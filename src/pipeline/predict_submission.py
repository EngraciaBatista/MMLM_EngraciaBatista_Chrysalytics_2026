from __future__ import annotations

"""Generate the Kaggle submission file from trained models.

Checklist Steps 13-15.

Flow:
  1. Load the sample submission template (exact ID order preserved).
  2. Parse each ID → Season, TeamLow, TeamHigh.
  3. Assign league from MTeams / WTeams team ID lookups.
  4. For each (league, season), load team_features and build pairwise features.
  5. Predict probabilities with the calibrated model.
  6. Fill any missing predictions with 0.5, log them, and write submission.csv.

Usage
-----
    python -m src.pipeline.predict_submission --config configs/model_config.yaml
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from ..features.matchup_features import MATCHUP_BASE_FEATURES, build_matchup_pair
from ..models.tournament_models import CalibratedTournamentModel


def _parse_id(row_id: str) -> tuple[int, int, int]:
    """Parse 'Season_TeamLow_TeamHigh' into (season, team_low, team_high)."""
    parts = str(row_id).strip().split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid ID format: {row_id!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Kaggle submission for NCAA MMLM 2026."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg["paths"]
    data_dir = Path(paths_cfg["data_dir"])
    models_dir = Path(paths_cfg["models_dir"])
    submissions_dir = Path(paths_cfg["submissions_dir"])
    submissions_dir.mkdir(parents=True, exist_ok=True)

    template_name = paths_cfg.get("submission_template", "SampleSubmissionStage2.csv")
    final_name = paths_cfg.get("final_submission_file", "FinalSubmissionStage.csv")

    template_path = data_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(
            f"Submission template not found: {template_path}\n"
            "Place SampleSubmissionStage2.csv in the data folder."
        )

    # ── Load template (preserve exact row order) ──────────────────────────────
    template = pd.read_csv(template_path)
    if "ID" not in template.columns or "Pred" not in template.columns:
        raise ValueError(
            f"Template must have columns 'ID' and 'Pred'; got {list(template.columns)}"
        )

    # Parse IDs into Season / TeamLow / TeamHigh columns.
    parsed = template["ID"].apply(_parse_id)
    template["_Season"] = [p[0] for p in parsed]
    template["_TeamLow"] = [p[1] for p in parsed]
    template["_TeamHigh"] = [p[2] for p in parsed]

    # ── Assign league from team-ID lookups ────────────────────────────────────
    men_ids: set[int] = set()
    women_ids: set[int] = set()
    for league, fname in [("M", "MTeams.csv"), ("W", "WTeams.csv")]:
        p = data_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            ids = set(df["TeamID"].astype(int).tolist())
            if league == "M":
                men_ids = ids
            else:
                women_ids = ids

    if not men_ids and not women_ids:
        raise FileNotFoundError("Need MTeams.csv or WTeams.csv to assign leagues.")

    template["_League"] = None
    id_to_reason: dict[str, str] = {}

    for i in range(len(template)):
        tl = int(template.at[i, "_TeamLow"])
        th = int(template.at[i, "_TeamHigh"])
        row_id = template.at[i, "ID"]
        if tl in men_ids and th in men_ids:
            template.at[i, "_League"] = "M"
        elif tl in women_ids and th in women_ids:
            template.at[i, "_League"] = "W"
        else:
            template.at[i, "_League"] = None
            id_to_reason[row_id] = (
                f"Unknown league: TeamLow={tl} and/or TeamHigh={th} not found in "
                "MTeams.csv or WTeams.csv."
            )

    # ── Detect model mode: combined or per-league ─────────────────────────────
    combined_model_path = models_dir / "combined_tourney_model.pkl"
    use_combined = combined_model_path.exists()

    if use_combined:
        print("Using combined Men's+Women's model (recommended).")
        combined_model: CalibratedTournamentModel = joblib.load(combined_model_path)
        combined_feat_names: list[str] = combined_model.features
        # Load fold models for ensemble averaging.
        combined_fold_models = []
        i = 0
        while (models_dir / f"combined_tourney_fold_{i}.pkl").exists():
            combined_fold_models.append(joblib.load(models_dir / f"combined_tourney_fold_{i}.pkl"))
            i += 1
        print(f"  Loaded {len(combined_fold_models)} fold models for ensemble.")
    else:
        # Fallback: load per-league models.
        league_data: dict[str, dict] = {}
        for league in ("M", "W"):
            feat_path = models_dir / f"{league}_team_features.csv"
            model_path = models_dir / f"{league}_tourney_model.pkl"
            if not feat_path.exists() or not model_path.exists():
                print(f"[Warning] Model/features missing for league {league} — skipping.")
                continue
            m = joblib.load(model_path)
            fold_models = []
            j = 0
            while (models_dir / f"{league}_tourney_fold_{j}.pkl").exists():
                fold_models.append(joblib.load(models_dir / f"{league}_tourney_fold_{j}.pkl"))
                j += 1
            league_data[league] = {
                "team_features": pd.read_csv(feat_path),
                "model": m,
                "features": m.features,
                "fold_models": fold_models,
            }

    # ── Load team feature tables ───────────────────────────────────────────────
    league_features: dict[str, pd.DataFrame] = {}
    for league in ("M", "W"):
        fp = models_dir / f"{league}_team_features.csv"
        if fp.exists():
            league_features[league] = pd.read_csv(fp)

    # ── Generate predictions ──────────────────────────────────────────────────
    id_to_pred: dict[str, float] = {}

    def _predict_batch(
        records: list[dict],
        feat_names: list[str],
        main_model,
        fold_models: list,
        men_women_val: int | None = None,
    ) -> np.ndarray:
        """Run ensemble prediction: average fold models + final model."""
        df_pairs = pd.DataFrame.from_records(records)
        # Inject gender flag for combined model.
        if men_women_val is not None and "men_women" in feat_names:
            df_pairs["men_women"] = float(men_women_val)
        # Fill any NaN features with 0 (should rarely occur at prediction time).
        for col in feat_names:
            if col not in df_pairs.columns:
                df_pairs[col] = 0.0
            else:
                df_pairs[col] = df_pairs[col].fillna(0.0)
        X = df_pairs[feat_names].values.astype(float)
        # Ensemble: average spline-calibrated probabilities from all fold models + final.
        prob_arrays = []
        margin_clip = main_model.margin_clip
        spline = main_model.spline
        for fm in fold_models:
            margins = fm.predict(X)
            probs = np.clip(spline(np.clip(margins, -margin_clip, margin_clip)), 0.0, 1.0)
            prob_arrays.append(probs)
        # Add final model predictions.
        prob_arrays.append(main_model.predict_proba(X))
        preds = np.mean(prob_arrays, axis=0) if prob_arrays else main_model.predict_proba(X)
        # Confidence boost: predictions below 0.85 are nudged upward (winner's technique).
        preds = np.where(preds < 0.85, preds * 1.10, preds)
        # Tighter clip: [0.01, 0.99] instead of [0.05, 0.95] to allow near-certain matchups.
        preds = np.clip(preds, 0.01, 0.99)
        return preds

    for league in ("M", "W"):
        if league not in league_features:
            continue
        if use_combined:
            main_model = combined_model
            feat_names = combined_feat_names
            fold_mdls = combined_fold_models
            men_women_val = 1 if league == "M" else 0
        else:
            if league not in league_data:
                continue
            main_model = league_data[league]["model"]
            feat_names = league_data[league]["features"]
            fold_mdls = league_data[league]["fold_models"]
            men_women_val = None

        team_features = league_features[league]
        sub = template[template["_League"] == league]
        if sub.empty:
            continue

        available_seasons = sorted(team_features["Season"].dropna().unique().tolist())
        if not available_seasons:
            for _, r in sub.iterrows():
                id_to_reason[str(r["ID"])] = f"No team features at all for league {league}."
            continue

        for season in sub["_Season"].unique():
            season_feats = team_features[team_features["Season"] == season]
            season_rows = sub[sub["_Season"] == season]

            if season_feats.empty:
                proxy = max(available_seasons)
                season_feats = team_features[team_features["Season"] == proxy]
                if season_feats.empty:
                    for _, r in season_rows.iterrows():
                        id_to_reason[str(r["ID"])] = (
                            f"No features for {league}/{season} and no proxy available."
                        )
                    continue
                print(f"  [Warning] No features for {league}/{season}; using season {proxy} as proxy.")

            records = []
            indices = []
            for _, r in season_rows.iterrows():
                rec = build_matchup_pair(
                    season=int(r["_Season"]),
                    team_low=int(r["_TeamLow"]),
                    team_high=int(r["_TeamHigh"]),
                    season_feats=season_feats,
                    base_feature_names=MATCHUP_BASE_FEATURES,
                )
                if rec is None:
                    id_to_reason[str(r["ID"])] = (
                        f"TeamLow={int(r['_TeamLow'])} or TeamHigh={int(r['_TeamHigh'])} "
                        f"missing from {league} team features for season {season}."
                    )
                    continue
                records.append(rec)
                indices.append(r.name)

            if not records:
                continue

            preds = _predict_batch(records, feat_names, main_model, fold_mdls, men_women_val)

            for k, idx in enumerate(indices):
                row_id = str(template.at[idx, "ID"])
                id_to_pred[row_id] = float(preds[k])

    # ── Fill template and report missing predictions ──────────────────────────
    template["Pred"] = template["ID"].map(id_to_pred)
    missing_mask = template["Pred"].isna()
    n_missing = int(missing_mask.sum())

    if n_missing > 0:
        missing_ids = template.loc[missing_mask, "ID"].tolist()
        missing_with_reasons = [
            (id_, id_to_reason.get(id_, "Pipeline error: no reason recorded."))
            for id_ in missing_ids
        ]

        print("\n" + "=" * 80)
        print(f"*** ALERT: {n_missing} missing prediction(s) filled with 0.5 ***")
        print("=" * 80)
        for id_, reason in missing_with_reasons:
            print(f"  {id_}")
            print(f"    -> {reason}\n")

        missing_df = pd.DataFrame(missing_with_reasons, columns=["ID", "Reason"])
        report_path = submissions_dir / "missing_predictions_report.csv"
        missing_df.to_csv(report_path, index=False)
        print(f"  Full report saved to: {report_path}")
        print("=" * 80 + "\n")

    template["Pred"] = template["Pred"].fillna(0.5)
    out_df = template[["ID", "Pred"]]

    # Validate (Step 15)
    assert out_df["Pred"].between(0, 1).all(), "Probabilities out of [0,1] range!"
    assert out_df["Pred"].notna().all(), "NaN probabilities in output!"

    out_path = submissions_dir / final_name
    out_df.to_csv(out_path, index=False)
    print(
        f"Submission saved: {out_path}  "
        f"({len(out_df):,} rows, template: {template_name})"
    )


if __name__ == "__main__":
    main()
