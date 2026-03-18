from __future__ import annotations

"""Train the tournament matchup model for each league.

Architecture (winner-inspired):
  - LightGBM REGRESSOR predicts point differential (not win probability).
  - A spline is fitted on out-of-fold (predicted_margin, actual_win) pairs to
    convert margins to calibrated win probabilities.
  - Optional Optuna hyperparameter sweep (--tune flag, 50 trials).

Why margin regression beats classification:
  A 30-point win and a 1-point win both become label=1 in classification,
  throwing away 95% of the signal.  Regression on margin preserves that
  information, and the spline handles the non-linear margin → P(win) mapping.

Usage
-----
    python -m src.pipeline.train_tournament_model --config configs/model_config.yaml
    python -m src.pipeline.train_tournament_model --config configs/model_config.yaml --tune
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from ..data.load_data import DataPaths, load_tourney_results
from ..features.matchup_features import (
    MATCHUP_BASE_FEATURES,
    MATCHUP_DIFF_FEATURES,
    build_tournament_training_frame,
)
from ..models.tournament_models import (
    MarginRegressionTournamentModel,
    TournamentModelConfig,
    fit_spline_calibrator,
    train_margin_model,
)
from ..validation.metrics import brier_score


def _load_matchups_for_league(
    league: str,
    cfg: dict,
    models_dir: Path,
    data_dir: Path,
) -> pd.DataFrame:
    """Load team features + tournament results and build the matchup training frame."""
    paths = DataPaths(base_dir=data_dir)
    min_season = cfg["seasons"]["min_season"]
    max_season = cfg["seasons"]["max_season"]

    feat_path = models_dir / f"{league}_team_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Team features not found: {feat_path}\nRun build_features.py first."
        )
    team_features = pd.read_csv(feat_path)

    tourney = load_tourney_results(paths, league)
    tourney = tourney[
        (tourney["Season"] >= min_season) & (tourney["Season"] <= max_season)
    ].copy()

    matchups = build_tournament_training_frame(
        tourney_results=tourney,
        team_features=team_features,
        base_feature_names=MATCHUP_BASE_FEATURES,
    )
    return matchups


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def _run_optuna_tuning(
    cv_data: pd.DataFrame,
    available_feats: list[str],
    cv_test_seasons: list[int],
    min_season: int,
    margin_clip: float,
    n_trials: int = 50,
) -> dict:
    """Optuna sweep minimising mean CV Brier (via spline-calibrated margins)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "num_leaves": trial.suggest_int("num_leaves", 8, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            "random_state": 2026,
            "verbose": -1,
        }
        # Also tune margin_clip: range [15, 35] around the default of 25.
        trial_margin_clip = trial.suggest_float("margin_clip", 15.0, 35.0)
        cfg = TournamentModelConfig(features=available_feats, lgbm_params=params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            oof_margins, oof_labels = [], []
            for test_season in sorted(cv_test_seasons):
                train_df = cv_data[cv_data["Season"] < test_season]
                test_df = cv_data[cv_data["Season"] == test_season]
                if train_df.empty or test_df.empty:
                    continue
                lgbm = train_margin_model(train_df, cfg)
                oof_margins.extend(lgbm.predict(test_df[available_feats].values.astype(float)))
                oof_labels.extend(test_df["label"].values)

            if not oof_margins:
                return 0.25

            spline = fit_spline_calibrator(
                np.array(oof_margins), np.array(oof_labels), trial_margin_clip
            )

            briers = []
            for test_season in sorted(cv_test_seasons):
                test_df = cv_data[cv_data["Season"] == test_season]
                if test_df.empty:
                    continue
                train_df = cv_data[cv_data["Season"] < test_season]
                if train_df.empty:
                    continue
                lgbm = train_margin_model(
                    train_df,
                    TournamentModelConfig(features=available_feats, lgbm_params=params),
                )
                margins = lgbm.predict(test_df[available_feats].values.astype(float))
                probs = np.clip(
                    spline(np.clip(margins, -trial_margin_clip, trial_margin_clip)), 0.0, 1.0
                )
                briers.append(brier_score(test_df["label"].values.astype(int), probs))

        return float(np.mean(briers)) if briers else 0.25

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    # Extract margin_clip separately; remaining keys are LightGBM params.
    best_margin_clip = float(best.pop("margin_clip", margin_clip))
    best["random_state"] = 2026
    best["verbose"] = -1
    return best, best_margin_clip


# ── Shared training core ──────────────────────────────────────────────────────

def _prepare_matchups(
    all_matchups: pd.DataFrame,
    available_feats: list[str],
    tag: str,
) -> pd.DataFrame:
    """Impute NaN features with season means; drop only rows with missing target."""
    n_before = len(all_matchups)
    for feat_col in available_feats:
        if feat_col in all_matchups.columns and all_matchups[feat_col].isna().any():
            season_means = all_matchups.groupby("Season")[feat_col].transform("mean")
            global_mean = all_matchups[feat_col].mean()
            all_matchups[feat_col] = (
                all_matchups[feat_col].fillna(season_means).fillna(global_mean)
            )
    if "point_diff" in all_matchups.columns:
        all_matchups = all_matchups.dropna(subset=["point_diff"]).reset_index(drop=True)
    n_dropped = n_before - len(all_matchups)
    if n_dropped > 0:
        print(f"[{tag}] Dropped {n_dropped} rows with missing target ({n_dropped/n_before*100:.1f}%).")
    else:
        print(f"[{tag}] All {n_before} rows retained after NaN imputation.")
    return all_matchups


def _train_and_save(
    tag: str,
    all_matchups: pd.DataFrame,
    available_feats: list[str],
    cfg: dict,
    models_dir: Path,
    tune: bool,
) -> None:
    """Core training loop: Optuna tuning → OOF spline calibration → fold ensemble → save."""
    cv_test_seasons = cfg["seasons"]["cv_test_seasons"]
    calib_seasons: list[int] = cfg["seasons"].get("calib_seasons", [])
    margin_clip: float = cfg.get("tournament_model", {}).get("margin_clip", 25.0)

    cv_data = all_matchups[~all_matchups["Season"].isin(calib_seasons)]

    lgbm_params = cfg.get("tournament_model", {}).get("lgbm_params", {
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
        "num_leaves": 15, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 2026, "verbose": -1,
    })

    if tune:
        n_trials = cfg.get("tournament_model", {}).get("optuna_trials", 50)
        print(f"\n[{tag}] Running Optuna tuning ({n_trials} trials)...")
        lgbm_params, margin_clip = _run_optuna_tuning(
            cv_data=cv_data,
            available_feats=available_feats,
            cv_test_seasons=cv_test_seasons,
            min_season=cfg["seasons"]["min_season"],
            margin_clip=margin_clip,
            n_trials=n_trials,
        )
        print(f"[{tag}] Best LGBM params: {lgbm_params}")
        print(f"[{tag}] Best margin_clip: {margin_clip:.2f}")

    tm_config = TournamentModelConfig(
        features=available_feats,
        lgbm_params=lgbm_params,
        margin_clip=margin_clip,
    )

    # ── LOSO OOF collection → spline calibration + fold ensemble ─────────────
    print(f"\n[{tag}] Collecting OOF margin predictions for spline calibration...")
    oof_margins: list[float] = []
    oof_labels: list[int] = []
    oof_season_list: list[int] = []
    fold_models: list = []  # save each fold's trained model for the ensemble

    for test_season in sorted(cv_test_seasons):
        train_df = cv_data[cv_data["Season"] < test_season].reset_index(drop=True)
        test_df = cv_data[cv_data["Season"] == test_season].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue
        fold_lgbm = train_margin_model(train_df, tm_config)
        fold_models.append(fold_lgbm)
        fold_margins = fold_lgbm.predict(test_df[available_feats].values.astype(float))
        oof_margins.extend(fold_margins.tolist())
        oof_labels.extend(test_df["label"].values.tolist())
        oof_season_list.extend(test_df["Season"].values.tolist())

    if not oof_margins:
        raise RuntimeError(f"[{tag}] No OOF predictions collected — check cv_test_seasons.")

    spline = fit_spline_calibrator(
        np.array(oof_margins), np.array(oof_labels), margin_clip
    )
    print(f"[{tag}] Spline fitted on {len(oof_margins)} OOF predictions.")

    # ── CV evaluation ─────────────────────────────────────────────────────────
    print(f"\n[{tag}] CV evaluation on seasons: {cv_test_seasons}")
    cv_briers = []
    for test_season in sorted(cv_test_seasons):
        season_mask = [s == test_season for s in oof_season_list]
        if not any(season_mask):
            continue
        s_margins = np.array([m for m, flag in zip(oof_margins, season_mask) if flag])
        s_labels = np.array([l for l, flag in zip(oof_labels, season_mask) if flag])
        s_probs = np.clip(spline(np.clip(s_margins, -margin_clip, margin_clip)), 0.0, 1.0)
        s_brier = brier_score(s_labels.astype(int), s_probs)
        n_games = len(s_labels) // 2
        cv_briers.append(s_brier)
        print(f"  Season {test_season}: Brier={s_brier:.5f}  (n={n_games})")

    if cv_briers:
        print(f"  Mean CV Brier: {np.mean(cv_briers):.5f}")

    # ── Final model (trained on all non-calib data) ────────────────────────────
    if calib_seasons:
        final_train = all_matchups[
            ~all_matchups["Season"].isin(calib_seasons)
        ].reset_index(drop=True)
        print(
            f"\n[{tag}] Final model: {len(final_train)} training rows "
            f"(calib seasons {calib_seasons} held out for spline)."
        )
    else:
        final_train = all_matchups.copy()

    print(f"[{tag}] Training final margin regression model...")
    final_lgbm = train_margin_model(final_train, tm_config)

    final_model = MarginRegressionTournamentModel(
        lgbm=final_lgbm,
        spline=spline,
        features=available_feats,
        margin_clip=margin_clip,
    )

    # ── Persist artifacts ─────────────────────────────────────────────────────
    joblib.dump(final_model, models_dir / f"{tag}_tourney_model.pkl")

    # Save each fold model for the ensemble.
    for i, fm in enumerate(fold_models):
        joblib.dump(fm, models_dir / f"{tag}_tourney_fold_{i}.pkl")
    print(f"[{tag}] Saved {len(fold_models)} fold models for ensemble.")

    # Save metadata.
    pd.Series(available_feats, name="feature").to_csv(
        models_dir / f"{tag}_tourney_features.csv", index=False
    )
    pd.DataFrame(
        [{"season": s, "brier": b} for s, b in zip(sorted(cv_test_seasons), cv_briers)]
    ).to_csv(models_dir / f"{tag}_tourney_cv.csv", index=False)
    # Record spline and margin_clip alongside fold count for predict_submission.
    pd.DataFrame([{
        "tag": tag,
        "n_folds": len(fold_models),
        "margin_clip": margin_clip,
        "features": ",".join(available_feats),
    }]).to_csv(models_dir / f"{tag}_tourney_meta.csv", index=False)
    print(f"[{tag}] Model saved → {models_dir / f'{tag}_tourney_model.pkl'}")


# ── Per-league training ───────────────────────────────────────────────────────

def train_for_league(config_path: Path, league: str, tune: bool = False) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    all_matchups = _load_matchups_for_league(league, cfg, models_dir, data_dir)

    config_feats = cfg.get("tournament_model", {}).get("features", MATCHUP_DIFF_FEATURES)
    # men_women feature only makes sense for the combined model; skip for per-league.
    config_feats = [f for f in config_feats if f != "men_women"]
    available_feats = [f for f in config_feats if f in all_matchups.columns]
    missing_feats = sorted(set(config_feats) - set(available_feats))
    if missing_feats:
        print(f"[{league}] Skipping unavailable features: {missing_feats}")

    all_matchups = _prepare_matchups(all_matchups, available_feats, league)
    _train_and_save(league, all_matchups, available_feats, cfg, models_dir, tune)


# ── Combined M+W training ─────────────────────────────────────────────────────

def train_combined(config_path: Path, tune: bool = False) -> None:
    """Train a single model on Men's + Women's data with a men_women gender flag.

    This doubles training data and gives the spline calibrator more OOF points,
    which is one of the key techniques from the 2025 winning solution.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    m_matchups = _load_matchups_for_league("M", cfg, models_dir, data_dir)
    w_matchups = _load_matchups_for_league("W", cfg, models_dir, data_dir)

    m_matchups["men_women"] = 1
    w_matchups["men_women"] = 0

    all_matchups = pd.concat([m_matchups, w_matchups], ignore_index=True)
    print(
        f"[combined] Men's: {len(m_matchups)} rows, Women's: {len(w_matchups)} rows, "
        f"Total: {len(all_matchups)} rows."
    )

    config_feats = cfg.get("tournament_model", {}).get("features", MATCHUP_DIFF_FEATURES)
    available_feats = [f for f in config_feats if f in all_matchups.columns]
    missing_feats = sorted(set(config_feats) - set(available_feats))
    if missing_feats:
        print(f"[combined] Skipping unavailable features: {missing_feats}")

    all_matchups = _prepare_matchups(all_matchups, available_feats, "combined")
    _train_and_save("combined", all_matchups, available_feats, cfg, models_dir, tune)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train margin regression tournament model for NCAA MMLM."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--league",
        type=str,
        choices=["M", "W", "both", "combined"],
        default="combined",
        help="'combined' trains one M+W model (recommended). 'both' trains separate models.",
    )
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter sweep.")
    args = parser.parse_args()
    config_path = Path(args.config)

    if args.league == "combined":
        train_combined(config_path, tune=args.tune)
    else:
        leagues = ["M", "W"] if args.league == "both" else [args.league]
        for league in leagues:
            train_for_league(config_path, league, tune=args.tune)


if __name__ == "__main__":
    main()
