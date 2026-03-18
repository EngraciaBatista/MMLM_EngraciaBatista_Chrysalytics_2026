from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import yaml

from ..data.load_data import DataPaths, load_regular_season_detailed
from ..features.season_stats import compute_team_season_stats
from ..features.elo import EloConfig, compute_elo
from ..features.massey_features import compute_massey_team_ranks
from ..models.power_rating_model import (
    PowerModelConfig,
    build_power_training_frame,
    compute_team_power_ratings,
    train_power_model,
)
from ..data.load_data import load_massey_ordinals, load_tourney_seeds


def train_for_league(config_path: Path, league: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    min_season = cfg["seasons"]["min_season"]
    max_season = cfg["seasons"]["max_season"]

    paths = DataPaths(base_dir=data_dir)

    reg = load_regular_season_detailed(paths, league)  # type: ignore[arg-type]
    reg = reg[(reg["Season"] >= min_season) & (reg["Season"] <= max_season)].copy()

    team_stats = compute_team_season_stats(reg)

    elo_cfg = EloConfig(
        base_rating=cfg["elo"]["base_rating"],
        k_factor=cfg["elo"]["k_factor"],
        mov_scale=cfg["elo"]["mov_scale"],
        regression_factor=cfg["elo"]["regression_factor"],
        home_advantage=cfg["elo"].get("home_advantage", 0.0),
    )
    _, season_elos = compute_elo(reg, elo_cfg)

    massey_df = load_massey_ordinals(paths, league)  # type: ignore[arg-type]
    if massey_df.empty:
        # No Massey data for this league (e.g., women's); use empty ranking features.
        massey_team = compute_massey_team_ranks(massey_df)
    else:
        massey_df = massey_df[massey_df["Season"].between(min_season, max_season)]
        massey_team = compute_massey_team_ranks(massey_df)

    seeds_df = load_tourney_seeds(paths, league)  # type: ignore[arg-type]

    train_df = build_power_training_frame(reg, team_stats, season_elos, massey_team, seeds_df)

    # Allow league-specific feature availability (e.g. no Massey for women).
    requested_features = list(cfg["power_model"]["features"])
    available_features = [f for f in requested_features if f in train_df.columns]
    if len(available_features) != len(requested_features):
        missing = sorted(set(requested_features) - set(available_features))
        print(f"[{league}] Skipping unavailable power model features: {missing}")

    power_cfg = PowerModelConfig(
        margin_cap=cfg["power_model"]["margin_cap"],
        features=available_features,
        lgbm_params=cfg["power_model"]["lightgbm_params"],
    )

    model = train_power_model(train_df, power_cfg)

    power_ratings = compute_team_power_ratings(
        model=model,
        team_stats=team_stats,
        elo_season=season_elos,
        massey=massey_team,
        seeds=seeds_df,
        feature_names=available_features,
    )

    # Persist artifacts.
    joblib.dump(model, models_dir / f"{league}_power_model.pkl")
    power_ratings.to_csv(models_dir / f"{league}_team_power_ratings.csv", index=False)
    team_stats.to_csv(models_dir / f"{league}_team_season_stats.csv", index=False)
    season_elos.to_csv(models_dir / f"{league}_team_elo.csv", index=False)
    massey_team.to_csv(models_dir / f"{league}_team_massey.csv", index=False)
    seeds_df.to_csv(models_dir / f"{league}_tourney_seeds.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train power rating model for NCAA MMLM.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)

    for league in ("M", "W"):
        train_for_league(config_path, league)


if __name__ == "__main__":
    main()

