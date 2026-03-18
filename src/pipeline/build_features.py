from __future__ import annotations

"""Build and persist the team feature table for both leagues.

Checklist Steps 1-7.

Produces one CSV per league:  outputs/models/{M,W}_team_features.csv

Columns in the feature table (checklist Step 7):
    Season, TeamID,
    strength_rating,               -- Bradley-Terry logistic regression rating
    off_eff, def_eff,              -- raw offensive / defensive efficiency
    net_rating,                    -- off_eff - def_eff
    adj_off_eff, adj_def_eff,      -- SOS-adjusted efficiencies
    adj_net_rating,                -- adj_off_eff - adj_def_eff
    win_pct, avg_margin, pace,     -- other season stats
    seed, seed_strength,           -- tournament seed (raw + inverted: 17 - seed)

Usage
-----
    python -m src.pipeline.build_features --config configs/model_config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from ..data.load_data import DataPaths, load_regular_season_detailed, load_tourney_seeds
from ..features.season_stats import (
    compute_team_season_stats,
    compute_sos_adjusted_stats,
    compute_opponent_stats,
)
from ..features.strength_rating import compute_strength_ratings
from ..features.margin_quality import compute_margin_quality
from ..features.elo import compute_elo, EloConfig
from ..features.matchup_features import process_seeds


def build_for_league(config_path: Path, league: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    min_season = cfg["seasons"]["min_season"]
    max_season = cfg["seasons"]["max_season"]

    paths = DataPaths(base_dir=data_dir)

    # ── Step 1: Load regular season detailed results ──────────────────────────
    print(f"[{league}] Loading regular season detailed results...")
    reg = load_regular_season_detailed(paths, league)  # type: ignore[arg-type]
    reg = reg[(reg["Season"] >= min_season) & (reg["Season"] <= max_season)].copy()
    print(f"[{league}]   {len(reg):,} games loaded.")

    # ── Steps 2-4: Team season stats ─────────────────────────────────────────
    print(f"[{league}] Computing team season stats...")
    team_stats = compute_team_season_stats(reg)
    print(f"[{league}]   {len(team_stats):,} team-season rows.")

    # ── SOS-adjusted efficiency (medium-impact improvement) ───────────────────
    print(f"[{league}] Computing SOS-adjusted efficiency stats...")
    sos_adj = compute_sos_adjusted_stats(team_stats, reg)
    print(f"[{league}]   SOS adjustment done.")

    # ── Step 5: Team strength ratings (Bradley-Terry) ─────────────────────────
    print(f"[{league}] Fitting Bradley-Terry strength ratings...")
    strength = compute_strength_ratings(reg)
    print(f"[{league}]   {len(strength):,} team-season ratings computed.")

    # ── Margin quality (OLS/Massey — winner's key feature) ────────────────────
    # Fits point_diff ~ team_T1 - team_T2 via Ridge regression.
    # Unlike Bradley-Terry (win/loss only), this uses the actual margin,
    # giving ~10x more information per game.
    print(f"[{league}] Computing OLS margin quality ratings...")
    margin_qual = compute_margin_quality(reg)
    print(f"[{league}]   {len(margin_qual):,} team-season margin quality scores.")

    # ── Elo ratings (late-season momentum) ───────────────────────────────────
    print(f"[{league}] Computing Elo ratings...")
    elo_cfg = EloConfig(k_factor=20.0, regression_factor=0.75)
    _, season_elos = compute_elo(reg, elo_cfg)
    print(f"[{league}]   {len(season_elos):,} team-season Elo ratings computed.")

    # ── Opponent stats (what opponents average against each team) ─────────────
    print(f"[{league}] Computing opponent stats...")
    opp_stats = compute_opponent_stats(reg)
    print(f"[{league}]   {len(opp_stats):,} team-season opponent stat rows.")

    # ── Step 6: Tournament seeds ──────────────────────────────────────────────
    print(f"[{league}] Processing tournament seeds...")
    raw_seeds = load_tourney_seeds(paths, league)  # type: ignore[arg-type]
    raw_seeds = raw_seeds[
        (raw_seeds["Season"] >= min_season) & (raw_seeds["Season"] <= max_season)
    ].copy()
    seeds = process_seeds(raw_seeds)
    print(f"[{league}]   {len(seeds):,} seed entries.")

    # ── Step 7: Merge unified team feature table ──────────────────────────────
    print(f"[{league}] Merging team feature table...")
    team_features = (
        team_stats
        .merge(sos_adj, on=["Season", "TeamID"], how="left")
        .merge(strength, on=["Season", "TeamID"], how="left")
        .merge(margin_qual, on=["Season", "TeamID"], how="left")
        .merge(season_elos, on=["Season", "TeamID"], how="left")
        .merge(opp_stats, on=["Season", "TeamID"], how="left")
        .merge(seeds, on=["Season", "TeamID"], how="left")
    )

    n_null_rating = team_features["strength_rating"].isna().sum()
    if n_null_rating > 0:
        print(f"[{league}]   WARNING: {n_null_rating} rows missing strength_rating.")

    out_path = models_dir / f"{league}_team_features.csv"
    team_features.to_csv(out_path, index=False)
    print(f"[{league}] Saved → {out_path}  ({len(team_features):,} rows)")
    print(f"[{league}] Columns: {list(team_features.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build team feature table for NCAA MMLM (Steps 1-7)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--league",
        type=str,
        choices=["M", "W", "both"],
        default="both",
        help="Which league to process (default: both).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    leagues = ["M", "W"] if args.league == "both" else [args.league]
    for league in leagues:
        build_for_league(config_path, league)


if __name__ == "__main__":
    main()
