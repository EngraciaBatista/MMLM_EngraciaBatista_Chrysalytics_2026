from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


@dataclass
class PowerModelConfig:
    margin_cap: int
    features: Sequence[str]
    lgbm_params: dict


def build_power_training_frame(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    elo_season: pd.DataFrame,
    massey: pd.DataFrame,
    seeds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct training DataFrame for the power rating model.

    games: regular season detailed results.
    team_stats, elo_season, massey: per-team-per-season tables.
    seeds: optional seed table with Season, TeamID, Seed.
    """
    # Pre-compute team features table.
    feat = team_stats.merge(elo_season, on=["Season", "TeamID"], how="left")
    feat = feat.merge(massey, on=["Season", "TeamID"], how="left")

    if seeds is not None and not seeds.empty:
        seeds_proc = seeds.copy()
        seeds_proc["SeedNum"] = seeds_proc["Seed"].str[1:3].astype(int)
        seeds_proc = seeds_proc[["Season", "TeamID", "SeedNum"]]
        feat = feat.merge(seeds_proc, on=["Season", "TeamID"], how="left")
    else:
        feat["SeedNum"] = np.nan

    # Build one row per game with ordered teams: WTeamID vs LTeamID.
    df = games[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]].copy()
    df["margin"] = df["WScore"] - df["LScore"]

    # Attach features for both teams.
    left = feat.rename(columns=lambda c: c if c in {"Season", "TeamID"} else f"A_{c}")
    right = feat.rename(columns=lambda c: c if c in {"Season", "TeamID"} else f"B_{c}")

    df = df.merge(
        left,
        left_on=["Season", "WTeamID"],
        right_on=["Season", "TeamID"],
        how="left",
    ).drop(columns=["TeamID"])
    df = df.merge(
        right,
        left_on=["Season", "LTeamID"],
        right_on=["Season", "TeamID"],
        how="left",
    ).drop(columns=["TeamID"])

    # Difference features A - B.
    for col in [
        "off_rating",
        "def_rating",
        "tempo",
        "win_pct",
        "avg_margin",
        "mean_rank",
        "median_rank",
        "best_rank",
        "worst_rank",
        "elo_rating",
        "SeedNum",
    ]:
        a = f"A_{col}"
        b = f"B_{col}"
        if a in df.columns and b in df.columns:
            df[f"{col}_diff"] = df[a] - df[b]

    # Backwards-compatible aliases expected by config:
    # elo_diff -> elo_rating_diff, rank_diff -> median_rank_diff.
    if "elo_rating_diff" in df.columns and "elo_diff" not in df.columns:
        df["elo_diff"] = df["elo_rating_diff"]
    if "median_rank_diff" in df.columns and "rank_diff" not in df.columns:
        df["rank_diff"] = df["median_rank_diff"]

    return df


def add_recency_weights(df: pd.DataFrame) -> pd.Series:
    """Compute recency-based sample weights using DayNum within a season.

    Later-season games receive higher weight. The exact functional form can
    be tuned; here we use a simple linear scaling in [0.5, 1.5].
    """
    # Normalize DayNum within each season to [0, 1].
    day_min = df.groupby("Season")["DayNum"].transform("min")
    day_max = df.groupby("Season")["DayNum"].transform("max")
    norm = (df["DayNum"] - day_min) / (day_max - day_min + 1e-9)

    return 0.5 + norm  # ranges from 0.5 to 1.5 approximately


def train_power_model(
    train_df: pd.DataFrame,
    config: PowerModelConfig,
) -> LGBMRegressor:
    """Train LightGBM regressor to predict score margin."""
    df = train_df.copy()
    df["margin_capped"] = df["margin"].clip(-config.margin_cap, config.margin_cap)

    X = df[list(config.features)].values
    y = df["margin_capped"].values
    sample_weight = add_recency_weights(df)

    model = LGBMRegressor(**config.lgbm_params)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def compute_team_power_ratings(
    model: LGBMRegressor,
    team_stats: pd.DataFrame,
    elo_season: pd.DataFrame,
    massey: pd.DataFrame,
    seeds: pd.DataFrame | None,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Compute TeamPowerRating per team-season using the trained model.

    Strategy:
        - Build synthetic "vs league-average" opponent feature vectors:
          subtract global means so power rating reflects relative strength.
    """
    feat = team_stats.merge(elo_season, on=["Season", "TeamID"], how="left")
    feat = feat.merge(massey, on=["Season", "TeamID"], how="left")

    if seeds is not None and not seeds.empty:
        seeds_proc = seeds.copy()
        seeds_proc["SeedNum"] = seeds_proc["Seed"].str[1:3].astype(int)
        seeds_proc = seeds_proc[["Season", "TeamID", "SeedNum"]]
        feat = feat.merge(seeds_proc, on=["Season", "TeamID"], how="left")
    else:
        feat["SeedNum"] = np.nan

    feature_cols = {
        "off_rating": "off_rating",
        "def_rating": "def_rating",
        "tempo": "tempo",
        "win_pct": "win_pct",
        "avg_margin": "avg_margin",
        "mean_rank": "mean_rank",
        "median_rank": "median_rank",
        "best_rank": "best_rank",
        "worst_rank": "worst_rank",
        "elo_rating": "elo_rating",
        "SeedNum": "SeedNum",
    }

    # Compute league averages per season for opponent baseline.
    season_means = feat.groupby("Season")[list(feature_cols.values())].mean().rename(
        columns=lambda c: f"opp_{c}"
    )
    feat = feat.merge(season_means, on="Season", how="left")

    # Build difference features Team - OppAverage.
    for base_col in feature_cols.values():
        opp_col = f"opp_{base_col}"
        diff_col = f"{base_col}_diff"
        feat[diff_col] = feat[base_col] - feat[opp_col]

    # Backwards-compatible aliases for config feature names.
    if "elo_rating_diff" in feat.columns and "elo_diff" not in feat.columns:
        feat["elo_diff"] = feat["elo_rating_diff"]
    if "median_rank_diff" in feat.columns and "rank_diff" not in feat.columns:
        feat["rank_diff"] = feat["median_rank_diff"]

    X = feat[list(feature_names)].values
    power = model.predict(X)

    out = feat[["Season", "TeamID"]].copy()
    out["TeamPowerRating"] = power
    return out

