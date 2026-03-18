from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MasseyConfig:
    """Configuration for aggregating Massey Ordinals."""

    # Use the last ranking for each (Season, TeamID, SystemName) by default.
    # Optionally, a DayNum cutoff window could be added here later.
    pass


def compute_massey_team_ranks(massey_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Massey Ordinals to per-team-per-season rank features.

    Strategy:
        - For each (Season, TeamID, SystemName), take the latest DayNum ranking.
        - Then aggregate across systems to get mean/median/best/worst rank.
    """
    if massey_df.empty:
        return pd.DataFrame(
            columns=["Season", "TeamID", "mean_rank", "median_rank", "best_rank", "worst_rank"]
        )

    # Kaggle schema uses RankingDayNum instead of DayNum.
    massey_df = massey_df.copy()
    if "DayNum" not in massey_df.columns and "RankingDayNum" in massey_df.columns:
        massey_df = massey_df.rename(columns={"RankingDayNum": "DayNum"})

    # Latest ranking per system (by DayNum) before the tournament.
    massey_df = massey_df.sort_values(["Season", "SystemName", "TeamID", "DayNum"])
    latest = massey_df.groupby(["Season", "SystemName", "TeamID"], as_index=False).tail(1)

    agg = (
        latest.groupby(["Season", "TeamID"])["OrdinalRank"]
        .agg(
            mean_rank="mean",
            median_rank="median",
            best_rank="min",
            worst_rank="max",
        )
        .reset_index()
    )

    return agg

