from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

League = Literal["M", "W"]


@dataclass
class SeasonStatsConfig:
    """Configuration for season statistics computation."""

    min_season: int
    max_season: int


# ── Shared game-level view builder ────────────────────────────────────────────

def _build_game_level_view(detailed_results: pd.DataFrame) -> pd.DataFrame:
    """Build a symmetrical per-team-per-game view from raw detailed results.

    Includes OT normalization: all counting stats are divided by
        ot_factor = (40 + 5 * NumOT) / 40
    so every game is expressed in per-regulation-minute equivalents.
    This removes the bias where OT games inflate counting stats.

    Returns columns:
        Season, DayNum, TeamID, OppTeamID,
        Score, OppScore, FGA, OppFGA, FTA, OppFTA, TO, OppTO, OR, OppOR,
        is_win, margin
    """
    w_cols = {
        "Season": "Season",
        "DayNum": "DayNum",
        "NumOT": "NumOT",
        "WTeamID": "TeamID",
        "LTeamID": "OppTeamID",
        "WScore": "Score",
        "LScore": "OppScore",
        "WFGA": "FGA",
        "LFGA": "OppFGA",
        "WFTA": "FTA",
        "LFTA": "OppFTA",
        "WTO": "TO",
        "LTO": "OppTO",
        "WOR": "OR",
        "LOR": "OppOR",
        "WDR": "DR",
        "LDR": "OppDR",
        "WBlk": "Blk",
        "LBlk": "OppBlk",
        "WPF": "PF",
        "LPF": "OppPF",
    }
    l_cols = {
        "Season": "Season",
        "DayNum": "DayNum",
        "NumOT": "NumOT",
        "LTeamID": "TeamID",
        "WTeamID": "OppTeamID",
        "LScore": "Score",
        "WScore": "OppScore",
        "LFGA": "FGA",
        "WFGA": "OppFGA",
        "LFTA": "FTA",
        "WFTA": "OppFTA",
        "LTO": "TO",
        "WTO": "OppTO",
        "LOR": "OR",
        "WOR": "OppOR",
        "LDR": "DR",
        "WDR": "OppDR",
        "LBlk": "Blk",
        "WBlk": "OppBlk",
        "LPF": "PF",
        "WPF": "OppPF",
    }

    # Gracefully handle missing NumOT column (older data formats).
    src = detailed_results.copy()
    if "NumOT" not in src.columns:
        src["NumOT"] = 0
    # Gracefully handle missing box-score columns (DR, Blk, PF).
    for col in ["WDR", "LDR", "WBlk", "LBlk", "WPF", "LPF"]:
        if col not in src.columns:
            src[col] = np.nan

    w_games = src[list(w_cols.keys())].rename(columns=w_cols)
    l_games = src[list(l_cols.keys())].rename(columns=l_cols)
    games = pd.concat([w_games, l_games], ignore_index=True)

    # ── OT normalisation ──────────────────────────────────────────────────────
    # Divide all counting stats by (40 + 5*NumOT)/40 so OT games are comparable
    # to regulation games on a per-40-minute basis.
    ot_factor = (40.0 + 5.0 * games["NumOT"]) / 40.0
    for col in ["Score", "OppScore", "FGA", "OppFGA", "FTA", "OppFTA", "TO", "OppTO", "OR", "OppOR",
                "DR", "OppDR", "Blk", "OppBlk", "PF", "OppPF"]:
        if col in games.columns:
            games[col] = games[col] / ot_factor

    games["is_win"] = (games["Score"] > games["OppScore"]).astype(int)
    games["margin"] = games["Score"] - games["OppScore"]

    return games


# ── Public feature functions ──────────────────────────────────────────────────

def compute_team_season_stats(detailed_results: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team-per-season statistics from detailed results.

    Applies OT normalisation before computing all stats.

    Possessions formula (checklist Step 2):
        poss = FGA - OR + TO + 0.475 * FTA

    Output columns:
        Season, TeamID,
        off_eff    -- offensive efficiency = points scored / possessions
        def_eff    -- defensive efficiency = points allowed / possessions
        net_rating -- off_eff - def_eff
        win_pct    -- fraction of games won
        avg_margin -- average score margin
        pace       -- average possessions per game
    """
    games = _build_game_level_view(detailed_results)

    # Possessions: FGA - OR + TO + 0.475 * FTA  (checklist formula, Step 2)
    games["possessions"] = (
        games["FGA"] - games["OR"] + games["TO"] + 0.475 * games["FTA"]
    )
    # Guard against zero / negative possessions (data quality edge case).
    games["possessions"] = games["possessions"].clip(lower=1.0)

    agg_dict = dict(
        games_played=("TeamID", "size"),
        wins=("is_win", "sum"),
        points_scored=("Score", "sum"),
        points_allowed=("OppScore", "sum"),
        total_margin=("margin", "sum"),
        total_possessions=("possessions", "sum"),
    )
    # Include box-score totals only when the columns are present and non-null.
    for raw_col, agg_name in [("DR", "total_dr"), ("Blk", "total_blk"), ("PF", "total_pf")]:
        if raw_col in games.columns and games[raw_col].notna().any():
            agg_dict[agg_name] = (raw_col, "sum")

    grouped = games.groupby(["Season", "TeamID"], as_index=False).agg(**agg_dict)

    grouped["win_pct"] = grouped["wins"] / grouped["games_played"]
    grouped["avg_margin"] = grouped["total_margin"] / grouped["games_played"]

    grouped["off_eff"] = grouped["points_scored"] / grouped["total_possessions"]
    grouped["def_eff"] = grouped["points_allowed"] / grouped["total_possessions"]
    grouped["net_rating"] = grouped["off_eff"] - grouped["def_eff"]
    grouped["pace"] = grouped["total_possessions"] / grouped["games_played"]

    # Per-game box score averages (only when data was available).
    extra_cols = []
    for total_col, avg_col in [("total_dr", "avg_dr"), ("total_blk", "avg_blk"), ("total_pf", "avg_pf")]:
        if total_col in grouped.columns:
            grouped[avg_col] = grouped[total_col] / grouped["games_played"]
            extra_cols.append(avg_col)

    # Replace any infinities or NaNs produced by sparse data.
    grouped = grouped.replace([np.inf, -np.inf], np.nan)

    base_cols = ["Season", "TeamID", "off_eff", "def_eff", "net_rating", "win_pct", "avg_margin", "pace"]
    return grouped[base_cols + extra_cols]


def compute_opponent_stats(detailed_results: pd.DataFrame) -> pd.DataFrame:
    """Compute what opponents average when playing *against* each team.

    These features capture a team's defensive context in raw form rather than
    through efficiency ratios, letting the model find non-linear interactions
    (e.g. a 90-ppg offence playing a team that holds opponents to 55 ppg
    is different from a 70-ppg offence vs a 65-ppg defence).

    Output columns:
        Season, TeamID,
        opp_avg_score  -- avg points opponents score against this team
        opp_avg_fga    -- avg FGA opponents attempt against this team
    """
    games = _build_game_level_view(detailed_results)

    opp = games.groupby(["Season", "TeamID"], as_index=False).agg(
        opp_avg_score=("OppScore", "mean"),
        opp_avg_fga=("OppFGA", "mean"),
    )

    return opp.replace([np.inf, -np.inf], np.nan)


def compute_sos_adjusted_stats(
    raw_stats: pd.DataFrame,
    detailed_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compute strength-of-schedule-adjusted efficiency stats (single-iteration).

    For each team:
        adj_off_eff  = off_eff  * (league_avg_def_eff / opp_avg_def_eff)
        adj_def_eff  = def_eff  * (league_avg_off_eff / opp_avg_off_eff)
        adj_net_rating = adj_off_eff - adj_def_eff

    Parameters
    ----------
    raw_stats:
        Output of compute_team_season_stats (must contain off_eff, def_eff).
    detailed_results:
        Raw Kaggle detailed results with WTeamID / LTeamID columns.

    Returns
    -------
    DataFrame with Season, TeamID, adj_off_eff, adj_def_eff, adj_net_rating.
    """
    # Build a flat opponent map: (Season, TeamID) → [OppTeamIDs played that season]
    w_view = detailed_results[["Season", "WTeamID", "LTeamID"]].rename(
        columns={"WTeamID": "TeamID", "LTeamID": "OppTeamID"}
    )
    l_view = detailed_results[["Season", "LTeamID", "WTeamID"]].rename(
        columns={"LTeamID": "TeamID", "WTeamID": "OppTeamID"}
    )
    opp_map = pd.concat([w_view, l_view], ignore_index=True)

    # Attach opponents' raw efficiency stats.
    opp_stats = raw_stats[["Season", "TeamID", "off_eff", "def_eff"]].rename(
        columns={"TeamID": "OppTeamID", "off_eff": "opp_off_eff", "def_eff": "opp_def_eff"}
    )
    opp_map = opp_map.merge(opp_stats, on=["Season", "OppTeamID"], how="left")

    # Average opponent efficiency per (Season, TeamID).
    opp_avgs = opp_map.groupby(["Season", "TeamID"], as_index=False).agg(
        opp_avg_off_eff=("opp_off_eff", "mean"),
        opp_avg_def_eff=("opp_def_eff", "mean"),
    )

    # League-average efficiency per season (used as the neutral baseline).
    league_avgs = raw_stats.groupby("Season", as_index=False).agg(
        league_avg_off_eff=("off_eff", "mean"),
        league_avg_def_eff=("def_eff", "mean"),
    )

    adj = (
        raw_stats[["Season", "TeamID", "off_eff", "def_eff"]]
        .merge(opp_avgs, on=["Season", "TeamID"], how="left")
        .merge(league_avgs, on="Season", how="left")
    )

    adj["adj_off_eff"] = (
        adj["off_eff"] * adj["league_avg_def_eff"] / adj["opp_avg_def_eff"]
    )
    adj["adj_def_eff"] = (
        adj["def_eff"] * adj["league_avg_off_eff"] / adj["opp_avg_off_eff"]
    )
    adj["adj_net_rating"] = adj["adj_off_eff"] - adj["adj_def_eff"]

    adj = adj.replace([np.inf, -np.inf], np.nan)
    return adj[["Season", "TeamID", "adj_off_eff", "adj_def_eff", "adj_net_rating"]]
