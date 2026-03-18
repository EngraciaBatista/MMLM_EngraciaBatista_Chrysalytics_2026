from __future__ import annotations

"""Matchup feature construction for tournament prediction.

Checklist Steps 6-9.

Feature convention:
    All diffs are A_feat - B_feat where A = TeamLow (lower TeamID),
    B = TeamHigh (higher TeamID).  seed_strength (= 17 - seed) is used
    instead of raw seed so that higher values always mean a stronger team,
    keeping diff signs consistent with other features.
"""

import numpy as np
import pandas as pd


# Per-team base features that get differenced into matchup features.
# Must all exist as columns in the team_features table from build_features.py.
MATCHUP_BASE_FEATURES = [
    "strength_rating",
    "margin_quality",   # OLS/Massey margin-based team quality (winner's key feature)
    "seed_strength",
    "elo_rating",       # Elo rating at end of regular season (captures momentum)
    "off_eff",
    "def_eff",
    "net_rating",
    "adj_off_eff",      # SOS-adjusted offensive efficiency
    "adj_def_eff",      # SOS-adjusted defensive efficiency
    "adj_net_rating",   # adj_off_eff - adj_def_eff
    "opp_avg_score",    # points opponents score against this team (raw defensive quality)
    "opp_avg_fga",      # shot attempts opponents take against this team
    "win_pct",
    "avg_margin",
    "pace",
    "avg_dr",           # defensive rebounds per game (when available)
    "avg_blk",          # blocks per game (when available)
    "avg_pf",           # personal fouls per game (when available)
]

# All diff columns exposed to the model (base diffs + computed interactions).
MATCHUP_DIFF_FEATURES = [f"{f}_diff" for f in MATCHUP_BASE_FEATURES] + [
    "strength_seed_interaction",  # strength_rating_diff * seed_strength_diff
]


def parse_seed(seed_str: str) -> int:
    """Convert a seed string like 'W01', 'X16', 'Y04' to an integer seed number."""
    return int(str(seed_str)[1:3])


def process_seeds(seeds_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw seeds table to (Season, TeamID, seed, seed_strength).

    Checklist Step 6: seed_strength = 17 - seed
    so that seed-1 teams (best) have the highest seed_strength (16).
    """
    out = seeds_df[["Season", "TeamID", "Seed"]].copy()
    out["seed"] = out["Seed"].apply(parse_seed)
    out["seed_strength"] = 17 - out["seed"]
    return out[["Season", "TeamID", "seed", "seed_strength"]]


def _attach_team_features(
    df: pd.DataFrame,
    team_features: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> pd.DataFrame:
    """Attach per-team-season features for two teams (A and B) in each row."""
    feat_left = team_features.rename(
        columns=lambda c: c if c in {"Season", "TeamID"} else f"A_{c}"
    )
    feat_right = team_features.rename(
        columns=lambda c: c if c in {"Season", "TeamID"} else f"B_{c}"
    )
    df = df.merge(
        feat_left,
        left_on=["Season", left_col],
        right_on=["Season", "TeamID"],
        how="left",
    ).drop(columns=["TeamID"])
    df = df.merge(
        feat_right,
        left_on=["Season", right_col],
        right_on=["Season", "TeamID"],
        how="left",
    ).drop(columns=["TeamID"])
    return df


def _compute_diffs(df: pd.DataFrame, base_features: list) -> pd.DataFrame:
    """Compute A_feat - B_feat differences for each base feature."""
    for col in base_features:
        a_col = f"A_{col}"
        b_col = f"B_{col}"
        if a_col in df.columns and b_col in df.columns:
            df[f"{col}_diff"] = df[a_col] - df[b_col]
        else:
            df[f"{col}_diff"] = np.nan
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived interaction features after base diffs are computed."""
    # Interaction: both strength and seed agree → high confidence; disagree → upset risk.
    if "strength_rating_diff" in df.columns and "seed_strength_diff" in df.columns:
        df["strength_seed_interaction"] = (
            df["strength_rating_diff"] * df["seed_strength_diff"]
        )
    else:
        df["strength_seed_interaction"] = np.nan
    return df


def build_tournament_training_frame(
    tourney_results: pd.DataFrame,
    team_features: pd.DataFrame,
    base_feature_names=None,
) -> pd.DataFrame:
    """Construct a tournament matchup training DataFrame.

    Checklist Steps 8-9.

    Uses canonical TeamLow/TeamHigh ordering (by TeamID) and mirrors every
    row with reversed teams + flipped label to remove team-order bias.

    Returns
    -------
    DataFrame with: Season, TeamLow, TeamHigh, label, <feature>_diff columns,
    and strength_seed_interaction.
    """
    if base_feature_names is None:
        base_feature_names = MATCHUP_BASE_FEATURES

    df = tourney_results.copy()
    df["TeamLow"] = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["TeamHigh"] = df[["WTeamID", "LTeamID"]].max(axis=1)
    df["TeamLowWin"] = (df["WTeamID"] == df["TeamLow"]).astype(int)

    # Point differential from TeamLow perspective (positive = TeamLow won).
    # Used as regression target for the margin model.
    if "WScore" in df.columns and "LScore" in df.columns:
        df["point_diff"] = np.where(
            df["WTeamID"] == df["TeamLow"],
            df["WScore"] - df["LScore"],    # TeamLow won: positive
            df["LScore"] - df["WScore"],    # TeamHigh won: negative
        )
    else:
        df["point_diff"] = np.nan

    matchups = df[["Season", "TeamLow", "TeamHigh", "TeamLowWin", "point_diff"]].drop_duplicates()
    matchups = _attach_team_features(matchups, team_features, "TeamLow", "TeamHigh")
    matchups = _compute_diffs(matchups, base_feature_names)
    matchups = _add_interactions(matchups)
    matchups["label"] = matchups["TeamLowWin"]

    # Mirror dataset: swap teams, flip diffs, flip label + flip point_diff (Step 9)
    reversed_df = matchups.copy()
    tl = reversed_df["TeamLow"].copy()
    th = reversed_df["TeamHigh"].copy()
    reversed_df["TeamLow"] = th
    reversed_df["TeamHigh"] = tl
    reversed_df["label"] = 1 - reversed_df["label"]
    if "point_diff" in reversed_df.columns:
        reversed_df["point_diff"] = -reversed_df["point_diff"]
    for col in base_feature_names:
        diff_col = f"{col}_diff"
        if diff_col in reversed_df.columns:
            reversed_df[diff_col] = -reversed_df[diff_col]
    # Interaction sign flips too (both factors flip → product stays same sign, but
    # we want it symmetric, so re-compute rather than negate)
    reversed_df = _add_interactions(reversed_df)

    combined = pd.concat([matchups, reversed_df], ignore_index=True)
    return combined


def build_matchup_pair(
    season: int,
    team_low: int,
    team_high: int,
    season_feats: pd.DataFrame,
    base_feature_names=None,
):
    """Build one feature dict for a (season, team_low, team_high) prediction pair.

    Returns None if either team is missing from season_feats.
    Used by predict_submission.py.
    """
    if base_feature_names is None:
        base_feature_names = MATCHUP_BASE_FEATURES

    a = season_feats[season_feats["TeamID"] == team_low]
    b = season_feats[season_feats["TeamID"] == team_high]

    if a.empty or b.empty:
        return None

    a_row = a.iloc[0]
    b_row = b.iloc[0]

    rec = {
        "Season": season,
        "TeamLow": team_low,
        "TeamHigh": team_high,
    }
    for col in base_feature_names:
        a_val = float(a_row[col]) if col in a_row.index else np.nan
        b_val = float(b_row[col]) if col in b_row.index else np.nan
        rec[f"{col}_diff"] = a_val - b_val

    # Interaction feature
    r_diff = rec.get("strength_rating_diff", np.nan)
    s_diff = rec.get("seed_strength_diff", np.nan)
    rec["strength_seed_interaction"] = (
        r_diff * s_diff if (not np.isnan(r_diff) and not np.isnan(s_diff)) else np.nan
    )

    return rec
