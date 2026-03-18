from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

League = Literal["M", "W"]


@dataclass
class EloConfig:
    base_rating: float = 1500.0
    k_factor: float = 20.0
    mov_scale: float = 400.0
    regression_factor: float = 0.75
    home_advantage: float = 0.0


def _expected_prob(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / 400.0))


def _mov_multiplier(margin: float, rating_diff: float) -> float:
    # Common logistic-style MOV scaling used in Elo variants.
    return np.log(abs(margin) + 1.0) * (2.2 / (2.2 + abs(rating_diff) * 0.001))


def compute_elo(
    detailed_results: pd.DataFrame,
    config: EloConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-game and per-team-season Elo ratings.

    Returns:
        game_elos: one row per game with pre-game Elo ratings and win probability.
        season_elos: final per-team-per-season Elo rating.
    """
    df = detailed_results.sort_values(["Season", "DayNum"]).copy()

    ratings: dict[int, float] = {}
    records = []

    current_season = None

    for _, row in df.iterrows():
        season = int(row["Season"])
        if current_season is None or season != current_season:
            # Regress all ratings toward the base at the beginning of a new season.
            current_season = season
            for team_id in list(ratings.keys()):
                ratings[team_id] = config.base_rating + config.regression_factor * (
                    ratings[team_id] - config.base_rating
                )

        w = int(row["WTeamID"])
        l = int(row["LTeamID"])
        margin = float(row["WScore"] - row["LScore"])
        loc = row.get("WLoc", "N")

        rw = ratings.get(w, config.base_rating)
        rl = ratings.get(l, config.base_rating)

        # Apply home-court advantage to ratings.
        if config.home_advantage != 0.0:
            if loc == "H":
                rw_eff = rw + config.home_advantage
                rl_eff = rl
            elif loc == "A":
                rw_eff = rw
                rl_eff = rl + config.home_advantage
            else:
                rw_eff = rw
                rl_eff = rl
        else:
            rw_eff = rw
            rl_eff = rl

        exp_w = _expected_prob(rw_eff, rl_eff)
        rating_diff = rw_eff - rl_eff
        mult = _mov_multiplier(margin, rating_diff)

        delta = config.k_factor * mult * (1.0 - exp_w)
        rw_new = rw + delta
        rl_new = rl - delta

        ratings[w] = rw_new
        ratings[l] = rl_new

        records.append(
            {
                "Season": season,
                "DayNum": int(row["DayNum"]),
                "WTeamID": w,
                "LTeamID": l,
                "W_Rating_Before": rw,
                "L_Rating_Before": rl,
                "EloWinProb": exp_w,
                "Margin": margin,
            }
        )

    game_elos = pd.DataFrame.from_records(records)

    # Convert ratings dict to season-level DataFrame by replaying last rating per season/team.
    # To avoid replaying, reconstruct from records.
    team_records: dict[tuple[int, int], float] = {}
    for rec in records:
        season = rec["Season"]
        w = rec["WTeamID"]
        l = rec["LTeamID"]
        team_records[(season, w)] = ratings.get(w, config.base_rating)
        team_records[(season, l)] = ratings.get(l, config.base_rating)

    season_rows = [
        {"Season": s, "TeamID": t, "elo_rating": r} for (s, t), r in team_records.items()
    ]
    season_elos = pd.DataFrame(season_rows)

    return game_elos, season_elos

