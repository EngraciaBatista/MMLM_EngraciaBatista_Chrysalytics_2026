from __future__ import annotations

"""Margin-based team quality rating (Massey / OLS style).

Conceptual difference vs. Bradley-Terry (strength_rating.py):
  - Bradley-Terry: logit(P(win)) = r_A - r_B   → fits on binary win/loss labels
  - Margin quality: E(margin) = q_A - q_B       → fits on actual point differentials

Using the actual margin gives ~10x more signal per game because a +30 win
carries very different information than a +1 win.  The coefficient for each
team is its "quality" — how many points better/worse it is than the average.

Implementation:
  For each (season), build a sparse design matrix:
    row i: X[i, winner_idx] = +1, X[i, loser_idx] = -1, y[i] = WScore - LScore
  Fit Ridge(fit_intercept=False) to solve for quality coefficients.
  The winner's coefficient is its margin quality rating.

  Ridge regularisation (alpha) keeps ratings finite for sparse schedules and
  shrinks extreme values for teams with few games.
"""

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.linear_model import Ridge


def compute_margin_quality(
    regular_season: pd.DataFrame,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Compute margin-based team quality ratings per season.

    Parameters
    ----------
    regular_season:
        Raw Kaggle regular season results (WTeamID, LTeamID, WScore, LScore, Season).
    alpha:
        Ridge regularisation strength (higher = more shrinkage toward 0).

    Returns
    -------
    DataFrame with columns: Season, TeamID, margin_quality
    """
    records = []

    for season, grp in regular_season.groupby("Season"):
        teams = pd.unique(
            pd.concat([grp["WTeamID"], grp["LTeamID"]])
        )
        team_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)
        n_games = len(grp)

        # Build design matrix: +1 for winner position, -1 for loser position.
        X = lil_matrix((n_games, n_teams), dtype=np.float32)
        y = np.empty(n_games, dtype=np.float32)

        for i, (_, row) in enumerate(grp.iterrows()):
            w = int(row["WTeamID"])
            l = int(row["LTeamID"])
            margin = float(row["WScore"] - row["LScore"])

            X[i, team_idx[w]] = 1.0
            X[i, team_idx[l]] = -1.0
            y[i] = margin

        model = Ridge(alpha=alpha, fit_intercept=False, max_iter=2000)
        model.fit(X.tocsr(), y)

        for team, idx in team_idx.items():
            records.append(
                {
                    "Season": season,
                    "TeamID": int(team),
                    "margin_quality": float(model.coef_[idx]),
                }
            )

    return pd.DataFrame(records)
