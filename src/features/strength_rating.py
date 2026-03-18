from __future__ import annotations

"""Bradley-Terry team strength ratings via logistic regression.

Checklist Step 5 — CRITICAL.

Model: logit(P(A beats B)) = r_A - r_B

For each season we fit a logistic regression with no intercept where the
design matrix has one column per team.  For game i (winner W, loser L) the
row is: +1 in column W, -1 in column L, label = 1.  This directly encodes
the log-odds difference and recovers team-level strength ratings as the
fitted coefficients.

KEY RULES (checklist):
- Only REGULAR SEASON games are used (NO tournament games).
- One rating per team per season.
"""

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression


def compute_strength_ratings(regular_season: pd.DataFrame) -> pd.DataFrame:
    """Fit a Bradley-Terry strength model per season.

    Parameters
    ----------
    regular_season:
        Regular season results with at minimum columns:
        Season, WTeamID, LTeamID.
        Must NOT include tournament games.

    Returns
    -------
    DataFrame with columns: Season, TeamID, strength_rating.
    Ratings are on a log-odds scale; larger = stronger team.
    """
    all_rows: list[dict] = []

    for season, season_games in regular_season.groupby("Season"):
        season_games = season_games.reset_index(drop=True)

        # Build ordered list of all teams this season.
        teams = sorted(
            set(season_games["WTeamID"].tolist() + season_games["LTeamID"].tolist())
        )
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)
        n_games = len(season_games)

        if n_games == 0 or n_teams < 2:
            continue

        # Build sparse design matrix with mirrored rows so sklearn sees both classes.
        # For each game (W, L) we add TWO rows:
        #   Row A: W=+1, L=-1, label=1  (winner beats loser)
        #   Row B: W=-1, L=+1, label=0  (loser loses to winner)
        # This gives the full Bradley-Terry likelihood and satisfies sklearn's
        # requirement for at least 2 classes in the training data.
        n_rows = 2 * n_games
        X = lil_matrix((n_rows, n_teams), dtype=np.float32)
        y = np.empty(n_rows, dtype=np.int32)

        for game_idx, row in season_games.iterrows():
            w = int(row["WTeamID"])
            l = int(row["LTeamID"])
            # Forward row: winner wins
            X[game_idx, team_to_idx[w]] = 1.0
            X[game_idx, team_to_idx[l]] = -1.0
            y[game_idx] = 1
            # Mirror row: loser loses
            mirror = game_idx + n_games
            X[mirror, team_to_idx[w]] = -1.0
            X[mirror, team_to_idx[l]] = 1.0
            y[mirror] = 0

        # Fit logistic regression without intercept.
        # C=1.0 provides L2 regularisation that keeps ratings finite and
        # comparable across seasons.
        model = LogisticRegression(
            fit_intercept=False,
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=2026,
        )
        model.fit(X.tocsr(), y)

        coefs = model.coef_[0]
        for team, idx in team_to_idx.items():
            all_rows.append(
                {
                    "Season": int(season),
                    "TeamID": int(team),
                    "strength_rating": float(coefs[idx]),
                }
            )

    return pd.DataFrame(all_rows, columns=["Season", "TeamID", "strength_rating"])
