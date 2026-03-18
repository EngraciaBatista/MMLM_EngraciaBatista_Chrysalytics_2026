from __future__ import annotations

"""Season-based cross-validation for the tournament matchup model.

Checklist Step 10.

Strategy:
  - For each test_season, train on all seasons in [min_season, test_season-1].
  - Validate on test_season.
  - The model trained in each fold uses the *previous* fold's validation set as
    its calibration set (leave-one-season-out calibration proxy).
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import brier_score, log_loss_safe
from ..models.tournament_models import (
    CalibratedTournamentModel,
    TournamentModelConfig,
    train_tournament_model,
)


@dataclass
class FoldResult:
    test_season: int
    brier: float
    log_loss: float
    n_games: int


def season_based_cv(
    data: pd.DataFrame,
    config: TournamentModelConfig,
    test_seasons: Sequence[int],
    min_season: int,
) -> list[FoldResult]:
    """Season-based cross-validation for tournament models.

    Trains on seasons [min_season, test_season-1], validates on test_season.
    No calibration is applied during CV (calibration is fitted on the final
    model using the held-out last season before full-data refit).

    Parameters
    ----------
    data:
        Mirrored matchup training frame with 'Season', feature diff columns, 'label'.
    config:
        Model configuration.
    test_seasons:
        Seasons to use as validation folds, in chronological order.
    min_season:
        Earliest season to include in training data.

    Returns
    -------
    List of FoldResult, one per test season.
    """
    results: list[FoldResult] = []

    for test_season in sorted(test_seasons):
        train_mask = (data["Season"] >= min_season) & (data["Season"] < test_season)
        test_mask = data["Season"] == test_season

        train_df = data.loc[train_mask].reset_index(drop=True)
        test_df = data.loc[test_mask].reset_index(drop=True)

        if train_df.empty or test_df.empty:
            continue

        # No calibration set during CV (pure hold-out evaluation).
        model: CalibratedTournamentModel = train_tournament_model(
            train_df, config, calib_df=None
        )

        X_test = test_df[list(config.features)].values
        y_test = test_df["label"].values.astype(int)

        p = model.predict_proba(X_test)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        brier = brier_score(y_test, p)
        ll = log_loss_safe(y_test, p)
        results.append(
            FoldResult(
                test_season=test_season,
                brier=brier,
                log_loss=ll,
                n_games=len(test_df),
            )
        )

    return results
