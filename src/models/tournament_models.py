from __future__ import annotations

"""Tournament matchup models.

Two model architectures are available:

1. MarginRegressionTournamentModel  (primary — winner's approach)
   - LightGBM regressor predicts point differential (continuous target).
   - A spline fitted on out-of-fold (predicted_margin, actual_win) pairs
     converts margins to win probabilities.  This is far more powerful than
     direct probability prediction because the margin carries ~10x more signal
     per game than a binary win/loss label.

2. CalibratedTournamentModel  (legacy — classification approach)
   - LightGBM classifier + isotonic calibration + LR blend.
   - Kept for reference / fallback.

Both expose the same .predict_proba(X) interface, so predict_submission.py
works with either without modification.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


# ── Shared config ─────────────────────────────────────────────────────────────

# Features used by the logistic regression "prior" in the classification model.
LR_FEATURES = ["strength_rating_diff", "seed_strength_diff"]


@dataclass
class TournamentModelConfig:
    """Configuration for tournament matchup models (both architectures)."""

    features: Sequence[str]
    lgbm_params: dict = field(default_factory=dict)
    lr_weight: float = 0.25     # blend weight for LR in classification model
    margin_clip: float = 25.0   # clip margins at ±N before spline lookup


# ══════════════════════════════════════════════════════════════════════════════
# 1. Margin Regression Model  (primary)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarginRegressionTournamentModel:
    """LightGBM regressor + spline calibration.

    Attributes
    ----------
    lgbm:        Fitted LGBMRegressor predicting point differential.
    spline:      UnivariateSpline mapping clipped margin → P(win).
                 Fitted on out-of-fold predictions from the training pipeline.
                 If None, a sigmoid fallback is used.
    features:    Feature column names used during training.
    margin_clip: Clip margins at ±clip before applying the spline.
    """

    lgbm: LGBMRegressor
    spline: Any          # UnivariateSpline | None
    features: list[str]
    margin_clip: float = 25.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return win probabilities for the positive class (TeamLow wins)."""
        margins = self.lgbm.predict(X.astype(float))
        clipped = np.clip(margins, -self.margin_clip, self.margin_clip)

        if self.spline is not None:
            probs = np.array(self.spline(clipped), dtype=float)
            return np.clip(probs, 0.0, 1.0)

        # Fallback: soft sigmoid when no spline is available.
        return 1.0 / (1.0 + np.exp(-clipped / 8.0))


def fit_spline_calibrator(
    oof_margins: np.ndarray,
    oof_labels: np.ndarray,
    margin_clip: float = 25.0,
    k: int = 5,
) -> UnivariateSpline:
    """Fit a UnivariateSpline mapping predicted margin → win probability.

    Parameters
    ----------
    oof_margins:  Out-of-fold predicted margins (any order).
    oof_labels:   Corresponding binary win labels (0 or 1).
    margin_clip:  Clip margins at ±this value before fitting.
    k:            Spline degree (5 = quintic, same as original winner's approach).

    Returns
    -------
    Fitted UnivariateSpline ready for predict calls.
    """
    clipped = np.clip(oof_margins, -margin_clip, margin_clip)

    # Sort by predicted margin (required by UnivariateSpline).
    sort_idx = np.argsort(clipped)
    x_sorted = clipped[sort_idx]
    y_sorted = oof_labels[sort_idx].astype(float)

    spline = UnivariateSpline(x_sorted, y_sorted, k=k)
    return spline


def train_margin_model(
    train_df: pd.DataFrame,
    config: TournamentModelConfig,
) -> LGBMRegressor:
    """Train a LightGBM regressor on point differential.

    Returns the raw fitted LGBMRegressor (without spline — the spline is
    fitted externally on OOF predictions across multiple folds).
    """
    features = list(config.features)
    X_train = train_df[features].values.astype(float)
    y_train = train_df["point_diff"].values.astype(float)

    lgbm_params = {k: v for k, v in config.lgbm_params.items()}

    lgbm = LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train)
    return lgbm


# ══════════════════════════════════════════════════════════════════════════════
# 2. Classification Model  (legacy / fallback)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CalibratedTournamentModel:
    """LightGBM classifier + LR blend + isotonic calibration (legacy).

    Attributes
    ----------
    lgbm:        Fitted LGBMClassifier (uncalibrated).
    lr_model:    Fitted LogisticRegression on LR_FEATURES.  None if blend disabled.
    lr_scaler:   StandardScaler fitted on training LR features.
    lr_weight:   Blend weight for the LR (e.g. 0.25 → 75% LGBM + 25% LR).
    calibrator:  Fitted IsotonicRegression.  None if no calibration was applied.
    features:    Ordered list of feature column names used during training.
    """

    lgbm: LGBMClassifier
    lr_model: LogisticRegression | None
    lr_scaler: StandardScaler | None
    lr_weight: float
    calibrator: IsotonicRegression | None
    features: list[str]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated blended win probabilities for the positive class."""
        lgbm_pred = self.lgbm.predict_proba(X.astype(float))[:, 1]

        if self.lr_model is not None and self.lr_scaler is not None:
            lr_indices = [
                self.features.index(f) for f in LR_FEATURES if f in self.features
            ]
            if lr_indices:
                X_lr = X[:, lr_indices].copy()
                X_lr = np.where(np.isnan(X_lr), 0.0, X_lr)
                X_lr = self.lr_scaler.transform(X_lr)
                lr_pred = self.lr_model.predict_proba(X_lr)[:, 1]
                blended = (1 - self.lr_weight) * lgbm_pred + self.lr_weight * lr_pred
            else:
                blended = lgbm_pred
        else:
            blended = lgbm_pred

        if self.calibrator is not None:
            return self.calibrator.predict(blended)
        return blended


def train_tournament_model(
    train_df: pd.DataFrame,
    config: TournamentModelConfig,
    calib_df: pd.DataFrame | None = None,
) -> CalibratedTournamentModel:
    """Train a LightGBM + LR blended classifier and optionally calibrate."""
    features = list(config.features)

    X_train = train_df[features].values.astype(float)
    y_train = train_df["label"].values.astype(int)

    lgbm = LGBMClassifier(**config.lgbm_params)
    lgbm.fit(X_train, y_train)

    lr_model: LogisticRegression | None = None
    lr_scaler: StandardScaler | None = None
    lr_weight = config.lr_weight

    if lr_weight > 0:
        lr_indices = [i for i, f in enumerate(features) if f in LR_FEATURES]
        if len(lr_indices) > 0:
            X_lr_train = X_train[:, lr_indices]
            lr_scaler = StandardScaler()
            X_lr_scaled = lr_scaler.fit_transform(X_lr_train)
            lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=2026)
            lr_model.fit(X_lr_scaled, y_train)

    calibrator: IsotonicRegression | None = None

    if calib_df is not None and not calib_df.empty:
        X_calib = calib_df[features].values.astype(float)
        y_calib = calib_df["label"].values.astype(int)

        lgbm_raw = lgbm.predict_proba(X_calib)[:, 1]
        if lr_model is not None and lr_scaler is not None:
            lr_indices = [i for i, f in enumerate(features) if f in LR_FEATURES]
            X_lr_calib = X_calib[:, lr_indices].copy()
            X_lr_calib = np.where(np.isnan(X_lr_calib), 0.0, X_lr_calib)
            X_lr_calib = lr_scaler.transform(X_lr_calib)
            lr_raw = lr_model.predict_proba(X_lr_calib)[:, 1]
            blended_raw = (1 - lr_weight) * lgbm_raw + lr_weight * lr_raw
        else:
            blended_raw = lgbm_raw

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(blended_raw, y_calib)

    return CalibratedTournamentModel(
        lgbm=lgbm,
        lr_model=lr_model,
        lr_scaler=lr_scaler,
        lr_weight=lr_weight,
        calibrator=calibrator,
        features=features,
    )
