"""
Survival (time-to-voluntary-exit) model with a classifier-style interface.

Why survival instead of a 12-month yes/no classifier
----------------------------------------------------
A classifier can only learn from snapshots whose full 12-month window has
passed, so the most recent year of data is unusable, and it treats "left in
month 1" and "left in month 11" the same. An accelerated-failure-time (AFT)
model learns time-to-exit and accepts censored rows ("still here after N
months"), so it uses every snapshot up to the training cutoff. In out-of-time
backtests it improved PR-AUC from ~0.25 to ~0.36 and precision in the top-10%
risk group from ~22% to ~29% (see README).

Interface
---------
`SurvivalAttritionModel.predict_proba(X)[:, 1]` = calibrated probability of a
voluntary exit within `horizon` months, so the UI, simulator and evaluation
code treat it like any binary classifier.

The risk score is linear in the booster margin (log of predicted time):

    score = (ln(horizon) − margin) / scale        # higher = sooner exit
    P(exit ≤ horizon) = sigmoid(a · score + b)    # Platt calibration

so per-feature contributions in log-odds space are exactly
`−a / scale × pred_contribs`, which `risk_contributions` returns for SHAP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from config.constants import (
    ML_RANDOM_STATE,
    ML_RECENCY_HALF_LIFE_YEARS,
    ML_TARGET_DEPARTURE_REASON,
    XGB_AFT_PARAMS,
    XGB_AFT_ROUNDS,
)

_DAYS_PER_MONTH = 30.44


def recency_weights(
    snapshots: pd.Series, reference: pd.Timestamp, half_life_years: float = ML_RECENCY_HALF_LIFE_YEARS
) -> np.ndarray:
    """Weight 1.0 at `reference`, halving every `half_life_years` further back."""
    age_years = (reference - snapshots).dt.days.to_numpy() / 365.25
    return 0.5 ** (np.clip(age_years, 0, None) / half_life_years)


def survival_targets(
    snapshots: pd.Series, outcomes: pd.DataFrame, cutoff: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray]:
    """AFT interval bounds (months from snapshot) using only what was known on `cutoff`.

    Voluntary exit on or before cutoff → exact event time (lower = upper).
    Anything else (still employed, other exit, exit after cutoff) → right-censored.
    """
    dep = outcomes["Departure Date"]
    known_exit = dep.notna() & (dep <= cutoff)
    voluntary = known_exit & (outcomes["Departure Reason Description"] == ML_TARGET_DEPARTURE_REASON)
    end = dep.where(known_exit, cutoff)
    t = ((end - snapshots).dt.days / _DAYS_PER_MONTH).clip(lower=0.25).to_numpy()
    return t, np.where(voluntary.to_numpy(), t, np.inf)


class SurvivalAttritionModel:
    """XGBoost AFT survival model exposing P(voluntary exit within horizon)."""

    def __init__(self, booster: xgb.Booster, feature_names: list[str], horizon_months: int):
        self.booster = booster
        self.feature_names = list(feature_names)
        self.horizon_months = horizon_months
        self.scale = float(XGB_AFT_PARAMS["aft_loss_distribution_scale"])
        # Before calibration: logistic approximation of the normal CDF
        self.calib_a, self.calib_b = 1.7, 0.0

    # ---- scoring ---------------------------------------------------
    def _dmatrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(X[self.feature_names], feature_names=self.feature_names)

    def risk_score(self, X: pd.DataFrame) -> np.ndarray:
        margin = self.booster.predict(self._dmatrix(X), output_margin=True)
        return (np.log(self.horizon_months) - margin) / self.scale

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(self.calib_a * self.risk_score(X) + self.calib_b)))
        return np.column_stack([1.0 - p, p])

    def predicted_months(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted median months until a voluntary exit."""
        return self.booster.predict(self._dmatrix(X))

    # ---- calibration -----------------------------------------------
    def calibrate(self, X: pd.DataFrame, y: pd.Series) -> "SurvivalAttritionModel":
        """Platt-scale the risk score on fully observed horizon labels."""
        return self.calibrate_scores(self.risk_score(X), y)

    def calibrate_scores(self, scores: np.ndarray, y) -> "SurvivalAttritionModel":
        y = np.asarray(y)
        if len(np.unique(y)) > 1:
            lr = LogisticRegression(C=1e6).fit(scores.reshape(-1, 1), y)
            self.calib_a, self.calib_b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    # ---- explanations ----------------------------------------------
    def risk_contributions(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Exact per-feature contributions to the calibrated log-odds of exiting.

        Positive = pushes risk up. Returns (N×F array, expected value).
        """
        contribs = self.booster.predict(self._dmatrix(X), pred_contribs=True)
        k = -self.calib_a / self.scale
        bias = self.calib_a * (np.log(self.horizon_months) - contribs[0, -1]) / self.scale + self.calib_b
        return contribs[:, :-1] * k, float(bias)

    @property
    def feature_importances_(self) -> np.ndarray:
        gain = self.booster.get_score(importance_type="total_gain")
        imp = np.array([gain.get(f, 0.0) for f in self.feature_names])
        return imp / imp.sum() if imp.sum() else imp


def fit_survival(
    X: pd.DataFrame,
    snapshots: pd.Series,
    outcomes: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_months: int,
    seed: int = ML_RANDOM_STATE,
) -> SurvivalAttritionModel | None:
    """Train on snapshots strictly before `cutoff`, outcomes censored at `cutoff`."""
    mask = (snapshots < cutoff).to_numpy()
    if mask.sum() == 0:
        return None
    lo, hi = survival_targets(snapshots[mask], outcomes.loc[mask], cutoff)
    if not np.isfinite(hi).any():
        return None   # no events before cutoff
    dtrain = xgb.DMatrix(
        X.loc[mask], feature_names=list(X.columns),
        weight=recency_weights(snapshots[mask], snapshots[mask].max()),
    )
    dtrain.set_float_info("label_lower_bound", lo)
    dtrain.set_float_info("label_upper_bound", hi)
    booster = xgb.train({**XGB_AFT_PARAMS, "seed": seed}, dtrain, num_boost_round=XGB_AFT_ROUNDS)
    return SurvivalAttritionModel(booster, list(X.columns), horizon_months)
