"""SHAP explanations: global, dependence, per-employee.

When the trained model is wrapped in `CalibratedClassifierCV`, the wrapper
itself is not a tree model — TreeExplainer needs the underlying XGBoost
estimator. This module unwraps it transparently.

Important caveat: SHAP values come from the *uncalibrated* base model. The
calibration layer monotonically transforms probabilities, so the *direction*
and *relative importance* of features remains the same — but the magnitudes
are on the pre-calibration log-odds scale. Always interpret SHAP rankings
qualitatively, never as exact probability deltas.

Why two implementation paths exist
----------------------------------
`shap.TreeExplainer(xgb_model)` works by reading XGBoost's serialised binary
model format. SHAP's parser expects the OLD binary layout (xgboost ≤ 1.x).
With modern XGBoost (2.x+), the binary often parses correctly but sometimes
SHAP grabs garbage bytes when reading internal strings, producing:

    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xXX

This is a known SHAP issue (github.com/shap/shap/issues/1215, #2764).

We try `shap.TreeExplainer` first because it gives proper `Explainer` objects
that work with `shap.summary_plot` etc. If it fails, we fall back to
XGBoost's native `predict(pred_contribs=True)` which computes the same SHAP
values via the model's own code path — no binary parsing involved.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from config.constants import SHAP_SAMPLE_SIZE_GLOBAL

log = logging.getLogger(__name__)


@dataclass
class ShapBundle:
    explainer: shap.TreeExplainer | None     # may be None if we used the fallback
    shap_values: np.ndarray                  # for class 1 if binary
    X_sample: pd.DataFrame
    expected_value: float


def _unwrap_to_tree_model(model):
    """Find the underlying tree model inside any wrapper.

    Handles:
    - CalibratedClassifierCV(FrozenEstimator(xgb))     # sklearn ≥1.6
    - CalibratedClassifierCV(xgb, cv='prefit')         # sklearn ≤1.7
    - Plain XGBClassifier
    """
    def _is_frozen(obj) -> bool:
        return type(obj).__name__ == "FrozenEstimator"

    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        cal = model.calibrated_classifiers_[0]
        for attr in ("estimator", "base_estimator"):
            if hasattr(cal, attr):
                inner = getattr(cal, attr)
                if inner is None:
                    continue
                if _is_frozen(inner) and hasattr(inner, "estimator"):
                    return inner.estimator
                return inner

    if _is_frozen(model) and hasattr(model, "estimator"):
        return model.estimator

    return model


def _select_class(values, expected_value, class_idx: int = 1):
    """Pick the right slice if SHAP returns a list (multiclass)."""
    if isinstance(values, list):
        idx = class_idx if len(values) > 1 else 0
        sv = values[idx]
        ev = expected_value[idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        return sv, ev
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0]
    return values, expected_value


def _xgb_native_shap(xgb_model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Compute SHAP values via XGBoost's own pred_contribs.

    Returns (shap_values_shape_NxF, expected_value).
    `pred_contribs=True` returns a (N, F+1) array where the last column is the
    bias term (the model's expected output for the dataset).
    """
    import xgboost as xgb
    booster = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model
    dmat = xgb.DMatrix(X.values, feature_names=list(X.columns))
    contribs = booster.predict(dmat, pred_contribs=True)
    # Last column = bias / expected value (same for every row)
    expected_value = float(contribs[0, -1])
    shap_values = contribs[:, :-1]
    return shap_values, expected_value


def _safe_tree_explainer(xgb_model, X: pd.DataFrame):
    """Try shap.TreeExplainer, fall back to XGBoost-native pred_contribs.

    Returns (explainer_or_None, shap_values, expected_value).
    explainer is None when the fallback path was used.
    """
    if hasattr(xgb_model, "risk_contributions"):
        # SurvivalAttritionModel: exact contributions to the calibrated log-odds
        # of exiting (sign already in the "raises risk" direction)
        sv, ev = xgb_model.risk_contributions(X)
        return None, sv, ev
    try:
        explainer = shap.TreeExplainer(xgb_model)
        raw = explainer.shap_values(X, check_additivity=False)
        sv, ev = _select_class(raw, explainer.expected_value)
        return explainer, sv, float(ev)
    except (UnicodeDecodeError, KeyError, AttributeError, IndexError) as e:
        log.warning(
            "shap.TreeExplainer failed (%s: %s). "
            "Falling back to XGBoost-native pred_contribs (no SHAP plots, only values).",
            type(e).__name__, e,
        )
        sv, ev = _xgb_native_shap(xgb_model, X)
        return None, sv, ev


def compute_global_shap(
    model,
    X_train: pd.DataFrame,
    sample_size: int = SHAP_SAMPLE_SIZE_GLOBAL,
    random_state: int = 42,
) -> ShapBundle:
    """Compute SHAP on a sample of X_train. Returns class-1 slice for binary models."""
    tree_model = _unwrap_to_tree_model(model)
    X_num = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    n = min(sample_size, X_num.shape[0])
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X_num.shape[0], size=n, replace=False)
    X_sample = X_num.iloc[idx].copy()

    explainer, sv, ev = _safe_tree_explainer(tree_model, X_sample)
    return ShapBundle(
        explainer=explainer,
        shap_values=sv,
        X_sample=X_sample,
        expected_value=ev,
    )


def top_drivers_per_employee(
    model,
    X_likely: pd.DataFrame,
    top_n: int = 3,
    display_names: dict[str, str] | None = None,
) -> list[str]:
    """Return a list of strings: 'feature1 (↑), feature2 (↓), ...' per row.

    If `display_names` is provided (safe-name → original-name map from
    FeatureEncoder), the strings use the original (e.g. Greek) labels.
    """
    tree_model = _unwrap_to_tree_model(model)
    X_num = X_likely.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    _, shap_arr, _ = _safe_tree_explainer(tree_model, X_num)

    feature_names = X_num.columns.tolist()
    label_of = (lambda name: display_names.get(name, name)) if display_names else (lambda name: name)
    out: list[str] = []
    for i in range(X_num.shape[0]):
        row = shap_arr[i]
        top_idx = np.argsort(-np.abs(row))[:top_n]
        out.append(
            ", ".join(
                f"{label_of(feature_names[j])} ({'↑' if row[j] > 0 else '↓'})"
                for j in top_idx
            )
        )
    return out


def shap_for_single_row(model, row_X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values + expected value for a single employee."""
    tree_model = _unwrap_to_tree_model(model)
    row_num = row_X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_num.columns = row_num.columns.astype(str)
    _, sv, ev = _safe_tree_explainer(tree_model, row_num)
    return sv[0], ev
