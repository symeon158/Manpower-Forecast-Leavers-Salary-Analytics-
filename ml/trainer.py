"""
Training / evaluation for the survival attrition model (see ml/survival.py,
ml/data_prep.py).

Three models are fitted, each only on what was known at its cutoff (training
rows = snapshots before the cutoff, outcomes censored at the cutoff):

1. **Validation model** — cutoff V = T − horizon. Scored on snapshot V (whose
   12-month outcomes are known by T). Its predictions fit the probability
   calibration and pick the decision threshold, so neither is tuned on the test.
2. **Evaluation model** — cutoff T = most recent fully-labelled snapshot.
   Scored on snapshot T with the validation calibration: a genuine "train on
   the past, predict the next 12 months" estimate. The reported test metrics.
3. **Final model** — cutoff = export date (all snapshots). Scores current
   employees, drives SHAP and the What-If simulator. Its calibration is
   re-fitted on the evaluation model's test-snapshot predictions (the most
   recent out-of-sample evidence).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.constants import (
    ML_BACKTEST_SNAPSHOTS,
    ML_BACKTEST_SPACING_MONTHS,
    ML_HORIZON_MONTHS,
    ML_RECENCY_HALF_LIFE_YEARS,
    XGB_AFT_PARAMS,
    XGB_AFT_ROUNDS,
)
from ml.evaluation import sweep_thresholds
from ml.survival import SurvivalAttritionModel, fit_survival


# --------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------
@dataclass
class TrainResult:
    model: SurvivalAttritionModel       # final model (all snapshots, censored at export)
    base_model: SurvivalAttritionModel  # same object; kept for SHAP call sites
    best_params: dict
    X_train: pd.DataFrame               # rows the evaluation model was trained on
    X_test: pd.DataFrame                # test snapshot rows (fully labelled)
    y_train: pd.Series                  # fully-labelled subset of the training rows
    y_test: pd.Series
    test_proba: np.ndarray              # evaluation-model predictions on X_test
    threshold: float                    # F1-optimal on the validation snapshot
    train_columns: list[str]
    test_snapshot: pd.Timestamp
    val_snapshot: pd.Timestamp
    backtest: pd.DataFrame              # per-snapshot out-of-time metrics


@dataclass
class EvalResult:
    auc: float
    precision: float
    recall: float
    f1: float
    classification_report: str
    confusion_matrix: np.ndarray
    threshold: float


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def precision_at_top(y_true: np.ndarray, proba: np.ndarray, fraction: float = 0.10) -> float:
    """Share of actual leavers among the `fraction` highest-risk employees."""
    k = max(1, int(round(len(proba) * fraction)))
    top = np.argsort(-proba)[:k]
    return float(np.asarray(y_true)[top].mean())


def _labelled_rows(y: pd.Series, snapshots: pd.Series, t: pd.Timestamp) -> np.ndarray:
    return ((snapshots == t) & y.notna()).to_numpy()


def _backtest_row(t, y_t: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "Snapshot": t.date(),
        "Employees": int(len(y_t)),
        "Left voluntarily (12m)": int(y_t.sum()),
        "Base rate": float(y_t.mean()),
        "ROC AUC": float(roc_auc_score(y_t, proba)),
        "PR AUC": float(average_precision_score(y_t, proba)),
        "Precision @ top 10%": precision_at_top(y_t, proba),
    }


# --------------------------------------------------------------------
# Training
# --------------------------------------------------------------------
def train_attrition_model(
    X: pd.DataFrame,
    y: pd.Series,
    snapshots: pd.Series,
    outcomes: pd.DataFrame,
    export_date: pd.Timestamp,
    horizon: int = ML_HORIZON_MONTHS,
) -> TrainResult:
    """`outcomes` needs 'Departure Date' and 'Departure Reason Description' aligned to X."""
    X, y, snapshots = X.reset_index(drop=True), y.reset_index(drop=True), snapshots.reset_index(drop=True)
    outcomes = outcomes.reset_index(drop=True)
    H = pd.DateOffset(months=horizon)

    labelled_snaps = sorted(snapshots[y.notna()].unique())
    if not labelled_snaps:
        raise ValueError("Δεν υπάρχει snapshot με πλήρη 12μηνη παρακολούθηση.")
    test_snap = labelled_snaps[-1]
    earlier = [s for s in labelled_snaps if s <= test_snap - H]
    if not earlier:
        raise ValueError("Not enough history for an out-of-time test (need > 2 × horizon of data).")
    val_snap = earlier[-1]

    def fit(cutoff):
        model = fit_survival(X, snapshots, outcomes, cutoff, horizon)
        if model is None:
            raise ValueError(f"Δεν υπάρχουν εθελοντικές αποχωρήσεις πριν από {cutoff:%Y-%m-%d}.")
        return model

    # 1. Validation: calibration + threshold
    val_model = fit(val_snap)
    vmask = _labelled_rows(y, snapshots, val_snap)
    val_model.calibrate(X[vmask], y[vmask])
    val_proba = val_model.predict_proba(X[vmask])[:, 1]
    threshold = (
        sweep_thresholds(y[vmask].to_numpy(), val_proba).optimal_f1_threshold
        if y[vmask].nunique() > 1 else 0.5
    )

    # 2. Out-of-time test, using the validation calibration
    eval_model = fit(test_snap)
    eval_model.calib_a, eval_model.calib_b = val_model.calib_a, val_model.calib_b
    tmask = _labelled_rows(y, snapshots, test_snap)
    test_proba = eval_model.predict_proba(X[tmask])[:, 1]

    # 3. Final model, calibrated on the latest out-of-sample evidence
    final_model = fit(export_date)
    final_model.calibrate_scores(eval_model.risk_score(X[tmask]), y[tmask])

    # Backtest table (re-uses the two models above where the dates coincide)
    rows = []
    for k in reversed(range(ML_BACKTEST_SNAPSHOTS)):
        t = test_snap - pd.DateOffset(months=k * ML_BACKTEST_SPACING_MONTHS)
        mask = _labelled_rows(y, snapshots, t)
        if mask.sum() == 0 or y[mask].nunique() < 2:
            continue
        model = {test_snap: eval_model, val_snap: val_model}.get(t) or fit_survival(
            X, snapshots, outcomes, t, horizon
        )
        if model is not None:
            rows.append(_backtest_row(t, y[mask].to_numpy(), model.risk_score(X[mask])))

    train_mask = (snapshots < test_snap).to_numpy()
    return TrainResult(
        model=final_model,
        base_model=final_model,
        best_params={
            **XGB_AFT_PARAMS, "num_boost_round": XGB_AFT_ROUNDS,
            "recency_half_life_years": ML_RECENCY_HALF_LIFE_YEARS,
            "calibration (a, b)": [round(final_model.calib_a, 4), round(final_model.calib_b, 4)],
        },
        X_train=X[train_mask],
        X_test=X[tmask],
        y_train=y[train_mask & y.notna().to_numpy()],
        y_test=y[tmask].astype(int),
        test_proba=test_proba,
        threshold=float(threshold),
        train_columns=X.columns.tolist(),
        test_snapshot=test_snap,
        val_snapshot=val_snap,
        backtest=pd.DataFrame(rows),
    )


# --------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------
def evaluate(y_test: pd.Series, proba: np.ndarray, threshold: float) -> EvalResult:
    pred = (proba >= threshold).astype(int)
    return EvalResult(
        auc=float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan"),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
        classification_report=classification_report(y_test, pred, digits=3, zero_division=0),
        confusion_matrix=confusion_matrix(y_test, pred, labels=[0, 1]),
        threshold=threshold,
    )


def fairness_table(
    y_true: np.ndarray, proba: np.ndarray, groups: pd.Series, threshold: float, min_size: int = 30
) -> pd.DataFrame:
    """Flag rate, actual exit rate and recall per group at `threshold`.

    The model does not use these attributes, but a large gap in flag rate that
    is not explained by a gap in actual exit rate would mean proxies are at work.
    """
    pred = proba >= threshold
    rows = []
    for g in pd.unique(groups.dropna()):
        m = (groups == g).to_numpy()
        if m.sum() < min_size:
            continue
        pos = y_true[m] == 1
        rows.append({
            "Group": str(g),
            "N": int(m.sum()),
            "Actual exit rate": float(y_true[m].mean()),
            "Flag rate": float(pred[m].mean()),
            "Mean predicted risk": float(proba[m].mean()),
            "Recall": float(pred[m][pos].mean()) if pos.any() else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------
def align_columns(X: pd.DataFrame, train_columns: list[str]) -> pd.DataFrame:
    """Add missing columns as 0, drop extras, in training order.

    Mostly redundant now that FeatureEncoder produces a stable column set, but
    kept as a defensive fallback.
    """
    out = X.copy()
    missing = set(train_columns) - set(out.columns)
    for c in missing:
        out[c] = 0
    return out[train_columns]


def predict_active_employees(
    model,
    X_current: pd.DataFrame,
    train_columns: list[str],
    threshold: float,
) -> pd.DataFrame:
    """Score currently active employees (features as of the export date)."""
    X_aligned = align_columns(X_current, train_columns)
    out = pd.DataFrame(index=X_current.index)
    out["Attrition_Probability"] = model.predict_proba(X_aligned)[:, 1]
    out["Predicted_Attrition"] = (out["Attrition_Probability"] >= threshold).astype(int)
    if hasattr(model, "predicted_months"):
        out["Predicted_Months_To_Exit"] = model.predicted_months(X_aligned)
    return out
