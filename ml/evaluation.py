"""
Model evaluation utilities.

These don't change what the model predicts. They give you trust signals so you
can answer questions like:
- How wide is the confidence interval on the AUC I'm reporting?
- Are my "70%" probabilities actually 70% in the long run?
- Does the model work as well for engineers as for operators?
- Am I underfit, overfit, or near the right complexity?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve

from config.constants import ML_BOOTSTRAP_ITERATIONS, ML_LEARNING_CURVE_POINTS


# --------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------
@dataclass
class ThresholdSweep:
    thresholds: np.ndarray
    f1_scores: np.ndarray
    precisions: np.ndarray
    recalls: np.ndarray
    optimal_f1_threshold: float
    optimal_fbeta_threshold: float        # default β=2 (recall-weighted)


def sweep_thresholds(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0) -> ThresholdSweep:
    """Compute precision/recall/F1 across all PR-curve thresholds.

    The returned `optimal_f1_threshold` is the threshold that maximises F1 on
    this set. The `optimal_fbeta_threshold` weights recall β times more heavily
    than precision (β=2 is recall-favouring; β=0.5 favours precision).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # PR curve returns one extra point with no threshold; drop it
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1 = (2 * precisions * recalls) / np.where(
        (precisions + recalls) > 0, precisions + recalls, 1
    )
    fbeta = ((1 + beta**2) * precisions * recalls) / np.where(
        (beta**2 * precisions + recalls) > 0, beta**2 * precisions + recalls, 1
    )

    if len(thresholds) == 0:
        return ThresholdSweep(
            thresholds=np.array([0.5]),
            f1_scores=np.array([0.0]),
            precisions=np.array([0.0]),
            recalls=np.array([0.0]),
            optimal_f1_threshold=0.5,
            optimal_fbeta_threshold=0.5,
        )

    f1_idx = int(np.nanargmax(f1))
    fb_idx = int(np.nanargmax(fbeta))
    return ThresholdSweep(
        thresholds=thresholds,
        f1_scores=f1,
        precisions=precisions,
        recalls=recalls,
        optimal_f1_threshold=float(thresholds[f1_idx]),
        optimal_fbeta_threshold=float(thresholds[fb_idx]),
    )


# --------------------------------------------------------------------
# Bootstrap confidence intervals
# --------------------------------------------------------------------
@dataclass
class BootstrapCI:
    auc_mean: float
    auc_low: float
    auc_high: float
    f1_mean: float
    f1_low: float
    f1_high: float
    n_iterations: int


def bootstrap_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    n_iter: int = ML_BOOTSTRAP_ITERATIONS,
    random_state: int = 42,
) -> BootstrapCI:
    """Resample the test set with replacement and compute 95% CI on AUC and F1.

    A wide CI tells you the test set is too small to discriminate models.
    A narrow CI tells you the metric is meaningful.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    if n < 20:
        # Bootstrap is unreliable on tiny test sets
        return BootstrapCI(
            auc_mean=float("nan"), auc_low=float("nan"), auc_high=float("nan"),
            f1_mean=float("nan"), f1_low=float("nan"), f1_high=float("nan"),
            n_iterations=0,
        )

    aucs, f1s = [], []
    pred = (y_proba >= threshold).astype(int)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            # Bootstrap sample lost a class – skip; AUC undefined
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
            f1s.append(f1_score(y_true[idx], pred[idx], zero_division=0))
        except ValueError:
            continue

    if not aucs:
        return BootstrapCI(
            auc_mean=float("nan"), auc_low=float("nan"), auc_high=float("nan"),
            f1_mean=float("nan"), f1_low=float("nan"), f1_high=float("nan"),
            n_iterations=0,
        )

    return BootstrapCI(
        auc_mean=float(np.mean(aucs)),
        auc_low=float(np.percentile(aucs, 2.5)),
        auc_high=float(np.percentile(aucs, 97.5)),
        f1_mean=float(np.mean(f1s)),
        f1_low=float(np.percentile(f1s, 2.5)),
        f1_high=float(np.percentile(f1s, 97.5)),
        n_iterations=len(aucs),
    )


# --------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------
@dataclass
class CalibrationData:
    fraction_positive: np.ndarray   # actual rate of positives in each bin
    mean_predicted: np.ndarray      # average predicted prob in each bin
    brier_score: float              # lower is better; perfect = 0


def compute_calibration(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> CalibrationData:
    """Reliability diagram data + Brier score."""
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    # Brier = mean squared error of predicted prob vs binary label
    brier = float(np.mean((y_proba - y_true) ** 2))
    return CalibrationData(
        fraction_positive=frac_pos, mean_predicted=mean_pred, brier_score=brier
    )


# --------------------------------------------------------------------
# Learning curve
# --------------------------------------------------------------------
@dataclass
class LearningCurveData:
    train_sizes: np.ndarray
    train_scores_mean: np.ndarray
    train_scores_std: np.ndarray
    val_scores_mean: np.ndarray
    val_scores_std: np.ndarray
    scoring: str


def compute_learning_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: str = "f1",
    n_points: int = ML_LEARNING_CURVE_POINTS,
    n_jobs: int = -1,
) -> LearningCurveData:
    """Learning curve over increasing training-set size.

    Big gap between train and val → high variance (overfitting).
    Both flat and low → high bias (underfitting).
    Curves still climbing at full size → more data would help.
    """
    train_sizes = np.linspace(0.2, 1.0, n_points)
    sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=42,
        shuffle=True,
    )
    return LearningCurveData(
        train_sizes=sizes_abs,
        train_scores_mean=train_scores.mean(axis=1),
        train_scores_std=train_scores.std(axis=1),
        val_scores_mean=val_scores.mean(axis=1),
        val_scores_std=val_scores.std(axis=1),
        scoring=scoring,
    )


# --------------------------------------------------------------------
# Per-segment performance
# --------------------------------------------------------------------
def segment_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    segments: pd.Series,
    threshold: float,
    min_segment_size: int = 30,
) -> pd.DataFrame:
    """Compute per-segment AUC, precision, recall, F1, n, and base rate.

    A model that's accurate overall but bad for one segment (e.g., 'OPERATIONAL')
    is dangerous to deploy. This surfaces that.

    `segments` must be aligned positionally with `y_true` and `y_proba`
    (i.e. `len(segments) == len(y_true) == len(y_proba)` and same order).
    """
    pred = (y_proba >= threshold).astype(int)
    rows = []
    # reset_index so groupby positions match y_true / y_proba positions
    seg_array = np.asarray(segments)
    unique_segs = pd.unique(seg_array)
    for seg in unique_segs:
        mask = seg_array == seg
        n = int(mask.sum())
        if n < min_segment_size:
            continue
        y_s, p_s, pr_s = y_true[mask], y_proba[mask], pred[mask]
        try:
            auc = roc_auc_score(y_s, p_s) if len(np.unique(y_s)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        rows.append({
            "Segment": str(seg),
            "N": n,
            "Base rate": float(y_s.mean()),
            "AUC": auc,
            "Precision": float(precision_score(y_s, pr_s, zero_division=0)),
            "Recall": float(recall_score(y_s, pr_s, zero_division=0)),
            "F1": float(f1_score(y_s, pr_s, zero_division=0)),
        })
    return pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------
# Score helpers
# --------------------------------------------------------------------
def score_with_metric(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
    """Pick the right metric implementation given a name."""
    if metric == "average_precision":
        return float(average_precision_score(y_true, y_proba))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_proba))
    # Threshold-dependent metrics get the F1-optimal threshold
    sweep = sweep_thresholds(y_true, y_proba)
    pred = (y_proba >= sweep.optimal_f1_threshold).astype(int)
    if metric == "f1":
        return float(f1_score(y_true, pred, zero_division=0))
    if metric == "f2":
        return float(fbeta_score(y_true, pred, beta=2.0, zero_division=0))
    if metric == "precision":
        return float(precision_score(y_true, pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, pred, zero_division=0))
    raise ValueError(f"Unknown metric: {metric}")
