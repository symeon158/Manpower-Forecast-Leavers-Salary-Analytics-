"""Tab 5: Attrition ML (XGBoost + SHAP + What-If Simulator).

This UI exposes the new robustness features:
- Auto-derived F1-optimal threshold (with manual override)
- Bootstrap 95% CI on AUC/F1
- Reliability diagram + Brier score (calibration quality)
- Learning curve (over/under-fit detection)
- Per-segment metrics (Job Property: Operational vs Administrative)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from config.constants import (
    ML_RANDOMSEARCH_ITERATIONS,
    ML_SCORING,
    ML_SPLIT_STRATEGY,
    RISK_LOW_MAX,
    RISK_MEDIUM_MAX,
)
from ml import (
    MLPreparedData,
    align_columns,
    bootstrap_metrics,
    compute_calibration,
    compute_learning_curve,
    evaluate,
    predict_active_employees,
    prepare_ml_data,
    segment_metrics,
    split_data,
    sweep_thresholds,
    train_xgb_with_search,
)
from ui.tab_shap import render_shap_section
from ui.tab_simulator import render_new_hire_simulator


# --------------------------------------------------------------------
# Cache key helpers
# --------------------------------------------------------------------
def _file_id(uploaded_file) -> str:
    return f"{uploaded_file.name}::{uploaded_file.size}"


def _ensure_model_in_state(
    uploaded_file,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int,
    hire_dates_train: pd.Series | None,
) -> None:
    """Train if no cached model, file changed, or user asked for retrain."""
    fid = _file_id(uploaded_file)
    needs_retrain = (
        st.session_state.get("xgb_model") is None
        or st.session_state.get("xgb_file_id") != fid
        or st.session_state.get("xgb_needs_retrain", False)
    )
    if not needs_retrain:
        return

    with st.spinner(
        f"🔁 Εκπαίδευση XGBoost ({n_iter} iter, scoring={ML_SCORING}, "
        f"split={ML_SPLIT_STRATEGY}, calibration on)..."
    ):
        result = train_xgb_with_search(
            X_train, y_train,
            n_iter=n_iter,
            hire_dates_train=hire_dates_train,
        )

    st.session_state["xgb_model"] = result.model
    st.session_state["xgb_base_model"] = result.base_model
    st.session_state["xgb_best_params"] = result.best_params
    st.session_state["xgb_train_columns"] = result.train_columns
    st.session_state["xgb_is_calibrated"] = result.is_calibrated
    st.session_state["xgb_scoring"] = result.scoring
    st.session_state["xgb_split_strategy"] = result.split_strategy
    st.session_state["xgb_file_id"] = fid
    st.session_state["xgb_needs_retrain"] = False

    # Invalidate downstream caches
    for key in (
        "shap_global_bundle",
        "shap_interactions",
        "shap_interactions_X_sample",
        "lc_data",
        "auto_threshold",
    ):
        st.session_state.pop(key, None)

    st.success("✅ Νέο μοντέλο XGBoost εκπαιδεύτηκε.")


# --------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------
def _render_robustness_panel(
    y_test: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> None:
    """Bootstrap CI + calibration plot + Brier score."""
    col1, col2 = st.columns([1, 1])

    # Bootstrap CI
    with col1:
        st.markdown("**🎯 Bootstrap 95% CI (test set, 200 resamples)**")
        ci = bootstrap_metrics(y_test, proba, threshold)
        if ci.n_iterations > 0:
            st.write(f"AUC: **{ci.auc_mean:.3f}** [{ci.auc_low:.3f} – {ci.auc_high:.3f}]")
            st.write(f"F1:  **{ci.f1_mean:.3f}** [{ci.f1_low:.3f} – {ci.f1_high:.3f}]")
            spread = ci.auc_high - ci.auc_low
            if spread > 0.15:
                st.warning(
                    f"⚠️ Wide CI on AUC (spread = {spread:.2f}). The test set may be "
                    "too small or class-imbalanced for stable estimates."
                )
        else:
            st.info("Test set too small for bootstrap.")

    # Calibration
    with col2:
        st.markdown("**📐 Calibration (reliability diagram)**")
        cal = compute_calibration(y_test, proba, n_bins=10)
        st.write(f"Brier score: **{cal.brier_score:.4f}** (lower is better; perfect = 0)")

        fig_cal, ax_cal = plt.subplots(figsize=(4.2, 3.5))
        ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax_cal.plot(cal.mean_predicted, cal.fraction_positive, "o-", label="Model")
        ax_cal.set_xlabel("Mean predicted probability")
        ax_cal.set_ylabel("Fraction of positives")
        ax_cal.set_title("Reliability diagram")
        ax_cal.legend(loc="best", fontsize=8)
        ax_cal.grid(alpha=0.3)
        st.pyplot(fig_cal)
        plt.close(fig_cal)


def _render_learning_curve(model, X: pd.DataFrame, y: pd.Series, scoring: str) -> None:
    st.markdown("**📈 Learning Curve** — αν train ≫ val: overfit, αν και τα δύο πέφτουν: underfit.")
    if "lc_data" not in st.session_state:
        with st.spinner("Computing learning curve..."):
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            try:
                lc = compute_learning_curve(model, X, y, cv=cv, scoring=scoring, n_points=4)
                st.session_state["lc_data"] = lc
            except Exception as e:
                st.warning(f"Learning curve unavailable: {e}")
                return
    lc = st.session_state["lc_data"]

    fig_lc, ax_lc = plt.subplots(figsize=(6, 3.5))
    ax_lc.plot(lc.train_sizes, lc.train_scores_mean, "o-", label="Train", color="C0")
    ax_lc.fill_between(
        lc.train_sizes,
        lc.train_scores_mean - lc.train_scores_std,
        lc.train_scores_mean + lc.train_scores_std,
        alpha=0.2, color="C0",
    )
    ax_lc.plot(lc.train_sizes, lc.val_scores_mean, "s-", label="Cross-val", color="C1")
    ax_lc.fill_between(
        lc.train_sizes,
        lc.val_scores_mean - lc.val_scores_std,
        lc.val_scores_mean + lc.val_scores_std,
        alpha=0.2, color="C1",
    )
    ax_lc.set_xlabel("Training samples")
    ax_lc.set_ylabel(f"{lc.scoring}")
    ax_lc.set_title("Learning curve")
    ax_lc.legend()
    ax_lc.grid(alpha=0.3)
    st.pyplot(fig_lc)
    plt.close(fig_lc)


def _render_segment_metrics(
    y_test: np.ndarray,
    proba: np.ndarray,
    df_meta_test: pd.DataFrame,
    threshold: float,
) -> None:
    st.markdown("**👥 Per-segment performance** — does the model work equally for all groups?")
    # Use Job Property if available; fall back to Division
    seg_col = None
    for candidate in ("Job Property", "Division", "Department"):
        if candidate in df_meta_test.columns and df_meta_test[candidate].notna().any():
            seg_col = candidate
            break
    if seg_col is None:
        st.caption("No segment column available.")
        return

    seg = df_meta_test[seg_col].astype(str).fillna("Unknown")
    # Reset to positional Series — segment_metrics requires positional alignment
    seg_positional = seg.reset_index(drop=True)
    df_seg = segment_metrics(y_test, proba, seg_positional, threshold, min_segment_size=20)
    if df_seg.empty:
        st.caption(f"No segment in `{seg_col}` is large enough (≥20 rows) for reliable metrics.")
        return
    st.caption(f"Segmented by `{seg_col}` — segments with <20 rows hidden.")
    st.dataframe(df_seg.style.format({
        "Base rate": "{:.2%}", "AUC": "{:.3f}",
        "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}",
    }), width="stretch")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def render_ml_tab(uploaded_file) -> None:
    st.markdown("### 🤖 Attrition Prediction (Robust XGBoost + SHAP)")
    st.caption(
        f"Scoring: `{ML_SCORING}` · Split: `{ML_SPLIT_STRATEGY}` · "
        f"Probability calibration: on · Threshold auto-tuned for F1"
    )

    # 1. Prepare data
    try:
        prepared: MLPreparedData = prepare_ml_data(uploaded_file)
    except ValueError as e:
        st.error(str(e))
        return

    X, y = prepared.X, prepared.y
    st.write(
        f"📏 ML observations: **{X.shape[0]}** rows, **{X.shape[1]}** features. "
        f"Attrition rate: **{y.mean():.2%}**. "
        f"Snapshot date: **{prepared.snapshot_date.strftime('%Y-%m-%d')}**"
    )

    if X.empty or y.nunique() < 2:
        st.error("Δεν υπάρχουν επαρκή δεδομένα ή ο στόχος είναι μονοδιάστατος.")
        return

    # 2. Split (temporal by default)
    X_train, X_test, y_train, y_test = split_data(
        X, y, hire_dates=prepared.hire_date_series, strategy=ML_SPLIT_STRATEGY,
    )
    hire_dates_train = prepared.hire_date_series.loc[X_train.index]

    st.caption(
        f"Train: {len(X_train):,} rows ({y_train.mean():.1%} positive) · "
        f"Test: {len(X_test):,} rows ({y_test.mean():.1%} positive)"
    )

    # 3. Train (or reuse cached)
    st.markdown("### ⚙️ Training")
    if st.button("🔁 Train / update attrition model"):
        st.session_state["xgb_needs_retrain"] = True

    _ensure_model_in_state(
        uploaded_file, X_train, y_train,
        n_iter=ML_RANDOMSEARCH_ITERATIONS,
        hire_dates_train=hire_dates_train,
    )

    model = st.session_state["xgb_model"]
    train_columns = st.session_state["xgb_train_columns"]

    with st.expander("Best XGBoost parameters (last training)", expanded=False):
        st.json(st.session_state.get("xgb_best_params", {}))

    # 4. Auto-derive optimal threshold from test set, allow override
    X_test_aligned = align_columns(X_test, train_columns)
    proba_test = model.predict_proba(X_test_aligned)[:, 1]

    sweep = sweep_thresholds(y_test.values, proba_test)
    auto_thr = float(sweep.optimal_f1_threshold)
    st.session_state.setdefault("auto_threshold", auto_thr)

    st.markdown("### 📊 Test Set Performance")
    st.info(
        f"💡 F1-optimal threshold (auto-derived from test set): **{auto_thr:.3f}**. "
        "Move the slider only if you have a specific business reason "
        "(e.g. you can only afford a fixed number of follow-up interviews)."
    )

    threshold = st.slider(
        "Decision threshold for Attrition = 1",
        min_value=0.05, max_value=0.95,
        value=float(st.session_state["auto_threshold"]),
        step=0.01,
        help="Lower threshold → more predicted leavers (↑recall, ↓precision).",
    )

    metrics = evaluate(model, X_test_aligned, y_test, threshold)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC AUC", f"{metrics.auc:.3f}")
    m2.metric("F1", f"{metrics.f1:.3f}")
    m3.metric("Precision", f"{metrics.precision:.3f}")
    m4.metric("Recall", f"{metrics.recall:.3f}")

    st.text("Classification Report (Test Set):")
    st.text(metrics.classification_report)

    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(
        metrics.confusion_matrix, display_labels=["No Attrition", "Attrition"]
    ).plot(cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # 5. Robustness panel
    st.markdown("### 🔬 Model robustness")
    _render_robustness_panel(y_test.values, proba_test, threshold)

    # 6. Learning curve (cached on first compute)
    with st.expander("📈 Learning curve (slow on first run)", expanded=False):
        _render_learning_curve(
            st.session_state["xgb_base_model"], X_train, y_train, scoring=ML_SCORING,
        )

    # 7. Segment metrics
    with st.expander("👥 Per-segment metrics", expanded=False):
        # Recover segment columns from df_ui aligned to X_test
        df_ui = prepared.df_ui
        df_meta_test = df_ui.reindex(X_test.index)
        _render_segment_metrics(y_test.values, proba_test, df_meta_test, threshold)

    # 8. Predict on active employees
    active = predict_active_employees(
        model, prepared.df_transformed, train_columns, threshold
    )
    likely = active[active["Predicted_Attrition"] == 1]
    X_active = align_columns(
        active.drop(columns=["Attrition", "Registry Number",
                             "Attrition_Probability", "Predicted_Attrition"],
                    errors="ignore"),
        train_columns,
    )

    # 9. SHAP
    render_shap_section(
        model=st.session_state["xgb_base_model"],   # SHAP needs the raw tree model
        X_train=X_train,
        X_active=X_active,
        likely_to_attrite=likely,
        registry_numbers=prepared.registry_numbers,
        meta_cols=prepared.meta_cols,
    )

    # 10. What-If simulator
    st.markdown("---")
    render_new_hire_simulator(
        prepared.df_ui, st.session_state["xgb_base_model"], train_columns,
    )
