"""Tab 5: Attrition ML (XGBoost + SHAP + What-If Simulator).

Predicts VOLUNTARY exit within the next ML_HORIZON_MONTHS months, trained on
point-in-time snapshots (see ml/data_prep.py). The UI shows:
- Out-of-time test metrics (incl. precision / lift in the top-10% risk group)
- Threshold derived on a validation snapshot (with manual override)
- Bootstrap 95% CI on AUC/F1, reliability diagram + Brier score
- Per-snapshot backtest table
- Per-segment metrics (Job Property: Operational vs Administrative)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, roc_auc_score

from config.constants import (
    ML_HORIZON_MONTHS,
    ML_SNAPSHOT_STEP_MONTHS,
    ML_TARGET_COMPANY,
    ML_TARGET_DEPARTURE_REASON,
    ML_WORK_RELATIONSHIP,
)
from ml import (
    MLPreparedData,
    TrainResult,
    align_columns,
    bootstrap_metrics,
    compute_calibration,
    evaluate,
    fairness_table,
    precision_at_top,
    predict_active_employees,
    prepare_ml_data,
    segment_metrics,
    train_attrition_model,
)
from ui.tab_shap import render_shap_section
from ui.tab_simulator import render_new_hire_simulator


# --------------------------------------------------------------------
# Cache key helpers
# --------------------------------------------------------------------
def _file_id(uploaded_file) -> str:
    return f"{uploaded_file.name}::{uploaded_file.size}"


# Bump this when the feature pipeline changes in a way that makes cached
# models incompatible with newly-encoded data (e.g. ASCII sanitization,
# new engineered features, removed columns). Any change forces all sessions
# to retrain on next page load.
_PIPELINE_VERSION = "v4.0-survival"


def get_prepared(uploaded_file) -> MLPreparedData:
    """prepare_ml_data, cached in session_state per file + pipeline version (shared by tabs)."""
    key = (_file_id(uploaded_file), _PIPELINE_VERSION)
    if st.session_state.get("ml_prepared_key") != key:
        st.session_state["ml_prepared"] = prepare_ml_data(uploaded_file)
        st.session_state["ml_prepared_key"] = key
    return st.session_state["ml_prepared"]


def get_ml_state(uploaded_file) -> tuple[MLPreparedData, TrainResult]:
    """Prepared data + trained model, training once per file (used by the Executive tab)."""
    prepared = get_prepared(uploaded_file)
    _ensure_model_in_state(uploaded_file, prepared, announce=False)
    return prepared, st.session_state["xgb_result"]


def _ensure_model_in_state(uploaded_file, prepared: MLPreparedData, announce: bool = True) -> None:
    """Train if no cached model, file changed, pipeline version changed, or user asked for retrain."""
    fid = _file_id(uploaded_file)
    needs_retrain = (
        st.session_state.get("xgb_result") is None
        or st.session_state.get("xgb_file_id") != fid
        or st.session_state.get("xgb_pipeline_version") != _PIPELINE_VERSION
        or st.session_state.get("xgb_needs_retrain", False)
    )
    if not needs_retrain:
        return

    with st.spinner("🔁 Εκπαίδευση XGBoost (validation, out-of-time test, final model, backtest)..."):
        result = train_attrition_model(
            prepared.X, prepared.y, prepared.snapshots, prepared.panel_meta,
            export_date=prepared.snapshot_date, horizon=prepared.horizon_months,
        )

    st.session_state["xgb_result"] = result
    st.session_state["xgb_file_id"] = fid
    st.session_state["xgb_pipeline_version"] = _PIPELINE_VERSION
    st.session_state["xgb_needs_retrain"] = False

    # Invalidate downstream caches
    for key in (
        "shap_global_bundle",
        "shap_interactions",
        "shap_interactions_X_sample",
    ):
        st.session_state.pop(key, None)

    if announce:
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


def _render_segment_metrics(
    y_test: np.ndarray,
    proba: np.ndarray,
    df_meta_test: pd.DataFrame,
    threshold: float,
) -> None:
    st.markdown("**👥 Per-segment performance** — does the model work equally for all groups?")
    options = [
        c for c in ("Job Property", "Job Family", "Division", "Department")
        if c in df_meta_test.columns and df_meta_test[c].notna().any()
    ]
    if not options:
        st.caption("No segment column available.")
        return
    seg_col = st.selectbox("Segment by", options, key="segment_metrics_col")

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


def _render_fairness(
    y_test: np.ndarray, proba: np.ndarray, df_meta_test: pd.DataFrame, threshold: float,
) -> None:
    st.markdown(
        "**⚖️ Fairness check** — φύλο και ηλικία **δεν** είναι features του μοντέλου. "
        "Αν το *Flag rate* μιας ομάδας απέχει πολύ από το *Actual exit rate*, άλλα features "
        "(π.χ. προϋπηρεσία, μισθός) λειτουργούν ως proxies — χρήση με προσοχή."
    )
    fmt = {"Actual exit rate": "{:.1%}", "Flag rate": "{:.1%}",
           "Mean predicted risk": "{:.1%}", "Recall": "{:.1%}"}
    groups = {}
    if "Gender" in df_meta_test.columns:
        groups["Gender (κωδικός export)"] = df_meta_test["Gender"]
    if "Age" in df_meta_test.columns:
        groups["Age band"] = pd.cut(
            df_meta_test["Age"], [0, 30, 40, 50, 120], right=False,
            labels=["<30", "30-39", "40-49", "50+"],
        ).astype(str).where(df_meta_test["Age"].notna())
    for name, g in groups.items():
        table = fairness_table(y_test, proba, g.reset_index(drop=True), threshold)
        if not table.empty:
            st.caption(name)
            st.dataframe(table.style.format(fmt), width="stretch", hide_index=True)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def _render_model_definition(prepared: MLPreparedData, result: TrainResult) -> None:
    n_snaps = prepared.snapshots.nunique()
    with st.expander("ℹ️ Τι ακριβώς προβλέπει το μοντέλο", expanded=True):
        st.markdown(
            f"""
- **Στόχος:** πιθανότητα **εθελοντικής αποχώρησης** (`{ML_TARGET_DEPARTURE_REASON}`)
  μέσα στους **επόμενους {prepared.horizon_months} μήνες**.
- **Πληθυσμός:** `{ML_TARGET_COMPANY}`, σύμβαση `{ML_WORK_RELATIONSHIP}`.
  Απολύσεις, συνταξιοδοτήσεις κ.λπ. **δεν** μετράνε ως θετικό αποτέλεσμα.
- **Μοντέλο:** XGBoost **survival (AFT)** — μαθαίνει *σε πόσους μήνες* αποχωρεί κάποιος, άρα
  αξιοποιεί και τα πιο πρόσφατα snapshots που δεν έχουν ακόμη πλήρη 12μηνη παρακολούθηση.
  Η πιθανότητα 12μήνου βαθμονομείται (calibration) σε προηγούμενη περίοδο.
- **Εκπαίδευση:** {n_snaps} μηνιαίες "φωτογραφίες" (snapshots) του ενεργού προσωπικού
  ({prepared.snapshots.min():%Y-%m} → {prepared.snapshots.max():%Y-%m}).
  Η προϋπηρεσία υπολογίζεται **κατά την ημερομηνία κάθε snapshot**.
- **Features:** μισθός (σε σημερινές τιμές), μισθός vs grade, προϋπηρεσία, job family / level,
  job property, τμήμα, division, θέση, πόλη, εκπαίδευση.
  **Δεν** χρησιμοποιούνται φύλο, ηλικία, οικογενειακή κατάσταση, φορολογική κατηγορία.
- **Μισθοί** αναγάγονται σε σημερινές τιμές (το export κρατά τον μισθό αποχώρησης),
  με δείκτη από τα δεδομένα: {", ".join(f"{y}: ×{f:.2f}" for y, f in prepared.salary_index.items() if f > 1.001) or "—"}.
- **Test (out-of-time):** εκπαίδευση μόνο με ό,τι ήταν γνωστό στις {result.test_snapshot:%Y-%m-%d},
  πρόβλεψη για όσους ήταν ενεργοί τότε και σύγκριση με το τι έγινε τους επόμενους
  {prepared.horizon_months} μήνες. Calibration & threshold από το snapshot {result.val_snapshot:%Y-%m-%d}.
- **Βαθμολόγηση:** οι σημερινοί εργαζόμενοι με στοιχεία στις {prepared.snapshot_date:%Y-%m-%d}.
"""
        )
        st.caption(
            "Περιορισμός: το export περιέχει μόνο την τελευταία τιμή ανά εργαζόμενο για Grade, "
            "τμήμα, θέση και μισθό. Γι' αυτό το Grade δεν χρησιμοποιείται ως feature "
            "και τα παλαιότερα snapshots έχουν μικρότερο βάρος."
        )


def render_ml_tab(uploaded_file) -> None:
    st.markdown("### 🤖 Voluntary Attrition Prediction (XGBoost + SHAP)")
    st.caption(
        f"Survival model · monthly snapshots · horizon {ML_HORIZON_MONTHS} μήνες · "
        "out-of-time test · calibration & threshold από validation snapshot"
    )

    # 1. Prepare data
    try:
        prepared: MLPreparedData = get_prepared(uploaded_file)
    except ValueError as e:
        st.error(str(e))
        return

    X, y = prepared.X, prepared.y
    st.write(
        f"📏 Training panel: **{X.shape[0]:,}** employee-snapshots "
        f"(**{int(y.notna().sum()):,}** with a complete 12-month outcome) "
        f"({prepared.snapshots.nunique()} snapshots), **{X.shape[1]}** features. "
        f"12-month voluntary exit rate: **{y.mean():.1%}**. "
        f"Current employees scored: **{len(prepared.X_current):,}** "
        f"(as of {prepared.snapshot_date:%Y-%m-%d})."
    )

    if X.empty or y.nunique() < 2:
        st.error("Δεν υπάρχουν επαρκή δεδομένα ή ο στόχος είναι μονοδιάστατος.")
        return

    # 2. Train (or reuse cached)
    st.markdown("### ⚙️ Training")
    if st.button("🔁 Train / update attrition model"):
        st.session_state["xgb_needs_retrain"] = True
    try:
        _ensure_model_in_state(uploaded_file, prepared)
    except ValueError as e:
        st.error(str(e))
        return

    result: TrainResult = st.session_state["xgb_result"]
    model = result.model
    train_columns = result.train_columns

    _render_model_definition(prepared, result)

    with st.expander("XGBoost parameters", expanded=False):
        st.json(result.best_params)

    # 3. Out-of-time test performance
    y_test = result.y_test
    proba_test = result.test_proba
    st.markdown(f"### 📊 Out-of-time Test ({result.test_snapshot:%Y-%m-%d} → +{ML_HORIZON_MONTHS}m)")
    st.caption(
        f"Train: {len(result.X_train):,} snapshot rows before {result.test_snapshot:%Y-%m-%d} · "
        f"Test: {len(y_test):,} employees, {int(y_test.sum())} left voluntarily ({y_test.mean():.1%})"
    )

    p_top = precision_at_top(y_test.to_numpy(), proba_test)
    base = float(y_test.mean())
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("ROC AUC", f"{roc_auc_score(y_test, proba_test):.3f}")
    r2.metric("PR AUC", f"{average_precision_score(y_test, proba_test):.3f}",
              help=f"Τυχαίο μοντέλο = {base:.3f} (base rate)")
    r3.metric("Precision @ top 10%", f"{p_top:.1%}",
              help="Από το 10% με το υψηλότερο ρίσκο, πόσοι αποχώρησαν πράγματι.")
    r4.metric("Lift @ top 10%", f"{p_top / base:.1f}×" if base else "N/A",
              help="Πόσες φορές καλύτερο από τυχαία επιλογή.")

    st.info(
        f"💡 Threshold (F1-optimal στο validation snapshot {result.val_snapshot:%Y-%m-%d}): "
        f"**{result.threshold:.3f}**. Μετακίνησέ το μόνο για επιχειρησιακό λόγο "
        "(π.χ. συγκεκριμένος αριθμός stay interviews)."
    )
    threshold = st.slider(
        "Decision threshold for Attrition = 1",
        min_value=0.01, max_value=0.99,
        value=float(np.clip(result.threshold, 0.01, 0.99)),
        step=0.01,
        help="Lower threshold → more predicted leavers (↑recall, ↓precision).",
    )

    metrics = evaluate(y_test, proba_test, threshold)
    m1, m2, m3 = st.columns(3)
    m1.metric("F1", f"{metrics.f1:.3f}")
    m2.metric("Precision", f"{metrics.precision:.3f}")
    m3.metric("Recall", f"{metrics.recall:.3f}")

    st.text("Classification Report (Test Snapshot):")
    st.text(metrics.classification_report)

    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(
        metrics.confusion_matrix, display_labels=["Stayed", "Left voluntarily"]
    ).plot(cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # 4. Robustness panel
    st.markdown("### 🔬 Model robustness")
    _render_robustness_panel(y_test.values, proba_test, threshold)

    with st.expander("📅 Backtest: απόδοση ανά snapshot", expanded=False):
        st.caption(
            "Για κάθε snapshot: εκπαίδευση μόνο με ό,τι ήταν γνωστό εκείνη την ημέρα, πρόβλεψη των "
            "επόμενων 12 μηνών. Τα παλαιότερα snapshots μπορεί να δείχνουν υψηλότερα νούμερα, "
            "επειδή μισθός/θέση είναι οι τελευταίες γνωστές τιμές και όχι οι τότε."
        )
        st.dataframe(
            result.backtest.style.format({
                "Base rate": "{:.1%}", "ROC AUC": "{:.3f}", "PR AUC": "{:.3f}",
                "Precision @ top 10%": "{:.1%}",
            }),
            width="stretch", hide_index=True,
        )

    # 5. Segment metrics
    with st.expander("👥 Per-segment metrics", expanded=False):
        df_meta_test = prepared.panel_meta.loc[result.X_test.index]
        _render_segment_metrics(y_test.values, proba_test, df_meta_test, threshold)

    with st.expander("⚖️ Fairness check (φύλο / ηλικία)", expanded=False):
        _render_fairness(y_test.values, proba_test, prepared.panel_meta.loc[result.X_test.index], threshold)

    # 6. Score current employees
    active = predict_active_employees(model, prepared.X_current, train_columns, threshold)
    likely = active[active["Predicted_Attrition"] == 1]
    X_active = align_columns(prepared.X_current, train_columns)

    st.markdown("### 👥 Current employees")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted leavers (threshold)", f"{len(likely):,}")
    c2.metric("Expected voluntary exits (12m)", f"{active['Attrition_Probability'].sum():,.0f}",
              help="Άθροισμα πιθανοτήτων — αξιόπιστο όσο καλή είναι η calibration (βλ. reliability diagram).")
    c3.metric("Median risk", f"{active['Attrition_Probability'].median():.1%}")

    # 7. SHAP
    render_shap_section(
        model=model,
        X_train=X,
        X_active=X_active,
        likely_to_attrite=likely,
        registry_numbers=prepared.registry_numbers,
        meta_cols=prepared.meta_cols,
        display_names=prepared.encoder.display_names,
    )

    # 8. What-If simulator
    st.markdown("---")
    render_new_hire_simulator(prepared.df_ui, model, train_columns)
