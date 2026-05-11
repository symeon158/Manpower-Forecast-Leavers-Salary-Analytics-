"""SHAP rendering: global summary, dependence, per-employee drivers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from config.constants import (
    SHAP_SAMPLE_SIZE_DEPENDENCE,
    SHAP_SAMPLE_SIZE_GLOBAL,
    TOP_N_DRIVERS_DEFAULT,
)
from ml import compute_global_shap, shap_for_single_row, top_drivers_per_employee
from ui.components import shap_plot_safely


def render_shap_section(
    model,
    X_train: pd.DataFrame,
    X_active: pd.DataFrame,
    likely_to_attrite: pd.DataFrame,
    registry_numbers: pd.Series,
    meta_cols: pd.DataFrame,
) -> None:
    st.markdown("## 🧠 SHAP – Model Explainability")

    if model is None or X_train is None or X_train.empty:
        st.info("Το μοντέλο ML δεν είναι διαθέσιμο ή δεν υπάρχουν δεδομένα για SHAP.")
        return

    _render_predicted_leavers_table(
        model, X_active, likely_to_attrite, registry_numbers, meta_cols
    )

    bundle = _get_or_compute_global_shap(model, X_train)

    _render_global_summaries(bundle)
    _render_dependence_plots(bundle)
    _render_employee_drilldown(model, X_active, likely_to_attrite, registry_numbers, meta_cols)


# --------------------------------------------------------------------
# Sub-sections
# --------------------------------------------------------------------
def _get_or_compute_global_shap(model, X_train):
    """Cache SHAP bundle in session_state, invalidate on model change."""
    cur_id = id(model)
    bundle = st.session_state.get("shap_global_bundle")
    if bundle is None or st.session_state.get("shap_global_model_id") != cur_id:
        with st.spinner("Υπολογισμός SHAP values (sample)..."):
            bundle = compute_global_shap(model, X_train, sample_size=SHAP_SAMPLE_SIZE_GLOBAL)
        st.session_state["shap_global_bundle"] = bundle
        st.session_state["shap_global_model_id"] = cur_id
    return bundle


def _render_predicted_leavers_table(
    model, X_active, likely_to_attrite, registry_numbers, meta_cols
) -> None:
    st.markdown("### 📌 Predicted Attrition & Top SHAP Drivers (Active Employees)")
    n = 0 if likely_to_attrite is None else likely_to_attrite.shape[0]
    st.write(f"Employees predicted to attrite (next period): **{n}**")
    if n == 0:
        st.info("Δεν υπάρχουν ενεργοί εργαζόμενοι που να προβλέπονται ως leavers με το τρέχον threshold.")
        return

    with st.spinner("Υπολογισμός τοπικών SHAP drivers για predicted leavers..."):
        idx = likely_to_attrite.index
        X_likely = X_active.loc[idx]
        drivers = top_drivers_per_employee(model, X_likely, top_n=3)

    out = pd.DataFrame(
        {
            "Registry Number": registry_numbers.loc[idx],
            "Όνομα": meta_cols["First Name"].loc[idx],
            "Επώνυμο": meta_cols["Last Name"].loc[idx],
            "Division": meta_cols["Division"].loc[idx],
            "Department": meta_cols["Department"].loc[idx],
            "Job Title": meta_cols["Job Position"].loc[idx],
            "Attrition Probability": likely_to_attrite["Attrition_Probability"],
            "Top 3 SHAP Drivers": drivers,
        },
        index=idx,
    ).sort_values("Attrition Probability", ascending=False)

    st.dataframe(out.head(50), width="stretch")
    st.download_button(
        "⬇️ Download Predicted Attrition Employees (CSV)",
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name="predicted_attrition_employees_with_drivers.csv",
        mime="text/csv",
    )


def _render_global_summaries(bundle) -> None:
    with st.expander("📊 Global SHAP Feature Importance (Summary)", expanded=True):
        st.markdown("#### 🔵 SHAP Summary Plot (Dot)")
        shap_plot_safely(
            lambda: shap.summary_plot(
                bundle.shap_values, bundle.X_sample,
                feature_names=bundle.X_sample.columns,
                plot_type="dot", show=False, color_bar=False,
            )
        )
        st.markdown("#### 🟦 SHAP Summary Plot (Bar)")
        shap_plot_safely(
            lambda: shap.summary_plot(
                bundle.shap_values, bundle.X_sample,
                feature_names=bundle.X_sample.columns,
                plot_type="bar", show=False, color_bar=False,
            )
        )


def _render_dependence_plots(bundle) -> None:
    with st.expander("🔎 SHAP Dependence Plots (Top Drivers)", expanded=False):
        mean_abs = np.abs(bundle.shap_values).mean(axis=0)
        top_idx = int(np.argmax(mean_abs))
        top_feature = bundle.X_sample.columns[top_idx]
        second_idx = (
            int(np.argsort(-mean_abs)[1]) if bundle.X_sample.shape[1] > 1 else top_idx
        )
        second_feature = bundle.X_sample.columns[second_idx]

        st.markdown(
            f"#### 1️⃣ Dependence plot για κύριο driver: **{top_feature}** "
            f"(color = {second_feature})"
        )
        shap_plot_safely(
            lambda: shap.dependence_plot(
                top_feature, bundle.shap_values, bundle.X_sample,
                interaction_index=second_feature, show=False,
            )
        )

        feature_choice = st.selectbox(
            "Επίλεξε άλλο feature για dependence plot:",
            options=list(bundle.X_sample.columns),
            index=top_idx,
        )
        st.markdown(f"#### 2️⃣ Dependence plot για **{feature_choice}**")
        shap_plot_safely(
            lambda: shap.dependence_plot(
                feature_choice, bundle.shap_values, bundle.X_sample, show=False,
            )
        )


def _render_employee_drilldown(
    model, X_active, likely_to_attrite, registry_numbers, meta_cols
) -> None:
    with st.expander("🌊 SHAP – Top drivers για συγκεκριμένο εργαζόμενο", expanded=False):
        if likely_to_attrite is None or likely_to_attrite.empty:
            st.info("Δεν υπάρχουν predicted leavers για ανάλυση.")
            return

        reg_series = registry_numbers.loc[likely_to_attrite.index]
        selected_reg = st.selectbox(
            "Επίλεξε Registry Number:",
            options=reg_series.astype(str).tolist(),
        )
        idx_selected = reg_series[reg_series.astype(str) == selected_reg].index[0]

        full_name = (
            f"{meta_cols.loc[idx_selected, 'First Name']} "
            f"{meta_cols.loc[idx_selected, 'Last Name']}"
        )

        row_X = X_active.loc[[idx_selected]]
        shap_row, _ = shap_for_single_row(model, row_X)

        df_local = pd.DataFrame({
            "Feature": row_X.columns.tolist(),
            "SHAP": shap_row,
            "AbsSHAP": np.abs(shap_row),
        }).sort_values("AbsSHAP", ascending=False)

        top_n = st.slider("Πλήθος κορυφαίων drivers", 3, 20, TOP_N_DRIVERS_DEFAULT)
        df_top = df_local.head(top_n)

        st.markdown(
            f"#### 🌊 Top {top_n} SHAP drivers για **{full_name}** (Registry: {selected_reg})"
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(df_top["Feature"], df_top["SHAP"])
        ax.invert_yaxis()
        ax.set_xlabel("SHAP value (συμβολή στην πιθανότητα αποχώρησης)")
        ax.set_ylabel("Feature")
        st.pyplot(fig)
        plt.close(fig)
