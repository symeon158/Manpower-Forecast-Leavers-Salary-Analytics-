"""Tab 2: Salary & Grade."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from forecasting import classify_voluntary


def render_salary_tab(
    df_filtered: pd.DataFrame,
    df_leavers_period: pd.DataFrame,
    metric: str,
    date_range: tuple,
) -> None:
    # Population depends on metric
    if metric == "Hires" and "HireDate" in df_filtered.columns:
        df_pop = df_filtered[
            (df_filtered["HireDate"] >= date_range[0])
            & (df_filtered["HireDate"] <= date_range[1])
        ].copy()
        pop_label = "Hires (επιλεγμένη περίοδος)"
    else:
        df_pop = df_leavers_period.copy()
        pop_label = "Leavers (επιλεγμένη περίοδος)"

    st.markdown(f"### 💶 Salary & Grade Analysis για {pop_label}")

    has_salary = (
        not df_pop.empty
        and "NominalSalary" in df_pop.columns
        and df_pop["NominalSalary"].notna().any()
    )

    if has_salary:
        col1, col2 = st.columns(2)
        with col1:
            fig_box = px.box(
                df_pop, x="JobTitle", y="NominalSalary",
                title=f"Distribution of Nominal Salary by Job Title ({pop_label})",
            )
            fig_box.update_layout(xaxis_title="Job Title", yaxis_title="Nominal Salary")
            st.plotly_chart(fig_box, width="stretch")

        with col2:
            if "GRADE_clean" in df_pop.columns and df_pop["GRADE_clean"].notna().any():
                grade_salary = (
                    df_pop.dropna(subset=["GRADE_clean"])
                    .groupby("GRADE_clean")["NominalSalary"]
                    .mean().reset_index().sort_values("GRADE_clean")
                )
                fig_grade = px.bar(
                    grade_salary, x="GRADE_clean", y="NominalSalary",
                    title=f"Average Nominal Salary by Grade ({pop_label})",
                )
                fig_grade.update_layout(xaxis_title="Grade", yaxis_title="Average Nominal Salary")
                st.plotly_chart(fig_grade, width="stretch")
            else:
                st.info("Δεν υπάρχουν έγκυρες τιμές Grade για να γίνει ανάλυση.")
    else:
        st.info(f"Δεν βρέθηκαν έγκυρες τιμές στο NominalSalary για {pop_label}.")

    st.markdown("### 🧩 Voluntary vs Non-voluntary departures (per month)")
    if not df_leavers_period.empty:
        df = df_leavers_period.copy()
        df["VolCategory"] = df["Departure Type"].apply(classify_voluntary)
        vol_ts = (
            df.assign(Month=lambda x: x["DepartureDate"].dt.to_period("M").dt.to_timestamp())
            .groupby(["Month", "VolCategory"])
            .size()
            .reset_index(name="Departures")
        )
        fig_vol = px.bar(
            vol_ts, x="Month", y="Departures", color="VolCategory",
            barmode="stack",
            title="Voluntary vs Non-voluntary departures per month",
        )
        fig_vol.update_layout(xaxis_title="Month", yaxis_title="Departures")
        st.plotly_chart(fig_vol, width="stretch")
    else:
        st.info("Δεν υπάρχουν αποχωρήσεις για ανάλυση voluntary/non-voluntary.")
