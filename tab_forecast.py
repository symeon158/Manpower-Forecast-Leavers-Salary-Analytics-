"""
Tabs 3 & 4: Churn / Hire profiles.
Both share the same shape - aggregate by JobTitle, compute ratios, format, render.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.constants import TOP_N_PROFILE_DEFAULT


def _format_currency(x) -> str:
    return f"{x:,.0f}" if pd.notna(x) else "N/A"


def _format_pct(x) -> str:
    return f"{x:,.1f}%" if pd.notna(x) else "N/A"


def _format_decimal(x) -> str:
    return f"{x:,.2f}" if pd.notna(x) else "N/A"


def _aggregate_profile(
    df: pd.DataFrame,
    total_months: int,
    count_label: str,                # "Departures" or "Hires"
    common_cat_col: str,             # "Departure Type" or "Job Property"
    common_cat_alias: str,           # "Common Departure Type" / "Common Job Property"
) -> pd.DataFrame:
    """Aggregate per-job profile metrics. Returns table with both formatted + numeric helper cols.

    NOTE: missing values stay as `pd.NA` here (not the string 'N/A'), so the numeric
    columns keep a stable dtype. String formatting happens later in `_format_profile`.
    """
    grouped = df.groupby("JobTitle").agg(
        Total=("JobTitle", "size"),
        Avg_Per_Month=("JobTitle", lambda x: x.size / total_months),
        Avg_Salary=("NominalSalary", "mean"),
        Most_Common_Grade=(
            "GRADE_clean",
            lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA,
        ),
        Most_Common_Cat=(
            common_cat_col,
            lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA,
        ),
    ).reset_index()

    total_grand = grouped["Total"].sum()
    grouped["Ratio (%)"] = (grouped["Total"] / total_grand) * 100 if total_grand else 0
    grouped["Cost Index"] = grouped["Total"] * grouped["Avg_Salary"].fillna(0)
    grouped["CostIndex_num"] = grouped["Cost Index"]

    # Rename for display
    return grouped.rename(
        columns={
            "Total": f"Total {count_label} (History)",
            "Avg_Per_Month": f"Avg {count_label} / Month",
            "Avg_Salary": f"Avg Salary ({count_label})",
            "Most_Common_Grade": "Common Grade",
            "Most_Common_Cat": common_cat_alias,
        }
    )


def _format_profile(df: pd.DataFrame, count_label: str, common_cat_alias: str) -> pd.DataFrame:
    """Convert all display columns to strings.

    Streamlit's Arrow serializer requires a consistent dtype per column. Once we map
    every column to `str` (with 'N/A' for missing), there's no mixed-type ambiguity.
    """
    df[f"Avg Salary ({count_label})"] = df[f"Avg Salary ({count_label})"].map(_format_currency)
    df[f"Avg {count_label} / Month"] = df[f"Avg {count_label} / Month"].map(_format_decimal)
    df["Ratio (%)"] = df["Ratio (%)"].map(_format_pct)
    df["Cost Index"] = df["Cost Index"].map(_format_currency)
    # Force the categorical/text columns to plain str so Arrow gets one dtype
    df["Common Grade"] = df["Common Grade"].astype(object).where(
        df["Common Grade"].notna(), "N/A"
    ).astype(str)
    df[common_cat_alias] = df[common_cat_alias].astype(object).where(
        df[common_cat_alias].notna(), "N/A"
    ).astype(str)
    df["JobTitle"] = df["JobTitle"].astype(str)
    df[f"Total {count_label} (History)"] = df[f"Total {count_label} (History)"].astype(str)
    return df


def _render_profile(
    df: pd.DataFrame,
    total_months: int,
    count_label: str,                # "Leavers" / "Hires"
    common_cat_col: str,
    common_cat_alias: str,
    title: str,
    subtitle: str,
    css_id_filename: str,
    cost_index_top_label: str,
) -> None:
    st.markdown(f"### {title}")
    st.caption(subtitle)

    if df.empty or "JobTitle" not in df.columns:
        st.info(f"Δεν υπάρχουν δεδομένα για να δημιουργηθεί προφίλ.")
        return

    df_clean = df.dropna(subset=["JobTitle"])
    if df_clean.empty:
        st.info("Δεν υπάρχουν επαρκή δεδομένα με JobTitle.")
        return

    profile = _aggregate_profile(
        df_clean, total_months, count_label, common_cat_col, common_cat_alias
    )
    profile = profile.sort_values(f"Total {count_label} (History)", ascending=False)
    profile_display = _format_profile(profile.copy(), count_label, common_cat_alias)

    top_n = st.slider(
        f"Display Top N {count_label} Job Titles:",
        min_value=5, max_value=50, value=TOP_N_PROFILE_DEFAULT,
        key=f"top_n_{css_id_filename}",
    )

    display_cols = [
        "JobTitle",
        f"Total {count_label} (History)",
        f"Avg {count_label} / Month",
        "Ratio (%)",
        f"Avg Salary ({count_label})",
        "Common Grade",
        common_cat_alias,
        "Cost Index",
    ]
    st.dataframe(profile_display[display_cols].head(top_n), width="stretch")

    # Top-10 by cost index
    top10 = profile_display.assign(_cost=profile["CostIndex_num"]).sort_values(
        "_cost", ascending=False
    ).head(10)
    if not top10.empty:
        st.markdown(f"#### 🏅 {cost_index_top_label}")
        bullets = [
            f"- **{row['JobTitle']}**: {row[f'Total {count_label} (History)']} {count_label.lower()}, "
            f"Avg Salary ~ {row[f'Avg Salary ({count_label})']}, "
            f"Ratio {row['Ratio (%)']}, Cost Index {row['Cost Index']}"
            for _, row in top10.iterrows()
        ]
        st.markdown("\n".join(bullets))

    csv = profile_display[display_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        f"⬇️ Download Full {count_label} Profile (CSV)",
        data=csv,
        file_name=f"{css_id_filename}.csv",
        mime="text/csv",
    )


def render_churn_tab(df_leavers_period: pd.DataFrame, total_months: int) -> None:
    _render_profile(
        df=df_leavers_period,
        total_months=total_months,
        count_label="Leavers",
        common_cat_col="Departure Type",
        common_cat_alias="Common Departure Type",
        title="🎯 High-Risk Profile Summary (Leavers)",
        subtitle=(
            "Σύνοψη χαρακτηριστικών των εργαζομένων που αποχώρησαν, "
            "με μέσο μηνιαίο κίνδυνο και churn ratio."
        ),
        css_id_filename="churn_profile_jobtitles",
        cost_index_top_label="Top 10 ρόλοι με τον υψηλότερο 'Churn Cost Index'",
    )


def render_hires_tab(
    df_filtered: pd.DataFrame, date_range: tuple, total_months: int
) -> None:
    if "HireDate" not in df_filtered.columns:
        st.info("Δεν υπάρχει στήλη HireDate στο dataset.")
        return

    df_hires = df_filtered[
        (df_filtered["HireDate"] >= date_range[0])
        & (df_filtered["HireDate"] <= date_range[1])
    ].copy()

    _render_profile(
        df=df_hires,
        total_months=total_months,
        count_label="Hires",
        common_cat_col="Job Property",
        common_cat_alias="Common Job Property",
        title="🧲 High-Opportunity Profile Summary (Hires)",
        subtitle=(
            "Σύνοψη χαρακτηριστικών των προσλήψεων στην επιλεγμένη περίοδο, "
            "με μέσο μηνιαίο ρυθμό και hire ratio."
        ),
        css_id_filename="hire_profile_jobtitles",
        cost_index_top_label="Top 10 ρόλοι με τον υψηλότερο 'Hiring Cost Index'",
    )
