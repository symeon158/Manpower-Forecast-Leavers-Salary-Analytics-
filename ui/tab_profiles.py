"""
Tabs 3 & 4: Churn / Hire profiles.
Both share the same shape - aggregate by JobTitle, compute ratios, format, render.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config.constants import MONTHS_PER_YEAR, TOP_N_PROFILE_DEFAULT
from forecasting.time_series import classify_voluntary

# Below this average headcount, one leaver moves a role's turnover by ≥ 10 pp
SMALL_SAMPLE_HEADCOUNT = 10


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


def _headcount_by_title(df: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Active headcount per JobTitle at the first day of each month (titles x months).

    Same 'active' definition as forecasting.time_series._headcount_series.
    """
    hire = df["HireDate"]
    dep = df["DepartureDate"]
    snapshots = {
        m: df.loc[(hire <= m) & (dep.isna() | (dep > m)), "JobTitle"].value_counts()
        for m in months
    }
    return pd.DataFrame(snapshots).fillna(0)


def _aggregate_turnover(
    df_all: pd.DataFrame, df_leavers: pd.DataFrame, months: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per-JobTitle turnover: leavers in period / average monthly headcount."""
    n_months = len(months)
    avg_hc = _headcount_by_title(df_all, months).mean(axis=1).rename("Avg Headcount")
    start_hc = _headcount_by_title(df_all, months[:1]).iloc[:, 0].rename("Headcount (Start)")
    # Active on the last day of the period (not on the 1st of the last month)
    period_end = pd.DatetimeIndex([months[-1] + pd.offsets.MonthEnd(0)])
    end_hc = _headcount_by_title(df_all, period_end).iloc[:, 0].rename("Headcount (End)")

    leavers = df_leavers.groupby("JobTitle").size().rename("Leavers")
    out = pd.concat([start_hc, end_hc, avg_hc, leavers], axis=1).fillna(0)

    if "Departure Type" in df_leavers.columns:
        is_vol = df_leavers["Departure Type"].map(classify_voluntary) == "Voluntary"
        out["Voluntary Leavers"] = (
            df_leavers[is_vol].groupby("JobTitle").size().reindex(out.index, fill_value=0)
        )
    if "TenureMonths" in df_leavers.columns:
        out["Avg Tenure of Leavers (Months)"] = df_leavers.groupby("JobTitle")["TenureMonths"].mean()

    hc = out["Avg Headcount"].where(out["Avg Headcount"] > 0)
    out["Period Turnover (%)"] = out["Leavers"] / hc * 100
    out["Annualized Turnover (%)"] = out["Period Turnover (%)"] * MONTHS_PER_YEAR / n_months
    if "Voluntary Leavers" in out.columns:
        out["Voluntary Turnover (%)"] = (
            out["Voluntary Leavers"] / hc * 100 * MONTHS_PER_YEAR / n_months
        )

    out.index.name = "JobTitle"
    return out.reset_index()


def _render_turnover_by_title(
    df_all: pd.DataFrame, df_leavers: pd.DataFrame, date_range: tuple
) -> None:
    st.markdown("### 🔄 Turnover Report per Job Title")

    needed = {"JobTitle", "HireDate", "DepartureDate"}
    if df_all.empty or not needed.issubset(df_all.columns):
        st.info("Δεν υπάρχουν τα απαραίτητα δεδομένα (JobTitle, Hire/Departure Date) για turnover.")
        return

    months = pd.date_range(
        pd.Timestamp(date_range[0]).replace(day=1), pd.Timestamp(date_range[1]), freq="MS"
    )
    if len(months) == 0:
        st.info("Η επιλεγμένη περίοδος δεν περιέχει ολόκληρο μήνα.")
        return

    df_all = df_all.dropna(subset=["JobTitle"])
    df_leavers = (
        df_leavers.dropna(subset=["JobTitle"]) if "JobTitle" in df_leavers.columns
        else df_leavers.iloc[0:0].assign(JobTitle=pd.Series(dtype=str))
    )
    report = _aggregate_turnover(df_all, df_leavers, months)
    st.caption(
        f"Περίοδος: **{months[0]:%Y-%m} → {months[-1]:%Y-%m} ({len(months)} μήνες)**. "
        "**Annualized turnover** = αποχωρήσεις / μέσο μηνιαίο headcount, αναγόμενο σε 12 μήνες — "
        "αυτό είναι συγκρίσιμο μεταξύ ρόλων. *Period turnover* = το ίδιο για όλη την περίοδο "
        "(σε πολυετή περίοδο μπορεί να ξεπερνά το 100%, όταν οι θέσεις αναπληρώνονται)."
    )

    # Voluntary columns duplicate Leavers when the departure-type filter keeps only voluntary exits
    vol_cols = ["Voluntary Leavers", "Voluntary Turnover (%)"]
    if "Voluntary Leavers" in report.columns and (report["Voluntary Leavers"] == report["Leavers"]).all():
        report = report.drop(columns=vol_cols)
        if report["Leavers"].sum() > 0:
            st.caption("Όλες οι αποχωρήσεις της επιλογής είναι εθελοντικές — οι στήλες *Voluntary* παραλείπονται.")

    # Overall rate for comparison
    total_avg_hc = report["Avg Headcount"].sum()
    total_leavers = report["Leavers"].sum()
    overall = (
        total_leavers / total_avg_hc * 100 * MONTHS_PER_YEAR / len(months) if total_avg_hc else float("nan")
    )
    report["vs Overall (pp)"] = report["Annualized Turnover (%)"] - overall

    c1, c2, c3 = st.columns(3)
    c1.metric("Annualized turnover (σύνολο)", _format_pct(overall))
    c2.metric("Αποχωρήσεις περιόδου", f"{int(total_leavers):,}")
    c3.metric("Μέσο headcount", f"{total_avg_hc:,.0f}")

    min_hc = st.number_input(
        "Ελάχιστο μέσο headcount ρόλου (αποκλείει μικρούς ρόλους με παραπλανητικά ποσοστά — "
        f"κάτω από {SMALL_SAMPLE_HEADCOUNT} άτομα ένας αποχωρών αλλάζει πολύ το ποσοστό):",
        min_value=0, value=5, step=1, key="turnover_min_hc",
    )
    report["Sample"] = np.where(report["Avg Headcount"] < SMALL_SAMPLE_HEADCOUNT,
                                f"⚠️ < {SMALL_SAMPLE_HEADCOUNT} άτομα", "OK")
    report = report[report["Avg Headcount"] >= min_hc].sort_values(
        ["Annualized Turnover (%)", "Leavers"], ascending=False, na_position="last"
    )
    if report.empty:
        st.info("Κανένας ρόλος δεν πληροί το ελάχιστο headcount.")
        return

    top_n = st.slider(
        "Display Top N Job Titles by Turnover:",
        min_value=5, max_value=50, value=TOP_N_PROFILE_DEFAULT, key="top_n_turnover_jobtitles",
    )

    display_cols = [c for c in [
        "JobTitle", "Headcount (Start)", "Headcount (End)", "Avg Headcount", "Leavers",
        "Voluntary Leavers", "Annualized Turnover (%)", "Voluntary Turnover (%)", "vs Overall (pp)",
        "Period Turnover (%)", "Avg Tenure of Leavers (Months)", "Sample",
    ] if c in report.columns]

    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        report[display_cols].head(top_n),
        width="stretch",
        hide_index=True,
        column_config={
            "Headcount (Start)": st.column_config.NumberColumn(format="%d"),
            "Headcount (End)": st.column_config.NumberColumn(format="%d"),
            "Avg Headcount": st.column_config.NumberColumn(format="%.1f"),
            "Leavers": st.column_config.NumberColumn(format="%d"),
            "Voluntary Leavers": st.column_config.NumberColumn(format="%d"),
            "Period Turnover (%)": pct,
            "Annualized Turnover (%)": pct,
            "Voluntary Turnover (%)": pct,
            "vs Overall (pp)": st.column_config.NumberColumn(format="%+.1f"),
            "Avg Tenure of Leavers (Months)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    csv = report[display_cols].round(2).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download Full Turnover Report (CSV)",
        data=csv,
        file_name="turnover_by_jobtitle.csv",
        mime="text/csv",
    )


def render_churn_tab(
    df_leavers_period: pd.DataFrame,
    total_months: int,
    df_filtered: pd.DataFrame | None = None,
    date_range: tuple | None = None,
) -> None:
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

    if df_filtered is not None and date_range is not None:
        st.markdown("---")
        _render_turnover_by_title(df_filtered, df_leavers_period, date_range)


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
