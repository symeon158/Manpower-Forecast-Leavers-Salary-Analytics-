"""
Workforce Intelligence Platform - main entry point.

Run with:
    streamlit run app.py

Project structure:
    config/         - constants, static data
    data_loaders/   - CSV reading & preprocessing
    forecasting/    - Prophet, time series, Optuna
    ml/             - XGBoost training, SHAP
    ui/             - Streamlit components and tabs
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the directory containing this file is on sys.path so that the
# config/, data_loaders/, forecasting/, ml/, ui/ packages can be imported
# regardless of the current working directory streamlit was launched from.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# Apply NumPy 1.24+ compatibility shims for older SHAP versions BEFORE any
# third-party library that might import SHAP transitively. See _compat.py.
import _compat  # noqa: F401  (side-effect import)

import numpy as np
import pandas as pd
import streamlit as st

from config.constants import (
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_PROPHET_CHANGEPOINT_PRIOR,
    DEFAULT_PROPHET_SEASONALITY_PRIOR,
)
from data_loaders import load_leavers_data
from forecasting import build_monthly_time_series, forecast_best, run_optuna_tuning
from ui import HERO_CSS, HERO_HTML, KPI_CSS, kpi_card, kpi_card_with_donut, render_sidebar_filters
from ui.tab_executive import render_executive_tab
from ui.tab_forecast import render_forecast_tab
from ui.tab_ml import render_ml_tab
from ui.tab_pay_ranges import render_pay_ranges_tab
from ui.tab_profiles import render_churn_tab, render_hires_tab
from ui.tab_salary import render_salary_tab

logging.basicConfig(level=logging.WARNING)


# --------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------
st.set_page_config(page_title="Manpower Time Series & Leavers Salary", layout="wide")
st.markdown(KPI_CSS, unsafe_allow_html=True)
st.markdown(HERO_CSS + HERO_HTML, unsafe_allow_html=True)


# --------------------------------------------------------------------
# Sidebar - upload
# --------------------------------------------------------------------
st.sidebar.header("📂 Δεδομένα")
uploaded_file = st.sidebar.file_uploader(
    "Upload leavers/full employee CSV (; separator, Greek encoding)",
    type=["csv"],
    help="Export με Company, Division, Department, Job Title, Hire Date, Departure Date, ...",
)

if uploaded_file is None:
    st.info("👈 Ανέβασε πρώτα το αρχείο των εργαζομένων (CSV).")
    st.stop()


# --------------------------------------------------------------------
# Load data + filters
# --------------------------------------------------------------------
df = load_leavers_data(uploaded_file)
filters = render_sidebar_filters(df)
df_filtered = filters.df_filtered

if df_filtered.empty:
    st.warning("Δεν υπάρχουν δεδομένα για τα επιλεγμένα φίλτρα.")
    st.stop()

st.write(
    f"- **Company:** {filters.company}  \n"
    f"- **Division:** {filters.division}  \n"
    f"- **Department:** {filters.department}  \n"
    f"- **Job Title:** {filters.job_title}  \n"
    f"- **Job Property:** {filters.job_property}"
)


# --------------------------------------------------------------------
# Build time series + period selector
# --------------------------------------------------------------------
ts_full = build_monthly_time_series(df_filtered, filters.selected_dep_types)
if ts_full.empty:
    st.warning("Δεν βρέθηκαν επαρκή χρονικά δεδομένα (Hire/Departure).")
    st.stop()

global_start = ts_full["Date"].min().to_pydatetime()
global_end = ts_full["Date"].max().to_pydatetime()

st.markdown("### 🗓️ Περίοδος Ανάλυσης Time Series")

period_preset = st.radio(
    "Γρήγορη επιλογή περιόδου:",
    ["Όλη η διαθέσιμη", "Τελευταίοι 12 μήνες", "Τελευταίοι 24 μήνες"],
    horizontal=True,
)

if period_preset == "Τελευταίοι 12 μήνες":
    default_start = max(global_start, (global_end - pd.DateOffset(months=11)).to_pydatetime())
elif period_preset == "Τελευταίοι 24 μήνες":
    default_start = max(global_start, (global_end - pd.DateOffset(months=23)).to_pydatetime())
else:
    default_start = global_start

date_range = st.slider(
    "Επίλεξε την περίοδο ανάλυσης:",
    min_value=global_start, max_value=global_end,
    value=(default_start, global_end),
    format="YYYY-MM",
)

ts = ts_full[(ts_full["Date"] >= date_range[0]) & (ts_full["Date"] <= date_range[1])].copy()
if ts.empty:
    st.warning("Δεν υπάρχουν δεδομένα στην επιλεγμένη περίοδο.")
    st.stop()


# --------------------------------------------------------------------
# Metric & horizon
# --------------------------------------------------------------------
metric = st.selectbox("Metric για forecast:", ["Departures", "Hires", "NetHires"], index=0)
horizon = st.slider("Forecast horizon (μήνες):", 6, 36, DEFAULT_FORECAST_HORIZON)


# --------------------------------------------------------------------
# Optuna params
# --------------------------------------------------------------------
if st.sidebar.button("Re-Run Optuna HPO"):
    # Only invalidate the Optuna cache (NOT the entire data cache)
    run_optuna_tuning.clear()  # type: ignore[attr-defined]

if filters.use_optuna:
    best_params = run_optuna_tuning(
        ts, metric, filters.n_trials, recent_months=filters.recent_months
    )
else:
    best_params = {}

if best_params:
    st.sidebar.markdown("**✅ Best Parameters Found:**")
    st.sidebar.json({k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()})
    final_cp = best_params.get("changepoint_prior_scale", DEFAULT_PROPHET_CHANGEPOINT_PRIOR)
    final_sp = best_params.get("seasonality_prior_scale", DEFAULT_PROPHET_SEASONALITY_PRIOR)
else:
    if filters.use_optuna:
        st.sidebar.markdown("**⚠️ Optuna δεν επέστρεψε αποτελέσματα – default τιμές.**")
    final_cp = DEFAULT_PROPHET_CHANGEPOINT_PRIOR
    final_sp = DEFAULT_PROPHET_SEASONALITY_PRIOR


# --------------------------------------------------------------------
# Top KPI row
# --------------------------------------------------------------------
total_dep = int(ts["Departures"].sum())
total_hires = int(ts["Hires"].sum())
last_dep = int(ts.iloc[-1]["Departures"]) if not ts.empty else 0
period_label = f"{ts['Date'].min():%Y-%m} → {ts['Date'].max():%Y-%m}"

# One backtest + forecast, shared by the KPI row and the forecast tab (cached).
fc = forecast_best(ts, metric, horizon, final_cp, final_sp)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    kpi_card("Συνολικές αποχωρήσεις", f"{total_dep:,}", period_label, "Departures")
with c2:
    kpi_card("Συνολικές προσλήψεις", f"{total_hires:,}", period_label, "Hires")
with c3:
    kpi_card("Τελευταίος μήνας - αποχωρήσεις", str(last_dep),
             f"{ts['Date'].max():%Y-%m}", "Latest complete month")
with c4:
    if not np.isnan(fc.backtest_mae):
        kpi_card("Backtest MAE / μήνα", f"±{fc.backtest_mae:.1f}", f"Metric: {metric}", fc.model_name)
    else:
        kpi_card("Backtest MAE / μήνα", "N/A")
with c5:
    if not np.isnan(fc.backtest_total_err_pct):
        kpi_card("Σφάλμα 12μηνου συνόλου", f"{fc.backtest_total_err_pct:.0f}%",
                 f"Metric: {metric}", "Backtest accuracy")
    else:
        kpi_card("Σφάλμα 12μηνου συνόλου", "N/A")
with c6:
    kpi_card(
        f"Total Predicted {metric}",
        f"{fc.total:,.0f}",
        f"Next {horizon} months · {fc.total_lower:,.0f}–{fc.total_upper:,.0f}",
        "Forecast",
    )

st.markdown(f"**Περίοδος δεδομένων:** {period_label}")
st.markdown("---")


# --------------------------------------------------------------------
# Leavers in period & second KPI row (donuts)
# --------------------------------------------------------------------
df_leavers_period = (
    df_filtered.dropna(subset=["DepartureDate"]).copy()
    if "DepartureDate" in df_filtered.columns
    else pd.DataFrame()
)
if not df_leavers_period.empty:
    if "Departure Type" in df_leavers_period.columns and filters.selected_dep_types:
        df_leavers_period = df_leavers_period[
            df_leavers_period["Departure Type"].isin(filters.selected_dep_types)
        ]
    df_leavers_period = df_leavers_period[
        (df_leavers_period["DepartureDate"] >= date_range[0])
        & (df_leavers_period["DepartureDate"] <= date_range[1] + pd.offsets.MonthEnd(0))
    ]

# Total of all leaver types in same period (denominator)
df_leavers_all = (
    df_filtered.dropna(subset=["DepartureDate"]).dropna(subset=["NominalSalary"])
    if "DepartureDate" in df_filtered.columns and "NominalSalary" in df_filtered.columns
    else pd.DataFrame()
)
if not df_leavers_all.empty:
    df_leavers_all = df_leavers_all[
        (df_leavers_all["DepartureDate"] >= date_range[0])
        & (df_leavers_all["DepartureDate"] <= date_range[1] + pd.offsets.MonthEnd(0))
    ]

total_leavers_curr = df_leavers_period.shape[0]
total_leavers_all = df_leavers_all.shape[0]
pct_of_total = (total_leavers_curr / total_leavers_all * 100) if total_leavers_all > 0 else 0.0

if not df_leavers_period.empty:
    df_leavers_period["TenureMonths"] = (
        (df_leavers_period["DepartureDate"] - df_leavers_period["HireDate"]).dt.days / 30.4
    )
    early_count = int((df_leavers_period["TenureMonths"] < filters.early_threshold).sum())
    early_pct = (early_count / total_leavers_curr * 100) if total_leavers_curr else 0.0
else:
    early_count, early_pct = 0, 0.0

# Type label
if not filters.selected_dep_types or len(filters.selected_dep_types) == len(filters.available_dep_types):
    type_label = " (Όλοι οι Τύποι)"
elif len(filters.selected_dep_types) > 2:
    type_label = f" ({', '.join(filters.selected_dep_types[:2])} κ.α.)"
else:
    type_label = f" ({', '.join(filters.selected_dep_types)})"

period_text = f" ({ts['Date'].min():%Y-%m} → {ts['Date'].max():%Y-%m})"
full_label = f"Leavers{period_text}{type_label}"

kpi_cols = st.columns(2)
with kpi_cols[0]:
    kpi_card_with_donut(
        label=full_label,
        value=f"{total_leavers_curr:,}",
        subtitle=f"{pct_of_total:.1f}% of all leavers ({total_leavers_all:,})",
        badge="Selected leavers",
        percentage=pct_of_total,
    )
with kpi_cols[1]:
    kpi_card_with_donut(
        label=f"Early leavers (<{filters.early_threshold} μήνες)",
        value=f"{early_count:,}",
        subtitle=f"{early_pct:.1f}% του συνόλου",
        badge="Onboarding risk",
        percentage=early_pct,
    )

st.markdown("---")


# --------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------
tab_exec, tab_forecast, tab_salary, tab_churn, tab_hires, tab_ml, tab_pay = st.tabs([
    "🏛️ Executive Summary",
    "📈 Forecast & Time Series",
    "💶 Salary & Grade",
    "🎯 Churn Profiles",
    "🧲 Hire Profiles",
    "🤖 Attrition ML (XGBoost & SHAP)",
    "💰 Pay Ranges",
])

with tab_exec:
    scope = " · ".join(
        f"{k}: {v}" for k, v in [("Company", filters.company), ("Division", filters.division),
                                 ("Department", filters.department), ("Job Title", filters.job_title),
                                 ("Job Property", filters.job_property)]
        if v and v != "(All)"
    ) or "Όλος ο όμιλος"
    render_executive_tab(df_filtered, uploaded_file, scope)

with tab_forecast:
    render_forecast_tab(ts, metric, horizon, final_cp, final_sp, fc)

with tab_salary:
    render_salary_tab(df_filtered, df_leavers_period, metric, date_range, company=filters.company)

with tab_churn:
    render_churn_tab(
        df_leavers_period, total_months=max(ts.shape[0], 1),
        df_filtered=df_filtered, date_range=date_range,
    )

with tab_hires:
    render_hires_tab(df_filtered, date_range, total_months=max(ts.shape[0], 1))

with tab_ml:
    render_ml_tab(uploaded_file)

with tab_pay:
    render_pay_ranges_tab()
