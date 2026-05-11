"""
Sidebar filter widgets. Single entry point: `render_sidebar_filters(df)`.
Returns a `FilterState` dataclass with everything the rest of the app needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from config.constants import EARLY_LEAVER_DEFAULT_MONTHS


@dataclass
class FilterState:
    df_filtered: pd.DataFrame
    company: str
    division: str
    department: str
    job_title: str
    job_property: str
    selected_dep_types: list[str] | None
    early_threshold: int
    use_optuna: bool
    n_trials: int
    recent_months: int
    available_dep_types: list[str] = field(default_factory=list)


def _select_unique(label: str, df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return "(All)"
    options = ["(All)"] + sorted(df[col].dropna().astype(str).unique().tolist())
    return st.sidebar.selectbox(label, options)


def _apply_filter(df: pd.DataFrame, col: str, value: str) -> pd.DataFrame:
    if value == "(All)" or col not in df.columns:
        return df
    return df[df[col] == value]


def render_sidebar_filters(df: pd.DataFrame) -> FilterState:
    st.sidebar.header("🔎 Φίλτρα")

    # Cascading filters
    company = _select_unique("Company", df, "Company")
    df_f = _apply_filter(df, "Company", company)

    division = _select_unique("Division", df_f, "Division")
    df_f = _apply_filter(df_f, "Division", division)

    department = _select_unique("Department", df_f, "Department")
    df_f = _apply_filter(df_f, "Department", department)

    job_title = _select_unique("Job Title", df_f, "JobTitle")
    df_f = _apply_filter(df_f, "JobTitle", job_title)

    job_property = _select_unique("Job Property", df_f, "Job Property")
    df_f = _apply_filter(df_f, "Job Property", job_property)

    # Departure type filter (only for leavers)
    available_dep_types: list[str] = []
    selected_dep_types: list[str] | None = None
    leavers_in_filter = df_f.dropna(subset=["DepartureDate"]) if "DepartureDate" in df_f.columns else pd.DataFrame()
    if not leavers_in_filter.empty and "Departure Type" in leavers_in_filter.columns:
        available_dep_types = sorted(leavers_in_filter["Departure Type"].dropna().unique().tolist())
        selected_dep_types = st.sidebar.multiselect(
            "Departure Type (Filters Leavers Only)",
            options=available_dep_types,
            default=available_dep_types,
            help="Επέλεξε τύπους αποχώρησης (π.χ. Voluntary, Retirement, Fixed-term).",
        )

    early_threshold = st.sidebar.slider(
        "Early leaver threshold (months)",
        min_value=3,
        max_value=24,
        value=EARLY_LEAVER_DEFAULT_MONTHS,
        step=1,
        help="Όριο μηνών υπηρεσίας κάτω από το οποίο ένας leaver θεωρείται 'early'.",
    )

    use_optuna = st.sidebar.checkbox("Χρήση Optuna tuning για Prophet", value=False)

    st.sidebar.header("⚙️ Prophet Tuning (Optuna HPO)")
    n_trials = st.sidebar.slider("Optuna Trials (Search Depth):", 10, 100, 30, step=10)
    recent_months = st.sidebar.slider(
        "History used for tuning (months):",
        min_value=12, max_value=60, value=36, step=12,
        help="Χρονικό βάθος ιστορικών δεδομένων για το Optuna.",
    )

    return FilterState(
        df_filtered=df_f,
        company=company,
        division=division,
        department=department,
        job_title=job_title,
        job_property=job_property,
        selected_dep_types=selected_dep_types,
        early_threshold=early_threshold,
        use_optuna=use_optuna,
        n_trials=n_trials,
        recent_months=recent_months,
        available_dep_types=available_dep_types,
    )
