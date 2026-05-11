"""Build monthly time series of hires / departures / headcount."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from config.constants import START_YEAR_FOR_TS


# --------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------
def _monthly_counts(dates: pd.Series, name: str) -> pd.Series:
    """Count rows per calendar month, indexed at month-start."""
    s = (
        dates.dropna()
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )
    s.index = s.index.to_timestamp()
    s.name = name
    return s


def _headcount_series(
    df: pd.DataFrame, monthly_index: pd.DatetimeIndex
) -> np.ndarray:
    """Active headcount snapshot at the first day of each month."""
    hire = df["HireDate"]
    dep = df["DepartureDate"]

    out = np.empty(len(monthly_index), dtype=int)
    for i, m in enumerate(monthly_index):
        out[i] = int(((hire <= m) & (dep.isna() | (dep > m))).sum())
    return out


def _resolve_date_range(
    df: pd.DataFrame, df_dep: pd.DataFrame, start_year: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_hire = df["HireDate"].min().replace(day=1)
    start = max(pd.Timestamp(datetime(start_year, 1, 1)), min_hire)

    if df_dep.empty:
        end_src = df["HireDate"].max() if not df["HireDate"].empty else datetime.now()
    else:
        end_src = df_dep["DepartureDate"].max()
    end = pd.Timestamp(end_src).replace(day=1)
    return start, end


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def build_monthly_time_series(
    df: pd.DataFrame,
    selected_dep_types: Sequence[str] | None = None,
    start_year: int = START_YEAR_FOR_TS,
) -> pd.DataFrame:
    """
    Build a monthly DataFrame with:
      Date | Hires | Departures | NetHires | Headcount | TurnoverRate

    Departures filtered by `selected_dep_types` (Departure Type column).
    """
    cols = ["Date", "Hires", "Departures", "NetHires", "Headcount", "TurnoverRate"]
    if df.empty or df["HireDate"].isna().all():
        return pd.DataFrame(columns=cols)

    # Departures (filtered)
    df_dep = df.dropna(subset=["DepartureDate"]).copy()
    if "Departure Type" in df_dep.columns and selected_dep_types:
        df_dep = df_dep[df_dep["Departure Type"].isin(selected_dep_types)]

    start, end = _resolve_date_range(df, df_dep, start_year)
    monthly_index = pd.date_range(start, end, freq="MS")
    if len(monthly_index) == 0:
        return pd.DataFrame(columns=cols)

    hires_s = _monthly_counts(df["HireDate"], "Hires")
    dep_s = _monthly_counts(df_dep["DepartureDate"], "Departures")

    ts = pd.DataFrame(index=monthly_index)
    ts.index.name = "Date"
    ts["Hires"] = hires_s.reindex(monthly_index, fill_value=0)
    ts["Departures"] = dep_s.reindex(monthly_index, fill_value=0)
    ts["NetHires"] = ts["Hires"] - ts["Departures"]
    ts["Headcount"] = _headcount_series(df, monthly_index)
    ts["TurnoverRate"] = np.where(
        ts["Headcount"] > 0, ts["Departures"] / ts["Headcount"] * 100, np.nan
    )
    return ts.reset_index()


def monthly_seasonality(ts: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average value per calendar month (1–12) for `metric`."""
    out = ts.copy()
    out["Month"] = out["Date"].dt.month
    return out.groupby("Month")[metric].mean().reset_index().sort_values("Month")


def get_recent_ts(ts: pd.DataFrame, months: int) -> pd.DataFrame:
    """Tail `months` rows for tuning purposes."""
    if ts.empty:
        return ts
    return ts.sort_values("Date").tail(months)


def classify_voluntary(dep_type: str | None) -> str:
    """Voluntary vs Non-voluntary classifier (with involuntary checked first)."""
    if pd.isna(dep_type):
        return "Other / Unknown"
    s = str(dep_type).lower()
    if "involuntary" in s:
        return "Non-voluntary / Other"
    if "voluntary" in s:
        return "Voluntary"
    return "Non-voluntary / Other"


def departure_insight_text(ts: pd.DataFrame) -> str:
    """Short auto-text insight on departures (for sidebar/tab)."""
    if ts.empty or ts["Departures"].sum() == 0:
        return "Δεν υπάρχουν αποχωρήσεις στο επιλεγμένο φίλτρο και στην επιλεγμένη περίοδο."

    max_row = ts.loc[ts["Departures"].idxmax()]
    return (
        f"🔍 *Insight για αποχωρήσεις:*\n\n"
        f"- Μήνας με τις περισσότερες αποχωρήσεις: **{max_row['Date']:%Y-%m}** "
        f"με **{int(max_row['Departures'])}** αποχωρήσεις.\n"
        f"- Μέσος όρος αποχωρήσεων ανά μήνα: **{ts['Departures'].mean():.1f}**.\n"
    )
