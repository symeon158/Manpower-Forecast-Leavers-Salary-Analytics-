"""
Prophet forecasting helpers + Optuna hyperparameter tuning + baseline.
All functions are pure (no Streamlit calls) except where caching is needed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

from config.constants import (
    DEFAULT_PROPHET_CHANGEPOINT_PRIOR,
    DEFAULT_PROPHET_SEASONALITY_PRIOR,
    DEFAULT_PROPHET_WEEKLY_SEASONALITY,
    DEFAULT_RECENT_MONTHS_FOR_TUNING,
)
from forecasting.time_series import get_recent_ts

# Silence Optuna logs in Streamlit
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------
# Internal
# --------------------------------------------------------------------
def _to_prophet_df(ts: pd.DataFrame, metric: str) -> pd.DataFrame:
    return ts[["Date", metric]].rename(columns={"Date": "ds", metric: "y"})


def _is_too_sparse(df_prophet: pd.DataFrame) -> bool:
    return df_prophet["y"].sum() == 0 or df_prophet["y"].dropna().shape[0] < 3


def _fit_prophet(
    df_prophet: pd.DataFrame,
    cp_scale: float,
    s_scale: float,
    weekly_seasonality: bool,
) -> Prophet:
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        changepoint_prior_scale=cp_scale,
        seasonality_prior_scale=s_scale,
    )
    m.fit(df_prophet)
    return m


# --------------------------------------------------------------------
# Optuna
# --------------------------------------------------------------------
def _optuna_objective(trial: optuna.Trial, ts: pd.DataFrame, metric: str) -> float:
    df_prophet = _to_prophet_df(ts, metric)
    if _is_too_sparse(df_prophet):
        return float("inf")

    cp = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
    sp = trial.suggest_float("seasonality_prior_scale", 0.1, 30.0, log=True)
    weekly = trial.suggest_categorical("weekly_seasonality", [True, False])

    try:
        m = _fit_prophet(df_prophet, cp, sp, weekly)
    except Exception:
        return float("inf")

    forecast = m.predict(df_prophet[["ds"]])
    return float(np.mean((df_prophet["y"].values - forecast["yhat"].values) ** 2))


@st.cache_data(show_spinner="🔍 Optuna hyperparameter optimization...")
def run_optuna_tuning(
    ts: pd.DataFrame,
    metric: str,
    n_trials: int,
    recent_months: int = DEFAULT_RECENT_MONTHS_FOR_TUNING,
) -> dict[str, Any]:
    """Search best Prophet hyperparams on the recent slice of `ts`."""
    if ts.empty:
        return {}
    df_recent = get_recent_ts(ts, months=recent_months)
    if _is_too_sparse(_to_prophet_df(df_recent, metric)):
        return {}

    study = optuna.create_study(direction="minimize")
    try:
        study.optimize(
            lambda t: _optuna_objective(t, df_recent, metric),
            n_trials=n_trials,
            show_progress_bar=False,
        )
    except Exception:
        return {}
    return study.best_params


# --------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------
def run_prophet_forecast(
    ts: pd.DataFrame,
    metric: str,
    periods: int,
    changepoint_prior_scale: float = DEFAULT_PROPHET_CHANGEPOINT_PRIOR,
    seasonality_prior_scale: float = DEFAULT_PROPHET_SEASONALITY_PRIOR,
    weekly_seasonality: bool = DEFAULT_PROPHET_WEEKLY_SEASONALITY,
) -> tuple[Prophet | None, pd.DataFrame | None, dict | None]:
    """Fit Prophet on `ts[metric]` and forecast `periods` months ahead."""
    df_prophet = _to_prophet_df(ts, metric)
    if _is_too_sparse(df_prophet):
        return None, None, None

    try:
        m = _fit_prophet(
            df_prophet,
            changepoint_prior_scale,
            seasonality_prior_scale,
            weekly_seasonality,
        )
    except Exception:
        return None, None, None

    future = m.make_future_dataframe(periods=periods, freq="MS")
    forecast = m.predict(future)
    return m, forecast, calculate_metrics(df_prophet, forecast)


# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------
def calculate_metrics(df_prophet: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """In-sample MAE & MAPE. Returns NaN when not computable."""
    merged = pd.merge(df_prophet, forecast, on="ds", how="inner").dropna(
        subset=["y", "yhat"]
    )
    if merged.empty:
        return {"mae": float("nan"), "mape": float("nan")}

    y_true = merged["y"].values
    y_pred = merged["yhat"].values
    mae = mean_absolute_error(y_true, y_pred)

    mask = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.any()
        else float("nan")
    )
    return {"mae": float(mae), "mape": mape}


def baseline_metrics(ts: pd.DataFrame, metric: str) -> dict | None:
    """Naive seasonal baseline: y_t = y_{t-12}."""
    if ts.empty:
        return None
    s = ts.set_index("Date")[metric].astype(float)
    if s.dropna().shape[0] < 13:
        return None

    baseline = s.shift(12)
    mask = baseline.notna() & s.notna()
    if not mask.any():
        return None

    y_true = s[mask].values
    y_pred = baseline[mask].values
    mae = mean_absolute_error(y_true, y_pred)
    nz = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100)
        if nz.any()
        else float("nan")
    )
    return {"mae": float(mae), "mape": mape}


# --------------------------------------------------------------------
# Narrative
# --------------------------------------------------------------------
def business_summary(
    ts: pd.DataFrame,
    metric: str,
    horizon: int,
    forecast: pd.DataFrame | None,
    metrics: dict | None,
) -> str:
    if ts.empty or forecast is None or metrics is None:
        return "Δεν υπάρχουν αρκετά δεδομένα για παραγωγή περίληψης."

    last_history = ts["Date"].max()
    future = forecast[forecast["ds"] > last_history]
    mae = metrics.get("mae", float("nan"))
    mape = metrics.get("mape", float("nan"))
    quality = (
        "ικανοποιητικής"
        if not np.isnan(mape) and mape < 50
        else "μέτριας / θορυβώδους"
    )

    return (
        f"- Το μοντέλο καλύπτει **{ts['Date'].min():%Y-%m} → {ts['Date'].max():%Y-%m}**.\n"
        f"- Συνολικά ιστορικά {metric}: **{ts[metric].sum():.0f}** "
        f"(μέσος όρος **{ts[metric].mean():.1f}** ανά μήνα).\n"
        f"- Για τους επόμενους **{horizon}** μήνες, πρόβλεψη **{future['yhat'].sum():.0f}** {metric}.\n"
        f"- MAE: **{mae:.2f}**, MAPE: **{mape:.1f}%** — {quality} προσαρμογή."
    )
