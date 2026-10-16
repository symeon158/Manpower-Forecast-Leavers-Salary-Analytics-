"""
Prophet forecasting helpers + Optuna hyperparameter tuning (on backtest error).
Model selection / backtesting lives in forecasting/models.py.
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
    FORECAST_BACKTEST_HORIZON,
    OPTUNA_BACKTEST_ORIGINS,
)
from forecasting.models import backtest_origins, make_prophet

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
def _optuna_objective(
    trial: optuna.Trial, ts: pd.DataFrame, metric: str, origins: list[pd.Timestamp]
) -> float:
    """Mean absolute 12-month-ahead error over rolling origins.

    The previous objective was in-sample MSE, which rewards the most flexible
    settings (they fit history best) — and those forecast worst out of sample.
    """
    cp = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
    sp = trial.suggest_float("seasonality_prior_scale", 0.1, 30.0, log=True)
    fn = make_prophet(cp, sp)
    errors = []
    for origin in origins:
        hist = ts[ts["Date"] <= origin]
        actual = ts[ts["Date"] > origin][metric].head(FORECAST_BACKTEST_HORIZON).to_numpy(float)
        try:
            errors.append(np.abs(actual - fn(hist, metric, len(actual))).mean())
        except Exception:
            return float("inf")
    return float(np.mean(errors)) if errors else float("inf")


@st.cache_data(show_spinner="🔍 Optuna: tuning Prophet on rolling backtest error...")
def run_optuna_tuning(
    ts: pd.DataFrame,
    metric: str,
    n_trials: int,
    recent_months: int = DEFAULT_RECENT_MONTHS_FOR_TUNING,
) -> dict[str, Any]:
    """Search Prophet priors that minimise backtest error on origins within the last `recent_months`."""
    if ts.empty or _is_too_sparse(_to_prophet_df(ts, metric)):
        return {}
    cutoff = ts["Date"].max() - pd.DateOffset(months=recent_months)
    origins = [o for o in backtest_origins(ts) if o >= cutoff][-OPTUNA_BACKTEST_ORIGINS:]
    if not origins:
        return {}

    study = optuna.create_study(direction="minimize")
    try:
        study.optimize(
            lambda t: _optuna_objective(t, ts, metric, origins),
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
