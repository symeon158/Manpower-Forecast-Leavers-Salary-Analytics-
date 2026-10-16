"""
Forecast models + rolling-origin backtest + automatic model selection.

Why not "always Prophet"
------------------------
Monthly HR counts are short, noisy series. In rolling-origin backtests on the
group export (12-month-ahead forecasts from 12 origins over 3 years, for
departures / hires / net hires / single company / voluntary only), simple
seasonal methods beat Prophet's default settings by ~15% monthly MAE, and the
"flexible" Prophet settings that in-sample tuning picks were the worst of all
(~45% worse). No single model won every series, so the app backtests a few
candidates on the *currently selected* series and uses the best one.

Candidates
----------
- Seasonal mean      : last-12-month level × seasonal index (shrunk 50% toward 1)
- Rate × headcount   : departures only — recent turnover rate × seasonal index × headcount
- Prophet            : conservative priors (or Optuna-tuned on backtest error)
- Ensemble           : average of seasonal mean and Prophet

Prediction intervals are empirical, built from the selected model's backtest
errors (see `_intervals` / `_total_interval`), so their width reflects how wrong
it actually was.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from config.constants import (
    FORECAST_BACKTEST_HORIZON,
    FORECAST_BACKTEST_ORIGINS,
    FORECAST_BACKTEST_STEP_MONTHS,
    FORECAST_INTERVAL_LEVEL,
    FORECAST_MIN_HISTORY_MONTHS,
    FORECAST_SEASONAL_SHRINK,
)

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

ForecastFn = Callable[[pd.DataFrame, str, int], np.ndarray]


# --------------------------------------------------------------------
# Models: f(history, metric, h) -> np.ndarray of length h
# --------------------------------------------------------------------
def _future_months(hist: pd.DataFrame, h: int) -> np.ndarray:
    start = hist["Date"].iloc[-1] + pd.offsets.MonthBegin(1)
    return pd.date_range(start, periods=h, freq="MS").month.to_numpy()


def _seasonal_index(hist: pd.DataFrame, col: str, years: int = 4) -> pd.Series:
    """Multiplicative calendar-month index over the last `years`, shrunk toward 1."""
    tail = hist.tail(12 * years)
    overall = tail[col].mean()
    if not overall or np.isnan(overall):
        return pd.Series(1.0, index=range(1, 13))
    idx = (tail.groupby(tail["Date"].dt.month)[col].mean() / overall).reindex(range(1, 13)).fillna(1.0)
    return 1 + FORECAST_SEASONAL_SHRINK * (idx - 1)


def seasonal_mean(hist: pd.DataFrame, metric: str, h: int) -> np.ndarray:
    level = hist[metric].tail(12).mean()
    months = _future_months(hist, h)
    if (hist[metric] < 0).any() or level <= 0:
        # Additive profile for series that can be negative (NetHires)
        tail = hist.tail(48)
        profile = tail.groupby(tail["Date"].dt.month)[metric].mean() - tail[metric].mean()
        return np.array([level + FORECAST_SEASONAL_SHRINK * profile.get(m, 0.0) for m in months])
    idx = _seasonal_index(hist, metric)
    return np.array([level * idx[m] for m in months])


def rate_times_headcount(hist: pd.DataFrame, metric: str, h: int) -> np.ndarray:
    """Departures = recent monthly turnover rate × seasonal index × latest headcount."""
    hc = hist["Headcount"].replace(0, np.nan)
    tmp = hist.assign(_rate=hist["Departures"] / hc)
    rate = tmp["_rate"].tail(12).mean()
    if np.isnan(rate):
        return seasonal_mean(hist, metric, h)
    idx = _seasonal_index(tmp, "_rate")
    return np.array([rate * idx[m] * hist["Headcount"].iloc[-1] for m in _future_months(hist, h)])


def make_prophet(cp_scale: float, s_scale: float) -> ForecastFn:
    def prophet_fn(hist: pd.DataFrame, metric: str, h: int) -> np.ndarray:
        from prophet import Prophet

        df = hist[["Date", metric]].rename(columns={"Date": "ds", metric: "y"})
        m = Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            changepoint_prior_scale=cp_scale, seasonality_prior_scale=s_scale,
        )
        m.fit(df)
        out = m.predict(m.make_future_dataframe(periods=h, freq="MS"))["yhat"].to_numpy()[-h:]
        return out
    return prophet_fn


def make_ensemble(*fns: ForecastFn) -> ForecastFn:
    def ensemble_fn(hist: pd.DataFrame, metric: str, h: int) -> np.ndarray:
        return np.mean([f(hist, metric, h) for f in fns], axis=0)
    return ensemble_fn


def candidate_models(metric: str, cp_scale: float, s_scale: float) -> dict[str, ForecastFn]:
    prophet = make_prophet(cp_scale, s_scale)
    models = {
        "Seasonal mean": seasonal_mean,
        "Prophet": prophet,
        "Ensemble (seasonal mean + Prophet)": make_ensemble(seasonal_mean, prophet),
    }
    if metric == "Departures":
        models["Rate × headcount"] = rate_times_headcount
    return models


def _clip(metric: str, y: np.ndarray) -> np.ndarray:
    return y if metric == "NetHires" else np.clip(y, 0, None)


# --------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------
def backtest_origins(ts: pd.DataFrame, horizon: int = FORECAST_BACKTEST_HORIZON) -> list[pd.Timestamp]:
    """Forecast origins: every STEP months, latest one leaving `horizon` months of actuals."""
    last_origin = ts["Date"].max() - pd.DateOffset(months=horizon)
    first_allowed = ts["Date"].min() + pd.DateOffset(months=FORECAST_MIN_HISTORY_MONTHS - 1)
    origins = [
        last_origin - pd.DateOffset(months=k * FORECAST_BACKTEST_STEP_MONTHS)
        for k in range(FORECAST_BACKTEST_ORIGINS)
    ]
    return sorted(o for o in origins if o >= first_allowed)


def backtest(
    ts: pd.DataFrame, metric: str, models: dict[str, ForecastFn],
    horizon: int = FORECAST_BACKTEST_HORIZON,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Rolling-origin backtest. Returns (leaderboard, errors[model] as origins × horizon)."""
    errors: dict[str, list[np.ndarray]] = {name: [] for name in models}
    totals: dict[str, list[tuple[float, float]]] = {name: [] for name in models}
    for origin in backtest_origins(ts, horizon):
        hist = ts[ts["Date"] <= origin]
        actual = ts[ts["Date"] > origin][metric].head(horizon).to_numpy(float)
        if len(actual) < horizon:
            continue
        for name, fn in models.items():
            try:
                pred = _clip(metric, fn(hist, metric, horizon))
            except Exception:   # noqa: BLE001 — a failing candidate just drops out
                continue
            errors[name].append(actual - pred)
            totals[name].append((pred.sum(), actual.sum()))

    rows = []
    for name in models:
        if not errors[name]:
            continue
        e = np.vstack(errors[name])
        pred_tot, act_tot = np.array(totals[name]).T
        denom = np.abs(act_tot).mean()
        rows.append({
            "Model": name,
            "Monthly MAE": float(np.abs(e).mean()),
            "12m total error %": float(np.abs(pred_tot - act_tot).mean() / denom * 100) if denom else np.nan,
            "Bias / month": float(-e.mean()),     # + = over-forecasts
            "Origins": len(e),
        })
    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values("Monthly MAE").reset_index(drop=True)
    return board, {k: np.vstack(v) for k, v in errors.items() if v}


# --------------------------------------------------------------------
# Selection + final forecast
# --------------------------------------------------------------------
@dataclass
class ForecastResult:
    model_name: str
    forecast: pd.DataFrame          # Date, yhat, yhat_lower, yhat_upper (future months only)
    leaderboard: pd.DataFrame       # backtest results, best first (empty if too little history)
    backtest_mae: float
    backtest_total_err_pct: float
    interval_level: float
    total: float                    # forecast sum over the horizon
    total_lower: float              # interval for the sum (not the sum of monthly bounds)
    total_upper: float


def _intervals(errors: np.ndarray | None, yhat: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Monthly interval: `level` quantile of |backtest error| pooled over all horizon steps.

    Per-step quantiles from only ~8 origins were too narrow (73% actual coverage
    for a nominal 80% on unseen periods); pooling ~100 errors gave 89%, so we
    err on the cautious side. Widened by √(step/12) beyond the backtest horizon.
    """
    h = len(yhat)
    if errors is None or len(errors) < 3:
        return yhat, yhat
    q = float(np.quantile(np.abs(errors), level))
    hb = errors.shape[1]
    widen = np.sqrt(np.maximum(np.arange(1, h + 1), hb) / hb)
    return yhat - q * widen, yhat + q * widen


def _total_interval(
    errors: np.ndarray | None, total: float, h: int, level: float
) -> tuple[float, float]:
    """Interval for the h-month total from backtest errors of 12-month totals.

    Monthly errors partly cancel, so summing monthly bounds is far too wide.
    With few origins, a t multiplier (n−1 dof) corrects the small-sample RMSE.
    """
    from scipy.stats import t as student_t

    if errors is None or len(errors) < 3:
        return total, total
    totals = errors.sum(axis=1)
    rmse = float(np.sqrt(np.mean(totals ** 2)))
    half = student_t.ppf((1 + level) / 2, df=len(totals) - 1) * rmse * np.sqrt(h / errors.shape[1])
    return total - half, total + half


@st.cache_data(show_spinner="📈 Backtesting forecast models...")
def forecast_best(
    ts: pd.DataFrame,
    metric: str,
    horizon: int,
    cp_scale: float,
    s_scale: float,
    interval_level: float = FORECAST_INTERVAL_LEVEL,
) -> ForecastResult:
    """Backtest the candidates on `ts[metric]`, refit the best on all history, forecast `horizon`."""
    models = candidate_models(metric, cp_scale, s_scale)
    board, errors = backtest(ts, metric, models)

    if board.empty:
        name = "Seasonal mean"     # too little history to backtest — safest default
        mae = total_err = float("nan")
    else:
        name = board.iloc[0]["Model"]
        mae = float(board.iloc[0]["Monthly MAE"])
        total_err = float(board.iloc[0]["12m total error %"])

    yhat = _clip(metric, models[name](ts, metric, horizon))
    lo, hi = _intervals(errors.get(name), yhat, interval_level)
    t_lo, t_hi = _total_interval(errors.get(name), float(yhat.sum()), horizon, interval_level)
    dates = pd.date_range(ts["Date"].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    fc = pd.DataFrame({
        "Date": dates, "yhat": yhat,
        "yhat_lower": _clip(metric, lo), "yhat_upper": _clip(metric, hi),
    })
    if metric != "NetHires":
        t_lo = max(t_lo, 0.0)
    return ForecastResult(
        model_name=name, forecast=fc, leaderboard=board,
        backtest_mae=mae, backtest_total_err_pct=total_err, interval_level=interval_level,
        total=float(yhat.sum()), total_lower=float(t_lo), total_upper=float(t_hi),
    )
