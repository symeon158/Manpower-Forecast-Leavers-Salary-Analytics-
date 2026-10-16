"""Tab 1: Forecast & Time Series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecasting import ForecastResult, monthly_seasonality, run_prophet_forecast


def _business_summary(ts: pd.DataFrame, metric: str, horizon: int, fc: ForecastResult) -> str:
    lines = [
        f"- Ιστορικό: **{ts['Date'].min():%Y-%m} → {ts['Date'].max():%Y-%m}** (μόνο ολοκληρωμένοι μήνες). "
        f"Συνολικά {metric}: **{ts[metric].sum():.0f}** (μέσος όρος **{ts[metric].mean():.1f}** / μήνα, "
        f"τελευταίοι 12 μήνες: **{ts[metric].tail(12).sum():.0f}**).",
        f"- Πρόβλεψη επόμενων **{horizon}** μηνών: **{fc.total:.0f}** {metric} "
        f"(εύρος {fc.interval_level:.0%}: {fc.total_lower:.0f} – {fc.total_upper:.0f}).",
        f"- Μοντέλο: **{fc.model_name}** — επιλέχθηκε αυτόματα ως το πιο ακριβές στο backtest.",
    ]
    if not np.isnan(fc.backtest_mae):
        lines.append(
            f"- Ιστορική ακρίβεια (backtest, πρόβλεψη 12 μηνών μπροστά): μέσο σφάλμα "
            f"**±{fc.backtest_mae:.1f}** {metric} / μήνα, σφάλμα στο 12μηνο σύνολο "
            f"**{fc.backtest_total_err_pct:.0f}%**."
        )
    else:
        lines.append("- Λίγο ιστορικό για backtest — χρησιμοποιείται seasonal mean χωρίς διάστημα εμπιστοσύνης.")
    return "\n".join(lines)


def _forecast_chart(ts: pd.DataFrame, metric: str, fc: ForecastResult) -> go.Figure:
    f = fc.forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([f["Date"], f["Date"][::-1]]),
        y=pd.concat([f["yhat_upper"], f["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.18)", line=dict(width=0),
        name=f"{fc.interval_level:.0%} interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=ts["Date"], y=ts[metric], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(
        x=f["Date"], y=f["yhat"], mode="lines+markers", name=f"Forecast ({fc.model_name})",
        line=dict(dash="dash"),
    ))
    fig.update_layout(xaxis_title="Month", yaxis_title=metric, legend_title="", hovermode="x unified")
    return fig


def render_forecast_tab(
    ts: pd.DataFrame,
    metric: str,
    horizon: int,
    cp_scale: float,
    s_scale: float,
    fc: ForecastResult,
) -> None:
    st.markdown("### 🧾 Business Summary (HR Narrative)")
    st.markdown(_business_summary(ts, metric, horizon, fc))

    st.markdown("### 🔮 Forecast")
    st.plotly_chart(_forecast_chart(ts, metric, fc), width="stretch")
    st.caption(
        f"Το διάστημα {fc.interval_level:.0%} βασίζεται στα πραγματικά σφάλματα του μοντέλου στο backtest "
        "(όχι σε θεωρητικές υποθέσεις)."
    )

    st.markdown("### ⚖️ Model selection (rolling-origin backtest)")
    if fc.leaderboard.empty:
        st.info("Χρειάζονται τουλάχιστον 4 χρόνια ολοκληρωμένων μηνών για backtest.")
    else:
        st.caption(
            "Κάθε μοντέλο προβλέπει 12 μήνες μπροστά από διάφορα σημεία του παρελθόντος, χρησιμοποιώντας "
            "μόνο τα δεδομένα μέχρι εκείνο το σημείο, και συγκρίνεται με ό,τι πραγματικά έγινε. "
            "Επιλέγεται αυτό με το μικρότερο μηνιαίο MAE. *Bias* > 0 = υπερεκτίμηση."
        )
        st.dataframe(
            fc.leaderboard.style.format({
                "Monthly MAE": "{:.2f}", "12m total error %": "{:.1f}%", "Bias / month": "{:+.2f}",
            }).apply(
                lambda r: ["font-weight: bold" if r["Model"] == fc.model_name else ""] * len(r), axis=1
            ),
            width="stretch", hide_index=True,
        )

    st.markdown("### 📊 Ιστορικό & εποχικότητα")
    fig_hist = px.line(ts, x="Date", y=metric, markers=True, title=f"Historical {metric} per Month")
    fig_hist.update_layout(xaxis_title="Month", yaxis_title=metric)
    st.plotly_chart(fig_hist, width="stretch")

    ma = monthly_seasonality(ts, metric)
    fig_month = px.bar(ma, x="Month", y=metric, title=f"Average {metric} by Calendar Month")
    fig_month.update_layout(xaxis_title="Month (1-12)", yaxis_title=f"Average {metric}")
    st.plotly_chart(fig_month, width="stretch")

    with st.expander("Prophet components (trend & seasonality)", expanded=False):
        model, prophet_fc, _ = run_prophet_forecast(
            ts, metric=metric, periods=horizon,
            changepoint_prior_scale=cp_scale, seasonality_prior_scale=s_scale,
        )
        if model is None:
            st.info(f"Δεν υπάρχουν αρκετά μη-μηδενικά δεδομένα για {metric} ώστε να γίνει Prophet fit.")
        else:
            st.pyplot(model.plot_components(prophet_fc))

    st.markdown("### 📋 Forecast data table")
    view = fc.forecast.rename(columns={
        "yhat": f"{metric}_forecast",
        "yhat_lower": f"Lower ({fc.interval_level:.0%})",
        "yhat_upper": f"Upper ({fc.interval_level:.0%})",
    }).round(1)
    view["Date"] = view["Date"].dt.strftime("%Y-%m")
    st.dataframe(view, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Download Forecast (CSV)",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{metric}_forecast_{horizon}m.csv",
        mime="text/csv",
    )
