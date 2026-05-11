"""Tab 1: Forecast & Time Series."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from prophet.plot import plot_plotly

from forecasting import (
    baseline_metrics,
    business_summary,
    monthly_seasonality,
    run_prophet_forecast,
)


def render_forecast_tab(
    ts: pd.DataFrame,
    metric: str,
    horizon: int,
    cp_scale: float,
    s_scale: float,
    weekly_seasonality: bool,
) -> tuple:
    """Renders the forecast tab. Returns (model, forecast, model_metrics)."""
    model, forecast, metrics_model = run_prophet_forecast(
        ts,
        metric=metric,
        periods=horizon,
        changepoint_prior_scale=cp_scale,
        seasonality_prior_scale=s_scale,
        weekly_seasonality=weekly_seasonality,
    )

    st.markdown("### 🧾 Business Summary (HR Narrative)")
    if model is not None and metrics_model is not None:
        st.markdown(business_summary(ts, metric, horizon, forecast, metrics_model))
    else:
        st.info("Δεν υπάρχουν αρκετά δεδομένα ή forecast για δημιουργία περίληψης.")

    st.markdown("### 📊 Ιστορικό Time Series (επιλεγμένη περίοδος)")
    fig_hist = px.line(ts, x="Date", y=metric, markers=True,
                       title=f"Historical {metric} per Month")
    fig_hist.update_layout(xaxis_title="Month", yaxis_title=metric)
    st.plotly_chart(fig_hist, width="stretch")

    st.markdown("### 🔮 Optimized Forecast & Seasonality")

    if model is None or forecast is None:
        st.warning(
            f"Δεν υπάρχουν αρκετά μη-μηδενικά δεδομένα για {metric} ώστε να γίνει forecast. "
            "Δοκίμασε άλλο metric ή πιο γενικό φίλτρο."
        )
    else:
        st.subheader("Forecast – Ιστορικό + Μέλλον")
        fig_forecast = plot_plotly(model, forecast)
        fig_forecast.update_layout(xaxis_title="Month", yaxis_title=metric, legend_title="Legend")
        st.plotly_chart(fig_forecast, width="stretch")

        st.subheader("Prophet Components (Trend & Seasonality)")
        st.pyplot(model.plot_components(forecast))

        st.subheader("📆 Monthly Seasonality (ιστορικός μέσος όρος ανά μήνα)")
        ma = monthly_seasonality(ts, metric)
        fig_month = px.bar(ma, x="Month", y=metric, title=f"Average {metric} by Calendar Month")
        fig_month.update_layout(xaxis_title="Month (1-12)", yaxis_title=f"Average {metric}")
        st.plotly_chart(fig_month, width="stretch")

    # Model comparison
    st.markdown("### ⚖️ Model Comparison: Prophet vs Baseline")
    bm = baseline_metrics(ts, metric)
    if metrics_model is not None and bm is not None:
        comp_df = pd.DataFrame([
            {"Model": "Prophet", "MAE": metrics_model["mae"], "MAPE (%)": metrics_model["mape"]},
            {"Model": "Baseline (shift 12m)", "MAE": bm["mae"], "MAPE (%)": bm["mape"]},
        ])
        st.dataframe(
            comp_df.style.format({"MAE": "{:.2f}", "MAPE (%)": "{:.2f}"}),
            width="stretch",
        )
    else:
        st.info("Δεν ήταν δυνατή η σύγκριση Prophet με baseline (ανεπαρκή δεδομένα).")

    if model is not None and forecast is not None:
        st.markdown("### 📋 Forecast data table")
        view = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={
                "ds": "Date",
                "yhat": f"{metric}_forecast",
                "yhat_lower": "Lower CI",
                "yhat_upper": "Upper CI",
            }
        )
        st.dataframe(view, width="stretch")
        st.download_button(
            "⬇️ Download Forecast (CSV)",
            data=view.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{metric}_forecast_{horizon}m.csv",
            mime="text/csv",
        )

    return model, forecast, metrics_model
