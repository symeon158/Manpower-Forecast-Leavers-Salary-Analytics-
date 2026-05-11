"""Tab 6: Pay Ranges (reference markets)."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loaders import load_pay_ranges_df


def render_pay_ranges_tab() -> None:
    st.markdown("### 📊 Pay Ranges – Reference Markets (Monthly)")
    df_pay = load_pay_ranges_df()

    c1, c2, c3 = st.columns(3)
    with c1:
        loc_sel = st.multiselect(
            "Location",
            options=sorted(df_pay["location"].unique().tolist()),
            default=sorted(df_pay["location"].unique().tolist()),
        )
    with c2:
        zone_sel = st.multiselect(
            "Pay Zone",
            options=sorted(df_pay["pay_zone"].unique().tolist()),
            default=sorted(df_pay["pay_zone"].unique().tolist()),
        )
    with c3:
        show_group = st.checkbox("Show grouped min/max columns", value=True)

    dfv = df_pay[df_pay["location"].isin(loc_sel) & df_pay["pay_zone"].isin(zone_sel)].copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{dfv.shape[0]:,}")
    k2.metric("Locations", f"{dfv['location'].nunique():,}")
    k3.metric("Pay Zones", f"{dfv['pay_zone'].nunique():,}")
    k4.metric("Ref Levels", f"{dfv['ref_level'].nunique():,}")

    st.markdown("#### 🧾 Data table")
    cols = ["ref_level", "location", "pay_zone", "p25", "p50", "p75", "rfl"]
    if show_group:
        cols += ["group_min", "group_max", "diff_min_max"]

    st.dataframe(
        dfv.sort_values(["ref_level_num", "pay_zone", "location"])[cols],
        width="stretch",
    )

    st.download_button(
        "⬇️ Download pay ranges (CSV)",
        data=dfv[cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="pay_ranges_reference_markets.csv",
        mime="text/csv",
    )

    _render_band_chart(dfv)
    _render_spread_heatmap(dfv)
    _render_drilldown(dfv)


def _render_band_chart(dfv) -> None:
    st.markdown("---")
    st.markdown("#### 📈 Range Band Chart (P25–P75 with Median)")
    st.caption("Δείχνει εύρος αγοράς (P25–P75) και τον median (P50) ανά Reference Level, ανά Pay Zone.")

    fig = go.Figure()
    for z in sorted(dfv["pay_zone"].unique()):
        dzz = dfv[dfv["pay_zone"] == z].sort_values("ref_level_num")
        if dzz.empty:
            continue

        x = dzz["ref_level_num"]
        fig.add_trace(go.Scatter(
            x=x, y=dzz["p75"], mode="lines", name=f"{z} P75",
            line=dict(width=1), showlegend=False,
            hovertemplate="Ref: %{x}<br>P75: %{y:,.0f}€<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=dzz["p25"], mode="lines", name=f"{z} P25",
            line=dict(width=1), fill="tonexty", opacity=0.25,
            hovertemplate="Ref: %{x}<br>P25: %{y:,.0f}€<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=dzz["p50"], mode="lines+markers", name=f"{z} P50",
            line=dict(width=2),
            hovertemplate="Ref: %{x}<br>P50: %{y:,.0f}€<extra></extra>",
        ))

    fig.update_layout(
        height=520,
        xaxis_title="Reference Level (S plotted as 14.5)",
        yaxis_title="Monthly Pay (€)",
        legend_title="Pay Zone",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def _render_spread_heatmap(dfv) -> None:
    st.markdown("---")
    st.markdown("#### 🔥 Heatmap: Spread (P75 - P25)")

    df = dfv.copy()
    df["spread"] = df["p75"] - df["p25"]

    spread_agg = df.groupby(["pay_zone", "ref_level"], as_index=False)["spread"].mean()
    pivot = spread_agg.pivot(index="pay_zone", columns="ref_level", values="spread")

    order = (
        df[["ref_level", "ref_level_num"]]
        .drop_duplicates()
        .sort_values("ref_level_num")["ref_level"]
        .tolist()
    )
    pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns])
    pivot = pivot.dropna(axis=1, how="all")

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="ref_level", y="pay_zone", color="Spread (€)"),
        title="Average spread (P75 - P25) by Pay Zone and Ref Level",
    )
    fig.update_xaxes(type="category")
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")


def _render_drilldown(dfv) -> None:
    st.markdown("---")
    st.markdown("#### 🎯 Row Drill-down (Bullet-style view)")
    st.caption("Επίλεξε μία γραμμή και δες P25/P50/P75 σε ένα mini chart.")

    if dfv.empty:
        return

    row_key = st.selectbox(
        "Select row",
        options=dfv.index.tolist(),
        format_func=lambda i: f"Ref {dfv.loc[i,'ref_level']} | {dfv.loc[i,'location']} | {dfv.loc[i,'pay_zone']}",
    )
    r = dfv.loc[row_key]

    bullet = go.Figure()
    bullet.add_trace(go.Bar(
        x=[r["p75"] - r["p25"]],
        y=["Range P25–P75"],
        base=[r["p25"]],
        orientation="h",
        hovertemplate="P25: %{base:,.0f}€<br>P75: %{x:,.0f}€<extra></extra>",
    ))
    bullet.add_vline(x=r["p50"], line_width=3, line_dash="solid")
    bullet.update_layout(
        height=160,
        xaxis_title="€ / month",
        yaxis_title="",
        title=f"Ref {r['ref_level']} | {r['location']} | {r['pay_zone']}  (P50 line)",
        showlegend=False,
    )
    st.plotly_chart(bullet, width="stretch")
