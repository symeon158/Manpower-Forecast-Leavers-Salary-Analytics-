"""Reusable UI atoms: KPI cards, donut chart."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def kpi_card(
    label: str,
    value: str,
    subtitle: str | None = None,
    badge: str | None = None,
) -> None:
    """Render a KPI card. Styled by KPI_CSS (loaded in main app)."""
    parts = [
        '<div class="kpi-card">',
        f'<div class="kpi-label">{label}</div>',
        f'<div class="kpi-value">{value}</div>',
    ]
    if subtitle:
        parts.append(f'<div class="kpi-subtitle">{subtitle}</div>')
    if badge:
        parts.append(f'<div class="kpi-badge">{badge}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def create_donut_chart(percentage: float) -> go.Figure:
    """Small donut showing `percentage` covered vs remaining."""
    pct = max(0.0, min(100.0, float(percentage)))
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Covered", "Remaining"],
                values=[pct, 100 - pct],
                hole=0.7,
                marker_colors=["#1f77b4", "#e2e8f0"],
                textinfo="none",
                hoverinfo="skip",
                direction="clockwise",
                sort=False,
            )
        ]
    )
    fig.add_annotation(
        text=f"{pct:.1f}%",
        x=0.5, y=0.5,
        font=dict(size=14, color="#0d1b2a"),
        showarrow=False,
    )
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False,
        height=100, width=100,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def kpi_card_with_donut(
    label: str,
    value: str,
    subtitle: str,
    badge: str,
    percentage: float | None,
) -> None:
    """KPI card with a small donut on the right."""
    container = st.container()
    text_col, chart_col = container.columns([3, 1])
    with text_col:
        kpi_card(label=label, value=value, subtitle=subtitle, badge=badge)
    if percentage is None or percentage != percentage:  # NaN
        return
    with chart_col:
        st.markdown(
            '<div style="display:flex;justify-content:center;align-items:center;height:100%;">',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            create_donut_chart(percentage),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)


def shap_plot_safely(plot_func) -> None:
    """SHAP plots use plt.gcf(); wrap so Streamlit gets a clean figure."""
    import matplotlib.pyplot as plt

    plt.clf()
    plt.close("all")
    plot_func()
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()
    plt.close("all")
