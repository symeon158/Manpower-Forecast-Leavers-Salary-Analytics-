"""Executive Summary tab: the 2-minute view for upper management.

Leads with money and risk: cost of voluntary attrition, where it concentrates
(statistically, not by raw rate), what the next 12 months look like, and where
retention spending would pay off (€ at risk from the ML model).
"""
from __future__ import annotations

import html
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from config.constants import (
    DEFAULT_PROPHET_CHANGEPOINT_PRIOR,
    DEFAULT_PROPHET_SEASONALITY_PRIOR,
    EXEC_AMBER_TOLERANCE,
    EXEC_TARGET_EARLY_LEAVER_SHARE,
    EXEC_TARGET_VOLUNTARY_TURNOVER,
    ML_TARGET_COMPANY,
    ML_WORK_RELATIONSHIP,
    REPLACEMENT_COST_RATES,
)
from forecasting import build_monthly_time_series, classify_voluntary, forecast_best
from insights.executive import (
    LEVEL_NAMES,
    MIN_UNIT_HEADCOUNT,
    P_HOTSPOT,
    P_WATCH,
    euro_at_risk,
    funnel_limits,
    period_kpis,
    rolling_windows,
    status_vs_target,
    unit_breakdown,
)
from ui.viz_theme import STATUS, STATUS_ICON, hex_to_rgba, theme

STATUS_STYLE = {   # funnel status -> (status color key, icon)
    "Significant hotspot": ("critical", "⛔"),
    "Watch": ("warning", "⚠️"),
    "Better than average": ("good", "✅"),
    "Within normal range": (None, "•"),
    "Too small to judge": (None, "·"),
}


# --------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------
def eur(x: float) -> str:
    if x is None or np.isnan(x):
        return "N/A"
    if abs(x) >= 1e6:
        return f"€{x / 1e6:.2f}m"
    if abs(x) >= 1e5:
        return f"€{x / 1e3:.0f}k"
    if abs(x) >= 1e3:
        return f"€{x / 1e3:.1f}k"
    return f"€{x:,.0f}"


def pct(x: float, digits: int = 1) -> str:
    return "N/A" if x is None or np.isnan(x) else f"{x * 100:.{digits}f}%"


def _rates_from_state() -> dict[int, float]:
    return st.session_state.get("exec_rates", dict(REPLACEMENT_COST_RATES))


def _render_assumptions() -> dict[int, float]:
    rates = _rates_from_state()
    with st.expander("⚙️ Υποθέσεις κόστους αντικατάστασης (placeholder — να επιβεβαιωθούν από HR / Finance)"):
        st.caption(
            "Κόστος αντικατάστασης = μηνιαίος ονομαστικός μισθός × μισθοί ανά έτος (14 για τις ελληνικές "
            "εταιρείες, 12 για τις υπόλοιπες) × ποσοστό ανά επίπεδο θέσης (από τον τίτλο θέσης). "
            "Κενός / μηδενικός μισθός → διάμεσος της ίδιας εταιρείας & επιπέδου. Περιλαμβάνει πρόσληψη, "
            "εκπαίδευση, κενό θέσης και χρόνο προσαρμογής. Αλλάξτε τα ποσοστά — όλα τα € ποσά ενημερώνονται."
        )
        table = pd.DataFrame({
            "Level": list(LEVEL_NAMES), "Job level": list(LEVEL_NAMES.values()),
            "% of annual salary": [rates[k] * 100 for k in LEVEL_NAMES],
        })
        edited = st.data_editor(
            table, hide_index=True, disabled=["Level", "Job level"], key="exec_rates_editor",
            column_config={"% of annual salary": st.column_config.NumberColumn(
                min_value=0, max_value=400, step=5, format="%d%%")},
        )
        new_rates = {int(r["Level"]): float(r["% of annual salary"]) / 100 for _, r in edited.iterrows()}
        if new_rates != rates:
            st.session_state["exec_rates"] = new_rates
            rates = new_rates
    return rates


# --------------------------------------------------------------------
# KPI tiles
# --------------------------------------------------------------------
def _status_chip(status: str, text: str) -> str:
    color = STATUS.get(status)
    icon = STATUS_ICON.get(status, "•")
    border = f"border-left: 4px solid {color};" if color else ""
    return (f'<div style="font-size:0.82rem; padding:2px 8px; margin-top:-6px; {border}">'
            f'{icon} {html.escape(text)}</div>')


def _target_status_text(status: str, target: float) -> str:
    return {"good": f"Εντός στόχου (≤ {pct(target, 0)})",
            "warning": f"Κοντά στον στόχο ({pct(target, 0)})",
            "critical": f"Εκτός στόχου ({pct(target, 0)})"}.get(status, "")


def _render_kpis(cur, prev, projection) -> None:
    cols = st.columns(6)
    vol_status = status_vs_target(cur.voluntary_rate, EXEC_TARGET_VOLUNTARY_TURNOVER, EXEC_AMBER_TOLERANCE)
    early_status = status_vs_target(cur.early_leaver_share, EXEC_TARGET_EARLY_LEAVER_SHARE, EXEC_AMBER_TOLERANCE)
    cost_delta = (cur.cost / prev.cost - 1) if prev.cost else float("nan")
    with cols[0]:
        st.metric("Headcount", f"{cur.headcount_end:,}",
                  f"{cur.headcount_end - prev.headcount_end:+,} vs 12m πριν", delta_color="off")
    with cols[1]:
        st.metric("Voluntary turnover (12m)", pct(cur.voluntary_rate),
                  f"{(cur.voluntary_rate - prev.voluntary_rate) * 100:+.1f} pp", delta_color="inverse")
        st.markdown(_status_chip(vol_status, _target_status_text(vol_status, EXEC_TARGET_VOLUNTARY_TURNOVER)),
                    unsafe_allow_html=True)
    with cols[2]:
        st.metric("Voluntary leavers (12m)", f"{cur.voluntary_leavers:,}",
                  f"{cur.voluntary_leavers - prev.voluntary_leavers:+,}", delta_color="inverse")
    with cols[3]:
        st.metric("Κόστος εθελοντικών αποχωρήσεων (12m)", eur(cur.cost),
                  "N/A" if np.isnan(cost_delta) else f"{cost_delta * 100:+.0f}%", delta_color="inverse",
                  help=f"Εκτιμώμενο κόστος αντικατάστασης — βλ. υποθέσεις. Χωρίς τις {cur.exec_leavers} "
                       f"αποχωρήσεις διευθυντικού επιπέδου: {eur(cur.cost_excl_exec)} "
                       f"(πέρσι {eur(prev.cost_excl_exec)}).")
    with cols[4]:
        st.metric("Αποχωρήσεις < 12 μηνών", pct(cur.early_leaver_share, 0),
                  f"{(cur.early_leaver_share - prev.early_leaver_share) * 100:+.0f} pp", delta_color="inverse",
                  help="Μερίδιο των εθελοντικών αποχωρήσεων με προϋπηρεσία < 12 μήνες (επιλογή / onboarding).")
        st.markdown(_status_chip(early_status, _target_status_text(early_status, EXEC_TARGET_EARLY_LEAVER_SHARE)),
                    unsafe_allow_html=True)
    with cols[5]:
        if projection:
            st.metric("Πρόβλεψη εθελοντικών αποχωρήσεων (επόμενοι 12m)", f"{projection['exits']:,.0f}",
                      f"≈ {eur(projection['cost'])}", delta_color="off",
                      help=f"Εύρος 80%: {projection['lo']:,.0f}–{projection['hi']:,.0f} αποχωρήσεις. "
                           f"Μοντέλο: {projection['model']}. Κόστος = προβλεπόμενες αποχωρήσεις × "
                           f"{eur(projection['unit_cost'])} (φετινό συνολικό κόστος ÷ αποχωρήσεις).")
        else:
            st.metric("Πρόβλεψη (επόμενοι 12m)", "N/A")


# --------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------
def _cost_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, color: str) -> go.Figure:
    d = df.sort_values(value_col, ascending=True).tail(12)
    labels = [eur(v) if i >= len(d) - 3 else "" for i, v in enumerate(d[value_col])]   # top 3 labelled
    fig = go.Figure(go.Bar(
        x=d[value_col], y=d[label_col], orientation="h", marker=dict(color=color, cornerradius=4),
        text=labels, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:,.0f} €<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=max(260, 34 * len(d) + 90),
                      margin=dict(l=10, r=60, t=50, b=10), bargap=0.35, showlegend=False)
    fig.update_xaxes(tickprefix="€", tickformat="~s", gridcolor=theme()["grid"], zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def _funnel_plot(units: pd.DataFrame, p0: float, by: str) -> go.Figure:
    th = theme()
    fig = go.Figure()
    lim95 = funnel_limits(p0, units["Avg headcount"].max(), P_WATCH)
    lim998 = funnel_limits(p0, units["Avg headcount"].max(), P_HOTSPOT)
    for lim, dash, name in ((lim95, "dash", "95% limits"), (lim998, "dot", "99.8% limits")):
        for side in ("upper", "lower"):
            fig.add_trace(go.Scatter(
                x=lim["n"], y=lim[side] * 100, mode="lines", line=dict(color=th["muted"], width=1.5, dash=dash),
                name=name, legendgroup=name, showlegend=side == "upper", hoverinfo="skip",
            ))
    fig.add_hline(y=p0 * 100, line=dict(color=th["ref"], width=1.5),
                  annotation_text=f"Συνολικό {pct(p0)}", annotation_position="top left")
    for status, (color_key, icon) in STATUS_STYLE.items():
        d = units[units["Status"] == status]
        if d.empty:
            continue
        color = STATUS[color_key] if color_key else (th["series"][0] if status == "Within normal range" else th["muted"])
        fig.add_trace(go.Scatter(
            x=d["Avg headcount"], y=d["Turnover rate"] * 100, mode="markers", name=f"{icon} {status}",
            marker=dict(size=10 if color_key else 8, color=color if color_key else hex_to_rgba(color, 0.55),
                        line=dict(width=1.5, color="rgba(255,255,255,0.9)")),
            customdata=np.column_stack([d[by], d["Voluntary leavers"], d["Expected leavers"], d["Cost (€)"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>Μέσο headcount %{x:.0f}<br>Turnover %{y:.1f}%"
                          "<br>Αποχωρήσεις %{customdata[1]:.0f} (αναμενόμενες %{customdata[2]:.1f})"
                          "<br>Κόστος €%{customdata[3]:,.0f}<extra>" + status + "</extra>",
        ))
    flagged = units[units["Status"].isin(["Significant hotspot", "Watch"])]
    for i, (_, r) in enumerate(flagged.sort_values("Avg headcount").iterrows()):
        # log x-axis: annotation positions are given in log10 units; alternate offsets to avoid collisions
        fig.add_annotation(x=np.log10(r["Avg headcount"]), y=r["Turnover rate"] * 100, text=str(r[by]),
                           showarrow=False, xanchor="left", xshift=8, yshift=9 if i % 2 == 0 else -9,
                           font=dict(size=11))
    ymax = min(units.loc[units["Avg headcount"] >= MIN_UNIT_HEADCOUNT, "Turnover rate"].max() * 100 * 1.15, 100) \
        if (units["Avg headcount"] >= MIN_UNIT_HEADCOUNT).any() else 100
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                      legend=dict(orientation="h", y=1.08, x=0, title_text=""))
    sizes = units["Avg headcount"].clip(lower=0.5)
    fig.update_xaxes(title_text="Μέσο headcount (μέγεθος μονάδας, log)", gridcolor=th["grid"], zeroline=False,
                     type="log", range=[np.log10(sizes.min() * 0.8), np.log10(sizes.max() * 1.6)])
    fig.update_yaxes(title_text="Εθελοντικό turnover (12m, %)", ticksuffix="%", range=[0, max(ymax, p0 * 300)],
                     gridcolor=th["grid"], zeroline=False)
    return fig


# --------------------------------------------------------------------
# Narrative
# --------------------------------------------------------------------
def _narrative(cur, prev, units, by, projection, risk) -> list[str]:
    arrow = "▲" if cur.voluntary_rate > prev.voluntary_rate else "▼"
    def change(a: float, b: float) -> str:
        return f"{(a / b - 1) * 100:+.0f}% έναντι {eur(b)}" if b else f"έναντι {eur(b)}"

    by_lvl = cur.cost_by_level
    cost_line = (f"Εκτιμώμενο κόστος αντικατάστασης: **{eur(cur.cost)}** ({change(cur.cost, prev.cost)}). "
                 f"Τυπικό κόστος ανά αποχώρηση **{eur(cur.median_cost_per_leaver)}** (διάμεσος)")
    if len(by_lvl) > 1:
        cost_line += (f" — από {eur(by_lvl['Typical cost'].iloc[0])} ({by_lvl['Level'].iloc[0].lower()}) "
                      f"έως {eur(by_lvl['Typical cost'].iloc[-1])} ({by_lvl['Level'].iloc[-1].lower()})")
    cost_line += "."
    if cur.exec_leavers:
        cost_line += (f" Από το συνολικό κόστος, **{eur(cur.exec_cost)}** αφορά **{cur.exec_leavers}** αποχωρήσεις διευθυντικού "
                      f"επιπέδου· χωρίς αυτές: **{eur(cur.cost_excl_exec)}** ({change(cur.cost_excl_exec, prev.cost_excl_exec)}).")
    lines = [
        f"Τους τελευταίους 12 μήνες ({cur.window.label()}) αποχώρησαν εθελοντικά **{cur.voluntary_leavers:,}** "
        f"εργαζόμενοι — turnover **{pct(cur.voluntary_rate)}** ({arrow} {abs(cur.voluntary_rate - prev.voluntary_rate) * 100:.1f} pp "
        f"έναντι των προηγούμενων 12 μηνών· στόχος ≤ {pct(EXEC_TARGET_VOLUNTARY_TURNOVER, 0)}).",
        cost_line,
    ]
    hot = units[units["Status"] == "Significant hotspot"]
    watch = units[units["Status"] == "Watch"]
    if not hot.empty:
        names = ", ".join(f"**{r[by]}** ({r['Voluntary leavers']:.0f} αποχωρήσεις έναντι ~{r['Expected leavers']:.0f} αναμενόμενων)"
                          for _, r in hot.head(3).iterrows())
        lines.append(f"Στατιστικά σημαντικά hotspots ({by}): {names} — κόστος {eur(hot['Cost (€)'].sum())}.")
    elif not watch.empty:
        lines.append(f"Δεν υπάρχουν στατιστικά σημαντικά hotspots ({by}). Υπό παρακολούθηση: "
                     + ", ".join(f"**{n}**" for n in watch[by].head(3)) + ".")
    else:
        lines.append(f"Καμία μονάδα ({by}) δεν ξεφεύγει στατιστικά από το συνολικό turnover.")
    if cur.early_leaver_share > EXEC_TARGET_EARLY_LEAVER_SHARE:
        lines.append(f"**{pct(cur.early_leaver_share, 0)}** των εθελοντικών αποχωρήσεων έγιναν μέσα στον πρώτο χρόνο "
                     f"(στόχος ≤ {pct(EXEC_TARGET_EARLY_LEAVER_SHARE, 0)}) — θέμα επιλογής / onboarding.")
    if projection:
        lines.append(f"Πρόβλεψη: ~**{projection['exits']:,.0f}** εθελοντικές αποχωρήσεις τους επόμενους 12 μήνες "
                     f"(εύρος {projection['lo']:,.0f}–{projection['hi']:,.0f}), κόστος ≈ **{eur(projection['cost'])}** "
                     f"(με το φετινό κόστος ανά αποχώρηση, {eur(projection['unit_cost'])} = σύνολο ÷ αποχωρήσεις).")
    if risk is not None:
        lines.append(f"ML ({ML_TARGET_COMPANY}): **{eur(risk['total'])}** αναμενόμενο κόστος σε κίνδυνο· "
                     f"το **{pct(risk['top10_share'], 0)}** συγκεντρώνεται στο 10% των εργαζομένων με το υψηλότερο ρίσκο "
                     "— στοχευμένα μέτρα διακράτησης είναι αποδοτικά.")
    return lines


# --------------------------------------------------------------------
# Board pack (downloadable HTML)
# --------------------------------------------------------------------
def _board_pack_html(scope: str, cur, prev, lines: list[str], figures: list[tuple[str, go.Figure]],
                     tables: list[tuple[str, pd.DataFrame]]) -> bytes:
    def md(s: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(s))
    kpi_rows = [
        ("Headcount", f"{cur.headcount_end:,}", f"{prev.headcount_end:,}"),
        ("Voluntary turnover (12m)", pct(cur.voluntary_rate), pct(prev.voluntary_rate)),
        ("Voluntary leavers (12m)", f"{cur.voluntary_leavers:,}", f"{prev.voluntary_leavers:,}"),
        ("Typical cost per leaver (median)", eur(cur.median_cost_per_leaver), eur(prev.median_cost_per_leaver)),
        ("Replacement cost (12m)", eur(cur.cost), eur(prev.cost)),
        ("… excl. director-level exits", eur(cur.cost_excl_exec), eur(prev.cost_excl_exec)),
        ("Director-level voluntary exits", f"{cur.exec_leavers}", f"{prev.exec_leavers}"),
        ("Leavers < 12 months tenure", pct(cur.early_leaver_share, 0), pct(prev.early_leaver_share, 0)),
    ]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Workforce Executive Summary</title><style>",
        "body{font-family:Segoe UI,Arial,sans-serif;color:#0b0b0b;background:#fcfcfb;max-width:1100px;margin:24px auto;padding:0 16px}",
        "h1{font-size:24px;margin-bottom:4px}h2{font-size:18px;margin-top:28px}.muted{color:#52514e;font-size:13px}",
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e4e0;padding:6px 8px;text-align:left}",
        "th{color:#52514e;font-weight:600}li{margin:6px 0;line-height:1.45}@media print{.page{page-break-before:always}}",
        "</style></head><body>",
        "<h1>Workforce Executive Summary</h1>",
        f"<div class='muted'>{html.escape(scope)} · {cur.window.label()} vs {prev.window.label()} · generated {date.today():%Y-%m-%d}</div>",
        "<h2>Key messages</h2><ul>" + "".join(f"<li>{md(l)}</li>" for l in lines) + "</ul>",
        "<h2>KPIs</h2><table><tr><th>KPI</th><th>Last 12 months</th><th>Previous 12 months</th></tr>"
        + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in kpi_rows) + "</table>",
    ]
    for i, (title, fig) in enumerate(figures):
        parts.append(f"<h2 class='page'>{html.escape(title)}</h2>")
        parts.append(pio.to_html(fig, include_plotlyjs="inline" if i == 0 else False, full_html=False,
                                 config={"displaylogo": False}))
    for title, t in tables:
        parts.append(f"<h2>{html.escape(title)}</h2>" + t.to_html(index=False, border=0, escape=True))
    parts.append("<p class='muted'>Replacement cost = monthly salary × salary payments per year (14 Greek "
                 "companies, 12 others) × rate by job level (assumptions to be confirmed by HR / Finance); "
                 "missing salaries filled with the company & job-level median. Hotspots: exact Poisson test vs the overall "
                 "rate (95% / 99.8% limits); units under 5 people are not classified.</p></body></html>")
    return "\n".join(parts).encode("utf-8")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def render_executive_tab(df_filtered: pd.DataFrame, uploaded_file, scope_label: str) -> None:
    st.markdown("### 🏛️ Executive Summary")
    rates = _render_assumptions()

    ts = build_monthly_time_series(df_filtered)
    if ts.empty or len(ts) < 24:
        st.info("Χρειάζονται τουλάχιστον 24 ολοκληρωμένοι μήνες δεδομένων για τη σύγκριση 12 μηνών.")
        return
    cur_w, prev_w = rolling_windows(ts["Date"].max())
    cur, prev = period_kpis(df_filtered, cur_w, rates), period_kpis(df_filtered, prev_w, rates)
    filled = (f" Μισθός συμπληρώθηκε (διάμεσος εταιρείας & επιπέδου) για {cur.salaries_filled} / "
              f"{prev.salaries_filled} αποχωρήσεις (τρέχον / προηγούμενο έτος)."
              if cur.salaries_filled or prev.salaries_filled else "")
    st.caption(f"Scope: {scope_label} · Τελευταίοι 12 ολοκληρωμένοι μήνες **{cur_w.label()}** "
               f"έναντι **{prev_w.label()}**. Μόνο εθελοντικές αποχωρήσεις.{filled}")

    # Forecast of voluntary exits (shared, cached backtest + model selection)
    projection = None
    vol_types = [t for t in df_filtered["Departure Type"].dropna().unique() if classify_voluntary(t) == "Voluntary"] \
        if "Departure Type" in df_filtered.columns else []
    if vol_types:
        ts_vol = build_monthly_time_series(df_filtered, vol_types)
        if len(ts_vol) >= 24:
            fc = forecast_best(ts_vol, "Departures", 12, DEFAULT_PROPHET_CHANGEPOINT_PRIOR,
                               DEFAULT_PROPHET_SEASONALITY_PRIOR)
            unit_cost = cur.avg_cost_per_leaver if not np.isnan(cur.avg_cost_per_leaver) else 0.0
            projection = {"exits": fc.total, "lo": fc.total_lower, "hi": fc.total_upper,
                          "cost": fc.total * unit_cost, "unit_cost": unit_cost, "model": fc.model_name}

    # € at risk from the ML model (its own population: target company, indefinite contracts)
    risk, risk_table = None, None
    try:
        from ui.tab_ml import get_ml_state
        from ml import align_columns
        prepared, result = get_ml_state(uploaded_file)
        proba = pd.Series(
            result.model.predict_proba(align_columns(prepared.X_current, result.train_columns))[:, 1],
            index=prepared.X_current.index,
        )
        risk_table = euro_at_risk(prepared.df_ui, proba, rates)
        top = risk_table["Probability"] >= risk_table["Probability"].quantile(0.9)
        risk = {"total": float(risk_table["€ at risk"].sum()),
                "top10_share": float(risk_table.loc[top, "€ at risk"].sum() / risk_table["€ at risk"].sum())}
    except Exception as e:   # noqa: BLE001 — the summary must render even if the ML part cannot
        st.caption(f"€ at risk μη διαθέσιμο ({type(e).__name__}).")

    unit_options = [c for c in ("Department", "Division", "Job Family", "JobTitle")
                    if c in df_filtered.columns or c == "Job Family"]
    by = st.session_state.get("exec_unit", unit_options[0])
    units, p0 = unit_breakdown(df_filtered, cur_w, by, rates)

    # ---- Key messages + KPIs ------------------------------------------
    lines = _narrative(cur, prev, units, by, projection, risk)
    with st.container(border=True):
        st.markdown("#### Key messages")
        st.markdown("\n".join(f"- {l}" for l in lines))
    _render_kpis(cur, prev, projection)

    # ---- Where the money goes ----------------------------------------
    st.markdown("#### 💶 Πού χάνονται τα χρήματα")
    th = theme()
    div_units, _ = unit_breakdown(df_filtered, cur_w, "Division", rates) if "Division" in df_filtered.columns \
        else (pd.DataFrame(), None)
    figures: list[tuple[str, go.Figure]] = []
    c1, c2 = st.columns(2)
    with c1:
        if not div_units.empty:
            fig_cost = _cost_bar(div_units, "Division", "Cost (€)",
                                 "Κόστος εθελοντικών αποχωρήσεων ανά division (12m)", th["series"][0])
            st.plotly_chart(fig_cost, width="stretch")
            figures.append(("Cost of voluntary attrition by division (last 12 months)", fig_cost))
    with c2:
        if risk_table is not None:
            by_div = risk_table.groupby("Division", dropna=True)["€ at risk"].sum().reset_index()
            fig_risk = _cost_bar(by_div, "Division", "€ at risk",
                                 f"€ at risk — επόμενοι 12m ({ML_TARGET_COMPANY}, ML)", th["series"][0])
            st.plotly_chart(fig_risk, width="stretch")
            st.caption(f"Πιθανότητα εθελοντικής αποχώρησης × κόστος αντικατάστασης, για τους ενεργούς "
                       f"εργαζόμενους του ML μοντέλου ({ML_TARGET_COMPANY}, {ML_WORK_RELATIONSHIP}).")
            figures.append((f"€ at risk next 12 months by division ({ML_TARGET_COMPANY})", fig_risk))

    st.markdown("**Κόστος ανά επίπεδο θέσης (12m)** — λίγες αποχωρήσεις υψηλού επιπέδου μπορεί να "
                "κοστίζουν όσο δεκάδες αποχωρήσεις πρώτης γραμμής.")
    st.dataframe(cur.cost_by_level, hide_index=True, width="stretch", column_config={
        "Leavers": st.column_config.NumberColumn(format="%d"),
        "Typical cost": st.column_config.NumberColumn("Typical cost per leaver (median)", format="€%.0f"),
        "Total cost": st.column_config.NumberColumn(format="€%.0f"),
        "Share of cost": st.column_config.NumberColumn(format="percent"),
    })

    # ---- Hotspots -------------------------------------------------------
    st.markdown("#### 🎯 Hotspots — πραγματικά προβλήματα ή θόρυβος;")
    st.selectbox("Ανάλυση ανά", unit_options, key="exec_unit")
    st.caption(
        "Κάθε σημείο είναι μια μονάδα. Οι καμπύλες δείχνουν πόσο turnover είναι αναμενόμενο από τύχη για το "
        "μέγεθός της· μόνο ό,τι βγαίνει πάνω από τις καμπύλες είναι πραγματικό hotspot. Μικρές μονάδες "
        f"(< {MIN_UNIT_HEADCOUNT} άτομα) δεν αξιολογούνται."
    )
    fig_funnel = _funnel_plot(units, p0, by)
    st.plotly_chart(fig_funnel, width="stretch")
    figures.append((f"Turnover hotspots by {by} (funnel plot)", fig_funnel))

    flagged = units[units["Status"].isin(["Significant hotspot", "Watch", "Better than average"])].copy()
    flagged_view = None
    if not flagged.empty:
        flagged["Status"] = flagged["Status"].map(lambda s: f"{STATUS_STYLE[s][1]} {s}")
        flagged_view = flagged[[by, "Status", "Avg headcount", "Voluntary leavers", "Expected leavers",
                                "Turnover rate", "Cost (€)"]].sort_values("Cost (€)", ascending=False)
        st.dataframe(
            flagged_view, hide_index=True, width="stretch",
            column_config={
                "Avg headcount": st.column_config.NumberColumn(format="%.0f"),
                "Voluntary leavers": st.column_config.NumberColumn(format="%d"),
                "Expected leavers": st.column_config.NumberColumn(format="%.1f"),
                "Turnover rate": st.column_config.NumberColumn(format="percent"),
                "Cost (€)": st.column_config.NumberColumn(format="€%.0f"),
            },
        )

    # ---- € at risk detail ----------------------------------------------
    risk_view = None
    if risk_table is not None:
        with st.expander("🔎 € at risk ανά job family (ML)", expanded=False):
            risk_view = (risk_table.groupby("Job Family")
                         .agg(Employees=("Probability", "size"), **{"Avg risk": ("Probability", "mean"),
                              "€ at risk": ("€ at risk", "sum")})
                         .sort_values("€ at risk", ascending=False).reset_index())
            st.dataframe(risk_view, hide_index=True, width="stretch", column_config={
                "Avg risk": st.column_config.NumberColumn(format="percent"),
                "€ at risk": st.column_config.NumberColumn(format="€%.0f"),
            })

    # ---- Board pack ----------------------------------------------------
    lvl_table = cur.cost_by_level.copy()
    lvl_table["Typical cost"] = lvl_table["Typical cost"].map(eur)
    lvl_table["Total cost"] = lvl_table["Total cost"].map(eur)
    lvl_table["Share of cost"] = lvl_table["Share of cost"].map(lambda v: pct(v, 0))
    tables = [("Cost of voluntary exits by job level (last 12 months)",
               lvl_table.rename(columns={"Typical cost": "Typical cost per leaver (median)"}))]
    if flagged_view is not None:
        t = flagged_view.copy()
        t["Turnover rate"] = t["Turnover rate"].map(lambda v: pct(v))
        t["Cost (€)"] = t["Cost (€)"].map(eur)
        t["Expected leavers"] = t["Expected leavers"].round(1)
        t["Avg headcount"] = t["Avg headcount"].round(0).astype(int)
        tables.append((f"Flagged units by {by}", t))
    if risk_view is not None:
        t = risk_view.copy()
        t["Avg risk"] = t["Avg risk"].map(lambda v: pct(v))
        t["€ at risk"] = t["€ at risk"].map(eur)
        tables.append((f"€ at risk by job family ({ML_TARGET_COMPANY})", t))
    st.download_button(
        "⬇️ Download board pack (HTML — ανοίγει σε browser, εκτυπώνεται σε PDF)",
        data=_board_pack_html(scope_label, cur, prev, lines, figures, tables),
        file_name=f"executive_summary_{cur_w.end:%Y%m}.html", mime="text/html",
    )
