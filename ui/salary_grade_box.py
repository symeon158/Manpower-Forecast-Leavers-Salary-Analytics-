"""Salary & Grade tab: box plot of nominal salary by grade for ACTIVE employees.

Design notes
------------
- One company at a time: salary levels (and possibly currencies) differ a lot
  between subsidiaries, so a cross-company grade box plot would mislead.
- Grade 0.1 (raw "0,1") marks ungraded staff; excluded by default.
- Grades with fewer than `min_n` people are hidden — a box of 2 points is noise.
- Colors: validated categorical slots 1–2 (blue / orange), stepped for dark mode.
  The pay-range overlay is drawn in secondary ink so it reads as a reference,
  not as a third data series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loaders.pay_ranges_loader import load_pay_ranges_df
from ui.viz_theme import hex_to_rgba as _hex_to_rgba, theme as _theme

UNGRADED_VALUE = 0.1
PAY_RANGE_COMPANY = "ΑΛΟΥΜΥΛ Α.Ε."      # pay ranges in config/pay_ranges_data.py are for this company

def _grade_label(g: float) -> str:
    return "Ungraded" if g == UNGRADED_VALUE else f"{g:g}"


def active_employees(df: pd.DataFrame) -> pd.DataFrame:
    """Employees still active at the data's as-of date (latest departure date in the file)."""
    if "DepartureDate" not in df.columns:
        return df.copy()
    as_of = df["DepartureDate"].max()
    active = df["DepartureDate"].isna() | (df["DepartureDate"] > as_of)
    return df[active].copy()


def _grade_stats(d: pd.DataFrame, pay_mid: pd.Series | None) -> pd.DataFrame:
    g = d.groupby("GRADE_clean")["NominalSalary"]
    stats = pd.DataFrame({
        "Employees": g.size(),
        "P10": g.quantile(0.10),
        "Q1": g.quantile(0.25),
        "Median": g.median(),
        "Q3": g.quantile(0.75),
        "P90": g.quantile(0.90),
        "Mean": g.mean(),
    })
    stats["IQR"] = stats["Q3"] - stats["Q1"]
    stats["Spread P90/P10"] = stats["P90"] / stats["P10"].replace(0, np.nan)
    graded = stats.index != UNGRADED_VALUE
    stats.loc[graded, "Median step vs prev. grade (%)"] = stats.loc[graded, "Median"].pct_change() * 100
    if pay_mid is not None:
        stats["Median / pay-range P50"] = stats["Median"] / stats.index.map(pay_mid)
    stats.index = [_grade_label(x) for x in stats.index]
    stats.index.name = "Grade"
    return stats.reset_index()


def render_salary_grade_boxplot(df_filtered: pd.DataFrame, sidebar_company: str) -> None:
    st.markdown("### 📦 Nominal Salary by Grade — active employees")

    needed = {"GRADE_clean", "NominalSalary", "Company"}
    if df_filtered.empty or not needed.issubset(df_filtered.columns):
        st.info("Χρειάζονται στήλες Grade, NominalSalary και Company.")
        return

    act = active_employees(df_filtered)
    act = act[act["NominalSalary"].notna() & (act["NominalSalary"] > 0) & act["GRADE_clean"].notna()]
    if act.empty:
        st.info("Δεν υπάρχουν ενεργοί εργαζόμενοι με Grade και μισθό στα επιλεγμένα φίλτρα.")
        return

    # ---- Controls (one row) -------------------------------------------
    companies = act["Company"].value_counts().index.tolist()
    default_company = sidebar_company if sidebar_company in companies else companies[0]
    c1, c2, c3, c4 = st.columns([2.2, 1.3, 1.5, 1.2])
    with c1:
        company = st.selectbox(
            "Εταιρεία", companies, index=companies.index(default_company), key="sgb_company",
            help="Ένα chart ανά εταιρεία: τα επίπεδα μισθών (και πιθανόν το νόμισμα) διαφέρουν ανά χώρα.",
        )
    with c2:
        split = st.selectbox("Split by", ["None", "Job Property"], key="sgb_split")
    with c3:
        points_mode = st.selectbox(
            "Points", ["Outliers only", "All employees", "None"], key="sgb_points",
        )
    with c4:
        min_n = st.number_input("Min employees / grade", 1, 50, 5, key="sgb_min_n")

    c5, c6, c7, c8 = st.columns([1.3, 1.1, 1.1, 2.2])
    with c5:
        include_ungraded = st.checkbox("Include ungraded (0.1)", value=False, key="sgb_ungraded")
    with c6:
        show_sd = st.checkbox("Mean ± SD", value=False, key="sgb_sd")
    with c7:
        log_y = st.checkbox("Log scale", value=False, key="sgb_log")
        full_range = st.checkbox("Full range", value=False, key="sgb_full",
                                 help="Default: ο άξονας χωράει τα whiskers· ακραίες τιμές πάνω από αυτό σημειώνονται.")
    pay_zone = None
    with c8:
        if company == PAY_RANGE_COMPANY:
            zones = sorted(load_pay_ranges_df()["pay_zone"].unique())
            choice = st.selectbox("Pay range overlay", ["None"] + zones, key="sgb_zone",
                                  help="P25 – P50 – P75 του pay zone, με την υπόθεση Grade = Ref Level.")
            pay_zone = None if choice == "None" else choice

    d = act[act["Company"] == company]
    n_ungraded = int((d["GRADE_clean"] == UNGRADED_VALUE).sum())
    if not include_ungraded:
        d = d[d["GRADE_clean"] != UNGRADED_VALUE]
    counts = d["GRADE_clean"].value_counts()
    small = sorted(counts[counts < min_n].index)
    d = d[d["GRADE_clean"].isin(counts[counts >= min_n].index)]
    if d.empty:
        st.info("Κανένα grade δεν έχει αρκετούς εργαζόμενους με τα τρέχοντα φίλτρα.")
        return

    grades = sorted(d["GRADE_clean"].unique())
    labels = [_grade_label(g) for g in grades]
    n_by_grade = d["GRADE_clean"].value_counts()
    ticktext = [f"{lbl}<br><sub>n={n_by_grade[g]}</sub>" for g, lbl in zip(grades, labels)]
    d = d.assign(_grade=d["GRADE_clean"].map(_grade_label))

    # Pay ranges for the overlay / table (grade ↔ ref_level)
    pay = None
    if pay_zone:
        pr = load_pay_ranges_df()
        pr = pr[(pr["pay_zone"] == pay_zone) & pr["ref_level_num"].isin(grades)]
        pay = pr.groupby("ref_level_num")[["p25", "p50", "p75"]].mean()

    # ---- Figure --------------------------------------------------------
    theme = _theme()
    boxpoints = {"Outliers only": "outliers", "All employees": "all", "None": False}[points_mode]
    groups = (
        [(None, d)] if split == "None" or "Job Property" not in d.columns
        else [(k, g) for k, g in d.groupby(d["Job Property"].fillna("Unknown"))][:2]
    )
    hidden_cells = []
    if len(groups) > 1:
        # Same minimum per (grade, group) cell — a box of 1–2 people is noise
        kept = []
        for name, g in groups:
            sizes = g["GRADE_clean"].value_counts()
            hidden_cells += [f"{_grade_label(x)}/{name}" for x in sorted(sizes[sizes < min_n].index)]
            kept.append((name, g[g["GRADE_clean"].isin(sizes[sizes >= min_n].index)]))
        groups = kept
    fig = go.Figure()
    for i, (name, g) in enumerate(groups):
        color = theme["series"][i]
        hover_text = (
            g["JobTitle"].fillna("—").astype(str) + " · " + g.get("Department", pd.Series("", index=g.index)).fillna("").astype(str)
            if "JobTitle" in g.columns else None
        )
        fig.add_trace(go.Box(
            x=g["_grade"], y=g["NominalSalary"], name=name or "Active employees",
            marker=dict(color=color, size=5, opacity=0.55 if boxpoints == "all" else 0.9,
                        line=dict(width=0)),
            line=dict(color=color, width=1.6),
            fillcolor=_hex_to_rgba(color, 0.22),
            boxmean="sd" if show_sd else True,
            boxpoints=boxpoints, jitter=0.45 if boxpoints == "all" else 0, pointpos=0,
            text=hover_text, hoverlabel=dict(namelength=-1),
            offsetgroup=str(i), whiskerwidth=0.4,
        ))

    if pay is not None and not pay.empty:
        # Reference band behind each box: P25–P75 rectangle + dashed P50 line (secondary ink)
        ref = theme["ref"]
        for g_val, row in pay.iterrows():
            i = labels.index(_grade_label(g_val))
            fig.add_shape(type="rect", xref="x", yref="y", x0=i - 0.46, x1=i + 0.46,
                          y0=row["p25"], y1=row["p75"], fillcolor=_hex_to_rgba(ref, 0.13),
                          line=dict(width=0), layer="below")
            fig.add_shape(type="line", xref="x", yref="y", x0=i - 0.46, x1=i + 0.46,
                          y0=row["p50"], y1=row["p50"], line=dict(color=ref, width=2, dash="dash"),
                          layer="below")
        # Invisible-but-hoverable markers + legend entry for the band
        fig.add_trace(go.Scatter(
            x=[_grade_label(g) for g in pay.index], y=pay["p50"], mode="markers",
            name=f"Pay range {pay_zone}: P25–P75 band, P50 dashed",
            marker=dict(symbol="square", size=14, color=_hex_to_rgba(ref, 0.35)),
            customdata=np.column_stack([pay["p25"], pay["p75"]]),
            hovertemplate="Grade %{x} · pay range<br>P25 €%{customdata[0]:,.0f}<br>P50 €%{y:,.0f}"
                          "<br>P75 €%{customdata[1]:,.0f}<extra></extra>",
            opacity=0.0, showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=f"Pay range {pay_zone}: P25–P75 band, P50 dashed",
            marker=dict(symbol="square", size=12, color=_hex_to_rgba(ref, 0.35)), hoverinfo="skip",
        ))

    fig.update_layout(
        boxmode="group" if len(groups) > 1 else "overlay",
        boxgap=0.35, boxgroupgap=0.12,
        height=520, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        showlegend=len(groups) > 1 or pay is not None,
        hovermode="closest",
    )
    fig.update_xaxes(
        title_text="Grade", type="category", categoryorder="array", categoryarray=labels,
        tickmode="array", tickvals=labels, ticktext=ticktext, showgrid=False,
    )
    fig.update_yaxes(
        title_text="Nominal salary (€ / month)", tickprefix="€", tickformat=",.0f",
        type="log" if log_y else "linear", gridcolor=theme["grid"], zeroline=False,
    )
    if not full_range and not log_y:
        # Fit the whiskers (Q1−1.5·IQR … Q3+1.5·IQR, capped at the data), not single extreme salaries
        q = d.groupby("GRADE_clean")["NominalSalary"].quantile([0.25, 0.75]).unstack()
        iqr = q[0.75] - q[0.25]
        top = float(np.minimum(q[0.75] + 1.5 * iqr, d.groupby("GRADE_clean")["NominalSalary"].max()).max())
        bottom = float(np.maximum(q[0.25] - 1.5 * iqr, d.groupby("GRADE_clean")["NominalSalary"].min()).min())
        if pay is not None and not pay.empty:
            top, bottom = max(top, pay["p75"].max()), min(bottom, pay["p25"].min())
        pad = (top - bottom) * 0.06
        fig.update_yaxes(range=[max(0, bottom - pad), top + pad])
        n_above = int((d["NominalSalary"] > top + pad).sum())
        if n_above:
            fig.add_annotation(
                xref="paper", yref="paper", x=1, y=1.0, xanchor="right", yanchor="bottom",
                text=f"▲ {n_above} τιμή(ές) πάνω από το εύρος — δες 'Full range'",
                showarrow=False, font=dict(size=11, color=theme["ref"]),
            )
    st.plotly_chart(fig, width="stretch")

    notes = [f"**{company}** · {len(d):,} ενεργοί εργαζόμενοι · {len(grades)} grades."]
    if n_ungraded and not include_ungraded:
        notes.append(f"{n_ungraded:,} ungraded (0.1) εξαιρούνται.")
    if small:
        notes.append(f"Grades με < {min_n} άτομα κρύβονται: {', '.join(_grade_label(g) for g in small)}.")
    if hidden_cells:
        notes.append(f"Ομάδες με < {min_n} άτομα κρύβονται: {', '.join(hidden_cells)}.")
    if pay is not None:
        notes.append("Pay range overlay: υπόθεση ότι Grade = Ref Level του pay-range πίνακα.")
    st.caption(" ".join(notes) + " Box = Q1–Q3, γραμμή = διάμεσος, διακεκομμένη = μέσος όρος, "
               "whiskers = 1.5×IQR.")

    # ---- Statistics table (accessible alternative to the chart) --------
    stats = _grade_stats(d, pay["p50"] if pay is not None else None)
    inversions = stats["Median step vs prev. grade (%)"].lt(0).sum()
    with st.expander(
        f"📋 Στατιστικά ανά grade" + (f" — ⚠️ {inversions} grade(s) με χαμηλότερο διάμεσο από το προηγούμενο"
                                      if inversions else ""),
        expanded=False,
    ):
        fmt = {c: "€{:,.0f}" for c in ("P10", "Q1", "Median", "Q3", "P90", "Mean", "IQR")}
        fmt.update({"Spread P90/P10": "{:.2f}×", "Median step vs prev. grade (%)": "{:+.1f}%",
                    "Median / pay-range P50": "{:.2f}"})
        st.dataframe(
            stats.style.format(fmt, na_rep="—").apply(
                lambda col: ["color: #e34948; font-weight: 600" if v < 0 else "" for v in col],
                subset=["Median step vs prev. grade (%)"],
            ),
            width="stretch", hide_index=True,
        )
        st.download_button(
            "⬇️ Download grade statistics (CSV)",
            data=stats.round(2).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"salary_by_grade_{company}.csv", mime="text/csv",
        )
