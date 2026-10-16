"""New Hire What-If Simulator: predict attrition probability for a synthetic profile."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from config.constants import RISK_LOW_MAX, RISK_MEDIUM_MAX
from ml.job_families import job_family, job_level


# --------------------------------------------------------------------
# Range derivation
# --------------------------------------------------------------------
@dataclass
class _NumRange:
    min: int
    max: int
    median: int


def _derive_salary_range(df: pd.DataFrame) -> _NumRange:
    """Return (min, max, median) for the salary slider, with sensible global bounds."""
    if "Nominal Salary" not in df.columns:
        return _NumRange(800, 4000, 1500)

    s = (
        df["Nominal Salary"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )
    s = s.where(s >= 90, s * 26)  # daily/weekly fix
    s = s[(s.notna()) & (s > 0)]
    if s.empty:
        return _NumRange(800, 4000, 1500)
    return _NumRange(
        min=max(600, int(s.quantile(0.05))),
        max=max(4000, int(s.quantile(0.95))),
        median=int(s.median()),
    )


def _derive_grade_choices(df: pd.DataFrame) -> list[float]:
    if "Grade" not in df.columns:
        return [0.1, 1, 2, 3, 4, 5]
    g = pd.to_numeric(df["Grade"], errors="coerce")
    valid = sorted(g[(g.notna()) & (g > 0)].unique().tolist())
    return valid or [0.1, 1, 2, 3, 4, 5]


def _safe_unique(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())


# --------------------------------------------------------------------
# Feature row builder
# --------------------------------------------------------------------
def _build_feature_row(
    train_columns: list[str],
    *,
    salary: float,
    grade: float,
    job_prop: str,
    division: str,
    department: str,
    job_pos: str,
    city: str,
    age: int | None = None,
    tax_cat: int | None = None,
    gender_value: float | None = None,
    marital: str = "",
    tenure_years: float = 0.0,
    peer_median_salary: float | None = None,
    education: str = "",
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the training feature schema.

    For a *new hire* simulator, Tenure defaults to 0 (just hired). For modelling
    an existing employee profile, pass `tenure_years`. Age / tax category /
    gender / marital status are only set if the model was trained with them
    (the current model is not). Job family and level are derived from `job_pos`.

    `peer_median_salary` is used to compute SalaryToGradeRatio. When None, the
    ratio is left as NaN — XGBoost handles it natively.
    """
    from ml.features import _ascii_safe

    x = pd.DataFrame(0.0, index=[0], columns=train_columns)

    def set_num(name: str, value: float | None) -> None:
        # Model columns are ASCII-sanitised ("Log Nominal Salary" -> "Log_Nominal_Salary")
        if value is None or pd.isna(value):
            return
        for col in (_ascii_safe(name), name):
            if col in x.columns:
                x.at[0, col] = float(value)
                return

    # --- Numeric features ---
    set_num("Age", age)
    set_num("Tax Category", tax_cat)
    set_num("Log Nominal Salary", np.log1p(salary))
    set_num("Grade", grade)
    set_num("Gender", gender_value)
    set_num("Job Property", 1.0 if job_prop == "ADMINISTRATIVE" else 0.0)

    # --- Engineered features ---
    set_num("Tenure", tenure_years)
    set_num("LogTenure", np.log1p(max(tenure_years, 0)))
    if age is not None:
        set_num("AgeAtHire", float(age) - float(tenure_years))
    if job_pos and job_pos != "(none)":
        set_num("Job Level", job_level(job_pos))
    if peer_median_salary and peer_median_salary > 0:
        set_num("SalaryToGradeRatio", float(salary) / float(peer_median_salary))

    def set_dummy(prefix: str, value: str) -> None:
        if not value or value == "(none)":
            return
        # The encoder ASCII-sanitises column names, so map both the prefix and value
        col_raw = f"{prefix}_{value}"
        col_safe = _ascii_safe(col_raw)
        if col_safe in x.columns:
            x.at[0, col_safe] = 1.0
        elif col_raw in x.columns:   # backward-compat for non-sanitised columns
            x.at[0, col_raw] = 1.0

    set_dummy("Division", division)
    set_dummy("Department", department)
    set_dummy("Job Position", job_pos)
    if job_pos and job_pos != "(none)":
        set_dummy("Job Family", job_family(job_pos))
    set_dummy("City", city)
    set_dummy("Marital Status", marital)
    set_dummy("Education Level", education)

    return x.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _risk_label(prob: float) -> tuple[str, str, str]:
    if prob < RISK_LOW_MAX:
        return "Low risk", "🟢", "Το μοντέλο εκτιμά χαμηλή πιθανότητα αποχώρησης για αυτό το προφίλ."
    if prob < RISK_MEDIUM_MAX:
        return "Medium risk", "🟡", "Μέτριο ρίσκο – αξίζει να εξεταστούν προσαρμογές σε μισθό/θέση/grade."
    return "High risk", "🔴", "Υψηλό ρίσκο αποχώρησης – χρειάζεται ιδιαίτερη προσοχή στον σχεδιασμό."


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def render_new_hire_simulator(
    df_ui: pd.DataFrame,
    model,
    train_columns: list[str],
) -> None:
    st.markdown("## 🧪 New Hire What-If Simulator")
    st.caption(
        "Δημιούργησε ένα προφίλ νέου εργαζόμενου και δες την πιθανότητα αποχώρησης "
        "σύμφωνα με το ML μοντέλο."
    )

    if model is None or not train_columns:
        st.info("Πρώτα εκπαίδευσε το μοντέλο ML και μετά δοκίμασε τον simulator.")
        return

    # Derive ranges
    sal = _derive_salary_range(df_ui)
    grades = _derive_grade_choices(df_ui)

    divisions = _safe_unique(df_ui, "Division")
    departments = _safe_unique(df_ui, "Department")
    job_positions = _safe_unique(df_ui, "Job Position")
    cities = _safe_unique(df_ui, "City")

    # Form
    with st.form("new_hire_sim_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            salary = st.slider("Ονομαστικός μισθός (€)", sal.min, sal.max, sal.median, 50)
            tenure_years = st.slider(
                "Προϋπηρεσία στην εταιρεία (έτη)", 0.0, 30.0, 0.0, 0.5,
                help="0 = νέα πρόσληψη. Αύξησέ το για να δεις ένα υπάρχον προφίλ.",
            )

        with col2:
            grade_default_idx = grades.index(0.1) if 0.1 in grades else 0
            grade = st.selectbox("Grade", options=grades, index=grade_default_idx)
            division = st.selectbox("Division", options=divisions or ["(none)"])
            department = st.selectbox("Department", options=departments or ["(none)"])

        with col3:
            job_pos = st.selectbox("Job Position", options=job_positions or ["(none)"])
            city = st.selectbox("Πόλη (City)", options=cities or ["(none)"])
            job_prop = st.selectbox(
                "Job Property", options=["OPERATIONAL", "ADMINISTRATIVE"], index=0
            )

        submitted = st.form_submit_button("🔮 Υπολόγισε Attrition Risk")

    if not submitted:
        return

    # Build row & predict
    # Compute peer median salary for the chosen grade so SalaryToGradeRatio
    # is meaningful in the prediction (instead of NaN/0).
    peer_median_salary: float | None = None
    if "Grade" in df_ui.columns and "Nominal Salary" in df_ui.columns:
        peers = df_ui[df_ui["Grade"].round(0) == round(grade)]["Nominal Salary"]
        if len(peers) >= 5:
            peer_median_salary = float(peers.median())

    x_new = _build_feature_row(
        train_columns,
        salary=salary,
        grade=grade,
        job_prop=job_prop,
        division=division,
        department=department,
        job_pos=job_pos,
        city=city,
        tenure_years=tenure_years,
        peer_median_salary=peer_median_salary,
    )

    prob = float(model.predict_proba(x_new)[0, 1])
    label, emoji, comment = _risk_label(prob)

    st.markdown("---")
    st.markdown("### Αποτέλεσμα Πρόβλεψης για το Προφίλ Νέου Εργαζόμενου")
    st.metric(
        label="Voluntary Attrition Probability (12m)",
        value=f"{prob*100:.1f}%",
        help="Πιθανότητα εθελοντικής αποχώρησης μέσα στους επόμενους 12 μήνες.",
    )
    st.write(f"{emoji} **Risk level:** {label}")
    st.write(comment)

    st.markdown(
        f"""
        **Προφίλ που αξιολογήθηκε:**
        - Μισθός: **{salary:,.0f} €**
        - Προϋπηρεσία: **{tenure_years:g} έτη**
        - Grade: **{grade}** (μόνο για τη σύγκριση μισθού με το grade)
        - Job Property: **{job_prop}**
        - Division: **{division}**
        - Department: **{department}**
        - Job Position: **{job_pos}**
        - City: **{city}**
        - Job family / level: **{job_family(job_pos)} / {job_level(job_pos)}**

        Φύλο, ηλικία, οικογενειακή κατάσταση και φορολογική κατηγορία δεν χρησιμοποιούνται από το μοντέλο.
        """
    )
