"""
ML data preparation — point-in-time ("snapshot") design.

Question the model answers
--------------------------
"Given what we knew about an employee on date T, will they leave VOLUNTARILY
within the next `ML_HORIZON_MONTHS` months?"

Why snapshots instead of one row per employee
---------------------------------------------
The HR export is a single cross-section: leavers carry their values *as of the
day they left*, active employees carry *today's* values. Training directly on
that (the previous design) leaks the departure date into every feature:

- **Salary** is frozen at departure, so a 2019 leaver looks "underpaid" next to
  a 2026 active employee purely because of 7 years of wage growth.
- **Age** is today's age for everyone, so leavers look older than they were.
- **Tenure** is time-to-departure for leavers but time-to-today for actives,
  i.e. the label is partly encoded in the feature.
- Active employees were the negative class *and* the scoring set, so their
  scores were in-sample.

The snapshot design fixes this:

1. Pick snapshot dates T every `ML_SNAPSHOT_STEP_MONTHS` months, from
   `ML_DEPARTURE_DATE_CUTOFF` up to the export date. Recent snapshots have less
   than `horizon` of follow-up; the survival model (ml/survival.py) uses them as
   censored observations.
2. At each T, take everyone active on T and compute Age / Tenure *as of T*.
3. Label `y` = 1 if they left voluntarily in (T, T + horizon], 0 if they were
   still there. It is NaN when the window is not fully observed yet or the
   person left for another reason inside it (competing outcome, not "stayed").
   `y` is only used for evaluation; training uses exit dates in `panel_meta`.
4. Salaries are restated in today's money with a wage index estimated from the
   data (see `_salary_growth_factor`), so old observations are comparable.
5. Score current employees with features as of the export date. They are
   never training labels at that date, so the scores are genuinely out-of-sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config.constants import (
    LOW_SALARY_MULTIPLIER,
    LOW_SALARY_THRESHOLD,
    ML_COLUMN_RENAME_MAP,
    ML_DEPARTURE_DATE_CUTOFF,
    ML_HORIZON_MONTHS,
    ML_REPLACE_DEPARTMENT_PATTERN,
    ML_REPLACE_REASON,
    ML_SNAPSHOT_DATE,
    ML_SNAPSHOT_STEP_MONTHS,
    ML_TARGET_COMPANY,
    ML_TARGET_DEPARTURE_REASON,
    ML_WORK_RELATIONSHIP,
)
from data_loaders.leavers_loader import robust_read_csv
from ml.features import FeatureEncoder, fit_encoder
from ml.job_families import add_job_features

META_BASE_COLS = [
    "Registry Number",
    "First Name",
    "Last Name",
    "Division",
    "Department",
    "Job Position",
]

# Categorical columns kept for one-hot encoding
#
# Gender, age, marital status and tax category (which reflects family
# situation) are deliberately excluded: a tool that flags people for HR action
# should not score them on protected characteristics, and in backtests removing
# them cost no accuracy. They are kept in `panel_meta` for fairness checks.
CATEGORICAL_COLS = [
    "City",
    "Division",
    "Job Position",
    "Job Family",       # function grouping of Job Position (ml/job_families.py)
    "Department",
    "Education Level",
]

# Numeric columns kept as-is (NaN passes through to XGBoost natively).
# Age / Tenure / salary features are all computed AS OF the snapshot date.
#
# `Grade` itself is deliberately NOT a feature: the export holds each person's
# latest grade, so leavers keep the grade they left with while stayers carry
# later re-gradings. In backtests it inflated older-snapshot scores and lowered
# out-of-time AUC (0.69 → 0.77 when removed). It is still used for the
# within-grade salary ratio. A department-turnover feature was also tested
# and dropped (no out-of-time gain).
NUMERIC_COLS = [
    "Log Nominal Salary",          # log of salary restated in today's money
    "Tenure",
    "SalaryToGradeRatio",          # vs same-grade peers active on the same date
    "Job Property",
    "Job Level",                   # 1 = entry … 5 = executive, from the job title
    "Promoted",
]


@dataclass
class MLPreparedData:
    """Container for everything the ML pipeline needs."""
    X: pd.DataFrame                     # encoded features, one row per (employee, snapshot)
    y: pd.Series                        # 1/0 = left voluntarily within the horizon; NaN = not fully observed
    snapshots: pd.Series                # snapshot date of each row in X
    panel_meta: pd.DataFrame            # raw columns aligned to X: segments, Gender/Age, exit date + reason
    X_current: pd.DataFrame             # encoded features of currently active employees
    registry_numbers: pd.Series         # aligned to X_current.index
    meta_cols: pd.DataFrame             # aligned to X_current.index
    df_ui: pd.DataFrame                 # raw cleaned, used in What-If simulator
    encoder: FeatureEncoder             # fitted feature encoder
    snapshot_date: pd.Timestamp         # export date: features of current employees are as of this date
    horizon_months: int
    salary_index: pd.Series             # year -> factor applied to restate salaries in today's money


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _parse_dates(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce", dayfirst=True)


def _clean_bracketed_numeric(s: pd.Series) -> pd.Series:
    """Strip [, ], replace , with ., to numeric."""
    return pd.to_numeric(
        s.astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _resolve_snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """Pick the export date (the moment the data was 'frozen').

    Priority:
      1. ML_SNAPSHOT_DATE constant if set
      2. Latest known Departure Date in the dataset
      3. Today as last resort
    """
    if ML_SNAPSHOT_DATE:
        return pd.Timestamp(ML_SNAPSHOT_DATE)
    if "Departure Date" in df.columns:
        max_dep = df["Departure Date"].dropna().max()
        if pd.notna(max_dep):
            return pd.Timestamp(max_dep).normalize()
    return pd.Timestamp(datetime.today().date())


def _apply_business_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Target company, indefinite contract, post-cutoff.

    Unlike the old design, non-voluntary leavers are KEPT: they were active
    (and correctly labelled 0) in every snapshot before the window in which
    they left. Snapshot labelling drops them only for that final window.
    """
    if "Department" in df.columns and "Departure Reason Description" in df.columns:
        mask = df["Department"].astype(str).str.contains(
            ML_REPLACE_DEPARTMENT_PATTERN, na=False
        )
        df.loc[mask, "Departure Reason Description"] = ML_REPLACE_REASON

    if "Company" in df.columns:
        df = df[df["Company"] == ML_TARGET_COMPANY]

    df = df[df["Work Relationship"] == ML_WORK_RELATIONSHIP]
    if "Departure Date" in df.columns:
        df = df[
            (df["Departure Date"] > ML_DEPARTURE_DATE_CUTOFF) | df["Departure Date"].isna()
        ]
    return df


def _engineer_salary(df: pd.DataFrame) -> pd.DataFrame:
    df["Nominal Salary"] = _clean_bracketed_numeric(df["Nominal Salary"])
    df = df[df["Nominal Salary"] > 0].copy()
    # Daily/weekly fix: low values × multiplier
    low_mask = df["Nominal Salary"] < LOW_SALARY_THRESHOLD
    df.loc[low_mask, "Nominal Salary"] = df.loc[low_mask, "Nominal Salary"] * LOW_SALARY_MULTIPLIER
    return df


def _engineer_grade(df: pd.DataFrame) -> pd.DataFrame:
    if "Grade" in df.columns:
        df["Grade"] = _clean_bracketed_numeric(df["Grade"].replace({"99999": "0.1"}))
    return df


def _convert_misc_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if "Gender" in df.columns:
        df["Gender"] = pd.to_numeric(df["Gender"], errors="coerce")
    if "Promoted" in df.columns:
        df["Promoted"] = pd.to_numeric(df["Promoted"], errors="coerce")
    if "Job Property" in df.columns:
        df["Job Property Label"] = df["Job Property"]
        df["Job Property"] = df["Job Property"].map(
            {"ADMINISTRATIVE": 1, "OPERATIONAL": 0}
        )
    if "Tax Category" in df.columns:
        df["Tax Category"] = _clean_bracketed_numeric(df["Tax Category"])
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df


def _salary_growth_factor(df: pd.DataFrame, export: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """Restate every salary in export-year money.

    A leaver's salary is the one they had when they left; an active employee's
    is today's. We estimate a yearly wage index from *leavers only* (a
    population observed consistently across years): the median log-salary
    residual after removing grade effects, per departure year. Each row's
    salary is then multiplied by index[export_year] / index[observation_year].

    Returns (per-row factor, per-year factor table for display).
    """
    obs_year = df["Departure Date"].dt.year.fillna(export.year).clip(upper=export.year).astype(int)
    leavers = df["Departure Date"].notna()
    log_sal = np.log(df["Nominal Salary"])

    if "Grade" in df.columns:
        grade_med = log_sal[leavers].groupby(df.loc[leavers, "Grade"]).median()
        baseline = df["Grade"].map(grade_med).fillna(log_sal[leavers].median())
    else:
        baseline = pd.Series(log_sal[leavers].median(), index=df.index)
    resid = (log_sal - baseline)[leavers]

    by_year = resid.groupby(obs_year[leavers]).agg(["median", "size"])
    by_year = by_year[by_year["size"] >= 15]["median"]
    if by_year.empty or export.year not in by_year.index:
        # Not enough leaver history to estimate wage growth — leave salaries as-is
        factors = pd.Series(1.0, index=sorted(obs_year.unique()))
        return obs_year.map(factors).astype(float), factors

    years = pd.Index(range(int(obs_year.min()), export.year + 1))
    idx = by_year.reindex(years).interpolate(limit_direction="both")
    # Wage index shouldn't fall over time; enforce monotone non-decreasing
    idx = idx.cummax()
    factors = np.exp(idx.loc[export.year] - idx).clip(1.0, 2.0)
    return obs_year.map(factors).astype(float), factors


def _snapshot_dates(export: pd.Timestamp, step: int) -> list[pd.Timestamp]:
    """Snapshots every `step` months, anchored on the export date, back to the cutoff."""
    start = pd.Timestamp(ML_DEPARTURE_DATE_CUTOFF) + pd.Timedelta(days=1)
    out = []
    k = 1
    while (t := export - pd.DateOffset(months=k * step)) >= start:
        out.append(t)
        k += 1
    return sorted(out)


def _features_at(df: pd.DataFrame, t: pd.Timestamp, export: pd.Timestamp) -> pd.DataFrame:
    """Raw feature frame for everyone active on date `t`, with time-varying features as of `t`."""
    active = (df["Hire Date"] <= t) & (df["Departure Date"].isna() | (df["Departure Date"] > t))
    out = df.loc[active].copy()

    out["Tenure"] = (t - out["Hire Date"]).dt.days / 365.25
    age_from_birth = (
        (t - out["Birth Date"]).dt.days / 365.25 if "Birth Date" in out.columns else np.nan
    )
    # 'Age' in the export is today's age for everyone; shift it back as a fallback
    age_shifted = out["Age"] - (export - t).days / 365.25 if "Age" in out.columns else np.nan
    out["Age"] = pd.Series(age_from_birth, index=out.index).fillna(
        pd.Series(age_shifted, index=out.index)
    )

    out["Log Nominal Salary"] = np.log1p(out["Salary (today)"])
    if "Grade" in out.columns:
        peer_median = out.groupby("Grade")["Salary (today)"].transform("median")
        out["SalaryToGradeRatio"] = (out["Salary (today)"] / peer_median).replace(
            [np.inf, -np.inf], np.nan
        )

    out["Snapshot"] = t
    return out


def _label(rows: pd.DataFrame, t: pd.Timestamp, horizon: int, export: pd.Timestamp) -> pd.Series:
    """1 = voluntary exit in (t, t+horizon]; 0 = still there; NaN = other exit / not yet observable."""
    end = t + pd.DateOffset(months=horizon)
    if end > export:
        return pd.Series(np.nan, index=rows.index)
    left_in_window = rows["Departure Date"].notna() & (rows["Departure Date"] <= end)
    voluntary = rows["Departure Reason Description"] == ML_TARGET_DEPARTURE_REASON
    y = pd.Series(0.0, index=rows.index)
    y[left_in_window & voluntary] = 1.0
    y[left_in_window & ~voluntary] = np.nan
    return y


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def prepare_ml_data(
    uploaded_file: Any,
    horizon_months: int = ML_HORIZON_MONTHS,
    step_months: int = ML_SNAPSHOT_STEP_MONTHS,
) -> MLPreparedData:
    """End-to-end ML preparation pipeline (snapshot panel + current scoring set)."""
    df = robust_read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=ML_COLUMN_RENAME_MAP)

    if "Registry Number" not in df.columns:
        raise ValueError("Δεν βρέθηκε στήλη 'Αριθμός μητρώου' (Registry Number).")
    if "Work Relationship" not in df.columns:
        raise ValueError(
            "Δεν βρέθηκε στήλη 'Σχέση Εργασίας'. Δεν μπορεί να εφαρμοστεί ML attrition μοντέλο."
        )
    if "Departure Date" not in df.columns:
        raise ValueError("Δεν βρέθηκε στήλη 'Departure Date' στο αρχείο.")
    if "Departure Reason Description" not in df.columns:
        raise ValueError("Δεν βρέθηκε στήλη 'departure_type' (Departure Reason Description).")

    _parse_dates(df, ["Birth Date", "Hire Date", "Departure Date"])
    df = _apply_business_filters(df)

    if "Nominal Salary" not in df.columns:
        raise ValueError("Δεν βρέθηκε στήλη Nominal Salary.")
    df = _engineer_salary(df)
    df = _engineer_grade(df)
    df = _convert_misc_numeric(df)
    df = add_job_features(df)
    df = df.dropna(subset=["Registry Number", "Hire Date"]).drop_duplicates(subset="Registry Number")

    export = _resolve_snapshot_date(df)
    factor, salary_index = _salary_growth_factor(df, export)
    df["Salary (today)"] = df["Nominal Salary"] * factor

    # ---- Historical snapshot panel (training / evaluation) ----
    frames = []
    for t in _snapshot_dates(export, step_months):
        rows = _features_at(df, t, export)
        rows["Attrition"] = _label(rows, t, horizon_months, export)
        frames.append(rows)
    if not frames or all(f["Attrition"].isna().all() for f in frames):
        raise ValueError(
            f"Δεν υπάρχει αρκετό ιστορικό: χρειάζονται δεδομένα τουλάχιστον "
            f"{horizon_months} μηνών μετά το {ML_DEPARTURE_DATE_CUTOFF}."
        )
    panel = pd.concat(frames, ignore_index=True)

    # ---- Current employees (scoring set) ----
    current = _features_at(df, export, export)

    available_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    available_nums = [c for c in NUMERIC_COLS if c in panel.columns]

    # The encoder learns category lists / rare-category lumping only — no labels —
    # so fitting it on the whole panel does not leak the target.
    encoder, X = fit_encoder(
        panel[available_cats + available_nums],
        categorical_cols=available_cats,
        numeric_cols=available_nums,
    )
    X.columns = X.columns.astype(str)
    X_current = encoder.transform(current[available_cats + available_nums])

    meta_cols = pd.DataFrame(
        {col: current[col] if col in current.columns else pd.Series(index=current.index, dtype="object")
         for col in META_BASE_COLS}
    )
    # Segment columns (UI breakdowns, fairness checks) + outcome dates (survival-style analyses)
    seg_cols = [
        c for c in ("Job Property Label", "Division", "Department", "Job Family", "Gender", "Age",
                    "Departure Date", "Departure Reason Description")
        if c in panel.columns
    ]
    panel_meta = panel[seg_cols].rename(columns={"Job Property Label": "Job Property"})

    return MLPreparedData(
        X=X,
        y=panel["Attrition"],
        snapshots=panel["Snapshot"],
        panel_meta=panel_meta,
        X_current=X_current,
        registry_numbers=current["Registry Number"],
        meta_cols=meta_cols,
        df_ui=current,
        encoder=encoder,
        snapshot_date=export,
        horizon_months=horizon_months,
        salary_index=salary_index,
    )
