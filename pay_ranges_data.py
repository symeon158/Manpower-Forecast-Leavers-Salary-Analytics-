"""
Data loading & cleaning.
- Reads the CSV with multiple Greek encodings.
- Cleans salaries / grades / dates with rules from config.
- Returns a canonical DataFrame.
"""
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from config.constants import (
    COLUMN_RENAME_MAP,
    CSV_ENCODINGS,
    CSV_SEPARATOR,
    LOW_SALARY_MULTIPLIER,
    LOW_SALARY_THRESHOLD,
    SPECIAL_SALARY_COMPANIES,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Low-level CSV reading
# --------------------------------------------------------------------
def robust_read_csv(uploaded_file: Any) -> pd.DataFrame:
    """
    Read uploaded_file (Streamlit UploadedFile) as CSV trying several
    Greek encodings. Never raises UnicodeDecodeError - falls back to
    latin1 with replacement.
    """
    raw_bytes = uploaded_file.getvalue()  # safe even if read multiple times

    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), sep=CSV_SEPARATOR, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:  # malformed CSV, etc.
            log.debug("CSV read failed with %s: %s", enc, e)
            continue

    # Final fallback: never fails
    log.warning("All preferred encodings failed - using latin1 with replacement.")
    return pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=CSV_SEPARATOR,
        encoding="latin1",
        encoding_errors="replace",
    )


# --------------------------------------------------------------------
# Salary parsing
# --------------------------------------------------------------------
def parse_european_money(x: Any) -> float:
    """
    Parse '€ 1.178,29' -> 1178.29.
    Handles thousand '.' and decimal ','. Returns NaN on failure.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return np.nan
    s = s.replace("€", "").replace("\u00a0", " ").strip()
    s = s.replace(".", "").replace(",", ".")
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    return pd.to_numeric(s, errors="coerce")


def _clean_salary_series(s: pd.Series) -> pd.Series:
    """Vectorised salary cleaner: 'X.XXX,YY' style → float."""
    return (
        s.astype(str)
        .str.strip()
        .replace({"": np.nan, "-": np.nan, "nan": np.nan, "None": np.nan})
        .str.replace(".", "", regex=False)   # thousands
        .str.replace(",", ".", regex=False)  # decimal
        .pipe(pd.to_numeric, errors="coerce")
    )


def _apply_low_salary_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    For SPECIAL_SALARY_COMPANIES, multiply salaries below the threshold
    (probably daily/weekly figures) by LOW_SALARY_MULTIPLIER.
    """
    if "NominalSalary" not in df.columns or "Company" not in df.columns:
        return df

    mask = (
        df["Company"].isin(SPECIAL_SALARY_COMPANIES)
        & df["NominalSalary"].notna()
        & (df["NominalSalary"] < LOW_SALARY_THRESHOLD)
    )
    df.loc[mask, "NominalSalary"] = df.loc[mask, "NominalSalary"] * LOW_SALARY_MULTIPLIER
    return df


# --------------------------------------------------------------------
# Public loader
# --------------------------------------------------------------------
@st.cache_data(show_spinner="📂 Loading and preprocessing data...")
def load_leavers_data(uploaded_file: Any) -> pd.DataFrame:
    """
    Load + preprocess the leavers/full employee dataset.
    Cached on file content (Streamlit hashes the UploadedFile).
    """
    df = robust_read_csv(uploaded_file)

    # Normalise headers, then rename to canonical names
    df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False).str.lower()
    df = df.rename(columns=COLUMN_RENAME_MAP, errors="ignore")

    # Salary column auto-detect
    salary_col = next(
        (c for c in df.columns if "nominal" in c.lower() and "sal" in c.lower()),
        None,
    )
    if salary_col and salary_col != "NominalSalary":
        df = df.rename(columns={salary_col: "NominalSalary"})

    # Parse dates
    df["HireDate"] = pd.to_datetime(df.get("HireDate"), dayfirst=True, errors="coerce")
    df["DepartureDate"] = pd.to_datetime(
        df.get("DepartureDate"), dayfirst=True, errors="coerce"
    ) if "DepartureDate" in df.columns else pd.NaT

    df = df.dropna(subset=["HireDate"])

    # Clean salary
    if "NominalSalary" in df.columns:
        df["NominalSalary"] = _clean_salary_series(df["NominalSalary"])
        df = _apply_low_salary_rule(df)

    # Clean grade
    grade_cols = [c for c in df.columns if "grade" in c.lower()]
    if grade_cols:
        gc = grade_cols[0]
        df["GRADE_clean"] = (
            df[gc].astype(str).str.replace(",", ".", regex=False).replace({"99999": "0.1"})
        )
        df["GRADE_clean"] = pd.to_numeric(df["GRADE_clean"], errors="coerce")

    # Deduplicate by registry number if available
    if "Registry_Number" in df.columns:
        df = df.drop_duplicates(subset=["Registry_Number"])

    return df.reset_index(drop=True)
