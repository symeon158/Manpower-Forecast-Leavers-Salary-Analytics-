"""
Feature encoding with persistence.

Replaces the old `pd.get_dummies` + `align_columns` pair. The encoder is
fitted on training data, knows its full column set, lumps rare categories
into `__OTHER__`, and silently passes through unseen categories at inference.

Why this matters
----------------
`pd.get_dummies` is non-deterministic across runs: the columns it produces
depend on what's in the dataframe. Two consequences:

1. A category present in next month's CSV but not in last month's training
   would be silently dropped at inference.
2. A category that disappears between training and scoring would mean the
   model has a column with no signal but still occupying feature space.

Using `OneHotEncoder(handle_unknown="ignore")` with a saved category list
fixes both problems and is what production ML pipelines do.

Feature-name sanitization
-------------------------
Greek (and other non-ASCII) characters in category values become part of
one-hot column names like `Department_ΠΩΛΗΣΕΙΣ`. When XGBoost serialises its
internal model with such names, SHAP's `XGBTreeModelLoader` chokes on the
non-UTF-8 bytes (a known SHAP bug, github.com/shap/shap/issues/1260).

To avoid this, we ASCII-fy column names before training and keep a
`display_names` map that the UI uses to show the original Greek labels.
The model only ever sees safe ASCII column names.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from config.constants import RARE_CATEGORY_MIN_FRACTION


def _stable_hash(s: str, digits: int = 4) -> str:
    """Deterministic short hash, unlike Python's hash() which is process-random."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return h[:digits]


def _ascii_safe(name: str) -> str:
    """Convert a column name to a safe ASCII identifier.

    1. Normalise unicode (NFKD), strip diacritics
    2. Transliterate non-ASCII chars to '?' then drop them
    3. Replace non-alphanumeric chars with '_'
    4. Collapse repeated '_' and strip edges
    5. If the result lost the distinguishing part (e.g. "Department_ΠΩΛΗΣΕΙΣ"
       → "Department" because all the Greek was stripped), append a short
       deterministic hash of the original so distinct inputs map to distinct
       safe names.

    The output is *not* meant to be human-readable — it's just a stable,
    UTF-8-safe identifier that XGBoost and SHAP can serialize without issues.
    The UI uses `FeatureEncoder.display_names` to show the original label.
    """
    if not name:
        return "_unnamed"
    # NFKD then drop combining marks
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = nfkd.encode("ascii", errors="ignore").decode("ascii")
    # Replace any non-alphanumeric/underscore with _
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", ascii_only).strip("_")
    cleaned = re.sub(r"_{2,}", "_", cleaned)

    if not cleaned:
        # Pure non-ASCII (e.g. Greek-only label) — fall back to a stable hash
        cleaned = f"col_{_stable_hash(name, digits=8)}"
    elif any(ord(c) > 127 for c in name) and cleaned != name:
        # We lost information; append a stable hash so distinct inputs map
        # to distinct safe names.
        cleaned = f"{cleaned}_{_stable_hash(name, digits=4)}"
    return cleaned


def _dedupe(names: list[str]) -> list[str]:
    """Ensure column names are unique by appending _2, _3, ... to collisions."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
    return out


@dataclass
class FeatureEncoder:
    """Stateful encoder. Fit on train, transform train + new data identically."""
    categorical_cols: list[str]
    numeric_cols: list[str]
    encoder: OneHotEncoder
    rare_replacements: dict[str, set[str]]  # col -> set of rare values that became __OTHER__
    feature_names: list[str]                # ASCII-safe column names actually used for fit/predict
    display_names: dict[str, str] = field(default_factory=dict)   # safe -> original (for UI)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned encoding to new data. Unseen categories become all-zeros."""
        df = df.copy()

        # Lump rare categories the same way they were lumped in training
        for col, rare in self.rare_replacements.items():
            if col in df.columns and rare:
                df[col] = df[col].where(~df[col].isin(rare), other="__OTHER__")

        # Force any column missing from input to a plain "__MISSING__" so the
        # encoder doesn't crash; with handle_unknown=ignore it'll go to all-zero.
        for col in self.categorical_cols:
            if col not in df.columns:
                df[col] = "__MISSING__"
            df[col] = df[col].fillna("__MISSING__").astype(str)

        cat_arr = self.encoder.transform(df[self.categorical_cols])
        raw_cat_names = self.encoder.get_feature_names_out(self.categorical_cols)
        cat_df = pd.DataFrame(cat_arr, columns=raw_cat_names, index=df.index)

        # Numeric columns — keep NaN for XGBoost to handle natively
        num_df = pd.DataFrame(index=df.index)
        for c in self.numeric_cols:
            num_df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

        out = pd.concat([num_df, cat_df], axis=1)
        # Map raw column names → safe ASCII names using display_names (reversed)
        rev = {orig: safe for safe, orig in self.display_names.items()}
        out.columns = [rev.get(c, _ascii_safe(c)) for c in out.columns]
        # Reorder to match training feature order, fill missing with 0
        for c in self.feature_names:
            if c not in out.columns:
                out[c] = 0.0
        return out[self.feature_names]

    def display_name(self, safe_name: str) -> str:
        """UI-facing original name for a sanitized feature."""
        return self.display_names.get(safe_name, safe_name)


def _lump_rare(series: pd.Series, min_fraction: float) -> tuple[pd.Series, set[str]]:
    """Replace categories with frequency below `min_fraction` with __OTHER__."""
    counts = series.value_counts(normalize=True, dropna=False)
    rare = set(counts[counts < min_fraction].index.tolist())
    if not rare:
        return series, set()
    out = series.where(~series.isin(rare), other="__OTHER__")
    return out, rare


def fit_encoder(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    min_fraction: float = RARE_CATEGORY_MIN_FRACTION,
) -> tuple[FeatureEncoder, pd.DataFrame]:
    """Fit the encoder on training data and return the encoder + transformed df.

    The returned dataframe is what you'd train on. Apply `encoder.transform`
    to any future data (test set, active employees, what-if rows).

    All output column names are ASCII-safe so XGBoost can serialize them
    without the SHAP/XGBoost UTF-8 incompatibility bug surfacing.
    """
    df = df.copy()

    # Lump rare categories per column, remember the rare sets
    rare_replacements: dict[str, set[str]] = {}
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "__MISSING__"
        df[col] = df[col].fillna("__MISSING__").astype(str)
        df[col], rare_replacements[col] = _lump_rare(df[col], min_fraction)

    # Encode categoricals
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        drop=None,        # keep all levels — XGB handles redundancy
        dtype=np.float32,
    )
    cat_arr = encoder.fit_transform(df[categorical_cols])
    raw_cat_names = encoder.get_feature_names_out(categorical_cols).tolist()
    cat_df = pd.DataFrame(cat_arr, columns=raw_cat_names, index=df.index)

    # Numeric: keep raw, allow NaN (XGBoost handles natively)
    num_df = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        num_df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

    out = pd.concat([num_df, cat_df], axis=1)
    raw_names = num_df.columns.tolist() + raw_cat_names

    # Sanitize: ASCII-safe + unique
    safe_names = _dedupe([_ascii_safe(n) for n in raw_names])
    display_names = dict(zip(safe_names, raw_names))   # safe -> original

    out.columns = safe_names

    fe = FeatureEncoder(
        categorical_cols=list(categorical_cols),
        numeric_cols=list(numeric_cols),
        encoder=encoder,
        rare_replacements=rare_replacements,
        feature_names=safe_names,
        display_names=display_names,
    )
    return fe, out
