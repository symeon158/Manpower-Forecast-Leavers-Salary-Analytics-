"""Loader for the static pay ranges reference market."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config.pay_ranges_data import PAY_RANGES_ROWS


def _ref_level_to_num(x: str) -> float:
    """Map ref_level to numeric (S → 14.5 for plotting order)."""
    s = str(x).strip().upper()
    if s == "S":
        return 14.5
    try:
        return float(s)
    except ValueError:
        return np.nan


@st.cache_data
def load_pay_ranges_df() -> pd.DataFrame:
    df = pd.DataFrame(PAY_RANGES_ROWS)
    df["ref_level_num"] = df["ref_level"].map(_ref_level_to_num)
    return df
