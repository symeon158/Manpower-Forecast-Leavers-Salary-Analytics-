"""
Executive insights: cost of attrition, rolling-12-month KPIs, hotspot (funnel)
statistics. Pure functions — no Streamlit — so they can be tested and reused
for the downloadable board pack.

Conventions
-----------
- "Last 12 months" = the 12 most recent COMPLETE months of the data; compared
  with the 12 months before that.
- Turnover rate = voluntary leavers in the window / average headcount on the
  1st of each month in the window (12-month window → already an annual rate).
- Replacement cost = monthly nominal salary × salary payments per year (14 for
  the Greek companies, 12 elsewhere) × rate for the job level derived from the
  job title (config REPLACEMENT_COST_RATES). Missing / zero salaries are filled
  with the median of the same company and job level (counted and reported).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.constants import (
    ANNUAL_SALARY_MONTHS,
    ANNUAL_SALARY_MONTHS_DEFAULT,
    ANNUAL_SALARY_MONTHS_OVERRIDES,
    FOURTEEN_SALARY_COMPANIES,
    REPLACEMENT_COST_RATES,
)
from forecasting.time_series import classify_voluntary
from ml.job_families import job_family, job_level

# Funnel-plot control limits (one-sided exact Poisson tail probabilities)
P_WATCH = 0.025     # ≈ 95% two-sided limit
P_HOTSPOT = 0.001   # ≈ 99.8% two-sided limit
MIN_UNIT_HEADCOUNT = 5   # smaller units are shown but never classified


# --------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------
def level_of(titles: pd.Series) -> pd.Series:
    """Integer job level 1–5 from job titles (unknown titles → 2, professional)."""
    uniq = {t: job_level(t) for t in titles.dropna().unique()}
    lvl = titles.map(uniq).fillna(2.0)
    return np.floor(lvl + 0.5).clip(1, 5).astype(int)


EXEC_LEVEL = 5


def salary_months(companies: pd.Series) -> pd.Series:
    """Salary payments per year for each row's company."""
    def months(c) -> int:
        if c in ANNUAL_SALARY_MONTHS_OVERRIDES:
            return ANNUAL_SALARY_MONTHS_OVERRIDES[c]
        return ANNUAL_SALARY_MONTHS if c in FOURTEEN_SALARY_COMPANIES else ANNUAL_SALARY_MONTHS_DEFAULT
    return companies.map({c: months(c) for c in companies.dropna().unique()}).fillna(ANNUAL_SALARY_MONTHS_DEFAULT)


def _columns(df: pd.DataFrame) -> tuple[str, str]:
    title_col = "JobTitle" if "JobTitle" in df.columns else "Job Position"
    salary_col = "NominalSalary" if "NominalSalary" in df.columns else "Nominal Salary"
    return title_col, salary_col


def _levels(df: pd.DataFrame) -> pd.Series:
    title_col, _ = _columns(df)
    return level_of(df[title_col]) if title_col in df.columns else pd.Series(2, index=df.index)


def filled_salary(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """Monthly salary with missing / ≤0 values filled; returns (salary, was_filled mask).

    Fill order: median of the same company & job level in `reference`, then the
    same job level anywhere in `reference`, then the overall median.
    """
    _, salary_col = _columns(df)
    salary = df[salary_col].where(df[salary_col] > 0)
    missing = salary.isna()
    if not missing.any():
        return salary, missing
    ref = reference if reference is not None else df
    _, ref_sal_col = _columns(ref)
    ref = ref.assign(_lvl=_levels(ref), _sal=ref[ref_sal_col].where(ref[ref_sal_col] > 0))
    lvl = _levels(df)
    fill = pd.Series(np.nan, index=df.index)
    if "Company" in ref.columns and "Company" in df.columns:
        by_company_level = ref.groupby(["Company", "_lvl"])["_sal"].median()
        fill = pd.Series([by_company_level.get(k, np.nan) for k in zip(df["Company"], lvl)], index=df.index)
    fill = fill.fillna(lvl.map(ref.groupby("_lvl")["_sal"].median())).fillna(ref["_sal"].median())
    return salary.fillna(fill.where(missing)), missing


def replacement_cost(
    df: pd.DataFrame, rates: dict[int, float] | None = None, reference: pd.DataFrame | None = None
) -> pd.Series:
    """€ replacement cost per row: salary × payments per year × rate for the job level."""
    rates = rates or REPLACEMENT_COST_RATES
    salary, _ = filled_salary(df, reference)
    months = salary_months(df["Company"]) if "Company" in df.columns else ANNUAL_SALARY_MONTHS_DEFAULT
    return salary * months * _levels(df).map(rates)


def is_voluntary(df: pd.DataFrame) -> pd.Series:
    if "Departure Type" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["Departure Type"].map(classify_voluntary).eq("Voluntary") & df["DepartureDate"].notna()


def active_on(df: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    return (df["HireDate"] <= day) & (df["DepartureDate"].isna() | (df["DepartureDate"] > day))


@dataclass
class Window:
    start: pd.Timestamp     # first day of first month
    end: pd.Timestamp       # last day of last month

    @property
    def month_starts(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq="MS")

    def label(self) -> str:
        return f"{self.start:%Y-%m} → {self.end:%Y-%m}"


def rolling_windows(last_complete_month: pd.Timestamp) -> tuple[Window, Window]:
    """(last 12 complete months, the 12 months before)."""
    end = last_complete_month + pd.offsets.MonthEnd(0)
    start = last_complete_month - pd.DateOffset(months=11)
    prev = Window(start - pd.DateOffset(months=12), start - pd.Timedelta(days=1))
    return Window(start, end), prev


def avg_headcount(df: pd.DataFrame, w: Window, by: str | None = None):
    snaps = [df.loc[active_on(df, m), by].value_counts() if by else pd.Series({"_": active_on(df, m).sum()})
             for m in w.month_starts]
    avg = pd.concat(snaps, axis=1).fillna(0).mean(axis=1)
    return avg if by else float(avg.iloc[0])


def leavers_in(df: pd.DataFrame, w: Window) -> pd.DataFrame:
    m = df["DepartureDate"].between(w.start, w.end) & is_voluntary(df)
    return df[m]


# --------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------
@dataclass
class PeriodKPIs:
    window: Window
    headcount_end: int
    avg_headcount: float
    voluntary_leavers: int
    voluntary_rate: float
    cost: float
    avg_cost_per_leaver: float      # total ÷ leavers — used only to cost the projection
    median_cost_per_leaver: float   # the typical departure (robust to a few senior exits)
    cost_by_level: pd.DataFrame     # Level, Leavers, Typical cost, Total cost, Share of cost
    early_leaver_share: float       # voluntary leavers with < 12 months tenure
    exec_leavers: int               # director / general-manager level (job level 5)
    exec_cost: float
    salaries_filled: int            # leavers whose missing salary was filled with a peer median

    @property
    def cost_excl_exec(self) -> float:
        return self.cost - self.exec_cost


LEVEL_NAMES = {1: "Entry / operational", 2: "Professional", 3: "Supervisor / coordinator",
               4: "Manager / lead", 5: "Director / executive"}


def _cost_by_level(cost: pd.Series, levels: pd.Series) -> pd.DataFrame:
    g = cost.groupby(levels)
    out = pd.DataFrame({"Leavers": g.size(), "Typical cost": g.median(), "Total cost": g.sum()})
    out = out.reindex(sorted(LEVEL_NAMES)).dropna(how="all")
    out["Share of cost"] = out["Total cost"] / cost.sum() if cost.sum() else np.nan
    out.insert(0, "Level", [LEVEL_NAMES[i] for i in out.index])
    return out.reset_index(drop=True)


def period_kpis(df: pd.DataFrame, w: Window, rates: dict[int, float]) -> PeriodKPIs:
    lv = leavers_in(df, w)
    hc = avg_headcount(df, w)
    cost = replacement_cost(lv, rates, reference=df)
    is_exec = _levels(lv) == EXEC_LEVEL
    tenure_m = (lv["DepartureDate"] - lv["HireDate"]).dt.days / 30.44
    return PeriodKPIs(
        window=w,
        headcount_end=int(active_on(df, w.end).sum()),
        avg_headcount=hc,
        voluntary_leavers=len(lv),
        voluntary_rate=len(lv) / hc if hc else float("nan"),
        cost=float(cost.sum()),
        avg_cost_per_leaver=float(cost.mean()) if len(lv) else float("nan"),
        median_cost_per_leaver=float(cost.median()) if len(lv) else float("nan"),
        cost_by_level=_cost_by_level(cost, _levels(lv)),
        early_leaver_share=float((tenure_m < 12).mean()) if len(lv) else float("nan"),
        exec_leavers=int(is_exec.sum()),
        exec_cost=float(cost[is_exec].sum()),
        salaries_filled=int(filled_salary(lv, df)[1].sum()),
    )


def status_vs_target(value: float, target: float, tolerance: float, lower_is_better: bool = True) -> str:
    """'good' / 'warning' / 'critical' for a KPI against its target."""
    if np.isnan(value):
        return "neutral"
    if lower_is_better:
        if value <= target:
            return "good"
        return "warning" if value <= target * (1 + tolerance) else "critical"
    if value >= target:
        return "good"
    return "warning" if value >= target * (1 - tolerance) else "critical"


# --------------------------------------------------------------------
# Breakdown + funnel (hotspot) statistics
# --------------------------------------------------------------------
def unit_breakdown(
    df: pd.DataFrame, w: Window, by: str, rates: dict[int, float]
) -> tuple[pd.DataFrame, float]:
    """Per-unit avg headcount, voluntary leavers, rate, cost and hotspot status; plus the overall rate.

    Status uses an exact Poisson test of the unit's leavers against what the
    overall rate predicts for its size, so small units are not flagged on noise.
    """
    from scipy.stats import poisson

    if by == "Job Family":
        df = df.assign(**{"Job Family": df["JobTitle"].map(
            {t: job_family(t) for t in df["JobTitle"].dropna().unique()})})
    df = df[df[by].notna()]
    hc = avg_headcount(df, w, by=by)
    lv = leavers_in(df, w)
    out = pd.DataFrame({
        "Avg headcount": hc,
        "Voluntary leavers": lv.groupby(by).size(),
        "Cost (€)": replacement_cost(lv, rates, reference=df).groupby(lv[by]).sum(),
    }).fillna(0)
    out = out[out["Avg headcount"] > 0]
    out["Turnover rate"] = out["Voluntary leavers"] / out["Avg headcount"]

    p0 = out["Voluntary leavers"].sum() / out["Avg headcount"].sum()
    expected = p0 * out["Avg headcount"]
    out["Expected leavers"] = expected
    p_high = poisson.sf(out["Voluntary leavers"] - 1, expected)    # P(X ≥ observed)
    p_low = poisson.cdf(out["Voluntary leavers"], expected)        # P(X ≤ observed)
    out["p-value (high)"] = p_high
    out["Status"] = np.select(
        [out["Avg headcount"] < MIN_UNIT_HEADCOUNT, p_high < P_HOTSPOT, p_high < P_WATCH, p_low < P_WATCH],
        ["Too small to judge", "Significant hotspot", "Watch", "Better than average"],
        default="Within normal range",
    )
    out.index.name = by
    return out.reset_index().sort_values("Cost (€)", ascending=False), p0


def funnel_limits(p0: float, n_max: float, tail: float, points: int = 200) -> pd.DataFrame:
    """Smooth control limits (as rates) matching the exact Poisson test in `unit_breakdown`.

    P(X ≥ k | λ) equals the regularised lower incomplete gamma P(k, λ), which is
    continuous in k. Solving P(k, λ) = tail for k gives a smooth upper limit; a
    unit lies above it exactly when its test p-value is below `tail`. Same idea
    for the lower limit with P(X ≤ k) = Q(k + 1, λ).
    """
    from scipy.optimize import brentq
    from scipy.special import gammainc, gammaincc

    n = np.geomspace(MIN_UNIT_HEADCOUNT, max(n_max * 1.05, MIN_UNIT_HEADCOUNT + 1), points)
    upper, lower = [], []
    for lam in p0 * n:
        hi = lam + 20 * np.sqrt(lam) + 20
        upper.append(brentq(lambda k: gammainc(k, lam) - tail, 1e-9, hi))
        lower.append(brentq(lambda k: gammaincc(k + 1, lam) - tail, -1 + 1e-9, hi)
                     if gammaincc(1e-9, lam) < tail else 0.0)
    return pd.DataFrame({"n": n, "upper": np.array(upper) / n, "lower": np.clip(np.array(lower), 0, None) / n})


# --------------------------------------------------------------------
# € at risk (ML predictions × replacement cost)
# --------------------------------------------------------------------
def euro_at_risk(df_current: pd.DataFrame, proba: pd.Series, rates: dict[int, float]) -> pd.DataFrame:
    """Per current employee: probability, replacement cost and expected cost (p × cost)."""
    cost = replacement_cost(df_current, rates)
    return pd.DataFrame({
        "Division": df_current.get("Division"),
        "Job Family": df_current.get("Job Family"),
        "Probability": proba,
        "Replacement cost (€)": cost,
        "€ at risk": proba * cost,
    })
