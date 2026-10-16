# **AI-Powered Workforce Intelligence Platform**
### *Executive Insights • Manpower Forecasting • Voluntary-Attrition Prediction • Salary Analytics*

![python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge)
![streamlit](https://img.shields.io/badge/Streamlit-1.5x-success?style=for-the-badge)
![xgboost](https://img.shields.io/badge/XGBoost-survival_AFT-orange?style=for-the-badge)
![prophet](https://img.shields.io/badge/Prophet-backtested-purple?style=for-the-badge)
![shap](https://img.shields.io/badge/SHAP-Explainable_AI-red?style=for-the-badge)

---

## 📌 Overview

A **Streamlit** application that turns a raw HR export (one CSV) into decision-ready
insight for **workforce planning, retention and compensation**. It is built for
HR leadership, HR Business Partners and C&B analysts — and, through the
Executive Summary, for upper management.

Version 2 is a substantial rework focused on one principle: **every number the
app shows must be honest** — forecasts and models are judged on data they have
not seen, money is shown with its assumptions, and small numbers are not
presented as problems.

<img width="1859" height="770" alt="image" src="https://github.com/user-attachments/assets/96ecdaf9-1389-454f-9f21-6b03dc6493ef" />

---

## 🆕 What's new in v2

| Area | v1 | v2 |
|---|---|---|
| **Management view** | — | New **Executive Summary**: cost of attrition (€), € at risk, statistically tested hotspots, red/amber/green KPIs, auto-written key messages, downloadable board pack |
| **Attrition model** | One row per employee; features leaked the departure date (frozen salary, today's age), threshold tuned on the test set | **Point-in-time monthly snapshots + XGBoost survival (AFT) model**, out-of-time test, calibrated 12-month probabilities |
| **Forecasting** | Prophet only; Optuna tuned on *in-sample* error; in-sample MAE/MAPE shown | **Backtest-driven model selection** (seasonal, rate × headcount, Prophet, ensemble), honest prediction intervals, partial last month excluded |
| **Fairness** | Gender, age, marital status used as features | **Protected attributes removed** (no accuracy loss) + fairness table by gender / age band |
| **Turnover** | Share of leavers per job title | **Turnover rate per job title** (leavers ÷ average headcount), annualised, small-sample flag |
| **Salary** | Bar chart of average salary by grade | **Box plot by grade for active employees** with pay-range overlay and grade statistics |
| **Code** | Flat files | Packaged modules (`config/`, `data_loaders/`, `forecasting/`, `ml/`, `insights/`, `ui/`) |

---

## 🚀 Features (one tab each)

### 🏛️ 1. Executive Summary
The two-minute view for upper management, for the last 12 complete months vs the 12 before:
- **Key messages** written automatically from the data (turnover vs target, cost, hotspots, early leavers, forecast, € at risk).
- **KPI tiles** with deltas and **red / amber / green status** against configurable targets.
- **Cost of voluntary attrition (€)** = salary × payments per year × replacement-cost rate by job level — split into director-level vs all other exits and broken down by job level, so a few senior departures cannot hide the trend. The typical (median) cost per departure is shown instead of a skewed average.
- **€ at risk**: the ML probability of a voluntary exit × replacement cost, by division and job family.
- **Hotspot funnel plot**: each unit's turnover against its size with exact Poisson 95% / 99.8% limits — only units above the limits are flagged as real problems.
- **Board pack**: one self-contained HTML file (charts included) to open in a browser and print to PDF.

### 📈 2. Forecast & Time Series
- Monthly **departures, hires and net hires** with automatic **model selection by rolling-origin backtest**: every candidate forecasts 12 months from 8 past points using only data available then; the most accurate one is used.
- Candidates: seasonal mean, rate × headcount (departures), Prophet with conservative priors, and an ensemble.
- **Prediction intervals from real backtest errors** — separately for each month and for the 12-month total.
- Leaderboard with monthly MAE, 12-month total error and bias; optional **Optuna tuning of Prophet on backtest error**.

### 💶 3. Salary & Grade
- **Box plot of nominal salary by grade for active employees**, one company at a time; split by job property; outliers / all points; mean ± SD; log scale.
- Optional **pay-range overlay** (P25–P75 band, P50 line) from the reference pay ranges.
- **Grade statistics table**: percentiles, spread, step-up between grades (flags grades whose median is lower than the grade below), median vs pay-range midpoint.
- Leaver / hire salary distributions and voluntary vs non-voluntary departures per month.

### 🎯 4. Churn Profiles
- High-risk profile of leavers per job title (counts, ratios, salary, grade, cost index).
- **Turnover report per job title**: start / end / average headcount, leavers, voluntary leavers, **annualised turnover**, comparison with the overall rate, average tenure of leavers and a small-sample flag.

### 🧲 5. Hire Profiles
Hiring patterns per job title in the selected period.

### 🤖 6. Voluntary Attrition ML (XGBoost survival + SHAP)
- Predicts the **probability that an employee leaves voluntarily within the next 12 months** (dismissals, retirements etc. are not counted as positives).
- **Monthly point-in-time snapshots**: tenure is computed as of each snapshot; salaries are restated in today's money; drifting fields (grade) are not used as features; older snapshots are down-weighted.
- **Survival model (XGBoost AFT)** learns *time to a voluntary exit*, so recent employees with less than 12 months of follow-up still contribute (censored).
- **Out-of-time evaluation**: trained only on what was known at the test date; calibration and decision threshold come from an earlier validation snapshot — never from the test set.
- Robustness panel: ROC / PR AUC, precision and lift in the top-10% risk group, bootstrap CIs, reliability diagram, per-snapshot backtest, per-segment metrics and a **fairness check** by gender and age band.
- **SHAP** global importance, dependence plots and per-employee drivers (exact contributions in log-odds).
- **What-If simulator** for a new-hire or existing profile.
- **Job families & levels**: free-text job titles mapped to 14 families and 5 seniority levels (`ml/job_families.py`).

### 💰 7. Pay Ranges
Reference pay ranges by level, location and pay zone: band chart, spread heatmap and drill-down.
> The repository ships **sample (fictitious) pay ranges** — replace `config/pay_ranges_data.py` with your own.

---

## 🧠 Methodology highlights

- **No leakage by design.** HR exports keep only each person's latest values (leavers keep their exit salary, everyone carries today's age). The ML pipeline rebuilds the history as monthly snapshots so the model only sees what was known on each date.
- **Judged out of sample.** Both the forecasts and the attrition model are evaluated on periods they were not trained on; reported metrics are never in-sample.
- **Simple beats complex when it is more accurate.** The forecast tab picks the winning model per series from the backtest instead of always using Prophet.
- **Uncertainty is shown.** Intervals come from observed errors; monthly and 12-month-total intervals are computed separately.
- **Signal vs noise.** Hotspots are flagged with exact statistical limits; units under 5 people are never classified; turnover tables flag small samples.
- **Money with assumptions.** All € figures use replacement-cost rates that are editable in the app and clearly labelled as assumptions.

---

## 🎯 Business value

- **Leadership**: what attrition costs, where it concentrates, what the next 12 months look like, and where retention spending pays off.
- **HR Business Partners**: at-risk employees with explainable drivers, statistically real hotspots, early-leaver signals.
- **C&B**: pay positioning by grade, grade inversions, distance from pay-range midpoints.
- **Workforce planning**: backtested forecasts of hires and departures with realistic ranges.

---

## 📂 Data requirements

Upload one `;`-separated CSV (UTF-8 or Greek encodings are detected automatically) with one row per employee — active and former. Expected columns:

`Κωδικός εργαζόμενου` (employee id), `Ονομα`, `Επώνυμο`, `Φύλο`, `Ηλικία`, `Ημ/νία γέννησης`, `hire_date`, `departure_date`, `departure_type`, `Σχέση Εργασίας` (contract type), `Division`, `Job Property`, `job_title`, `GRADE`, `Περιγραφή Υποκαταστήματος` (city), `Department`, `Οικογενειακή κατάσταση`, `NominalSalary`, `Φορολογική κατηγορία`, `Βαθμίδα Εκπαίδευσης`, `Company`, optional `Promoted`.

Dates as `dd/mm/yyyy`; salaries as monthly amounts (`1.234,56` format supported).
The ML tab models the company and contract type set in `config/constants.py` (`ML_TARGET_COMPANY`, `ML_WORK_RELATIONSHIP`).

**No HR data is included in this repository**, and `.gitignore` excludes CSV / Excel files.

---

## ⚙️ Getting started

```bash
git clone https://github.com/symeon158/Manpower-Forecast-Leavers-Salary-Analytics-.git
cd Manpower-Forecast-Leavers-Salary-Analytics-
pip install -r requirements.txt
streamlit run app.py
```

Run `streamlit run app.py` from the repository root so the packages next to `app.py` can be imported.

---

## 🔧 Configuration (`config/constants.py`)

| Want to change… | Setting |
|---|---|
| Replacement-cost rates per job level (**placeholders — agree with HR / Finance**) | `REPLACEMENT_COST_RATES` |
| Salary payments per year (14 for listed companies, 12 elsewhere, overrides) | `ANNUAL_SALARY_MONTHS`, `ANNUAL_SALARY_MONTHS_DEFAULT`, `FOURTEEN_SALARY_COMPANIES`, `ANNUAL_SALARY_MONTHS_OVERRIDES` |
| Executive KPI targets | `EXEC_TARGET_VOLUNTARY_TURNOVER`, `EXEC_TARGET_EARLY_LEAVER_SHARE`, `EXEC_AMBER_TOLERANCE` |
| ML population | `ML_TARGET_COMPANY`, `ML_WORK_RELATIONSHIP`, `ML_TARGET_DEPARTURE_REASON` |
| Prediction horizon, snapshot spacing, recency weighting | `ML_HORIZON_MONTHS`, `ML_SNAPSHOT_STEP_MONTHS`, `ML_RECENCY_HALF_LIFE_YEARS` |
| Survival model settings | `XGB_AFT_PARAMS`, `XGB_AFT_ROUNDS` |
| Forecast backtest, intervals, seasonality strength | `FORECAST_*` |
| Prophet priors / Optuna | `DEFAULT_PROPHET_*`, `OPTUNA_*` |
| Salary cleaning rules | `LOW_SALARY_THRESHOLD`, `LOW_SALARY_MULTIPLIER`, `SPECIAL_SALARY_COMPANIES` |
| Job title → family / level rules | `ml/job_families.py` |
| Pay ranges | `config/pay_ranges_data.py` |

---

## 🗂 Project structure

```
├── app.py                    # Entry point: upload, filters, KPI row, tabs
├── _compat.py                # NumPy / SHAP compatibility shims
├── requirements.txt
├── config/
│   ├── constants.py          # All business rules & tunables
│   └── pay_ranges_data.py    # Reference pay ranges (sample data)
├── data_loaders/             # CSV reading & cleaning, pay-range loader
├── forecasting/
│   ├── time_series.py        # Monthly series (complete months only)
│   ├── models.py             # Candidate models, backtest, auto-selection, intervals
│   └── prophet_model.py      # Prophet fit + Optuna (tuned on backtest error)
├── insights/
│   └── executive.py          # Cost of attrition, rolling KPIs, hotspot statistics
├── ml/
│   ├── data_prep.py          # Monthly point-in-time snapshot panel
│   ├── survival.py           # XGBoost AFT survival model (classifier-style API)
│   ├── trainer.py            # Validation / out-of-time test / final model, backtest, fairness
│   ├── job_families.py       # Job title → family & level
│   ├── features.py           # One-hot encoding with rare-category folding
│   ├── evaluation.py         # Thresholds, bootstrap CI, calibration, segments
│   └── shap_explain.py       # SHAP utilities
└── ui/
    ├── tab_executive.py      # Executive Summary + board pack
    ├── tab_forecast.py       # Forecast & model leaderboard
    ├── tab_salary.py         # Salary & Grade (+ salary_grade_box.py)
    ├── tab_profiles.py       # Churn / hire profiles, turnover per job title
    ├── tab_ml.py             # Attrition model, robustness, fairness
    ├── tab_shap.py, tab_simulator.py, tab_pay_ranges.py
    ├── sidebar.py, components.py, styles.py
    └── viz_theme.py          # Validated chart colours (light / dark)
```

---

## 🔒 Privacy & responsible use

- Keep HR exports and real compensation data **out of public repositories**.
- The attrition model **does not use gender, age, marital status or tax category**; review the fairness table before acting on predictions.
- Predictions are **risk indicators for supportive conversations** (e.g. stay interviews), not grounds for decisions about individuals.
- € figures depend on the replacement-cost assumptions — confirm them with HR / Finance before presenting.
