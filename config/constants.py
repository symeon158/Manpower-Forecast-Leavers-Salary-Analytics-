"""
Central configuration. Every magic number lives here.
Edit ONLY this file when business rules change.
"""
from __future__ import annotations

# --------------------------------------------------------------------
# Time / calendar
# --------------------------------------------------------------------
DAYS_PER_MONTH: float = 30.4375          # avg, used for tenure calc
MONTHS_PER_YEAR: int = 12

# --------------------------------------------------------------------
# Forecasting defaults
# --------------------------------------------------------------------
DEFAULT_FORECAST_HORIZON: int = 12
DEFAULT_RECENT_MONTHS_FOR_TUNING: int = 36
# Conservative priors: in rolling backtests they beat Prophet's defaults
# (0.05 / 10) on every series tested; flexible priors were the worst option.
DEFAULT_PROPHET_CHANGEPOINT_PRIOR: float = 0.01
DEFAULT_PROPHET_SEASONALITY_PRIOR: float = 1.0
DEFAULT_PROPHET_WEEKLY_SEASONALITY: bool = False   # monthly data: weekly seasonality is meaningless
START_YEAR_FOR_TS: int = 2019

# Rolling-origin backtest used to pick the forecast model (forecasting/models.py)
FORECAST_BACKTEST_HORIZON: int = 12        # months ahead evaluated from each origin
FORECAST_BACKTEST_ORIGINS: int = 8         # number of forecast origins
FORECAST_BACKTEST_STEP_MONTHS: int = 3     # spacing between origins
FORECAST_MIN_HISTORY_MONTHS: int = 36      # history required before the first origin
FORECAST_SEASONAL_SHRINK: float = 0.5      # 0 = no seasonality, 1 = full historical seasonal index
FORECAST_INTERVAL_LEVEL: float = 0.80      # empirical prediction interval from backtest errors

# Optuna (tunes Prophet on backtest error; slower than the default path)
OPTUNA_DEFAULT_TRIALS: int = 30
OPTUNA_TRIAL_RANGE: tuple[int, int] = (10, 100)
OPTUNA_BACKTEST_ORIGINS: int = 4

# --------------------------------------------------------------------
# Salary cleaning rules
# --------------------------------------------------------------------
# Below this monthly value, we assume the figure is daily/weekly and multiply
LOW_SALARY_THRESHOLD: float = 90.0
LOW_SALARY_MULTIPLIER: float = 26.0  # ~working days/month

# Companies that need the low-salary fix
SPECIAL_SALARY_COMPANIES: list[str] = [
    "ΑΛΟΥΜΥΛ Α.Ε.",
    "CFT CARBON FIBER TECHNOLOGIES P.C.",
    "ALUTRADE ΕΜΠΟΡΙΟ ΑΛΟΥΜΙΝΙΟΥ Α.Ε.",
    "BMP Α.Ε.",
    "ALUSEAL A.E.",
    "GLM HELLAS ΑΕ",
    "ΑΛΟΥΜΥΛ ΑΡΧΙΤΕΚΤΟΝΙΚΑ ΣΥΣΤΗΜΑΤΑ  Α.Ε.",
    "ΓΑ ΒΙΟΜΗΧ. ΠΛΑΣΤ. ΥΛΩΝ  Α.Ε.",
    "BUILDING SYSTEMS INNOVATION CENTRE ΙΔΙΩΤΙΚΗ ΚΕΦΑΛΑΙΟΥΧΙΚΗ ΕΤΑΙΡΕΙΑ",
    "ΝΕΑ ΑΛΟΥΦΟΝΤ ΜΟΝΟΠΡΟΣΩΠΗ ΑΝΩΝΥΜΗ ΕΤΑΙΡΕΙΑ",
]

# --------------------------------------------------------------------
# ML business filters
# --------------------------------------------------------------------
ML_TARGET_COMPANY: str = "ΑΛΟΥΜΥΛ Α.Ε."
ML_WORK_RELATIONSHIP: str = "ΑΟΡΙΣΤΟΥ ΧΡΟΝΟΥ"
ML_TARGET_DEPARTURE_REASON: str = "VOLUNTARY DEPARTURE"
ML_DEPARTURE_DATE_CUTOFF: str = "2018-12-31"
ML_REPLACE_DEPARTMENT_PATTERN: str = "ΕΠΑΝΑΤΙΜΟΛΟΓΗΣΗ"
ML_REPLACE_REASON: str = "ΜΕΤΑΦΟΡΑ ΣΕ ΑΛΛΗ ΕΤΑΙΡΕΙΑ"

# --------------------------------------------------------------------
# ML model
# --------------------------------------------------------------------
ML_RANDOM_STATE: int = 42

# Export date of the data: current employees are scored with features as of
# this date. None ⇒ latest Departure Date in the file (≈ when it was exported).
ML_SNAPSHOT_DATE: str | None = None  # e.g. "2024-12-31"

# --- Point-in-time snapshot design (see ml/data_prep.py) ---
# The model predicts "voluntary exit within the next ML_HORIZON_MONTHS months".
# Rows are snapshots of everyone active every ML_SNAPSHOT_STEP_MONTHS, up to the
# export date. Monthly snapshots beat 6-monthly ones in backtests (PR-AUC
# 0.32 → 0.36 for the survival model).
ML_HORIZON_MONTHS: int = 12
ML_SNAPSHOT_STEP_MONTHS: int = 1

# Older snapshots are down-weighted: their salary/grade values are less
# accurate (the export holds only the latest value per person) and the
# workforce has changed since. Weight halves every N years of snapshot age.
ML_RECENCY_HALF_LIFE_YEARS: float = 1.0

# Categories appearing in fewer than this fraction of rows are folded into __OTHER__.
# Keeps OneHot dimensionality sane and stabilizes SHAP attributions.
RARE_CATEGORY_MIN_FRACTION: float = 0.01

# XGBoost survival (AFT) model — see ml/survival.py. Chosen by out-of-time
# backtests on the 2024-10 / 2025-04 / 2025-10 snapshots; results were stable
# across distribution (normal vs logistic), scale (0.8–1.6) and depth (3–4),
# so the simplest regularised setting is used. A randomized search over older
# folds was no better out-of-time and mostly rewarded drifting features.
XGB_AFT_PARAMS: dict = {
    "objective": "survival:aft",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.2,
    "max_depth": 3,
    "eta": 0.03,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "lambda": 5.0,
    "eval_metric": "aft-nloglik",
}
XGB_AFT_ROUNDS: int = 300

# Out-of-time backtest table: this many snapshots, spaced ML_BACKTEST_SPACING_MONTHS apart.
ML_BACKTEST_SNAPSHOTS: int = 4
ML_BACKTEST_SPACING_MONTHS: int = 6

# --- Evaluation ---
ML_BOOTSTRAP_ITERATIONS: int = 200    # for confidence intervals on AUC / F1
ML_LEARNING_CURVE_POINTS: int = 5     # # train-size samples in learning curve

# Risk thresholds (probability)
RISK_LOW_MAX: float = 0.15
RISK_MEDIUM_MAX: float = 0.35

# --------------------------------------------------------------------
# Executive summary: cost of attrition & targets (ui/tab_executive.py)
# --------------------------------------------------------------------
# ⚠️ PLACEHOLDERS — agree these with HR / Finance before presenting € figures.
# Replacement cost of one leaver as a share of their ANNUAL salary, by job level
# (ml/job_families.py: 1 = entry/operational … 5 = executive). Covers recruiting,
# onboarding, lost productivity while the seat is empty and while the new hire
# ramps up. Ranges commonly cited in HR literature: ~20–50% for operational and
# entry roles, 50–100% for professionals, 100–200% for senior leaders.
REPLACEMENT_COST_RATES: dict[int, float] = {
    1: 0.30,   # entry / operational
    2: 0.50,   # professional
    3: 0.75,   # supervisor / coordinator
    4: 1.00,   # manager / lead
    5: 1.50,   # director / executive
}
# Salary payments per year: 14 for the Greek companies (Christmas / Easter /
# holiday bonuses), 12 elsewhere unless overridden per company below.
ANNUAL_SALARY_MONTHS: int = 14                      # Greek companies
ANNUAL_SALARY_MONTHS_DEFAULT: int = 12              # all other companies
FOURTEEN_SALARY_COMPANIES: list[str] = list(SPECIAL_SALARY_COMPANIES)   # the Greek entities
ANNUAL_SALARY_MONTHS_OVERRIDES: dict[str, int] = {}  # e.g. {"ALUMIL CY LTD": 13}

# Targets for the red / amber / green status of the executive KPIs
EXEC_TARGET_VOLUNTARY_TURNOVER: float = 0.10     # annual voluntary turnover rate
EXEC_TARGET_EARLY_LEAVER_SHARE: float = 0.30     # share of voluntary leavers with < 12 months tenure
EXEC_AMBER_TOLERANCE: float = 0.10               # within +10% of target = amber

# --------------------------------------------------------------------
# Display / UI
# --------------------------------------------------------------------
EARLY_LEAVER_DEFAULT_MONTHS: int = 12
TOP_N_PROFILE_DEFAULT: int = 30
TOP_N_DRIVERS_DEFAULT: int = 10
SHAP_SAMPLE_SIZE_GLOBAL: int = 1000
SHAP_SAMPLE_SIZE_DEPENDENCE: int = 800

# --------------------------------------------------------------------
# Encodings tried when reading the CSV
# --------------------------------------------------------------------
CSV_ENCODINGS: tuple[str, ...] = ("utf-8", "cp1253", "iso-8859-7", "latin1")
CSV_SEPARATOR: str = ";"

# --------------------------------------------------------------------
# Column rename map (raw → canonical)
# --------------------------------------------------------------------
COLUMN_RENAME_MAP: dict[str, str] = {
    # leavers/full file (lowercased)
    "hire_date": "HireDate",
    "departure_date": "DepartureDate",
    "job_title": "JobTitle",
    "departure_type": "Departure Type",
    "company": "Company",
    "division": "Division",
    "department": "Department",
    "κωδικός_εργαζόμενου": "Registry_Number",
    "job_property": "Job Property",
}

# ML rename map (raw → canonical)
ML_COLUMN_RENAME_MAP: dict[str, str] = {
    "Κωδικός εργαζόμενου": "Registry Number",
    "Όνομα": "First Name",
    "Ονομα": "First Name",
    "Επώνυμο": "Last Name",
    "Φύλο": "Gender",
    "Ηλικία": "Age",
    "Ημ/νία γέννησης": "Birth Date",
    "Ημ/νία γένν": "Birth Date",
    "hire_date": "Hire Date",
    "departure_date": "Departure Date",
    "departure_type": "Departure Reason Description",
    "NominalSalary": "Nominal Salary",
    "Σχέση Εργασίας": "Work Relationship",
    "Σχέση Εργ": "Work Relationship",
    "Περιγραφή Υποκαταστήματος": "City",
    "Division": "Division",
    "Job Property": "Job Property",
    "Ιδιότητα Πρ": "Job Property",
    "job_title": "Job Position",
    "GRADE": "Grade",
    "Department": "Department",
    "Οικογενειακή κατάσταση": "Marital Status",
    "NominalSa": "Nominal Salary",
    "Φορολογική κατηγορία": "Tax Category",
    "Βαθμίδα Εκπαίδευσης": "Education Level",
    "Βαθμίδα Εκ": "Education Level",
    "Company": "Company",
}
