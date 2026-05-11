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
DEFAULT_PROPHET_CHANGEPOINT_PRIOR: float = 0.05
DEFAULT_PROPHET_SEASONALITY_PRIOR: float = 10.0
DEFAULT_PROPHET_WEEKLY_SEASONALITY: bool = False
START_YEAR_FOR_TS: int = 2019

# Optuna
OPTUNA_DEFAULT_TRIALS: int = 30
OPTUNA_TRIAL_RANGE: tuple[int, int] = (10, 100)

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
ML_TEST_SIZE: float = 0.2
ML_RANDOM_STATE: int = 42
ML_CV_FOLDS: int = 5
ML_RANDOMSEARCH_ITERATIONS: int = 50
ML_DEFAULT_THRESHOLD: float = 0.4

# --- New: scoring + temporal handling ---
# Use F1 by default; "average_precision" (PR-AUC) is also a strong choice for
# imbalanced data. Avoid pure "recall" — its degenerate optimum is predicting 1
# for everyone. See ml/trainer.py for the rationale.
ML_SCORING: str = "f1"

# Switch CV / split strategy.
# "temporal" → train on people whose Hire Date (or Departure Date) is older,
#              test on more recent rows. Honest estimate of production behavior.
# "random"   → classic stratified shuffle. Optimistic but matches old behavior.
ML_SPLIT_STRATEGY: str = "temporal"

# Reference date used to compute Tenure for *active* employees.  Setting it to a
# fixed point avoids the leak where the same active employee scores higher each
# month simply because their tenure keeps growing.  None ⇒ use today.
ML_SNAPSHOT_DATE: str | None = None  # e.g. "2024-12-31"

# --- New: feature engineering ---
# Categories appearing in fewer than this fraction of rows are folded into __OTHER__.
# Keeps OneHot dimensionality sane and stabilizes SHAP attributions.
RARE_CATEGORY_MIN_FRACTION: float = 0.01

# --- New: calibration ---
# When True, wrap the trained classifier in CalibratedClassifierCV so that
# `predict_proba` outputs are interpretable as actual probabilities.
ML_CALIBRATE_PROBABILITIES: bool = True
ML_CALIBRATION_METHOD: str = "isotonic"  # or "sigmoid" (Platt scaling)
ML_CALIBRATION_HOLDOUT_FRACTION: float = 0.15  # carved out of train

# --- New: evaluation ---
ML_BOOTSTRAP_ITERATIONS: int = 200    # for confidence intervals on AUC / F1
ML_LEARNING_CURVE_POINTS: int = 5     # # train-size samples in learning curve

# XGBoost hyperparameter search space
XGB_PARAM_GRID: dict = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "gamma": [0, 0.1, 0.5],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0.001, 0.01, 0.1, 1],
    "reg_lambda": [0.001, 0.01, 0.1, 1],
    "min_child_weight": [1, 3, 5],   # NEW: helps with imbalance
}

# Risk thresholds (probability)
RISK_LOW_MAX: float = 0.15
RISK_MEDIUM_MAX: float = 0.35

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
