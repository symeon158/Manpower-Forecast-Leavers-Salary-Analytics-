import io
from datetime import datetime

import numpy as np
import numpy as np

# --- NumPy 2.x compatibility shims for old libraries (shap, xgboost κλπ.) ---
if not hasattr(np, "float"):
    np.float = float  # deprecated alias, needed by shap
if not hasattr(np, "int"):
    np.int = int      # για τυχόν κλήσεις np.int
if not hasattr(np, "bool"):
    np.bool = bool    # για τυχόν κλήσεις np.bool

import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import optuna

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
import shap

# CUSTOM CSS FOR KPI CARDS
# -------------------------------------------------------------------
kpi_css = """
<style>
.kpi-card {
    background: linear-gradient(135deg, #f0f4f8 0%, #dce6f2 100%);
    border-radius: 16px;
    padding: 16px 18px;
    border: 1px solid #e2e8f0;
    color: #0d1b2a;
    box-shadow: 0 3px 10px rgba(0,0,0,0.07);
    font-family: 'Segoe UI', sans-serif;
}

.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a5568;
    font-weight: 700;   /* BOLD */
}

.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1f77b4;
    margin-top: 4px;
}

.kpi-subtitle {
    font-size: 0.80rem;
    color: #4a5568;
    margin-top: 2px;
}

.kpi-badge {
    display: inline-block;
    font-size: 0.70rem;
    padding: 2px 10px;
    border-radius: 10px;
    background: #edf2f7;
    color: #1f77b4;
    margin-top: 6px;
    border: 1px solid #d1d9e0;
}
</style>
"""
st.markdown(kpi_css, unsafe_allow_html=True)




def kpi_card(label: str, value: str, subtitle: str | None = None, badge: str | None = None):
    """Render a single KPI card with custom CSS."""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ''}
        {f'<div class="kpi-badge">{badge}</div>' if badge else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Manpower Time Series & Leavers Salary",
    layout="wide"
)

st.title("📈 Manpower Forecast & Leavers Salary Analytics (Optimized)")
st.caption(
    "Time series forecasting με seasonality, salary insights, Optuna tuning, HR KPIs, "
    "σύγκριση Prophet vs baseline και ML attrition analysis."
)

# =============================================================================
# HELPERS - DATA LOADING & PREPROCESSING
# =============================================================================
def robust_read(uploaded_file) -> pd.DataFrame:
    """
    Διαβάζει το uploaded_file δοκιμάζοντας πολλές ελληνικές κωδικοποιήσεις.
    Χρησιμοποιεί BytesIO, οπότε δεν μπλέκουμε με seek/position.
    ΠΟΤΕ δεν πετάει UnicodeDecodeError.
    """
    raw_bytes = uploaded_file.getvalue()  # πάρε όλα τα bytes μία φορά

    encodings = ["utf-8", "cp1253", "iso-8859-7", "latin1"]

    for enc in encodings:
        try:
            return pd.read_csv(
                io.BytesIO(raw_bytes),
                sep=";",
                encoding=enc
            )
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    # Τελικό fallback – διάβασε τα πάντα, αντικατάστησε «χαλασμένους» χαρακτήρες
    return pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=";",
        encoding="latin1",
        errors="replace"
    )


@st.cache_data
def load_leavers_data(uploaded_file) -> pd.DataFrame:
    """Load and preprocess the leavers/full employee dataset."""
    df = robust_read(uploaded_file)

    # Normalize column names (πιο εύχρηστα)
    df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False).str.lower()

    # Standard names
    df = df.rename(
        columns={
            "hire_date": "HireDate",
            "departure_date": "DepartureDate",
            "job_title": "JobTitle",
            "departure_type": "Departure Type",
            "company": "Company",
            "division": "Division",
            "department": "Department",
            "κωδικός_εργαζόμενου": "Registry_Number",
            "job_property": "Job Property"
        },
        errors="ignore"
    )

    # Detect salary column
    salary_col = None
    for c in df.columns:
        cl = c.lower()
        if "nominal" in cl and "sal" in cl:
            salary_col = c
            break

    if salary_col is not None:
        df.rename(columns={salary_col: "NominalSalary"}, inplace=True)

    # Parse dates
    df["HireDate"] = pd.to_datetime(df.get("HireDate"), dayfirst=True, errors="coerce")
    if "DepartureDate" in df.columns:
        df["DepartureDate"] = pd.to_datetime(df.get("DepartureDate"), dayfirst=True, errors="coerce")
    else:
        df["DepartureDate"] = pd.NaT

    # Keep rows with HireDate
    df = df.dropna(subset=["HireDate"])

    # Clean salary
    if "NominalSalary" in df.columns:
        df["NominalSalary"] = (
            df["NominalSalary"]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "-": np.nan})
            .str.replace(".", "", regex=False)   # thousands separator
            .str.replace(",", ".", regex=False)  # decimal
        )
        df["NominalSalary"] = pd.to_numeric(df["NominalSalary"], errors="coerce")
        # 2) Company list for special salary rule

        special_companies = [
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

        threshold = 90  # <<< CHANGE THIS to your preferred cutoff

        if "NominalSalary" in df.columns and "Company" in df.columns:
            df.loc[
                (df["Company"].isin(special_companies)) &
                (df["NominalSalary"].notna()) &
                (df["NominalSalary"] < threshold),
                "NominalSalary"
            ] = df["NominalSalary"] * 26

    # Clean grade
    grade_cols = [col for col in df.columns if "grade" in col]
    if grade_cols:
        grade_col_name = grade_cols[0]
        df["GRADE_clean"] = (
            df[grade_col_name]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df["GRADE_clean"] = df["GRADE_clean"].replace({"99999": "0.1"})
        df["GRADE_clean"] = pd.to_numeric(df["GRADE_clean"], errors="coerce")
    df.drop_duplicates(subset=["Registry_Number"], inplace=True)

    return df


def build_monthly_time_series(df_slice: pd.DataFrame, selected_dep_types, start_year: int = 2019) -> pd.DataFrame:
    """
    Build a monthly time series where:
    - Hires = all HireDates
    - Departures = rows with DepartureDate, optionally filtered by Departure Type
    - Headcount = snapshot count ανά μήνα (active employees)
    """
    if df_slice["HireDate"].isna().all():
        return pd.DataFrame(columns=["Date", "Hires", "Departures", "NetHires", "Headcount", "TurnoverRate"])

    # Monthly hires
    hires = (
        df_slice
        .groupby(df_slice["HireDate"].dt.to_period("M"))
        .size()
        .rename("Hires")
    )
    hires.index = hires.index.to_timestamp()

    # Monthly departures (filtered)
    df_departures = df_slice.dropna(subset=["DepartureDate"]).copy()
    if "Departure Type" in df_departures.columns and selected_dep_types:
        df_departures = df_departures[df_departures["Departure Type"].isin(selected_dep_types)]

    departures = (
        df_departures
        .groupby(df_departures["DepartureDate"].dt.to_period("M"))
        .size()
        .rename("Departures")
    )
    departures.index = departures.index.to_timestamp()

    # Date range
    min_hire = df_slice["HireDate"].min().replace(day=1)
    start_date = datetime(start_year, 1, 1)
    start_date = max(start_date, min_hire)

    if df_departures.empty:
        if not df_slice["HireDate"].empty:
            end_date = df_slice["HireDate"].max().replace(day=1)
        else:
            end_date = datetime.now().replace(day=1)
    else:
        end_date = df_departures["DepartureDate"].max().replace(day=1)

    monthly_index = pd.date_range(start_date, end_date, freq="MS")
    ts = pd.DataFrame(index=monthly_index)
    ts.index.name = "Date"

    ts["Hires"] = hires.reindex(monthly_index, fill_value=0)
    ts["Departures"] = departures.reindex(monthly_index, fill_value=0)
    ts["NetHires"] = ts["Hires"] - ts["Departures"]

    # Headcount per month (snapshot at first of month)
    headcounts = []
    for m in monthly_index:
        mask = (df_slice["HireDate"] <= m) & (
            df_slice["DepartureDate"].isna() | (df_slice["DepartureDate"] > m)
        )
        headcounts.append(int(mask.sum()))
    ts["Headcount"] = headcounts

    ts["TurnoverRate"] = np.where(
        ts["Headcount"] > 0,
        ts["Departures"] / ts["Headcount"] * 100,
        np.nan
    )

    ts = ts.reset_index()
    return ts


def compute_monthly_seasonality(ts: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average value per calendar month (1–12) for the selected metric."""
    df = ts.copy()
    df["Month"] = df["Date"].dt.month
    month_avg = (
        df.groupby("Month")[metric]
        .mean()
        .reset_index()
        .sort_values("Month")
    )
    return month_avg


def generate_departure_insights(ts: pd.DataFrame) -> str:
    """Simple auto-text insights based on departures."""
    if ts.empty or ts["Departures"].sum() == 0:
        return "Δεν υπάρχουν αποχωρήσεις στο επιλεγμένο φίλτρο και στην επιλεγμένη περίοδο."

    max_row = ts.loc[ts["Departures"].idxmax()]
    max_month = max_row["Date"].strftime("%Y-%m")
    max_value = int(max_row["Departures"])
    avg_dep = ts["Departures"].mean()

    insight = (
        f"🔍 *Insight για αποχωρήσεις (με βάση την επιλεγμένη περίοδο):*\n\n"
        f"- Ο μήνας με τις περισσότερες αποχωρήσεις είναι ο **{max_month}** "
        f"με **{max_value}** αποχωρήσεις.\n"
        f"- Ο μέσος όρος αποχωρήσεων ανά μήνα είναι περίπου **{avg_dep:.1f}**.\n"
    )
    return insight


def get_recent_ts(ts: pd.DataFrame, months: int = 36) -> pd.DataFrame:
    """Return only the most recent N months of ts for tuning."""
    if ts.empty:
        return ts
    ts_sorted = ts.sort_values("Date")
    if ts_sorted.shape[0] <= months:
        return ts_sorted
    return ts_sorted.tail(months)

# =============================================================================
# PROPHET / OPTUNA / METRICS HELPERS
# =============================================================================
def objective(trial, ts: pd.DataFrame, metric: str):
    """Optuna objective: minimize MSE of Prophet on historical data."""
    df_prophet = ts[["Date", metric]].rename(columns={"Date": "ds", metric: "y"})

    if df_prophet["y"].sum() == 0 or df_prophet["y"].dropna().shape[0] < 3:
        return np.inf

    cp_scale = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
    s_scale = trial.suggest_float("seasonality_prior_scale", 0.1, 30.0, log=True)
    weekly_seasonality_param = trial.suggest_categorical("weekly_seasonality", [True, False])

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=weekly_seasonality_param,
        daily_seasonality=False,
        changepoint_prior_scale=cp_scale,
        seasonality_prior_scale=s_scale,
    )

    m.fit(df_prophet)

    future = df_prophet[["ds"]]
    forecast = m.predict(future)

    y_true = df_prophet["y"].values
    y_pred = forecast["yhat"].values
    mse = np.mean((y_true - y_pred) ** 2)

    return mse


@st.cache_data(show_spinner="🔍 Εκτέλεση Optuna hyperparameter optimization...")
def run_optuna_tuning(ts_data: pd.DataFrame, metric: str, n_trials: int, recent_months: int = 36) -> dict:
    """Run Optuna HPO on recent subset of ts to find best Prophet params."""
    if ts_data.empty or ts_data[metric].sum() == 0 or ts_data[metric].dropna().shape[0] < 3:
        return {}

    ts_recent = get_recent_ts(ts_data, months=recent_months)
    func = lambda trial: objective(trial, ts_recent, metric)

    study = optuna.create_study(direction="minimize")
    try:
        study.optimize(func, n_trials=n_trials, show_progress_bar=False)
    except Exception:
        return {}

    return study.best_params


def calculate_metrics(df_prophet: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """Calculate MAE and MAPE on historical part of forecast."""
    df_merged = pd.merge(df_prophet, forecast, on="ds", how="inner")
    df_merged = df_merged.dropna(subset=["y", "yhat"])

    if df_merged.empty:
        return {"mae": np.nan, "mape": np.nan}

    y_true = df_merged["y"].values
    y_pred = df_merged["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)

    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan

    return {"mae": mae, "mape": mape}


def run_prophet_forecast(
    ts: pd.DataFrame,
    metric: str,
    periods: int = 12,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    weekly_seasonality: bool = False,
):
    """Run Prophet forecast on a selected metric with tuned parameters."""
    df_prophet = ts[["Date", metric]].rename(columns={"Date": "ds", metric: "y"})

    if df_prophet["y"].sum() == 0 or df_prophet["y"].dropna().shape[0] < 3:
        return None, None, None

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
    )

    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=periods, freq="MS")
    forecast = m.predict(future)

    metrics = calculate_metrics(df_prophet, forecast)

    return m, forecast, metrics


def generate_business_summary(ts: pd.DataFrame, metric: str, horizon: int, forecast: pd.DataFrame, metrics: dict) -> str:
    """Create a short HR-style narrative summary based on history and forecast."""
    if ts.empty or forecast is None or metrics is None:
        return "Δεν υπάρχουν αρκετά δεδομένα για παραγωγή περίληψης."

    total_hist = ts[metric].sum()
    avg_month = ts[metric].mean()
    first_date = ts["Date"].min().strftime("%Y-%m")
    last_date = ts["Date"].max().strftime("%Y-%m")

    last_history_date = ts["Date"].max()
    future_forecast = forecast[forecast["ds"] > last_history_date]
    total_future = future_forecast["yhat"].sum()

    mae = metrics.get("mae", np.nan)
    mape = metrics.get("mape", np.nan)

    summary = f"""
- Το μοντέλο (στην επιλεγμένη περίοδο) καλύπτει την περίοδο **{first_date} → {last_date}**.
- Συνολικά ιστορικά {metric}: **{total_hist:.0f}**, με μέσο όρο **{avg_month:.1f}** ανά μήνα.
- Για τους επόμενους **{horizon}** μήνες, το μοντέλο προβλέπει συνολικά **{total_future:.0f}** {metric}.
- Η μέση απόλυτη απόκλιση (MAE) είναι **{mae:.2f}** και το MAPE είναι περίπου **{mape:.1f}%**, 
  ένδειξη {"ικανοποιητικής" if not np.isnan(mape) and mape < 50 else "μέτριας / θορυβώδους"} προσαρμογής, 
  ειδικά αν οι τιμές είναι μικρά counts ανά μήνα.
"""
    return summary


def compute_baseline_metrics(ts: pd.DataFrame, metric: str) -> dict | None:
    """
    Naive baseline: value of same calendar position 12 months earlier (shift(12)).
    """
    if ts.empty:
        return None

    s = ts.set_index("Date")[metric].astype(float)
    if s.dropna().shape[0] < 13:
        return None

    baseline = s.shift(12)

    mask = (~baseline.isna()) & (~s.isna())
    if mask.sum() == 0:
        return None

    y_true = s[mask].values
    y_pred = baseline[mask].values

    mae = mean_absolute_error(y_true, y_pred)
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan

    return {"mae": mae, "mape": mape}


def classify_voluntary(dep_type):
    """Classify departure type as Voluntary vs Non-voluntary/Other."""
    if pd.isna(dep_type):
        return "Other / Unknown"
    s = str(dep_type).lower()

    # ΠΡΩΤΑ ελέγχουμε για involuntary, για να μην «πιαστεί» από το voluntary
    if "involuntary" in s:
        return "Non-voluntary / Other"

    # contains 'voluntary'
    if "voluntary" in s:
        return "Voluntary"

    return "Non-voluntary / Other"


# =============================================================================
# SIDEBAR – FILE UPLOAD
# =============================================================================
st.sidebar.header("📂 Δεδομένα")

uploaded_file = st.sidebar.file_uploader(
    "Upload leavers/full employee CSV (; separator, Greek encoding)",
    type=["csv"],
    help="Χρησιμοποίησε export με Company, Division, Department, Job Title, Hire Date, Departure Date κ.λπ."
)

if uploaded_file is None:
    st.info("👈 Ανέβασε πρώτα το αρχείο των εργαζομένων (CSV).")
    st.stop()

df = load_leavers_data(uploaded_file)

# =============================================================================
# SIDEBAR – FILTERS
# =============================================================================
st.sidebar.header("🔎 Φίλτρα")

company_list = sorted(df["Company"].dropna().unique()) if "Company" in df.columns else []
selected_company = st.sidebar.selectbox("Company", ["(All)"] + list(company_list))

df_filtered = df.copy()
if selected_company != "(All)":
    df_filtered = df_filtered[df_filtered["Company"] == selected_company]

division_list = sorted(df_filtered["Division"].dropna().unique()) if "Division" in df_filtered.columns else []
selected_division = st.sidebar.selectbox("Division", ["(All)"] + list(division_list))
if selected_division != "(All)":
    df_filtered = df_filtered[df_filtered["Division"] == selected_division]

dept_list = sorted(df_filtered["Department"].dropna().unique()) if "Department" in df_filtered.columns else []
selected_department = st.sidebar.selectbox("Department", ["(All)"] + list(dept_list))
if selected_department != "(All)":
    df_filtered = df_filtered[df_filtered["Department"] == selected_department]

job_list = sorted(df_filtered["JobTitle"].dropna().unique()) if "JobTitle" in df_filtered.columns else []
selected_job = st.sidebar.selectbox("Job Title", ["(All)"] + list(job_list))
if selected_job != "(All)":
    df_filtered = df_filtered[df_filtered["JobTitle"] == selected_job]

# --- JOB PROPERTY FILTER ---
job_property_list = (
    sorted(df_filtered["Job Property"].dropna().unique())
    if "Job Property" in df_filtered.columns
    else []
)

selected_job_property = st.sidebar.selectbox(
    "Job Property",
    ["(All)"] + list(job_property_list)
)

if selected_job_property != "(All)":
    df_filtered = df_filtered[df_filtered["Job Property"] == selected_job_property]


# Departure Type filter for leavers only
df_leavers_only_for_filter = (
    df_filtered.dropna(subset=["DepartureDate"]) if "DepartureDate" in df_filtered.columns else pd.DataFrame()
)
selected_dep_types = None

if not df_leavers_only_for_filter.empty and "Departure Type" in df_leavers_only_for_filter.columns:
    dep_types = sorted(df_leavers_only_for_filter["Departure Type"].dropna().unique())
    selected_dep_types = st.sidebar.multiselect(
        "Departure Type (Filters Leavers Only)",
        options=dep_types,
        default=dep_types,
        help="Επέλεξε τύπους αποχώρησης (π.χ. Voluntary, Retirement, Fixed-term)."
    )

# Early leaver threshold (months)
early_threshold = st.sidebar.slider(
    "Early leaver threshold (months)",
    min_value=3,
    max_value=24,
    value=12,
    step=1,
    help="Όριο μηνών υπηρεσίας κάτω από το οποίο θεωρούμε έναν leaver ως 'early leaver'."
)

# Toggle Optuna
use_optuna = st.sidebar.checkbox("Χρήση Optuna tuning για Prophet", value=False)

if df_filtered.empty:
    st.warning("Δεν υπάρχουν δεδομένα για τα επιλεγμένα φίλτρα. Χαλάρωσε λίγο τα φίλτρα.")
    st.stop()

st.write(
    f"- **Company:** {selected_company}  \n"
    f"- **Division:** {selected_division}  \n"
    f"- **Department:** {selected_department}  \n"
    f"- **Job Title:** {selected_job}  \n"
    f"- **Job Property:** {selected_job_property}"
)


# =============================================================================
# BUILD TIME SERIES (FULL) & DATE RANGE SLIDER
# =============================================================================
ts_full = build_monthly_time_series(df_filtered, selected_dep_types)

if ts_full.empty:
    st.warning("Δεν βρέθηκαν επαρκή χρονικά δεδομένα (Hire/Departure) για την επιλεγμένη επιλογή.")
    st.stop()

global_start = ts_full["Date"].min().to_pydatetime()
global_end = ts_full["Date"].max().to_pydatetime()

st.markdown("### 🗓️ Περίοδος Ανάλυσης Time Series")

period_preset = st.radio(
    "Γρήγορη επιλογή περιόδου:",
    ["Όλη η διαθέσιμη", "Τελευταίοι 12 μήνες", "Τελευταίοι 24 μήνες"],
    horizontal=True
)

if period_preset == "Τελευταίοι 12 μήνες":
    default_start = max(global_start, (global_end - pd.DateOffset(months=11)).to_pydatetime())
elif period_preset == "Τελευταίοι 24 μήνες":
    default_start = max(global_start, (global_end - pd.DateOffset(months=23)).to_pydatetime())
else:
    default_start = global_start

date_range = st.slider(
    "Επίλεξε την περίοδο ανάλυσης:",
    min_value=global_start,
    max_value=global_end,
    value=(default_start, global_end),
    format="YYYY-MM"
)

ts = ts_full[(ts_full["Date"] >= date_range[0]) & (ts_full["Date"] <= date_range[1])].copy()

if ts.empty:
    st.warning("Δεν υπάρχουν δεδομένα στην επιλεγμένη περίοδο. Δοκίμασε μεγαλύτερο εύρος ημερομηνιών.")
    st.stop()

# =============================================================================
# METRIC & HORIZON
# =============================================================================
metric = st.selectbox(
    "Metric για forecast:",
    ["Departures", "Hires", "NetHires"],
    index=0
)

horizon = st.slider(
    "Forecast horizon (μήνες):",
    min_value=6,
    max_value=36,
    value=12
)

# =============================================================================
# OPTUNA TUNING PARAMS
# =============================================================================
st.sidebar.header("⚙️ Prophet Tuning (Optuna HPO)")

n_trials = st.sidebar.slider(
    "Optuna Trials (Search Depth):",
    min_value=10,
    max_value=100,
    value=30,
    step=10
)

recent_months = st.sidebar.slider(
    "History used for tuning (months):",
    min_value=12,
    max_value=60,
    value=36,
    step=12,
    help="Χρονικό βάθος ιστορικών δεδομένων που θα χρησιμοποιηθεί για το Optuna."
)

if st.sidebar.button("Re-Run Optuna HPO"):
    st.cache_data.clear()

if use_optuna:
    best_params = run_optuna_tuning(ts, metric, n_trials, recent_months=recent_months)
else:
    best_params = {}

if best_params:
    st.sidebar.markdown("**✅ Best Parameters Found:**")
    st.sidebar.json({k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()})
    final_cp_scale = best_params.get("changepoint_prior_scale", 0.05)
    final_s_scale = best_params.get("seasonality_prior_scale", 10.0)
    final_weekly_seasonality = best_params.get("weekly_seasonality", False)
else:
    if use_optuna:
        st.sidebar.markdown("**⚠️ Optuna δεν επέστρεψε αποτελέσματα – χρησιμοποιούνται default τιμές.**")
    final_cp_scale = 0.05
    final_s_scale = 10.0
    final_weekly_seasonality = False

# =============================================================================
# RUN PROPHET FORECAST & BASELINE
# =============================================================================
model, forecast, metrics_model = run_prophet_forecast(
    ts,
    metric=metric,
    periods=horizon,
    changepoint_prior_scale=final_cp_scale,
    seasonality_prior_scale=final_s_scale,
    weekly_seasonality=final_weekly_seasonality
)

baseline_metrics = compute_baseline_metrics(ts, metric)

# =============================================================================
# KPI CARDS
# =============================================================================
# TOP KPIS (FIRST ROW)
# -------------------------------------------------------------------
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5, col_kpi6 = st.columns(6)

total_dep = int(ts["Departures"].sum())
total_hires = int(ts["Hires"].sum())
last_dep = int(ts.iloc[-1]["Departures"]) if not ts.empty else 0
period_label = f"{ts['Date'].min():%Y-%m} → {ts['Date'].max():%Y-%m}"

with col_kpi1:
    kpi_card(
        label="Συνολικές αποχωρήσεις",
        value=f"{total_dep:,}",
        subtitle=period_label,
        badge="Departures"
    )

with col_kpi2:
    kpi_card(
        label="Συνολικές προσλήψεις",
        value=f"{total_hires:,}",
        subtitle=period_label,
        badge="Hires"
    )

with col_kpi3:
    kpi_card(
        label="Τελευταίος μήνας - αποχωρήσεις",
        value=str(last_dep),
        subtitle=f"{ts['Date'].max():%Y-%m}" if not ts.empty else "",
        badge="Latest month"
    )

with col_kpi4:
    if metrics_model is not None and not np.isnan(metrics_model["mae"]):
        kpi_card(
            label="MAE Prophet",
            value=f"{metrics_model['mae']:.2f}",
            subtitle=f"Metric: {metric}",
            badge="Model accuracy"
        )
    else:
        kpi_card("MAE Prophet", "N/A")

with col_kpi5:
    if metrics_model is not None and not np.isnan(metrics_model["mape"]):
        kpi_card(
            label="MAPE Prophet",
            value=f"{metrics_model['mape']:.2f}%",
            subtitle=f"Metric: {metric}",
            badge="Model accuracy"
        )
    else:
        kpi_card("MAPE Prophet", "N/A")

with col_kpi6:
    if model is not None and forecast is not None:
        last_history_date = ts["Date"].max()
        future_forecast = forecast[forecast["ds"] > last_history_date]
        total_predicted_sum = int(future_forecast["yhat"].sum().round(0))
        kpi_card(
            label=f"Total Predicted {metric}",
            value=f"{total_predicted_sum:,}",
            subtitle=f"Next {horizon} months",
            badge="Forecast"
        )
    else:
        kpi_card(f"Total Predicted {metric}", "N/A")


period_label = f"{ts['Date'].min():%Y-%m} → {ts['Date'].max():%Y-%m}"
st.markdown(f"**Περίοδος δεδομένων (επιλεγμένο time series):** {period_label}")

st.markdown("---")

# =============================================================================
# LEAVERS DATA (FILTERED BY PERIOD & DEPARTURE TYPE)
# =============================================================================
# =============================================================================
# LEAVERS DATA (FILTERED BY PERIOD & DEPARTURE TYPE)
# =============================================================================
df_leavers_period = df_filtered.dropna(subset=["DepartureDate"]).copy()

if "Departure Type" in df_leavers_period.columns and selected_dep_types:
    df_leavers_period = df_leavers_period[df_leavers_period["Departure Type"].isin(selected_dep_types)]

# Note: df_leavers_period now contains the exact subset of employees that form the 
# Departures count in the 'ts' time series for the entire selected time range.

df_leavers_period = df_leavers_period[
    (df_leavers_period["DepartureDate"] >= date_range[0]) &
    (df_leavers_period["DepartureDate"] <= date_range[1])
]

early_kpi1, early_kpi2, early_kpi3 = st.columns(3)

if not df_leavers_period.empty:
    # 👉 Βάση = σύνολο αποχωρήσεων από το time series (ίδιο με ΣΥΝΟΛΙΚΕΣ ΑΠΟΧΩΡΗΣΕΙΣ)
    total_leavers_curr = total_dep  

    # Tenure
    df_leavers_period["TenureDays"] = (
        df_leavers_period["DepartureDate"] - df_leavers_period["HireDate"]
    ).dt.days
    df_leavers_period["TenureMonths"] = df_leavers_period["TenureDays"] / 30.4

    early_mask = df_leavers_period["TenureMonths"] < early_threshold
    early_count = int(early_mask.sum())
    early_pct = early_count / total_leavers_curr * 100 if total_leavers_curr > 0 else 0

    # Voluntary vs Non-voluntary (με χρήση της νέας classify_voluntary)
    df_leavers_period["VolCategory"] = df_leavers_period["Departure Type"].apply(classify_voluntary)
    vol_count = int((df_leavers_period["VolCategory"] == "Voluntary").sum())
    vol_pct = vol_count / total_leavers_curr * 100 if total_leavers_curr > 0 else 0

    with early_kpi1:
        kpi_card(
            label="Leavers (επιλεγμένη περίοδος)",
            value=f"{total_leavers_curr:,}",
            subtitle=period_label,
            badge="All leavers"
        )

    with early_kpi2:
        kpi_card(
            label=f"Early leavers (<{early_threshold} μήνες)",
            value=f"{early_count:,}",
            subtitle=f"{early_pct:.1f}% του συνόλου",
            badge="Onboarding risk"
        )

    with early_kpi3:
        kpi_card(
            label="Voluntary departures",
            value=f"{vol_count:,}",
            subtitle=f"{vol_pct:.1f}% του συνόλου",
            badge="Voluntary churn"
        )

else:
    with early_kpi1:
        kpi_card("Leavers (επιλεγμένη περίοδος)", "0")
    with early_kpi2:
        kpi_card(f"Early leavers (<{early_threshold} μήνες)", "0", subtitle="0.0%")
    with early_kpi3:
        kpi_card("Voluntary departures", "0", subtitle="0.0%")



st.markdown("---")

# =============================================================================
# TABS LAYOUT
# =============================================================================
tab_forecast, tab_salary, tab_churn, tab_hires, tab_ml = st.tabs([
    "📈 Forecast & Time Series",
    "💶 Salary & Grade",
    "🎯 Churn Profiles",
    "🧲 Hire Profiles",
    "🤖 Attrition ML (XGBoost & SHAP)"
])

# =============================================================================
# TAB 1: FORECAST & TIME SERIES
# =============================================================================
with tab_forecast:
    st.markdown("### 🧾 Business Summary (HR Narrative)")
    if model is not None and forecast is not None and metrics_model is not None:
        summary_text = generate_business_summary(ts, metric, horizon, forecast, metrics_model)
        st.markdown(summary_text)
    else:
        st.info("Δεν υπάρχουν αρκετά δεδομένα ή forecast για δημιουργία περίληψης.")

    st.markdown("### 📊 Ιστορικό Time Series (επιλεγμένη περίοδος)")
    fig_hist = px.line(
        ts,
        x="Date",
        y=metric,
        markers=True,
        title=f"Historical {metric} per Month",
    )
    fig_hist.update_layout(xaxis_title="Month", yaxis_title=metric)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### 🔮 Optimized Forecast & Seasonality")

    if model is None or forecast is None:
        st.warning(
            f"Δεν υπάρχουν αρκετά μη-μηδενικά δεδομένα για {metric} ώστε να γίνει forecast στην επιλεγμένη περίοδο. "
            "Δοκίμασε άλλο metric ή πιο γενικό φίλτρο."
        )
    else:
        st.subheader("Forecast – Ιστορικό + Μέλλον")
        fig_forecast = plot_plotly(model, forecast)
        fig_forecast.update_layout(
            xaxis_title="Month",
            yaxis_title=metric,
            legend_title="Legend",
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("Prophet Components (Trend & Seasonality)")
        comp_fig = model.plot_components(forecast)
        st.pyplot(comp_fig)

        st.subheader("📆 Monthly Seasonality (ιστορικός μέσος όρος ανά μήνα)")
        month_avg = compute_monthly_seasonality(ts, metric)
        fig_month = px.bar(
            month_avg,
            x="Month",
            y=metric,
            title=f"Average {metric} by Calendar Month",
        )
        fig_month.update_layout(xaxis_title="Month (1-12)", yaxis_title=f"Average {metric}")
        st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("### ⚖️ Model Comparison: Prophet vs Baseline")

    if metrics_model is not None and baseline_metrics is not None:
        comp_df = pd.DataFrame([
            {"Model": "Prophet", "MAE": metrics_model["mae"], "MAPE (%)": metrics_model["mape"]},
            {"Model": "Baseline (shift 12 months)", "MAE": baseline_metrics["mae"], "MAPE (%)": baseline_metrics["mape"]},
        ])
        st.dataframe(
            comp_df.style.format({"MAE": "{:.2f}", "MAPE (%)": "{:.2f}"}),
            use_container_width=True
        )
    else:
        st.info("Δεν ήταν δυνατή η σύγκριση Prophet με baseline (ανεπαρκή δεδομένα για baseline ή model).")

    if model is not None and forecast is not None:
        st.markdown("### 📋 Forecast data table")
        display_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        forecast_view = forecast[display_cols].rename(
            columns={
                "ds": "Date",
                "yhat": f"{metric}_forecast",
                "yhat_lower": "Lower CI",
                "yhat_upper": "Upper CI",
            }
        )
        st.dataframe(forecast_view, use_container_width=True)

        csv_forecast = forecast_view.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download Forecast (CSV)",
            data=csv_forecast,
            file_name=f"{metric}_forecast_{horizon}m.csv",
            mime="text/csv",
        )

# =============================================================================
# TAB 2: SALARY & GRADE
# =============================================================================
with tab_salary:
    # Επιλογή πληθυσμού ανάλογα με το metric
    if metric == "Hires":
        # Μισθοί για όσους προσλήφθηκαν στην επιλεγμένη περίοδο
        if "HireDate" in df_filtered.columns:
            df_salary = df_filtered.copy()
            df_salary = df_salary[
                (df_salary["HireDate"] >= date_range[0]) &
                (df_salary["HireDate"] <= date_range[1])
            ]
        else:
            df_salary = pd.DataFrame()
        population_label = "Hires (επιλεγμένη περίοδος)"
    else:
        # Default: Leavers όπως πριν (Departures ή NetHires)
        df_salary = df_leavers_period.copy()
        population_label = "Leavers (επιλεγμένη περίοδος)"

    st.markdown(f"### 💶 Salary & Grade Analysis για {population_label}")

    if (
        not df_salary.empty
        and "NominalSalary" in df_salary.columns
        and df_salary["NominalSalary"].notna().any()
    ):
        col1, col2 = st.columns(2)

        with col1:
            fig_sal_box = px.box(
                df_salary,
                x="JobTitle",
                y="NominalSalary",
                title=f"Distribution of Nominal Salary by Job Title ({population_label})",
            )
            fig_sal_box.update_layout(xaxis_title="Job Title", yaxis_title="Nominal Salary")
            st.plotly_chart(fig_sal_box, use_container_width=True)

        with col2:
            if "GRADE_clean" in df_salary.columns and df_salary["GRADE_clean"].notna().any():
                grade_salary = (
                    df_salary
                    .dropna(subset=["GRADE_clean"])
                    .groupby("GRADE_clean")["NominalSalary"]
                    .mean()
                    .reset_index()
                    .sort_values("GRADE_clean")
                )
                fig_grade = px.bar(
                    grade_salary,
                    x="GRADE_clean",
                    y="NominalSalary",
                    title=f"Average Nominal Salary by Grade ({population_label})",
                )
                fig_grade.update_layout(xaxis_title="Grade", yaxis_title="Average Nominal Salary")
                st.plotly_chart(fig_grade, use_container_width=True)
            else:
                st.info("Δεν υπάρχουν έγκυρες τιμές Grade για να γίνει ανάλυση.")
    else:
        st.info(f"Δεν βρέθηκαν έγκυρες τιμές στο NominalSalary για {population_label} στην επιλεγμένη περίοδο.")
    st.markdown("### 🧩 Voluntary vs Non-voluntary departures (per month)")
    if not df_leavers_period.empty:
        df_leavers_period["VolCategory"] = df_leavers_period["Departure Type"].apply(classify_voluntary)
        vol_ts = (
            df_leavers_period
            .assign(Month=lambda x: x["DepartureDate"].dt.to_period("M").dt.to_timestamp())
            .groupby(["Month", "VolCategory"])
            .size()
            .reset_index(name="Departures")
        )

        fig_vol = px.bar(
            vol_ts,
            x="Month",
            y="Departures",
            color="VolCategory",
            barmode="stack",
            title="Voluntary vs Non-voluntary departures per month"
        )
        fig_vol.update_layout(xaxis_title="Month", yaxis_title="Departures")
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("Δεν υπάρχουν αποχωρήσεις για να αναλυθούν ως voluntary/non-voluntary στην επιλεγμένη περίοδο.")

# =============================================================================
# TAB 3: CHURN PROFILES
# =============================================================================
with tab_churn:
    st.markdown("### 🎯 High-Risk Profile Summary (Leavers στην επιλεγμένη περίοδο)")
    st.caption(
        "Σύνοψη των χαρακτηριστικών των υπαλλήλων που έχουν αποχωρήσει, "
        "με έμφαση στον μέσο μηνιαίο κίνδυνο ανά ρόλο και τη σχετική βαρύτητα (churn ratio)."
    )

    if not df_leavers_period.empty:
        total_months = ts.shape[0] if ts.shape[0] > 0 else 1

        df_profile = df_leavers_period.copy().dropna(subset=["JobTitle"])

        profile_summary = df_profile.groupby("JobTitle").agg(
            Total_Departures=("JobTitle", "size"),
            Avg_Departures_per_Month=("JobTitle", lambda x: x.size / total_months),
            Avg_Salary=("NominalSalary", "mean"),
            Most_Common_Grade=("GRADE_clean", lambda x: x.mode()[0] if not x.mode().empty else "N/A"),
            Most_Common_Departure=("Departure Type", lambda x: x.mode()[0] if not x.mode().empty else "N/A"),
        ).reset_index()

        total_leavers_grand = profile_summary["Total_Departures"].sum()
        profile_summary["Churn Ratio (%)"] = (profile_summary["Total_Departures"] / total_leavers_grand) * 100

        # 👉 Numeric Churn Cost Index
        profile_summary["Churn Cost Index"] = (
            profile_summary["Total_Departures"] * profile_summary["Avg_Salary"].fillna(0)
        )

        # 👉 Κρατάμε numeric έκδοση για σωστό sort
        profile_summary["ChurnCostIndex_num"] = profile_summary["Churn Cost Index"]

        profile_summary = profile_summary.rename(
            columns={
                "Total_Departures": "Total Leavers (History)",
                "Avg_Departures_per_Month": "Avg Departs / Month",
                "Avg_Salary": "Avg Salary (Leavers)",
                "Most_Common_Grade": "Common Grade",
                "Most_Common_Departure": "Common Departure Type",
            }
        )

        # Formatting για εμφάνιση (ΔΕΝ πειράζει το numeric helper)
        profile_summary["Avg Salary (Leavers)"] = profile_summary["Avg Salary (Leavers)"].map(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A"
        )
        profile_summary["Avg Departs / Month"] = profile_summary["Avg Departs / Month"].map(
            lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A"
        )
        profile_summary["Churn Ratio (%)"] = profile_summary["Churn Ratio (%)"].map(
            lambda x: f"{x:,.1f}%" if pd.notnull(x) else "N/A"
        )
        profile_summary["Churn Cost Index"] = profile_summary["Churn Cost Index"].map(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A"
        )

        # Βασική ταξινόμηση πίνακα
        profile_summary = profile_summary.sort_values("Total Leavers (History)", ascending=False)

        top_n = st.slider("Display Top N Leaver Job Titles:", min_value=5, max_value=50, value=10)

        display_cols = [
            "JobTitle",
            "Total Leavers (History)",
            "Avg Departs / Month",
            "Churn Ratio (%)",
            "Avg Salary (Leavers)",
            "Common Grade",
            "Common Departure Type",
            "Churn Cost Index",
        ]

        st.dataframe(profile_summary[display_cols].head(top_n), use_container_width=True)

       
        top3 = profile_summary.sort_values("ChurnCostIndex_num", ascending=False).head(10)
        if not top3.empty:
            st.markdown("#### 🏅 Top 10 ρόλοι με τον υψηλότερο 'Churn Cost Index'")
            bullets = []
            for _, row in top3.iterrows():
                bullets.append(
                    f"- **{row['JobTitle']}**: {row['Total Leavers (History)']} leavers, "
                    f"Avg Salary ~ {row['Avg Salary (Leavers)']}, "
                    f"Churn Ratio {row['Churn Ratio (%)']}, "
                    f"Churn Cost Index {row['Churn Cost Index']}"
                )
            st.markdown("\n".join(bullets))

        # Download button για πλήρες profile
        csv_profile = profile_summary[display_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download Full Churn Profile (CSV)",
            data=csv_profile,
            file_name="churn_profile_jobtitles.csv",
            mime="text/csv",
        )

        st.markdown(
            "*Note: Το 'Avg Departs / Month' εκφράζει τον μέσο μηνιαίο φόρτο αντικατάστασης για κάθε ρόλο, "
            "ενώ το 'Churn Cost Index' είναι προσεγγιστικό μέτρο που συνδυάζει όγκο αποχωρήσεων και μέση μισθοδοσία.*"
        )
    else:
        st.info("Δεν υπάρχουν αποχωρήσεις για να δημιουργηθεί προφίλ κινδύνου στην επιλεγμένη περίοδο.")

# TAB 4: HIRE PROFILES (HIRES)
# =============================================================================
with tab_hires:
    st.markdown("### 🧲 High-Opportunity Profile Summary (Hires στην επιλεγμένη περίοδο)")
    st.caption(
        "Σύνοψη των χαρακτηριστικών των υπαλλήλων που προσλήφθηκαν στην επιλεγμένη περίοδο, "
        "με έμφαση στον μέσο μηνιαίο ρυθμό πρόσληψης ανά ρόλο και τη σχετική βαρύτητα (hire ratio)."
    )

    if "HireDate" not in df_filtered.columns:
        st.info("Δεν υπάρχει στήλη HireDate στο dataset.")
    else:
        df_hires_period = df_filtered.copy()
        df_hires_period = df_hires_period[
            (df_hires_period["HireDate"] >= date_range[0]) &
            (df_hires_period["HireDate"] <= date_range[1])
        ]

        if df_hires_period.empty:
            st.info("Δεν υπάρχουν προσλήψεις για να δημιουργηθεί προφίλ στην επιλεγμένη περίοδο.")
        elif "JobTitle" not in df_hires_period.columns:
            st.info("Δεν υπάρχει στήλη JobTitle για να δημιουργηθεί hire profile.")
        else:
            total_months_h = ts.shape[0] if ts.shape[0] > 0 else 1
            df_profile_h = df_hires_period.copy().dropna(subset=["JobTitle"])

            if df_profile_h.empty:
                st.info("Δεν υπάρχουν επαρκή δεδομένα hires με JobTitle.")
            else:
                # Aggregation for hires
                profile_hires = df_profile_h.groupby("JobTitle").agg(
                    Total_Hires=("JobTitle", "size"),
                    Avg_Hires_per_Month=("JobTitle", lambda x: x.size / total_months_h),
                    Avg_Salary=("NominalSalary", "mean"),
                    Most_Common_Grade=("GRADE_clean", lambda x: x.mode()[0] if not x.mode().empty else "N/A"),
                    Most_Common_JobProp=("Job Property", lambda x: x.mode()[0] if not x.mode().empty else "N/A"),
                ).reset_index()

                total_hires_grand = profile_hires["Total_Hires"].sum()
                profile_hires["Hire Ratio (%)"] = (profile_hires["Total_Hires"] / total_hires_grand) * 100

                # 👉 Numeric Hiring Cost Index (proxy)
                profile_hires["Hiring Cost Index"] = (
                    profile_hires["Total_Hires"] * profile_hires["Avg_Salary"].fillna(0)
                )
                profile_hires["HiringCostIndex_num"] = profile_hires["Hiring Cost Index"]

                # Rename columns for display
                profile_hires = profile_hires.rename(
                    columns={
                        "Total_Hires": "Total Hires (History)",
                        "Avg_Hires_per_Month": "Avg Hires / Month",
                        "Avg_Salary": "Avg Salary (Hires)",
                        "Most_Common_Grade": "Common Grade",
                        "Most_Common_JobProp": "Common Job Property",
                    }
                )

                # Formatting
                profile_hires["Avg Salary (Hires)"] = profile_hires["Avg Salary (Hires)"].map(
                    lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A"
                )
                profile_hires["Avg Hires / Month"] = profile_hires["Avg Hires / Month"].map(
                    lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A"
                )
                profile_hires["Hire Ratio (%)"] = profile_hires["Hire Ratio (%)"].map(
                    lambda x: f"{x:,.1f}%" if pd.notnull(x) else "N/A"
                )
                profile_hires["Hiring Cost Index"] = profile_hires["Hiring Cost Index"].map(
                    lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A"
                )

                # Sort by total hires
                profile_hires = profile_hires.sort_values("Total Hires (History)", ascending=False)

                top_n_h = st.slider(
                    "Display Top N Hire Job Titles:",
                    min_value=5,
                    max_value=50,
                    value=10
                )

                display_cols_h = [
                    "JobTitle",
                    "Total Hires (History)",
                    "Avg Hires / Month",
                    "Hire Ratio (%)",
                    "Avg Salary (Hires)",
                    "Common Grade",
                    "Common Job Property",
                    "Hiring Cost Index",
                ]

                st.dataframe(profile_hires[display_cols_h].head(top_n_h), use_container_width=True)

                top10_h = profile_hires.sort_values("HiringCostIndex_num", ascending=False).head(10)
                if not top10_h.empty:
                    st.markdown("#### 🏅 Top 10 ρόλοι με τον υψηλότερο 'Hiring Cost Index'")
                    bullets_h = []
                    for _, row in top10_h.iterrows():
                        bullets_h.append(
                            f"- **{row['JobTitle']}**: {row['Total Hires (History)']} hires, "
                            f"Avg Salary ~ {row['Avg Salary (Hires)']}, "
                            f"Hire Ratio {row['Hire Ratio (%)']}, "
                            f"Hiring Cost Index {row['Hiring Cost Index']}"
                        )
                    st.markdown("\n".join(bullets_h))

                # Download button για πλήρες hire profile
                csv_profile_h = profile_hires[display_cols_h].to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ Download Full Hire Profile (CSV)",
                    data=csv_profile_h,
                    file_name="hire_profile_jobtitles.csv",
                    mime="text/csv",
                )

                st.markdown(
                    "*Note: Το 'Avg Hires / Month' εκφράζει τον μέσο μηνιαίο ρυθμό προσλήψεων ανά ρόλο, "
                    "ενώ το 'Hiring Cost Index' είναι προσεγγιστικό μέτρο που συνδυάζει όγκο προσλήψεων και μέση μισθοδοσία.*"
                )

# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# 🔧 Helper: safe SHAP plotting for Streamlit + Matplotlib
# ---------------------------------------------------------------------
def shap_plot_safely(plot_func):
    """
    Wrapper to avoid SHAP/Matplotlib axis issues in Streamlit.
    - Clears any previous figures
    - Runs the SHAP plot function (which uses global plt.gcf())
    - Grabs the current figure and sends it to Streamlit
    - Clears/close again
    """
    plt.clf()
    plt.close("all")

    plot_func()  # this should call a shap.*plot(..., show=False)

    fig = plt.gcf()
    st.pyplot(fig)

    plt.clf()
    plt.close("all")


# ---------------------------------------------------------------------
# 🧠 SHAP – Advanced Explainability
# ---------------------------------------------------------------------
def render_shap_advanced(
    best_xgb_ml,
    X_train_ml: pd.DataFrame,
    X_active_ml: pd.DataFrame,
    likely_to_attrite_ml: pd.DataFrame,
    registry_numbers_ml: pd.Series,
    meta_cols: pd.DataFrame,
):
    """
    Advanced SHAP analysis & visualizations for the ML attrition model.

    Parameters
    ----------
    best_xgb_ml : trained XGBClassifier
    X_train_ml : pd.DataFrame
        Training feature matrix.
    X_active_ml : pd.DataFrame
        Features for active employees (aligned with predictions).
    likely_to_attrite_ml : pd.DataFrame
        Subset of active_df_full_ml με Predicted_Attrition == 1.
    registry_numbers_ml : pd.Series
        Registry numbers aligned on original index.
    meta_cols : pd.DataFrame
        Περιέχει First Name, Last Name, Division, Department, Job Position.
    """

    st.markdown("## 🧠 SHAP – Model Explainability")

    if best_xgb_ml is None or X_train_ml is None or X_train_ml.empty:
        st.info("Το μοντέλο ML δεν είναι διαθέσιμο ή δεν υπάρχουν δεδομένα για SHAP.")
        return

    # ------------------------------------------------------------------
    # 0️⃣ PER-EMPLOYEE TOP DRIVERS TABLE (PREDICTED LEAVERS)
    # ------------------------------------------------------------------
    st.markdown("### 📌 Predicted Attrition & Top SHAP Drivers (Active Employees)")

    num_likely_ml = 0 if likely_to_attrite_ml is None else likely_to_attrite_ml.shape[0]
    st.write(f"Employees predicted to attrite (next period): **{num_likely_ml}**")

    if num_likely_ml > 0:
        with st.spinner("Υπολογισμός τοπικών SHAP drivers για predicted leavers..."):
            X_likely = X_active_ml.loc[likely_to_attrite_ml.index].copy()
            X_likely = X_likely.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            explainer_local = shap.TreeExplainer(best_xgb_ml)
            shap_likely_all = explainer_local.shap_values(X_likely, check_additivity=False)

            # Deal with binary/multiclass: keep class 1 (Attrition=1) if list
            if isinstance(shap_likely_all, list):
                class_idx = 1 if len(shap_likely_all) > 1 else 0
                shap_likely = shap_likely_all[class_idx]
            else:
                shap_likely = shap_likely_all

            feature_names = X_likely.columns
            top_drivers_list = []

            for i in range(X_likely.shape[0]):
                row_vals = shap_likely[i]
                idx_sorted = np.argsort(-np.abs(row_vals))
                top_idx_row = idx_sorted[:3]

                drivers = []
                for j in top_idx_row:
                    fname = feature_names[j]
                    contrib = row_vals[j]
                    sign = "↑" if contrib > 0 else "↓"
                    drivers.append(f"{fname} ({sign})")

                top_drivers_list.append(", ".join(drivers))

            output_df = pd.DataFrame(
                {
                    "Registry Number": registry_numbers_ml.loc[likely_to_attrite_ml.index],
                    "Όνομα": meta_cols["First Name"].loc[likely_to_attrite_ml.index],
                    "Επώνυμο": meta_cols["Last Name"].loc[likely_to_attrite_ml.index],
                    "Division": meta_cols["Division"].loc[likely_to_attrite_ml.index],
                    "Department": meta_cols["Department"].loc[likely_to_attrite_ml.index],
                    "Job Title": meta_cols["Job Position"].loc[likely_to_attrite_ml.index],
                    "Attrition Probability": likely_to_attrite_ml["Attrition_Probability"],
                    "Top 3 SHAP Drivers": top_drivers_list,
                },
                index=likely_to_attrite_ml.index,
            ).sort_values("Attrition Probability", ascending=False)

            st.dataframe(output_df.head(50), use_container_width=True)

            csv_pred = output_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ Download Predicted Attrition Employees (CSV)",
                data=csv_pred,
                file_name="predicted_attrition_employees_with_drivers.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "Δεν υπάρχουν ενεργοί εργαζόμενοι που να προβλέπονται ως leavers με το τρέχον threshold."
        )

    # Ensure numeric train matrix
    X_train_num = X_train_ml.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # ------------------------------------------------------------------
    # 1️⃣ GLOBAL SHAP SUMMARIES (DOT + BAR)
    # ------------------------------------------------------------------
    with st.expander("📊 Global SHAP Feature Importance (Summary)", expanded=True):
        with st.spinner("Υπολογισμός SHAP values (sample)..."):
            # Sample for speed
            sample_size_ml = min(1000, X_train_num.shape[0])
            idx_sample_ml = np.random.choice(
                X_train_num.shape[0], size=sample_size_ml, replace=False
            )
            X_sample = X_train_num.iloc[idx_sample_ml].copy()

            explainer = shap.TreeExplainer(best_xgb_ml)
            shap_values_all = explainer.shap_values(X_sample, check_additivity=False)

            # Handle binary/multiclass: choose class 1 (Attrition=1) if list
            if isinstance(shap_values_all, list):
                class_idx = 1 if len(shap_values_all) > 1 else 0
                shap_values = shap_values_all[class_idx]
            else:
                shap_values = shap_values_all

            # SHAP summary dot plot
            st.markdown("#### 🔵 SHAP Summary Plot (Dot)")
            shap_plot_safely(
                lambda: shap.summary_plot(
                    shap_values,
                    X_sample,
                    feature_names=X_sample.columns,
                    plot_type="dot",
                    show=False,
                    color_bar=False,
                )
            )

            # SHAP summary bar plot
            st.markdown("#### 🟦 SHAP Summary Plot (Bar)")
            shap_plot_safely(
                lambda: shap.summary_plot(
                    shap_values,
                    X_sample,
                    feature_names=X_sample.columns,
                    plot_type="bar",
                    show=False,
                    color_bar=False,
                )
            )

    # ------------------------------------------------------------------
    # 2️⃣ DEPENDENCE PLOTS (Top feature + user-selected)
    # ------------------------------------------------------------------
    with st.expander("🔎 SHAP Dependence Plots (Top Drivers)", expanded=False):
        # Αν για κάποιο λόγο δεν υπάρχουν shap_values από πάνω, τα υπολογίζουμε ξανά
        if "shap_values" not in locals():
            sample_size_ml = min(800, X_train_num.shape[0])
            idx_sample_ml = np.random.choice(
                X_train_num.shape[0], size=sample_size_ml, replace=False
            )
            X_sample = X_train_num.iloc[idx_sample_ml].copy()

            explainer = shap.TreeExplainer(best_xgb_ml)
            shap_values_all = explainer.shap_values(X_sample, check_additivity=False)
            if isinstance(shap_values_all, list):
                class_idx = 1 if len(shap_values_all) > 1 else 0
                shap_values = shap_values_all[class_idx]
            else:
                shap_values = shap_values_all

        # Mean absolute impact per feature
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = int(np.argmax(mean_abs))
        top_feature = X_sample.columns[top_idx]

        # 2ο πιο σημαντικό feature για interaction coloring
        second_idx = int(np.argsort(-mean_abs)[1]) if X_sample.shape[1] > 1 else top_idx
        second_feature = X_sample.columns[second_idx]

        st.markdown(
            f"#### 1️⃣ Dependence plot για κύριο driver: **{top_feature}** "
            f"(color = {second_feature})"
        )

        shap_plot_safely(
            lambda: shap.dependence_plot(
                top_feature,
                shap_values,
                X_sample,
                interaction_index=second_feature,
                show=False,
            )
        )

        # Επιλογή άλλου feature από τον χρήστη
        feature_choice = st.selectbox(
            "Επίλεξε άλλο feature για dependence plot:",
            options=list(X_sample.columns),
            index=top_idx,
        )
        st.markdown(f"#### 2️⃣ Dependence plot για **{feature_choice}**")

        shap_plot_safely(
            lambda: shap.dependence_plot(
                feature_choice,
                shap_values,
                X_sample,
                show=False,
            )
        )

    # ------------------------------------------------------------------
    # 3️⃣ OPTIONAL: INTERACTION SUMMARY (cached)
    # ------------------------------------------------------------------
    with st.expander("🔗 SHAP Feature Interactions (optional)", expanded=False):

        # Μικρό toggle για να μην τρέχει καν αν δεν το θες
        enable_interactions = st.checkbox(
            "Υπολογισμός / εμφάνιση interactions (βαριά πράξη – μόνο όταν χρειάζεται)",
            value=False,
            key="show_shap_interactions",
        )

        if not enable_interactions:
            st.info("Ενεργοποίησε το checkbox για να υπολογιστούν τα SHAP interaction values.")
        else:
            try:
                # Αν δεν τα έχουμε ήδη στη session_state → υπολογισμός ΜΙΑ φορά
                if "shap_interactions" not in st.session_state:
                    with st.spinner("Υπολογισμός SHAP interaction values (μπορεί να αργήσει λίγο)..."):
                        expl_inter = shap.TreeExplainer(best_xgb_ml)
                        shap_inter_all = expl_inter.shap_interaction_values(X_sample)

                        # Αν είναι list (multiclass), παίρνουμε την Attrition=1 κλάση
                        if isinstance(shap_inter_all, list):
                            class_idx = 1 if len(shap_inter_all) > 1 else 0
                            shap_inter_to_plot = shap_inter_all[class_idx]
                        else:
                            shap_inter_to_plot = shap_inter_all

                        # Αποθήκευση στη session_state για reuse
                        st.session_state["shap_interactions"] = shap_inter_to_plot
                        st.session_state["shap_interactions_X_sample"] = X_sample
                        st.session_state["shap_interactions_model_id"] = id(best_xgb_ml)

                else:
                    # Αν έχουμε αποθηκευμένα interactions, αλλάζει μόνο αν άλλαξε μοντέλο
                    if st.session_state.get("shap_interactions_model_id") != id(best_xgb_ml):
                        st.session_state.pop("shap_interactions", None)
                        st.session_state.pop("shap_interactions_X_sample", None)
                        st.session_state.pop("shap_interactions_model_id", None)
                        st.warning(
                            "Το μοντέλο άλλαξε – χρειάζεται εκ νέου υπολογισμός interactions. "
                            "Ξανατσέκαρε το checkbox."
                        )
                        st.stop()

                    shap_inter_to_plot = st.session_state["shap_interactions"]
                    X_sample = st.session_state["shap_interactions_X_sample"]

                st.markdown("#### 🔗 SHAP Interaction Summary (Top pairwise effects)")

                shap_plot_safely(
                    lambda: shap.summary_plot(
                        shap_inter_to_plot,
                        X_sample,
                        show=False,
                    )
                )

            except TypeError:
                st.info("Η τωρινή έκδοση του SHAP δυσκολεύεται με interaction values για αυτό το μοντέλο.")
            except Exception as e:
                st.info(f"Δεν ήταν δυνατός ο υπολογισμός interaction values: {e}")

    # ------------------------------------------------------------------
    # 4️⃣ TOP DRIVERS ΓΙΑ ΣΥΓΚΕΚΡΙΜΕΝΟ ΕΡΓΑΖΟΜΕΝΟ
    # ------------------------------------------------------------------
    with st.expander("🌊 SHAP – Top drivers για συγκεκριμένο εργαζόμενο", expanded=False):
        if likely_to_attrite_ml is None or likely_to_attrite_ml.empty:
            st.info("Δεν υπάρχουν predicted leavers για ανάλυση.")
        else:
            # Registry numbers των predicted leavers
            reg_series = registry_numbers_ml.loc[likely_to_attrite_ml.index]
            reg_list = reg_series.astype(str).tolist()

            selected_reg = st.selectbox(
                "Επίλεξε Registry Number:",
                options=reg_list,
            )

            # index του επιλεγμένου
            idx_selected = reg_series[reg_series.astype(str) == selected_reg].index[0]

            # 🔹 Full Name
            first_name = meta_cols.loc[idx_selected, "First Name"]
            last_name = meta_cols.loc[idx_selected, "Last Name"]
            full_name = f"{first_name} {last_name}"

            # 1 γραμμή features για τον συγκεκριμένο εργαζόμενο
            row_X = X_active_ml.loc[[idx_selected]].copy()
            row_X = row_X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            row_X.columns = row_X.columns.astype(str)

            expl_local = shap.TreeExplainer(best_xgb_ml)
            shap_values_all = expl_local.shap_values(row_X, check_additivity=False)

            # --- Επιλογή σωστής κλάσης & 1D vector ---
            if isinstance(shap_values_all, list):
                class_idx = 1 if len(shap_values_all) > 1 else 0
                shap_row = shap_values_all[class_idx][0]
                base_val_all = expl_local.expected_value
                if isinstance(base_val_all, (list, np.ndarray)):
                    base_val = base_val_all[class_idx]
                else:
                    base_val = base_val_all
            else:
                shap_row = shap_values_all[0]
                base_val = expl_local.expected_value
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = base_val[0]

            # Φτιάχνουμε πίνακα με SHAP values
            feature_names = row_X.columns.tolist()
            df_local = pd.DataFrame({
                "Feature": feature_names,
                "SHAP": shap_row,
                "AbsSHAP": np.abs(shap_row),
            }).sort_values("AbsSHAP", ascending=False)

            top_n = st.slider("Πλήθος κορυφαίων drivers", 3, 20, 10)
            df_top = df_local.head(top_n)

            st.markdown(
                f"#### 🌊 Top {top_n} SHAP drivers για **{full_name}** (Registry: {selected_reg})"
            )

            # Horizontal bar chart (not SHAP's own plotting, so normal Matplotlib is fine)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(df_top["Feature"], df_top["SHAP"])
            ax.invert_yaxis()
            ax.set_xlabel("SHAP value (συμβολή στην πιθανότητα αποχώρησης)")
            ax.set_ylabel("Feature")
            st.pyplot(fig)
            plt.clf()
            plt.close("all")


# ---------------------------------------------------------------------
# 🤖 MAIN: Attrition ML (XGBoost & SHAP)
# ---------------------------------------------------------------------
def run_ml_prediction(uploaded_file, df_filtered):
    st.markdown("### 🤖 Attrition Prediction (Optimized XGBoost + SHAP)")
    st.caption(
        "Το μοντέλο βελτιστοποιείται ως προς **recall** (να πιάσει όσο το δυνατόν περισσότερους μελλοντικούς leavers). "
        "Η απόφαση για Attrition=1 γίνεται με ρυθμιζόμενο threshold."
    )

    # Αν αλλάξει αρχείο, αναγκαστικά retrain
    if "current_file_name" not in st.session_state or st.session_state["current_file_name"] != uploaded_file.name:
        st.session_state["current_file_name"] = uploaded_file.name
        st.session_state.pop("xgb_model", None)
        st.session_state["xgb_needs_retrain"] = True

    # --- 1. Reload raw file for ML ---
    uploaded_file.seek(0)
    df_ml = robust_read(uploaded_file)
    df_ml.columns = df_ml.columns.str.strip()

    # Rename columns (σύμφωνα με το Excel screenshot)
    rename_map = {
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
    df_ml.rename(columns=rename_map, inplace=True)

    today = datetime.today()

    # Κρατάμε μετα-πληροφορίες για output predicted leavers
    meta_cols_dict = {}
    meta_base_cols = ["Registry Number", "First Name", "Last Name", "Division", "Department", "Job Position"]
    for col in meta_base_cols:
        if col in df_ml.columns:
            meta_cols_dict[col] = df_ml[col].copy()
        else:
            meta_cols_dict[col] = pd.Series(index=df_ml.index, dtype="object")
    meta_cols = pd.DataFrame(meta_cols_dict)

    # --- 3. Date features & Attrition ---
    date_columns_ml = ["Birth Date", "Hire Date", "Departure Date"]
    for col in date_columns_ml:
        if col in df_ml.columns:
            df_ml[col] = pd.to_datetime(
                df_ml[col],
                format="%d/%m/%Y",
                errors="coerce",
                dayfirst=True,
            )

    # Calculate Tenure (θα drop-αριστεί αργότερα για leakage)
    if "Hire Date" in df_ml.columns and "Departure Date" in df_ml.columns:
        df_ml["Tenure"] = (
            df_ml["Departure Date"].fillna(today) - df_ml["Hire Date"]
        ).dt.days // 365
    else:
        st.error(
            "Οι στήλες 'Hire Date' ή 'Departure Date' δεν βρέθηκαν μετά τη μετονομασία. Αδύνατη η πρόβλεψη."
        )
        st.stop()

    # Calculate Attrition target
    if "Departure Date" in df_ml.columns:
        df_ml["Attrition"] = df_ml["Departure Date"].notnull().astype(int)
    else:
        st.error(
            "Δεν βρέθηκε στήλη 'Departure Date' στο αρχείο – δεν μπορεί να οριστεί Attrition."
        )
        st.stop()

    # --- 4. Business filters & cleaning ---
    if "Work Relationship" not in df_ml.columns:
        st.error(
            "Δεν βρέθηκε στήλη 'Σχέση Εργασίας' (Work Relationship). Δεν μπορεί να εφαρμοστεί ML attrition μοντέλο."
        )
        st.stop()

    if "Department" in df_ml.columns:
        df_ml.loc[
            df_ml["Department"].astype(str).str.contains("ΕΠΑΝΑΤΙΜΟΛΟΓΗΣΗ", na=False),
            "Departure Reason Description",
        ] = "ΜΕΤΑΦΟΡΑ ΣΕ ΑΛΛΗ ΕΤΑΙΡΕΙΑ"

    # Εταιρεία / σχέση εργασίας / voluntary only
    if "Company" in df_ml.columns:
        df_ml = df_ml[df_ml["Company"] == "ΑΛΟΥΜΥΛ Α.Ε."]

    df_ml = df_ml[df_ml["Work Relationship"] == "ΑΟΡΙΣΤΟΥ ΧΡΟΝΟΥ"]
    df_ml = df_ml[
        (df_ml["Departure Reason Description"] == "VOLUNTARY DEPARTURE")
        | (df_ml["Departure Reason Description"].isnull())
    ]

    if "Departure Date" in df_ml.columns:
        df_ml = df_ml[
            (df_ml["Departure Date"] > "2018-12-31")
            | (df_ml["Departure Date"].isnull())
        ]

    # --- 5. Salary Cleaning and Log Transform ---
    if "Nominal Salary" in df_ml.columns:
        df_ml["Nominal Salary"] = (
            df_ml["Nominal Salary"]
            .astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df_ml["Nominal Salary"] = pd.to_numeric(
            df_ml["Nominal Salary"], errors="coerce"
        )

        # Clean extreme low values (e.g. day/weekly)
        df_ml["Nominal Salary"] = np.where(
            df_ml["Nominal Salary"] < 90,
            df_ml["Nominal Salary"] * 26,  # Assuming low values are weekly/daily
            df_ml["Nominal Salary"],
        )
        df_ml["Nominal Salary"].fillna(
            df_ml["Nominal Salary"].median(), inplace=True
        )

        # Log transform για να μειώσουμε skewness
        df_ml["Log Nominal Salary"] = np.log1p(df_ml["Nominal Salary"])
    else:
        st.error("Δεν βρέθηκε στήλη Nominal Salary.")
        st.stop()

    # --- 6. Grade Cleaning ---
    if "Grade" in df_ml.columns:
        df_ml["Grade"] = (
            df_ml["Grade"]
            .astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df_ml["Grade"] = df_ml["Grade"].replace({"99999": "0.1"})
        df_ml["Grade"] = pd.to_numeric(df_ml["Grade"], errors="coerce")

    # --- 7. Basic checks for Registry Number ---
    if "Registry Number" not in df_ml.columns:
        st.error("Δεν βρέθηκε στήλη 'Αριθμός μητρώου' (Registry Number).")
        st.stop()

    # Drop rows χωρίς Registry Number & drop duplicates on Registry Number
    df_ml = df_ml.dropna(subset=["Registry Number"])
    df_ml.drop_duplicates(subset="Registry Number", inplace=True)

    # --- 8. Convert numeric fields ---
    if "Gender" in df_ml.columns:
        df_ml["Gender"] = pd.to_numeric(df_ml["Gender"], errors="coerce")

    if "Job Property" in df_ml.columns:
        df_ml["Job Property"] = df_ml["Job Property"].map(
            {"ADMINISTRATIVE": 1, "OPERATIONAL": 0}
        )

    if "Tax Category" in df_ml.columns:
        df_ml["Tax Category"] = (
            df_ml["Tax Category"]
            .astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
            .str.strip()
        )
        df_ml["Tax Category"] = pd.to_numeric(
            df_ml["Tax Category"], errors="coerce"
        )

    if "Age" in df_ml.columns:
        df_ml["Age"] = pd.to_numeric(df_ml["Age"], errors="coerce")

    # --- 9. Drop leaking / unused columns + Tenure ---
    registry_numbers_ml = df_ml["Registry Number"].copy()

    df_ml.drop(
        columns=[
            "Departure Date",
            "Hire Date",
            "Work Relationship",
            "Registry Number",
            "Departure Reason Description",
            "Birth Date",
            "Company",
            "Nominal Salary",  # Dropping original salary after log transform
            "Education Level",
            "First Name",
            "Last Name",
        ],
        inplace=True,
        errors="ignore",
    )

    # --- 10. Drop NaN σε βασικά numeric features πριν τα dummies ---
    core_numeric = [
        col
        for col in ["Gender", "Age", "Tax Category", "Log Nominal Salary", "Grade"]
        if col in df_ml.columns
    ]
    if core_numeric:
        df_ml = df_ml.dropna(subset=core_numeric)

    # --- 11. Dummies & feature matrix ---
    categorical_columns_ml = [
        col
        for col in ["City", "Division", "Job Position", "Department", "Marital Status"]
        if col in df_ml.columns
    ]

    df_transformed_ml = pd.get_dummies(
        df_ml, columns=categorical_columns_ml, drop_first=True
    )
    df_transformed_ml.columns = df_transformed_ml.columns.astype(str)

    if "Attrition" not in df_transformed_ml.columns:
        st.error(
            "Δεν μπόρεσε να εντοπιστεί η στήλη στόχου Attrition μετά τα dummies."
        )
        st.stop()

    # Drop any remaining NaNs after transformations
    df_transformed_ml.dropna(inplace=True)

    X_ml = df_transformed_ml.drop(columns=["Attrition"])
    y_ml = df_transformed_ml["Attrition"]

    # Convert bool → int
    bool_cols = X_ml.select_dtypes(include=["bool"]).columns
    X_ml[bool_cols] = X_ml[bool_cols].astype(int)

    # Ensure numeric
    X_ml = X_ml.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    st.write(
        f"📏 Παρατηρήσεις για ML: {X_ml.shape[0]} rows, {X_ml.shape[1]} features. "
        f"Attrition rate: {y_ml.mean():.2%}"
    )

    # --- 13. Train/Test split ---
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
        X_ml, y_ml, test_size=0.2, stratify=y_ml, random_state=42
    )

    # --- Imbalance Weight ---
    neg_ml = np.sum(y_train_ml == 0)
    pos_ml = np.sum(y_train_ml == 1)
    scale_pos_weight_ml = neg_ml / max(pos_ml, 1)

    st.markdown("### ⚙️ Εκπαίδευση XGBoost (βελτιστοποιημένο σε Recall)")
    if "xgb_needs_retrain" not in st.session_state:
        st.session_state["xgb_needs_retrain"] = True

    train_button = st.button("🔁 Train / update attrition model")

    if train_button:
        st.session_state["xgb_needs_retrain"] = True

    if "xgb_model" not in st.session_state or st.session_state["xgb_needs_retrain"]:
        param_grid_ml = {
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "gamma": [0, 0.1, 0.5],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0.001, 0.01, 0.1, 1],
            "reg_lambda": [0.001, 0.01, 0.1, 1],
        }

        xgb_clf = XGBClassifier(
            scale_pos_weight=scale_pos_weight_ml,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=500,
        )

        kf_ml = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        n_iter_search = 50

        with st.spinner(
            f"🔁 Εκπαίδευση XGBoost + RandomizedSearchCV ({n_iter_search} iterations, 5-fold, scoring=recall)..."
        ):
            random_search_ml = RandomizedSearchCV(
                estimator=xgb_clf,
                param_distributions=param_grid_ml,
                n_iter=n_iter_search,
                cv=kf_ml,
                scoring="recall",
                verbose=0,
                n_jobs=-1,
                refit=True,
                random_state=42,
            )

            random_search_ml.fit(X_train_ml, y_train_ml)
            best_xgb_ml = random_search_ml.best_estimator_
            best_xgb_ml.fit(X_train_ml, y_train_ml)

        st.session_state["xgb_model"] = best_xgb_ml
        st.session_state["xgb_best_params"] = random_search_ml.best_params_
        st.session_state["xgb_train_columns"] = X_train_ml.columns.tolist()
        st.session_state["xgb_needs_retrain"] = False

        st.success("✅ Νέο μοντέλο XGBoost εκπαιδεύτηκε.")

        # 🔄 Invalidate cached SHAP interactions after retrain
        for key in ["shap_interactions", "shap_interactions_X_sample", "shap_interactions_model_id"]:
            st.session_state.pop(key, None)

    else:
        best_xgb_ml = st.session_state["xgb_model"]
        X_train_ml = X_train_ml[st.session_state["xgb_train_columns"]]
        st.info("Χρησιμοποιείται το ήδη εκπαιδευμένο XGBoost μοντέλο (δεν έγινε retrain).")

    st.write("Best XGBoost parameters (last training):")
    st.json(st.session_state.get("xgb_best_params", {}))

    # --- 15. Evaluation on test set ---
    st.markdown("### 📊 Test Set Performance (με έμφαση στη Recall)")
    threshold = st.slider(
        "Decision threshold για Attrition = 1",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,
        help="Μικρότερο threshold → περισσότερους προβλεπόμενους leavers (υψηλότερη recall, χαμηλότερη precision).",
    )

    # φροντίζουμε οι στήλες του test να είναι ίδιες με του train
    X_test_ml = X_test_ml[st.session_state["xgb_train_columns"]]

    y_test_proba_ml = best_xgb_ml.predict_proba(X_test_ml)[:, 1]
    y_test_pred_ml = (y_test_proba_ml >= threshold).astype(int)

    auc_ml = roc_auc_score(y_test_ml, y_test_proba_ml)
    precision_ml = precision_score(y_test_ml, y_test_pred_ml)
    recall_val = recall_score(y_test_ml, y_test_pred_ml)
    f1_ml = f1_score(y_test_ml, y_test_pred_ml)

    st.write(f"ROC AUC: **{auc_ml:.3f}**")
    st.write(f"Recall (target metric): **{recall_val:.3f}**")
    st.write(f"Precision: **{precision_ml:.3f}**")
    st.write(f"F1 Score: **{f1_ml:.3f}**")

    st.text("Classification Report (Test Set):")
    st.text(classification_report(y_test_ml, y_test_pred_ml, digits=3))

    cm_ml = confusion_matrix(y_test_ml, y_test_pred_ml)
    fig_cm, ax_cm = plt.subplots()
    disp_ml = ConfusionMatrixDisplay(
        cm_ml, display_labels=["No Attrition", "Attrition"]
    )
    disp_ml.plot(cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)
    plt.clf()
    plt.close("all")

    # --- 16. Predict attrition for active employees ---
    active_indices_ml = df_transformed_ml[
        df_transformed_ml["Attrition"] == 0
    ].index
    active_df_full_ml = df_transformed_ml.loc[active_indices_ml].copy()
    active_df_full_ml["Registry Number"] = registry_numbers_ml.loc[active_indices_ml]

    X_active_ml = active_df_full_ml.drop(
        columns=["Attrition", "Registry Number"], errors="ignore"
    )

    # Ensure same columns as training data
    missing_cols_ml = set(st.session_state["xgb_train_columns"]) - set(X_active_ml.columns)
    for col in missing_cols_ml:
        X_active_ml[col] = 0
    X_active_ml = X_active_ml[st.session_state["xgb_train_columns"]]

    active_df_full_ml["Attrition_Probability"] = best_xgb_ml.predict_proba(
        X_active_ml
    )[:, 1]
    active_df_full_ml["Predicted_Attrition"] = (
        active_df_full_ml["Attrition_Probability"] >= threshold
    ).astype(int)

    likely_to_attrite_ml = active_df_full_ml[
        active_df_full_ml["Predicted_Attrition"] == 1
    ]

    # 🔙 Επιστρέφουμε ό,τι χρειάζεται το SHAP
    return {
        "best_xgb_ml": best_xgb_ml,
        "X_train_ml": X_train_ml,
        "X_active_ml": X_active_ml,
        "likely_to_attrite_ml": likely_to_attrite_ml,
        "registry_numbers_ml": registry_numbers_ml,
        "meta_cols": meta_cols,
    }


# ---------------------------------------------------------------------
# 📌 Χρήση μέσα στο tab
# ---------------------------------------------------------------------
with tab_ml:
    results = run_ml_prediction(uploaded_file, df_filtered)

    # Αν το training / prediction πέτυχε, κάνουμε SHAP analysis
    if results is not None:
        render_shap_advanced(**results)
