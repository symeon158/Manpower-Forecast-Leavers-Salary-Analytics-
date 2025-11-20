import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import optuna


# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Manpower Time Series & Leavers Salary",
    layout="wide"
)

st.title("📈 Manpower Forecast & Leavers Salary Analytics (Optimized)")
st.caption(
    "Time series forecasting με seasonality, salary insights, Optuna tuning, HR KPIs "
    "και σύγκριση Prophet vs baseline."
)

# -------------------------------------------------------------------
# HELPERS - DATA LOADING & PREPROCESSING
# -------------------------------------------------------------------

@st.cache_data
def load_leavers_data(uploaded_file: bytes) -> pd.DataFrame:
    """Load and preprocess the leavers/full employee dataset."""
    df = pd.read_csv(uploaded_file, encoding="iso-8859-7", sep=";")

    # Normalize column names
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
        },
        errors="ignore"
    )
    for col in df.columns:
        if "κωδ" in col and "εργαζ" in col:
            employee_code_col = col
            break

    # If detected, drop duplicates keeping the latest row
    if employee_code_col:
        df = df.sort_values(by=df.columns.tolist()).drop_duplicates(subset=[employee_code_col], keep="last")
        if "JobTitle" in df.columns:
            df["JobTitle"] = df["JobTitle"].replace("ΕΡΓΑΤΗΣ", "WORKER")

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
        df["GRADE_clean"] = pd.to_numeric(df["GRADE_clean"], errors="coerce")

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

# -------------------------------------------------------------------
# PROPHET / OPTUNA / METRICS
# -------------------------------------------------------------------

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
    Αποφεύγουμε .loc[Timestamp] για να μην έχουμε KeyError.
    """
    if ts.empty:
        return None

    s = ts.set_index("Date")[metric].astype(float)
    # Χρειαζόμαστε τουλάχιστον 13 σημεία για να έχει νόημα η shift(12)
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
    """Classify departure type as Voluntary vs Non-voluntary/Other (simple heuristic)."""
    if pd.isna(dep_type):
        return "Other / Unknown"
    s = str(dep_type).lower()
    if "vol" in s:  # matches 'Voluntary', etc.
        return "Voluntary"
    return "Non-voluntary / Other"

# -------------------------------------------------------------------
# SIDEBAR – FILE UPLOAD
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# SIDEBAR – FILTERS
# -------------------------------------------------------------------
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
use_optuna = st.sidebar.checkbox("Χρήση Optuna tuning για Prophet", value=True)

if df_filtered.empty:
    st.warning("Δεν υπάρχουν δεδομένα για τα επιλεγμένα φίλτρα. Χαλάρωσε λίγο τα φίλτρα.")
    st.stop()

st.markdown("### 🎯 Τρέχουσα Επιλογή")
st.write(
    f"- **Company:** {selected_company}  \n"
    f"- **Division:** {selected_division}  \n"
    f"- **Department:** {selected_department}  \n"
    f"- **Job Title:** {selected_job}"
)

# -------------------------------------------------------------------
# BUILD TIME SERIES (FULL) & DATE RANGE SLIDER
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# BUILD TIME SERIES (FULL) & DATE RANGE SLIDER
# -------------------------------------------------------------------
ts_full = build_monthly_time_series(df_filtered, selected_dep_types)

if ts_full.empty:
    st.warning("Δεν βρέθηκαν επαρκή χρονικά δεδομένα (Hire/Departure) για την επιλεγμένη επιλογή.")
    st.stop()

# --- CORRECTION: Convert Timestamps to standard datetime objects ---
global_start = ts_full["Date"].min().to_pydatetime()
global_end = ts_full["Date"].max().to_pydatetime()

st.markdown("### 🗓️ Περίοδος Ανάλυσης Time Series")

period_preset = st.radio(
    "Γρήγορη επιλογή περιόδου:",
    ["Όλη η διαθέσιμη", "Τελευταίοι 12 μήνες", "Τελευταίοι 24 μήνες"],
    horizontal=True
)

if period_preset == "Τελευταίοι 12 μήνες":
    # Ensure default_start is also converted to standard datetime if it's a Timestamp
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

# -------------------------------------------------------------------
# METRIC & HORIZON
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# OPTUNA TUNING PARAMS
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# RUN PROPHET FORECAST & BASELINE
# -------------------------------------------------------------------
model, forecast, metrics_model = run_prophet_forecast(
    ts,
    metric=metric,
    periods=horizon,
    changepoint_prior_scale=final_cp_scale,
    seasonality_prior_scale=final_s_scale,
    weekly_seasonality=final_weekly_seasonality
)

baseline_metrics = compute_baseline_metrics(ts, metric)


df_leavers_period = df_filtered.dropna(subset=["DepartureDate"]).copy()

if "Departure Type" in df_leavers_period.columns and selected_dep_types:
    df_leavers_period = df_leavers_period[df_leavers_period["Departure Type"].isin(selected_dep_types)]

df_leavers_period = df_leavers_period[
    (df_leavers_period["DepartureDate"] >= date_range[0]) &
    (df_leavers_period["DepartureDate"] <= date_range[1])
]

# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# LEAVERS DATA (FILTERED BY PERIOD & DEPARTURE TYPE)
# -------------------------------------------------------------------
# Base leavers set
# -------------------------------------------------------------------
# LEAVERS DATA (FILTERED BY PERIOD & DEPARTURE TYPE)
# -------------------------------------------------------------------


early_kpi1, early_kpi2, early_kpi3 = st.columns(3)

if not df_leavers_period.empty:

    df_leavers_period["TenureDays"] = (df_leavers_period["DepartureDate"] - df_leavers_period["HireDate"]).dt.days
    df_leavers_period["TenureMonths"] = df_leavers_period["TenureDays"] / 30.4

    early_mask = df_leavers_period["TenureMonths"] < early_threshold
    early_count = int(early_mask.sum())
    total_leavers_curr = int(df_leavers_period.shape[0])
    early_pct = early_count / total_leavers_curr * 100 if total_leavers_curr > 0 else 0

    df_leavers_period["VolCategory"] = df_leavers_period["Departure Type"].apply(classify_voluntary)
    vol_count = int((df_leavers_period["VolCategory"] == "Voluntary").sum())
    vol_pct = vol_count / total_leavers_curr * 100 if total_leavers_curr > 0 else 0

    with early_kpi1:
        kpi_card(
            label="Leavers (επιλεγμένη περίοδος)",
            value=f"{total_dep:,}",
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

# -------------------------------------------------------------------
# TABS LAYOUT
# -------------------------------------------------------------------
tab_forecast, tab_salary, tab_churn, tab_raw = st.tabs([
    "📈 Forecast & Time Series",
    "💶 Salary & Grade",
    "🎯 Churn Profiles",
    "📊 Raw Data"
])

# -------------------------------------------------------------------
# TAB 1: FORECAST & TIME SERIES
# -------------------------------------------------------------------
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

    # Forecast table + download
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

# -------------------------------------------------------------------
# TAB 2: SALARY & GRADE
# -------------------------------------------------------------------
with tab_salary:
    st.markdown("### 💶 Salary & Grade Analysis για όσους αποχωρούν (στην επιλεγμένη περίοδο)")

    if (
        not df_leavers_period.empty
        and "NominalSalary" in df_leavers_period.columns
        and df_leavers_period["NominalSalary"].notna().any()
    ):
        col1, col2 = st.columns(2)

        with col1:
            fig_sal_box = px.box(
                df_leavers_period,
                x="JobTitle",
                y="NominalSalary",
                title="Distribution of Nominal Salary by Job Title (Leavers)",
            )
            fig_sal_box.update_layout(xaxis_title="Job Title", yaxis_title="Nominal Salary")
            st.plotly_chart(fig_sal_box, use_container_width=True)

        with col2:
            if "GRADE_clean" in df_leavers_period.columns and df_leavers_period["GRADE_clean"].notna().any():
                grade_salary = (
                    df_leavers_period
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
                    title="Average Nominal Salary by Grade (Leavers)",
                )
                fig_grade.update_layout(xaxis_title="Grade", yaxis_title="Average Nominal Salary")
                st.plotly_chart(fig_grade, use_container_width=True)
            else:
                st.info("Δεν υπάρχουν έγκυρες τιμές Grade για να γίνει ανάλυση.")
    else:
        st.info("Δεν βρέθηκαν έγκυρες τιμές στο NominalSalary για τους leavers στην επιλεγμένη περίοδο.")

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

# -------------------------------------------------------------------
# TAB 3: CHURN PROFILES
# -------------------------------------------------------------------
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

        top_n = st.slider("Display Top N Leaver Job Titles:", min_value=5, max_value=25, value=10)

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

        # 🏅 Τώρα το Top 3 χρησιμοποιεί το numeric ChurnCostIndex_num
        top3 = profile_summary.sort_values("ChurnCostIndex_num", ascending=False).head(3)
        if not top3.empty:
            st.markdown("#### 🏅 Top 3 ρόλοι με τον υψηλότερο 'Churn Cost Index'")
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


# -------------------------------------------------------------------
# TAB 4: RAW DATA
# -------------------------------------------------------------------
with tab_raw:
    st.markdown("### 📊 Raw filtered data (employees)")
    st.dataframe(df_filtered, use_container_width=True)

    csv_raw = df_filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Download filtered dataset (CSV)",
        data=csv_raw,
        file_name="filtered_employees.csv",
        mime="text/csv",
    )

    if not df_leavers_period.empty:
        st.markdown("### 📊 Raw leavers data (filtered by period & departure type)")
        st.dataframe(df_leavers_period, use_container_width=True)

        csv_leavers = df_leavers_period.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download leavers dataset (CSV)",
            data=csv_leavers,
            file_name="filtered_leavers.csv",
            mime="text/csv",
        )
