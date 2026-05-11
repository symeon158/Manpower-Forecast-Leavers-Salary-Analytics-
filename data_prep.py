# Workforce Intelligence Platform

A modular Streamlit app for HR forecasting, attrition prediction, and pay range analytics.

## Run

```bash
cd workforce_app
pip install -r requirements.txt
streamlit run app.py
```

> **Important:** run `streamlit run app.py` **from inside the `workforce_app/`
> directory** (the same folder that contains `app.py`). The packages
> (`config`, `data_loaders`, `forecasting`, `ml`, `ui`) live next to `app.py`
> and are imported by name. `app.py` includes a `sys.path` bootstrap so it
> still works if you launch from elsewhere, but staying inside the folder is
> the cleanest setup.

## Project Structure

```
workforce_app/
├── app.py                    # Thin orchestration layer
├── _compat.py                # NumPy 1.24+ shim for older SHAP
├── config/
│   ├── constants.py          # All magic numbers, thresholds, business rules
│   └── pay_ranges_data.py    # Static pay-range reference data
├── data_loaders/
│   ├── leavers_loader.py     # CSV reading, salary cleaning
│   └── pay_ranges_loader.py  # Pay-ranges DataFrame
├── forecasting/
│   ├── time_series.py        # Monthly TS builder, seasonality, classifiers
│   └── prophet_model.py      # Prophet + Optuna + baseline + summary
├── ml/
│   ├── data_prep.py          # ML feature engineering pipeline
│   ├── features.py           # Stateful OneHotEncoder w/ rare-cat lumping
│   ├── trainer.py            # XGBoost + RandomizedSearchCV + calibration
│   ├── evaluation.py         # Threshold sweep, bootstrap CI, calibration, segments
│   └── shap_explain.py       # SHAP utilities (global, per-row, drivers)
└── ui/
    ├── styles.py             # CSS (KPI cards, hero)
    ├── components.py         # KPI card, donut chart, safe SHAP plot
    ├── sidebar.py            # Filter widgets → FilterState dataclass
    ├── tab_forecast.py
    ├── tab_salary.py
    ├── tab_profiles.py       # Shared logic for Churn & Hires tabs
    ├── tab_ml.py             # ML orchestration + robustness panels
    ├── tab_shap.py           # SHAP UI
    ├── tab_simulator.py      # What-If simulator
    └── tab_pay_ranges.py
```

## ML Robustness Upgrades

The attrition model went through a substantial overhaul. **Behaviour changes
visible in the UI:**

| Upgrade | Why it matters |
|---|---|
| **Snapshot-date Tenure** | Previously, an active employee's tenure grew every time you re-scored them. Now tenure is computed against a fixed snapshot date — same input, same prediction, every time. |
| **Temporal train/test split** | Train on people hired earlier, test on people hired later — an honest estimate of production behaviour. Random shuffle is still available via `ML_SPLIT_STRATEGY="random"`. |
| **`OneHotEncoder` instead of `pd.get_dummies`** | New categorical values in next month's CSV are now silently passed through (zero-vector) instead of breaking the column alignment. Rare categories (<1%) are folded into `__OTHER__` to keep dimensionality sane. |
| **F1 scoring instead of `recall`** | `recall` alone has a degenerate optimum (predict 1 for everyone). F1 balances precision and recall properly. Configurable via `ML_SCORING`. |
| **Auto-derived F1-optimal threshold** | The slider now defaults to the threshold that maximises F1 on the test set, instead of a hardcoded 0.4. |
| **Probability calibration** | XGBoost + `scale_pos_weight` produces miscalibrated probabilities. The model is now wrapped in `CalibratedClassifierCV` (isotonic, prefit) so a "65%" prediction actually means 65%. Toggle via `ML_CALIBRATE_PROBABILITIES`. |
| **Bootstrap 95% CI on AUC/F1** | Tells you whether your reported metric is stable or just noise from a small test set. |
| **Reliability diagram + Brier score** | Visual + numeric proof that calibration is working (or that it's not — wide deviations from the diagonal mean the model is over-/under-confident). |
| **Learning curve** | Detect over- vs under-fit. If train ≫ val, you have variance issues; if both are flat-low, you have bias. |
| **Per-segment metrics** | Confusion matrix broken out by Job Property / Division. A model that's accurate overall but bad for one segment is dangerous to deploy. |
| **3 new engineered features** | `SalaryToGradeRatio` (compensation premium vs same-grade peers), `AgeAtHire`, `LogTenure`. Strong attrition signals that pure salary or pure tenure don't capture. |
| **`Education Level` reinstated** | Was previously dropped silently. Now encoded as a categorical feature. |
| **No median-fill on numeric NaN** | XGBoost handles missing values natively via internal split direction. Median-filling leaks training distribution into inference. |
| **Removed `use_label_encoder` from XGBClassifier** | Deprecated since xgboost 1.6. |

### Tunables (in `config/constants.py`)

```python
ML_SCORING               = "f1"            # or "average_precision", "f2", "recall"
ML_SPLIT_STRATEGY        = "temporal"       # or "random"
ML_SNAPSHOT_DATE         = None             # e.g. "2024-12-31"; None = latest departure date
ML_CALIBRATE_PROBABILITIES = True
ML_CALIBRATION_METHOD    = "isotonic"       # or "sigmoid"
RARE_CATEGORY_MIN_FRACTION = 0.01
ML_BOOTSTRAP_ITERATIONS  = 200
```

## Key changes vs. the original single-file version

### Bugs fixed
- **Removed duplicate prediction block** in the What-If simulator (the entire
  prediction logic was written twice, only the second one ran).
- **Removed `np.float = float` etc. monkey-patches.** These break with NumPy 2.x
  for downstream libraries; pinning compatible SHAP/XGBoost is the right fix.
- **Removed redundant `numpy as np` and other duplicate imports.**
- **`st.cache_data.clear()`** was nuking *all* caches just to rerun Optuna.
  Now we use `run_optuna_tuning.clear()` which only invalidates that one cache,
  so the heavy CSV loader is not re-run.
- **Removed `uploaded_file.seek(0)`** which was useless after `getvalue()`.
- The original `# 12.` numbered ML section was missing in the comments and the
  flow had hidden state bugs across reruns. The new version uses a single
  `_ensure_model_in_state` helper keyed on file id + retrain flag.

### Maintainability
- **All magic numbers live in `config/constants.py`** — change a threshold once.
- **Pay ranges data** moved to a pure data file. Analyst can edit without
  touching code.
- **Cascading sidebar filters** wrapped in a `FilterState` dataclass instead of
  10 free-floating variables.
- **Churn and Hire profile tabs** share a single `_render_profile` builder
  (was 95% duplicated code before).
- **SHAP and trainer logic** are pure functions in `ml/`; the UI only consumes
  `TrainResult`, `EvalResult`, `ShapBundle`.
- **No more globals leaked across tabs** — each tab gets exactly the data it
  needs, passed in.
- **Type hints + dataclasses** everywhere.

### Performance
- SHAP bundle is cached in `st.session_state` keyed on `id(model)` and
  invalidated on retrain.
- Headcount snapshot loop is unchanged in semantics but isolated for future
  vectorisation.
- File-id-based cache keying for the trained model so a new upload
  automatically triggers retraining without manual clearing.

## Editing checklist

| Want to change... | Edit |
|---|---|
| Optuna trial count default | `config/constants.py` → `OPTUNA_DEFAULT_TRIALS` |
| Salary low-value rule | `config/constants.py` → `LOW_SALARY_THRESHOLD` / `_MULTIPLIER` |
| Risk thresholds for the simulator | `config/constants.py` → `RISK_LOW_MAX`, `RISK_MEDIUM_MAX` |
| ML business filters (company, contract type) | `config/constants.py` → `ML_*` |
| ML scoring metric | `config/constants.py` → `ML_SCORING` (`"f1"`, `"average_precision"`, `"f2"`) |
| Train/test split strategy | `config/constants.py` → `ML_SPLIT_STRATEGY` (`"temporal"` or `"random"`) |
| Snapshot date for tenure | `config/constants.py` → `ML_SNAPSHOT_DATE` (`None` = latest departure date in data) |
| Probability calibration on/off | `config/constants.py` → `ML_CALIBRATE_PROBABILITIES` |
| Rare-category folding threshold | `config/constants.py` → `RARE_CATEGORY_MIN_FRACTION` |
| Categorical/numeric feature lists | `ml/data_prep.py` → `CATEGORICAL_COLS` / `NUMERIC_COLS` |
| Add a new tab | New `ui/tab_<name>.py` + register in `app.py` |
| Add a new pay-range row | `config/pay_ranges_data.py` |
| KPI card style | `ui/styles.py` → `KPI_CSS` |
