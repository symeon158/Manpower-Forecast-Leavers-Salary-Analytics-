# **AI-Powered Workforce Intelligence Platform**  
### *Manpower Forecasting • Attrition Prediction • SHAP Explainability • Salary Analytics*

![banner](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge)
![streamlit](https://img.shields.io/badge/Streamlit-1.51-success?style=for-the-badge)
![xgboost](https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge)
![prophet](https://img.shields.io/badge/Prophet-1.2-purple?style=for-the-badge)
![shap](https://img.shields.io/badge/SHAP-Explainable_AI-red?style=for-the-badge)

---

## 📌 Overview

This project is an **AI-powered Workforce Intelligence platform** that transforms raw HR data into dynamic insights for **workforce planning, attrition prevention, and salary analytics**.  
Built with **Streamlit**, it provides a unified, fully automated environment for HR teams to analyze workforce trends, forecast leavers, and understand the drivers behind employee attrition.

The app has been designed for **HR departments**, **HR Business Partners**, **C&B analysts**, and **HR leadership**, providing actionable insights that previously required multiple systems, manual reporting, and Excel-based work.

---

## 🚀 Core Features

### 🔮 1. AI Attrition Prediction (XGBoost)
- Highly optimized XGBoost classifier focused on **recall** — detecting as many true leavers as possible.
- Clean preprocessing pipeline for HR datasets including:
  - Tenure calculation  
  - Salary normalization & log-transform  
  - Encoding of categorical features  
  - Leakage prevention (safe removal of future-only columns)
- Real-time prediction of **Attrition Probability** for all active employees.

---

### 🧠 2. Explainable AI (SHAP)
- **Global SHAP importance** (dot & bar summary plots)  
- **Per-employee explanation** with:
  - Top SHAP drivers  
  - Employee name, registry number, role, division/department  
- Optional **SHAP interaction effects** with caching to avoid repeated heavy calculations.
- **Scenario-ready explainability** supporting HRBP conversations.

---

### 📈 3. Manpower Forecasting (Prophet + Optuna)
- Monthly forecasting of:
  - Leavers  
  - Hires  
  - Net headcount  
- Integrated **hyperparameter optimization** with Optuna for improved forecast accuracy.
- Multiple error metrics:
  - **RMSE** (default & robust)  
  - MAE  
  - Bias estimate  
- Forecast visualizations with confidence intervals.

---

### 💶 4. Salary & Grade Analytics
- Salary distributions of:
  - New hires  
  - Leavers  
  - Active employees  
- Grade patterns, job property analysis, and group-level compensation structure.
- Supports C&B teams in **cost planning** and salary strategy.

---

### 🗂 5. Workforce Dashboards & KPIs
- Full HR KPI suite:
  - Turnover Rate  
  - Early Leavers (<12 months)  
  - Voluntary vs Non-voluntary exits  
  - Workforce stability indicators  
- Multi-level filters (Company → Division → Department → Job Title)
- Interactive drill-down tables and detailed managerial insights.

---

### 📤 6. Automated Outputs
- Downloadable CSVs:
  - Predicted leavers + SHAP drivers  
  - Cleaned datasets  
  - Forecast results  
- Consistent formatting for integration with Power BI & HR reporting workflows.

---

## 🎯 Business Value / Impact

### ➤ For HR Departments
- Automated headcount & attrition analysis  
- Faster, more accurate workforce planning  
- Early identification of at-risk employees  

### ➤ For HR Business Partners
- Case-by-case explainability  
- Better preparation for sensitive discussions  
- Improved support for line managers  

### ➤ For C&B Teams
- Salary structure insights (hiring vs attrition)  
- Grade alignment analysis  
- Stronger budget planning  

### ➤ For HR Leadership
- A **single source of truth** for all HR metrics  
- Predictive insights that support annual & strategic planning  
- Dashboards replacing slow/manual Excel workflows  

---

## 🛠 Tech Stack

| Category       | Tools |
|----------------|-------|
| Web App        | **Streamlit** |
| ML / AI        | **XGBoost**, **scikit-learn**, **SHAP** |
| Forecasting    | **Prophet**, **Optuna** |
| Visualization  | **Plotly**, **Matplotlib**, **Seaborn** |
| Data           | **Pandas**, **NumPy**, **OpenPyXL** |
| Deployment     | **Streamlit Cloud** |

---

## 📦 Project Structure

