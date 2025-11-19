# 📈 Manpower Forecast & Leavers Salary Analytics  
*A Streamlit-Based HR Analytics & Workforce Forecasting Platform*

This repository contains a comprehensive **HR Analytics application** built with **Python** and **Streamlit**, designed to provide deep insights into workforce movements and separation trends across an organization.

The app integrates **time series forecasting**, **salary analysis**, and **churn profiling**, offering HR leaders a powerful tool for decision-making, risk mitigation, and workforce planning.

---

## 🚀 Key Features

### 🔮 Time Series Forecasting (Prophet + Optuna)
- Forecasts **Departures**, **Hires**, and **Net Hires**  
- Automated hyperparameter tuning via **Optuna**  
- Trend & yearly seasonality decomposition  
- Evaluation against a **12-month baseline model**  
- Exportable forecast table (CSV)

---

### 📊 Workforce Time Series
- Dynamic analysis period (full history, last 12 months, last 24 months)
- Monthly metrics:
  - **Hires**
  - **Departures**
  - **Net Hires**
  - **Headcount (snapshot)**
  - **Turnover Rate (%) — historical KPI only**
- Interactive line charts and bar charts

---

### 💶 Leavers Salary & Grade Analytics
- Salary distribution of leavers (boxplots)
- Average salary by **Grade**
- Job Title–based compensation patterns
- Support for mixed decimal separators (European locale)

---

### 🎯 Churn Profile Intelligence
- Identification of roles with highest:
  - **Leavers count**
  - **Monthly churn load**
  - **Churn Ratio (%)**
  - **Salary impact**
  - **Churn Cost Index** (high-level approximation)
- “Top 3 Critical Roles” section based on churn cost
- Exportable churn profile dataset

---

### 🧩 Voluntary vs Non-Voluntary Departures
- Automatic classification of departure types  
- Monthly stacked bar charts  
- KPIs:
  - **Voluntary %**
  - **Early leavers** (configurable threshold in months)

---

### 🗂️ Filter Controls
All analyses are dynamically filtered by:
- **Company**
- **Division**
- **Department**
- **Job Title**
- **Departure Type**
- **Date range (timeline slider)**

---

### 🧾 Business Summary (HR Narrative)
The app automatically generates a short strategic summary describing:
- Historical workforce dynamics  
- Average monthly departure patterns  
- Forecast expectations for next X months  
- Model accuracy commentary  

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Prophet**
- **Optuna**
- **Plotly**
- **Matplotlib**

---

## 📦 Installation

```bash
pip install -r requirements.txt

