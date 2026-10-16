"""All inline CSS used across the app, in one place."""

KPI_CSS = """
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
    font-weight: 700;
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

HERO_CSS = """
<style>
.app-hero {
    padding: 10px 0 18px 0;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.35);
}
.app-hero-title {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    letter-spacing: 0.03em;
    background: linear-gradient(90deg, #0f172a, #1d4ed8, #22c55e);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.app-hero-subtitle {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 0.95rem;
    margin-top: 6px;
    color: rgba(15,23,42,0.78);
}
.app-hero-badges {
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.app-hero-badge {
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.5);
    background: rgba(248, 250, 252, 0.6);
    color: rgba(30, 64, 175, 0.95);
}
@media (prefers-color-scheme: dark) {
    .app-hero-subtitle { color: rgba(226,232,240,0.9); }
    .app-hero-badge {
        background: rgba(15,23,42,0.8);
        border-color: rgba(148, 163, 184, 0.6);
        color: rgba(191, 219, 254, 0.96);
    }
}
</style>
"""

HERO_HTML = """
<div class="app-hero">
  <h1 class="app-hero-title">🧠 Workforce Intelligence Platform</h1>
  <div class="app-hero-subtitle">
    Manpower Forecast, Attrition Risk & Leavers' Salary Analytics for HR Decision Support
  </div>
  <div class="app-hero-badges">
    <span class="app-hero-badge">XGBoost · SHAP Explainability</span>
    <span class="app-hero-badge">Time-series Forecasting</span>
    <span class="app-hero-badge">HR & C&B Analytics</span>
  </div>
</div>
"""
