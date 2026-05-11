streamlit>=1.30
pandas>=2.0
# NumPy < 2.0 keeps the deprecated np.int/np.float aliases that some
# transitive deps (older SHAP/imbalanced-learn) still reference.
# If you must use NumPy 2.x, the _compat.py shim handles it.
numpy>=1.24,<2.0
prophet>=1.1.5
plotly>=5.18
matplotlib>=3.7
scikit-learn>=1.3
xgboost>=1.7
# 0.46+ has clean NumPy 1.24+ support (no np.bool / np.int references).
shap>=0.46
optuna>=3.4
