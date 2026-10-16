"""
NumPy 1.24+ compatibility shim.

Older versions of SHAP (and a handful of other libs) still reference the
deprecated `np.int` / `np.float` / `np.bool` / `np.object` / `np.long` aliases.
NumPy removed these in 1.24. Importing this module once, *before* any of those
libraries are imported, restores the aliases so the app works without
downgrading NumPy.

This is intentionally isolated to a single file so the workaround is easy to
find and remove the day we upgrade SHAP everywhere to a NumPy-2.x-clean
release. Do NOT add any other top-of-file monkey-patches.

Usage: import this module FIRST in app.py — before importing shap, sklearn,
xgboost, prophet, or anything that transitively imports SHAP.
"""
from __future__ import annotations

import numpy as _np

# Each shim is added only if the alias is missing — safe to import multiple
# times and safe on NumPy versions where the aliases still exist.
_DEPRECATED_ALIASES = {
    "int": int,
    "float": float,
    "bool": bool,
    "object": object,
    "long": int,
    "str": str,
    "complex": complex,
}

for _name, _builtin in _DEPRECATED_ALIASES.items():
    if not hasattr(_np, _name):
        setattr(_np, _name, _builtin)
