"""Shared chart colors (validated categorical slots, status palette) for light / dark mode.

Categorical slots 1–2 pass the palette validator (CVD ΔE ≥ 24, contrast ≥ 3:1)
in both modes. Status colors are reserved for state (good / warning / serious /
critical) and are always paired with an icon + label, never used for a series.
"""
from __future__ import annotations

import streamlit as st

PALETTE = {
    "light": {"series": ["#2a78d6", "#eb6834"], "ref": "#52514e", "muted": "#8a8984",
              "grid": "rgba(0,0,0,0.08)"},
    "dark":  {"series": ["#3987e5", "#d95926"], "ref": "#c3c2b7", "muted": "#8f8e86",
              "grid": "rgba(255,255,255,0.10)"},
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_ICON = {"good": "✅", "warning": "⚠️", "serious": "🔶", "critical": "⛔", "neutral": "•"}


def theme() -> dict:
    try:
        mode = "dark" if st.context.theme.type == "dark" else "light"
    except Exception:   # older Streamlit: no theme context
        mode = "light"
    return PALETTE[mode]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
