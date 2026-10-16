"""
Map free-text job titles to a job family (function) and a job level (seniority).

~97% of job titles occur too rarely to get their own one-hot column, so the
encoder folds them into a single __OTHER__ bucket and the model can't tell an
accountant from a CNC supervisor. Family + level keep that information in a
form with enough rows per value to learn from.

Rules are ordered keyword regexes on the upper-cased title; the first match
wins. Edit `_FAMILY_RULES` / `_LEVEL_RULES` when new titles appear —
`unmapped_titles()` lists titles that fell through to "OTHER".
"""
from __future__ import annotations

import re

import pandas as pd

_FAMILY_RULES: list[tuple[str, str]] = [
    ("EXECUTIVE",      r"\bCEO\b|\bCHIEF\b|MANAGING DIRECTOR|^GENERAL MANAGER$"),
    ("HR",             r"\bHR\b|PAYROLL|PERSONNEL|RECRUIT|TALENT|LEARNING & DEVELOPMENT|COMPENSATION|REWARDS|EMPLOYEE RELATIONS"),
    ("R&D",            r"R&D"),
    ("MARKETING",      r"MARKETING|GRAPHIC|\b3D\b|VIDEO|CONTENT|COPYWRITER|ART EXECUTIVE|SOCIAL MEDIA|PROMOTION|WEB GROWTH|DESIGN PROMOTER"),
    ("IT",             r"\bIT\b|ΙΤ |SOFTWARE|DATABASE|SYSTEM ADMIN|NETWORK|\bBI\b|\bAI\b|ANALYTICS|INFORMATION SECURITY|SERVICEDESK|WEBSITE|DIGITAL|SYSTEMS ADMINISTRATOR|INFORMATION SYSTEMS|DATA ANALYST|DATA ADMINISTRATOR"),
    ("QUALITY_HSE",    r"QUALITY|QHSE|\bHSE\b|HEALTH & SAFETY|ENVIRO|SUSTAINAB|\bESG\b|MANAGEMENT SYSTEMS|LABORATORY|ENERGY|SAFETY"),
    ("FINANCE_LEGAL",  r"ACCOUNTANT|ACCOUNTING|ACCOUNTS|FINANCE|FINANCIAL|CASHIER|CREDIT|\bCOST\b|BUDGET|CONSOLIDATION|INVOICING|AUDIT|TREASUR|BUSINESS CONTROL|LEGAL|COMPLIANCE MANAGER"),
    ("SUPPLY_CHAIN",   r"PLANNER|PLANNING|SUPPLY CHAIN|PROCUREMENT|BUYER|\bMRP\b|MATERIALS?\b"),
    ("SALES_CUSTOMER", r"SALES|KEY ACCOUNT|CUSTOMER|SHOWROOM|COMMERCIAL|CALL CENTER|ARCHITECT CONSULTANT|BUSINESS CONSULTANT|ΠΕΛΑΤΩΝ|ALUMILCAL|ESTIMATION|PRICING|PRODUCT MANAGER|SOLAR|INTERNATIONAL BUSINESS"),
    ("MAINTENANCE",    r"MAINTEN|ELECTRIC|MECHANICAL|TECHNICIAN|MACHINERY|STEAM BOILER|ΜΗΧΑΝΟΤΕΧΝΙΤΗΣ|FACILITIES|BUILDINGS|CIVIL WORKS|ELECTROSHELLER"),
    ("ENGINEERING",    r"ENGINEER|DESIGN OFFICE|DESIGNER|\bBIM\b|ROBOTICS|AUTOMATION|INNOVATION|PROCESS IMPROVEMENT|\bLEAN\b|TECHNICAL|ΜΗΧΑΝ ΕΡΓΩΝ|PROJECT|OPTIMIZATION"),
    ("LOGISTICS",      r"WAREHOUSE|LOGISTIC|TRANSPORT|DRIVER|FORKLIFT|LOADER|LOADING|STOCK|FLEET|CUSTOMS|SPARE PARTS|DISTRIBUTION|RECEIVABLES"),
    ("PRODUCTION",     r"WORKER|ΕΡΓΑΤΗΣ|OPERATOR|PRODUCTION|SHIFT|EXTRUSION|\bCNC\b|\bDIES?\b|DIE CORRECT|FOUNDRY|ANODIZING|POWDER COATING|PACKAGING|PROFILE|ΧΕΙΡΙΣΤΗΣ|THERMAL-BREAK|\bIPD\b|MACHINING"),
    ("ADMIN_SUPPORT",  r"OFFICE EMPLOYEE|RECEPTION|FRONT DESK|ASSISTANT|HOSPITALITY|CLEANING|EU CO-FUNDED|OPERATIONS|CORPORATE SOCIAL|ACADEMY|DIRECTOR|HEAD OF"),
]

# 5 = executive … 1 = entry / operational. Checked in order; default = 2 (professional).
_LEVEL_RULES: list[tuple[float, str]] = [
    (5.0, r"\bCEO\b|\bCHIEF\b|DIRECTOR|HEAD OF|^GENERAL MANAGER$"),
    (4.0, r"\bMANAGER\b|\bLEAD\b|\bLEADER\b"),
    (3.0, r"SUPERVISOR|COORDINATOR|FOREMAN"),
    (1.0, r"WORKER|ΕΡΓΑΤΗΣ|DRIVER|OPERATOR|ASSISTANT|EMPLOYEE|AGENT|CLEANING|CASHIER|REPRESENTATIVE|"
          r"HEATER|LOADER|ΧΕΙΡΙΣΤΗΣ|ΜΗΧΑΝΟΤΕΧΝΙΤΗΣ|TECHNICIAN|ELECTRICIAN|ELECTROSHELLER|DIE CORRECTOR|"
          r"SALES SUPPORT|HOSPITALITY|FRONT DESK|RECEPTION"),
]

_FAMILY_RX = [(fam, re.compile(rx)) for fam, rx in _FAMILY_RULES]
_LEVEL_RX = [(lvl, re.compile(rx)) for lvl, rx in _LEVEL_RULES]


def _norm(title) -> str:
    return " ".join(str(title).upper().split()) if pd.notna(title) else ""


def job_family(title) -> str:
    t = _norm(title)
    if not t:
        return "UNKNOWN"
    for fam, rx in _FAMILY_RX:
        if rx.search(t):
            return fam
    return "OTHER"


def job_level(title) -> float:
    t = _norm(title)
    if not t:
        return float("nan")
    level = next((lvl for lvl, rx in _LEVEL_RX if rx.search(t)), 2.0)
    if "JUNIOR" in t:
        level = max(1.0, level - 0.5)
    if "SENIOR" in t:
        level += 0.5
    return level


def add_job_features(df: pd.DataFrame, title_col: str = "Job Position") -> pd.DataFrame:
    """Add `Job Family` (categorical) and `Job Level` (numeric) columns."""
    if title_col in df.columns:
        titles = df[title_col]
        uniq = titles.dropna().unique()
        fam = {t: job_family(t) for t in uniq}
        lvl = {t: job_level(t) for t in uniq}
        df["Job Family"] = titles.map(fam).fillna("UNKNOWN")
        df["Job Level"] = titles.map(lvl)
    return df


def unmapped_titles(titles: pd.Series) -> list[str]:
    """Titles that matched no family rule — review these when extending the rules."""
    return sorted({t for t in titles.dropna().unique() if job_family(t) == "OTHER"})
