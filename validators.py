# validators.py
# -------------------------------------------------------------
# Reusable validation checks for the contact center pipeline.
# Verifies:
#   - final tables schema & uniqueness
#   - UNKNOWN members present in dimensions
#   - foreign key integrity (facts -> dims), allowing UNKNOWN
#   - support_report schema & value sanity
# -------------------------------------------------------------

import os
import re
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

# Where the pipeline writes outputs
FINAL_DIR = "data/final"
REPORT_DIR = "data/report"

# --- Expected schemas ---

# Minimal required columns for each final table
REQUIRED_COLS: Dict[str, List[str]] = {
    "agents_final": ["agent_id"],  # agent_name optional
    "contact_centers_final": ["contact_center_id", "contact_center_name"],
    "service_categories_final": ["category_id", "department"],  # category_name optional
    "interactions_final": ["interaction_id", "agent_id", "contact_center_id", "category_id"],
}

# Required columns for the aggregated report
REPORT_COLS: List[str] = [
    "month", "contact_center_name", "department",
    "total_interactions", "total_calls", "total_call_duration",
]

# YYYY-MM
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


# --- Loaders ---

def _load_csv(path: str) -> pd.DataFrame:
    """Load CSV or raise a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_csv(path)

def load_finals() -> Dict[str, pd.DataFrame]:
    """Load all final-state CSVs produced by the pipeline."""
    return {
        "agents_final": _load_csv(os.path.join(FINAL_DIR, "agents_final.csv")),
        "contact_centers_final": _load_csv(os.path.join(FINAL_DIR, "contact_centers_final.csv")),
        "service_categories_final": _load_csv(os.path.join(FINAL_DIR, "service_categories_final.csv")),
        "interactions_final": _load_csv(os.path.join(FINAL_DIR, "interactions_final.csv")),
    }

def load_report() -> pd.DataFrame:
    """Load the aggregated support_report."""
    return _load_csv(os.path.join(REPORT_DIR, "support_report.csv"))


# --- Atomic checks (each returns a list of error messages) ---

def check_required_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Ensure all required columns exist."""
    missing = [c for c in cols if c not in df.columns]
    return [f"Missing columns: {missing}"] if missing else []

def check_unique(df: pd.DataFrame, col: str) -> List[str]:
    """Ensure the identifier column has unique values."""
    dups = df[col][df[col].duplicated()].unique()
    return [f"Duplicate IDs in {col}: {dups[:10]} (showing up to 10)"] if len(dups) else []

def check_unknown_present(df: pd.DataFrame, col: str) -> List[str]:
    """Ensure the dimension contains an UNKNOWN member."""
    if "UNKNOWN" not in set(df[col].astype(str)):
        return [f"UNKNOWN member missing in {col}"]
    return []

def check_fk_integrity(facts: pd.DataFrame,
                       dim_df: pd.DataFrame,
                       fact_key: str, dim_key: str) -> List[str]:
    """
    Ensure all fact FK values are present in the dimension OR equal to 'UNKNOWN'.
    """
    facts = facts.copy()
    facts[fact_key] = facts[fact_key].astype(str)
    dim_df = dim_df.copy()
    dim_df[dim_key] = dim_df[dim_key].astype(str)
    valid = set(dim_df[dim_key])
    mask = (~facts[fact_key].isin(valid)) & (facts[fact_key] != "UNKNOWN")
    bad = facts.loc[mask, fact_key].unique()
    return [f"Orphan foreign keys in {fact_key}: {bad[:10]} (showing up to 10)"] if len(bad) else []

def check_month_format(report: pd.DataFrame) -> List[str]:
    """Validate month strings are in YYYY-MM format."""
    bad = report["month"].astype(str).apply(lambda x: not bool(MONTH_RE.match(x)))
    if bad.any():
        return [f"Bad month formats found: {report.loc[bad, 'month'].unique()[:10]}"]
    return []

def check_non_negative(report: pd.DataFrame) -> List[str]:
    """Ensure counts and durations are non-negative."""
    msgs = []
    for c in ["total_interactions", "total_calls", "total_call_duration"]:
        if (report[c] < 0).any():
            msgs.append(f"Negative values in {c}")
    return msgs

def check_calls_not_exceed_interactions(report: pd.DataFrame) -> List[str]:
    """Sanity: calls should never exceed interactions in any group."""
    bad = report["total_calls"] > report["total_interactions"]
    return ["total_calls > total_interactions for some groups"] if bad.any() else []

def check_report_internal_consistency(report: pd.DataFrame) -> List[str]:
    """Sanity: totals that should be integer-like are integer-like."""
    msgs = []
    for c in ["total_interactions", "total_calls"]:
        # allow NaN -> fill 0, then mod 1 check
        if not np.all(np.mod(report[c].fillna(0), 1) == 0):
            msgs.append(f"{c} has non-integer values")
    return msgs


# --- Orchestrator (exported) ---

def run_core_validations() -> Tuple[bool, List[str]]:
    """
    Run all validation checks against final tables and support_report.

    Returns:
      ok (bool): True if all checks pass, False otherwise
      messages (list[str]): List of validation error messages
    """
    messages: List[str] = []

    # Load data
    finals = load_finals()
    report = load_report()

    # 1) Schema checks for final tables
    for name, cols in REQUIRED_COLS.items():
        messages += [f"[{name}] {m}" for m in check_required_columns(finals[name], cols)]

    # 2) Uniqueness checks on ID columns
    messages += [f"[agents_final] {m}" for m in check_unique(finals["agents_final"], "agent_id")]
    messages += [f"[contact_centers_final] {m}" for m in check_unique(finals["contact_centers_final"], "contact_center_id")]
    messages += [f"[service_categories_final] {m}" for m in check_unique(finals["service_categories_final"], "category_id")]
    messages += [f"[interactions_final] {m}" for m in check_unique(finals["interactions_final"], "interaction_id")]

    # 3) UNKNOWN existence in dimensions
    messages += [f"[agents_final] {m}" for m in check_unknown_present(finals["agents_final"], "agent_id")]
    messages += [f"[contact_centers_final] {m}" for m in check_unknown_present(finals["contact_centers_final"], "contact_center_id")]
    messages += [f"[service_categories_final] {m}" for m in check_unknown_present(finals["service_categories_final"], "category_id")]

    # 4) Foreign key integrity (facts -> dims), allowing UNKNOWN
    facts = finals["interactions_final"]
    messages += [f"[facts->agents] {m}" for m in check_fk_integrity(facts, finals["agents_final"], "agent_id", "agent_id")]
    messages += [f"[facts->centers] {m}" for m in check_fk_integrity(facts, finals["contact_centers_final"], "contact_center_id", "contact_center_id")]
    messages += [f"[facts->categories] {m}" for m in check_fk_integrity(facts, finals["service_categories_final"], "category_id", "category_id")]

    # 5) Report schema
    messages += [f"[support_report] {m}" for m in check_required_columns(report, REPORT_COLS)]

    # 6) Report value sanity
    messages += [f"[support_report] {m}" for m in check_month_format(report)]
    messages += [f"[support_report] {m}" for m in check_non_negative(report)]
    messages += [f"[support_report] {m}" for m in check_calls_not_exceed_interactions(report)]
    messages += [f"[support_report] {m}" for m in check_report_internal_consistency(report)]

    ok = len(messages) == 0
    return ok, messages
