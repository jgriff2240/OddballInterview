# validators.py
# -------------------------------------------------------------
# Reusable validation checks for the contact center pipeline.
# Supports CSV, Parquet, and JSON inputs for finals & report.
# Exposes: run_core_validations() -> (ok: bool, messages: List[str])
# -------------------------------------------------------------

import os
import re
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np

# ---------- Flexible loaders: CSV | Parquet | JSON ----------

SUPPORTED_EXTS = {".csv", ".parquet", ".json"}

def _read_with_ext(path_with_ext: str) -> pd.DataFrame:
    ext = os.path.splitext(path_with_ext)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path_with_ext)
    elif ext == ".parquet":
        # Requires pyarrow or fastparquet
        return pd.read_parquet(path_with_ext)
    elif ext == ".json":
        # Pipeline writes records (list of objects). This also handles records.
        return pd.read_json(path_with_ext, orient="records")
    else:
        raise ValueError(f"Unsupported extension: {ext}")

def load_any(base_or_full: str) -> pd.DataFrame:
    """
    Load a DataFrame from CSV/Parquet/JSON.
    - If `base_or_full` has an extension and exists -> read it.
    - Else try <base>.csv -> <base>.parquet -> <base>.json (first that exists).
    """
    base, ext = os.path.splitext(base_or_full)
    if ext and ext.lower() in SUPPORTED_EXTS and os.path.exists(base_or_full):
        return _read_with_ext(base_or_full)

    candidates = [base_or_full + ".csv", base_or_full + ".parquet", base_or_full + ".json"]
    last_err: Optional[Exception] = None
    for p in candidates:
        if os.path.exists(p):
            try:
                return _read_with_ext(p)
            except Exception as e:
                last_err = e
    if last_err:
        raise RuntimeError(f"Found a candidate report/table but failed to read it: {last_err}")
    raise FileNotFoundError(f"Missing file: tried {', '.join(candidates)}")

# ---------- Expected schemas ----------

REQUIRED_COLS: Dict[str, List[str]] = {
    "agents_final": ["agent_id"],  # agent_name optional
    "contact_centers_final": ["contact_center_id", "contact_center_name"],
    "service_categories_final": ["category_id", "department"],  # category_name optional
    "interactions_final": ["interaction_id", "agent_id", "contact_center_id", "category_id"],
}

REPORT_COLS: List[str] = [
    "month", "contact_center_name", "department",
    "total_interactions", "total_calls", "total_call_duration",
]

# ---------- Generic checks ----------

def check_required_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    missing = [c for c in required if c not in df.columns]
    return [f"Missing required column(s): {missing}"] if missing else []

def check_unique(df: pd.DataFrame, cols: List[str], label: str) -> List[str]:
    if not all(c in df.columns for c in cols):
        return [f"[{label}] cannot check uniqueness; missing cols: {[c for c in cols if c not in df.columns]}"]
    dup = df.duplicated(subset=cols, keep=False)
    if dup.any():
        bad = df.loc[dup, cols].drop_duplicates().to_dict(orient="records")
        return [f"[{label}] duplicates present on {cols}: {bad}"]
    return []

def check_unknown_member_present(df: pd.DataFrame, id_col: str, label: str) -> List[str]:
    if id_col not in df.columns:
        return [f"[{label}] missing id column: {id_col}"]
    if "UNKNOWN" not in set(df[id_col].astype(str)):
        return [f"[{label}] must contain UNKNOWN member in {id_col}"]
    return []

def check_fk_integrity(facts: pd.DataFrame, dim: pd.DataFrame, fk_col: str, dim_id: str) -> List[str]:
    msgs: List[str] = []
    if fk_col not in facts.columns:
        return [f"facts missing FK column: {fk_col}"]
    if dim_id not in dim.columns:
        return [f"dimension missing ID column: {dim_id}"]
    fk_vals = facts[fk_col].astype(str)
    valid = set(dim[dim_id].astype(str))
    missing_mask = ~fk_vals.isin(valid)
    # Allow UNKNOWN; treat any other invalid as an error
    if missing_mask.any():
        offenders = facts.loc[missing_mask, fk_col].astype(str)
        non_unknown = offenders[offenders != "UNKNOWN"]
        if len(non_unknown) > 0:
            sample = non_unknown.head(10).tolist()
            msgs.append(f"Invalid FK values in {fk_col} (not in dimension {dim_id}). Example: {sample} "
                        f"(total invalid excluding UNKNOWN: {len(non_unknown)})")
    return msgs

# ---------- Report-specific checks ----------

_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")  # YYYY-MM

def check_month_format(report: pd.DataFrame) -> List[str]:
    if "month" not in report.columns:
        return ["report missing 'month'"]
    bad = report.loc[~report["month"].astype(str).str.match(_MONTH_RE), "month"]
    return [f"Invalid month format (expect YYYY-MM): {bad.unique().tolist()}"] if len(bad) else []

def check_non_negative(report: pd.DataFrame) -> List[str]:
    msgs: List[str] = []
    for c in ["total_interactions", "total_calls", "total_call_duration"]:
        if c not in report.columns:
            msgs.append(f"report missing column: {c}")
            continue
        if (report[c] < 0).any():
            msgs.append(f"{c} contains negative values")
    return msgs

def check_calls_not_exceed_interactions(report: pd.DataFrame) -> List[str]:
    if not all(c in report.columns for c in ["total_calls", "total_interactions"]):
        return ["cannot check calls<=interactions; missing cols"]
    bad = report[report["total_calls"] > report["total_interactions"]]
    return [f"rows where total_calls > total_interactions: {len(bad)}"] if len(bad) else []

def check_report_internal_consistency(report: pd.DataFrame) -> List[str]:
    """
    Sanity: interactions should be integer counts; calls likewise.
    """
    msgs: List[str] = []
    for c in ["total_interactions", "total_calls"]:
        if c not in report.columns:
            msgs.append(f"report missing column: {c}")
            continue
        # allow NaN -> treat as error
        if report[c].isna().any():
            msgs.append(f"{c} contains NaN")
        # values should be whole numbers
        non_int = report[c].dropna() % 1 != 0
        if non_int.any():
            msgs.append(f"{c} has non-integer values")
    return msgs

# ---------- Main entry point ----------

def run_core_validations() -> Tuple[bool, List[str]]:
    """
    Loads final tables & support_report (csv/parquet/json) and runs checks.
    Returns (ok, messages). ok==True if no errors found.
    """
    messages: List[str] = []

    # --- Load finals using flexible loader ---
    finals: Dict[str, pd.DataFrame] = {}
    base_dir = os.path.join("data", "final")
    table_bases = {
        "agents_final": os.path.join(base_dir, "agents_final"),
        "contact_centers_final": os.path.join(base_dir, "contact_centers_final"),
        "service_categories_final": os.path.join(base_dir, "service_categories_final"),
        "interactions_final": os.path.join(base_dir, "interactions_final"),
    }

    for name, base in table_bases.items():
        try:
            df = load_any(base)  # tries .csv -> .parquet -> .json
        except Exception as e:
            messages.append(f"[{name}] failed to load: {e}")
            continue
        finals[name] = df

    # --- Load report (flexible) ---
    report_base = os.path.join("data", "report", "support_report")
    try:
        report = load_any(report_base)
    except Exception as e:
        messages.append(f"[support_report] failed to load: {e}")
        report = None

    # If any critical table missing, return early with errors
    required_loaded = set(table_bases.keys())
    if set(finals.keys()) != required_loaded or report is None:
        return False, messages

    # --- 1) Required columns ---
    for name, req in REQUIRED_COLS.items():
        messages += [f"[{name}] {m}" for m in check_required_columns(finals[name], req)]
    messages += [f"[support_report] {m}" for m in check_required_columns(report, REPORT_COLS)]

    # --- 2) Uniqueness on natural keys / IDs ---
    messages += [f"[agents_final] {m}" for m in check_unique(finals["agents_final"], ["agent_id"], "agent_id")]
    messages += [f"[contact_centers_final] {m}" for m in check_unique(finals["contact_centers_final"], ["contact_center_id"], "contact_center_id")]
    messages += [f"[service_categories_final] {m}" for m in check_unique(finals["service_categories_final"], ["category_id"], "category_id")]
    messages += [f"[interactions_final] {m}" for m in check_unique(finals["interactions_final"], ["interaction_id"], "interaction_id")]

    # --- 3) UNKNOWN members present in dims ---
    messages += [f"[agents_final] {m}" for m in check_unknown_member_present(finals["agents_final"], "agent_id", "agents_final")]
    messages += [f"[contact_centers_final] {m}" for m in check_unknown_member_present(finals["contact_centers_final"], "contact_center_id", "contact_centers_final")]
    messages += [f"[service_categories_final] {m}" for m in check_unknown_member_present(finals["service_categories_final"], "category_id", "service_categories_final")]

    # --- 4) FK integrity (facts -> dims), allowing UNKNOWN ---
    messages += [f"[facts->agents] {m}" for m in check_fk_integrity(finals["interactions_final"], finals["agents_final"], "agent_id", "agent_id")]
    messages += [f"[facts->centers] {m}" for m in check_fk_integrity(finals["interactions_final"], finals["contact_centers_final"], "contact_center_id", "contact_center_id")]
    messages += [f"[facts->categories] {m}" for m in check_fk_integrity(finals["interactions_final"], finals["service_categories_final"], "category_id", "category_id")]

    # --- 5) Report schema & values ---
    messages += [f"[support_report] {m}" for m in check_month_format(report)]
    messages += [f"[support_report] {m}" for m in check_non_negative(report)]
    messages += [f"[support_report] {m}" for m in check_calls_not_exceed_interactions(report)]
    messages += [f"[support_report] {m}" for m in check_report_internal_consistency(report)]

    ok = len(messages) == 0
    return ok, messages
