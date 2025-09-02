# `validators.py` — Line-by-Line, Syntax-Aware Commentary

This document explains the **validators** module line by line with extra detail on **purpose**, **syntax**, and **behavior**. The module provides reusable data validation utilities for your contact-center pipeline and a single entry point `run_core_validations()` that loads outputs and runs checks.

---

## File Header

```python
# validators.py
# -------------------------------------------------------------
# Reusable validation checks for the contact center pipeline.
# Supports CSV, Parquet, and JSON inputs for finals & report.
# Exposes: run_core_validations() -> (ok: bool, messages: List[str])
# -------------------------------------------------------------
```
- File banner describing responsibilities and the main function signature.  
- Pure comments (ignored by Python).

---

## Imports

```python
import os
import re
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
```
- `import os` — Standard library module for path ops (`os.path.splitext`, `os.path.exists`, `os.path.join`).  
- `import re` — Regular expressions (used for month format validation).  
- `from typing import ...` — **Type hints** to document function signatures and structures.  
- `import pandas as pd` — Pandas for DataFrames, IO, and manipulation.  
- `import numpy as np` — NumPy available (not heavily used here, but common in data checks).

---

## Flexible Loaders (CSV | Parquet | JSON)

```python
SUPPORTED_EXTS = {".csv", ".parquet", ".json"}  # whitelist of extensions we know how to read
```
- A **set** of supported file extensions; fast membership checks (`in`).

```python
def _read_with_ext(path_with_ext: str) -> pd.DataFrame:
    """
    Read a file based on its extension.
    - .csv      → pandas.read_csv
    - .parquet  → pandas.read_parquet
    - .json     → pandas.read_json (records format)
    """
```
- Private helper (leading `_`) that dispatches to the proper Pandas reader based on extension.  
- The docstring documents expected behavior.

```python
    ext = os.path.splitext(path_with_ext)[1].lower()  # grab extension like ".csv"
```
- `os.path.splitext(path)` returns `(root, ext)`. Taking `[1]` gets the extension; `.lower()` normalizes case.

```python
    if ext == ".csv":
        return pd.read_csv(path_with_ext)
    elif ext == ".parquet":
        # Requires either pyarrow or fastparquet installed in the environment
        return pd.read_parquet(path_with_ext)
    elif ext == ".json":
        # Pipeline writes JSON as a list of record objects
        return pd.read_json(path_with_ext, orient="records")
    else:
        raise ValueError(f"Unsupported extension: {ext}")
```
- Conditional branches for each known extension; for JSON, explicitly use `orient="records"` to parse arrays of objects.  
- Raises a clear error for unsupported formats.

```python
def load_any(base_or_full: str) -> pd.DataFrame:
    """
    Load a DataFrame from CSV/Parquet/JSON.
    - If a full filename with extension is given and exists → read it.
    - Else, try <base>.csv → <base>.parquet → <base>.json (first one that exists).
    """
```
- Public utility to load a dataset either from an explicit path with extension or by trying known extensions in order.  
- Returns a DataFrame or raises a descriptive error.

```python
    base, ext = os.path.splitext(base_or_full)
```
- Splits the provided path into `(base, ext)`; `ext` includes the dot when present.

```python
    # Case 1: user gave a file with extension explicitly
    if ext and ext.lower() in SUPPORTED_EXTS and os.path.exists(base_or_full):
        return _read_with_ext(base_or_full)
```
- If caller passed ".../file.csv" (etc.) and it exists, read immediately using the helper.

```python
    # Case 2: try the known suffixes in order of preference
    candidates = [base_or_full + ".csv", base_or_full + ".parquet", base_or_full + ".json"]
    last_err: Optional[Exception] = None
```
- If no extension, build candidate paths. `last_err` stores the most recent exception (if reads fail).

```python
    for p in candidates:
        if os.path.exists(p):
            try:
                return _read_with_ext(p)
            except Exception as e:
                # Save the last error so we can report it later
                last_err = e
```
- Iterate candidates; if the file exists, try to read it; keep the error and continue if it fails.

```python
    if last_err:
        # Found a file but it failed to read
        raise RuntimeError(f"Found a candidate report/table but failed to read it: {last_err}")
```
- If at least one candidate existed but all attempts failed, raise `RuntimeError` with the last error message.

```python
    # Nothing found at all
    raise FileNotFoundError(f"Missing file: tried {', '.join(candidates)}")
```
- If no candidate file exists, raise a clear `FileNotFoundError` listing attempts.

## Expected Schemas

```python
REQUIRED_COLS: Dict[str, List[str]] = {
    "agents_final": ["agent_id"],  # agent_name is optional
    "contact_centers_final": ["contact_center_id", "contact_center_name"],
    "service_categories_final": ["category_id", "department"],  # category_name optional
    "interactions_final": ["interaction_id", "agent_id", "contact_center_id", "category_id"],
}
```
- **Minimum required columns** for each final output table.  
- Used by validation to assert schema completeness. Hints indicate optional fields.

```python
REPORT_COLS: List[str] = [
    "month", "contact_center_name", "department",
    "total_interactions", "total_calls", "total_call_duration",
]
```
- Expected column names for `support_report` used in downstream analytics.

## Generic Checks

```python
def check_required_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    """
    Verify required columns exist in the DataFrame.
    Return a list of error messages if any are missing.
    """
```
- Verifies presence of mandatory columns; returns a **list of string messages** (empty if OK).

```python
    missing = [c for c in required if c not in df.columns]
    return [f"Missing required column(s): {missing}"] if missing else []
```
- List comprehension to collect missing names; one concise message if any are missing.

```python
def check_unique(df: pd.DataFrame, cols: List[str], label: str) -> List[str]:
    """
    Ensure the given set of columns uniquely identify rows.
    Returns error messages if duplicates are found.
    """
```
- Validates uniqueness of a set of columns (e.g., primary key).

```python
    if not all(c in df.columns for c in cols):
        return [f"[{label}] cannot check uniqueness; missing cols: {[c for c in cols if c not in df.columns]}"]
```
- If required columns aren’t present, return a descriptive message immediately.

```python
    dup = df.duplicated(subset=cols, keep=False)  # mark duplicates
```
- `duplicated(..., keep=False)` marks **all** rows that belong to a duplicate group.

```python
    if dup.any():
        bad = df.loc[dup, cols].drop_duplicates().to_dict(orient="records")
        return [f"[{label}] duplicates present on {cols}: {bad}"]
    return []
```
- If duplicates exist, extract unique offending key combinations and return them as a message.  
- Otherwise, return empty list (no problems).

```python
def check_unknown_member_present(df: pd.DataFrame, id_col: str, label: str) -> List[str]:
    """
    Validate that the dimension table contains a special UNKNOWN row
    (used for repairing bad/missing foreign keys).
    """
```
- Confirms the presence of the `"UNKNOWN"` member used to repair referential integrity.

```python
    if id_col not in df.columns:
        return [f"[{label}] missing id column: {id_col}"]
    if "UNKNOWN" not in set(df[id_col].astype(str)):
        return [f"[{label}] must contain UNKNOWN member in {id_col}"]
    return []
```
- If ID column missing, report. If no `"UNKNOWN"` row, report. Else, return OK (empty list).

```python
def check_fk_integrity(facts: pd.DataFrame, dim: pd.DataFrame, fk_col: str, dim_id: str) -> List[str]:
    """
    Verify that foreign key values in the fact table exist in the dimension.
    - Allows UNKNOWN as a fallback.
    - Reports invalid values (not found in the dimension).
    """
```
- Validates that fact foreign keys match valid IDs in the dimension table. `"UNKNOWN"` is allowed.

```python
    msgs: List[str] = []
    if fk_col not in facts.columns:
        return [f"facts missing FK column: {fk_col}"]
    if dim_id not in dim.columns:
        return [f"dimension missing ID column: {dim_id}"]
```
- Early exits if required columns are missing, returning single-item message lists.

```python
    fk_vals = facts[fk_col].astype(str)
    valid = set(dim[dim_id].astype(str))
    missing_mask = ~fk_vals.isin(valid)
```
- Normalize to string dtype for consistent comparisons; build a boolean mask of invalid FKs.

```python
    if missing_mask.any():
        offenders = facts.loc[missing_mask, fk_col].astype(str)
        non_unknown = offenders[offenders != "UNKNOWN"]  # exclude the allowed UNKNOWN
        if len(non_unknown) > 0:
            sample = non_unknown.head(10).tolist()
            msgs.append(
                f"Invalid FK values in {fk_col} (not in dimension {dim_id}). Example: {sample} "
                f"(total invalid excluding UNKNOWN: {len(non_unknown)})"
            )
    return msgs
```
- If there are invalid FKs (excluding `"UNKNOWN"`), emit a message with a small sample and a count.  
- Return the list of messages (possibly empty).

## Report-Specific Checks

```python
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
```
- Compiled regex for months in `YYYY-MM` format. `^` anchors start, `$` anchors end.

```python
def check_month_format(report: pd.DataFrame) -> List[str]:
    """Ensure the month column exists and follows YYYY-MM format."""
```
- Validates presence and format of the `month` column.

```python
    if "month" not in report.columns:
        return ["report missing 'month'"]
    bad = report.loc[~report["month"].astype(str).str.match(_MONTH_RE), "month"]
    return [f"Invalid month format (expect YYYY-MM): {bad.unique().tolist()}"] if len(bad) else []
```
- If column missing, report. Otherwise, find non-matching rows and report their unique values.

```python
def check_non_negative(report: pd.DataFrame) -> List[str]:
    """Check numeric metrics are non-negative."""
```
- Ensures `total_interactions`, `total_calls`, and `total_call_duration` are ≥ 0.

```python
    msgs: List[str] = []
    for c in ["total_interactions", "total_calls", "total_call_duration"]:
        if c not in report.columns:
            msgs.append(f"report missing column: {c}")
            continue
        if (report[c] < 0).any():
            msgs.append(f"{c} contains negative values")
    return msgs
```
- Iterates columns, guarding for absence; for present ones, flags any negative values.

```python
def check_calls_not_exceed_interactions(report: pd.DataFrame) -> List[str]:
    """Validate logical rule: calls ≤ interactions."""
```
- Sanity constraint: number of calls cannot exceed total interactions.

```python
    if not all(c in report.columns for c in ["total_calls", "total_interactions"]):
        return ["cannot check calls<=interactions; missing cols"]
    bad = report[report["total_calls"] > report["total_interactions"]]
    return [f"rows where total_calls > total_interactions: {len(bad)}"] if len(bad) else []
```
- If needed columns missing, return a single message. Else count rows violating the rule and report if any.

```python
def check_report_internal_consistency(report: pd.DataFrame) -> List[str]:
    """
    Extra sanity: interactions and calls should be whole numbers, not NaN or floats.
    """
```
- Ensures integer-like counts and no NaNs in `total_interactions` and `total_calls` columns.

```python
    msgs: List[str] = []
    for c in ["total_interactions", "total_calls"]:
        if c not in report.columns:
            msgs.append(f"report missing column: {c}")
            continue
        if report[c].isna().any():
            msgs.append(f"{c} contains NaN")
        non_int = report[c].dropna() % 1 != 0
        if non_int.any():
            msgs.append(f"{c} has non-integer values")
    return msgs
```
- For each column: flag missing, NaN presence, and non-integer remainders (value % 1 ≠ 0).

## Main Entry Point

```python
def run_core_validations() -> Tuple[bool, List[str]]:
    """
    Loads final tables & support_report (csv/parquet/json) and runs checks.
    Returns (ok, messages). ok == True if no errors found.
    """
```
- The orchestrator: loads all final tables and the report, runs validations, and returns `(ok, messages)`.

```python
    messages: List[str] = []
```
- Collects all validation messages in a list.

```python
    # --- Load finals using flexible loader ---
    finals: Dict[str, pd.DataFrame] = {}
    base_dir = os.path.join("data", "final")
    table_bases = {
        "agents_final": os.path.join(base_dir, "agents_final"),
        "contact_centers_final": os.path.join(base_dir, "contact_centers_final"),
        "service_categories_final": os.path.join(base_dir, "service_categories_final"),
        "interactions_final": os.path.join(base_dir, "interactions_final"),
    }
```
- Prepares a mapping of final table names → base paths (no extensions).

```python
    for name, base in table_bases.items():
        try:
            df = load_any(base)  # tries .csv -> .parquet -> .json
        except Exception as e:
            messages.append(f"[{name}] failed to load: {e}")
            continue
        finals[name] = df
```
- Attempts to load each final table via `load_any`; on failure, append a message and continue.  
- Successful DataFrames are stored in `finals` under their names.

```python
    # --- Load report (flexible) ---
    report_base = os.path.join("data", "report", "support_report")
    try:
        report = load_any(report_base)
    except Exception as e:
        messages.append(f"[support_report] failed to load: {e}")
        report = None
```
- Loads the aggregated report similarly; if it fails, set `report=None` and record a message.

```python
    # If any critical table/report is missing, return early
    required_loaded = set(table_bases.keys())
    if set(finals.keys()) != required_loaded or report is None:
        return False, messages
```
- Early exit: all finals plus the report must be present to proceed. If not, return failure with messages.

```python
    # --- 1) Required columns ---
    for name, req in REQUIRED_COLS.items():
        messages += [f"[{name}] {m}" for m in check_required_columns(finals[name], req)]
    messages += [f"[support_report] {m}" for m in check_required_columns(report, REPORT_COLS)]
```
- Check schema columns for finals and report; prefix messages with the table/report tag.

```python
    # --- 2) Uniqueness of IDs ---
    messages += [f"[agents_final] {m}" for m in check_unique(finals["agents_final"], ["agent_id"], "agent_id")]
    messages += [f"[contact_centers_final] {m}" for m in check_unique(finals["contact_centers_final"], ["contact_center_id"], "contact_center_id")]
    messages += [f"[service_categories_final] {m}" for m in check_unique(finals["service_categories_final"], ["category_id"], "category_id")]
    messages += [f"[interactions_final] {m}" for m in check_unique(finals["interactions_final"], ["interaction_id"], "interaction_id")]
```
- Validate primary-key uniqueness per table; each call returns a list of messages (possibly empty).

```python
    # --- 3) UNKNOWN members present in dimensions ---
    messages += [f"[agents_final] {m}" for m in check_unknown_member_present(finals["agents_final"], "agent_id", "agents_final")]
    messages += [f"[contact_centers_final] {m}" for m in check_unknown_member_present(finals["contact_centers_final"], "contact_center_id", "contact_centers_final")]
    messages += [f"[service_categories_final] {m}" for m in check_unknown_member_present(finals["service_categories_final"], "category_id", "service_categories_final")]
```
- Ensure each dimension table contains an `UNKNOWN` member to support FK repairs.

```python
    # --- 4) FK integrity: facts → dims (allow UNKNOWN) ---
    messages += [f"[facts->agents] {m}" for m in check_fk_integrity(finals["interactions_final"], finals["agents_final"], "agent_id", "agent_id")]
    messages += [f"[facts->centers] {m}" for m in check_fk_integrity(finals["interactions_final"], finals["contact_centers_final"], "contact_center_id", "contact_center_id")]
    messages += [f"[facts->categories] {m}" for m in check_fk_integrity(finals["interactions_final"], finals["service_categories_final"], "category_id", "category_id")]
```
- Cross-check the fact table’s FK columns against dimension IDs; permit `"UNKNOWN"`; include table-pair tags in messages.

```python
    # --- 5) Report schema & values ---
    messages += [f"[support_report] {m}" for m in check_month_format(report)]
    messages += [f"[support_report] {m}" for m in check_non_negative(report)]
    messages += [f"[support_report] {m}" for m in check_calls_not_exceed_interactions(report)]
    messages += [f"[support_report] {m}" for m in check_report_internal_consistency(report)]
```
- Validate the report’s column presence, month format, metric ranges, logical constraints, and integer-ness for counts.

```python
    # Overall status: pass if no messages
    ok = len(messages) == 0
    return ok, messages
```
- If no messages were produced, the validation **passes** (`ok=True`); otherwise `ok=False` with accumulated messages.

## Design Notes & Conventions

- **Helper vs public**: Leading underscore on helper functions signals “internal use” by convention.  
- **Type hints**: Aid readability/tooling; not enforced at runtime.  
- **Graceful loading**: `load_any` tries multiple formats and errors clearly.  
- **Composability**: Checks are small, pure functions that return message lists—easy to extend and test.  
- **Early exits**: Fail fast if critical inputs are missing; don’t run misleading downstream checks.
