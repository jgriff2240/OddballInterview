# Pipeline Script — Line‑by‑Line, Syntax‑Aware Commentary

This document explains the provided script **line by line**, with extra context about **what each line does**, **why it’s written that way**, and **how the Python syntax works**. Sections mirror the code’s structure so you can quickly navigate.

---

## Imports

```python
import os
```
- Loads Python’s built‑in **OS** module. Used for filesystem operations (creating directories, joining paths, checking file existence).  
- **Syntax:** `import <module>` binds a module object to the given name.

```python
import sys
```
- Imports **sys** to access interpreter internals like `sys.argv` (command‑line arguments) and `sys.path`.  
- Same import syntax as above.

```python
import glob
```
- Imports **glob** to expand shell‑style filename patterns like `*_delta_*.csv`.  
- Returns lists of path strings.

```python
import json
```
- Imports **json** for reading/writing small JSON files (e.g., state registry `_state.json`).

```python
import logging
```
- Imports **logging** to emit structured log messages with levels (INFO/ERROR, etc.), handlers, and formatters.

```python
from typing import List, Optional, Dict
```
- Imports **type hints** for better editor support and readability.  
- **Syntax:** `from <module> import A, B` brings symbols directly into the local namespace.

```python
import numpy as np
```
- Imports **NumPy** and aliases it as `np` (convention). Useful for vectorized boolean ops and numerics.

```python
import pandas as pd
```
- Imports **Pandas** as `pd`. Provides DataFrame structures and IO helpers (CSV/JSON/Parquet).

---

## Paths & Logging

```python
initial_path = "data/initial"
```
- String constant pointing to initial/base CSV directory (first snapshot).

```python
delta_path   = "data/delta"
```
- Directory containing monthly delta CSVs to apply incrementally.

```python
final_path   = "data/final"
```
- Output directory for materialized “final” tables (used as the next run’s starting point when not resetting).

```python
report_path  = "data/report"
```
- Output directory for aggregated reporting artifacts (e.g., `support_report`).

```python
logger_path  = "logger/logger.log"
```
- File path for persisted logs.

```python
state_path   = "state/_state.json"
```
- File path for the state registry of already‑processed delta files.

```python
os.makedirs(os.path.dirname(logger_path), exist_ok=True)
```
- Creates the **parent directory** for the log file if it doesn’t exist.  
- `exist_ok=True` prevents errors when the directory already exists.  
- `os.path.dirname(p)` returns the directory portion of a path string.

```python
LOGGER = logging.getLogger("support_pipeline")
```
- Gets (or creates) a named logger. The name helps identify log origin.

```python
LOGGER.setLevel(logging.INFO)
```
- Sets the **minimum** level that this logger will process (`INFO` and above).

```python
if not LOGGER.handlers:
```
- Avoids adding duplicate handlers when the module is re‑imported (e.g., in REPL/IDE).  
- Empty list is falsy → this condition is `True` only once.

```python
    _file_handler = logging.FileHandler(logger_path, encoding="utf-8")
```
- Creates a **file handler** that writes logs to `logger/logger.log`.  
- Leading underscore `_file_handler` follows the convention for “internal” names.

```python
    _file_handler.setLevel(logging.INFO)
```
- Sets the handler’s own level (can differ from the logger’s).

```python
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
```
- Defines string formatting to render each log record.  
- Tokens (e.g., `%(asctime)s`) interpolate fields from the LogRecord.

```python
    _file_handler.setFormatter(_formatter)
```
- Attaches the formatter to the file handler.

```python
    _console_handler = logging.StreamHandler()
```
- Creates a handler that writes to the console (stderr by default).

```python
    _console_handler.setLevel(logging.INFO)
```
- Sets the console handler’s level to INFO and above.

```python
    _console_handler.setFormatter(_formatter)
```
- Uses the same format string for console output.

```python
    LOGGER.addHandler(_file_handler)
    LOGGER.addHandler(_console_handler)
```
- Registers both handlers with the logger so log messages flow to file and console.

---

## Helpers

```python
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
```
- Defines a function taking a Pandas DataFrame and returning a DataFrame.  
- Leading `_` marks it as an internal helper by convention.

```python
    df = df.copy()
```
- Avoids mutating the input DataFrame (functional style).

```python
    df.columns = [c.strip() for c in df.columns]
```
- **List comprehension** that applies `.strip()` to each column name to remove leading/trailing spaces.

```python
    return df
```
- Returns the modified copy.

---

```python
def convert_utc_to_est(series_utc: pd.Series) -> pd.Series:
```
- Converts a timestamp Series from UTC to EST, handling tz‑awareness and parsing failures.

```python
    s = pd.to_datetime(series_utc, utc=True, errors="coerce")
```
- Parses to timezone‑aware UTC datetimes. Bad inputs become `NaT` (Not‑a‑Time).

```python
    try:
        return s.dt.tz_convert("EST")
```
- For tz‑aware values, convert timezone directly to the fixed EST zone.

```python
    except TypeError:
        return s.dt.tz_localize(None) - pd.Timedelta(hours=5)
```
- Fallback if the Series is tz‑naive: drop tz info and subtract 5 hours to emulate EST.

---

```python
TIMESTAMP_COLS = {
    "timestamp",
    "interaction_start",
    "agent_resolution_timestamp",
    "interaction_end",
}
```
- A **set** of canonical timestamp column names. Sets give fast membership checks and convenient intersections.

```python
def convert_all_timestamps(df: pd.DataFrame) -> pd.DataFrame:
```
- Converts any of the known timestamp columns **that actually exist** in the given DataFrame.

```python
    df = df.copy()
```
- Work on a copy to avoid in‑place surprises.

```python
    for col in TIMESTAMP_COLS.intersection(df.columns):
        df[col] = convert_utc_to_est(df[col])
```
- Intersect the desired set with actual columns and convert each present one.  
- `set.intersection(iterable)` returns the common elements.

```python
    return df
```
- Return the transformed copy.

---

```python
def read_csv_required(path: str) -> pd.DataFrame:
```
- Reads a CSV file at `path` and applies normalization and timezone conversions.

```python
    np_path = os.path.normpath(path)
```
- Normalizes path separators (`/` vs `\\`) based on OS.

```python
    if not os.path.exists(np_path):
        raise FileNotFoundError(f"Missing required file: {np_path}")
```
- Pre‑flight validation with a descriptive error if missing.

```python
    df = pd.read_csv(np_path)
    df = _normalize_cols(df)
    df = convert_all_timestamps(df)
    return df
```
- Load, clean, convert timestamps, return.

---

```python
def read_any_required(base_no_ext: str) -> pd.DataFrame:
```
- Attempts to read a snapshot from `<base>.csv` → `<base>.parquet` → `<base>.json`.  
- Useful when resuming from **finals** regardless of previously chosen format.

```python
    candidates = [base_no_ext + ".csv", base_no_ext + ".parquet", base_no_ext + ".json"]
    last_err = None
```
- Ordered list of candidate file paths and a holder for the last exception.

```python
    for p in candidates:
        if os.path.exists(p):
            try:
                if p.endswith(".csv"):
                    df = pd.read_csv(p)
                elif p.endswith(".parquet"):
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_json(p, orient="records")
```
- Conditionally reads based on file extension.  
- `orient="records"` reads an array of row objects for JSON.

```python
                df = _normalize_cols(df)
                df = convert_all_timestamps(df)
                LOGGER.info("Loaded prior final snapshot: %s", os.path.basename(p))
                return df
```
- Standardize columns and timestamps; log and return.

```python
            except Exception as e:
                last_err = e
                LOGGER.warning("Failed to read %s (%s). Trying next.", p, e)
```
- If a read fails, warn and continue to the next candidate.

```python
    raise FileNotFoundError(f"No readable table found at {base_no_ext}.*") from last_err
```
- If none loaded, raise a chained error with the last exception as context.

---

```python
def ensure_unknown_member(df: pd.DataFrame, id_col: str, name_cols: List[str]) -> pd.DataFrame:
```
- Guarantees a dimension table includes an `UNKNOWN` member for robust joins and foreign key repair.

```python
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
```
- Avoid mutation; normalize ID dtype.

```python
    if "UNKNOWN" not in set(df[id_col]):
        pad = {id_col: "UNKNOWN"}
```
- Create a dict for a new row if not already present.

```python
        for c in name_cols:
            if c in df.columns:
                pad[c] = "Unknown"
```
- Fill expected descriptive/name columns with `"Unknown"` to keep dtypes stable.

```python
        for c in df.columns:
            if c not in pad:
                pad[c] = "Unknown"
```
- Fill any remaining columns for consistency.

```python
        df = pd.concat([df, pd.DataFrame([pad])], ignore_index=True)
```
- Append the new row and reindex.

```python
    return df
```
- Return the augmented DataFrame.

---

```python
def apply_delta(base: pd.DataFrame, delta: pd.DataFrame, id_col: str) -> pd.DataFrame:
```
- Applies `DELETE`, `ADD`, and `UPDATE` actions from a delta table to the base table.

```python
    base = base.copy()
    base[id_col] = base[id_col].astype(str)
```
- Work on a copy; normalize ID dtype to string.

```python
    d = delta.copy()
    if "action" not in d.columns:
        raise ValueError("Delta missing 'action' column")
```
- Ensure required control column exists.

```python
    d["action"] = d["action"].astype(str).str.upper().str.strip()
    d[id_col] = d[id_col].astype(str)
```
- Normalize action values and IDs (e.g., " add " → "ADD").

```python
    d = d.drop_duplicates(subset=[id_col], keep="last")
```
- If multiple actions exist for the same ID in one file, keep the final one.

```python
    del_ids = d.loc[d["action"] == "DELETE", id_col].tolist()
```
- List IDs to delete.

```python
    if del_ids:
        base = base[~base[id_col].isin(del_ids)]
```
- Filter out rows whose IDs are marked for deletion (`~` negates the boolean mask).

```python
    upserts = d[d["action"].isin(["ADD", "UPDATE"])].drop(columns=["action"], errors="ignore")
```
- Subset the rows to upsert and drop the control column.

```python
    if not upserts.empty:
        upserts = convert_all_timestamps(upserts)
        base = base[~base[id_col].isin(upserts[id_col])]
        base = pd.concat([base, upserts], ignore_index=True)
```
- Remove existing rows with those IDs, then append normalized replacements.

```python
    return base
```
- Return the updated table state.

---

```python
def save_df(df: pd.DataFrame, path_base: str, fmt: str):
```
- Saves a DataFrame to `csv`, `json`, or `parquet` based on the `fmt` string.

```python
    base = os.path.normpath(path_base)
```
- Normalize path for the OS.

```python
    if fmt == "csv":
        df.to_csv(base + ".csv", index=False)
    elif fmt == "json":
        df.to_json(base + ".json", orient="records")
    elif fmt == "parquet":
        df.to_parquet(base + ".parquet", index=False)
    else:
        raise ValueError("Unsupported format: " + fmt)
```
- Branch on format and call the appropriate writer. Raise if unsupported.

---

```python
def discover_deltas(delta_dir: str, month_filter: Optional[List[str]]) -> Dict[str, List[str]]:
```
- Scans the delta directory for `*_delta_YYYYMM.csv` files and returns a map of table→file list.  
- `month_filter` limits which YYYYMM files are included.

```python
    mapping: Dict[str, List[str]] = {
        "agents": [],
        "contact_centers": [],
        "service_categories": [],
        "interactions": [],
    }
```
- Initialize an allowlist of valid tables with empty lists.

```python
    if not os.path.isdir(delta_dir):
        return mapping
```
- If the directory doesn’t exist, return empty buckets (safe default).

```python
    for path in glob.glob(os.path.join(delta_dir, "*_delta_*.csv")):
        fname = os.path.basename(path)
        if "_delta_" not in fname:
            continue
        left, right = fname.split("_delta_", 1)
        table = left
        yyyymm = os.path.splitext(right)[0]
```
- Iterate matched files; parse out `table` and `yyyymm` from the filename.  
- `os.path.splitext(x)[0]` drops the file extension.

```python
        if table in mapping and (month_filter is None or yyyymm in month_filter):
            mapping[table].append(os.path.normpath(path))
```
- Apply table allowlist and optional month filter.

```python
    for k in mapping:
        mapping[k].sort(key=lambda p: os.path.basename(p).split("_delta_")[1].split(".")[0])
    return mapping
```
- Sort each list chronologically by YYYYMM (string sort works for year‑month). Return the mapping.

---

## Incremental Processing State

```python
def _blank_state() -> Dict[str, Dict[str, List[str]]]:
```
- Returns a fresh state structure with no processed files recorded.

```python
    return {
        "processed": {
            "agents": [],
            "contact_centers": [],
            "service_categories": [],
            "interactions": [],
        }
    }
```
- JSON‑serializable dict of lists keyed by table names.

```python
def load_state(path: str) -> Dict[str, Dict[str, List[str]]]:
```
- Loads state if present; otherwise returns a blank state. Repairs missing keys.

```python
    if not os.path.exists(path):
        return _blank_state()
```
- No state file → start clean.

```python
    with open(path, "r", encoding="utf-8") as f:
        try:
            state = json.load(f)
        except json.JSONDecodeError:
            LOGGER.warning("State file is corrupted; resetting: %s", path)
            state = _blank_state()
```
- Parse JSON; if corrupted, log a warning and reset to blank.

```python
    state.setdefault("processed", {})
    for k in ["agents", "contact_centers", "service_categories", "interactions"]:
        state["processed"].setdefault(k, [])
    return state
```
- Ensure required keys exist even if the file was partial; return the normalized state.

```python
def save_state(path: str, state: Dict[str, Dict[str, List[str]]]) -> None:
```
- Persists state to disk as pretty‑printed JSON.

```python
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
```
- Ensure directory exists, write JSON with indentation.

```python
def filter_unprocessed(deltas: Dict[str, List[str]], state: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
```
- Filters out delta files that are already recorded in state as processed.

```python
    out: Dict[str, List[str]] = {k: [] for k in deltas.keys()}
```
- Initialize output mapping with the same keys and empty lists.

```python
    for table, paths in deltas.items():
        processed_basenames = set(state["processed"].get(table, []))
        to_apply = []
        skipped = []
        for p in paths:
            b = os.path.basename(p)
            if b in processed_basenames:
                skipped.append(b)
            else:
                to_apply.append(p)
        if skipped:
            LOGGER.info("Skipping already-processed %s deltas: %s", table, ", ".join(skipped))
        out[table] = to_apply
    return out
```
- Build `to_apply` per table by comparing basenames. Log skipped items. Return new mapping.

```python
def mark_processed(state: Dict[str, Dict[str, List[str]]], applied: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
```
- Merges newly applied delta basenames into the state (deduped and sorted).

```python
    for table, paths in applied.items():
        if not paths:
            continue
        bn = [os.path.basename(p) for p in paths]
        existing = set(state["processed"].get(table, []))
        new_list = list(existing.union(bn))
        new_list.sort()
        state["processed"][table] = new_list
    return state
```
- Update, sort, return state.

---

## Main Pipeline

```python
def run_pipeline(
    output_format: str = "csv",
    month_filter: Optional[List[str]] = None,
    reset: bool = False,
) -> Dict[str, str]:
```
- Entry point to run ETL. Parameters: output format, an optional month filter (list of YYYYMM), and `reset` flag.

```python
    os.makedirs(final_path, exist_ok=True)
    os.makedirs(report_path, exist_ok=True)
```
- Ensure output directories exist.

```python
    if reset:
        LOGGER.info("--reset supplied: starting fresh from INITIAL and clearing processed-delta registry.")
        state = _blank_state()
        start_from_final = False
```
- `--reset` forces use of the `initial` snapshots and clears state.

```python
    else:
        state = load_state(state_path)

        def final_exists(name: str) -> bool:
            base = os.path.join(final_path, f"{name}_final")
            return any(os.path.exists(base + ext) for ext in [".csv", ".parquet", ".json"])
```
- Load state and define a helper to detect if any final snapshot exists for a table in any supported format.

```python
        start_from_final = any(final_exists(n) for n in ["agents", "contact_centers", "service_categories", "interactions"])
```
- If **any** final exists, prefer loading **all** tables from `final_path` to continue incrementally.

```python
    if start_from_final:
        LOGGER.info("Starting from prior FINAL snapshot in %s (incremental mode).", final_path)
        agents = read_any_required(os.path.join(final_path, "agents_final"))
        centers = read_any_required(os.path.join(final_path, "contact_centers_final"))
        cats   = read_any_required(os.path.join(final_path, "service_categories_final"))
        facts  = read_any_required(os.path.join(final_path, "interactions_final"))
```
- Resume from previously materialized finals (any format).

```python
    else:
        LOGGER.info("Starting from INITIAL snapshot in %s.", initial_path)
        agents = read_csv_required(os.path.join(initial_path, "agents.csv"))
        centers = read_csv_required(os.path.join(initial_path, "contact_centers.csv"))
        cats   = read_csv_required(os.path.join(initial_path, "service_categories.csv"))
        facts  = read_csv_required(os.path.join(initial_path, "interactions.csv"))
```
- Start from the initial CSVs (fresh baseline).

```python
    LOGGER.info(
        "Loaded rows — agents=%d, centers=%d, categories=%d, interactions=%d",
        len(agents), len(centers), len(cats), len(facts),
    )
```
- Log the row counts for quick sanity checks.

```python
    LOGGER.info("Discovering delta files...")
    discovered = discover_deltas(delta_path, month_filter)
    to_apply = filter_unprocessed(discovered, state)
    total_to_apply = sum(len(v) for v in to_apply.values())
    if total_to_apply == 0:
        LOGGER.info("No new delta files to apply (already up-to-date for the requested months).")
```
- Find delta files, filter out already processed ones, and report status.

```python
    LOGGER.info("Applying delta files...")
    id_map = {
        "agents": "agent_id",
        "contact_centers": "contact_center_id",
        "service_categories": "category_id",
        "interactions": "interaction_id",
    }
```
- Map each table to its primary key column for delta application.

```python
    tables: Dict[str, pd.DataFrame] = {
        "agents": agents,
        "contact_centers": centers,
        "service_categories": cats,
        "interactions": facts,
    }
```
- Working set of named DataFrames in a dictionary.

```python
    actually_applied: Dict[str, List[str]] = {k: [] for k in to_apply.keys()}
```
- Track which delta files get applied this run (for state update).

```python
    for t, paths in to_apply.items():
        for pth in paths:
            LOGGER.info("Applying %s delta: %s", t, os.path.basename(pth))
            ddf = read_csv_required(pth)
            before = len(tables[t])
            tables[t] = apply_delta(tables[t], ddf, id_map[t])
            after = len(tables[t])
            LOGGER.info(" -> %s rows: %d -> %d", t, before, after)
            actually_applied[t].append(pth)
```
- Apply each pending delta file in chronological order and log the row count change.

```python
    if "department" not in tables["service_categories"].columns:
        tables["service_categories"]["department"] = "Unknown"
```
- Ensure a `department` column exists for reporting joins/aggregations.

```python
    LOGGER.info("Ensuring UNKNOWN members in dimensions...")
    tables["agents"] = ensure_unknown_member(
        tables["agents"], "agent_id",
        ["agent_name"] if "agent_name" in tables["agents"].columns else [],
    )
    tables["contact_centers"] = ensure_unknown_member(
        tables["contact_centers"], "contact_center_id",
        ["contact_center_name"] if "contact_center_name" in tables["contact_centers"].columns else [],
    )
    name_cols = ["department"] + (["category_name"] if "category_name" in tables["service_categories"].columns else [])
    tables["service_categories"] = ensure_unknown_member(
        tables["service_categories"], "category_id", name_cols,
    )
```
- Add an `UNKNOWN` row to each dimension table to make joins robust and enable FK repair.

```python
    LOGGER.info("Repairing foreign key references in interactions...")
    f = tables["interactions"].copy()
```
- Work on a copy of facts for FK checks and repairs.

```python
    for col in ["agent_id", "contact_center_id", "category_id"]:
        if col in f.columns:
            f[col] = f[col].astype(str)
```
- Normalize FK dtypes to `str` to match the dimension ID types.

```python
    valid_agents  = set(tables["agents"]["agent_id"].astype(str))
    valid_centers = set(tables["contact_centers"]["contact_center_id"].astype(str))
    valid_cats    = set(tables["service_categories"]["category_id"].astype(str))
```
- Build sets of valid IDs (fast membership checks).

```python
    missing_agents  = (~f["agent_id"].isin(valid_agents)).sum() if "agent_id" in f.columns else 0
    missing_centers = (~f["contact_center_id"].isin(valid_centers)).sum() if "contact_center_id" in f.columns else 0
    missing_cats    = (~f["category_id"].isin(valid_cats)).sum() if "category_id" in f.columns else 0
    LOGGER.info("Missing FK counts — agents=%d, centers=%d, categories=%d",
                missing_agents, missing_centers, missing_cats)
```
- Count invalid foreign keys using boolean masks and log summary counts.

```python
    if "agent_id" in f.columns:
        f.loc[~f["agent_id"].isin(valid_agents), "agent_id"] = "UNKNOWN"
    if "contact_center_id" in f.columns:
        f.loc[~f["contact_center_id"].isin(valid_centers), "contact_center_id"] = "UNKNOWN"
    if "category_id" in f.columns:
        f.loc[~f["category_id"].isin(valid_cats), "category_id"] = "UNKNOWN"
```
- Replace invalid foreign key values with `"UNKNOWN"` to preserve referential integrity for reporting.

```python
    tables["interactions"] = f
```
- Write the repaired facts back to the working set.

```python
    LOGGER.info("Saving final tables to %s ...", final_path)
```
- Start persistence stage.

```python
    def save_final(name: str, df: pd.DataFrame) -> str:
        base = os.path.join(final_path, f"{name}_final")
        save_df(df, base, output_format)
        return f"{base}.{output_format}"
```
- Helper to write a final table with a standard `<name>_final.<ext>` pattern and return its path.

```python
    paths: Dict[str, str] = {
        "agents_final": save_final("agents", tables["agents"]),
        "contact_centers_final": save_final("contact_centers", tables["contact_centers"]),
        "service_categories_final": save_final("service_categories", tables["service_categories"]),
        "interactions_final": save_final("interactions", tables["interactions"]),
    }
```
- Collect the written file paths for return and logging.

```python
    LOGGER.info("Building support_report...")
    inter = tables["interactions"].copy()
```
- Prepare to build the aggregated report.

```python
    ts_col = "interaction_start" if "interaction_start" in inter.columns else None
    if ts_col is None:
        raise ValueError("Missing required 'interaction_start' column in interactions")
```
- Select the timestamp column used for monthly bucketing; fail early if absent.

```python
    ts = inter[ts_col]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce")
    inter["month"] = ts.dt.strftime("%Y-%m")
```
- Ensure datetime dtype and derive a `month` column formatted as `YYYY-MM`.

```python
    centers_dim = (
        tables["contact_centers"][["contact_center_id", "contact_center_name"]]
        if "contact_center_name" in tables["contact_centers"].columns
        else tables["contact_centers"][["contact_center_id"]].assign(contact_center_name="Unknown")
    )
    centers_dim = centers_dim.copy()
    centers_dim["contact_center_id"] = centers_dim["contact_center_id"].astype(str)
    inter["contact_center_id"] = inter["contact_center_id"].astype(str)
    inter = inter.merge(centers_dim, on="contact_center_id", how="left")
    inter["contact_center_name"] = inter["contact_center_name"].fillna("Unknown")
```
- Join contact center names to interactions (left join), normalizing ID types and filling missing values.

```python
    cats_dim = tables["service_categories"][["category_id", "department"]].copy()
    cats_dim["category_id"] = cats_dim["category_id"].astype(str)
    inter["category_id"] = inter["category_id"].astype(str)
    inter = inter.merge(cats_dim, on="category_id", how="left")
    inter["department"] = inter["department"].fillna("Unknown")
```
- Join `department` from categories, normalizing ID types and filling missing values.

```python
    ch = inter["channel"].str.lower().to_numpy() if "channel" in inter.columns else np.array([""] * len(inter))
    inter["is_call"] = (ch == "phone")
```
- Create a boolean `is_call` indicator based on channel name (vectorized comparison).

```python
    dur_col = next((c for c in ["call_duration_minutes"] if c in inter.columns), None)
    if dur_col is None:
        inter["duration_value"] = 0
        dur_col = "duration_value"
    inter[dur_col] = pd.to_numeric(inter[dur_col], errors="coerce").fillna(0)
```
- Choose a duration column if present; else create one with zeros. Coerce text to numeric, turning errors into 0.

```python
    report = (
        inter.groupby(["month", "contact_center_name", "department"], dropna=False)
        .agg(
            total_interactions=("interaction_id", "count"),
            total_calls=("is_call", "sum"),
            total_call_duration=(dur_col, "sum"),
        )
        .reset_index()
        .sort_values(["month", "contact_center_name", "department"])
    )
```
- Group and aggregate to produce the `support_report` with counts and sums per month/center/department.

```python
    report_base = os.path.join(report_path, "support_report")
    save_df(report, report_base, output_format)
    paths["support_report"] = f"{report_base}.{output_format}"
```
- Persist the report and record the path.

```python
    state = mark_processed(state, actually_applied)
    save_state(state_path, state)
```
- Update and persist the state with newly applied deltas.

```python
    LOGGER.info("Processed %d new delta files.", sum(len(v) for v in actually_applied.values()))
    LOGGER.info("Report written to %s", paths["support_report"])
    LOGGER.info("Done.")
    return paths
```
- Final logging and return value: a dict of artifact paths.

---

## CLI Entry Point

```python
if __name__ == "__main__":
```
- Standard Python idiom: only run the following as a script (not when imported as a module).

```python
    import argparse
```
- Imports **argparse** for command‑line parsing.

```python
    if len(sys.argv) == 1:
        LOGGER.info("No arguments provided. Running with defaults: format=csv, all months.")
        run_pipeline(output_format="csv", month_filter=None, reset=False)
```
- If the script was run without flags, use defaults (CSV, all months, no reset).

```python
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--format", default="csv", choices=["csv", "json", "parquet"])
        parser.add_argument(
            "--months",
            default=None,
            help="Comma-separated YYYYMM list to LIMIT processing (e.g., 202502,202503). Omit to process ALL deltas.",
        )
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Start from data/initial, ignore prior finals, and reset processed-deltas registry.",
        )
```
- Defines CLI flags with validation and help: `--format`, `--months`, `--reset` (boolean).

```python
        args = parser.parse_args()
        months = args.months.split(",") if args.months else None
```
- Parses arguments into `args`; converts the comma‑separated months into a list or `None`.

```python
        if months is None:
            LOGGER.info("No --months provided: processing ALL available delta months.")
        else:
            LOGGER.info("Restricting to delta months: %s", ",".join(months))
```
- Informational logging depending on presence of `--months`.

```python
        if args.reset:
            LOGGER.info("--reset is enabled: the run will ignore existing finals and clear processed registry.")
```
- Log when `--reset` is active.

```python
        run_pipeline(output_format=args.format, month_filter=months, reset=args.reset)
```
- Execute the pipeline with the specified CLI parameters.

---

## Notes on Conventions

- **Leading underscore** (e.g., `_file_handler`): Conventional “internal/private” hint, not enforced by Python.  
- **Type hints**: Aid readability and tooling; not enforced at runtime unless you add a type checker.  
- **Copy on write** (`df.copy()`): Prevents side effects and chained assignment warnings.  
- **Set operations**: Efficient for membership checks and deduping.  
- **Robustness**: Fallbacks for timezone handling, safe JSON parsing, and idempotent state tracking.
