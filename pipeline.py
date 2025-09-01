import os                   # stdlib: filesystem and path utilities
import sys                  # stdlib: command-line arguments (sys.argv)
import glob                 # stdlib: filename pattern matching ("*_delta_*.csv")
import json                 # stdlib: read/write small state JSON
import logging              # stdlib: structured logging
from typing import List, Optional, Dict  # type hints for readability/tooling

import numpy as np          # arrays and vectorized operations
import pandas as pd         # DataFrame/tabular processing

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ PATHS & LOGGING                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

# --- Data locations (plain strings used to build file paths) ---
initial_path = "data/initial"        # where initial/base CSVs live (Jan 2025)
delta_path   = "data/delta"          # where monthly delta CSVs live (e.g., Feb, Mar)
final_path   = "data/final"          # where we write/read final snapshots
report_path  = "data/report"         # where we write the aggregated report
logger_path  = "logger/logger.log"   # log file location under logger/

state_path   = os.path.join(final_path, "_state.json")  # small JSON registry of processed deltas

# Ensure the directory for the log file exists (no-op if already present)
os.makedirs(os.path.dirname(logger_path), exist_ok=True)  # exist_ok=True → do not error if it exists

# Create or get a named logger; levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
LOGGER = logging.getLogger("support_pipeline")
LOGGER.setLevel(logging.INFO)  # baseline level: ignore DEBUG unless changed

# Add handlers only once (avoid duplicate logs when re-imported in REPL/IDE)
if not LOGGER.handlers:  # len(LOGGER.handlers) == 0 → no handlers yet
    _file_handler = logging.FileHandler(logger_path, encoding="utf-8")  # write logs to file
    _file_handler.setLevel(logging.INFO)                                # file logs at INFO+
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")  # interpolation by logger
    _file_handler.setFormatter(_formatter)                              # attach the formatter

    _console_handler = logging.StreamHandler()  # print logs to stdout/stderr
    _console_handler.setLevel(logging.INFO)     # console logs at INFO+
    _console_handler.setFormatter(_formatter)

    LOGGER.addHandler(_file_handler)            # register handlers with the logger
    LOGGER.addHandler(_console_handler)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ HELPERS                                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names.
    """
    df = df.copy()                                  # avoid mutating caller's object
    df.columns = [c.strip() for c in df.columns]    # list comprehension: apply .strip() to each column name
    return df

def coerce_utc_to_est(series_utc: pd.Series) -> pd.Series:
    """
    Convert UTC timestamps to EST (fixed UTC-5).
    If parsing or tz handling fails, manually shift by -5 hours.
    """
    s = pd.to_datetime(series_utc, utc=True, errors="coerce")  # parse strings → datetime64[ns, UTC]; bad → NaT
    try:
        return s.dt.tz_convert("EST")                          # convert timezone-aware UTC → fixed EST
    except TypeError:
        # If Series is tz-naive somewhere, drop tz and subtract 5 hours to emulate EST
        return s.dt.tz_localize(None) - pd.Timedelta(hours=5)

# Only these columns are treated as timestamps throughout the pipeline
TIMESTAMP_COLS = {
    "timestamp",
    "interaction_start",
    "agent_resolution_timestamp",
    "interaction_end",
}  # a Python set of strings; set ops like .intersection(...) are O(1) per lookup on average

def convert_all_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any present TIMESTAMP_COLS from UTC → EST.
    """
    df = df.copy()
    # Set intersection: keep only timestamp columns that actually exist in df
    for col in TIMESTAMP_COLS.intersection(df.columns):  # iterate each timestamp column present
        df[col] = coerce_utc_to_est(df[col])             # per-column conversion
    return df

def read_csv_required(path: str) -> pd.DataFrame:
    """
    Read CSV with path existence check, normalize column names,
    and convert known timestamp columns to EST.
    """
    np_path = os.path.normpath(path)                      # normalize slashes/backslashes
    if not os.path.exists(np_path):                       # boolean file-exists check
        raise FileNotFoundError(f"Missing required file: {np_path}")
    df = pd.read_csv(np_path)                             # read CSV into a DataFrame
    df = _normalize_cols(df)                              # clean column names
    df = convert_all_timestamps(df)                       # UTC→EST on known timestamp columns
    return df

def read_any_required(base_no_ext: str) -> pd.DataFrame:
    """
    Try reading a table snapshot at base path (no extension) in this order:
      CSV → Parquet → JSON.
    Normalize columns and convert timestamps to EST after reading.
    """
    candidates = [base_no_ext + ".csv", base_no_ext + ".parquet", base_no_ext + ".json"]  # list of strings
    last_err = None  # keep last exception to chain if all fail
    for p in candidates:
        if os.path.exists(p):                             # check which candidate exists
            try:
                if p.endswith(".csv"):                   # str.endswith: suffix check
                    df = pd.read_csv(p)
                elif p.endswith(".parquet"):
                    df = pd.read_parquet(p)
                else:                                    # assume JSON
                    df = pd.read_json(p, orient="records")  # JSON array of row objects
                df = _normalize_cols(df)
                df = convert_all_timestamps(df)
                LOGGER.info("Loaded prior final snapshot: %s", os.path.basename(p))
                return df
            except Exception as e:                        # intentionally broad: any read error
                last_err = e
                LOGGER.warning("Failed to read %s (%s). Trying next.", p, e)
    # If we got here, none succeeded
    raise FileNotFoundError(f"No readable table found at {base_no_ext}.*") from last_err

def ensure_unknown_member(df: pd.DataFrame, id_col: str, name_cols: List[str]) -> pd.DataFrame:
    """
    Ensure the dimension has a special row with {id_col: "UNKNOWN"}.
    Fill other columns with 'Unknown' to keep dtypes stable.
    """
    df = df.copy()
    df[id_col] = df[id_col].astype(str)                # cast ID column to string for consistent matching

    if "UNKNOWN" not in set(df[id_col]):               # build set from Series; O(n) membership test
        pad = {id_col: "UNKNOWN"}                      # dict for a new row

        # Fill expected name-like columns
        for c in name_cols:                            # c: a column name (string)
            if c in df.columns:                        # guard against missing cols
                pad[c] = "Unknown"

        # Fill any other columns not explicitly set
        for c in df.columns:
            if c not in pad:                           # membership test in dict keys
                pad[c] = "Unknown"

        # Append a one-row DataFrame created from 'pad'
        df = pd.concat([df, pd.DataFrame([pad])], ignore_index=True)

    return df

def apply_delta(base: pd.DataFrame, delta: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Apply a delta file to the base table:
      - DELETE: remove rows with given IDs
      - ADD/UPDATE: upsert (replace existing ID, or add new one)
    """
    base = base.copy()                                 # defensive copy
    base[id_col] = base[id_col].astype(str)            # unify dtype for ID

    d = delta.copy()
    if "action" not in d.columns:                      # required column check
        raise ValueError("Delta missing 'action' column")
    d["action"] = d["action"].astype(str).str.upper().str.strip()  # normalize action strings
    d[id_col] = d[id_col].astype(str)                  # IDs as strings

    # If multiple actions for same ID, keep only the last one in the file
    d = d.drop_duplicates(subset=[id_col], keep="last")  # dedupe by id_col

    # DELETE: remove any base rows whose ID appears in del_ids
    del_ids = d.loc[d["action"] == "DELETE", id_col].tolist()
    #  d["action"] == "DELETE" → boolean mask
    #  d.loc[mask, id_col]     → filter rows by mask, then take the ID column
    #  .tolist()               → Python list of IDs (strings)
    if del_ids:
        base = base[~base[id_col].isin(del_ids)]
        #  base[id_col].isin(del_ids) → boolean mask: True if base ID ∈ del_ids
        #  ~                          → boolean NOT (invert True/False)
        #  base[mask]                 → row-filter DataFrame with mask

    # ADD/UPDATE: upsert
    upserts = d[d["action"].isin(["ADD", "UPDATE"])].drop(columns=["action"], errors="ignore")
    #  d["action"].isin([...]) → boolean mask for the two actions
    #  d[mask]                 → subset the delta rows
    #  .drop(columns=..., errors="ignore") → remove 'action' if present
    if not upserts.empty:
        upserts = convert_all_timestamps(upserts)      # normalize timestamp columns
        base = base[~base[id_col].isin(upserts[id_col])]  # drop existing rows that share IDs
        base = pd.concat([base, upserts], ignore_index=True)  # append upserts; reindex to 0..N

    return base

def save_df(df: pd.DataFrame, path_base: str, fmt: str):
    """
    Save a DataFrame in the requested format by appending extension to path_base.
    """
    base = os.path.normpath(path_base)                 # normalize path separators
    if fmt == "csv":
        df.to_csv(base + ".csv", index=False)          # index=False avoids writing the numeric index column
    elif fmt == "json":
        df.to_json(base + ".json", orient="records")   # JSON array of objects
    elif fmt == "parquet":
        df.to_parquet(base + ".parquet", index=False)  # compact columnar format
    else:
        raise ValueError("Unsupported format: " + fmt)

def discover_deltas(delta_dir: str, month_filter: Optional[List[str]]) -> Dict[str, List[str]]:
    """
    Scan delta directory for "*_delta_YYYYMM.csv" files.
    Return {table_name: [sorted delta paths]} and apply an optional month filter.
    """
    mapping: Dict[str, List[str]] = {
        "agents": [],
        "contact_centers": [],
        "service_categories": [],
        "interactions": [],
    }
    if not os.path.isdir(delta_dir):                   # guard if delta dir is missing
        return mapping

    # glob.glob returns a list[str] of paths that match the pattern
    for path in glob.glob(os.path.join(delta_dir, "*_delta_*.csv")):
        fname = os.path.basename(path)                 # strip directory → "agents_delta_202502.csv"
        if "_delta_" not in fname:                     # quick sanity filter
            continue
        left, right = fname.split("_delta_", 1)        # split once: ["agents", "202502.csv"]
        table = left                                   # table name (e.g., "agents")
        yyyymm = os.path.splitext(right)[0]            # remove extension: "202502"

        # Apply table allowlist and month filter (if provided)
        if table in mapping and (month_filter is None or yyyymm in month_filter):
            mapping[table].append(os.path.normpath(path))  # store normalized path

    # Sort lists in chronological order (string sort works for YYYYMM)
    for k in mapping:
        mapping[k].sort(key=lambda p: os.path.basename(p).split("_delta_")[1].split(".")[0])
    return mapping

# ---------- Incremental Processing State ----------

def _blank_state() -> Dict[str, Dict[str, List[str]]]:
    """
    Fresh state structure with no processed files recorded.
    """
    return {
        "processed": {
            "agents": [],
            "contact_centers": [],
            "service_categories": [],
            "interactions": [],
        }
    }

def load_state(path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Load state JSON if present; otherwise return a blank state.
    Structure:
      {"processed": {"agents": [...basenames...], "contact_centers": [...], ...}}
    """
    if not os.path.exists(path):
        return _blank_state()
    with open(path, "r", encoding="utf-8") as f:
        try:
            state = json.load(f)                       # parse JSON → dict
        except json.JSONDecodeError:
            LOGGER.warning("State file is corrupted; resetting: %s", path)
            state = _blank_state()
    # Ensure all keys exist even if file was partial
    state.setdefault("processed", {})
    for k in ["agents", "contact_centers", "service_categories", "interactions"]:
        state["processed"].setdefault(k, [])
    return state

def save_state(path: str, state: Dict[str, Dict[str, List[str]]]) -> None:
    """
    Persist state JSON (mkdir if needed).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)                  # pretty-printed JSON

def filter_unprocessed(deltas: Dict[str, List[str]], state: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """
    Remove delta files that have already been processed (by basename) per table.
    """
    out: Dict[str, List[str]] = {k: [] for k in deltas.keys()}   # initialize same keys with empty lists
    for table, paths in deltas.items():                          # iterate mapping items
        processed_basenames = set(state["processed"].get(table, []))  # set for O(1) membership checks
        to_apply = []                                            # collect new ones
        skipped = []                                             # collect already-processed for logging
        for p in paths:
            b = os.path.basename(p)                              # basename of delta file
            if b in processed_basenames:                         # already processed? (fast set lookup)
                skipped.append(b)
            else:
                to_apply.append(p)
        if skipped:
            LOGGER.info("Skipping already-processed %s deltas: %s", table, ", ".join(skipped))
        out[table] = to_apply
    return out

def mark_processed(state: Dict[str, Dict[str, List[str]]], applied: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Append applied delta basenames to state and keep lists sorted.
    """
    for table, paths in applied.items():
        if not paths:
            continue
        bn = [os.path.basename(p) for p in paths]     # list of basenames
        existing = set(state["processed"].get(table, []))
        new_list = list(existing.union(bn))           # union to avoid duplicates
        new_list.sort()                               # keep stable order (lexicographic)
        state["processed"][table] = new_list
    return state

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ MAIN PIPELINE                                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
def run_pipeline(
    output_format: str = "csv",                 # "csv" | "json" | "parquet"
    month_filter: Optional[List[str]] = None,   # list like ["202502", "202503"] or None for all
    reset: bool = False,                        # when True: ignore finals, reset state, start from initial
) -> Dict[str, str]:
    """
    Run the ETL pipeline with incremental processing:
      - If NOT reset and final snapshots exist: start from final; else start from initial.
      - Apply only *unprocessed* delta files (filtered by --months if provided).
      - Persist final tables and build report.
      - Track processed deltas in _state.json to avoid reprocessing.
    """

    os.makedirs(final_path, exist_ok=True)      # ensure output dirs exist
    os.makedirs(report_path, exist_ok=True)

    # ── STAGE 0: Select starting snapshot (final vs initial) ───────────
    if reset:
        LOGGER.info("--reset supplied: starting fresh from INITIAL and clearing processed-delta registry.")
        state = _blank_state()                  # fresh state (no processed files)
        start_from_final = False                # force initial regardless of existing final files
    else:
        state = load_state(state_path)          # load or create state registry

        def final_exists(name: str) -> bool:
            base = os.path.join(final_path, f"{name}_final")
            # any(...) → True if any of the files exists
            return any(os.path.exists(base + ext) for ext in [".csv", ".parquet", ".json"])

        # any(...) across tables: if any final snapshot exists, we load ALL from final
        start_from_final = any(final_exists(n) for n in ["agents", "contact_centers", "service_categories", "interactions"])

    if start_from_final:
        LOGGER.info("Starting from prior FINAL snapshot in %s (incremental mode).", final_path)
        # Read whichever format exists for each table; normalize & convert timestamps
        agents = read_any_required(os.path.join(final_path, "agents_final"))
        centers = read_any_required(os.path.join(final_path, "contact_centers_final"))
        cats   = read_any_required(os.path.join(final_path, "service_categories_final"))
        facts  = read_any_required(os.path.join(final_path, "interactions_final"))
    else:
        LOGGER.info("Starting from INITIAL snapshot in %s.", initial_path)
        agents = read_csv_required(os.path.join(initial_path, "agents.csv"))
        centers = read_csv_required(os.path.join(initial_path, "contact_centers.csv"))
        cats   = read_csv_required(os.path.join(initial_path, "service_categories.csv"))
        facts  = read_csv_required(os.path.join(initial_path, "interactions.csv"))

    LOGGER.info(
        "Loaded rows — agents=%d, centers=%d, categories=%d, interactions=%d",
        len(agents), len(centers), len(cats), len(facts),
    )

    # ── STAGE 1: Discover & filter deltas (months + unprocessed) ───────
    LOGGER.info("Discovering delta files...")
    discovered = discover_deltas(delta_path, month_filter)  # apply --months filter if provided
    to_apply = filter_unprocessed(discovered, state)        # drop already-processed (per state registry)
    total_to_apply = sum(len(v) for v in to_apply.values()) # sum lengths across table lists
    if total_to_apply == 0:
        LOGGER.info("No new delta files to apply (already up-to-date for the requested months).")

    # ── STAGE 2: Apply deltas ──────────────────────────────────────────
    LOGGER.info("Applying delta files...")
    id_map = {
        "agents": "agent_id",
        "contact_centers": "contact_center_id",
        "service_categories": "category_id",
        "interactions": "interaction_id",
    }  # per-table primary key column

    tables: Dict[str, pd.DataFrame] = {
        "agents": agents,
        "contact_centers": centers,
        "service_categories": cats,
        "interactions": facts,
    }  # working set of tables

    actually_applied: Dict[str, List[str]] = {k: [] for k in to_apply.keys()}  # for state update later

    for t, paths in to_apply.items():                      # iterate tables and their pending delta paths
        for pth in paths:                                  # pth: absolute/relative path to a delta CSV
            LOGGER.info("Applying %s delta: %s", t, os.path.basename(pth))
            ddf = read_csv_required(pth)                   # read + normalize timestamps
            before = len(tables[t])                        # rowcount before applying
            tables[t] = apply_delta(tables[t], ddf, id_map[t])  # apply upserts/deletes
            after = len(tables[t])                         # rowcount after applying
            LOGGER.info(" -> %s rows: %d -> %d", t, before, after)
            actually_applied[t].append(pth)                # remember this file as applied

    # Ensure department column exists (used later in reporting joins/aggregations)
    if "department" not in tables["service_categories"].columns:
        tables["service_categories"]["department"] = "Unknown"

    # ── STAGE 3: Add 'UNKNOWN' members to dimensions ───────────────────
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

    # ── STAGE 4: Repair fact foreign keys ──────────────────────────────
    LOGGER.info("Repairing foreign key references in interactions...")
    f = tables["interactions"].copy()                      # work on a copy to avoid chained assignments

    # Ensure FK columns are strings to match dimension ID dtypes
    for col in ["agent_id", "contact_center_id", "category_id"]:
        if col in f.columns:
            f[col] = f[col].astype(str)

    # Collect valid ID sets from dimensions (cast to str)
    valid_agents  = set(tables["agents"]["agent_id"].astype(str))
    valid_centers = set(tables["contact_centers"]["contact_center_id"].astype(str))
    valid_cats    = set(tables["service_categories"]["category_id"].astype(str))

    # Count invalid references using boolean masks
    missing_agents  = (~f["agent_id"].isin(valid_agents)).sum() if "agent_id" in f.columns else 0
    missing_centers = (~f["contact_center_id"].isin(valid_centers)).sum() if "contact_center_id" in f.columns else 0
    missing_cats    = (~f["category_id"].isin(valid_cats)).sum() if "category_id" in f.columns else 0
    LOGGER.info("Missing FK counts — agents=%d, centers=%d, categories=%d",
                missing_agents, missing_centers, missing_cats)

    # Overwrite invalid FK values with "UNKNOWN"
    if "agent_id" in f.columns:
        f.loc[~f["agent_id"].isin(valid_agents), "agent_id"] = "UNKNOWN"
    if "contact_center_id" in f.columns:
        f.loc[~f["contact_center_id"].isin(valid_centers), "contact_center_id"] = "UNKNOWN"
    if "category_id" in f.columns:
        f.loc[~f["category_id"].isin(valid_cats), "category_id"] = "UNKNOWN"

    tables["interactions"] = f                              # write repaired facts back

    # ── STAGE 5: Save final tables ─────────────────────────────────────
    LOGGER.info("Saving final tables to %s ...", final_path)

    def save_final(name: str, df: pd.DataFrame) -> str:
        """
        Persist a table with standard suffix '<name>_final.<ext>' and return the path.
        """
        base = os.path.join(final_path, f"{name}_final")
        save_df(df, base, output_format)                   # choose extension by output_format
        return f"{base}.{output_format}"

    paths: Dict[str, str] = {
        "agents_final": save_final("agents", tables["agents"]),
        "contact_centers_final": save_final("contact_centers", tables["contact_centers"]),
        "service_categories_final": save_final("service_categories", tables["service_categories"]),
        "interactions_final": save_final("interactions", tables["interactions"]),
    }

    # ── STAGE 6: Build support report ──────────────────────────────────
    LOGGER.info("Building support_report...")
    inter = tables["interactions"].copy()                   # local working copy for reporting

    # Month bucketing uses 'interaction_start' (already EST by design)
    ts_col = "interaction_start" if "interaction_start" in inter.columns else None
    if ts_col is None:
        raise ValueError("Missing required 'interaction_start' column in interactions")

    ts = inter[ts_col]
    if not pd.api.types.is_datetime64_any_dtype(ts):        # ensure datetime dtype if it isn't already
        ts = pd.to_datetime(ts, errors="coerce")
    inter["month"] = ts.dt.strftime("%Y-%m")                # format as "YYYY-MM" into a new column

    # Join contact center names
    centers_dim = (
        tables["contact_centers"][["contact_center_id", "contact_center_name"]]
        if "contact_center_name" in tables["contact_centers"].columns
        else tables["contact_centers"][["contact_center_id"]].assign(contact_center_name="Unknown")
    )
    centers_dim = centers_dim.copy()
    centers_dim["contact_center_id"] = centers_dim["contact_center_id"].astype(str)   # key dtype normalize
    inter["contact_center_id"] = inter["contact_center_id"].astype(str)               # FK dtype normalize
    inter = inter.merge(centers_dim, on="contact_center_id", how="left")              # left join
    inter["contact_center_name"] = inter["contact_center_name"].fillna("Unknown")     # fill missing joined values

    # Join department from categories
    cats_dim = tables["service_categories"][["category_id", "department"]].copy()
    cats_dim["category_id"] = cats_dim["category_id"].astype(str)
    inter["category_id"] = inter["category_id"].astype(str)
    inter = inter.merge(cats_dim, on="category_id", how="left")
    inter["department"] = inter["department"].fillna("Unknown")

    # Derived flags and measures
    ch = inter["channel"].str.lower().to_numpy() if "channel" in inter.columns else np.array([""] * len(inter))
    #  .str.lower() → vectorized lowercase on Series of strings
    #  .to_numpy()  → convert Series to a NumPy array for fast element-wise equality
    inter["is_call"] = (ch == "phone")  # element-wise comparison yields boolean array; assigned to new column

    dur_col = next((c for c in ["call_duration_minutes"] if c in inter.columns), None)  # first matching name or None
    if dur_col is None:
        inter["duration_value"] = 0
        dur_col = "duration_value"
    inter[dur_col] = pd.to_numeric(inter[dur_col], errors="coerce").fillna(0)  # parse numeric; bad → NaN → fill 0

    # Group and aggregate (counts and sums)
    report = (
        inter.groupby(["month", "contact_center_name", "department"], dropna=False)  # group keys; keep NaNs if any
        .agg(
            total_interactions=("interaction_id", "count"),  # count rows per group
            total_calls=("is_call", "sum"),                  # sum True values (True→1)
            total_call_duration=(dur_col, "sum"),           # sum of durations per group
        )
        .reset_index()                                       # convert group index back to columns
        .sort_values(["month", "contact_center_name", "department"])  # stable sorted output
    )

    report_base = os.path.join(report_path, "support_report")  # base path (no extension yet)
    save_df(report, report_base, output_format)                # persist report in chosen format
    paths["support_report"] = f"{report_base}.{output_format}"

    # ── STAGE 7: Update processed-delta state ──────────────────────────
    state = mark_processed(state, actually_applied)  # merge in applied deltas, sorted unique
    save_state(state_path, state)                    # write to disk

    LOGGER.info("Processed %d new delta files.", sum(len(v) for v in actually_applied.values()))
    LOGGER.info("Report written to %s", paths["support_report"])
    LOGGER.info("Done.")
    return paths

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ CLI ENTRY POINT                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    import argparse  # stdlib: easy command-line parsing

    # If no args, run with defaults (ALL months, CSV output)
    if len(sys.argv) == 1:
        LOGGER.info("No arguments provided. Running with defaults: format=csv, all months.")
        run_pipeline(output_format="csv", month_filter=None, reset=False)
    else:
        parser = argparse.ArgumentParser()
        # --format: restrict to known set of choices
        parser.add_argument("--format", default="csv", choices=["csv", "json", "parquet"])
        # --months: comma-separated list like "202502,202503"
        parser.add_argument(
            "--months",
            default=None,
            help="Comma-separated YYYYMM list to LIMIT processing (e.g., 202502,202503). Omit to process ALL deltas.",
        )
        # --reset: store_true means flag is a boolean; present → True, absent → False
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Start from data/initial, ignore prior finals, and reset processed-deltas registry.",
        )

        args = parser.parse_args()                                     # parse into a Namespace
        months = args.months.split(",") if args.months else None       # list[str] or None

        if months is None:
            LOGGER.info("No --months provided: processing ALL available delta months.")
        else:
            LOGGER.info("Restricting to delta months: %s", ",".join(months))

        if args.reset:
            LOGGER.info("--reset is enabled: the run will ignore existing finals and clear processed registry.")

        run_pipeline(output_format=args.format, month_filter=months, reset=args.reset)
