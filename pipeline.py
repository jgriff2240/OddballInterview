import os
import sys
import glob
import logging
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths: define where data is located and where results should be stored
# ---------------------------------------------------------------------
initial_path = "data/initial"    # initial extract (Jan 2025)
delta_path   = "data/delta"      # monthly deltas (Feb, Mar, etc.)
final_path   = "data/final"      # final state of all tables
report_path  = "data/report"     # aggregated reporting output
logger_path  = "logger/logger.log"  # log file location

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
os.makedirs(os.path.dirname(logger_path), exist_ok=True)

LOGGER = logging.getLogger("support_pipeline")
LOGGER.setLevel(logging.INFO)

# File handler (writes logs to file)
_file_handler = logging.FileHandler(logger_path, encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_file_handler.setFormatter(_formatter)

# Console handler (prints logs to console as well)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)

# Avoid duplicate handlers if script is re-run
if not LOGGER.handlers:
    LOGGER.addHandler(_file_handler)
    LOGGER.addHandler(_console_handler)

# ---------------------------
# Helpers
# ---------------------------

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from column names to standardize them."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def read_csv_required(path: str) -> pd.DataFrame:
    """Load a CSV file and normalize column names. Raise error if missing."""
    np_path = os.path.normpath(path)
    if not os.path.exists(np_path):
        raise FileNotFoundError(f"Missing required file: {np_path}")
    df = pd.read_csv(np_path)
    return _normalize_cols(df)

def ensure_unknown_member(df: pd.DataFrame, id_col: str, name_cols: List[str]) -> pd.DataFrame:
    """
    Ensure the dimension table has a special 'UNKNOWN' row.
    Used when facts reference deleted/missing dimension values.
    """
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    if "UNKNOWN" not in set(df[id_col]):
        pad = {id_col: "UNKNOWN"}
        for c in name_cols:
            if c in df.columns:
                pad[c] = "Unknown"
        for c in df.columns:
            if c not in pad:
                pad[c] = pd.NA
        df = pd.concat([df, pd.DataFrame([pad])], ignore_index=True)
    return df

def apply_delta(base: pd.DataFrame, delta: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Apply a delta file to the base table:
      - DELETE: remove rows with given IDs
      - ADD/UPDATE: upsert (replace existing ID, or add new one)
    """
    base = base.copy()
    base[id_col] = base[id_col].astype(str)
    d = delta.copy()
    if "action" not in d.columns:
        raise ValueError("Delta missing 'action' column")
    d["action"] = d["action"].str.upper().str.strip()
    d[id_col] = d[id_col].astype(str)

    # If multiple actions for same ID, keep only the last one
    d = d.drop_duplicates(subset=[id_col], keep="last")

    # DELETE -> drop IDs from base
    del_ids = d.loc[d["action"].eq("DELETE"), id_col].tolist()
    if del_ids:
        base = base[~base[id_col].isin(del_ids)]

    # ADD/UPDATE -> upsert
    upserts = d[d["action"].isin(["ADD", "UPDATE"])].drop(columns=["action"], errors="ignore")
    if not upserts.empty:
        base = base[~base[id_col].isin(upserts[id_col])]
        base = pd.concat([base, upserts], ignore_index=True)

    return base

def coerce_utc_to_est(series_utc: pd.Series) -> pd.Series:
    """
    Convert UTC timestamps to EST.
    If conversion fails (e.g. missing tz), shift by -5 hours.
    """
    s = pd.to_datetime(series_utc, utc=True, errors="coerce")
    try:
        return s.dt.tz_convert("EST")
    except Exception:
        return s.dt.tz_localize(None) - pd.Timedelta(hours=5)

def save_df(df: pd.DataFrame, path_base: str, fmt: str):
    """Save DataFrame in csv/json/parquet format."""
    base = os.path.normpath(path_base)
    if fmt == "csv":
        df.to_csv(base + ".csv", index=False)
    elif fmt == "json":
        df.to_json(base + ".json", orient="records")
    elif fmt == "parquet":
        df.to_parquet(base + ".parquet", index=False)
    else:
        raise ValueError("Unsupported format: " + fmt)

def discover_deltas(delta_dir: str, months: Optional[List[str]]) -> Dict[str, List[str]]:
    """
    Find delta files in the delta directory, filtered by table and months.
    Returns mapping of table_name -> list of delta file paths in chronological order.
    """
    mapping = {"agents":[], "contact_centers":[], "service_categories":[], "interactions":[]}
    if not os.path.isdir(delta_dir):
        return mapping
    for path in glob.glob(os.path.join(delta_dir, "*_delta_*.csv")):
        fname = os.path.basename(path)
        if "_delta_" not in fname:
            continue
        left, right = fname.split("_delta_", 1)
        table = left
        yyyymm = os.path.splitext(right)[0]
        if table in mapping and (months is None or yyyymm in months):
            mapping[table].append(os.path.normpath(path))
    for k in mapping:
        mapping[k].sort(key=lambda p: os.path.basename(p).split("_delta_")[1].split(".")[0])
    return mapping

# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(
    output_format: str = "csv",
    months: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Run the ETL pipeline:
      1. Load initial tables
      2. Apply deltas in chronological order
      3. Add UNKNOWN members to dimensions
      4. Repair fact foreign keys pointing to deleted dimensions
      5. Persist final state of all tables
      6. Build aggregated support_report
    """

    os.makedirs(final_path, exist_ok=True)
    os.makedirs(report_path, exist_ok=True)

    # --- Step 1: Load initial data
    LOGGER.info("Loading initial data...")
    agents = read_csv_required(os.path.join(initial_path, "agents.csv"))
    centers = read_csv_required(os.path.join(initial_path, "contact_centers.csv"))
    cats = read_csv_required(os.path.join(initial_path, "service_categories.csv"))
    facts = read_csv_required(os.path.join(initial_path, "interactions.csv"))

    LOGGER.info("Initial rows — agents=%d, centers=%d, categories=%d, interactions=%d",
                len(agents), len(centers), len(cats), len(facts))

    # --- Step 2: Apply deltas
    LOGGER.info("Applying delta files...")
    deltas = discover_deltas(delta_path, months)
    id_map = {"agents":"agent_id", "contact_centers":"contact_center_id", "service_categories":"category_id", "interactions":"interaction_id"}
    tables = {"agents":agents, "contact_centers":centers, "service_categories":cats, "interactions":facts}
    for t, paths in deltas.items():
        for pth in paths:
            LOGGER.info("Applying %s delta: %s", t, os.path.basename(pth))
            ddf = read_csv_required(pth)
            before = len(tables[t])
            tables[t] = apply_delta(tables[t], ddf, id_map[t])
            LOGGER.info(" -> %s rows: %d -> %d", t, before, len(tables[t]))

    # Ensure department column exists in service_categories
    if "department" not in tables["service_categories"].columns:
        tables["service_categories"]["department"] = "Unknown"

    # --- Step 3: Add UNKNOWN members
    LOGGER.info("Ensuring UNKNOWN members in dimensions...")
    tables["agents"] = ensure_unknown_member(tables["agents"], "agent_id", ["agent_name"] if "agent_name" in tables["agents"].columns else [])
    tables["contact_centers"] = ensure_unknown_member(tables["contact_centers"], "contact_center_id", ["contact_center_name"] if "contact_center_name" in tables["contact_centers"].columns else [])
    name_cols = ["department"] + (["category_name"] if "category_name" in tables["service_categories"].columns else [])
    tables["service_categories"] = ensure_unknown_member(tables["service_categories"], "category_id", name_cols)

    # --- Step 4: Repair FK references in fact table
    LOGGER.info("Repairing foreign key references in interactions...")
    f = tables["interactions"].copy()
    for col in ["agent_id","contact_center_id","category_id"]:
        if col in f.columns:
            f[col] = f[col].astype(str)
    valid_agents = set(tables["agents"]["agent_id"].astype(str))
    valid_centers = set(tables["contact_centers"]["contact_center_id"].astype(str))
    valid_cats = set(tables["service_categories"]["category_id"].astype(str))

    missing_agents = (~f["agent_id"].isin(valid_agents)).sum() if "agent_id" in f.columns else 0
    missing_centers = (~f["contact_center_id"].isin(valid_centers)).sum() if "contact_center_id" in f.columns else 0
    missing_cats = (~f["category_id"].isin(valid_cats)).sum() if "category_id" in f.columns else 0
    LOGGER.info("Missing FK counts — agents=%d, centers=%d, categories=%d",
                missing_agents, missing_centers, missing_cats)

    if "agent_id" in f.columns:
        f.loc[~f["agent_id"].isin(valid_agents), "agent_id"] = "UNKNOWN"
    if "contact_center_id" in f.columns:
        f.loc[~f["contact_center_id"].isin(valid_centers), "contact_center_id"] = "UNKNOWN"
    if "category_id" in f.columns:
        f.loc[~f["category_id"].isin(valid_cats), "category_id"] = "UNKNOWN"
    tables["interactions"] = f

    # --- Step 5: Save final tables
    LOGGER.info("Saving final tables to %s ...", final_path)
    def save_final(name, df):
        base = os.path.join(final_path, f"{name}_final")
        save_df(df, base, output_format)
        return f"{base}.{output_format}"

    paths = {
        "agents_final": save_final("agents", tables["agents"]),
        "contact_centers_final": save_final("contact_centers", tables["contact_centers"]),
        "service_categories_final": save_final("service_categories", tables["service_categories"]),
        "interactions_final": save_final("interactions", tables["interactions"]),
    }

    # --- Step 6: Build support_report
    LOGGER.info("Building support_report...")
    inter = tables["interactions"].copy()

    ts_col = next((c for c in ["timestamp","interaction_timestamp","created_at","interaction_time"] if c in inter.columns), None)
    if ts_col is None:
        raise ValueError("No timestamp column found in interactions")

    est = coerce_utc_to_est(inter[ts_col])
    inter["month"] = est.dt.strftime("%Y-%m")

    centers = tables["contact_centers"][["contact_center_id","contact_center_name"]] if "contact_center_name" in tables["contact_centers"].columns else tables["contact_centers"][["contact_center_id"]].assign(contact_center_name="Unknown")
    centers = centers.copy()
    centers["contact_center_id"] = centers["contact_center_id"].astype(str)
    inter["contact_center_id"] = inter["contact_center_id"].astype(str)
    inter = inter.merge(centers, on="contact_center_id", how="left")
    inter["contact_center_name"] = inter["contact_center_name"].fillna("Unknown")

    cats = tables["service_categories"][["category_id","department"]].copy()
    cats["category_id"] = cats["category_id"].astype(str)
    inter["category_id"] = inter["category_id"].astype(str)
    inter = inter.merge(cats, on="category_id", how="left")
    inter["department"] = inter["department"].fillna("Unknown")

    ch = inter["channel"].str.lower().to_numpy() if "channel" in inter.columns else np.array([""] * len(inter))
    inter["is_call"] = (ch == "phone")

    dur_col = next((c for c in ["call_duration_minutes"] if c in inter.columns), None)
    if dur_col is None:
        inter["duration_value"] = 0
        dur_col = "duration_value"
    inter[dur_col] = pd.to_numeric(inter[dur_col], errors="coerce").fillna(0)

    report = (
        inter.groupby(["month","contact_center_name","department"], dropna=False)
            .agg(total_interactions=("interaction_id","count"),
                 total_calls=("is_call","sum"),
                 total_call_duration=(dur_col,"sum"))
            .reset_index()
            .sort_values(["month","contact_center_name","department"])
    )

    report_base = os.path.join(report_path, "support_report")
    save_df(report, report_base, output_format)
    paths["support_report"] = f"{report_base}.{output_format}"

    LOGGER.info("Report written to %s", paths["support_report"])
    LOGGER.info("Done.")
    return paths

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    import argparse
    if len(sys.argv) == 1:
        # Default run: CSV format, process ALL deltas
        LOGGER.info("No arguments provided. Running with defaults: format=csv, all months.")
        run_pipeline(output_format="csv", months=None)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--format", default="csv", choices=["csv","json","parquet"])
        parser.add_argument("--months", default=None, help="Comma-separated YYYYMM list to LIMIT processing (e.g., 202502,202503). Omit to process ALL deltas.")
        args = parser.parse_args()
        months = args.months.split(",") if args.months else None
        if months is None:
            LOGGER.info("No --months provided: processing ALL available delta months.")
        else:
            LOGGER.info("Restricting to delta months: %s", ",".join(months))

        run_pipeline(output_format=args.format, months=months)
