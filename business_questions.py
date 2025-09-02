import os
import pandas as pd

# Use a base path (no extension). Your pipeline writes one of these:
#   data/report/support_report.csv | .json | .parquet
REPORT_BASE = os.path.join("data", "report", "support_report")

def load_report_any(base: str) -> pd.DataFrame:
    """
    Try reading <base>.csv, then <base>.parquet, then <base>.json (records).
    If `base` already ends with a supported extension, just read that.
    """
    # If user passed an explicit path with extension, honor it.
    _, ext = os.path.splitext(base)
    if ext.lower() in {".csv", ".json", ".parquet"}:
        return _read_with_ext(base)

    candidates = [base + ".csv", base + ".parquet", base + ".json"]
    last_err = None
    for p in candidates:
        if os.path.exists(p):
            try:
                return _read_with_ext(p)
            except Exception as e:
                last_err = e  # try next
    if last_err:
        raise RuntimeError(f"Found a report file but failed to read it: {last_err}")
    raise FileNotFoundError(
        f"Missing report file: tried {', '.join(candidates)}. Run pipeline.py first."
    )

def _read_with_ext(path_with_ext: str) -> pd.DataFrame:
    ext = os.path.splitext(path_with_ext)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path_with_ext)
    elif ext == ".parquet":
        # Requires pyarrow or fastparquet installed
        return pd.read_parquet(path_with_ext)
    elif ext == ".json":
        # Pipeline writes JSON as records; this handles either records or a list of dicts
        return pd.read_json(path_with_ext, orient="records")
    else:
        raise ValueError(f"Unsupported report extension: {ext}")

def main():
    df = load_report_any(REPORT_BASE)

    # Only Q1 2025 (Jan, Feb, Mar 2025)
    q1 = df[df["month"].isin(["2025-01","2025-02","2025-03"])].copy()

    print("\n--- Q1 2025 Business Questions ---\n")

    # Q1) Total interactions by contact center
    total_by_center = (
        q1.groupby("contact_center_name")["total_interactions"]
        .sum()
        .reset_index()
        .sort_values("total_interactions", ascending=False)
    )
    print("1) Total interactions by contact center in Q1 2025:")
    print(total_by_center.to_string(index=False))
    print()

    # Q2) Month with highest interaction volume
    month_totals = (
        q1.groupby("month")["total_interactions"]
        .sum()
        .reset_index()
        .sort_values("total_interactions", ascending=False)
    )
    top_month = month_totals.iloc[0]
    print("2) Month with highest total interaction volume:")
    print(month_totals.to_string(index=False))
    print(f"\nHighest = {top_month['month']} with {top_month['total_interactions']} interactions\n")

    # Q3) Contact center with longest average phone call duration
    total_duration = (
        q1.groupby("contact_center_name")
        .agg(
            total_calls=("total_calls", "sum"),
            total_call_duration=("total_call_duration", "sum")
        )
        .reset_index()
    )
    total_duration["avg_call_duration"] = total_duration.apply(
        lambda r: (r["total_call_duration"] / r["total_calls"]) if r["total_calls"] > 0 else 0,
        axis=1
    )
    avg_duration_sorted = total_duration.sort_values("avg_call_duration", ascending=False)
    print("3) Contact center with longest average phone call duration:")
    print(avg_duration_sorted.to_string(index=False))
    top_center = avg_duration_sorted.iloc[0]
    print(f"\nLongest average duration = {top_center['contact_center_name']} "
          f"({top_center['avg_call_duration']:.2f} seconds)\n")

    # Q4) Why might this be the case?
    dept_mix = (
        q1[q1["contact_center_name"] == top_center["contact_center_name"]]
        .groupby("department")
        .agg(total_calls=("total_calls","sum"),
             total_duration=("total_call_duration","sum"),
             total_interactions=("total_interactions","sum"))
        .reset_index()
        .sort_values("total_duration", ascending=False)
    )
    print("4) Department mix for top contact center:")
    print(dept_mix.to_string(index=False))
    print("\nInterpretation:")
    if top_center["avg_call_duration"] == 0:
        print("   All recorded durations are 0 → suggests missing or unrecorded call durations.")
    else:
        print("   Departments with higher total_duration likely drive the longer average calls.")

    # Q5) Recommendations for measuring agent work time more accurately
    print("\n5) Recommended approach to measure agent work time more accurately:")
    print("""\
   - Track explicit agent states (Login, Ready, Not Ready, On Call, After-Call Work, Break)
   - Separate talk time from After-Call Work (ACW)
   - Capture queue wait, transfers, and hold times
   - Log multi-channel concurrency (chat/email vs phone overlap)
   - Use session heartbeats to detect disconnects
   - Compare actual states vs schedules to calculate adherence/shrinkage""")

if __name__ == "__main__":
    main()
