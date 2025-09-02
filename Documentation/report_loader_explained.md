# Report Loader & Business Questions Script — Line-by-Line Commentary

This document explains the `report loader` script line by line, with details about **purpose**, **syntax**, and **behavior**.

---

## Imports

```python
import os
```
- Loads Python’s **os** module for filesystem operations (paths, extensions).

```python
import pandas as pd
```
- Imports **Pandas** as `pd`. Used for reading/writing tabular data, grouping, and analysis.

---

## Report Path

```python
# Use a base path (no extension). Your pipeline writes one of these:
#   data/report/support_report.csv | .json | .parquet
REPORT_BASE = os.path.join("data", "report", "support_report")
```
- Defines a base path for the `support_report` without an extension.  
- The pipeline may output `.csv`, `.json`, or `.parquet`.  
- `os.path.join` safely constructs a path for any OS.

---

## Report Loader

```python
def load_report_any(base: str) -> pd.DataFrame:
```
- Function to load a report file in any of the supported formats.  
- Input: base filename (string). Output: Pandas DataFrame.

```python
    _, ext = os.path.splitext(base)
    if ext.lower() in {".csv", ".json", ".parquet"}:
        return _read_with_ext(base)
```
- Splits path into `(root, extension)`.  
- If the user already passed a path with an extension, read it directly with helper.

```python
    candidates = [base + ".csv", base + ".parquet", base + ".json"]
    last_err = None
```
- Defines possible candidates if the base has no extension.  
- Tracks the last error encountered when trying to read.

```python
    for p in candidates:
        if os.path.exists(p):
            try:
                return _read_with_ext(p)
            except Exception as e:
                last_err = e  # try next
```
- Iterates over candidates. If file exists, try reading it.  
- If reading fails, remember the error and try the next.

```python
    if last_err:
        raise RuntimeError(f"Found a report file but failed to read it: {last_err}")
```
- If a file was found but all reads failed, raise `RuntimeError` with the last error.

```python
    raise FileNotFoundError(
        f"Missing report file: tried {', '.join(candidates)}. Run pipeline.py first."
    )
```
- If none of the candidate files exist, raise a clear error instructing user to run the pipeline.

---

## Format-Specific Reader

```python
def _read_with_ext(path_with_ext: str) -> pd.DataFrame:
```
- Helper function (leading `_` marks it internal).  
- Reads a report file given a path that already includes an extension.

```python
    ext = os.path.splitext(path_with_ext)[1].lower()
```
- Extracts the file extension.

```python
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
```
- Branches on extension to choose appropriate Pandas reader.  
- `.json` is read in `records` orientation (list of dicts).  
- Raises `ValueError` if extension is unknown.

---

## Main Analysis

```python
def main():
    df = load_report_any(REPORT_BASE)
```
- Loads the report DataFrame from one of the valid formats.

```python
    # Only Q1 2025 (Jan, Feb, Mar 2025)
    q1 = df[df["month"].isin(["2025-01","2025-02","2025-03"])].copy()
```
- Filters rows for Q1 2025 based on the `month` column.  
- `.isin([...])` checks membership. `.copy()` avoids SettingWithCopy warnings.

```python
    print("\n--- Q1 2025 Business Questions ---\n")
```
- Prints a header for clarity.

---

### Q1 — Total Interactions by Contact Center

```python
    total_by_center = (
        q1.groupby("contact_center_name")["total_interactions"]
        .sum()
        .reset_index()
        .sort_values("total_interactions", ascending=False)
    )
```
- Groups by contact center, sums interactions, resets the index, and sorts by descending count.

```python
    print("1) Total interactions by contact center in Q1 2025:")
    print(total_by_center.to_string(index=False))
    print()
```
- Prints the result table without row indices.

---

### Q2 — Month with Highest Interaction Volume

```python
    month_totals = (
        q1.groupby("month")["total_interactions"]
        .sum()
        .reset_index()
        .sort_values("total_interactions", ascending=False)
    )
```
- Groups by month, sums total interactions, sorts descending.

```python
    top_month = month_totals.iloc[0]
```
- Selects the first row (highest volume).

```python
    print("2) Month with highest total interaction volume:")
    print(month_totals.to_string(index=False))
    print(f"\nHighest = {top_month['month']} with {top_month['total_interactions']} interactions\n")
```
- Prints the month totals and highlights the top performer.

---

### Q3 — Longest Average Phone Call Duration

```python
    total_duration = (
        q1.groupby("contact_center_name")
        .agg(
            total_calls=("total_calls", "sum"),
            total_call_duration=("total_call_duration", "sum")
        )
        .reset_index()
    )
```
- Groups by contact center and aggregates total calls and total call duration.

```python
    total_duration["avg_call_duration"] = total_duration.apply(
        lambda r: (r["total_call_duration"] / r["total_calls"]) if r["total_calls"] > 0 else 0,
        axis=1
    )
```
- Calculates per-row average call duration. Uses `apply` with `axis=1` (row-wise).

```python
    avg_duration_sorted = total_duration.sort_values("avg_call_duration", ascending=False)
```
- Sorts by longest average call duration.

```python
    print("3) Contact center with longest average phone call duration:")
    print(avg_duration_sorted.to_string(index=False))
    top_center = avg_duration_sorted.iloc[0]
    print(f"\nLongest average duration = {top_center['contact_center_name']} "
          f"({top_center['avg_call_duration']:.2f} seconds)\n")
```
- Prints sorted results and highlights the top contact center with formatted float.

---

### Q4 — Department Mix for Top Contact Center

```python
    dept_mix = (
        q1[q1["contact_center_name"] == top_center["contact_center_name"]]
        .groupby("department")
        .agg(total_calls=("total_calls","sum"),
             total_duration=("total_call_duration","sum"),
             total_interactions=("total_interactions","sum"))
        .reset_index()
        .sort_values("total_duration", ascending=False)
    )
```
- Filters rows for the top contact center, groups by department, aggregates calls, duration, and interactions, sorts by duration.

```python
    print("4) Department mix for top contact center:")
    print(dept_mix.to_string(index=False))
    print("\nInterpretation:")
```
- Prints the department breakdown and a placeholder for interpretation.

```python
    if top_center["avg_call_duration"] == 0:
        print("   All recorded durations are 0 → suggests missing or unrecorded call durations.")
    else:
        print("   Departments with higher total_duration likely drive the longer average calls.")
```
- Provides simple interpretation logic based on whether durations were recorded.

---

### Q5 — Recommendations

```python
    print("\n5) Recommended approach to measure agent work time more accurately:")
    print(
   "- Track explicit agent states (Login, Ready, Not Ready, On Call, After-Call Work, Break)\n"
   "- Separate talk time from After-Call Work (ACW)\n"
   "- Capture queue wait, transfers, and hold times\n"
   "- Log multi-channel concurrency (chat/email vs phone overlap)\n"
   "- Use session heartbeats to detect disconnects\n"
   "- Compare actual states vs schedules to calculate adherence/shrinkage"
    )
```
- Prints out multi-line recommendations. Used concatenated string instead of triple quotes to avoid confusion.

---

## Script Entrypoint

```python
if __name__ == "__main__":
    main()
```
- Standard idiom: only run `main()` when the file is executed as a script, not when imported as a module.

---

## Key Takeaways

- Flexible file loading: works with `.csv`, `.json`, `.parquet`.  
- Pandas groupby/agg/sort pipelines answer business questions.  
- Simple printed outputs for clarity.  
- Designed as a CLI-friendly reporting utility on top of pipeline outputs.
