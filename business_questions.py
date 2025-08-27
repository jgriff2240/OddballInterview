import os
import pandas as pd

REPORT_PATH = os.path.join("data", "report", "support_report.csv")

def main():
    if not os.path.exists(REPORT_PATH):
        raise FileNotFoundError(f"Missing report file: {REPORT_PATH}. Run pipeline.py first.")

    df = pd.read_csv(REPORT_PATH)

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
    avg_duration = (
        q1.groupby("contact_center_name")
        .agg(total_calls=("total_calls","sum"),
             total_call_duration=("total_call_duration","sum"))
        .reset_index()
    )
    avg_duration["avg_call_duration"] = avg_duration.apply(
        lambda r: (r["total_call_duration"] / r["total_calls"]) if r["total_calls"] > 0 else 0,
        axis=1
    )
    avg_duration_sorted = avg_duration.sort_values("avg_call_duration", ascending=False)
    print("3) Contact center with longest average phone call duration:")
    print(avg_duration_sorted.to_string(index=False))
    top_center = avg_duration_sorted.iloc[0]
    print(f"\nLongest average duration = {top_center['contact_center_name']} "
          f"({top_center['avg_call_duration']:.2f} seconds)\n")

    # Q4) Why might this be the case?
    # Look at department mix for the top center
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
