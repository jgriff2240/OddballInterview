# Contact Center Pipeline

A Python 3.12 project that processes **contact center interaction data** with support for incremental updates, validation, and reporting.  

---

## 📦 Dependencies

This project requires the following libraries:

- [numpy](https://numpy.org/)  
- [pandas](https://pandas.pydata.org/)  
- [pytest](https://docs.pytest.org/)  
- [pyarrow](https://arrow.apache.org/docs/python/)

---

## ▶️ How to Run

1. **Navigate to the project directory**  
   ```bash
   cd C:\Users\Jackson\PycharmProjects\Oddball\OddballInterview
   ```

2. **Run the pipeline**  
   ```bash
   python3 pipeline.py
   ```

3. **Run business questions script**  
   ```bash
   python3 business_questions.py
   ```

---

## ⚙️ Options

- **Reset existing files** (start fresh from initial data):  
  ```bash
  python3 pipeline.py --reset
  ```

- **Incremental monthly updates** (example: process February & March 2025):  
  ```bash
  python3 pipeline.py --months=202502 --format=json --reset
  python3 pipeline.py --months=202503 --format=json
  ```

---

## ✅ Running Tests

Choose one of the following:

```bash
py -3.12 -m pytest -q
```

or  

```bash
python3 validate.py
```

---

## 📊 Business Questions & Answers

### 1. Total interactions handled by each contact center in Q1 2025
```text
contact_center_name   total_interactions
Boston MA NE                       13
Atlanta GA SE                       8
Richmond VA E                       7
```

**Code snippet**:
```python
q1 = df[df["month"].isin(["2025-01", "2025-02", "2025-03"])].copy()
total_by_center = (
    q1.groupby("contact_center_name")["total_interactions"]
    .sum()
    .reset_index()
    .sort_values("total_interactions", ascending=False)
)
```

---

### 2. Which month had the highest total interaction volume?
```text
month     total_interactions
2025-02   10
2025-01    9
2025-03    9

Highest = 2025-02 with 10 interactions
```

**Code snippet**:
```python
month_totals = (
    q1.groupby("month")["total_interactions"]
    .sum()
    .reset_index()
    .sort_values("total_interactions", ascending=False)
)
```

---

### 3. Which contact center had the longest average phone call duration?
```text
contact_center_name  total_calls  total_call_duration  avg_call_duration
Boston MA NE                  11                 140.0          12.727273
Richmond VA E                  5                  62.0          12.400000
Atlanta GA SE                  5                  54.0          10.800000

Longest average duration = Boston MA NE (12.73 seconds)
```

**Code snippet**:
```python
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
```
