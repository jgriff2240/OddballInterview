# tests/test_pipeline_validation.py
# -------------------------------------------------------------
# Unit tests using pytest.
# These tests import the same validators used by validate.py.
# They let you run `pytest` locally or in CI to enforce rules.
# -------------------------------------------------------------

import pandas as pd
from validators import load_finals, load_report, run_core_validations

def test_files_exist():
    """Check that pipeline produced all final CSVs and the report."""
    finals = load_finals()
    report = load_report()
    assert isinstance(finals["agents_final"], pd.DataFrame)
    assert isinstance(report, pd.DataFrame)

def test_core_validations_pass():
    """Check that all core validation checks pass."""
    ok, messages = run_core_validations()
    assert ok, f"Validation failed with messages: {messages}"