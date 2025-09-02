# tests/test_pipeline_validation.py
# -------------------------------------------------------------
# Unit tests using pytest.
# These tests import the same validators used by validate.py.
# They let you run `pytest` locally or in CI to enforce rules.
# -------------------------------------------------------------

import pandas as pd
from validators import run_core_validations, load_any

def test_core_validations_pass():
    """Check that all core validation checks pass."""
    ok, messages = run_core_validations()
    assert ok, "Validation failed with messages: {}".format(messages)

def test_report_loads():
    """Check that the support_report can be loaded in any format."""
    df = load_any("data/report/support_report")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
