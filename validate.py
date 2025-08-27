#!/usr/bin/env python3
# validate.py
# -------------------------------------------------------------
# Command-line validation runner.
# Calls the reusable checks in validators.py, prints results,
# and returns an appropriate exit code (0=pass, 1=fail).
# This allows integration into CI/CD pipelines.
# -------------------------------------------------------------

import sys
from validators import run_core_validations

def main():
    ok, messages = run_core_validations()
    if ok:
        print("✅ Validation PASSED: all checks succeeded.")
        sys.exit(0)   # success exit code
    else:
        print("❌ Validation FAILED:")
        for m in messages:
            print(" -", m)
        sys.exit(1)   # failure exit code (causes CI to fail)

if __name__ == "__main__":
    main()
