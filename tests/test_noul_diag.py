"""Structural tests for tools/noul_diag.py — no weights required.

Weight-dependent regression (running the diagnostic against a live checkpoint)
is covered by tests/test_local_e2e.py::noul-label-bias.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


# --------------------------------------------------------------- import resolves
from tools import noul_diag
from tools.noul_diag import check_noul_bias, format_report

check_true("tools.noul_diag imports", True)
check_true("check_noul_bias is callable", callable(check_noul_bias))
check_true("format_report is callable", callable(format_report))

# --------------------------------------------------------------- structural checks
check_true("has positive state", isinstance(noul_diag.POSITIVE_STATE, dict) and len(noul_diag.POSITIVE_STATE) > 0)
check_true("has negative state", isinstance(noul_diag.NEGATIVE_STATE, dict) and len(noul_diag.NEGATIVE_STATE) > 0)
check("basic question type is noul", noul_diag.BASIC_QUESTION["type"], "noul")
check_true("rich question has criteria", "criteria" in noul_diag.RICH_QUESTION)
check("rich criteria keys", set(noul_diag.RICH_QUESTION["criteria"].keys()), {"true", "false"})
check("custom labels non-boolean", set(noul_diag.CUSTOM_LABELS.values()), {"A", "B"})

# --------------------------------------------------------------- format_report works
mock_report = {
    "questions": {
        "plain": {"pass": False, "cells": {
            "positive": {"P(true)": 0.0, "confidence": 1.0, "want_true": True, "pass": False},
            "negative": {"P(true)": 0.0, "confidence": 1.0, "want_true": False, "pass": True},
        }},
        "custom": {"pass": True, "cells": {
            "positive": {"P(true)": 0.95, "confidence": 0.9, "want_true": True, "pass": True},
            "negative": {"P(true)": 0.05, "confidence": 0.85, "want_true": False, "pass": True},
        }},
    },
    "biased": True,
}
formatted = format_report(mock_report)
check_true("format_report shows BIASED", "BIASED" in formatted)
check_true("format_report shows overall", "overall:" in formatted)
check_true("format_report shows plain", "plain" in formatted)
check_true("format_report shows custom", "custom" in formatted)

# --------------------------------------------------------------- report shape contract
report = check_noul_bias.__wrapped__ if hasattr(check_noul_bias, "__wrapped__") else check_noul_bias
import inspect
sig = inspect.signature(report)
check_true("check_noul_bias accepts agent param", "agent" in sig.parameters)
check_true("check_noul_bias accepts states param", "states" in sig.parameters)
check_true("check_noul_bias accepts questions param", "questions" in sig.parameters)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
