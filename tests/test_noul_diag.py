"""Tests for laya.noul_diag diagnostic utility."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import Agent
from laya.noul_diag import check_noul_bias

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


# --------------------------------------------------------------- structural checks (no weights needed)
from laya import noul_diag

check_true("noul_diag/exposes check_noul_bias", callable(noul_diag.check_noul_bias))
check_true("noul_diag/exposes format_report", callable(noul_diag.format_report))
check_true("noul_diag/has positive state", isinstance(noul_diag.POSITIVE_STATE, dict) and len(noul_diag.POSITIVE_STATE) > 0)
check_true("noul_diag/has negative state", isinstance(noul_diag.NEGATIVE_STATE, dict) and len(noul_diag.NEGATIVE_STATE) > 0)
check("noul_diag/basic question is noul", noul_diag.BASIC_QUESTION["type"], "noul")
check_true("noul_diag/rich question has criteria", "criteria" in noul_diag.RICH_QUESTION)
check("noul_diag/rich criteria has true/false keys", set(noul_diag.RICH_QUESTION["criteria"].keys()), {"true", "false"})
check("noul_diag/custom labels are non-boolean", set(noul_diag.CUSTOM_LABELS.values()), {"A", "B"})


# --------------------------------------------------------------- runtime check (weights required)
def _run_with_weights():
    import laya

    agent = laya.load("convaiinnovations/laya", device="cpu")
    report = check_noul_bias(agent)

    check_true("report/has questions", "questions" in report)
    check_true("report/has biased flag", "biased" in report)
    check_true("report/covers plain", "plain" in report["questions"])
    check_true("report/covers rich", "rich" in report["questions"])
    check_true("report/covers custom_labels", "custom_labels" in report["questions"])

    for qid, result in report["questions"].items():
        check_true(f"{qid}/has pass flag", "pass" in result)
        for label in ("positive", "negative"):
            cell = result["cells"][label]
            check_true(f"{qid}/{label}/has P(true)", "P(true)" in cell)
            check_true(f"{qid}/{label}/has confidence", "confidence" in cell)

    formatted = noul_diag.format_report(report)
    check_true("format_report/mentions overall", "overall:" in formatted)


try:
    _run_with_weights()
except Exception as e:
    FAIL.append("weight-dependent check skipped or failed: %s" % e)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
