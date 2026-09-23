"""Diagnostic utilities for detecting boolean-label bias in noul questions.

The shipped Laya checkpoints can anchor on literal boolean option labels
(true/false, yes/no) and select the negative answer regardless of the input
state. This module runs the bias check from the project's issue tracker so
users can verify whether their question setup is affected and tune labels
before trusting noul probabilities.

Deprecation note: when the default noul model-facing labels are changed
(e.g. via the fix for issue #156 / PR #163), this tool remains useful for
verifying older checkpoints and custom deployments but will be less necessary
for new users on fixed checkpoints.
"""
from typing import Dict, List, Optional, Tuple


POSITIVE_STATE = {"body": "This product is excellent quality - six months in and not a single problem."}
NEGATIVE_STATE = {"body": "It arrived broken and nobody answers when I contact support."}

BASIC_QUESTION = {"type": "noul", "instructions": "Is this review positive?"}

RICH_QUESTION = {
    "type": "noul",
    "instructions": "Is this review positive?",
    "criteria": {"true": "the review is positive", "false": "the review is negative"},
}

CUSTOM_LABELS = {"false": "A", "true": "B"}
CUSTOM_QUESTION = {**RICH_QUESTION, "labels": CUSTOM_LABELS}


def _is_biased(got: bool, want: bool) -> bool:
    return got != want


def check_noul_bias(
    agent,
    states: Optional[Dict[str, Dict]] = None,
    questions: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """Run a noul bias diagnostic over the given agent.

    Uses the positive/negative review pair from issue #156 by default. Returns a
    dict with per-question pass/fail and an overall ``biased`` flag.
    """
    if states is None:
        states = {"positive": POSITIVE_STATE, "negative": NEGATIVE_STATE}
    if questions is None:
        questions = {
            "plain": BASIC_QUESTION,
            "rich": RICH_QUESTION,
            "custom_labels": CUSTOM_QUESTION,
        }

    report = {"questions": {}, "biased": False}
    for qid, qdef in questions.items():
        result = {"pass": True, "cells": {}}
        for label, state in states.items():
            want_true = label == "positive"
            answers = agent.predict(state, {qid: qdef})["answers"]
            probability = answers[qid]["noul"]
            confidence = answers[qid]["confidence"]
            cell = {
                "P(true)": round(probability, 4),
                "confidence": round(confidence, 4),
                "want_true": want_true,
                "pass": (probability > 0.5) == want_true,
            }
            result["cells"][label] = cell
            if not cell["pass"]:
                result["pass"] = False
                report["biased"] = True
        report["questions"][qid] = result
    return report


def format_report(report: Dict) -> str:
    lines = ["noul bias diagnostic:", ("=" * 40)]
    for qid, result in report["questions"].items():
        status = "PASS" if result["pass"] else "FAIL"
        lines.append(f"[{status}] {qid}:")
        for label, cell in result["cells"].items():
            mark = "OK" if cell["pass"] else "BIAS"
            lines.append(
                f"  {label:9s} P(true)={cell['P(true)']:.4f}  "
                f"conf={cell['confidence']:.4f}  want={cell['want_true']}  [{mark}]"
            )
    lines.append(("=" * 40))
    lines.append("overall: BIASED - boolean labels may be driving the answer" if report["biased"] else "overall: CLEAN - noul answers track the input state")
    return "\n".join(lines)
