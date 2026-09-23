"""Tests for Agent.fit_temperatures — offline, no real checkpoint needed.

Uses a stub DecisionModel that returns predetermined logits so the fitting
logic can be verified exactly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from laya.common import DecisionModel

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


class StubModel(nn.Module):
    """Returns fixed logits for each sample; ignores input."""

    def __init__(self, logits_per_sample):
        super().__init__()
        self.logits_per_sample = logits_per_sample
        self.encoder = nn.Identity()
        self.type_emb = nn.Identity()
        self.head = nn.Identity()
        self.scorer = nn.Identity()
        self.act_head = nn.Identity()
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        batch_size = input_ids.shape[0]
        logits = torch.tensor([self.logits_per_sample[i] for i in range(batch_size)],
                              dtype=torch.float32)
        act = torch.zeros(batch_size, 2)
        return logits, act


class StubTok:
    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        n = max(1, len(text.split()))
        return {"input_ids": [5] * n}


from laya.agent import Agent

# Build a minimal Agent with stub components.
a = object.__new__(Agent)
a.device = torch.device("cpu")
a.dtype = torch.float32
a.cfg = {"max_len": 512, "head_max_len": 192}
a.tok = StubTok()
a.model = StubModel([])
a.temperature = [1.0, 1.0, 1.0]
a.temperature_by_options = {}
a.model_name = "stub"

# Calibration data where over-confidence is harmful.
# Some samples are WRONG with high confidence (logits peak at wrong index).
# At T=1.0, wrong samples have high confidence → high ECE.
# At T>1.0, confidence drops → ECE improves.
states = [
    {"body": "wrong 1"},
    {"body": "wrong 2"},
    {"body": "wrong 3"},
    {"body": "correct 1"},
    {"body": "correct 2"},
]
questions = [
    {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None, "c": None}}},
    {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None, "c": None}}},
    {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None, "c": None}}},
    {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None, "c": None}}},
    {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None, "c": None}}},
]
# Targets: first 3 are "a" but logits peak at wrong index.
targets = ["a", "a", "a", "b", "c"]

# Wrong samples: logits peak at index 1 (b) but target is a (index 0).
# Correct samples: logits peak at correct index.
a.model.logits_per_sample = [
    [0.0, 10.0, 0.0],   # target=a, peaks at b → WRONG with high conf
    [0.0, 10.0, 0.0],   # target=a, peaks at b → WRONG with high conf
    [0.0, 10.0, 0.0],   # target=a, peaks at b → WRONG with high conf
    [0.0, 10.0, 0.0],   # target=b, peaks at b → CORRECT with high conf
    [0.0, 0.0, 10.0],   # target=c, peaks at c → CORRECT with high conf
]

result = a.fit_temperatures(states, questions, targets, t_range=(1.0, 5.0), n_steps=50)

check_true("fit returns temperature", "temperature" in result)
check_true("fit returns buckets", "ece" in result)
check_true("has choice:3-5 bucket", "choice:3-5" in result["temperature_by_options"])
# Over-confident wrong samples → optimal T should soften (T > 1).
t_fit = result["temperature_by_options"]["choice:3-5"]
check_true("optimal T > 1.0 (softens over-confident)", t_fit > 1.0, "(got %.4f)" % t_fit)
check_true("ECE reported", "choice:3-5" in result["ece"])
check_true("agent temperature updated", a.temperature_by_options["choice:3-5"] == t_fit)

# All wrong samples → optimal T should soften (T > 1).
a.temperature = [1.0, 1.0, 1.0]
a.temperature_by_options = {}
targets_all_wrong = ["billing", "billing", "billing", "billing", "billing"]
result2 = a.fit_temperatures(states, questions, targets_all_wrong, t_range=(1.0, 5.0), n_steps=50)
t_fit2 = result2["temperature_by_options"]["choice:3-5"]
check_true("all-wrong: optimal T > 1.0 (softens)", t_fit2 > 1.0, "(got %.4f)" % t_fit2)

# Under-confident correct samples → optimal T should sharpen (T < 1).
a.temperature = [1.0, 1.0, 1.0]
a.temperature_by_options = {}
# Logits that are correct but low confidence.
a.model.logits_per_sample = [
    [0.5, 0.4, 0.3],   # target=a, correct but low conf
    [0.3, 0.5, 0.4],   # target=b, correct but low conf
    [0.4, 0.3, 0.5],   # target=c, correct but low conf
    [0.5, 0.4, 0.3],   # target=a, correct but low conf
    [0.3, 0.5, 0.4],   # target=b, correct but low conf
]
targets_under = ["a", "b", "c", "a", "b"]
result3 = a.fit_temperatures(states, questions, targets_under, t_range=(0.5, 2.0), n_steps=50)
t_fit3 = result3["temperature_by_options"]["choice:3-5"]
check_true("under-confident: optimal T < 1.0 (sharpens)", t_fit3 < 1.0, "(got %.4f)" % t_fit3)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
