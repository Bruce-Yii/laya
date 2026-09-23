"""Tests for tools.calibration.fit_temperatures — offline, no real checkpoint."""
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

a = object.__new__(Agent)
a.device = torch.device("cpu")
a.dtype = torch.float32
a.cfg = {"max_len": 512, "head_max_len": 192}
a.tok = StubTok()
a.model = StubModel([])
a.temperature = [1.0, 1.0, 1.0]
a.temperature_by_options = {}
a.model_name = "stub"

from tools.calibration import fit_temperatures

# Over-confident wrong samples -> optimal T should soften (T > 1).
states = [{"body": "wrong 1"}, {"body": "wrong 2"}, {"body": "wrong 3"},
          {"body": "correct 1"}, {"body": "correct 2"}]
questions = [{"q": {"type": "choice", "instructions": "Which?",
               "criteria": {"a": None, "b": None, "c": None}}}] * 5
targets = [0, 0, 0, 1, 2]  # first 3 are wrong (logits peak at 1, target=0)

a.model.logits_per_sample = [
    [0.0, 10.0, 0.0],   # peaks at 1, target=0 -> WRONG
    [0.0, 10.0, 0.0],   # peaks at 1, target=0 -> WRONG
    [0.0, 10.0, 0.0],   # peaks at 1, target=0 -> WRONG
    [0.0, 10.0, 0.0],   # peaks at 1, target=1 -> CORRECT
    [0.0, 0.0, 10.0],   # peaks at 2, target=2 -> CORRECT
]

result = fit_temperatures(a, states, questions, targets, t_range=(1.0, 5.0))

check_true("fit returns temperature", "temperature" in result)
check_true("fit returns buckets", "ece" in result)
check_true("has choice:3-5 bucket", "choice:3-5" in result["temperature_by_options"])
t_fit = result["temperature_by_options"]["choice:3-5"]
check_true("over-confident wrong: T > 1.0", t_fit > 1.0, "(got %.4f)" % t_fit)
check_true("ECE reported", "choice:3-5" in result["ece"])
check_true("agent temperature NOT modified (caller applies)", a.temperature == [1.0, 1.0, 1.0])

# Under-confident correct samples -> optimal T should sharpen (T < 1).
a.temperature = [1.0, 1.0, 1.0]
a.temperature_by_options = {}
a.model.logits_per_sample = [
    [0.5, 0.4, 0.3],   # target=0, correct but low conf
    [0.3, 0.5, 0.4],   # target=1, correct but low conf
    [0.4, 0.3, 0.5],   # target=2, correct but low conf
    [0.5, 0.4, 0.3],   # target=0, correct but low conf
    [0.3, 0.5, 0.4],   # target=1, correct but low conf
]
targets_under = [0, 1, 2, 0, 1]
result2 = fit_temperatures(a, states, questions, targets_under, t_range=(0.5, 2.0))
t_fit2 = result2["temperature_by_options"]["choice:3-5"]
check_true("under-confident correct: T < 1.0", t_fit2 < 1.0, "(got %.4f)" % t_fit2)

# Target format: str labels and bool work too.
a.temperature = [1.0, 1.0, 1.0]
a.temperature_by_options = {}
a.model.logits_per_sample = [
    [0.0, 10.0, 0.0],   # peaks at b
    [0.0, 10.0, 0.0],   # peaks at b
]
states_str = [{"body": "s1"}, {"body": "s2"}]
questions_str = [
    {"q": {"type": "choice", "instructions": "Which?",
     "criteria": {"billing": None, "tech": None, "other": None}}},
    {"q": {"type": "choice", "instructions": "Which?",
     "criteria": {"billing": None, "tech": None, "other": None}}},
]
targets_str = ["tech", "tech"]  # label strings
result3 = fit_temperatures(a, states_str, questions_str, targets_str, t_range=(1.0, 5.0))
t_fit3 = result3["temperature_by_options"]["choice:3-5"]
check_true("str targets: T > 1.0 (both correct)", t_fit3 > 1.0, "(got %.4f)" % t_fit3)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
