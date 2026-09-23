"""Calibration temperature fitting for Laya agents.

The shipped multilingual checkpoint has no fitted temperatures, and the README
documents that both checkpoints are over-confident as shipped. This module
provides :func:`fit_temperatures`, which finds the temperature per bucket
(question type + option count) that minimizes Expected Calibration Error
on user-supplied calibration data.

Example::

    from tools.calibration import fit_temperatures

    result = fit_temperatures(
        agent,
        states=[{"body": "..."}, ...],
        questions=[{"q": {"type": "choice", ...}}, ...],
        targets=[0, 1, 2, ...],  # index of the correct answer
    )
    agent.temperature = result["temperature"]
    agent.temperature_by_options = result["temperature_by_options"]
"""
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from laya.common import QTYPES, TEMP_MAX, TEMP_MIN, clamp_temperature, ece_score, temp_bucket


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _to_target_index(qtype: str, target, crit_keys: Optional[List[str]] = None) -> int:
    """Unify target format to an integer index."""
    if isinstance(target, bool):
        return 1 if target else 0
    if isinstance(target, int):
        return target
    if isinstance(target, str) and crit_keys is not None:
        return crit_keys.index(target) if target in crit_keys else 0
    return 0


def fit_temperatures(
    agent,
    states: Sequence[Union[str, dict]],
    questions: Sequence[dict],
    targets: Sequence,
    t_range: tuple = (0.5, 5.0),
    n_bins: int = 15,
) -> Dict:
    """Fit calibration temperatures on user data.

    Finds, per bucket (question type + option count), the temperature that
    minimizes Expected Calibration Error on the supplied targets.

    Args:
        agent: a loaded Agent instance.
        states: list of state dicts/strs, one per calibration sample.
        questions: list of question dicts (public shape), one per sample.
        targets: list of target answers (str label / int index / bool).
        t_range: (min, max) temperature range to search.
        n_bins: number of confidence bins for ECE computation.

    Returns:
        {"temperature": [...], "temperature_by_options": {...}, "ece": {...}}
    """
    if not (len(states) == len(questions) == len(targets)):
        raise ValueError("states, questions, and targets must have the same length")

    buckets: Dict[str, List] = {}
    for idx, (state, qs, tgt) in enumerate(zip(states, questions, targets)):
        out = agent.system_one(state, qs, return_raw_logits=True)
        raw = out["usage"]["_raw_logits"]
        ids = list(qs.keys())
        for r, qid in enumerate(ids):
            q = qs[qid]
            qtype = q.get("type", q.get("t"))
            if qtype is None:
                raise ValueError("question %r missing 'type' key" % qid)
            k = len(q.get("criteria", {}).keys()) if qtype == "choice" else (
                len(q.get("criteria", [])) if qtype == "score" else 2)
            qt = QTYPES[qtype]
            logits = raw[r, :k].copy()
            crit_keys = list(q.get("criteria", {}).keys()) if qtype == "choice" else None
            target_idx = _to_target_index(qtype, tgt, crit_keys)
            bucket = temp_bucket(qt, k)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append((logits, target_idx, qt, k))

    new_temperature = list(agent.temperature)
    new_temperature_by_options = dict(agent.temperature_by_options)
    ece_report = {}

    for bucket, samples in buckets.items():
        qtype = samples[0][2]
        k = samples[0][3]
        logits = np.stack([s[0] for s in samples])
        correct = np.zeros(len(samples), dtype=float)

        for i, (lg, tgt_idx, qt, kk) in enumerate(samples):
            if qt == QTYPES["choice"]:
                pred_idx = int(lg.argmax())
                correct[i] = 1.0 if pred_idx == tgt_idx else 0.0
            elif qt == QTYPES["score"]:
                pred_idx = int((np.arange(kk) * _softmax(lg)).sum())
                correct[i] = 1.0 if pred_idx == tgt_idx else 0.0
            else:
                p = _softmax(lg)
                pred_true = 1 if p[1] > 0.5 else 0
                correct[i] = 1.0 if pred_true == tgt_idx else 0.0

        def ece_for_t(t: float) -> float:
            z = logits / t
            p = _softmax(z)
            conf = p.max(axis=1)
            return ece_score(conf, correct, bins=n_bins)

        lo, hi = t_range
        for _ in range(50):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            if ece_for_t(m1) < ece_for_t(m2):
                hi = m2
            else:
                lo = m1
        best_t = float((lo + hi) / 2)
        best_t = clamp_temperature(best_t)
        new_temperature_by_options[bucket] = best_t
        ece_report[bucket] = {"temperature": round(best_t, 4),
                              "ece": round(ece_for_t(best_t), 4)}

        if bucket == temp_bucket(qtype, 2) or bucket == temp_bucket(qtype, 3):
            new_temperature[qtype] = best_t

    return {
        "temperature": [clamp_temperature(t) for t in new_temperature],
        "temperature_by_options": {k: clamp_temperature(v)
                                  for k, v in new_temperature_by_options.items()},
        "ece": ece_report,
    }
