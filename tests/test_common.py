import json

import pytest

from laya.common import render_options, render_criterion


def test_render_options_sorts_criteria():
    """Criteria keys must be sorted so JSON key order doesn't change options or model decisions.

    Issue #779 reported that changing JSON key order in criteria modified the model's selected option.
    Since choice questions treat options as ordered slots, different sequences produce different
    slot embeddings and can flip decisions. The fix sorts criterion keys deterministically.
    """
    q1 = {"t": "choice", "crit": {"b": "desc b", "a": "desc a", "c": "desc c"}}
    q2 = {"t": "choice", "crit": {"a": "desc a", "b": "desc b", "c": "desc c"}}
    q3 = {"t": "choice", "crit": {"c": "desc c", "a": "desc a", "b": "desc b"}}

    opts1 = render_options(q1)
    opts2 = render_options(q2)
    opts3 = render_options(q3)

    expected = ["a: desc a", "b: desc b", "c: desc c"]
    assert opts1 == expected, f"Q1 failed: {opts1}"
    assert opts2 == expected, f"Q2 failed: {opts2}"
    assert opts3 == expected, f"Q3 failed: {opts3}"
    assert opts1 == opts2 == opts3, "Options must be identical across all key orders"


def test_render_options_with_numeric_keys():
    """Numeric keys should sort numerically when compared as strings."""
    q1 = {"t": "choice", "crit": {2: "desc 2", 1: "desc 1", 3: "desc 3"}}
    q2 = {"t": "choice", "crit": {3: "desc 3", 1: "desc 1", 2: "desc 2"}}

    opts1 = render_options(q1)
    opts2 = render_options(q2)

    expected = ["1: desc 1", "2: desc 2", "3: desc 3"]
    assert opts1 == expected, f"Q1 failed: {opts1}"
    assert opts2 == expected, f"Q2 failed: {opts2}"
    assert opts1 == opts2, "Options must be identical across different key orders"


def test_render_options_preserves_descriptions():
    """Descriptions must remain unchanged; only key order is normalized."""
    q = {"t": "choice", "crit": {"zebra": "last one", "apple": "first one", "middle": "in between"}}

    opts = render_options(q)

    expected = ["apple: first one", "middle: in between", "zebra: last one"]
    assert opts == expected, f"Expected {expected}, got {opts}"


def test_render_options_with_none_values():
    """None values should render as just the label without description prefix."""
    q = {"t": "choice", "crit": {"b": None, "a": "has desc", "c": ""}}

    opts = render_options(q)

    expected = ["a: has desc", "b", "c"]
    assert opts == expected, f"Expected {expected}, got {opts}"
