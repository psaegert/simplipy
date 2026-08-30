# test_normalization.py
"""The two canonical expression FORMS, on the explicit binary prefix that every
current caller passes. The dialect-invariance ruling and the all-forms intake are
pinned in test_normalization_forms.py; this file keeps the per-token behaviour the
helpers have always been about."""

import pytest

from simplipy import SimpliPyEngine, normalize_variable_token, to_expression, to_skeleton
from conftest import acj_config_path


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


def test_normalize_variable_token():
    assert normalize_variable_token("v1") == ("x1", True)
    assert normalize_variable_token("x2") == ("x2", True)
    assert normalize_variable_token("V3") == ("x3", True)   # case-insensitive
    assert normalize_variable_token("X10") == ("x10", True)
    assert normalize_variable_token("add") == ("add", False)
    assert normalize_variable_token("3.14") == ("3.14", False)


def test_to_skeleton_renames_vars_and_placeholders_constants(engine):
    assert to_skeleton(["+", "v1", "3.14"], engine) == ["+", "x1", "<constant>"]
    assert to_skeleton(["sin", "x2"], engine) == ["sin", "x2"]
    assert to_skeleton(["+", "x1", "<constant>"], engine) == ["+", "x1", "<constant>"]
    # `<c>` is the other spelling of the same placeholder and folds to it.
    assert to_skeleton(["+", "x1", "<c>"], engine) == ["+", "x1", "<constant>"]
    assert to_skeleton(["*", "-2", "v3"], engine) == ["*", "<constant>", "x3"]


def test_to_expression_keeps_literals(engine):
    assert to_expression(["+", "v1", "3"], engine) == ["+", "x1", "3"]
    assert to_expression(["sin", "V2"], engine) == ["sin", "x2"]


def test_none_passthrough(engine):
    assert to_skeleton(None, engine) is None
    assert to_expression(None, engine) is None


def test_renamed_skeletons_compare_equal(engine):
    # v-style and x-style of the same structure normalize identically
    assert to_skeleton(["*", "v1", "v2"], engine) == to_skeleton(["*", "x1", "x2"], engine)
    # different concrete constants collapse to the same skeleton
    assert to_skeleton(["+", "x1", "2.0"], engine) == to_skeleton(["+", "x1", "99"], engine)


def test_the_expression_form_carries_exact_values(engine):
    """The canonical state holds EXACT rationals, so a decimal literal is
    re-spelled as the fraction it is: `2.5` is `5/2`. The skeleton is unaffected
    -- both halves of a rational are one degree of freedom, and the front door's
    collect stage is what enforces that."""
    assert to_expression(["+", "x1", "2.5"], engine) == ["+", "x1", "/", "5", "2"]
    assert to_skeleton(["+", "x1", "2.5"], engine) == ["+", "x1", "<constant>"]


def test_reserved_spellings_are_refused_not_masked(engine):
    """H-046 (D2', 2026-08-05) closed the older defect that a bare `float()`
    probe masked the RESERVED spellings -- `inf`, `NAN`, `1_000` -- into a
    `<constant>` that is finite by doctrine, so a skeleton carrying `inf` compared
    equal to a finite-constant skeleton (fake coverage in holdout matching and
    recovery scoring). Canonicalising through the core closes it harder: the token
    grammar refuses those spellings at the boundary (H-007), so they can neither
    become a `<constant>` nor pass through into a key."""
    for reserved in (["+", "x0", "inf"], ["-", "NAN", "v1"], ["*", "-Infinity", "x0"],
                     ["*", "1_000", "x0"]):
        with pytest.raises(ValueError):
            to_skeleton(reserved, engine)
    # The canonical special spellings stay themselves (policy-owned sites, never
    # a fitted parameter).
    assert to_skeleton(["sin", 'float("nan")'], engine) == ['float("nan")']
    # Grammar-admitted literals still mask: exact fractions and e-notation.
    assert to_skeleton(["*", "1/3", "x1"], engine) == ["*", "<constant>", "x1"]
    assert to_skeleton(["+", "x1", "1e+16"], engine) == ["+", "x1", "<constant>"]
