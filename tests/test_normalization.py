# test_normalization.py

from simplipy import normalize_skeleton, normalize_expression, normalize_variable_token


def test_normalize_variable_token():
    assert normalize_variable_token("v1") == ("x1", True)
    assert normalize_variable_token("x2") == ("x2", True)
    assert normalize_variable_token("V3") == ("x3", True)   # case-insensitive
    assert normalize_variable_token("X10") == ("x10", True)
    assert normalize_variable_token("add") == ("add", False)
    assert normalize_variable_token("3.14") == ("3.14", False)


def test_normalize_skeleton_renames_vars_and_placeholders_constants():
    assert normalize_skeleton(["add", "v1", "3.14"]) == ["add", "x1", "<constant>"]
    assert normalize_skeleton(["sin", "x2"]) == ["sin", "x2"]
    assert normalize_skeleton(["add", "x1", "<constant>"]) == ["add", "x1", "<constant>"]
    assert normalize_skeleton(["add", "x1", "<c>"]) == ["add", "x1", "<constant>"]
    assert normalize_skeleton(["mul", "-2", "v3"]) == ["mul", "<constant>", "x3"]


def test_normalize_expression_keeps_literals():
    assert normalize_expression(["add", "v1", "3.14"]) == ["add", "x1", "3.14"]
    assert normalize_expression(["sin", "V2"]) == ["sin", "x2"]


def test_none_passthrough():
    assert normalize_skeleton(None) is None
    assert normalize_expression(None) is None


def test_renamed_skeletons_compare_equal():
    # v-style and x-style of the same structure normalize identically
    assert normalize_skeleton(["mul", "v1", "v2"]) == normalize_skeleton(["mul", "x1", "x2"])
    # different concrete constants collapse to the same skeleton
    assert normalize_skeleton(["add", "x1", "2.0"]) == normalize_skeleton(["add", "x1", "99"])


def test_reserved_spellings_are_not_constants():
    # H-046 (D2', 2026-08-05): constant-hood is decided by the normative token grammar
    # (is_numeric_string), not a bare float() probe. float() also accepts the RESERVED
    # spellings -- bare inf/nan (any case/sign) and underscore groupings -- and masking
    # those minted a finite-by-doctrine <constant> for a non-finite literal: a skeleton
    # carrying `inf` compared equal to a finite-constant skeleton (fake coverage).
    assert normalize_skeleton(["add", "x0", "inf"]) == ["add", "x0", "inf"]
    assert normalize_skeleton(["sub", "NAN", "v1"]) == ["sub", "NAN", "x1"]
    assert normalize_skeleton(["mul", "-Infinity", "x0"]) == ["mul", "-Infinity", "x0"]
    assert normalize_skeleton(["mul", "1_000", "x0"]) == ["mul", "1_000", "x0"]
    # The canonical special spellings stay themselves too (policy-owned sites).
    assert normalize_skeleton(["sin", 'float("nan")']) == ["sin", 'float("nan")']
    # Grammar-admitted literals still mask: exact fractions and e-notation.
    assert normalize_skeleton(["mul", "1/3", "1e19"]) == ["mul", "<constant>", "<constant>"]
    assert normalize_skeleton(["add", "x1", "1e+16"]) == ["add", "x1", "<constant>"]
