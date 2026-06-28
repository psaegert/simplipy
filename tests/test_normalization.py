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
