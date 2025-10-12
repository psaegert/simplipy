from __future__ import annotations

from pathlib import Path

import pytest

from simplipy import SimpliPyEngine


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("dev_7-3", local_dir=Path("simplipy-assets"))


def _to_prefix(engine: SimpliPyEngine, expression: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(expression, str):
        return engine.parse(expression)
    if isinstance(expression, tuple):
        return list(expression)
    return expression


def test_repeated_addition_does_not_emit_unsupported_multipliers(engine: SimpliPyEngine) -> None:
    expr = " + ".join(["x"] * 14)

    simplified = engine.simplify(expr, max_iter=1, verbose=False)
    simplified_prefix = _to_prefix(engine, simplified)

    assert "mult7" not in simplified_prefix
    assert "div7" not in simplified_prefix


def test_repeated_multiplication_avoids_unsupported_powers(engine: SimpliPyEngine) -> None:
    expr = "x / (" + " * ".join(["x"] * 15) + ")"

    simplified = engine.simplify(expr, max_iter=1, verbose=False)
    simplified_prefix = _to_prefix(engine, simplified)

    assert "pow7" not in simplified_prefix
    assert "mult7" not in simplified_prefix
    assert "div7" not in simplified_prefix
