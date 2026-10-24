"""Reading A at poles (f64 and real): the value at a point where the input jumps is not
compared -- an odd pole whose two sides disagree, or a jump between finite values. Where
both sides agree on a value or a signed infinity, the answer must have that limit.

The check probes the source and the answer along random lines through the point, on both
sides and at shrinking radii, and compares the two-sided limits; it does not read the value
computed AT the point. IEEE arithmetic gives a product of factors that vanish there a zero
with the sign of its coefficient, not of its sides: at the even pole of
``exp(x3)/((x3/x2 - x3)*sin(x3/x2))`` (x3 = 0) the limit is +inf for x2 = 1/2 and -inf for
x2 = 2, and every spelling computes one sign along the whole line, so no answer can carry
the limit as a computed value at both points. The function's limit is what an answer keeps.
"""
import numpy as np
import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path

# (source, point, the two-sided limit there: +inf, -inf, or None where the sides disagree)
CASES = [
    ("1/((x1 - 3)*(3 - x1))", {"x1": 3.0}, -np.inf),
    ("-1/(x1 - 3)**2", {"x1": 3.0}, -np.inf),
    ("-1/((3 - x1)*(x1 - 3))", {"x1": 3.0}, np.inf),
    ("30*(x1 - 1)*(x3 - 1)/((x1 - 10)*x2**2)", {"x1": -1.0, "x2": 0.0, "x3": -1.0}, -np.inf),
    ("exp(x3)/((x3/x2 - x3)*sin(x3/x2))", {"x1": 1.0, "x2": 0.5, "x3": 0.0}, np.inf),
    ("exp(x3)/((x3/x2 - x3)*sin(x3/x2))", {"x1": 1.0, "x2": 2.0, "x3": 0.0}, -np.inf),
    ("(x2/x3 + cos(x1/x3))/x2**2", {"x1": 0.5, "x2": 0.0, "x3": -1.0}, np.inf),
    ("1/(1 - x1 - x2)", {"x1": 0.5, "x2": 0.5}, None),
    ("-1/asin(x1 + x2 - 1)", {"x1": 0.5, "x2": 0.5}, None),
    ("atan(1/(3 - x1))", {"x1": 3.0}, None),
]


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


#: What `two_sided_limit` returns when a value near the point is NaN: no limit can be read
#: there, which is a different statement from two sides that disagree (`None`).
UNDEFINED = 'undefined'


def two_sided_limit(engine: SimpliPyEngine, tokens: list[str], point: dict[str, float]):
    """The limit along 24 random lines through the point (+-inf, or a finite value), None
    when the sides disagree, or `UNDEFINED` when the function is NaN near the point."""
    names = sorted(point)
    f = engine.as_callable(tokens, names)
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(24, len(names)))
    vals = []
    for h in (1e-6, -1e-6, 1e-8, -1e-8):
        args = [point[n] + h * dirs[:, i] for i, n in enumerate(names)]
        with np.errstate(all="ignore"):
            vals.append(np.broadcast_to(np.asarray(f(*args), dtype=float), (len(dirs),)))
    v = np.concatenate(vals)
    if np.isnan(v).any():
        return UNDEFINED
    if (np.abs(v) > 1e5).all():
        signs = set(np.sign(v).tolist())
        return None if len(signs) > 1 else signs.pop() * np.inf
    if np.isfinite(v).all() and np.allclose(v, v[0], rtol=1e-4, atol=1e-6):
        return round(float(v[0]), 3)
    return None


@pytest.mark.parametrize("mode", ["f64", "real", "permissive"])
@pytest.mark.parametrize("src,point,limit", CASES)
def test_answer_keeps_the_two_sided_limit(engine, mode, src, point, limit) -> None:
    tokens = list(engine.read_infix(src))
    assert two_sided_limit(engine, tokens, point) == limit, "the case's own premise moved"
    answer = list(engine.simplify(tokens, mode=mode))
    assert two_sided_limit(engine, answer, point) == limit, (mode, engine.to_infix(answer))
