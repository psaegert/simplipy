"""The PERMISSIVE literal fold (owner ruling 2026-09-03: "in the permissive mode, we can relax
this"). Falsifier-first: written red against 0.14.5, where the permissive tier printed
`2e29/426738538271436458205631863649` for `4 / (2.7167019109484434 * 3.1415926535897)`.

The exact rational core prices a literal with the two-codeword mu' codebook and prints the
cheaper EXACT codeword; a value with no finite decimal expansion has only the rational
codeword, so the quotient of two 16-digit decimals became a 60-digit fraction, three tokens
for one degree of freedom. The strict tiers cannot do better: a spelling denotes the state's
exact value. The permissive tier is licensed to move values, so there an exact literal folds
to its f64 nearest, spelled shortest, WHEN mu' prices that spelling strictly cheaper -- and
only then: `1/2`, `15/37`, `4366/8875` keep their fractions, drawn constants (already shortest
f64 spellings) never move.
"""
import math
import re
from fractions import Fraction

import pytest
import yaml

from conftest import acj_config_path
from simplipy import Mode, SimpliPyEngine

MONSTER = "4 / (2.7167019109484434 * 3.1415926535897)"


@pytest.fixture(scope="module")
def engine():
    # A rules-less engine over the acj vocabulary: the fold is a property of the canon and
    # the permissive chain, not of any mined rule.
    return SimpliPyEngine(operators=yaml.safe_load(open(acj_config_path()))["operators"], rules=[])


def prefix(engine, expr, mode):
    return list(engine.simplify(engine.infix_to_prefix(expr), mode=mode))


def exact_value(expr):
    """The expression's exact rational value, from its literals as Fractions."""
    src = re.sub(r"(\d+\.\d+(?:e-?\d+)?|\d+(?:e-?\d+)?)", lambda m: f"Fraction('{m.group(1)}')", expr)
    return eval(src, {"Fraction": Fraction, "pow": pow})


def test_permissive_folds_the_monster_to_its_float(engine):
    out = prefix(engine, MONSTER, Mode.permissive)
    # The f64 nearest to the EXACT quotient, spelled shortest: 0.46867105279529636 (float
    # arithmetic on the literals lands one ulp away, at 0.4686710527952964 -- the fold is
    # correctly rounded, not the quotient of two rounded components).
    assert out == [repr(float(exact_value(MONSTER)))] == ["0.46867105279529636"], out


def test_strict_tier_keeps_the_exact_fraction(engine):
    # (Mode.real needs a mined real ruleset, which a rules-less engine cannot serve; the fold
    # is gated on the permissive mode alone, so the f64 tier stands for both strict tiers.)
    out = prefix(engine, MONSTER, Mode.f64)
    assert out == ["/", "200000000000000000000000000000", "426738538271436458205631863649"], out


@pytest.mark.parametrize("expr,expected", [
    ("2.5 / 5", ["/", "1", "2"]),                # finite decimal, fraction cheaper
    ("1.5 / 3.7", ["/", "15", "37"]),            # no finite decimal, fraction still cheaper than any float
    ("0.2 * 4", ["0.8"]),                        # finite decimal, decimal cheaper
    ("2.7167019109484434 * x0", ["*", "2.7167019109484434", "x0"]),  # a drawn constant is its own fold
])
def test_cheap_fractions_and_drawn_constants_do_not_move(engine, expr, expected):
    assert prefix(engine, expr, Mode.permissive) == expected


def test_cleared_coefficient_keeps_its_cheap_fraction(engine):
    # 3.4928/7.1 = 4366/8875 prices 35 bits as a fraction, more as a 17-digit float: it stays.
    assert prefix(engine, "3.4928 / 7.1 * x0", Mode.permissive) == ["/", "*", "4366", "x0", "8875"]


def test_monster_coefficient_folds_into_one_literal(engine):
    out = prefix(engine, "x0 / (2.7167019109484434 * 3.1415926535897)", Mode.permissive)
    literals = [t for t in out if t not in ("*", "/", "x0")]
    assert len(literals) == 1 and "/" not in literals[0], out
    assert math.isclose(float(literals[0]), 1 / (2.7167019109484434 * 3.1415926535897), rel_tol=1e-15)


def test_fold_is_idempotent_and_descends_mu(engine):
    once = prefix(engine, MONSTER, Mode.permissive)
    assert list(engine.simplify(once, mode=Mode.permissive)) == once
    exact = prefix(engine, MONSTER, Mode.f64)
    assert engine.complexity(once) < engine.complexity(exact)


def test_huge_integer_folds_to_its_float_value(engine):
    # A 30-digit integer product: the fold keeps an integer spelling (the emitter prints
    # integers as digit strings) but the VALUE is the f64 nearest, priced by its scientific codeword.
    out = prefix(engine, "757269634452864785310003246961 * 1", Mode.permissive)
    assert len(out) == 1 and float(out[0]) == float(757269634452864785310003246961), out
    assert int(out[0]) != 757269634452864785310003246961


@pytest.mark.parametrize("expr", [
    MONSTER,
    "1 / (2.7167019109484434 * 3.1415926535897 * 1.0000000000000002)",   # an i128-overflow partition
    "757269634452864785310003246961 / 7",
    "1e-7 / 3.0000000000000004",
    "123456789.123456789 / 987654321.987654321",                          # reduces to a CHEAP fraction: stays
    "0.3333333333333333 * 0.3333333333333333",
    "74.22110491129965 - 92.12760331603121 * pow(9.440764872280846, 4)",  # exact fold overflows i128: the strict tier keeps all four operators
    "2.7167019109484434 * 3.1415926535897 * 1.4142135623730951 * 1.7320508075688772",
])
def test_fold_follows_the_mu_gate_and_is_correctly_rounded(engine, expr):
    """The permissive endpoint is the cheaper of the strict tier's exact spelling and the single
    literal float(Fraction(...)) -- the correctly rounded float, not the quotient of two rounded
    components -- priced by the engine's own mu."""
    exact = exact_value(expr)
    strict = prefix(engine, expr, Mode.f64)
    as_float = [repr(float(exact))]
    out = prefix(engine, expr, Mode.permissive)
    if engine.complexity(as_float) < engine.complexity(strict):
        # One literal, the float's value; the emitter spells it as the exact decimal of that
        # float (an integer as its digit string, a small value positionally), so compare values.
        assert len(out) == 1 and "/" not in out[0], out
        if len(strict) == 3 and strict[0] == "/":
            assert float(out[0]) == float(exact), (out, as_float)
        elif len(strict) == 1:
            assert float(out[0]) == float(exact), (out, as_float)
        else:
            # an i128-overflow PARTITION (the exact value never fit one rational): the members
            # fold piecewise, so the endpoint is correctly rounded per fold, within an ulp or two
            assert math.isclose(float(out[0]), float(exact), rel_tol=4e-16), (out, as_float)
    else:
        assert out == strict, (out, strict, engine.complexity(as_float), engine.complexity(strict))


# A constant-only draw from the v25.0-T7 training stream (2026-09-03): the argument of the cosine
# is a product of ten 16-digit constants whose exact value leaves i128, so the strict tier keeps
# the whole subtree, and the first fold cut left a 20-digit exact decimal behind (the emitter
# folded two literals at print time that the constructor had kept apart). The permissive endpoint
# must be ONE float literal and its own fixpoint.
STREAM_DRAW = ['-', '14.811588267158381', 'cos', '-', '5.442771642222319', '-', '0.04900091656348083', '-', '9.162716173529507', '*', '/', '*', '*', '*', '*', '-6.845112239925015', '-5.594387656783201', '/', '1.580838523296102', '-4.8070930465509525', '-9.834539098616455', '77.18352008764161', '4.611109313508952', '/', '*', '-1.1927945911146152', '-0.7126036273393542', '/', '-4.282923090520727', '-22.20645482072392']


def test_stream_constant_draw_folds_to_one_literal_and_is_idempotent(engine):
    once = list(engine.simplify(STREAM_DRAW, mode=Mode.permissive))
    assert len(once) == 1 and "/" not in once[0] and len(once[0].replace("-", "").replace(".", "").lstrip("0")) <= 17, once
    assert list(engine.simplify(once, mode=Mode.permissive)) == once
    strict = list(engine.simplify(STREAM_DRAW, mode=Mode.f64))
    assert len(strict) > 1   # the strict tier cannot fold it: the exact value leaves i128
