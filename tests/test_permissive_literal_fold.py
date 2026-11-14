"""The PERMISSIVE literal fold (the permissive mode relaxes the literal rule). Falsifier-first: written red against 0.14.5, where the permissive tier printed
`2e29/426738538271436458205631863649` for `4 / (2.7167019109484434 * 3.1415926535897)`.

The exact rational core prices a literal with the two-codeword mu' codebook and prints the
cheaper EXACT codeword; a value with no finite decimal expansion has only the rational
codeword, so the quotient of two 16-digit decimals became a 60-digit fraction, three tokens
for one degree of freedom. Strict writes float64's double of that one number where it is
cheaper (its nearest double). The permissive tier is licensed to move values, so there an exact
literal folds to the shorter-spelled of the two floats around it WHEN mu' prices that spelling
strictly cheaper -- and only then: `1/2`, `15/37`, `4366/8875` keep their fractions, drawn
constants (already shortest f64 spellings) never move.
"""
import math
import re
from fractions import Fraction

import pytest
import yaml

from conftest import acj_config_path
from simplipy import Mode, SimpliPyEngine
from test_mu_prime import mu_prime_expected as exact_price

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


def floats_around(value):
    """The float nearest to `value` and the float on its other side (one when exact)."""
    nearest = float(value)
    if Fraction(nearest) == value:
        return [nearest]
    return [nearest, math.nextafter(nearest, math.inf if value > Fraction(nearest) else -math.inf)]


def significant_digits(spelling):
    mantissa = spelling.lstrip('-').split('e')[0].replace('.', '')
    return len(mantissa.strip('0'))


def test_permissive_folds_the_monster_to_its_float(engine):
    out = prefix(engine, MONSTER, Mode.permissive)
    # The EXACT quotient lies between 0.46867105279529636 (the nearest float) and
    # 0.4686710527952964: permissive promises float precision, not the nearest float, so it
    # lands on the shorter spelling.
    assert float(out[0]) in floats_around(exact_value(MONSTER))
    assert out == ["0.4686710527952964"], out
    assert repr(float(exact_value(MONSTER))) == "0.46867105279529636"


def test_f64_writes_the_double_of_the_exact_value(engine):
    # f64 builds the EXACT value, 2e29/426738538271436458205631863649, and writes float64's
    # double of that one number, cheaper than the 60-digit fraction: the double nearest it,
    # where permissive takes the shorter-spelled of the two floats around it.
    out = prefix(engine, MONSTER, Mode.f64)
    assert out == [repr(float(exact_value(MONSTER)))] == ["0.46867105279529636"], out


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


def test_fold_is_idempotent_and_descends_the_exact_price(engine):
    once = prefix(engine, MONSTER, Mode.permissive)
    assert list(engine.simplify(once, mode=Mode.permissive)) == once
    exact = prefix(engine, MONSTER, Mode.f64)
    # Under the float cap the exact quotient costs its nearest float's shortest decimal; the
    # fold lands on the shorter neighbour, so the measure never rises; the fold descends the
    # EXACT price, which gates it.
    assert engine.complexity(once) <= engine.complexity(exact)
    assert exact_price(Fraction(once[0])) < exact_price(exact_value(MONSTER))


def test_huge_integer_folds_to_its_float_value(engine):
    # A 30-digit integer product: the fold takes the f64 nearest, an integer whose trailing zeros
    # print in scientific notation (`7.572696344528648e29`), priced by its scientific codeword.
    out = prefix(engine, "757269634452864785310003246961 * 1", Mode.permissive)
    assert out == ["7.572696344528648e29"], out
    assert float(out[0]) == float(757269634452864785310003246961)
    assert Fraction(out[0]) != 757269634452864785310003246961


def test_one_thirds_double_is_one_third(engine):
    # the recognition list reads 0.3333333333333333, 1/3's double, as 1/3, in both modes
    for mode in (Mode.f64, Mode.permissive):
        assert prefix(engine, "0.3333333333333333 * 0.3333333333333333", mode) == ["/", "1", "9"]


@pytest.mark.parametrize("expr", [
    MONSTER,
    "1 / (2.7167019109484434 * 3.1415926535897 * 1.0000000000000002)",
    "1e-7 / 3.0000000000000004",
    "0.3333333333333335 * 0.3333333333333335",
])
def test_f64_writes_a_double_where_cheaper(engine, expr):
    """Strict writes float64's value of the one exact number the literals make where that double
    is cheaper than the exact value: one of the two doubles around it, priced below it."""
    out = prefix(engine, expr, Mode.f64)
    exact = exact_value(expr)
    assert len(out) == 1 and "/" not in out[0], out
    assert float(out[0]) in floats_around(exact), (out, exact)
    assert exact_price(Fraction(out[0])) < exact_price(exact)


@pytest.mark.parametrize("expr,expected", [
    ("2.5 / 5", ["/", "1", "2"]),
    ("1.5 / 3.7", ["/", "15", "37"]),
    ("123456789.123456789 / 987654321.987654321", None),
])
def test_f64_keeps_an_exact_value_cheaper_than_its_double(engine, expr, expected):
    out = prefix(engine, expr, Mode.f64)
    if expected is not None:
        assert out == expected, out
    value = Fraction(out[0]) if len(out) == 1 else Fraction(out[1]) / Fraction(out[2])
    assert value == exact_value(expr), out


@pytest.mark.parametrize("expr", [
    MONSTER,
    "1 / (2.7167019109484434 * 3.1415926535897 * 1.0000000000000002)",   # beyond 128 bits (a partition before 2c)
    "757269634452864785310003246961 / 7",
    "1e-7 / 3.0000000000000004",
    "123456789.123456789 / 987654321.987654321",                          # reduces to a CHEAP fraction: stays
    "0.3333333333333335 * 0.3333333333333335",
    "74.22110491129965 - 92.12760331603121 * pow(9.440764872280846, 4)",  # beyond 128 bits: the strict tier kept all four operators before 2c
    "2.7167019109484434 * 3.1415926535897 * 1.4142135623730951 * 1.7320508075688772",
])
def test_fold_follows_the_mu_gate_and_lands_on_a_float_around_the_value(engine, expr):
    """The permissive endpoint is the cheaper of the strict tier's exact spelling and the single
    literal on one of the two floats around the exact value -- the shorter-spelled one, not the
    quotient of two rounded components -- priced by the EXACT codewords (where the fold fires,
    the fold changes what is printed and never raises the price)."""
    exact = exact_value(expr)
    strict = prefix(engine, expr, Mode.f64)
    as_float = [repr(float(exact))]
    out = prefix(engine, expr, Mode.permissive)
    if exact_price(Fraction(as_float[0])) < exact_price(exact):
        # One literal, the float's value; the emitter spells it as the exact decimal of that
        # float (an integer as its digit string, a small value positionally), so compare values.
        assert len(out) == 1 and "/" not in out[0], out
        if (len(strict) == 3 and strict[0] == "/") or len(strict) == 1:
            around = floats_around(exact)
            assert float(out[0]) in around, (out, as_float)
            assert significant_digits(out[0]) == min(significant_digits(repr(f)) for f in around), \
                (out, around)
        else:
            # the strict tier keeps more than one literal (a refused fold): the members fold
            # piecewise, so the endpoint is correctly rounded per fold, within an ulp or two
            assert math.isclose(float(out[0]), float(exact), rel_tol=4e-16), (out, as_float)
    else:
        assert out == strict, (out, strict, engine.complexity(as_float), engine.complexity(strict))


# A constant-only draw from the v25.0-T7 training stream: the argument of the cosine
# is a product of ten 16-digit constants whose exact value left i128, so before phase 2c the strict
# tier kept the whole subtree, and the first fold cut left a 20-digit exact decimal behind (the
# emitter folded two literals at print time that the constructor had kept apart). The permissive
# endpoint must be ONE float literal and its own fixpoint.
STREAM_DRAW = ['-', '14.811588267158381', 'cos', '-', '5.442771642222319', '-', '0.04900091656348083', '-', '9.162716173529507', '*', '/', '*', '*', '*', '*', '-6.845112239925015', '-5.594387656783201', '/', '1.580838523296102', '-4.8070930465509525', '-9.834539098616455', '77.18352008764161', '4.611109313508952', '/', '*', '-1.1927945911146152', '-0.7126036273393542', '/', '-4.282923090520727', '-22.20645482072392']


def test_stream_constant_draw_folds_to_one_literal_and_is_idempotent(engine):
    once = list(engine.simplify(STREAM_DRAW, mode=Mode.permissive))
    assert len(once) == 1 and "/" not in once[0] and len(once[0].replace("-", "").replace(".", "").lstrip("0")) <= 17, once
    assert list(engine.simplify(once, mode=Mode.permissive)) == once
    strict = list(engine.simplify(STREAM_DRAW, mode=Mode.f64))
    # Phase 2c: the exact product fits the 1,100-bit cap, so the strict tier folds the subtree
    # too (the cosine in float64, as f64 mode folds functions; the subtraction exactly). The two
    # modes' floats differ: permissive values the whole ground subtree by the table and rounds it
    # once (P3), the double nearest to the true value 14.817389436549068...; f64 folds in float64,
    # and the cosine of a 9,121-radian argument carries the rounding of its argument
    # (14.817389436549087).
    assert len(strict) == 1 and float(strict[0]) == 14.817389436549087, strict
    assert once == ['14.81738943654907'], once
