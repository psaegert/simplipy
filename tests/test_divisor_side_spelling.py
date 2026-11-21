"""Divisor-side coefficient spelling: a coefficient is written as the divisor where the
reciprocal codeword is its price.

Every f64-born literal is a dyadic rational with a finite decimal spelling, but its exact
reciprocal generally is not: canon of ``x0 / 0.333`` holds the exact reciprocal coefficient
1000/333 (one node fewer). The measure prices such a value by its reciprocal codeword, the
divisor ``0.333`` (a terminating decimal, the value's denominator past 24), and the emitters
write it as that divisor, behind the bag's ``<div>`` (tagged) or the ``/`` chain (explicit); a
long divisor keeps its spelling the same way (``x1/0.225147640565301``). Every other
coefficient is written by its own cheapest codeword: a fraction split across the product
(``22*x0/7``, ``x0/3``) or one decimal.

Pinned here:
* the state is the same whatever the spelling, so idempotence and round-trip identity hold for
  every output;
* a fraction keeps its numerator in the bag (``22/7``, ``1024/3``), and a value whose decimal
  is the price stays one numerator token;
* the sign never enters a divisor group.
"""
import os

import pytest

from simplipy import masking
from conftest import acj_config_path
from conftest import legacy_from_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = acj_config_path()


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj-4-3 config not staged')
    return legacy_from_config(CONFIG)


TAGS = {'<add>', '</add>', '<mul>', '</mul>', '<sub>', '<div>'}


def check(eng, tokens, expected):
    """`simplify` answers in the DIALECT it was handed since the conversion split, so
    the input is converted into the dialect the EXPECTATION is written in first --
    `simplify(to_tagged(x))` / `simplify(to_prefix(x))`, the exact migration for the
    retired `form=` parameter."""
    convert = eng.to_tagged if TAGS.intersection(expected) else eng.to_prefix
    out = eng.simplify(convert(list(tokens)))
    assert list(out) == list(expected), f'{tokens} -> {out}, expected {expected}'
    again = eng.simplify(list(out))
    assert list(again) == list(out), f'not idempotent: {out} -> {again}'
    return out


class TestDivisorSideSpelling:
    def test_a_reciprocal_coefficient_keeps_its_decimal_divisor(self, eng):
        # x0 / 0.333: the reciprocal coefficient 1000/333 has no finite decimal; its price is the
        # reciprocal codeword, the divisor 0.333, which is how it is written.
        check(eng, ['/', 'x0', '0.333'], ['<mul>', 'x0', '<div>', '0.333', '</mul>'])
        check(eng, ['/', 'x0', '0.333'], ['/', 'x0', '0.333'])

    def test_a_literal_read_as_one_thirds_double(self, eng):
        # float64 reads 0.3333333333333333 as exactly 1/3's double, so it is read as 1/3
        check(eng, ['/', 'x0', '0.3333333333333333'], ['*', '3', 'x0'])

    def test_unit_fraction_coefficient(self, eng):
        # (1/3)*x0 == x0/3: the fraction 1/3, its numerator 1 left out of the product.
        check(eng, ['*', '1/3', 'x0'], ['<mul>', 'x0', '<div>', '3', '</mul>'])
        check(eng, ['*', '1/3', 'x0'], ['/', 'x0', '3'])

    def test_tie_stays_coefficient_side(self, eng):
        # 22/7 is priced as the fraction (its denominator is within 24): no divisor; the tagged
        # form writes no one-token fraction, so the coefficient splits into the bag.
        check(eng, ['*', '22/7', 'x0'], ['<mul>', '22', 'x0', '<div>', '7', '</mul>'])

    def test_longer_reciprocal_decimal_stays(self, eng):
        # 1024/3 is priced as the fraction, not as the divisor 0.0029296875: it splits into the
        # bag (no one-token fraction in the tagged form).
        check(eng, ['*', '1024/3', 'x0'], ['<mul>', '1024', 'x0', '<div>', '3', '</mul>'])

    def test_exact_decimal_never_moves(self, eng):
        # `1/2` is priced as the fraction (2585 milli-bits against the decimal's 3585), written
        # `x0/2` like any other unit fraction, as `1/3` gives `x0/3` above; a coefficient whose
        # cheapest codeword is one token stays in the numerator (the 22/7 and 1024/3 rows).
        check(eng, ['*', '0.5', 'x0'], ['<mul>', 'x0', '<div>', '2', '</mul>'])

    def test_negative_coefficient_sign_splits_out(self, eng):
        # The sign never enters `<div>` -- a signed den
        # literal re-parses the sign into the divisor bag, where the sign-fold clause
        # may absorb it into a Const-bearing sum (a different parking than the state's
        # outer coefficient). The pure sign splits out as a `-1` bag literal; the den
        # token stays positive.
        check(eng, ['/', 'x0', '-0.333'], ['<mul>', '-1', 'x0', '<div>', '0.333', '</mul>'])

    def test_coefficient_split_fires_without_plain_factor(self, eng):
        # The spelling law: the split is UNCONDITIONAL for
        # in-bound fractions. (1/3)/x0 -- coefficient + inverse factor, no plain
        # numerator member -- dissolves into the bag: q joins <div>, and the unit
        # numerator's `1` is KEPT as the numerator member the ratified table pins
        # (an older emitter demanded a plain factor here and rendered the
        # coefficient as a NESTED structural fraction, the census's 8,918-event
        # family at 1M).
        check(eng, ['/', '1/3', 'x0'],
              ['<mul>', '1', '<div>', '3', 'x0', '</mul>'])
        # The sign rides the numerator (<div> members stay positive).
        check(eng, ['/', '(-2/3)', 'x0'],
              ['<mul>', '-2', '<div>', '3', 'x0', '</mul>'])
        # Coefficient-FREE den-only bags keep the established empty-numerator
        # spelling -- the spelling law does not touch them (see
        # TestLossyReciprocalRejoinProjection::test_certified_pair_stays_distributed).

    def test_joins_existing_divisor_section(self, eng):
        # x0 / (0.333 * x1): reciprocal token leads the den members.
        check(eng, ['/', 'x0', '*', '0.333', 'x1'],
              ['<mul>', 'x0', '<div>', '0.333', 'x1', '</mul>'])

    def test_masking_sees_the_divisor_side_coefficient(self, eng):
        out = eng.simplify(eng.to_tagged(['/', 'x0', '0.333']))
        sites = masking.literal_sites(out, eng)
        assert sites == [(3, '0.333', masking.Role.COEFFICIENT)]


@pytest.mark.parametrize('mode', ['f64', 'permissive'])
@pytest.mark.parametrize('divisor', ['6.674e-11', '1.380649e-23', '8.854187817e-12',
                                     '0.225147640565301', '3.7', '9.81'])
def test_a_reciprocal_is_priced_as_its_divisor(eng, mode, divisor):
    """The reciprocal of a terminating decimal is priced as that divisor (the measure's
    reciprocal codeword) and printed as it was priced: `x1/c` stays as written in both modes and
    in every dialect, at the price of its print, and `1/c` alone is written `/ 1 c` (or `c`'s
    reciprocal as one literal where that is cheaper)."""
    for src in (['/', 'x1', divisor], ['<mul>', 'x1', '<div>', divisor, '</mul>']):
        out = list(eng.simplify(list(src), mode=mode))
        assert out == src, (src, out)
        assert eng.complexity(out, mode=mode) == eng.complexity(src, mode=mode)
    assert eng.simplify(f'x1/{divisor}', mode=mode) == f'x1/{divisor}'
    lone = list(eng.simplify(['/', '1', divisor], mode=mode))
    assert lone == ['/', '1', divisor] or len(lone) == 1, lone
