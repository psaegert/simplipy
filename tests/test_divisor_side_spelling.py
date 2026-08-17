"""Divisor-side coefficient spelling (owner-ratified 2026-08-01; design/
UNIFIED_SIMPLICITY_MEASURE.md §8) -- a user-facing aesthetics mechanism, EMISSION ONLY.

Every f64-born literal is a dyadic rational with a finite decimal spelling, but its
exact reciprocal generally is not: canon of ``x0 / 0.3333333333333333`` takes the exact
reciprocal coefficient 10^16/3333333333333333 (a genuine descent -- one node fewer),
whose only exact one-token spelling is that monster fraction. The emitters therefore
spell a Mul coefficient on WHICHEVER SIDE has the shorter exact spelling: a coefficient
with NO exact decimal whose reciprocal spells strictly shorter as one literal token
moves behind the bag's ``<div>`` (tagged) / the ``/`` chain (explicit) as the
reciprocal's token. Same state, same value, cosmetic only.

Guardrails pinned here:
* spelling NEVER enters the measure or any mint/skip decision (spelling-dependence is
  the diagnosed Kruskal-skip disease) -- the internal state is identical either way,
  so idempotence and round-trip identity hold for every output;
* ties and both-fraction coefficients stay on the coefficient side (``22/7`` beats
  ``<div> 7/22``); exact decimals never move (``0.5 x0``); a longer reciprocal decimal
  never fires (``1024/3`` beats ``<div> 0.0029296875``);
* the move needs a PLAIN numerator factor to remain (a bag of only coefficient +
  inverse factors keeps the coefficient in the numerator).
"""
import os

import pytest

from simplipy import masking
from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = acj_config_path()


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj-4-3 config not staged')
    return SimpliPyEngine.from_config(CONFIG)


def check(eng, tokens, expected, form=None):
    kwargs = {'form': form} if form else {}
    out = eng.simplify(list(tokens), **kwargs)
    assert list(out) == list(expected), f'{tokens} -> {out}, expected {expected}'
    again = eng.simplify(list(out), **kwargs)
    assert list(again) == list(out), f'not idempotent: {out} -> {again}'
    return out


class TestDivisorSideSpelling:
    def test_the_owner_example(self, eng):
        # x0 / 0.3333333333333333: the reciprocal coefficient has no finite decimal;
        # the original divisor is the exact value's shortest spelling.
        check(eng, ['/', 'x0', '0.3333333333333333'],
              ['<mul>', 'x0', '<div>', '0.3333333333333333', '</mul>'])
        check(eng, ['/', 'x0', '0.3333333333333333'],
              ['/', 'x0', '0.3333333333333333'], form='explicit')

    def test_unit_fraction_coefficient(self, eng):
        # (1/3)*x0 == x0/3: the integer reciprocal is the shorter exact spelling
        # (and the tagged form catches up to what the explicit form already did).
        check(eng, ['*', '1/3', 'x0'], ['<mul>', 'x0', '<div>', '3', '</mul>'])
        check(eng, ['*', '1/3', 'x0'], ['/', 'x0', '3'], form='explicit')

    def test_tie_stays_coefficient_side(self, eng):
        # len("22/7") == len("7/22"): ties keep the coefficient in the numerator.
        check(eng, ['*', '22/7', 'x0'], ['<mul>', '22/7', 'x0', '</mul>'])

    def test_longer_reciprocal_decimal_stays(self, eng):
        # 1/(1024/3) = 0.0029296875 spells LONGER than 1024/3: no move.
        check(eng, ['*', '1024/3', 'x0'], ['<mul>', '1024/3', 'x0', '</mul>'])

    def test_exact_decimal_never_moves(self, eng):
        # RENAMED IN SPIRIT 2026-08-07 (owner-ratified argmin, §10.10(1); audit F31/F33):
        # "exact decimal" is no longer the operative category. `1/2` spells as the
        # FRACTION under argmin (2585 milli-bits against the decimal's 3585), so it is
        # not an atomic-decimal coefficient at all and takes the divisor side like any
        # other unit fraction -- `x0/2`, exactly as `1/3` gives `x0/3` above. What still
        # holds, and is what this row guards, is that a coefficient whose ARGMIN token is
        # atomic stays in the numerator; see the 22/7 and 1024/3 rows.
        check(eng, ['*', '0.5', 'x0'], ['<mul>', 'x0', '<div>', '2', '</mul>'])

    def test_negative_coefficient_sign_splits_out(self, eng):
        # H-020 amendment (2026-08-04): the sign never enters `<div>` -- a signed den
        # literal re-parses the sign into the divisor bag, where the sign-fold clause
        # may absorb it into a Const-bearing sum (a different parking than the state's
        # outer coefficient). The pure sign splits out as a `-1` bag literal; the den
        # token stays positive.
        check(eng, ['/', 'x0', '-0.3333333333333333'],
              ['<mul>', '-1', 'x0', '<div>', '0.3333333333333333', '</mul>'])

    def test_coefficient_split_fires_without_plain_factor(self, eng):
        # F80 (owner-ruled spelling law, 2026-08-10): the split is UNCONDITIONAL for
        # in-bound fractions. (1/3)/x0 -- coefficient + inverse factor, no plain
        # numerator member -- dissolves into the bag: q joins <div>, and the unit
        # numerator's `1` is KEPT as the numerator member the ratified table pins
        # (the pre-F80 emitter demanded a plain factor here and rendered the
        # coefficient as a NESTED structural fraction, the census's 8,918-event
        # family at 1M).
        check(eng, ['/', '1/3', 'x0'],
              ['<mul>', '1', '<div>', '3', 'x0', '</mul>'])
        # The sign rides the numerator (H-020: <div> members stay positive).
        check(eng, ['/', '(-2/3)', 'x0'],
              ['<mul>', '-2', '<div>', '3', 'x0', '</mul>'])
        # Coefficient-FREE den-only bags keep the established empty-numerator
        # spelling -- the F80 ruling did not touch them (see
        # TestLossyReciprocalRejoinProjection::test_certified_pair_stays_distributed).

    def test_joins_existing_divisor_section(self, eng):
        # x0 / (0.3333333333333333 * x1): reciprocal token leads the den members.
        check(eng, ['/', 'x0', '*', '0.3333333333333333', 'x1'],
              ['<mul>', 'x0', '<div>', '0.3333333333333333', 'x1', '</mul>'])

    def test_masking_sees_the_divisor_side_coefficient(self, eng):
        out = eng.simplify(['/', 'x0', '0.3333333333333333'])
        sites = masking.literal_sites(out, eng)
        assert sites == [(3, '0.3333333333333333', masking.Role.COEFFICIENT)]
