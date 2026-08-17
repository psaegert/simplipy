"""Falsifiers for the ruled class gates (F75/F77) in remine/fuzz_properties.py.

The gates replace hand-written row pins: a judge conviction may be tolerated
WITHOUT a pin iff its SOURCE lies in an owner-ruled MECHANICAL class (F75: the
extreme lane's degenerate-literal class, 2026-08-10; F77: the base lane's
inverse-pair-collapse-on-literal-arguments class, 2026-08-10). These tests pin
the two properties the gates must never lose:

  1. the DISCRIMINATOR: ruled-illegal group-D convictions (508487) are NOT
     absorbed -- a regression of the F82/F83 fixes must exit 1;
  2. the stream law: the row indices the falsifiers talk about regenerate the
     exact recorded corpus sources (rng call order copied from row(); drift
     between record and stream convicts here, not at hour N of a 1M run).

remine/ is the research-branch harness (gitignored on public checkouts):
everything skips when it is not staged.
"""
import os
import random
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'remine'))

fp = pytest.importorskip('fuzz_properties', reason='remine harness not staged')

from conftest import acj_config_path, require_or_skip  # noqa: E402

CONFIG = acj_config_path()

# Recorded corpus sources, VERBATIM from fuzz_*_f81e2_1M.json (frozen streams).
SRC_508487 = 'pow cos / - x1 -3 rootn 9007199254740991 -3 exp rootn / x0 -3 -1'
SRC_433087 = ('/ * - 9007199254740991 log pow np.pi '
              '1/170141183460469231731687303715884105727 - - + 1e40 '
              '170141183460469231731687303715884105727/3 x1 + rootn 0 -1 x3 np.pi')
SRC_57000 = 'pow float("-inf") / - 1.0000000000000002 1e19 5e-324'
SRC_241803 = 'rootn x3 atanh pow tanh -1 pow x3 0'


def _extreme_source(i):
    # EXACT copy of row()'s stream law: rng.choice consumes state BEFORE gen.
    rng = random.Random(20260805 * 1_000_003 + i)
    expr = fp.gen_extreme(rng, rng.choice([2, 3, 4, 5]))
    if len(expr) > 3000:
        expr = [rng.choice(fp.VARS)]
    return expr


def _base_source(i):
    rng = random.Random(20260803 * 1_000_003 + i)
    expr = fp.gen(rng, rng.choice([2, 3, 4, 5, 6]), literal_heavy=(i % 2 == 1))
    if len(expr) > 3000:
        expr = [rng.choice(fp.VARS)]
    return expr


class TestStreamLaw:
    def test_regenerated_sources_match_records(self):
        assert ' '.join(_extreme_source(508487)) == SRC_508487
        assert ' '.join(_extreme_source(433087)) == SRC_433087
        assert ' '.join(_extreme_source(57000)) == SRC_57000
        assert ' '.join(_base_source(241803)) == SRC_241803


class TestF75DegenerateLiteralGate:
    def test_group_d_conviction_is_not_absorbed(self):
        # THE discriminator: 508487 (ruled illegal, fixed by F83) has no special
        # literal and no ground pole -- rootn(9007199254740991, -3) is a small
        # finite number; the pole needs x0. If the fix regresses, the row lands
        # in judge_non_ok and the lane exits 1.
        assert not fp.f75_member(SRC_508487.split())

    def test_row_433087_absorbed_by_design(self):
        # DOCUMENTED BLIND SPOT, deliberate: 433087 was ALSO a group-D
        # conviction (F82, mixed-infinity b_mul collapse -- fixed), but its
        # source contains the ground constructed pole `rootn 0 -1` (= inv 0 =
        # +inf under contract semantics), so the SHAPE predicate absorbs it: a
        # future F82 regression on this row would be class-tolerated, not
        # convicted. That is acceptable ONLY because the rust falsifier
        # f82_mixed_infinity_survives_multiplication pins the minimal mixed
        # operand case AND the reduced 433087 row end-to-end. The asymmetry
        # with 508487 (non-member, the live discriminator) is the ruling's
        # intent, not an accident.
        assert fp.f75_member(SRC_433087.split())

    def test_members_by_each_disjunct(self):
        # disjunct 1: a literal special token (recorded row 57000)
        assert fp.f75_member(SRC_57000.split())
        assert fp.f75_member('* float("nan") x0'.split())
        # disjunct 2: a variable-free subterm that IS a pole under contract
        # semantics (evaluated, never token-scanned: the engine may spell
        # constructed poles without any inf/nan token)
        assert fp.f75_member('inv - 1 1'.split())
        assert fp.f75_member('+ x1 rootn 0 -1'.split())

    def test_non_members(self):
        # variable-carried poles and plain rows stay convictable
        assert not fp.f75_member('pow / neg x0 3 -1'.split())
        assert not fp.f75_member('* 2 sin x3'.split())
        assert not fp.f75_member('+ x0 1'.split())
        # an evaluator REFUSAL is never membership (single var token)
        assert not fp.f75_member(['x0'])


class TestF77InversePairGate:
    def test_family_row_is_member(self):
        require_or_skip(CONFIG, 'acj-4-3 config not staged')
        # the F64-register family (base row 241803): atanh(tanh(-1)^(x3^0=1))
        # collapses to the single finite literal -1 -- exactly the engine
        # behavior the judge's variable-free machinery may not corroborate.
        assert fp.f77_member(SRC_241803.split(), CONFIG)

    def test_non_members(self):
        require_or_skip(CONFIG, 'acj-4-3 config not staged')
        # variable collapse is excluded (single token but not a literal)
        assert not fp.f77_member('atanh tanh x0'.split(), CONFIG)
        assert not fp.f77_member('log exp x1'.split(), CONFIG)
        # no inverse pair present
        assert not fp.f77_member('+ x0 1'.split(), CONFIG)
        assert not fp.f77_member('* 2 sin x3'.split(), CONFIG)
