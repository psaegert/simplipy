"""The mine's evidence decision and its high-precision refutations.

A source has evidence iff the interval layer certifies a cell of its box on which it is finite
everywhere -- a property of the source, never of how many rows of the sampled matrix happen to
fall where it is finite -- and the rows it is judged on are topped up to the evidence floor from
that set. A near miss re-judged at high precision is refuted only by converged values, and a
collapse to `<constant>` is minted by the judge's criterion."""
import os

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.mining import RuleMiner

ACJ2_4 = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4', 'config.yaml')
MODES = ('default', 'real', 'permissive')


@pytest.fixture(scope='module')
def bare():
    """The acj2-4 operators with an empty rule set per mode: the core on its own."""
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_real=[], rules_permissive=[])


def x_with_band(engine, in_band: int, n: int = 512) -> np.ndarray:
    """The mine's mixture matrix (two columns) with exactly `in_band` rows whose x0 lies in
    [0, 1), the only place `atanh(floor(x0))` is finite."""
    X = RuleMiner(engine)._mining_sample_x(n, 2, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    x0 = X[:, 0]
    out = (x0 < 0) | (x0 >= 1)
    x0[~out] += 1.5  # every row leaves the band ...
    x0[:in_band] = rng.uniform(0.0, 1.0, in_band)  # ... and `in_band` rows enter it
    return X


@pytest.mark.parametrize('in_band', [0, 40, 63, 79, 200])
def test_a_narrow_domain_has_evidence_whatever_the_sample_holds(bare, in_band):
    """`asinh(atanh(floor(x0))) -> atanh(floor(x0))` holds on [0, 1), about 13% of the mine's
    mixture: the floor of 64 rows in 512 was met at one seed (79 rows) and missed at another
    (63), so the rule depended on the seed. The certified domain decides now, and the missing
    rows are drawn from it, so the pair certifies at every count, none included."""
    X = x_with_band(bare, in_band)
    for mode in MODES:
        ok, why = bare._core.judge_pair(
            'asinh atanh floor x0'.split(), 'atanh floor x0'.split(), ['x0', 'x1'],
            X.flatten().tolist(), X.shape[0], rtol=1e-11, atol=1e-12, rule_mode=mode)
        assert ok, (mode, in_band, why)
    assert bare._core.equivalent_no_const(
        'asinh atanh floor x0'.split(), 'atanh floor x0'.split(), ['x0', 'x1'],
        X.flatten().tolist(), X.shape[0])


def test_a_source_finite_on_a_null_set_has_no_evidence_however_many_rows_hit_it(bare):
    """`asin(cosh(x0))` is finite at x0 = 0 alone. A matrix with a third of its rows at that
    corner holds 170 source-finite rows, above the floor, and still certifies nothing: no cell
    of the box is finite."""
    X = RuleMiner(bare)._mining_sample_x(512, 2, np.random.default_rng(0))
    X[:170, 0] = 0.0
    xf = X.flatten().tolist()
    for mode in MODES:
        ok, why = bare._core.judge_pair('asin cosh x0'.split(), ['<constant>'], ['x0', 'x1'], xf,
                                        512, rtol=1e-11, atol=1e-12, rule_mode=mode)
        assert not ok and why == 'uncertified', (mode, why)
        assert bare._core.find_rule('asin cosh x0'.split(), 3, None, [['<constant>']],
                                    ['x0', 'x1'], xf, 512, rule_mode=mode) is None
    assert not bare._core.equivalent_no_const('asin cosh x0'.split(), ['1'], ['x0', 'x1'], xf, 512)


def test_a_refutation_needs_converged_values(bare):
    """At x = -937.8, `tanh(x)` is exactly -1 at 1024 and at 2048 bits, so `sin(acos(tanh(x)))`
    reads the residue of `sin(pi)` against `1/cosh(x)` ~ 1e-407 at both: an unconverged row,
    which used to refute the identity (and the rule was mined only when no matrix row lay below
    -700). It is judged again at 4096 and 8192 bits and holds. A false zero whose value is
    stable (`exp(sinh(-5))` is 5.9e-33) is still refuted."""
    rows = lambda x: [0.5, 0.0] * 7 + [x, 0.0]  # noqa: E731
    for x in (-937.8, -705.0, 705.0):
        ok, why = bare._core.judge_pair('sin acos tanh x0'.split(), 'inv cosh x0'.split(),
                                        ['x0', 'x1'], rows(x), 8, rtol=1e-11, atol=1e-12,
                                        min_informative=1, rule_mode='real')
        assert ok, (x, why)
    for mode in MODES:
        ok, _ = bare._core.judge_pair('exp sinh x0'.split(), ['0'], ['x0', 'x1'],
                                      [-5.0, 0.0] + [-40.0, 0.0] * 7, 8, rtol=1e-11, atol=1e-12,
                                      min_informative=1, rule_mode=mode)
        assert not ok, mode


def test_each_mode_reads_evidence_in_its_own_semantics(bare):
    """`acosh(erf(x0))` is NaN over the reals and finite (0) wherever float64 rounds erf to 1,
    x0 >= 5.92. `f64` reads float64, its tails exactly (the enclosure of erf there straddles 1
    by its outward step), so its rule has evidence; `real` and `permissive` judge over the
    reals, where the source is finite nowhere, and certify nothing on it."""
    X = RuleMiner(bare)._mining_sample_x(512, 2, np.random.default_rng(0))

    def certified(src, tgt, mode):
        return bare._core.judge_pair(src.split(), tgt.split(), ['x0', 'x1'], X.flatten().tolist(),
                                     512, rtol=1e-11, atol=1e-12, rule_mode=mode)
    ok, why = certified('atan acosh erf x0', 'acosh erf x0', 'default')
    assert ok, why
    for mode in ('real', 'permissive'):
        for src, tgt in (('atan acosh erf x0', 'acosh erf x0'),
                         ('acosh tanh x0', 'acosh erf x0')):
            assert certified(src, tgt, mode) == (False, 'uncertified'), (src, mode)


def test_the_evidence_decision_counts_the_rows_in_the_modes_values(bare):
    """`tan(atan2(x0, 0))` has no value off x0 = 0 in the reals (tan at pi/2), where float64
    gives 1.633e16; `real` and `permissive` read the source's rows by the table before they
    count its evidence, so it has none there and certifies nothing (it took every target, each
    refused by the judge). `f64` reads float64's 1.633e16, which no other target matches."""
    X = RuleMiner(bare)._mining_sample_x(512, 2, np.random.default_rng(42))
    for mode in MODES:
        for tgt in ('0', 'x0', 'tan x0'):
            assert bare._core.judge_pair(
                'tan atan2 x0 0'.split(), tgt.split(), ['x0', 'x1'], X.flatten().tolist(), 512,
                rtol=1e-11, atol=1e-12, rule_mode=mode) == (False, 'uncertified'), (mode, tgt)


def test_a_constant_collapse_is_minted_where_the_judge_licenses_it(bare):
    """A collapse to `<constant>` owes a witness only where its source is finite (the judge's
    `source_defined`), so the mint asks for a positive-measure finite part and nothing more of
    the source's class: `atan(atanh(c))` (finite on |c| <= 1, NaN beyond) and `acos(c)`
    certify, where the class guard refused them as a finite target for a mixed source, and so
    does `0^c` (0 for c > 0, 1 at 0; nothing is owed where it is +inf): the collapse licence. A
    source finite only at the integers (`(-1)^c`) has no evidence outside permissive, whose
    named constant collapse licenses it."""
    X = RuleMiner(bare)._mining_sample_x(512, 2, np.random.default_rng(0))
    xf = X.flatten().tolist()

    def certified(src, mode):
        return bare._core.judge_pair(src.split(), ['<constant>'], ['x0', 'x1'], xf, 512,
                                     rtol=1e-11, atol=1e-12, rule_mode=mode)[0]
    for mode in MODES:
        for src in ('atan atanh <constant>', 'acos <constant>', 'exp log <constant>',
                    'pow 0 <constant>'):
            assert certified(src, mode), (src, mode)
        assert not certified('log neg cosh <constant>', mode), mode
        assert certified('pow (-1) <constant>', mode) == (mode == 'permissive'), mode


def test_the_engine_collapses_what_the_mint_would(bare):
    """The engine's fold reads the collapse licence: a constant-only subtree is a fitted
    `<constant>` wherever a cell of its constants certifies it finite, so `log(c)`, `acos(c)`,
    `atan(atanh(c))` and `0^c` (0 for c > 0, 1 at 0, owed nothing where it is +inf) are
    `<constant>` in every mode, and the mine has nothing to write for them. `(-1)^c` (finite
    only at the integers) is not, outside permissive's recall. The mint reads the same
    licence."""
    for src in ('log <constant>', 'acos <constant>', 'atan atanh <constant>',
                'exp log <constant>', 'pow 0 <constant>'):
        for mode in ('f64', 'real', 'permissive'):
            assert bare.simplify(src.split(), mode=mode) == ['<constant>'], (src, mode)
    for mode in ('f64', 'real'):
        assert ' '.join(bare.simplify('pow (-1) <constant>'.split(), mode=mode)) == \
            'pow -1 <constant>', mode
