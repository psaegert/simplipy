"""Where the engine turns an exact literal into an f64, it rounds ONCE, to the nearest f64.

The deployed evaluator reads every literal correctly rounded (Python's ``float`` of a
decimal and ``int / int`` true division both round once), so any engine path that reads
a literal as an f64 has to land on the same double. Rounding the numerator and the
denominator separately and then dividing is off by an ulp or two once either leaves 53
bits, and a function of that argument can amplify the slip far beyond an ulp.
"""

import math
from fractions import Fraction

import pytest
import yaml

from simplipy import Mode
from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

# Each fraction has a component beyond 53 bits; its correctly rounded f64 differs from
# float(p) / float(q).
FRACTIONS = [
    (9161892985817857, 100000),
    (2 * 10**29, 426738538271436458205631863649),
    (673107593011939307760027002528, 810572757194796821120128085049),
]


@pytest.fixture(scope='module')
def bare():
    return SimpliPyEngine(operators=yaml.safe_load(open(acj_config_path()))['operators'], rules=[])


class TestF64FoldReadsTheNearestDouble:
    """`f64_fold` evaluates a ground function at the f64 nearest to its literal, with the
    system libm the deployed evaluator uses."""

    def test_a_large_argument_folds_to_the_evaluators_value(self, bare) -> None:
        # sin amplifies the double rounding of 9161892985817857 / 10^5 into a 3e-6 error:
        # the fold gave -0.9807961394411915.
        (folded,) = bare.simplify(['sin', '91618929858.17857'], mode=Mode.f64)
        assert float(folded) == math.sin(float('91618929858.17857'))

    @pytest.mark.parametrize('p, q', FRACTIONS)
    @pytest.mark.parametrize('fn', ['log', 'sin', 'atan'])
    def test_every_fold_reads_the_correctly_rounded_argument(self, bare, fn, p, q) -> None:
        (folded,) = bare.simplify([fn, '/', str(p), str(q)], mode=Mode.f64)
        assert float(folded) == getattr(math, fn)(float(Fraction(p, q)))


class TestFractionLeaves:
    """A one-token fraction `p/q` is read as Python's `int / int`: correctly rounded."""

    @pytest.mark.parametrize('p, q', FRACTIONS)
    def test_the_constant_folder_reads_the_nearest_double(self, bare, p, q) -> None:
        assert bare.evaluate_constants(['*', f'{p}/{q}', '1']) == [repr(float(Fraction(p, q)))]

    @pytest.mark.parametrize('p, q', FRACTIONS + [(10**40 + 1, 3 * 10**40)])
    def test_the_interval_leaf_encloses_the_fraction(self, bare, p, q) -> None:
        # B6: the old reader took float(p) / float(q) and bracketed it by one ulp, which
        # missed 673107593011939307760027002528/810572757194796821120128085049 (1.5 ulps
        # away). Components beyond i128 cannot be certified and get a wider bracket.
        box = bare._core.interval_value_set_box([f'{p}/{q}'], [0.0], [1.0])
        lo, hi = box[4], box[5]
        assert Fraction(lo) <= Fraction(p, q) <= Fraction(hi)
