"""Where the engine turns an exact literal into an f64, it rounds ONCE, to the nearest f64.

The deployed evaluator reads every literal correctly rounded (Python's ``float`` of a
decimal and ``int / int`` true division both round once), so any engine path that reads
a literal as an f64 has to land on the same double. Rounding the numerator and the
denominator separately and then dividing is off by an ulp or two once either leaves 53
bits, and a function of that argument can amplify the slip far beyond an ulp. The nearest
double is certified by exact midpoint tests at every magnitude a 128-bit fraction can have,
tiny ones included (a first version built the midpoints as 128-bit fractions and gave up
below about 2^-11).
"""

import math
import random
from fractions import Fraction

import pytest
import yaml

from simplipy import Mode
from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

# Each fraction has a component beyond 53 bits; its correctly rounded f64 differs from
# float(p) / float(q). The last two are below 2^-11, where midpoints need denominators
# beyond 128 bits.
FRACTIONS = [
    (9161892985817857, 100000),
    (2 * 10**29, 426738538271436458205631863649),
    (673107593011939307760027002528, 810572757194796821120128085049),
    (8159658314, 59197280150402860313),
    (1, 3 * 10**37 + 7),
]


def small_decimals(seed: int, n: int) -> list[str]:
    """Shortest-repr decimals spread over magnitudes 1e-20 .. 1e-3."""
    rng = random.Random(seed)
    return [repr(rng.uniform(1, 10) * 10.0 ** -rng.randrange(3, 21)) for _ in range(n)]


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

    # (the last fraction is left out: its fold is not cheaper than the fraction, so it stays)
    @pytest.mark.parametrize('p, q', FRACTIONS[:4])
    @pytest.mark.parametrize('fn', ['log', 'sin', 'atan'])
    def test_every_fold_reads_the_correctly_rounded_argument(self, bare, fn, p, q) -> None:
        (folded,) = bare.simplify([fn, '/', str(p), str(q)], mode=Mode.f64)
        assert float(folded) == getattr(math, fn)(float(Fraction(p, q)))

    @pytest.mark.parametrize('fn', ['sin', 'atan'])
    def test_small_arguments_fold_to_the_evaluators_value(self, bare, fn) -> None:
        # e.g. sin(4.7164035732078835e-06) folded to 4.716403573190399e-06, Python gives ...398
        for lit in ['4.7164035732078835e-06', '6.013849055450248e-10'] + small_decimals(0, 150):
            (folded,) = bare.simplify([fn, lit], mode=Mode.f64)
            assert float(folded) == getattr(math, fn)(float(lit)), (fn, lit, folded)


class TestFractionLeaves:
    """A one-token fraction `p/q` is read as Python's `int / int`: correctly rounded."""

    @pytest.mark.parametrize('p, q', FRACTIONS)
    def test_the_constant_folder_reads_the_nearest_double(self, bare, p, q) -> None:
        assert bare.evaluate_constants(['*', f'{p}/{q}', '1']) == [repr(float(Fraction(p, q)))]

    @pytest.mark.parametrize('p, q', FRACTIONS + [
        (10**40 + 1, 3 * 10**40),
        # beyond i128 and 2.27 ulps from float(p) / float(q): a one-ulp bracket misses it
        (88524220207673979580413052837552725604400, 89201998976178365554424057804963117690730),
    ])
    def test_the_interval_leaf_encloses_the_fraction(self, bare, p, q) -> None:
        # B6: the old reader took float(p) / float(q) and bracketed it by one ulp, which
        # missed 673107593011939307760027002528/810572757194796821120128085049 (1.5 ulps
        # away). Components beyond i128 cannot be certified and get a four-ulp bracket.
        box = bare._core.interval_value_set_box([f'{p}/{q}'], [0.0], [1.0])
        lo, hi = box[4], box[5]
        assert Fraction(lo) <= Fraction(p, q) <= Fraction(hi)

    @pytest.mark.parametrize('tok', [
        '10182333997706870151/10000',  # reads ...687.0, denotes ...687.0151
        '2311319199101329.25',  # reads ...329.0
        '9245276796405317/4',
    ])
    def test_a_non_integer_that_reads_as_an_integer_is_not_a_point(self, bare, tok) -> None:
        box = bare._core.interval_value_set_box([tok], [0.0], [1.0])
        value = Fraction(*map(int, tok.split('/'))) if '/' in tok else Fraction(tok)
        assert Fraction(box[4]) <= value <= Fraction(box[5])
        assert box[4] < box[5]

    @pytest.mark.parametrize('tok', ['7', '-12', '2.0', '1e3', '(-1)'])
    def test_an_integer_literal_stays_a_point(self, bare, tok) -> None:
        box = bare._core.interval_value_set_box([tok], [0.0], [1.0])
        assert box[4] == box[5]
