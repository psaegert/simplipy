"""The judge's exact routes (`simplipy.verify._exact`): a rule in the piecewise-affine fragment of
one variable is proved or refuted over its whole domain, a left side the interval layer encloses
by one literal over its whole domain is proved, and past the size limits the route answers
unknown."""
import pytest

from simplipy.verify import MODE_TIERS, effective_tier, verify_rule
from simplipy.verify import _contract, _exact

WIDTHS = ('0.001', '1e-10', '1e-50', '1e-100', '1e-200')


def judged(lhs: str, rhs: str) -> dict:
    return verify_rule(lhs.split(), rhs.split())


def route(lhs: str, rhs: str):
    tl = _contract.parse(lhs.split(), '<C_L>', number=True)
    tr = _contract.parse(rhs.split(), '<C_R>', number=True)
    return _exact.piecewise_affine(tl, tr, _contract.d_eval, _contract._table.table_f64_agree)


@pytest.mark.parametrize('width', WIDTHS)
@pytest.mark.parametrize('slot', ['?0', '_0'])
def test_a_narrow_tent_is_refuted_in_every_mode(width: str, slot: str) -> None:
    """`max(0, w - abs(x - 0.123456789)) -> 0` is false on (0.123456789 - w, 0.123456789 + w),
    narrower than every battery point's spacing: the exact route refutes it at 0.123456789,
    where the left side is w, in the reals, in permissive (w rounds to a nonzero double) and in
    float64 (its own w)."""
    lhs = f'max 0 - {width} abs - {slot} 0.123456789'
    d = judged(lhs, '0')
    assert d['exact']['route'] == 'piecewise-affine'
    assert (d['exact']['reals'], d['exact']['permissive'], d['exact']['f64']) == \
        ('refuted', 'refuted', 'refuted'), d['exact']
    assert d['exact']['witness'] == _contract.literal_value('0.123456789')
    assert d['verdict'] == 'KILL' and d['tier'] == 'reject', d


@pytest.mark.parametrize('lhs, rhs, f64', [
    ('- max ?0 0 min ?0 0', 'abs ?0', None),
    ('abs neg ?0', 'abs ?0', 'proved'),
    ('max ?0 ?0', '?0', 'proved'),
    ('min ?0 max ?0 1', '?0', 'proved'),
    ('abs - ?0 1', 'abs - 1 ?0', None),
    ('+ max ?0 1 min ?0 1', '+ ?0 1', None),
    ('* 2 abs ?0', 'abs * 2 ?0', None),
    ('/ abs ?0 4', 'abs * 0.25 ?0', None),
])
def test_an_identity_of_the_fragment_is_proved(lhs: str, rhs: str, f64) -> None:
    """Equal pieces on every cell: proved in the reals and permissive, and in float64 only where
    every operation is exact on doubles (`neg`, `abs`, `max`, `min`)."""
    out = route(lhs, rhs)
    assert (out['reals'], out['permissive'], out['f64']) == ('proved', 'proved', f64), out
    d = judged(lhs, rhs)
    assert d['verdict'] == 'CERTIFIED' and effective_tier(d) in MODE_TIERS['permissive'], d


def test_a_false_identity_is_refuted_with_its_witness() -> None:
    out = route('max ?0 0', 'abs ?0')
    assert out['reals'] == 'refuted' and out['witness'] < 0, out


def test_a_value_beyond_the_doubles_is_read_without_a_double() -> None:
    """A literal past the doubles' range has no double: the route decides the reals and leaves
    float64 to the battery, and an exact value past the range is compared at no float
    resolution (`as_double` is None)."""
    for lhs, rhs in (('max ?0 1e400', 'max 1e400 ?0'), ('* ?0 1e400', '* 1e400 ?0')):
        out = route(lhs, rhs)
        assert (out['reals'], out['permissive'], out['f64']) == ('proved', 'proved', None), out
    out = route('* ?0 1e400', '* ?0 2e400')
    assert (out['reals'], out['permissive']) == ('refuted', None), out
    assert _exact.as_double(10 ** 400) is None and _exact.as_double(10 ** 300) == 1e300


@pytest.mark.parametrize('lhs, rhs', [
    ('* ?0 ?0', '?0'),                     # a product of two varying factors
    ('+ ?0 ?1', '+ ?1 ?0'),                # two variables
    ('exp ?0', 'exp ?0'),                  # outside the fragment
    ('+ ?0 np.pi', '+ np.pi ?0'),          # a literal that is no rational
    ('* <constant> ?0', '* <constant> ?0'),  # a constant
    ('/ ?0 - ?0 ?0', '1'),                 # a quotient by zero
])
def test_outside_the_fragment_the_route_is_unknown(lhs: str, rhs: str) -> None:
    assert route(lhs, rhs) is None


def test_past_the_size_limit_the_route_is_unknown() -> None:
    """A sum of `abs(x - k)` with more kinks than `MAX_BREAKPOINTS` answers unknown; one with
    fewer is decided."""
    def kinks(n: int) -> str:
        terms = [f'abs - ?0 {k}' for k in range(n)]
        out = terms[0]
        for t in terms[1:]:
            out = f'+ {out} {t}'
        return out
    small = _exact.MAX_BREAKPOINTS // 2
    assert route(kinks(small), kinks(small))['reals'] == 'proved'
    assert route(kinks(_exact.MAX_BREAKPOINTS + 2), '0') is None


def test_a_left_side_one_literal_over_its_whole_domain_is_proved() -> None:
    """`0*exp(x)` is 0 at every finite binding (float64's overflow is a range event, where the
    true value stands): the interval enclosure over the whole domain is the literal."""
    d = judged('* 0 exp ?0', '0')
    assert d['exact']['route'] == 'whole-domain' and d['exact']['reals'] == 'proved', d
    assert d['verdict'] == 'CERTIFIED'


@pytest.mark.parametrize('lhs, rhs, f64, permissive', [
    ('cos np.pi', '-1', 'proved', 'proved'),                # one point, read exactly
    ('sin np.pi', '0', 'refused', 'proved'),                # float64's sin(pi) is 1.2e-16
    ('log -1', '0', 'refused', 'refused'),                  # no tolerance at one point
    ('acos <constant>', '<constant>', 'proved', 'proved'),  # the constant's own value
    ('abs neg x0', 'abs x0', 'proved', 'proved'),           # exact in float64 too
    ('- max x0 0 min x0 0', 'abs x0', 'evidence', 'proved'),  # float64 rounds the difference
    ('/ x0 x0', '1', 'evidence', 'evidence'),               # the battery's points alone
    ('atanh tanh x0', 'x0', 'refused', 'evidence'),
])
def test_every_verdict_says_what_it_rests_on(lhs: str, rhs: str, f64: str,
                                             permissive: str) -> None:
    """The basis of a verdict per mode (the specification's 3.4): proved by an exact route,
    held on the judge's points alone (evidence), or refused."""
    assert judged(lhs, rhs)['basis'] == {'f64': f64, 'permissive': permissive}


def test_an_unjudgeable_rule_is_refused_in_every_mode() -> None:
    assert judged('foo ?0', '?0')['basis'] == {'f64': 'refused', 'permissive': 'refused'}
