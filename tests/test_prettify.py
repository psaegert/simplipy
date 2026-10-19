"""`prettify`: the display of a canonical state.

Pinned here:
* the expected-output table at D = 3, row by row, plus the token forms;
* display only: prettify writes numbers and decides no value -- a function of numbers stays a
  function, and the display is not simplified again;
* P1, idempotence: `prettify(prettify(c)) == prettify(c)` in all three forms on the release
  corpus, and the display does not depend on how the state holds its reciprocals;
* P2, the read-back: every number of the display reads back as the float64 of the value it
  stands for -- the display evaluates as the state does on the corpus, and a one-ulp change to
  a displayed float is caught;
* the source guard: nothing inside SimpliPy calls prettify -- and the guard catches a call;
* the API: digits, form, dialects, empty input.
"""
import ast
import math
import os
import re
from decimal import Decimal
from fractions import Fraction

import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

CONFIG = acj_config_path()
PI = '3.141592653589793'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj-4-3 config not staged')
    return SimpliPyEngine.from_config(CONFIG)


def show(eng, text, **kw):
    """The display of the canonical state of an infix text."""
    return eng.prettify(eng.simplify(text), **kw)


# The plan's table (NUMBER_SPELLING_PLAN.md section 1, D = 3), row by row.
TABLE = [
    (f'1/(2*{PI})', f'1/(2*{PI})'),
    (f'3/(2*{PI})', f'3/(2*{PI})'),
    (f'x0/(2*{PI})', f'x0/(2*{PI})'),
    (f'exp(-x0^2/2)/rootn(2*{PI}, 2)', '0.3989422804014327*exp(-x0^2/2)'),  # Feynman I.6.2a
    (f'3*x0^2/(20*{PI}*x1*x2)', f'3*x0^2/(20*{PI}*x1*x2)'),  # Feynman 58
    ('x1/2', 'x1/2'),
    ('x1/12', 'x1/12'),
    ('x1/3.7', 'x1/3.7'),
    ('x1*5/21', 'x1/4.2'),
    ('22/7*x1', '22*x1/7'),
    ('1/16', '1/16'),
    ('x1/1024', '9.765625e-4*x1'),
    ('x1^(5/21)', 'x1^(5/21)'),
    # The plan writes this row's base `x2 + 0.5` by hand; rule 2 keeps a fraction that fits,
    # as for `x1/2`, so the base is `x2 + 1/2`. The row is about the exponent (rule 7).
    ('(x2 + 0.5)^(500000000000000/4427)', '(x2 + 1/2)^1.1294330246216399e11'),
    ('x1^2*x2^2/(6*3.1415926535897*8.854e-12*(2.99792458e8)^3)', '2.2238078067157093e-16*x1^2*x2^2'),  # Larmor
    ('x1*20847313721190114925292351/1875000000000000000000000000', '0.011118567317968061*x1'),
    ('sin(sin(9463/890))', 'sin(sin(10.632584269662921))'),
    ('x1*6.626e-34', '6.626e-34*x1'),
    ('x1*1e-40*3', '3e-40*x1'),
    ('x1*1e200*1e200', '1e200*1e200*x1'),
    ('x1^(-1/2)', '1/rootn(x1, 2)'),
    ('x1/0.225147640565301', '4.441529999999994*x1'),
    # rule 5: a divisor float64 holds as written stays a divisor beside another divisor
    ('(3.44153*x2 + 0.774852359434699*rootn(x1, 2))/(-0.225147640565301*(rootn(x1, 2) + sin(x2)))',
     '(3.44153*x2 + 0.774852359434699*rootn(x1, 2))/(-0.225147640565301*(rootn(x1, 2) + sin(x2)))'),
]


@pytest.mark.parametrize('text,expected', TABLE, ids=[t for t, _ in TABLE])
def test_the_plans_table(eng, text, expected):
    assert show(eng, text) == expected


@pytest.mark.parametrize('text,explicit,tagged', [
    (f'1/(2*{PI})', ['/', '1', '*', '2', PI], ['<mul>', '1', '<div>', '2', PI, '</mul>']),
    ('22/7*x1', ['/', '*', '22', 'x1', '7'], ['<mul>', '22', 'x1', '<div>', '7', '</mul>']),
    ('-x1/12', ['/', 'neg', 'x1', '12'], ['<mul>', '-1', 'x1', '<div>', '12', '</mul>']),
    ('x1/1024', ['*', '9.765625e-4', 'x1'], ['<mul>', '9.765625e-4', 'x1', '</mul>']),
    ('x1^(-1/2)', ['/', '1', 'rootn', 'x1', '2'], ['<mul>', '1', '<div>', 'rootn', 'x1', '2', '</mul>']),
])
def test_token_forms(eng, text, explicit, tagged):
    c = eng.simplify(text)
    assert eng.prettify(c, form='explicit') == explicit
    assert eng.prettify(c, form='tagged') == tagged


def test_long_literals_show_their_float(eng):
    """An exact literal longer than a float shows as the float64 nearest to it; one a hair from
    a short number shows as that number, coefficient 1 included."""
    assert show(eng, '2.14583142772210003669120562399025*x1') == '2.1458314277221002*x1'
    assert show(eng, '1.000000000000000000003*x1') == 'x1'
    assert show(eng, 'x1 + 1.000000000000000000003') == 'x1 + 1'
    assert show(eng, 'x1^(600000000000000000001/300000000000000000000)') == 'x1^2'
    assert show(eng, 'x1^(300000000000000000001/600000000000000000000)') == 'rootn(x1, 2)'


@pytest.mark.parametrize('text,precision,expected', [
    ('2.14583142772210003669120562399025*x1', None, '2.1458314277221002*x1'),
    ('2.14583142772210003669120562399025*x1', 3, '2.15*x1'),
    ('2.14583142772210003669120562399025*x1', 1, '2*x1'),
    ('x1/1024', 2, '9.8e-4*x1'),
    ('(x2 + 0.5)^(500000000000000/4427)', 3, '(x2 + 1/2)^1.13e11'),
    ('x1*sin(9463/890)', 4, 'x1*sin(10.63)'),
    ('9.9999*x1', 2, '10*x1'),
    ('1.000000000000000000003*x1', 3, 'x1'),
    (f'1/(2*{PI})', 3, '1/(2*3.14)'),
    ('x1/0.225147640565301', 3, '4.44*x1'),
    ('(3.44153*x2 + x1)/(0.225147640565301*(x1 + sin(x2)))', 3,
     '(x1 + 3.44*x2)/(0.225*(x1 + sin(x2)))'),
])
def test_precision(eng, text, precision, expected):
    """`precision=k` shows a number written as a float with k significant digits; `None` keeps
    the shortest decimal that reads back as the same float64."""
    assert show(eng, text, precision=precision) == expected


@pytest.mark.parametrize('text', ['x1*5/21', 'x1/12', '22/7*x1', 'x1^(5/21)', '3*x0^2/(20*x1*x2)'])
def test_precision_leaves_the_fractions_digits_keeps(eng, text):
    assert show(eng, text, precision=1) == show(eng, text)


def test_digit_budget(eng):
    assert show(eng, 'x1/12', digits=2) == '0.08333333333333333*x1'
    assert show(eng, 'x1/1024', digits=5) == 'x1/1024'
    assert show(eng, '3*x1/1024', digits=5) == '3*x1/1024'
    assert show(eng, '1234*x1') == '1234*x1'  # a float that writes as an integer
    assert show(eng, '1234567*x1') == '1.234567e6*x1'


@pytest.mark.parametrize('text,expected', [
    ('x1*exp(-381/2)', 'x1*exp(-190.5)'),
    ('x1*sin(sin(9463/890))', 'x1*sin(sin(10.632584269662921))'),
    ('x1*sin(9463/890)', 'x1*sin(10.632584269662921)'),
    ('x1 + sin(9463/890)', 'x1 + sin(10.632584269662921)'),
    ('sin(2) + x1', 'x1 + sin(2)'),
    # an odd function of a negative literal gives its sign to the joiner, as in `simplify`'s text
    ('x1 + sin(-2)', 'x1 - sin(2)'),
    # a power the state keeps for its size stays a power of its base, written as a float
    ('x1*0.001980784624751695158377988284554202796250646643885207061964241892776071925160468502^5',
     '0.001980784624751695^5*x1'),
    ('3^5000*x1', '3^5000*x1'),
    ('x1*2^(1/3)', '2^(1/3)*x1'),
])
def test_functions_of_numbers_stay_functions(eng, text, expected):
    """prettify writes numbers and evaluates nothing: what `simplify` keeps, it shows."""
    assert show(eng, text) == expected


# The display rounds `1.00000000000000000001` to `1` and the exponent `3 + 1e-22` to `3`; it does
# not simplify what it wrote, so the display neither collapses to `0` nor multiplies out.
@pytest.mark.parametrize('text,expected', [
    ('sin(1.00000000000000000001*x1) - sin(x1)', 'sin(x1) - sin(x1)'),
    ('x3 + (1815963197375679774584911*x1/1.1784e19)^(9000000000000000000001/3000000000000000000000)',
     'x3 + (154104.1409857162*x1)^3'),
])
def test_the_display_is_not_simplified(eng, text, expected):
    assert show(eng, text) == expected


def test_divisors(eng):
    """Rule 5: a float whose reciprocal float64 holds as written stays a divisor beside another
    divisor; a bare one is a multiplier."""
    assert show(eng, 'x1/243/x2') == 'x1/(243*x2)'
    assert show(eng, 'x1/243') == '0.00411522633744856*x1'
    assert show(eng, 'x0/(x1^2*x2)') == 'x0/(x1^2*x2)'


@pytest.mark.parametrize('text,expected', [
    ('x1/x2/exp(x3)', 'x1/(x2*exp(x3))'),
    ('x1/sin(x3)/(x2 + 1)', 'x1/(sin(x3)*(x2 + 1))'),
    ('x1/x2/(x3 + 1)', 'x1/(x2*(x3 + 1))'),
    ('x1/x2/x3^2', 'x1/(x2*x3^2)'),
    ('x0/(cos(x1)*x2)', 'x0/(x2*cos(x1))'),
    ('1/x1^(1/2)/sin(1/x1)', '1/(rootn(x1, 2)*sin(1/x1))'),
    ('-x1/(3*x2)/(x3 + 1)', '-x1/(3*x2*(x3 + 1))'),
])
def test_one_slash(eng, text, expected):
    """Every divisor of a product joins one denominator product, functions and sums included."""
    assert show(eng, text) == expected


def test_a_joined_reciprocal_keeps_its_sign_and_its_tagged_bag(eng):
    """A joined reciprocal's sign stays in the denominator, where the state holds it; the tagged
    section, which inverts each member on its own, keeps a joined reciprocal of a sum one bag."""
    assert eng.prettify('x3/(-(x11 + x9))') == 'x3/(-(x11 + x9))'
    assert eng.prettify(['/', 'x3', 'neg', '+', 'x11', 'x9']) == ['/', 'x3', 'neg', '+', 'x11', 'x9']
    assert eng.prettify('x1/(x2*(x3 + 1))', form='tagged') == [
        '<mul>', 'x1', '<div>', '<mul>', 'x2', '<add>', 'x3', '1', '</add>', '</mul>', '</mul>']


def test_a_number_beside_a_fitted_constant_keeps_its_place(eng):
    """A fitted constant absorbs a coefficient beside it, so the number of a joined reciprocal
    stays in the denominator: the display loses no number, and it is its own display."""
    import numpy as np
    text = '<constant>*x5/(0.2*(rootn(x2, 2) + <constant>*x7))'
    p = eng.prettify(text)
    assert eng.prettify(p) == p
    names, xs = ['x2', 'x5', 'x7'], np.random.default_rng(1).uniform(0.5, 2.0, size=(16, 3))
    a = values(eng, eng._core.parse(text, strict=True), names, xs)
    b = values(eng, eng._core.parse(p, strict=True), names, xs)
    assert np.allclose(a, b, rtol=1e-12, atol=0)


# ---------------------------------------------------------------- P1 and P2
def corpus():
    from simplipy.verify import default_corpus
    return default_corpus()


def in_form(eng, c, form):
    return eng.to_infix(c) if form == 'infix' else eng.to_tagged(c) if form == 'tagged' else c


def displays_that_move(eng, rows, form):
    """The rows whose display is not its own display: `prettify(p) != p`."""
    moved = []
    for row in rows:
        p = eng.prettify(in_form(eng, eng.simplify(row), form))
        if eng.prettify(p) != p:
            moved.append((row, p, eng.prettify(p)))
    return moved


@pytest.mark.parametrize('form', ['infix', 'explicit', 'tagged'])
def test_p1_displaying_a_display_gives_it_back_on_the_corpus(eng, form):
    moved = displays_that_move(eng, corpus(), form)
    assert not moved, moved[:5]


def test_p1_check_catches_a_display_that_moves(eng):
    """The gate can fail. Writing a number as its float can make two terms of the display alike
    that the state keeps apart; reading the display then collects them."""
    row = 'sin(1.00000000000000000001*x1) - sin(x1)'
    assert displays_that_move(eng, [row], 'infix') == [(row, 'sin(x1) - sin(x1)', '0')]


# Joined reciprocals of numbers, variables and their powers and roots: alike in every form.
GROUPS = ['x13/(3*x4*rootn(x10, 2))', '2*x1/(4.2*x2*x3^2)', 'x0/(3125*x1^5*x2^5)/sinh(x1)^5']
# Joined reciprocals holding functions and sums: alike in the forms with one denominator.
JOINED = ['x1/(x2*(x3 + 1))', 'x1/(sin(x3)*(x2 + 1))', '2*x1/(4.2*x2*sin(x3))',
          'x1/(x2*x3*(x4 + 1))/exp(x5)']


@pytest.mark.parametrize('text,form', [(t, f) for t in GROUPS for f in ('infix', 'explicit', 'tagged')]
                         + [(t, f) for t in JOINED for f in ('infix', 'explicit')])
def test_p1_the_display_does_not_depend_on_how_the_state_holds_its_reciprocals(eng, text, form):
    """The input reads as one joined reciprocal (`(x2*(x3 + 1))^-1`), the answer of `simplify`
    holds a reciprocal per factor (`x2^-1*(x3 + 1)^-1`): the two display alike."""
    joined = eng.prettify(in_form(eng, eng._core.parse(text, strict=True), form))
    split = eng.prettify(in_form(eng, eng.simplify(eng._core.parse(text, strict=True)), form))
    assert joined == split


def floats(tokens):
    """The float64 values of the numerals of a token answer, in order."""
    out = []
    for t in tokens:
        try:
            d = Decimal(t)
        except Exception:
            continue
        if d.is_finite():
            out.append(float(Fraction(d)))
    return out


def values(eng, tokens, names, xs):
    """The float64 values of `tokens` at the points `xs`, every `<constant>` set to one value."""
    import numpy as np
    params = [1.2345] * sum(t == '<constant>' for t in tokens)
    return np.asarray(eng._core.evaluate_batch(list(tokens), names, xs.ravel().tolist(), len(xs), params))


# The display writes each number exactly or as the float64 the evaluator reads it as, and writes
# the same operations; only their order can differ (a sum's association, the order of divisions).
# Measured on the corpus at these points: at most 9.1e-13 apart, where exp(sinh(...)) amplifies a
# re-associated sum. The gate catches a number written short of its float64 to a few digits; the
# one-ulp test below is the fine check that a float is written in full.
P2_RTOL = 1e-9


def test_p2_the_display_evaluates_as_the_state_does_on_the_corpus(eng):
    import numpy as np
    rng = np.random.default_rng(0)
    apart = []
    for row in corpus():
        c = list(eng.simplify(row))
        p = list(eng.prettify(c))
        names = sorted({t for t in c + p if re.fullmatch(r'x\d+', t)})
        xs = rng.uniform(0.5, 2.0, size=(64, max(len(names), 1)))
        a, b = values(eng, c, names or ['x0'], xs), values(eng, p, names or ['x0'], xs)
        if not np.allclose(a, b, rtol=P2_RTOL, atol=0, equal_nan=True):
            apart.append((eng.to_infix(c), eng.to_infix(p)))
    assert not apart, apart[:5]


def test_p2_holds_at_full_precision_only(eng):
    """With `precision=k` a number no longer reads back as its float64: the gate catches it."""
    import numpy as np
    c = list(eng.simplify(eng._core.parse('x1/3 + 7.123456789012345*x2', strict=True)))
    xs = np.random.default_rng(0).uniform(0.5, 2.0, size=(16, 2))
    a = values(eng, c, ['x1', 'x2'], xs)
    assert np.allclose(a, values(eng, eng.prettify(c), ['x1', 'x2'], xs), rtol=P2_RTOL, atol=0)
    assert not np.allclose(a, values(eng, eng.prettify(c, precision=3), ['x1', 'x2'], xs),
                           rtol=P2_RTOL, atol=0)


@pytest.mark.parametrize('text', ['x1/0.225147640565301', 'x1/3 + 7.123456789012345*x2',
                                  'x1^2*x2^2/(6*3.1415926535897*8.854e-12*(2.99792458e8)^3)'])
def test_p2_catches_a_one_ulp_change(eng, text):
    c = list(eng.simplify(eng._core.parse(text, strict=True)))
    p = eng.prettify(c)
    i = max(range(len(p)), key=lambda k: len(p[k]) if floats([p[k]]) else -1)
    assert floats(eng.simplify(p)) == floats(eng.simplify(eng.prettify(eng.simplify(p))))
    bumped = list(p)
    bumped[i] = repr(math.nextafter(float(Fraction(Decimal(p[i]))), math.inf))
    assert floats(eng.simplify(bumped)) != floats(eng.simplify(p))


# ---------------------------------------------------------------- the source guard
MODULES = {os.path.join('rust', 'ac', 'prettify.rs'), os.path.join('rust', 'engine', 'prettify.rs')}
# The only other lines that may name it: the module declarations and the FFI method.
ALLOWED_RUST = {
    os.path.join('rust', 'ac', 'mod.rs'): {'pub mod prettify;'},
    os.path.join('rust', 'engine', 'mod.rs'): {'mod prettify;', 'pub use prettify::{Pretty, PrettyForm};'},
    os.path.join('rust', 'lib.rs'): {'fn prettify(', 'self.inner.ac_prettify(&tokens, digits, form, mode, budget)'},
}
NAME = re.compile(r'prettify|PrettyForm|Pretty\b')


def guard_violations(root):
    """Every place under `root`'s `src/simplipy` and `rust` that reaches prettify, outside its
    module, the module declarations, the FFI method and the public method itself."""
    found = []
    for base in ('rust', os.path.join('src', 'simplipy')):
        for dirpath, _, files in os.walk(os.path.join(root, base)):
            for name in files:
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, root)
                if rel in MODULES:
                    continue
                if name.endswith('.rs'):
                    allowed = ALLOWED_RUST.get(rel, set())
                    for n, line in enumerate(open(path, encoding='utf-8'), 1):
                        code = line.split('//')[0].strip()
                        if NAME.search(code) and not any(code.startswith(a) for a in allowed):
                            if not (rel == os.path.join('rust', 'lib.rs') and _inside_ffi(path, n)):
                                found.append(f'{rel}:{n}: {code}')
                elif name.endswith('.py'):
                    found += [f'{rel}:{n}' for n in _python_uses(path)]
    return found


def _inside_ffi(path, lineno):
    """Is line `lineno` of lib.rs inside the body of `fn prettify(`?"""
    lines = open(path, encoding='utf-8').read().splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith('fn prettify(')), None)
    if start is None or lineno - 1 < start:
        return False
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == '    }')
    return lineno - 1 <= end


def _python_uses(path):
    """Lines of a Python file that name prettify outside `def prettify` itself."""
    tree = ast.parse(open(path, encoding='utf-8').read())
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'prettify':
            allowed |= {n.lineno for n in ast.walk(node) if hasattr(n, 'lineno')}
    uses = []
    for node in ast.walk(tree):
        named = (isinstance(node, ast.Attribute) and 'prettify' in node.attr) or \
                (isinstance(node, ast.Name) and 'prettify' in node.id)
        if named and node.lineno not in allowed:
            uses.append(node.lineno)
    return uses


def test_nothing_inside_simplipy_calls_prettify():
    assert guard_violations(REPO) == []


def test_the_source_guard_catches_a_call(tmp_path):
    for rel in ('rust/ac/expr.rs', 'src/simplipy/masking.py', 'rust/lib.rs'):
        os.makedirs(tmp_path / os.path.dirname(rel), exist_ok=True)
    (tmp_path / 'rust/ac/expr.rs').write_text(
        'fn f() {\n    let s = crate::ac::prettify::to_infix(&n);\n}\n')
    (tmp_path / 'src/simplipy/masking.py').write_text('def mask(e, x):\n    return e.prettify(x)\n')
    (tmp_path / 'rust/lib.rs').write_text('fn other() {\n    self.inner.ac_prettify(&t, 3, f, m, None)\n}\n')
    found = guard_violations(tmp_path)
    assert len(found) == 3, found


# ---------------------------------------------------------------- the API
def test_api(eng):
    c = eng.simplify('x1/12')
    assert eng.prettify(c) == 'x1/12'  # a str answers in infix
    assert eng.prettify(eng.simplify(['/', 'x1', '12'])) == ['/', 'x1', '12']  # tokens in tokens
    assert eng.prettify(eng.to_tagged(['/', 'x1', '12'])) == ['<mul>', 'x1', '<div>', '12', '</mul>']
    assert eng.prettify(['/', 'x1', '12'], form='infix') == 'x1/12'
    assert eng.prettify('') == ''
    assert eng.prettify([]) == []
    with pytest.raises(ValueError):
        eng.prettify(c, digits=0)
    with pytest.raises(TypeError):
        eng.prettify(c, digits=True)
    with pytest.raises(TypeError):
        eng.prettify(c, digits=2.5)
    with pytest.raises(ValueError):
        eng.prettify(c, form='prefix')
    with pytest.raises(ValueError):
        eng.prettify(['/', 'x1'])
    with pytest.raises(ValueError):
        eng.prettify(c, precision=0)
    with pytest.raises(TypeError):
        eng.prettify(c, precision=True)
    with pytest.raises(TypeError):
        eng.prettify(c, precision=2.5)


def test_prettify_never_changes_the_state(eng):
    """Display only: the canonical answer is the same before and after a display."""
    text = '2.14583142772210003669120562399025*x1'
    before = eng.simplify(text)
    eng.prettify(before)
    assert eng.simplify(text) == before == '2.14583142772210003669120562399025*x1'
