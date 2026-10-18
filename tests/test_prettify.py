"""`prettify`: the display of a canonical state (number plan v2, phase 3).

Pinned here:
* the plan's expected-output table at D = 3, row by row, plus the token forms;
* P1, the display fixed point: `prettify(simplify(p)) == p` for `p = prettify(c)`, in all three
  forms, on the release corpus -- and the check fails where it should (rule 11 off on a row
  whose rounding makes another form cheaper);
* P2, the round trip: `simplify(prettify(c))` is `c` on the corpus except where the display
  rounds a number, and a one-ulp change to a displayed float is caught;
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
    (f'exp(-x0^2/2)/rootn(2*{PI}, 2)', f'exp(-x0^2/2)*rootn(1/(2*{PI}), 2)'),  # Feynman I.6.2a
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
    ('sin(sin(9463/890))', '-0.8045036621463277'),
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


def test_digit_budget(eng):
    assert show(eng, 'x1/12', digits=2) == '0.08333333333333333*x1'
    assert show(eng, 'x1/1024', digits=5) == 'x1/1024'
    assert show(eng, '3*x1/1024', digits=5) == '3*x1/1024'
    assert show(eng, '1234*x1') == '1234*x1'  # a float that writes as an integer
    assert show(eng, '1234567*x1') == '1.234567e6*x1'


def test_functions_of_numbers_fold_only_when_every_number_is_a_float(eng):
    assert show(eng, 'sin(2) + x1') == 'x1 + sin(2)'
    assert show(eng, 'x1*sin(9463/890)') == '-0.9348393354311442*x1'
    assert show(eng, 'x1 + sin(9463/890)') == 'x1 - 0.9348393354311442'
    # an odd function of a negative literal gives its sign to the joiner, as in `simplify`'s text
    assert show(eng, 'x1 + sin(-2)') == 'x1 - sin(2)'
    # a power the state keeps only for its size is a number once its base is a float
    long = '0.001980784624751695158377988284554202796250646643885207061964241892776071925160468502'
    assert show(eng, f'x1*{long}^5') == '3.0492025996286705e-14*x1'
    assert show(eng, '3^5000*x1') == '3^5000*x1'
    assert show(eng, 'x1*2^(1/3)') == '2^(1/3)*x1'


def test_divisors(eng):
    """Rule 5 and the grouping: a float whose reciprocal float64 holds as written stays a divisor
    beside another divisor; numbers, variables and their powers group; any other divisor keeps
    its own slash, as the state keeps it."""
    assert show(eng, 'x1/243/x2') == 'x1/(243*x2)'
    assert show(eng, 'x1/243') == '0.00411522633744856*x1'
    assert show(eng, 'x0/(x1^2*x2)') == 'x0/(x1^2*x2)'
    assert show(eng, 'x0/(cos(x1)*x2)') == 'x0/x2/cos(x1)'
    assert show(eng, '1/x1^(1/2)/sin(1/x1)') == '1/rootn(x1, 2)/sin(1/x1)'


# ---------------------------------------------------------------- P1 and P2
def corpus():
    from simplipy.verify import default_corpus
    return default_corpus()


@pytest.mark.parametrize('form', ['infix', 'explicit', 'tagged'])
def test_p1_display_fixed_point_on_the_corpus(eng, form):
    failures = []
    for row in corpus():
        c = eng.simplify(row)
        c = eng.to_infix(c) if form == 'infix' else eng.to_tagged(c) if form == 'tagged' else c
        p = eng.prettify(c)
        if eng.prettify(eng.simplify(p)) != p:
            failures.append((row, p))
    assert not failures, failures[:5]


# Rounding `(c*x1)^(3 + 1e-22)` to `(c*x1)^3` makes the multiplied-out form cheaper: displaying
# the canonical state alone is no fixed point; rule 11 (simplify the display once and display
# that) is.
REROUND = 'x3 + (1815963197375679774584911*x1/1.1784e19)^(9000000000000000000001/3000000000000000000000)'


def test_p1_check_fails_without_rule_11(eng):
    c = list(eng.simplify(eng._core.parse(REROUND, strict=True)))
    raw = eng._core.prettify(c, 3, 'infix', 'default', False)
    assert raw == 'x3 + (154104.1409857162*x1)^3'
    assert eng.prettify(eng.simplify(raw)) != raw  # the P1 check catches it
    p = eng.prettify(c, form='infix')
    assert p == 'x3 + 3.6596784345305275e15*x1^3'
    assert eng.prettify(eng.simplify(p)) == p


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
    import numpy as np
    return np.asarray(eng._core.evaluate_batch(list(tokens), names, xs.ravel().tolist(), len(xs), []))


def test_p2_round_trip_on_the_corpus(eng):
    """Re-reading the display gives the state back, except where a number is rounded; there the
    two evaluate alike in float64 and display alike."""
    import numpy as np
    rng = np.random.default_rng(0)
    moved = []
    for row in corpus():
        c = list(eng.simplify(row))
        r = list(eng.simplify(eng.prettify(c)))
        if r != c:
            moved.append((c, r))
    assert len(moved) == 1, moved  # `19*x16/12` shows (and re-reads) as `1.5833333333333333*x16`
    for c, r in moved:
        assert eng.prettify(c) == eng.prettify(r)
        names = sorted({t for t in c + r if re.fullmatch(r'x\d+', t)})
        xs = rng.uniform(0.5, 2.0, size=(64, len(names)))
        a, b = values(eng, c, names, xs), values(eng, r, names, xs)
        assert np.allclose(a, b, rtol=4e-16, atol=0, equal_nan=True)


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


def test_prettify_never_changes_the_state(eng):
    """Display only: the canonical answer is the same before and after a display."""
    text = '2.14583142772210003669120562399025*x1'
    before = eng.simplify(text)
    eng.prettify(before)
    assert eng.simplify(text) == before == '2.14583142772210003669120562399025*x1'
