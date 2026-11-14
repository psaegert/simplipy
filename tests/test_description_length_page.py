"""The description-length guide (docs/guides/description-length.md) against the shipped measure.

1. Every price table on the page: each printed price is ``complexity()`` of its expression, in
   bits, and each listed part costs what the page says and the parts add up to the price.
2. The page's symbol table is the table of the formula below and the same for every engine,
   and every price written in its prose is recomputed.
3. THE FORMULA. Computed over the canonical form SimpliPy reads, without calling
   ``complexity()``, it gives ``complexity()``'s price on every expression of the page and on
   the 400 expressions of the benchmark corpus, in each mode. It is exact for every number
   SimpliPy holds as an exact fraction (p, q and the decimal digits m below 2^127).
4. Simplification never raises the price, on the same corpus, in each mode.
5. The rules do not change a price: an engine without rules prices every page expression the
   same.
6. The numbers of docs/formal.md that the guide links to: the named constants' bound, the
   variable's census basis, and the even-index root spelling of I4.

How the formula test reads the canonical form. An engine without rules simplifies an
expression to its canonical form and serializes it in the tagged dialect; no rule can fire,
so the answer is the form SimpliPy reads (re-reading it gives the same form). The tagged
serialization pins every part of that form but two, which the test therefore enumerates:

- a sum at the top or in a function's argument prints its terms with their signs, so it may
  stand for -1 times the sum of the negated terms (``-x1 - x2`` for ``-(x1 + x2)``);
- in ``permissive``, a divisor that is a product, ``1/(k*P)``, may hold its number ``k`` in
  front instead, ``(1/k) * 1/P``.

On the corpus the test asks that ``complexity()`` equal the formula for one of those forms. On
the page's own examples it asks for the cheapest of them, as the page's rule for sums states.
"""
import json
import math
import os
import random
import re
import struct
import warnings
from fractions import Fraction

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(REPO, 'docs', 'guides', 'description-length.md')
FORMAL = os.path.join(REPO, 'docs', 'formal.md')
CORPUS = os.path.join(REPO, 'benchmarks', 'corpus', 'raw_skeletons_nv.json')
MODES = ('f64', 'real', 'permissive')
_RULE_MODE = {'f64': 'default', 'real': 'real', 'permissive': 'permissive'}

# --------------------------------------------------------------------------------------------
# The formula, as the page states it
# --------------------------------------------------------------------------------------------

MILLI = 1000
#: c(s), in bits: the page's symbol table.
SYMBOLS = {'variable': 6, 'sum': 3, 'product': 3, 'power': 3, 'np.pi': 4, 'np.e': 4,
           'float("inf")': 8, 'float("-inf")': 8, 'float("nan")': 8, '<constant>': 67}
FUNCTIONS = {**{f: 6 for f in ('exp', 'log', 'abs', 'sin', 'cos', 'tan', 'rootn')},
             **{f: 8 for f in ('asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh',
                               'atanh')}}
OTHER_FUNCTION = 8


def _sum() -> int:
    return SYMBOLS['sum'] * MILLI


def _product_head() -> int:
    return SYMBOLS['product'] * MILLI


def _power() -> int:
    return SYMBOLS['power'] * MILLI


ODD = {'sin', 'tan', 'asin', 'atan', 'sinh', 'tanh', 'asinh', 'atanh'}
SIGNLESS = {'<constant>', 'float("inf")', 'float("-inf")', 'float("nan")'}
_NUMBER = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$|^[+-]?\d+/\d+$')


def ell(x) -> int:
    """l(x) = max(1, log2(1 + |x|)) in milli-bits, rounded to the nearest milli-bit."""
    m = abs(int(x)) + 1
    b = m.bit_length()
    return max(MILLI, round(MILLI * ((b - 1) + math.log2(m / (1 << (b - 1))))))


def L(n) -> int:
    """L(n) = 1 + [n < 0] + min(l(p) + [q != 1] l(q), l(m) + l(k)), in milli-bits."""
    n = Fraction(n)
    p, q = abs(n.numerator), n.denominator
    codes = [ell(p) + (ell(q) if q != 1 else 0)]
    rest, a, b = q, 0, 0
    while rest % 2 == 0:
        rest, a = rest // 2, a + 1
    while rest % 5 == 0:
        rest, b = rest // 5, b + 1
    if rest == 1 and p != 0:  # n = +-m * 10^k, m not ending in 0
        if q == 1:
            m, k = p, 0
            while m % 10 == 0:
                m, k = m // 10, k + 1
        else:
            k = max(a, b)
            m = p * 2 ** (k - a) * 5 ** (k - b)
        if k != 0:
            codes.append(ell(m) + ell(k))
    return MILLI + (MILLI if n < 0 else 0) + min(codes)


# --- reading the tagged serialization into parts --------------------------------------------
# ('num', Fraction) | ('sym', name) | ('fun', name, [args]) | ('pow', base, exponent)
# | ('add', [terms], free) | ('mul', coefficient, [factors]) | ('recip', k, [factors])
# | ('big', Fraction): a number not held exactly, kept as written; never merged into a
# coefficient or a sum's number, and a minus sign on it makes a product with -1.

#: A number is held exactly when its numerator and denominator are below 2^127.
_HELD = 2 ** 127


def _product(c, factors):
    if not factors:
        return ('num', c)
    if c == 1 and len(factors) == 1:
        return factors[0]
    return ('mul', c, factors)


def _negate(x):
    if x[0] == 'num':
        return ('num', -x[1])
    if x[0] == 'mul':
        return _product(-x[1], x[2])
    return ('mul', Fraction(-1), [x])


def parse(tokens, arity, recip_alternatives):
    pos = 0

    def section(close, inverse):
        nonlocal pos
        sides, side = ([], []), 0
        while tokens[pos] != close:
            if tokens[pos] == inverse:
                side, pos = 1, pos + 1
                continue
            sides[side].append(node(free=False))
        pos += 1
        return sides

    def node(free):
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == '<add>':
            plus, minus = section('</add>', '<sub>')
            terms = plus + [_negate(x) for x in minus]
            constant = sum((t[1] for t in terms if t[0] == 'num'), Fraction(0))
            terms = [t for t in terms if t[0] != 'num'] + ([('num', constant)] if constant else [])
            return ('add', terms, free)
        if tok == '<mul>':
            up, down = section('</mul>', '<div>')
            c, factors = Fraction(1), []
            for x in up:
                if x[0] == 'num':
                    c *= x[1]
                elif x[0] == 'mul':
                    c *= x[1]
                    factors += x[2]
                else:
                    factors.append(x)
            for x in down:
                if x[0] == 'num':
                    c /= x[1]
                elif x[0] == 'pow' and x[2][0] == 'num':
                    factors.append(('pow', x[1], ('num', -x[2][1])))
                elif x[0] == 'mul' and x[1] != 0 and recip_alternatives:
                    factors.append(('recip', x[1], x[2]))
                else:
                    factors.append(('pow', x, ('num', Fraction(-1))))
            return _product(c, factors)
        if tok == 'neg':
            return _negate(node(free=False))
        if tok == 'inv':
            x = node(free=False)
            if x[0] == 'mul' and x[1] != 0 and recip_alternatives:
                return ('mul', Fraction(1), [('recip', x[1], x[2])])
            return ('pow', x, ('num', Fraction(-1)))
        if tok == 'pow':
            base = node(free=False)
            return ('pow', base, node(free=False))
        if _NUMBER.match(tok):
            v = Fraction(tok)
            if max(abs(v.numerator), v.denominator) >= _HELD:  # kept as written: a part of its own
                return ('big', v)
            return ('num', v)
        if tok in SYMBOLS:
            return ('sym', tok)
        if arity.get(tok, 0):
            return ('fun', tok, [node(free=True) for _ in range(arity[tok])])
        return ('sym', 'variable')

    tree = node(free=True)
    assert pos == len(tokens), f'unread tokens in {tokens}'
    return tree


# --- the formula over the parts, for every form the serialization can stand for -------------

_CAP = 1 << 14


def _sums(sets):
    out = {0}
    for s in sets:
        out = {a + b for a in out for b in s}
        assert len(out) <= _CAP, 'too many forms to enumerate'
    return out


def _coefficient(c) -> int:
    return 0 if abs(c) == 1 else L(c)


def _odd_integer(t) -> bool:
    return t[0] == 'num' and t[1].denominator == 1 and t[1] >= 3 and t[1].numerator % 2 == 1


def _absorbing(t) -> bool:
    if t[0] == 'sym':
        return t[1] in SIGNLESS
    if t[0] == 'mul':
        return any(f[0] == 'sym' and f[1] in SIGNLESS for f in t[2])
    return False


def _carrier(f) -> bool:
    """A part whose sign can move inside it: an odd function, an odd power, a sum."""
    if f[0] == 'fun':
        return f[1] in ODD or (f[1] == 'rootn' and _odd_integer(f[2][1]))
    if f[0] == 'pow':
        return _odd_integer(f[2])
    if f[0] == 'add':
        return not any(_absorbing(t) for t in f[1])
    return False


def _head(name) -> int:
    return FUNCTIONS.get(name, OTHER_FUNCTION) * MILLI


def prices(t) -> set:
    """The formula's price of every form the part t can stand for, in milli-bits."""
    kind = t[0]
    if kind in ('num', 'big'):
        return {L(t[1])}
    if kind == 'sym':
        return {SYMBOLS[t[1]] * MILLI}
    if kind == 'fun':
        return {_head(t[1]) + s for s in _sums(prices(a) for a in t[2])}
    if kind == 'pow':
        e = t[2]
        exponent = {0} if (e[0] == 'num' and abs(e[1]) == 1) else prices(e)  # x^-1 is free
        return {_power() + s for s in _sums([prices(t[1]), exponent])}
    if kind == 'add':
        upright = {_sum() + s for s in _sums(prices(x) for x in t[1])}
        if not t[2] or any(_absorbing(x) for x in t[1]):
            return upright
        wrapped = _product_head() + _sum()  # -1 times the sum of the negated terms
        return upright | {wrapped + s for s in _sums(negated(x) for x in t[1])}
    if kind == 'mul':
        return _product_prices(t[1], t[2], flip=False)
    if kind == 'recip':
        return prices(('pow', ('mul', t[1], t[2]), ('num', Fraction(-1))))
    raise ValueError(t)


def _reciprocal(f):
    """1/f as a factor: a power's exponent changes sign, anything else gets the exponent -1."""
    if f[0] == 'pow' and f[2][0] == 'num':
        return ('pow', f[1], ('num', -f[2][1]))
    return ('pow', f, ('num', Fraction(-1)))


def _product_prices(c, factors, flip):
    """c * factors (or its negation), over each form a divisor product 1/(k*P) may take:
    k*P in the divisor, k in front of 1/P, or k in front of one reciprocal per factor."""
    recips = [f for f in factors if f[0] == 'recip']
    out = set()
    for choice in range(3 ** len(recips)):
        cc, fs, i = c, [], 0
        for f in factors:
            if f[0] != 'recip':
                fs.append(f)
                continue
            how = choice // 3 ** i % 3
            k, inner = f[1], f[2]
            if how == 0:
                fs.append(('pow', ('mul', k, inner), ('num', Fraction(-1))))
            elif how == 1:
                cc /= k
                fs.append(_reciprocal(inner[0] if len(inner) == 1 else ('mul', Fraction(1), inner)))
            else:
                cc /= k
                fs += [_reciprocal(g) for g in inner]
            i += 1
        cc = -cc if flip else cc
        if cc == 1 and len(fs) == 1:  # a lone factor is no product
            out |= prices(fs[0])
        else:
            out |= {_product_head() + _coefficient(cc) + s for s in _sums(prices(f) for f in fs)}
    return out


def negated(t) -> set:
    """The prices of -t: the sign rides a product's number, moves into an odd function or a
    sum, or a product with -1 carries it."""
    kind = t[0]
    if kind == 'num':
        return {L(-t[1])}
    if kind == 'sym':
        own = SYMBOLS[t[1]] * MILLI
        return {own} if t[1] in SIGNLESS else {own + _product_head()}
    if kind == 'mul':
        c, factors = t[1], t[2]
        if any(_absorbing(f) for f in factors):
            return prices(t)
        out = _product_prices(c, factors, flip=True)
        for i, f in enumerate(factors):
            if _carrier(f):
                rest = _sums(prices(g) for j, g in enumerate(factors) if j != i)
                out |= {_product_head() + _coefficient(c) + a + b
                        for a in rest for b in _negated_inside(f)}
        if len(factors) == 1 and c == -1:
            out |= prices(factors[0])
        return out
    out = {_product_head() + s for s in prices(t)}
    if _carrier(t):
        out |= _negated_inside(t)
    return out


def _negated_inside(f) -> set:
    if f[0] == 'fun':
        return {_head(f[1]) + s for s in _sums([negated(f[2][0])] + [prices(a) for a in f[2][1:]])}
    if f[0] == 'pow':
        return {_power() + s for s in _sums([negated(f[1]), prices(f[2])])}
    if f[0] == 'add':
        return {_sum() + s for s in _sums(negated(x) for x in f[1])}
    raise ValueError(f)


def written_price(tokens, arity) -> int:
    """The formula's price of a tagged expression exactly as written: no form enumerated."""
    (price,) = prices(_fixed(parse(tokens, arity, recip_alternatives=False)))
    return price


def _fixed(t):
    if t[0] == 'add':
        return ('add', [_fixed(x) for x in t[1]], False)
    if t[0] in ('fun',):
        return (t[0], t[1], [_fixed(a) for a in t[2]])
    if t[0] == 'mul':
        return ('mul', t[1], [_fixed(f) for f in t[2]])
    if t[0] == 'pow':
        return ('pow', _fixed(t[1]), _fixed(t[2]))
    return t


# --------------------------------------------------------------------------------------------
# Engines and the page
# --------------------------------------------------------------------------------------------

def _load(name):
    import simplipy as sp
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return sp.SimpliPyEngine.load(name, install=True)
    except Exception as error:  # noqa: BLE001 -- an unreachable asset is reported below
        if os.environ.get('SIMPLIPY_TEST_REQUIRE_ASSETS'):
            pytest.fail(f'SIMPLIPY_TEST_REQUIRE_ASSETS is set but {name} cannot be loaded: {error}')
        pytest.skip(f'{name} cannot be loaded: {error}')


@pytest.fixture(scope='module')
def engine():
    return _load('acj-5-4-llm')


@pytest.fixture(scope='module')
def bare():
    """The rules-less engine over the same vocabulary: its answer is the reading."""
    return _load('base')


def reading(bare, expression, mode):
    tokens = bare._core.parse(expression, True, False) if isinstance(expression, str) else list(expression)
    return bare._core.ac_simplify_in_mode(tokens, 1, _RULE_MODE[mode], 'tagged', 0)


def page_tables(marker):
    """Every markdown table that follows `marker` on the page: (header, rows)."""
    lines = open(PAGE).read().split('\n')
    tables = []
    for i, line in enumerate(lines):
        if line.strip() != marker:
            continue
        rows = []
        for row in lines[i + 1:]:
            if not row.startswith('|'):
                break
            rows.append([cell.strip() for cell in row.strip().strip('|').split('|')])
        tables.append((rows[0], [r for r in rows[2:]]))
    assert tables, f'no {marker} on the page'
    return tables


def _code(cell) -> str:
    m = re.match(r'^`([^`]+)`', cell)
    assert m, f'the first cell names no expression: {cell!r}'
    return m.group(1)


def _bits(value) -> int:
    return round(float(value) * MILLI)


def price_rows():
    """(expression, mode, printed price in milli-bits, parts cell or None) for every row."""
    out = []
    for header, rows in page_tables('<!-- price-table -->'):
        columns = {name.lower(): i for i, name in enumerate(header)}
        bits = next((i for name, i in columns.items() if name.startswith('bits')), None)
        for row in rows:
            expression = _code(row[0])
            if bits is not None:
                parts = row[columns['parts']] if 'parts' in columns else None
                out.append((expression, 'f64', _bits(row[bits]), parts))
            for mode in MODES:
                if mode in columns:
                    out.append((expression, mode, _bits(row[columns[mode]]), None))
    return out


PRICE_ROWS = price_rows()


# --------------------------------------------------------------------------------------------
# 1. the price tables
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('expression,mode,printed,parts', PRICE_ROWS,
                         ids=[f'{r[0]}|{r[1]}' for r in PRICE_ROWS])
def test_every_printed_price_is_the_shipped_price(engine, expression, mode, printed, parts):
    assert engine.complexity(expression, mode=mode) == printed


def _parts(cell):
    """[(name, milli-bits)] from a parts cell: `sum 3 + `x1` 6 + ...`."""
    out = []
    for piece in cell.split(' + '):
        m = re.match(r'^(`[^`]+`|[a-z]+) (\d+(?:\.\d+)?)$', piece.strip())
        assert m, f'unreadable part {piece!r}'
        out.append((m.group(1), _bits(m.group(2))))
    return out


@pytest.mark.parametrize('expression,parts', [(r[0], r[3]) for r in PRICE_ROWS if r[3]],
                         ids=[r[0] for r in PRICE_ROWS if r[3]])
def test_every_listed_part_costs_its_price_and_the_parts_add_up(engine, expression, parts):
    named = {'sum': SYMBOLS['sum'], 'product': SYMBOLS['product'], 'power': SYMBOLS['power']}
    total = 0
    for name, price in _parts(parts):
        if name in named:
            assert price == named[name] * MILLI, name
        else:
            symbol = name.strip('`')
            if symbol in FUNCTIONS:
                assert price == FUNCTIONS[symbol] * MILLI, symbol
            else:
                assert engine.complexity(symbol) == price, symbol
        total += price
    assert total == engine.complexity(expression)


def test_every_read_as_cell_is_the_typed_formulas_reading(bare):
    """Each `read as` cell, typed itself, reads to the same canonical form as its row's formula."""
    checked = 0
    for header, rows in page_tables('<!-- price-table -->'):
        columns = [name.lower() for name in header]
        if 'read as' not in columns:
            continue
        for row in rows:
            typed, shown = _code(row[0]), _code(row[columns.index('read as')])
            assert reading(bare, shown, 'f64') == reading(bare, typed, 'f64'), (typed, shown)
            checked += 1
    assert checked >= 30


# --------------------------------------------------------------------------------------------
# 2. the symbol table and the prices in the prose
# --------------------------------------------------------------------------------------------

_SYMBOL_WORDS = ('variable', 'sum', 'product', 'power')
_SYMBOL_NAMES = {'pi': 'np.pi', 'e': 'np.e', 'inf': 'float("inf")', '-inf': 'float("-inf")',
                 'nan': 'float("nan")', '<constant>': '<constant>'}


def test_the_symbol_table_is_the_formulas_table():
    (header, rows), = page_tables('<!-- symbol-table -->')
    seen = set()
    for row in rows:
        bits = _bits(row[1])
        if 'any other function' in row[0]:
            assert OTHER_FUNCTION * MILLI == bits
            seen.add('other')
        for word in (w for w in _SYMBOL_WORDS if re.search(rf'\b{w}\b', row[0])):
            assert SYMBOLS[word] * MILLI == bits, word
            seen.add(word)
        for name in re.findall(r'`([^`]+)`', row[0]):
            if name in FUNCTIONS:
                assert FUNCTIONS[name] * MILLI == bits, name
                seen.add(name)
            elif name in _SYMBOL_NAMES:
                assert SYMBOLS[_SYMBOL_NAMES[name]] * MILLI == bits, name
                seen.add(_SYMBOL_NAMES[name])
    assert seen == set(FUNCTIONS) | set(SYMBOLS) | {'other'}, 'a symbol is missing from the page'


def test_the_symbol_table_is_the_same_for_every_engine(engine, bare):
    """The prices are SimpliPy's: an engine without rules, and any variable name, agree."""
    for expression in ('x1', 'x12', 'y', 'pi', 'e', 'inf', 'nan', '<constant>', 'x1 + x2',
                       'x1*x2', 'x1^x2', *(f'{f}(x1)' for f in FUNCTIONS if f != 'rootn')):
        assert bare.complexity(expression) == engine.complexity(expression), expression
    assert {engine.complexity(v) for v in ('x1', 'x12', 'x99', 'y')} == {SYMBOLS['variable'] * MILLI}


def test_the_prices_in_the_prose(engine, bare):
    text = ' '.join(open(PAGE).read().split())
    arity = dict(bare.operator_arity)
    c = engine.complexity

    def written(expression):
        return written_price(bare.to_tagged(expression), arity)

    def read(expression):
        return reading(bare, expression, 'f64')

    from simplipy.engine import DEFAULT_ENGINE

    big = 2 ** 127
    sin2 = repr(math.sin(2.0))
    long_m = '34359738369/109951162777600000000000000000000'
    m_of_long_m = 34359738369 * 5 ** 40  # its decimal digits: m * 10^-60
    powers = [(a, b) for a in (2, 3, -1, -2) for b in (2, 3, -1, -3)]
    claims = [
        ("`install=True` downloads `acj-5-4-llm`, SimpliPy's default engine",
         [DEFAULT_ENGINE == 'acj-5-4-llm']),
        ('In `real` and `permissive`, reading also cancels some functions applied to their inverses, '
         'identities that hold for every real number: `atanh(tanh(x1))`, `log(exp(x1))`, '
         '`asinh(sinh(x1))` and `sinh(asinh(x1))` are read as `x1`, and `acosh(cosh(x1))` as '
         '`abs(x1)`',
         [reading(bare, f'{e}(x1))', m) == ['x1'] and reading(bare, f'{e}(x1))', 'f64') != ['x1']
          for e in ('atanh(tanh', 'log(exp', 'asinh(sinh', 'sinh(asinh') for m in ('real', 'permissive')]
         + [reading(bare, 'acosh(cosh(x1))', m) == reading(bare, 'abs(x1)', m) for m in ('real', 'permissive')]
         + [c('atanh(tanh(x1))') == 22000, c('atanh(tanh(x1))', mode='real') == 6000,
            c('atanh(tanh(x1))', mode='permissive') == 6000]),
        ('`permissive`, which accepts identities that fail for some inputs, also reads '
         '`rootn(x1^2, 2)` as `x1`. An engine without rules reads these the same way',
         [reading(bare, 'rootn(x1^2, 2)', 'permissive') == ['x1'],
          reading(bare, 'rootn(x1^2, 2)', 'real') != ['x1']]
         + [engine.complexity(e, mode=m) == bare.complexity(e, mode=m)
            for e in ('atanh(tanh(x1))', 'log(exp(x1))', 'acosh(cosh(x1))', 'rootn(x1^2, 2)')
            for m in ('real', 'permissive')]),
        ('a sum or a product costs 3 bits, `sin` 6 bits, and `asin` 8 bits',
         [c('x1 + x2') - c('x1') - c('x2') == 3000, c('x1*x2') - 2 * c('x1') == 3000,
          c('sin(x1)') - c('x1') == 6000, c('asin(x1)') - c('x1') == 8000]),
        ('`2` costs 2.585 bits, `3.14` costs 10.884 bits, and `3.141592653589793` costs 56.480 bits',
         [c('2') == 2585, c('3.14') == 10884, c('3.141592653589793') == 56480]),
        ('choosing among 3 equally likely options takes \\(\\log_2 3 \\approx 1.585\\) bits',
         [ell(2) == 1585]),
        ('`pi*x1` costs 13 bits and `3.141592653589793*x1` costs 65.480 bits, although their values '
         'agree to 16 digits',
         [c('pi*x1') == 13000, c('3.141592653589793*x1') == 65480, math.pi == 3.141592653589793]),
        # reading
        ('merges a power of a power into one power when both exponents are whole numbers, as '
         '`(x3^2)^-1` becomes `x3^-2`',
         [read(f'({x}^{a})^{b}') == read(f'{x}^{a * b}')
          for x in ('x1', '(x1 + x2)', '(x1 - x2)', 'sin(x1)') for a, b in powers]
         + [read('(x3^2)^-1') == read('x3^-2')]),
        ('as a fraction of two 128-bit integers, whose numerator and denominator are each below '
         '\\(2^{127} \\approx 1.7 \\cdot 10^{38}\\) in size',
         [round(big / 10 ** 38, 1) == 1.7,
          c(str(big - 1)) == L(big - 1), c(str(big)) != L(big),
          c(str(1 - big)) == L(1 - big), c(str(-big)) != L(-big),
          c(f'1/{big - 1}') == L(Fraction(1, big - 1)), c(f'1/{big + 1}') != L(Fraction(1, big + 1)),
          c(f'{big - 1}/{big - 2}') == L(Fraction(big - 1, big - 2))]),
        ('such as `1e-300`, whose denominator has 301 digits, is kept as written',
         [len(str(Fraction('1e-300').denominator)) == 301, read('1e-300') == ['1e-300'],
          engine.simplify('1e-300*x1') == '1e-300*x1']),
        ('and so is a computation whose result would fall outside it, such as `2^200`',
         [read('2^200') == ['pow', '2', '200'], 2 ** 200 > 2 ** 127,
          c('2^200') == _power() + L(2) + L(200),
          read('1e20*1e20') == ['<mul>', str(10 ** 20), str(10 ** 20), '</mul>']]),
        ('drops a factor 1, a term 0 and an exponent 1, and reads `x1^0` as `1` and `0*x1` as `0`',
         [read(e) == ['x1'] for e in ('1*x1', 'x1*1', 'x1 + 0', '0 + x1', 'x1^1', 'x1/1')]
         + [read('x1^0') == ['1'], read('0*x1') == ['0']]),
        ('replaces a function applied to a number, such as `sin(0)`, by its value when the value costs '
         'fewer bits than the function',
         [c('sin(0)') == c('0'), read('sin(0)') == read('0'), c('sin(2)') < L(Fraction(sin2))]),
        ('applies a few more identities that the mode accepts, such as `rootn(x1^3, 3)` read as `x1`',
         [read('rootn(x1^3, 3)') == ['x1']]),
        ('writes a power with the exponent \\(1/2\\), such as `x1^(1/2)`, as `rootn(x1, 2)`',
         [read('x1^(1/2)') == read('x1^0.5') == read('rootn(x1, 2)'),
          read('(x1 + x2)^(1/2)') == read('rootn(x1 + x2, 2)')]),
        ('drops a number that multiplies a `<constant>` or is added to it',
         [c('<constant> + 2') == c('<constant>/2') == c('<constant>') == 67000,
          c('2*<constant>*x1') == c('<constant>*x1'), c('x1 + <constant> + 2') == c('x1 + <constant>')]),
        ('`0.1 + 0.2` is read as \\(3/10\\): SimpliPy adds \\(1/10\\) and \\(1/5\\) exactly',
         [c('0.1 + 0.2') == c('3/10'), Fraction('0.2') == Fraction(1, 5),
          Fraction('0.1') + Fraction('0.2') == Fraction(3, 10)]),
        (f'`sin(2)` stays, because its float64 value, {sin2}, would cost more',
         [c('sin(2)') == written('sin(2)'), L(Fraction(sin2)) > c('sin(2)')]),
        ('`x1^(1/2)` costs 14.585 bits because it is read as `rootn(x1, 2)`',
         [c('x1^(1/2)') == c('rootn(x1, 2)') == 14585]),
        ('`x1/9.81` is read as `100/981*x1`: numbers are exact, so dividing by \\(9.81\\) multiplies by '
         'its exact reciprocal, \\(100/981\\)',
         [read('x1/9.81') == read('100/981*x1'), 1 / Fraction('9.81') == Fraction(100, 981)]),
        ('The fixed form can cost more than the typed one, as `x1^(1/2)` shows',
         [_power() + c('x1') + L(Fraction(1, 2)) == 12585 < c('x1^(1/2)')]),
        ('`acos(-1)` is read as written, at 11 bits, and simplification turns it into `pi`, at 4',
         [c('acos(-1)') == written('acos(-1)') == 11000, engine.simplify('acos(-1)') == 'pi',
          c('pi') == 4000]),
        # minus signs
        ('A minus sign on a number held exactly is its sign bit: `-2` costs 3.585 bits, `2` costs '
         '2.585',
         [c('-2') == 3585, c('2') == 2585]),
        ('A minus sign on a product that has a number held exactly merges into that number, as in '
         '`-2*x1`',
         [c('-2*x1') - c('2*x1') == 1000, read('-2*x1') == ['<mul>', '-2', 'x1', '</mul>']]),
        ('`-x1*x2` costs what `x1*x2` costs', [c('-x1*x2') == c('x1*x2')]),
        ('`-x1` costs 9 bits, and `-1e-300` costs 3 bits more than `1e-300`',
         [c('-x1') == 9000, c('-1e-300') - c('1e-300') == _product_head(),
          c('-1e-300*x1') == c('1e-300*x1')]),
        ('On `inf`, a minus sign gives the symbol `-inf`',
         [c('-inf') == c('inf') == SYMBOLS['float("-inf")'] * MILLI,
          read('-inf') == ['float("-inf")']]),
        ('`-x1 - x2 - x3` is read as `-(x1 + x2 + x3)`, since one factor \\(-1\\) costs less than '
         'three negated terms',
         [c('-x1 - x2 - x3') == 24000,
          prices(parse(read('-x1 - x2 - x3'), arity, False)) == {24000, 30000}]),
        ('Inside a function\'s argument the same holds: `sin(-x2 - x3)` is read as `sin(-(x2 + x3))`',
         [c('sin(-x2 - x3)') == c('sin(-(x2 + x3))') == 24000,
          prices(parse(read('sin(-x2 - x3)'), arity, False)) == {24000, 27000}]),
        ('`x1 - x2 - x3` and `-(-x1 + x2 + x3)` cost the same either way, and both read as '
         '`x1 - x2 - x3`',
         [read('x1 - x2 - x3') == read('-(-x1 + x2 + x3)'),
          prices(parse(read('x1 - x2 - x3'), arity, False)) == {27000},
          engine.simplify('-(-x1 + x2 + x3)') == engine.simplify('x1 - x2 - x3') == 'x1 - x2 - x3']),
        ('`sin`, `tan`, `asin`, `atan`, `sinh`, `tanh`, `asinh` and `atanh`',
         # a number inside each function's domain: 1/2, and 3/2 for acosh
         [all((read(f'x1*{f}(-{v})') == read(f'-x1*{f}({v})')) == (f in ODD)
              for f, v in ((f, '3/2' if f == 'acosh' else '1/2') for f in FUNCTIONS if f != 'rootn')),
          {'sin', 'tan', 'asin', 'atan', 'sinh', 'tanh', 'asinh', 'atanh'} == ODD]),
        ('When its argument is a number or a sum with terms of both signs, a minus sign can sit inside '
         'the argument or in front of the function',
         [read('-tanh(x1 - x2)') == read('tanh(x2 - x1)'),
          read('x1*tanh(x2 - x3 - x4)') == read('-x1*tanh(x3 + x4 - x2)'),
          read('-tanh(x1 + x2)') != read('tanh(-x1 - x2)'),
          read('x1*tanh(-x2 - x3)') != read('-x1*tanh(x2 + x3)'),
          c('x1*tanh(-x2 - x3)') > c('-x1*tanh(x2 + x3)')]),
        ('`x1*sin(-2)` is read as `-x1*sin(2)`, since the factor \\(-1\\) is free and the sign of '
         '\\(-2\\) costs a bit',
         [read('x1*sin(-2)') == read('-x1*sin(2)') == ['<mul>', '-1', 'x1', 'sin', '2', '</mul>'],
          L(-2) - L(2) == 1000, c('x1*sin(-2)') == 17585]),
        ('`-5*sin(2)` and `5*sin(-2)` cost the same, and both read as `5*sin(-2)`',
         [L(-5) + L(2) == L(5) + L(-2),
          read('-5*sin(2)') == read('5*sin(-2)') == ['<mul>', '5', 'sin', '-2', '</mul>']]),
        ('`-tanh(x1 - x2)` is read as `tanh(x2 - x1)`, which saves the 3 bits of the product',
         [c('-tanh(x1 - x2)') == c('tanh(x2 - x1)') == 26000,
          _product_head() + c('tanh(x1 - x2)') - c('-tanh(x1 - x2)') == 3000]),
        ('For any other argument of an odd function, such as `-x2`, `-2*x2` or `-(x2 + x3)`, the sign '
         'stays inside: `x1*sin(-x2)` costs 24 bits, although `-x1*sin(x2)` costs 21, and '
         '`x1*sin(-x2 - x3)` costs 33, although `-x1*sin(x2 + x3)` costs 30',
         [c('x1*sin(-x2)') == 24000, c('-x1*sin(x2)') == 21000,
          c('x1*sin(-x2 - x3)') == 33000, c('-x1*sin(x2 + x3)') == 30000,
          engine.simplify('x1*sin(-x2)') == 'x1*sin(-x2)']
         + [read(f'x1*sin(-{a})') != read(f'-x1*sin({a})') for a in ('x2', '2*x2', '(x2 + x3)')]
         + [engine.simplify(f'x1*sin(-{a})') == f'x1*sin(-{b})'
            for a, b in (('x2', 'x2'), ('2*x2', '2*x2'), ('(x2 + x3)', 'x2 - x3'))]),
        ('An odd power takes a minus sign out of its base, unless the base is a sum with terms of both '
         'signs, which the sum rule orients: `(-x1)^3` is read as `-x1^3`, at the same 15 bits. An '
         'even power drops a minus sign on its base, and `cos`, `cosh` and `abs` drop one on their '
         'argument: `(-x1)^2` is read as `x1^2`, and `cos(-x1)` as `cos(x1)`',
         [read(f'(-{u})^{n}') == read(f'-{u}^{n}') for u in ('x1', 'sin(x1)', 'exp(x1)', '(x1 + x2)')
          for n in (3, 5, -3)]
         + [c('(-x1)^3') == c('-x1^3') == 15000, written('(-x1)^3') == written('-x1^3') == 15000,
            read('-(x1 - x2)^3') == read('(x2 - x1)^3') == bare.to_tagged('(x2 - x1)^3'),
            read('(-x1)^3') == read('-x1^3'), read('(-x1)^5') == read('-x1^5'),
            read('(-x1)^2') == read('x1^2'), read('(-x1)^4') == read('x1^4'),
            read('(-x1 - x2)^2') == read('(x1 + x2)^2')]
         + [read(f'{f}(-x1)') == read(f'{f}(x1)') for f in ('cos', 'cosh', 'abs')]),
        ('so `-(x1 + x2 + x3)` prints as `-x1 - x2 - x3`',
         [engine.simplify('-(x1 + x2 + x3)') == '-x1 - x2 - x3']),
        # the formula
        ('the factor \\(-1\\) of a product that has no number held exactly, and the exponent '
         '\\(-1\\) of a power, as in `x1/x2` or `x1^-1`',
         [c('-x1*x2') == c('x1*x2'), c('-1e-300*x1') == c('1e-300*x1'),
          c('x1/x2') - c('x1*x2') == _power(), c('x1^-1') == c('1/x1') == _power() + c('x1'),
          c('(x1 + x2)^-1') == _power() + c('x1 + x2')]),
        ('\\(\\ell(1) = 1\\), \\(\\ell(2) = 1.585\\), \\(\\ell(3) = 2\\)',
         [ell(1) == 1000, ell(2) == 1585, ell(3) == 2000]),
        ('it is never below 1 bit, which matters only for the number 0',
         [ell(0) == 1000, math.log2(1 + 1) == 1]),
        ('\\(34359738369/(2^{60} \\cdot 5^{20})\\) has none, because its \\(m\\) is about '
         '\\(3.1 \\cdot 10^{38}\\)',
         [109951162777600000000000000000000 == 2 ** 60 * 5 ** 20,
          Fraction(long_m) == Fraction(m_of_long_m, 10 ** 60), m_of_long_m % 10 != 0,
          m_of_long_m >= big, round(m_of_long_m / 10 ** 38, 1) == 3.1,
          c(long_m) == MILLI + ell(34359738369) + ell(109951162777600000000000000000000),
          c(long_m) > L(Fraction(long_m))]),
        ('A number kept as written is measured from its written digits by the same two codes',
         [c('1e-300') == MILLI + ell(1) + ell(-300), c('1.000e-300') == c('1e-300')]),
        ('So `1000` and `0.001` both cost 4 bits, and `x1*x2` and `-x1*x2` both cost 15',
         [c('1000') == c('0.001') == 4000, c('x1*x2') == c('-x1*x2') == 15000]),
        # the symbol table
        ('its bits are close to what their frequencies give in a corpus of 400 templates', []),
        ('a kind of part, such as a sum or `tanh`, that makes up a fraction \\(f\\) of all parts in the '
         'corpus would cost \\(\\log_2(1/f)\\) bits', []),
        ('`pi` and `e` occur nowhere in the corpus; they cost less than the cheapest formula for '
         'either value',
         # any other formula holds an operator or a function and a part: at least 3 + 2 bits
         [c('pi') == c('e') == 4000 < min(SYMBOLS['sum'], *FUNCTIONS.values()) * MILLI + L(0),
          written('acos(-1)') == 11000, written('exp(1)') == 8000]),
        ('`<constant>` costs 67 bits: the largest \\(L(v)\\) of a float64 number written in its '
         'shortest decimal spelling is 66.649 bits, and 67 is that, rounded up',
         [L(Fraction('-8.9002954340287245e-308')) == 66649,
          repr(-8.9002954340287245e-308) == '-8.9002954340287245e-308',
          math.ceil(66.649) == SYMBOLS['<constant>'] == 67]),
        ('A negative number kept as written can still cost more than a slot, since its minus sign makes '
         'a 3-bit product instead of a sign bit: `-8.9002954340287245e-308` costs 68.649 bits',
         [c('-8.9002954340287245e-308') == 68649 > 67000,
          read('-8.9002954340287245e-308') == ['neg', '8.9002954340287245e-308'],
          c('-8.9002954340287245e-308') - c('8.9002954340287245e-308') == _product_head()]),
        # numbers
        ('`0.5` and `1/2` cost the same', [c('0.5') == c('1/2')]),
        ('Ten is the selector bit and two whole numbers of 1 bit each, 3 bits; seven is the selector '
         'bit and 3 bits for 7, 4 bits',
         [ell(1) == 1000, ell(7) == 3000, L(10) == 3000, L(7) == 4000]),
        ('writing the 6 costs 2.807 bits', [ell(6) == 2807]),
        ('`0.001` costs the same as `1000`', [c('0.001') == c('1000')]),
        # examples
        ('Its division by `x3^2` is the power `(x3^2)^-1`, which reading merges into `x3^-2`; the '
         'exponent \\(-2\\) costs 3.585 bits',
         [read('x1/x3^2') == read('x1*(x3^2)^-1') == read('x1*x3^-2'), c('-2') == 3585]),
        ('`x1^2 + 2*x1*x2 + x2^2` costs more than twice `(x1 + x2)^2`',
         [c('x1^2 + 2*x1*x2 + x2^2') > 2 * c('(x1 + x2)^2')]),
        ('it costs 95.663 bits instead of 44.768',
         [c('6.283185307179586*rootn(x1/9.81, 2)') == 95663, c('2*pi*rootn(x1/9.81, 2)') == 44768]),
        ('The decimal costs 57.480 bits, where `2` and `pi` cost 2.585 and 4',
         [c('6.283185307179586') == 57480, c('2') == 2585, c('pi') == 4000]),
        # calling complexity()
        ("In tokens, `pi` and `e` are written with NumPy's names, `np.pi` and `np.e`; a token `pi` "
         'costs 6 bits, as a variable does',
         [c(['np.pi']) == c('pi'), c(['np.e']) == c('e'), c(['pi']) == c(['e']) == c('x1') == 6000]),
        ('`sin(x1)` costs 12 bits and `asin(x1)` costs 14, with the same number of tokens in the same '
         'shape',
         [c('sin(x1)') == 12000, c('asin(x1)') == 14000, c(['sin', 'x1']) == 12000,
          c(['asin', 'x1']) == 14000]),
    ]
    for snippet, checks in claims:
        assert snippet in text, f'the page no longer says: {snippet}'
        assert all(checks), snippet


def test_the_symbol_table_follows_the_corpus_where_the_page_says(bare):
    """Sums, products, powers, rootn and the inverse and hyperbolic functions cost within one bit
    of log2(1/f) over the 400 templates of the benchmark corpus, where pi and e never occur."""
    with open(CORPUS) as f:
        corpus = json.load(f)
    assert len(corpus) == 400
    assert not any(t in ('pi', 'e', 'np.pi', 'np.e') for row in corpus for t in row)
    census = bare._core.ac_node_census([list(row) for row in corpus])
    total = sum(census.values())
    page = {'Add': SYMBOLS['sum'], 'Mul': SYMBOLS['product'], 'Pow': SYMBOLS['power'],
            **{f'Fun:{f}': bits for f, bits in FUNCTIONS.items()
               if f == 'rootn' or bits == 8}}
    for kind, bits in page.items():
        assert abs(math.log2(total / census[kind]) - bits) < 1, kind
    assert math.log2(total / census['Leaf']) < SYMBOLS['variable']


def test_no_float64_number_costs_more_than_66_649_bits_by_the_formula():
    """L of a float64 number in its shortest decimal spelling: the page's maximum, 66.649 bits,
    on random floats, on the floats around the maximum, and by the bound m < 10 * 2^53,
    |k| <= 343 of a shortest spelling."""
    rng = random.Random(67)
    floats = [struct.unpack('<d', struct.pack('<Q', rng.getrandbits(64)))[0] for _ in range(20000)]
    top = 2.0 ** -1020
    x = top
    for _ in range(2000):
        x = math.nextafter(x, 0.0)
        floats.append(x)
    worst = max(L(Fraction(repr(-abs(x)))) for x in floats if math.isfinite(x) and x)
    assert worst == 66649
    assert 2 * MILLI + ell(10 * 2 ** 53) + ell(343) < 67 * MILLI


def test_every_printed_price_has_three_decimals_or_none():
    for _, rows in page_tables('<!-- price-table -->'):
        for row in rows:
            for cell in row[1:]:
                for number in re.findall(r'(?<![\w.`/^-])\d+(?:\.\d+)?(?![\w.`/])', re.sub(r'`[^`]*`|\\\(.*?\\\)', '', cell)):
                    assert re.fullmatch(r'\d+(\.\d{3})?', number), (row[0], cell)


def test_the_formula_is_exact_for_every_number_held_as_an_exact_fraction(engine):
    """Every number whose p, q and decimal digits m are below 2^127."""
    rng = random.Random(14)
    checked = 0
    while checked < 3000:
        p = rng.randint(1, 10 ** rng.randint(1, 39) - 1)
        if rng.random() < 0.5:
            q = 2 ** rng.randint(0, 126) * 5 ** rng.randint(0, 54)
        else:
            q = rng.randint(1, 10 ** rng.randint(1, 39) - 1)
        n = Fraction(p, q) * (-1 if rng.random() < 0.3 else 1)
        a = b = 0
        rest = n.denominator
        while rest % 2 == 0:
            rest, a = rest // 2, a + 1
        while rest % 5 == 0:
            rest, b = rest // 5, b + 1
        m = abs(n.numerator) * 2 ** (max(a, b) - a) * 5 ** (max(a, b) - b) if rest == 1 else 0
        if max(abs(n.numerator), n.denominator, m) >= 2 ** 127:
            continue
        spelling = f'{n.numerator}/{n.denominator}' if n.denominator != 1 else str(n.numerator)
        assert engine.complexity(spelling) == L(n), spelling
        checked += 1


# --------------------------------------------------------------------------------------------
# 3. the formula
# --------------------------------------------------------------------------------------------

def test_the_price_formula_as_stated_gives_every_page_price(engine, bare):
    """The page's price formula, applied as the page states it to the read form of every row:
    each symbol at its table price, a sum, a product and a power once each, every number at
    L(v), and the factor -1 of a product without a number held exactly and the exponent -1 of
    a power at nothing. Of a sum's two forms, the cheaper one (the page's rule for sums)."""
    arity = dict(bare.operator_arity)
    for expression, mode, _, _ in PRICE_ROWS:
        form = reading(bare, expression, mode)
        options = prices(parse(form, arity, recip_alternatives=mode == 'permissive'))
        # a sum is read in whichever of its two forms costs less
        assert min(options) == engine.complexity(expression, mode=mode), (expression, mode, form)


def test_the_formula_prices_the_benchmark_corpus_in_each_mode(engine, bare):
    arity = dict(bare.operator_arity)
    with open(CORPUS) as f:
        corpus = json.load(f)
    assert len(corpus) == 400
    missed = []
    for tokens in corpus:
        for mode in MODES:
            form = reading(bare, tokens, mode)
            price = engine.complexity(form, mode=mode, canon='mode')
            if mode == 'f64':
                assert engine.complexity(form) == price  # the default call prices the same
            if price not in prices(parse(form, arity, recip_alternatives=mode == 'permissive')):
                missed.append((mode, ' '.join(form), price))
    assert not missed, missed[:5]


@pytest.mark.parametrize('value', [
    0, 1, 2, 3, 7, 10, 12, 100, 1000, 12345, 10 ** 37, 2 ** 100 + 1, -1, -2, -1000,
    '1/2', '1/3', '2/3', '22/7', '355/113', '0.1', '0.5', '0.25', '0.001', '3.14', '-0.75',
    '6.674e-11', '1e-300', '1.5e-40', '9.81', '100/981', '0.30000000000000004',
    '12345678901234567890123456789012345678', '0.12345678901234567890123456789012345678',
    2 ** 127 - 1, -(2 ** 127 - 1), f'1/{2 ** 127 - 1}', f'{2 ** 127 - 1}/{2 ** 127 - 2}',
])
def test_the_formula_prices_every_number_held_exactly(engine, value):
    assert engine.complexity(str(value)) == L(Fraction(str(value)))


def test_the_formula_stops_at_2_to_the_127(engine):
    """The page's bound is where the formula stops being exact, not a margin."""
    for value in (2 ** 127, -(2 ** 127), Fraction(1, 2 ** 127 + 1)):
        assert engine.complexity(str(value)) != L(value), value


# --------------------------------------------------------------------------------------------
# 4. the guarantee, and 5. the rules
# --------------------------------------------------------------------------------------------

def test_simplification_never_raises_the_price_on_the_corpus(engine):
    with open(CORPUS) as f:
        corpus = json.load(f)
    for tokens in corpus:
        e = list(tokens)
        assert engine.complexity(engine.simplify(e)) <= engine.complexity(e)
        for mode in ('real', 'permissive'):
            answer = engine.simplify(e, mode=mode)
            assert engine.complexity(answer, mode=mode, canon='mode') <= \
                engine.complexity(e, mode=mode, canon='mode')


def test_the_rules_do_not_change_a_price(engine, bare):
    for expression, mode, printed, _ in PRICE_ROWS:
        assert bare.complexity(expression, mode=mode) == printed, (expression, mode)


# --------------------------------------------------------------------------------------------
# 6. docs/formal.md
# --------------------------------------------------------------------------------------------

def test_the_formal_pages_numbers(engine, bare):
    text = ' '.join(open(FORMAL).read().split())
    arity = dict(bare.operator_arity)
    c = engine.complexity

    def read(expression):
        return reading(bare, expression, 'f64')

    with open(CORPUS) as f:
        corpus = json.load(f)
    census = bare._core.ac_node_census([list(row) for row in corpus])
    leaf = math.log2(sum(census.values()) / census['Leaf'])
    variables = {t for row in corpus for t in row if re.fullmatch(r'x\d+', t)}
    claims = [
        ('($\\mathrm{acos}(-1) = 11$ bits, $\\exp(1) = 8$, so $4$ has room)',
         [written_price(bare.to_tagged('acos(-1)'), arity) == 11000,
          written_price(bare.to_tagged('exp(1)'), arity) == 8000, c('pi') == c('e') == 4000]),
        ('the class `Leaf` carries $1.64$ bits of the corpus census, a specific variable among its '
         'seventeen (`x1` to `x17`) carries $1.64 + \\log_2 17 = 5.73$',
         [round(leaf, 2) == 1.64, variables == {f'x{i}' for i in range(1, 18)},
          round(leaf + math.log2(17), 2) == 5.73, round(1.64 + math.log2(17), 2) == 5.73,
          c('x1') == SYMBOLS['variable'] * MILLI]),
        ('The `pow` constructor writes $\\mathrm{Pow}(t, 1/k)$ for an even $k \\geq 2$ as '
         '$\\mathrm{rootn}(t, k)$: at an even index the two spellings are the same function, the '
         'principal root, NaN on negatives. An odd unit fraction stays a power',
         [read(f'x1^(1/{k})') == read(f'rootn(x1, {k})') for k in (2, 4, 6, 10)]
         + [read(f'x1^(1/{k})')[0] == 'pow' for k in (3, 5)]
         + [engine.simplify('rootn(-4, 2)') == engine.simplify('(-4)^(1/2)') == 'nan',
            engine.simplify('rootn(-8, 3)') == '-2', engine.simplify('(-8)^(1/3)') == 'nan']),
        ('at $k = 2$ the head and index of `rootn(t, 2)` cost $6 + 2.585 = 8.585$ bits, against '
         '$3 + 3.585 = 6.585$ for a power\'s head and exponent $1/2$',
         [c('rootn(x1, 2)') - c('x1') == FUNCTIONS['rootn'] * MILLI + L(2) == 8585,
          SYMBOLS['power'] * MILLI + L(Fraction(1, 2)) == 6585, c('x1^(1/2)') == c('rootn(x1, 2)')]),
    ]
    for snippet, checks in claims:
        assert snippet in text, f'formal.md no longer says: {snippet}'
        assert all(checks), snippet
