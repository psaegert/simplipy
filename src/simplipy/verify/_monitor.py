# mypy: ignore-errors
"""The deployed-realization monitor: an end-to-end sweep of the engine, judged by the rule judge.

The monitor exercises the DEPLOYED engine end to end (``simplify`` on real expression shapes:
an adversarial list and a sampled corpus, literal and pre-masked, as deployed traffic arrives)
and judges every input -> output rewrite as the rule judge judges a rule
(``_contract.judge_rule``): the rewrite holds in the mode it was made in exactly where its
judged tier is one that mode licenses (``verify.MODE_TIERS``), as a rule's does. The input's
variables are the rule's ``?`` slots, its ``<constant>`` leaves the source's constants and the
output's the fitted ones. So the monitor has no evaluator of its own: every value it compares is the core's
table, read through the judge, and the engine's composition of rules and constructors is held
to the criterion every rule is held to. Where a per-rule scan judges each rule at its own
points, the monitor judges what the rules do together on whole expressions.

Verdict classes per rewrite: OK | VIOLATION | UNSCORED (a rewrite the judge cannot read, or
reaches no decision on). A violation the rules-empty engine commits too is the engine's own
(NATIVE-LINE), not the rule set's.

WHAT A WHOLE ANSWER IS HELD TO (F8). Every rule step the engine took (each fire's rule instance,
`_core.ac_simplify_fires`) must be licensed in the mode by the judge, and the rewrite as a whole
either licensed in the mode or, the engine's own algebra being exact in real arithmetic, true in
the reals with float64's own events kept: its disagreements with the input what rounding does
(a sign or a zero near a cancellation or a pole included) or a value where float64 reads the
input as NaN, never a zero's sign (F7) or an infinity gained or lost (at an overflow or
underflow the true value is accepted, the judge's range standard). `x/(x/(x^2 - x)) -> x*(x - 1)` is exact, though float64's rounding near
x = 1 keeps the two apart: it is OK. Where a step holds in the mode and not in the reals (a rule
float64 holds and the reals do not), the engine's algebra around it cannot be read apart from the
step, and a rewrite the mode does not license as a whole is UNSCORED.

A STEP IS HELD AT ITS OWN POINTS, AND IN ITS CONTEXT ONLY TO A ZERO'S SIGN. Each step is true in
the reals and keeps float64's kinds at its own points; what an operation around it does with
float64's rounding of its input is no step's promise. Where float64's kinds alone refuse an answer
true in the reals, the smallest part of the input the answer changed that holds as a step (below
every operation the two share with its other arguments unchanged, a reciprocal's and a negation's
spellings read as one) is read as the step, and around it only a zero's sign is read
(`_contract.answer_zero_sign`): a value that turns on a zero's sign, an infinity included, on a
region. `1/log(exp(x0)) -> 1/x0` is OK (+inf against 1e17 at x0 = 1e-17 is rounding under `1/`),
and so is `1/sin(np.pi - x1) -> 1/sin(x1)` (+inf against 8.2e15 at x1 = np.pi: the step
`sin(np.pi - x1) -> sin(x1)` gives 0 against 1.2e-16 there); `1/(0 + 0*x0) -> 1/(0*x0)` is not
(+inf against -inf for x0 < 0). Where the answer respells a product or a quotient, the factors it
changed, less those the two sides share, are a part too, read under the reciprocal where one
divides: `x/(a*x^2 + b*x) -> 1/(a*x + b)` changes `(a*x^2 + b*x)/x -> a*x + b`, which holds as a
step where float64's divisor of the answer vanishes a double away from the input's. Several
factors changed, as many on each side, are paired one to one (`_factor_pairings`), each pair a
step of its own. Rounding alone apart, a whole answer whose value turns on a zero's
sign on a region is refused too (`atan2(0*x0, -1) -> pi`). An infinity's sign at an isolated pole
is not promised (the judge's reading of a pole's sign): `1/(-x0 - 3) -> -1/(x0 + 3)` is OK, and
nor is any value where the input has an isolated two-signed pole: `1/(x0 - 3) + 1/(3 - x0) -> 0`.

A FOLD IS A STEP. In f64 the engine writes a number its own algebra combines from literals (a
product's coefficient, a sum's constant term) as the double nearest the exact operation on the
literals' doubles, and a long number at its double (`ac_simplify_trace`'s folds). Where an answer as
written is refused, each of its numbers that a fold wrote is read as the numbers it was folded from,
through the exact products and sums the engine held between folds (`_unfolded`): each fold is
judged as the ground rule it makes (`_fold_rules`), and the answer so read as an answer. `27.940055475362794*(x1 + 0.2856797429345068*x2)`, distributed to
`27.940055475362794*x1 + 7.981907865777503*x2`, is OK where the reals refute the answer as written
by the fold's rounding.

A WHOLE ANSWER IS READ AT ITS ROOT. A zero's sign is a value where an operation reads it (F7),
and no operation of the answer reads its root's: there two zeros of opposite sign are one value,
while a value that turns on a zero's sign an operation inside reads is a change
(`_contract.strict_at_the_root`). `0*sinh(x0) -> 0` is OK in f64; the same step under `1/` is a
change of infinity. Each rule step stays held to the rule's criterion, which reads every context.

Self-test: :func:`selftest` poisons a copy of the rule set with known-unsound rules and fails
loudly unless the monitor flags and attributes every one of them.

Importable API:
  * :func:`judge_pair`  -- judge a single ``inp -> out`` rewrite.
  * :func:`sweep`       -- run the engine over a corpus and collect violations.
  * :func:`build_engine`, :func:`make_corpus` -- engine/corpus construction.
  * :func:`selftest`    -- poison self-test (returns True iff every poison is caught).
  * :func:`monitor`     -- top-level driver (optional self-test, then the main sweep).
"""
import itertools
import json
import os
import re

import numpy as np
import yaml

from ..engine import _TAGGED_DIALECT_TOKENS, SimpliPyEngine
from ..utils import count_expressions, sample_expression
from . import _contract
from ._modes import RULE_MODES, judge_mode

VAR_RE = re.compile(r'^x(\d+)$')

def _mode_name(mode):
    """The judge's mode (`strict` or `permissive`) of a mode name or `simplipy.Mode` member
    (`_modes.judge_mode`)."""
    return judge_mode(mode)


def _as_rule(inp, out, engine):
    """The rewrite as the judge reads a rule: binary prefix (a tagged side is expanded by
    `engine.to_prefix`), each variable `x<k>` the slot `?<k>`."""
    sides = []
    for side in (list(inp), list(out)):
        if _TAGGED_DIALECT_TOKENS.intersection(side):
            if engine is None:
                raise ValueError('a tagged side needs an engine to read it')
            side = list(engine.to_prefix(side))
        sides.append([f'?{m.group(1)}' if (m := VAR_RE.match(t)) else t for t in side])
    return sides


def _reading(detail, mode):
    """The mode's reading of a judgment (`_contract.judged_reading`: permissive including f64,
    and a fully numeric rule read on each mode's own value): True (holds), False (refused on a
    counterexample) or None (no answer: neither passed nor convicted)."""
    return _contract.judged_reading(detail, mode)

#: The reals' verdicts that are no answer: under F8 they leave open whether the engine's own
#: algebra was exact.
_REALS_UNDECIDED = ('NO-WITNESS', 'UNRESOLVED-COVERAGE', 'UNSUPPORTED-SHAPE', 'JUDGE-TIMEOUT',
                    'UNDECIDED-MEASURE')


#: Float64's disagreements F8 lets the engine's exact algebra leave: rounding and what rounding
#: does near a zero or a pole (a sign, a zero against a nonzero), a value where float64 reads
#: the input as NaN (the rewrite holds in the reals: the reals' value, or an extension on a null
#: set, `x/x -> 1` at 0), an infinity's sign at an isolated pole, which strict does not promise
#: (`1/(-x0 - 3) -> -1/(x0 + 3)`, +inf against -inf at x0 = -3 alone), and any value where the input
#: has an isolated two-signed pole, which counts as a point without a value
#: (`1/(x0 - 3) + 1/(3 - x0) -> 0`, +inf against 0 at x0 = 3). Two zeros of opposite sign
#: (`ZERO-SIGN`, which the f64 lane names at every point where the two sides' zeros differ) are
#: one value where nothing reads them: at isolated points, and at the answer's root, which no
#: operation reads (F7); a zero's sign on a region under an operation that reads it is read
#: apart (`_contract.answer_zero_sign`): `atan2(0*x0, -1) -> pi` stays refused. Not an infinity
#: gained or lost elsewhere.
_F8_KINDS = frozenset({'ROUNDING', 'SIGN', 'EXT', 'POLE-SIGN', 'ZERO-SIGN', 'LIMIT'})


def _trace(inp, mode, engine):
    """The rule steps the engine takes on `inp` in `mode` -- each fire's rule instance, as the judge
    reads a rule (`_as_rule`) -- and its folds of numbers into one number (`ac_simplify_trace`:
    `(op, [operand], written)`, each operand and the written number as prefix tokens)."""
    rule_mode = RULE_MODES[judge_mode(mode)]
    _, fires, folds = engine._core.ac_simplify_trace(list(inp), rule_mode=rule_mode)
    steps = list(dict.fromkeys(tuple(map(tuple, _as_rule(lhs, rhs, engine)))
                               for lhs, rhs in fires))
    return steps, folds


def _steps(inp, mode, engine):
    """The rule steps the engine takes on `inp` in `mode` (`_trace`)."""
    return _trace(inp, mode, engine)[0]


#: The arithmetic a fold's operands are made of, read exactly (`_exact_value`).
_EXACT_OPS = {
    '+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b,
    '/': lambda a, b: a / b if b else None, 'neg': lambda a: -a,
    'inv': lambda a: 1 / a if a else None,
}


def _exact_value(node):
    """The exact rational value of a `_contract._term` node made of literals and `+`, `-`, `*`,
    `/`, `neg` and `inv` (every number of strict is a rational, and arithmetic between numbers is
    exact), or None where it holds anything else (`np.pi`, a function, a variable)."""
    from fractions import Fraction
    head, kids = node
    if not kids:
        try:
            v = _contract.literal_value(head)
        except (ValueError, ArithmeticError):
            return None
        return v if isinstance(v, Fraction) else None
    args = [_exact_value(k) for k in kids]
    f = _EXACT_OPS.get(head)
    if f is None or any(a is None for a in args) or len(args) != f.__code__.co_argcount:
        return None
    return f(*args)


def _spelled_exactly(v):
    """A rational as one literal token the judge reads exactly (`p/q`, or an integer)."""
    return str(v.numerator) if v.denominator == 1 else f'{v.numerator}/{v.denominator}'


def _fold_rules(fold):
    """THE GROUND RULES A FOLD MAKES, as strict reads it: `op(m1, m2, ...) -> written`, in each of
    the two readings the engine writes a fold by. The literals a product's coefficient or a sum's
    constant term combines (`*`, `+`) are combined as written (their exact value, or float64's
    value in the written order), or read at their doubles, as float64 reads a literal, and combined
    exactly, which the engine writes rounded once: then the left side is that exact value as one
    number, at whose double the written number agrees or not. A number written as another
    (`literal`) is its exact value as one number. A fully numeric node written as its value
    (`ground`) is the node. A fold holds where one of its rules holds."""
    from fractions import Fraction
    op, operands, written = fold
    written = list(written)
    trees = [_contract._term(o) for o in operands]
    if op in ('*', '+') and trees:
        lhs = list(operands[0])
        for o in operands[1:]:
            lhs = [op] + lhs + list(o)
        rules = [(lhs, written)]
        values = [_exact_value(t) for t in trees]
        if all(v is not None for v in values):
            doubles = [Fraction(float(v)) for v in values]
            total = doubles[0]
            for d in doubles[1:]:
                total = total * d if op == '*' else total + d
            rules.append(([_spelled_exactly(total)], written))
        return rules
    if op == 'literal' and trees:
        v = _exact_value(trees[0])
        return [([_spelled_exactly(v)] if v is not None else list(operands[0]), written)]
    return [(list(operands[0]) if operands else [], written)]


#: The most candidates (recorded fold outputs, the most recent first, then input literals) a number
#: held between folds is read as a sum or a product of (`_unfolded`): signed sums and products of up
#: to six of them are searched, three a side (with 64, at most 333,312 signed sums and 41,664
#: products of three in a table, built once per rewrite).
_MADE_OF_CANDIDATES = 64


def _unfolded(lhs, rhs, folds):
    """THE REWRITE WITH ITS FOLDS READ AS STEPS. The engine reads a literal of the input that a
    double does not hold at its double (`literal`, a fold of its own), and its exact algebra then
    acts on that double: such a literal of `lhs` (up to its sign) is read as the double. Each number
    of `rhs` that the input so read does not hold (up to its sign) is read as the numbers it was
    made of: where a fold wrote it, as the numbers it was folded from -- their operation (`*`, `+`)
    or the one number (`literal`, `ground`) as the engine held it before it wrote the number, its
    sign the written number's; where no fold wrote it, as the exact product or sum of two numbers
    the engine held (the input's, a fold's, a term's unit coefficient: the engine holds
    `x - 0.05523072428633124*(2.0417762114714635*x)` as `0.88723122100982965*x`, which is
    `1 - 0.11276877899017035`, the fold of the two literals), else as a signed sum or a product of
    three to six of the recorded fold outputs and input literals (an expansion's coefficient is
    the exact sum of the products the folds wrote; `_MADE_OF_CANDIDATES`), which is the number
    itself, so the reading changes no value -- and so on, through the numbers those were made of. Then the rewrite reads
    as the engine's own algebra, which is exact, and each fold is a step of its own
    (`_fold_rules`). -> `(lhs, rhs, [fold used])`, or None where no number was folded; the last
    fold that wrote a number is the one read."""
    from fractions import Fraction
    try:
        tl, tr = _contract._term(lhs), _contract._term(rhs)
    except (ValueError, IndexError):
        return None

    def value(t):
        try:
            v = _contract.literal_value(t)
        except (ValueError, ArithmeticError):
            return None
        return v if isinstance(v, Fraction) else None

    used = []
    # the input's literals the engine read at their doubles, by their magnitude
    at_double = {}
    for fold in folds:
        op, operands, written = fold
        if op == 'literal' and len(operands) == 1 and len(operands[0]) == 1 and len(written) == 1:
            v, d = value(operands[0][0]), value(written[0])
            if v is not None and d is not None and v != 0:
                at_double.setdefault(abs(v), (fold, written[0].lstrip('-')))

    def read_input(node):
        head, kids = node
        if kids:
            return (head, tuple(read_input(k) for k in kids))
        v = value(head)
        if v is None or abs(v) not in at_double:
            return node
        fold, spelled = at_double[abs(v)]
        if spelled == head.lstrip('-'):
            return node
        used.append(fold)
        return ('-' + spelled if v < 0 else spelled, ())

    tl = read_input(tl)
    held = {abs(v) for n in _contract._nodes(tl) if not n[1] and (v := value(n[0])) is not None}
    wrote = {}
    for fold in folds:
        written = fold[2]
        if len(written) == 1 and (v := value(written[0])) is not None and v != 0:
            wrote[abs(v)] = fold
    # every number the engine held, signed, by its spelling
    spelled = {}
    for node in _contract._nodes(tl):
        if not node[1] and (v := value(node[0])) is not None:
            spelled.setdefault(v, node[0])
    for _, operands, written in folds:
        for tokens in list(operands) + [written]:
            for t in tokens:
                if (v := value(t)) is not None:
                    spelled.setdefault(v, t)
    spelled.setdefault(Fraction(1), '1')  # a term's unit coefficient: `x - c*x` is `(1 - c)*x`
    members = {s * v for v in spelled if v != 0 for s in (1, -1)}
    pool = sorted(members, key=lambda v: (v < 0, abs(v)))  # the positive ones first, smallest first
    # the candidates of a longer sum or product: the fold outputs the trace records, the most
    # recent first, then the input's literals and the numbers folds were made of
    recent = [abs(v) for _, _, written in reversed(folds) for t in written
              if (v := value(t)) is not None]
    recent += [abs(v) for node in _contract._nodes(tl) if not node[1]
               and (v := value(node[0])) is not None]
    recent += [abs(v) for _, operands, _ in folds for tokens in operands for t in tokens
               if (v := value(t)) is not None]
    candidates = [v for v in dict.fromkeys(recent) if v not in (0, 1)][:_MADE_OF_CANDIDATES]
    tables, products = {}, {}

    def number(v):
        if v in spelled:
            return (spelled[v], ())
        return ('neg', ((spelled[-v], ()),))

    def table(op):
        """{value: ((candidate index, sign), ...)} of the signed sums (`+`) or the products (`*`)
        of one, two and three distinct candidates, by their count, built once."""
        if op not in tables:
            by_count = [None, {}, {}, {}]
            for k in (1, 2, 3):
                for chosen in itertools.combinations(range(len(candidates)), k):
                    for signs in (itertools.product((1, -1), repeat=k) if op == '+' else [(1,) * k]):
                        if op == '+':
                            v = sum((g * candidates[i] for g, i in zip(signs, chosen)), Fraction(0))
                        else:
                            v = Fraction(1)
                            for i in chosen:
                                v *= candidates[i]
                        by_count[k].setdefault(v, tuple(zip(chosen, signs)))
            tables[op] = by_count
        return tables[op]

    def made_of(v):
        """The exact product or sum of two numbers the engine held that is `v`, else their product
        of three; else a signed sum or a product of three to six candidates, found by meeting in the
        middle (three a side)."""
        for op, rest in (('*', lambda a: v / a), ('+', lambda a: v - a)):
            for a in pool:
                b = rest(a)
                if b in members and abs(a) != abs(v) and abs(b) != abs(v):
                    return (op, (number(a), number(b)))
        if not products:
            for i, a in enumerate(pool):
                for b in pool[i:]:
                    if abs(a) != 1 and abs(b) != 1:
                        products.setdefault(a * b, (a, b))
        for c in pool:
            ab = products.get(v / c)
            if ab is not None and abs(c) != abs(v) and all(abs(x) != abs(v) for x in ab):
                return ('*', (('*', (number(ab[0]), number(ab[1]))), number(c)))
        for op in ('+', '*'):
            target, by_count = (v if op == '+' else abs(v)), table(op)
            for total in range(3, 2 * 3 + 1):
                for a in range(1, total // 2 + 1):
                    if total - a > 3:
                        continue
                    for left_value, left in by_count[a].items():
                        if op == '*' and left_value == 0:
                            continue
                        right = by_count[total - a].get(
                            target - left_value if op == '+' else target / left_value)
                        if right is None or {i for i, _ in left} & {i for i, _ in right} or \
                                any(candidates[i] == abs(v) for i, _ in left + right):
                            continue
                        terms = [number(g * candidates[i]) for i, g in left + right]
                        expr = terms[0]
                        for t in terms[1:]:
                            expr = (op, (expr, t))
                        return ('neg', (expr,)) if op == '*' and v < 0 else expr
        return None

    reading = {}

    def read(node):
        """-> `(node read, whether a fold was read in it)`."""
        head, kids = node
        if kids:
            parts = [read(k) for k in kids]
            return (head, tuple(p for p, _ in parts)), any(f for _, f in parts)
        v = value(head)
        if v is None or v == 0 or abs(v) in held:
            return node, False
        if v in reading:
            return reading[v] or (node, False)
        reading[v] = None  # a number read in its own reading stays as written
        if abs(v) in wrote:
            fold = wrote[abs(v)]
            op, operands, _ = fold
            trees = [read(_contract._term(o))[0] for o in operands]
            if not trees:
                return node, False
            expr = trees[0]
            if op in ('*', '+'):
                for t in trees[1:]:
                    expr = (op, (expr, t))
            exact = _exact_value(expr)
            if exact is not None and exact != 0 and (exact < 0) != (v < 0):
                expr = ('neg', (expr,))
            used.append(fold)
            result = (expr, True)
        else:
            # an integer no fold wrote is exact as written; a number made of numbers held as
            # written, no fold among them, reads the same as written
            made = None if v.denominator == 1 else made_of(v)
            expr, folded = read(made) if made is not None else (node, False)
            result = (expr, True) if folded else (node, False)
        reading[v] = result
        return result

    out, _ = read(tr)
    if not used:
        return None
    return (_contract._tokens_of(tl), _contract._tokens_of(out),
            list(dict.fromkeys((op, tuple(map(tuple, ops)), tuple(w)) for op, ops, w in used)))


def _arguments(tokens):
    """The arguments of the prefix expression `tokens` (`_contract.ARITY`), each its tokens."""
    out, i = [], 1
    for _ in range(_contract.ARITY.get(tokens[0], 0)):
        j, need = i, 1
        while need:
            need += _contract.ARITY.get(tokens[j], 0) - 1
            j += 1
        out.append(tokens[i:j])
        i = j
    return out


def _literal_is(token, value):
    """Is `token` a literal of the value `value` (`1`, `1.0`, `(-1)`)?"""
    try:
        return _contract.literal_value(token) == value
    except (ValueError, ArithmeticError):
        return False


def _signs_spelled_alike(node):
    """The `_contract._term` tree `node` with float64's identical spellings of a reciprocal and a
    negation made one: `1/t` as `inv(t)`, `-1/t` and `-inv(t)` as `inv(-t)`, `-1*t` and `t*-1` as
    `-t`. Each is the value of the spelling it replaces at every point, in the reals and bit for
    bit in float64 (a quotient's sign is its operands'), so the parts two sides share are found
    where the engine spelled a reciprocal or moved a sign out of it: `1/sin(x1 - pi) -> -1/sin(x1)`
    shares `inv` and changes `sin(x1 - pi) -> -sin(x1)`."""
    head, kids = node
    kids = tuple(_signs_spelled_alike(k) for k in kids)
    if head == '/' and not kids[0][1] and _literal_is(kids[0][0], 1):
        return ('inv', (kids[1],))
    if head == '/' and not kids[0][1] and _literal_is(kids[0][0], -1):
        return ('inv', (_signs_spelled_alike(('neg', (kids[1],))),))
    if head == 'neg' and kids[0][0] == 'inv':
        return ('inv', (_signs_spelled_alike(('neg', kids[0][1])),))
    if head == '*':
        for i in (0, 1):
            if not kids[i][1] and _literal_is(kids[i][0], -1):
                return _signs_spelled_alike(('neg', (kids[1 - i],)))
    return (head, kids)


def _changed_parts(lhs, rhs, factors=True, depth=0):
    """THE PARTS OF THE INPUT THE ANSWER CHANGED, outermost first: below every operation the two
    share with all its other arguments unchanged, each pair of subexpressions down to the smallest
    one where they differ, read with a reciprocal's and a negation's spellings made one
    (`_signs_spelled_alike`). The engine's steps all lie inside the smallest, and what lies around
    a part is its context: `inv(log(exp(x0))) -> inv(x0)` changes `log(exp(x0)) -> x0` under
    `inv`, and `1/sin(np.pi - x1) -> inv(sin(x1))` changes `np.pi - x1 -> x1`, inside
    `sin(np.pi - x1) -> sin(x1)`, under `inv`. A sum whose terms (`_terms`) are the other side's
    but one changes that one (`_changed_term`), up to the order of the terms. Where the smallest is
    a product or a quotient on either side, its changed factors follow (`_changed_factors`), each
    with its own parts. Only the whole rewrite where it is no one prefix expression on each
    side."""
    try:
        node_l = _signs_spelled_alike(_contract._term(lhs))
        node_r = _signs_spelled_alike(_contract._term(rhs))
    except (ValueError, IndexError):
        return [(list(lhs), list(rhs))]
    parts = [(_contract._tokens_of(node_l), _contract._tokens_of(node_r))]
    for _ in range(64):
        while node_l[0] == node_r[0] and node_l[1] and len(node_l[1]) == len(node_r[1]):
            pairs = [(a, b) for a, b in zip(node_l[1], node_r[1]) if a != b]
            if len(pairs) != 1:
                break
            node_l, node_r = pairs[0]
            parts.append((_contract._tokens_of(node_l), _contract._tokens_of(node_r)))
        term = _changed_term(node_l, node_r)
        if term is None or term == (node_l, node_r):
            break
        node_l, node_r = term
        parts.append((_contract._tokens_of(node_l), _contract._tokens_of(node_r)))
    if factors and depth < 3:
        for part_l, part_r in _changed_factors(node_l, node_r):
            said = (_contract._tokens_of(part_l), _contract._tokens_of(part_r))
            parts.append(said)
            parts.extend(_changed_parts(*said, depth=depth + 1)[1:])
    unique = dict.fromkeys((tuple(a), tuple(b)) for a, b in parts)
    return [(list(a), list(b)) for a, b in unique]


def _terms(node):
    """A sum's terms, each `(sign, term)`: `a - (b + -c)` is `[(1, a), (-1, b), (1, c)]`."""
    head, kids = node
    if head == '+' and len(kids) == 2:
        return _terms(kids[0]) + _terms(kids[1])
    if head == '-' and len(kids) == 2:
        return _terms(kids[0]) + [(-s, t) for s, t in _terms(kids[1])]
    if head == 'neg' and len(kids) == 1:
        return [(-s, t) for s, t in _terms(kids[0])]
    return [(1, node)]


def _unshared(left, right):
    """The members of two lists less those they share, counted with multiplicity."""
    rest, kept = list(right), []
    for x in left:
        if x in rest:
            rest.remove(x)
        else:
            kept.append(x)
    return kept, rest


def _changed_term(node_l, node_r):
    """THE ONE TERM A SUM CHANGED: where the two sides' terms (`_terms`) are the same but one on
    each side, that pair, the right one negated where their signs differ: `a + f(x) - b ->
    -b + g(x) + a` changes `f(x) -> g(x)`. None where a side is no sum, or more terms changed."""
    additive = ('+', '-', 'neg')
    if node_l[0] not in additive and node_r[0] not in additive:
        return None
    kept, rest = _unshared(_terms(node_l), _terms(node_r))
    if len(kept) != 1 or len(rest) != 1:
        return None
    (sign_l, a), (sign_r, b) = kept[0], rest[0]
    return (a, b) if sign_l == sign_r else (a, ('neg', (b,)))


def _factors(node):
    """A product's or a quotient's factors, `(sign, [(factor, 1 or -1)])`: `-x/(y*inv(z))` is
    `(-1, [(x, 1), (y, -1), (z, 1)])`, a negation and a factor -1 its sign."""
    head, kids = node
    if head in ('*', '/') and len(kids) == 2:
        (s1, f1), (s2, f2) = _factors(kids[0]), _factors(kids[1])
        return s1 * s2, f1 + (f2 if head == '*' else [(f, -e) for f, e in f2])
    if head in ('inv', 'neg') and len(kids) == 1:
        s, f = _factors(kids[0])
        return (s, [(g, -e) for g, e in f]) if head == 'inv' else (-s, f)
    for unit in (1, -1):
        if not kids and _literal_is(head, unit):
            return unit, []
    return 1, [(node, 1)]


def _quotient(sign, factors):
    """The `_contract._term` tree of `_factors`' reading `(sign, factors)`: the product of the
    factors of exponent 1 over the product of those of exponent -1, negated where `sign` is -1."""
    def product(nodes):
        out = nodes[0] if nodes else ('1', ())
        for f in nodes[1:]:
            out = ('*', (out, f))
        return out
    num = product([f for f, e in factors if e > 0])
    den = [f for f, e in factors if e < 0]
    out = ('/', (num, product(den))) if den else num
    return ('neg', (out,)) if sign < 0 else out


def _factor_pairings(lhs, rhs, limit=64):
    """THE CHANGED FACTORS OR TERMS PAIRED ONE TO ONE: below every operation the two sides share
    (`_changed_parts`), where a product or a quotient changed as many factors on each side
    (`_factors`, less those the two share, at least two), each pairing of a left factor with a
    right factor of the same exponent, the right factors of any set of pairs negated whose size
    has the parity of the sign the two sides do not share: `A*B*C -> C2*(A2*B)` pairs `A -> A2`
    and `C -> C2`, `sin(x0)*sin(x1) -> -sin(x0 - pi)*-sin(x1 - pi)` pairs
    `sin(x0) -> -sin(x0 - pi)` and `sin(x1) -> -sin(x1 - pi)`, each a step of its own, and a
    factor that divides is read under its reciprocal. Where a sum changed as many terms on each
    side (`_terms`), each pairing of its terms, a pair's signs read on the pair. -> `[[(part_l,
    part_r)]]` as prefix tokens, at most `limit` pairings."""
    parts = _changed_parts(lhs, rhs, factors=False)
    try:
        node_l, node_r = (_signs_spelled_alike(_contract._term(t)) for t in parts[-1])
    except (ValueError, IndexError):
        return []
    out = []
    if node_l[0] in ('+', '-', 'neg') or node_r[0] in ('+', '-', 'neg'):
        # a sum's terms: each pair's signs read on the pair
        kept, rest = _unshared(_terms(node_l), _terms(node_r))
        if 2 <= len(kept) == len(rest) <= 5:
            for order in itertools.permutations(range(len(rest))):
                pairing = []
                for (sign_l, a), (sign_r, b) in zip(kept, (rest[j] for j in order)):
                    b = b if sign_l == sign_r else ('neg', (b,))
                    pairing.append((_contract._tokens_of(a), _contract._tokens_of(b)))
                out.append(pairing)
                if len(out) >= limit:
                    return out
        return out
    (sign_l, fl), (sign_r, fr) = _factors(node_l), _factors(node_r)
    kept_l, rest_r = _unshared(fl, fr)
    if len(kept_l) < 2 or len(kept_l) != len(rest_r) or len(kept_l) > 5:
        return []
    for order in itertools.permutations(range(len(rest_r))):
        if any(kept_l[i][1] != rest_r[j][1] for i, j in enumerate(order)):
            continue
        pairs = [(kept_l[i][0], rest_r[j][0]) for i, j in enumerate(order)]
        for negated in itertools.product((False, True), repeat=len(pairs)):
            if (sum(negated) % 2 == 1) != (sign_l != sign_r):
                continue
            out.append([(_contract._tokens_of(a), _contract._tokens_of(('neg', (b,)) if n else b))
                        for (a, b), n in zip(pairs, negated)])
        if len(out) >= limit:
            break
    return out[:limit]


def _changed_factors(node_l, node_r):
    """THE FACTORS A PRODUCT OR A QUOTIENT CHANGED: each side's factors (`_factors`) less those
    the two share, as one quotient each, and, where a factor divides, its reciprocal. A shared
    factor multiplies the same value on both sides, and a reciprocal reads the zero of what it
    divides by as a pole, so around the changed factors float64's rounding of their step is
    magnified, as around any part: `-1/(x1*S) -> inv(T)/x1` changes `S` to `-T`, read under the
    reciprocal, and `x/(a*x^2 + b*x) -> 1/(a*x + b)` changes `(a*x^2 + b*x)/x` to `a*x + b`. ->
    `[(part_l, part_r)]`, none where neither side is a product or a quotient."""
    heads = ('*', '/', 'inv', 'neg')
    if node_l[0] not in heads and node_r[0] not in heads:
        return []
    (sign_l, fl), (sign_r, fr) = _factors(node_l), _factors(node_r)
    rest_r, kept_l = list(fr), []
    for f in fl:
        if f in rest_r:
            rest_r.remove(f)
        else:
            kept_l.append(f)
    if not kept_l and not rest_r:
        return []
    out = []
    if len(kept_l) < len(fl):
        out.append((_quotient(sign_l, kept_l), _quotient(sign_r, rest_r)))
    if any(e < 0 for _, e in kept_l + rest_r):
        out.append((_quotient(sign_l, [(f, -e) for f, e in kept_l]),
                    _quotient(sign_r, [(f, -e) for f, e in rest_r])))
    return [(a, b) for a, b in out if (a, b) != (node_l, node_r)]


def _held(detail, name, engine, exact_steps, part=False):
    """The mode's reading of the judged rewrite `detail` (`judge_pair`, its steps' verdicts
    `exact_steps` read already): `(verdict, why, kinds)`, where `kinds` names float64's events
    where a change of kind beyond rounding's alone refused a rewrite true in the reals (F8), else
    None: a refusal that names no event, or beside an undecided measure of float64's extension,
    is read as it stands. A `part` of an answer is no root: its context may read its zero's
    sign, so a zero's sign on a region refuses it, as it refuses a rule."""
    from . import MODE_TIERS, effective_tier
    verdict, tier = detail.get('verdict'), effective_tier(detail)
    if tier in MODE_TIERS[name] or _reading(detail, name) is True:
        # a fully numeric rewrite holds where the mode's own reading of its value holds
        return 'OK', f'{verdict} ({tier})', None
    if name == 'strict' and part and detail.get('zero_sign_refused'):
        return 'VIOLATION', (f'F8: {verdict} ({tier}), a zero\'s sign on a region, which the '
                             f'context may read: {str(detail.get("zero_sign"))[:160]}'), None
    if name == 'strict' and not part and not detail.get('ground') \
            and _contract.strict_at_the_root(detail) is True:
        # no operation reads the root's zero's sign (F7)
        return 'OK', (f'{verdict} ({tier}): strict at the root, two zeros of opposite sign one '
                      f'value there'), None
    if _reading(detail, name) is None:
        return 'UNSCORED', (f'{name} reaches no answer ({verdict}): '
                            f'{str(detail.get("detail"))[:200]}'), None
    if engine is not None and name == 'strict' and verdict in _REALS_UNDECIDED:
        # F8 excuses float64's rounding where the engine's algebra is exact in the reals, and
        # the reals reach no answer on it
        return 'UNSCORED', (f'F8: the reals reach no answer ({verdict}): '
                            f'{str(detail.get("detail"))[:200]}'), None
    if engine is not None and verdict in _contract.HOLDS_IN_THE_REALS:
        # F8: the rewrite holds in the reals; float64's own events must be kept
        kinds = set(detail.get('f64_kinds') or ())
        if detail.get('realised') is None:
            return 'UNSCORED', f'F8: {verdict} ({tier}), float64 read at no point', None
        if not kinds or not kinds <= _F8_KINDS:
            alone = kinds and detail.get('f64_measure') is not None
            return 'VIOLATION', (f'F8: {verdict} ({tier}), float64 {sorted(kinds - _F8_KINDS)}: '
                                 f'{str(detail.get("unrealised"))[:160]}'), \
                sorted(kinds) if alone else None
        if name == 'strict' and _contract.answer_zero_sign(detail) is False:
            # rounding alone apart, but a value that turns on a zero's sign on a region (F7):
            # `atan2(0*x0, -1) -> pi` is -pi against pi for x0 < 0
            return 'VIOLATION', (f'F8: {verdict} ({tier}), a value that turns on a zero\'s sign: '
                                 f'{str(detail.get("zero_sign_answer") or detail.get("zero_sign"))[:160]}'), \
                None
        if exact_steps:
            return 'OK', f'F8: {verdict} ({tier}), float64 rounding alone', None
        return 'UNSCORED', f'F8: {verdict} ({tier}) through a step that holds in {name} alone', None
    return 'VIOLATION', f'{verdict} ({tier}): {str(detail.get("detail"))[:200]}', None


def judge_pair(inp, out, rng=None, *, mode='strict', engine=None):
    """Judge one rewrite ``inp -> out`` made in ``mode`` (``'strict'``, ``'permissive'`` or a
    :class:`~simplipy.Mode` member, the pre-0.15 names ``'f64'`` and ``'default'`` read as
    ``'strict'`` with a ``DeprecationWarning`` until 0.16; the default is the engine's default
    mode) as the rule judge judges the rule ``inp -> out``.

    Returns ``(verdict, detail)``, by the mode's own reading of the judgment: ``OK`` where the
    judged tier is one the mode licenses (``verify.MODE_TIERS``) or, for a fully numeric rewrite,
    where the mode's own value of it holds, ``VIOLATION`` where the mode
    refuses it, and ``UNSCORED`` where the judge cannot read the rewrite or the mode reaches no
    answer on it (fail closed: the sweep counts it). The reals' verdict, a diagnostic, is no
    mode's label: `(x - x)/(x - x) -> 1` is a VIOLATION in f64, which refuses it on its own
    values, and OK in permissive, though the reals' measure is undecided. ``engine`` reads a
    tagged side (`to_prefix`). ``rng`` is accepted for the callers that pass one; the judgment is
    deterministic and draws nothing. With ``engine``, the rewrite is held to F8 (above): its rule
    steps are judged too, a rewrite true in the reals whose float64 disagreements are rounding
    alone is OK, a change of kind in the steps' context is read only for a zero's sign, and one
    f64 refuses whose exactness in the reals the judge cannot decide is UNSCORED."""
    from . import MODE_TIERS, effective_tier
    name = _mode_name(mode)
    try:
        lhs, rhs = _as_rule(inp, out, engine)
        with _contract.limits_off():
            # the whole answer is no step: the input's limit is a step's
            detail = _contract.judge_rule(lhs, rhs)
    except Exception as error:       # an unreadable rewrite is reported, never passed
        return 'UNSCORED', f'{type(error).__name__}: {error}'
    exact_steps = True
    folds = []
    if engine is not None:
        try:
            steps, folds = _trace(inp, name, engine)
        except Exception as error:
            return 'UNSCORED', f'steps: {type(error).__name__}: {error}'
        for step_lhs, step_rhs in steps:
            step = _contract.judge_rule(list(step_lhs), list(step_rhs))
            step_tier = effective_tier(step)
            said = f'{" ".join(step_lhs)} -> {" ".join(step_rhs)}'
            if step_tier not in MODE_TIERS[name] and _reading(step, name) is not True:
                if _reading(step, name) is None:
                    return 'UNSCORED', f'F8: the step {said}: {name} reaches no answer ' \
                                       f'({step.get("verdict")})'
                return 'VIOLATION', f'F8: the step {said} is {step_tier}, not {name}\'s'
            exact_steps = exact_steps and step.get('verdict') in _contract.HOLDS_IN_THE_REALS
    verdict, why = _whole_answer(lhs, rhs, detail, name, engine, exact_steps)
    if verdict == 'OK' or name != 'strict' or not folds:
        return verdict, why
    return _with_folds(lhs, rhs, folds, name, engine, exact_steps, verdict, why)


def _with_folds(lhs, rhs, folds, name, engine, exact_steps, verdict, why):
    """THE ANSWER READ WITH ITS FOLDS AS STEPS (`_unfolded`): the engine writes a number its own
    algebra combines from literals as a double (a product's coefficient, a sum's constant term, a
    long number at its double), a step of its own, which strict admits per step (`_fold_rules`, each
    judged as the ground rule it makes). Where the answer as written is refused, each number of it
    the engine folded is read as the numbers it was folded from; where every such fold holds in the
    mode and the answer so read holds as an answer (`_whole_answer`), the answer is OK. Otherwise
    the verdict stands, with what the reading found."""
    from . import MODE_TIERS, effective_tier
    unfolded = _unfolded(lhs, rhs, folds)
    if unfolded is None:
        return verdict, why
    read_lhs, read, used = unfolded
    for fold in used:
        said = None
        for fold_lhs, fold_rhs in _fold_rules(fold):
            said = said or f'{" ".join(fold_lhs)} -> {" ".join(fold_rhs)} ({fold[0]})'
            try:
                d = _contract.judge_rule(fold_lhs, fold_rhs)
            except Exception:  # noqa: BLE001  (a rule the judge cannot read holds nothing)
                continue
            if effective_tier(d) in MODE_TIERS[name] or _reading(d, name) is True:
                break
        else:
            return verdict, f'{why}; the fold {said} is not {name}' + "'s"
    try:
        with _contract.limits_off():
            detail = _contract.judge_rule(read_lhs, read)
    except Exception as error:
        return verdict, f'{why}; with its folds read as steps: {type(error).__name__}: {error}'
    read_verdict, read_why = _whole_answer(read_lhs, read, detail, name, engine, exact_steps)
    if read_verdict == 'OK':
        return 'OK', f'F8: {len(used)} folds read as steps, each {name}' + "'s" + f'; so read, {read_why}'
    if read_verdict == 'UNSCORED':
        # the answer as written is refused by its folds' rounding alone, and so read it reaches no
        # answer
        return 'UNSCORED', f'F8: {len(used)} folds read as steps, each {name}' + "'s" + \
            f'; so read, {read_why[:200]}'
    return verdict, f'{why}; with its folds read as steps: {read_verdict}: {read_why[:200]}'


def _whole_answer(lhs, rhs, detail, name, engine, exact_steps, depth=0, part=False):
    """The mode's reading of a whole answer `lhs -> rhs` judged as `detail` (`judge_pair`, its
    steps' verdicts read already): `_held`, and where float64's kinds alone refuse an answer true
    in the reals, the smallest changed part that holds as a step, around which only a zero's sign
    is read. -> `(verdict, why)`."""
    verdict, why, kinds = _held(detail, name, engine, exact_steps, part=part)
    if kinds is None:
        return verdict, why
    # float64's kinds alone refuse a rewrite true in the reals: each step is held at its own
    # points, and in its context only to a zero's sign. The smallest part of the input the answer
    # changed that holds as a step is read as one (the engine's steps lie inside the smallest
    # changed part, and so inside every part around it: `sin(np.pi - x1) -> sin(x1)` around
    # `np.pi - x1 -> x1`); around it, a change of kind is float64's rounding of the step's input
    # magnified by an operation of the context, unless a value turns on a zero's sign there
    held = {}

    def part_held(part_lhs, part_rhs, in_context=False):
        key = (tuple(part_lhs), tuple(part_rhs), in_context)
        if key not in held:
            try:
                part = _contract.judge_rule(list(part_lhs), list(part_rhs))
            except Exception as error:
                held[key] = 'UNSCORED', f'{type(error).__name__}: {error}'
            else:
                held[key] = _whole_answer(list(part_lhs), list(part_rhs), part, name, engine,
                                          exact_steps, depth + 1, True) if in_context \
                    else _held(part, name, engine, exact_steps, part=True)[:2]
        return held[key]

    first_why = None
    for part_lhs, part_rhs in reversed(_changed_parts(lhs, rhs)[1:]):
        said = f'the changed part {" ".join(part_lhs)} -> {" ".join(part_rhs)}'
        part_verdict, part_why = part_held(part_lhs, part_rhs)
        if part_verdict == 'OK':
            break
        first_why = first_why or f'{why}; {said}: {part_why}'
    else:
        # several factors or terms changed: each, paired one to one, holds as a step of its own,
        # read in its own context
        for pairing in (_factor_pairings(lhs, rhs) if depth < 2 else []):
            readings = []
            for a, b in pairing:
                readings.append(part_held(a, b, in_context=True))
                if readings[-1][0] != 'OK':
                    break
            if all(v == 'OK' for v, _ in readings):
                said = 'the changed factors ' + '; '.join(
                    f'{" ".join(a)} -> {" ".join(b)}' for a, b in pairing)
                part_why = ', '.join(w for _, w in readings)
                break
        else:
            return verdict, first_why or why
    if name == 'strict' and _contract.answer_zero_sign(detail) is False:
        return 'VIOLATION', (f'F8: a zero\'s sign under an operation that reads it, around '
                             f'{said}: '
                             f'{str(detail.get("zero_sign_answer") or detail.get("zero_sign"))[:160]}')
    if name == 'strict' and _contract.answer_keeps_poles(detail) is False:
        # around a held step a change of kind is float64's rounding of its input, but where the
        # left side has an infinity at a pole of the reals the answer keeps it
        return 'VIOLATION', (f'F8: an infinity the left side has at a pole of the reals, lost '
                             f'around {said}: {str(detail.get("pole_lost"))[:160]}')
    return 'OK', f'F8: {said} holds ({part_why}); around it float64 {kinds} from rounding'


# ---------------------------------------------------------------------------------------
# Corpus, engines, the sweep
# ---------------------------------------------------------------------------------------

ADVERSARIAL = [
    ['+', 'x0', 'pow', 'sin', 'x0', 'float("inf")'],
    ['pow', 'sin', 'x0', 'float("inf")'],
    # trigger for the point poison
    ['pow', 'sin', 'x0', 'float("-inf")'],
    ['*', '4', 'pow', 'x0', 'float("inf")'],
    ['-', 'exp', 'x0', 'exp', 'x0'],
    ['-', 'log', 'x0', 'log', 'x0'],
    ['inv', '*', '0', 'x0'],
    ['/', 'log', 'x0', 'log', 'x0'],
    ['atanh', 'tanh', 'sinh', 'x0'],
    ['*', 'x0', 'pow', 'cos', 'x1', 'float("inf")'],
    ['+', 'pow', 'tanh', 'x0', 'float("inf")', 'x1'],
    ['pow', '/', '7.0', '5', 'float("nan")'],
    ['*', 'sin', 'np.pi', 'x0'],
    ['abs', 'pow', 'x0', '*', '3', '1'],
    ['tan', '+', 'x0', 'np.pi'],
    ['inv', '+', '0', 'neg', 'pow', 'x0', '4'],
    ['pow', 'float("-inf")', 'rootn', '3.0', '3'],
    # triggers for the measure-clause poison (even-root round-trip via rootn)
    ['pow', 'rootn', 'x0', '2', '2'],
    ['*', '2', 'pow', 'rootn', 'x1', '2', '2'],
    # trigger for the product-over-a-sum poison
    ['*', 'x0', '+', 'x1', 'x2'],
    # PRE-MASKED shapes: deployed traffic arrives with literals already masked to
    # <constant> (masking is explicit, upstream of simplify), and they are the masked-scale
    # poisons' triggers on any rule set
    ['*', '<constant>', 'x0'],
    ['+', '*', '<constant>', 'x1', 'x0'],
    ['sin', '*', '<constant>', 'x0'],
    ['*', '<constant>', 'pow', 'x0', 'float("inf")'],
]

# Default leaf vocabulary used to sample the random half of the corpus.
DEFAULT_LEAVES = ['x0', 'x1', '0', '1', 'np.pi', '2.0', '(-0.5)', '3.0']

# Known-unsound rules for the poison self-test. Each must be DEPLOYABLE end to end (the load
# accepts it) and a trigger shape must exist in ADVERSARIAL. Catch requires ATTRIBUTION: a
# flagged input whose output CHANGED versus the unpoisoned engine.
POISON = [
    # a change of kind at a point f64 holds: float64's sin(fl(pi/2)) is exactly 1.0, where
    # pow(1, -inf) is 1 against inf (in the reals sin(fl(pi/2)) is below 1, and permissive
    # holds the rewrite)
    (('pow', 'sin', '?0', 'float("-inf")'), ('float("inf")',)),
    # masked-scale: drops the coefficient of pre-masked traffic
    (('*', '<constant>', '?0'), ('?0',)),
    # structural coefficient drop (fires on the pre-masked pow shape)
    (('*', '<constant>', 'pow', '?0', 'float("inf")'), ('pow', '?0', 'float("inf")')),
    # measure-clause poison: the even-root round-trip pow(rootn(x,2),2) -> x extends the
    # x < 0 half-line
    (('pow', 'rootn', '?0', '2', '2'), ('?0',)),
    # a dropped factor: x*(y+z) -> y+z
    (('*', '?0', '+', '?1', '?2'), ('+', '?1', '?2')),
]


def build_engine(rules, config_path):
    """Build a deployed engine from `config_path`, overriding its ruleset with `rules`.

    `config_path` is the deployed engine config the monitor realizes against; the driver
    passes the path in. `rules` is a list of (lhs, rhs) token-sequence pairs (empty for the
    rules-empty baseline engine), and it is the engine's whole rule set: no mode names a set
    of its own. Built in memory from the config's operator table, so nothing is written to
    disk."""
    base = yaml.safe_load(open(config_path))
    return SimpliPyEngine(operators=base['operators'], rules=[(tuple(l), tuple(r)) for l, r in rules])


def make_corpus(eng, n, rng, adversarial=None, leaves=None):
    if adversarial is None:
        adversarial = ADVERSARIAL
    if leaves is None:
        leaves = DEFAULT_LEAVES
    non_leaf = dict(sorted(eng.operator_arity.items(), key=lambda x: x[1]))
    counts = count_expressions(len(leaves), non_leaf, 10)
    corpus = [list(e) for e in adversarial]
    for _ in range(n):
        L = int(rng.integers(3, 10))
        corpus.append(list(sample_expression(L, leaves, non_leaf, counts, rng)))
    return corpus


class _JudgeTimeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _JudgeTimeout()


JUDGE_TIMEOUT_S = 10   # a single expression may not eat the sweep; a timeout is recorded as
                       # unresolved (with the offending expression printed) and skipped.


def sweep(rules, corpus, rng, baseline_engine, config_path, tag='',
          judge_timeout_s=JUDGE_TIMEOUT_S):
    # Timeout protection is SIGALRM-based and therefore main-thread-only. Install for
    # the duration of the sweep and RESTORE on the way out, the caller's pending timer
    # included (`_gate.caller_timer_kept`): clobbering the process handler permanently would
    # crash any off-main-thread caller.
    import threading
    from ._gate import caller_timer_kept
    if threading.current_thread() is not threading.main_thread():
        print(f'  [{tag}] off-main-thread sweep: judge timeout protection unavailable '
              f'(SIGALRM is main-thread-only)', flush=True)
        return _sweep_inner(rules, corpus, rng, baseline_engine, config_path, tag, 0)
    with caller_timer_kept(_alarm):
        return _sweep_inner(rules, corpus, rng, baseline_engine, config_path, tag,
                            judge_timeout_s)


def _sweep_inner(rules, corpus, rng, baseline_engine, config_path, tag,
                 judge_timeout_s):
    import signal
    eng = build_engine(rules, config_path)
    viol, native, unscored, changed, timeouts = [], [], 0, 0, 0

    def judged(expr, out, engine):
        """`expr -> out` judged with the rule steps `engine` takes on `expr`."""
        signal.alarm(judge_timeout_s)
        try:
            return judge_pair(list(expr), out, rng, engine=engine)
        finally:
            signal.alarm(0)

    for i, expr in enumerate(corpus):
        if tag and (i + 1) % 500 == 0:
            print(f'  [{tag} {i+1}/{len(corpus)}] changed={changed} viol={len(viol)} '
                  f'timeouts={timeouts}', flush=True)
        try:
            out = list(eng.simplify(list(expr)))
        except Exception as ex:
            viol.append((expr, ['<RAISED>'], f'simplify raised {type(ex).__name__}'))
            continue
        if out == list(expr):
            continue
        changed += 1
        try:
            v, detail = judged(expr, out, eng)
        except _JudgeTimeout:
            timeouts += 1
            print(f'  [{tag}] TIMEOUT-unresolved: {" ".join(expr)}  ->  {" ".join(out)}', flush=True)
            continue
        if v == 'UNSCORED':
            unscored += 1
        elif v == 'VIOLATION':
            try:
                base_out = list(baseline_engine.simplify(list(expr)))
            except Exception:
                base_out = None
            if base_out == out:
                native.append((expr, out, detail))
            elif base_out is not None and base_out != list(expr):
                # CAUSE attribution: if the rules-empty engine's own rewrite of this
                # expression ALREADY violates, the violation is native and the rule set
                # merely rewrote further on top. The baseline's answer is judged with the
                # baseline's own steps: the loaded engine's steps are the rule set's.
                try:
                    vb, _ = judged(expr, base_out, baseline_engine)
                except _JudgeTimeout:
                    vb = None
                if vb == 'VIOLATION':
                    native.append((expr, out, detail))
                else:
                    viol.append((expr, out, detail))
            else:
                viol.append((expr, out, detail))
    return viol, native, unscored, changed


def selftest(rules, config_path, corpus, baseline, seed, poison=None,
             judge_timeout_s=JUDGE_TIMEOUT_S):
    """Poison a copy of the ruleset with known-unsound rules; return True iff EVERY poison is
    caught. A catch requires ATTRIBUTION: a flagged input whose output differs from the
    unpoisoned (clean) engine's output. Prints a per-poison CAUGHT / *** MISSED *** line."""
    if poison is None:
        poison = POISON
    clean_eng = build_engine(rules, config_path)
    caught = 0
    for p in poison:
        v, *_ = sweep(rules + [p], corpus[:600], np.random.default_rng(seed),
                      baseline, config_path, tag="poison", judge_timeout_s=judge_timeout_s)
        attributed = []
        for expr, out, detail in v:
            try:
                clean_out = list(clean_eng.simplify(list(expr)))
            except Exception:
                clean_out = ['<RAISED>']
            # attribution = the poisoned engine BEHAVES DIFFERENTLY from the clean one on
            # this input; both raising is the same behaviour.
            if clean_out != list(out):
                attributed.append((expr, out, detail))
        hit = len(attributed) > 0
        caught += hit
        print(f'  poison {" ".join(p[0])} -> {" ".join(p[1])}: '
              f'{"CAUGHT" if hit else "*** MISSED ***"} '
              f'({len(attributed)} attributed / {len(v)} flagged)')
    return caught == len(poison)


class MonitorSelfTestError(RuntimeError):
    """Raised by :func:`monitor` when the poison self-test fails to flag a known-unsound rule.

    A self-test failure means the monitor cannot be trusted to catch an unsound ruleset, so
    the main sweep is not run: the gate has lost its own guarantee."""


class MonitorLivenessError(RuntimeError):
    """Raised by :func:`monitor` when the main sweep rewrites NOTHING on a non-trivial corpus.

    A sweep that judges only rewrites is vacuous when no rewrites happen: `changed == 0`
    over hundreds of sampled expressions means the engine under test is not applying
    anything (broken asset load, dead core plumbing) -- an instrument failure that must
    not be reported as 'clean' (0 violations of 0 rewrites)."""


def monitor(rules, config_path, corpus_n=6000, seed=20260718, run_selftest=False,
            adversarial=None, leaves=None, poison=None, judge_timeout_s=JUDGE_TIMEOUT_S,
            label=''):
    """Run the monitor against a deployed engine built from `config_path`.

    `rules` is either a list of (lhs, rhs) token-sequence pairs or a path to a JSON file of
    ``[[lhs, rhs], ...]`` (loaded as tuples). When `run_selftest` is True the poison self-test
    runs first and a failure raises :class:`MonitorSelfTestError` before the main sweep.

    Returns a dict with keys: rules, corpus, changed, violations, native, unscored,
    selftest_passed."""
    if adversarial is None:
        adversarial = ADVERSARIAL
    if leaves is None:
        leaves = DEFAULT_LEAVES
    if poison is None:
        poison = POISON
    rules = _load_rules(rules)
    rng = np.random.default_rng(seed)
    baseline = build_engine([], config_path)
    corpus = make_corpus(baseline, corpus_n, rng, adversarial=adversarial, leaves=leaves)
    print(f'monitor: {len(rules):,} rules | corpus {len(corpus):,} '
          f'({len(adversarial)} adversarial + {corpus_n:,} sampled)', flush=True)

    selftest_passed = None
    if run_selftest:
        selftest_passed = selftest(rules, config_path, corpus, baseline, seed,
                                   poison=poison, judge_timeout_s=judge_timeout_s)
        if not selftest_passed:
            print('SELF-TEST FAILED: the monitor cannot catch a poisoned artifact')
            raise MonitorSelfTestError(
                'poison self-test failed: a known-unsound rule was not flagged')
        print('SELF-TEST PASSED (all poisons caught, attribution-verified)\n', flush=True)

    viol, native, unscored, changed = sweep(
        rules, corpus, rng, baseline, config_path,
        tag="main", judge_timeout_s=judge_timeout_s)
    if changed == 0 and len(corpus) >= 100:
        raise MonitorLivenessError(
            f'main sweep rewrote 0 / {len(corpus)} expressions: the engine under test '
            f'applied nothing (broken asset load or dead core plumbing) -- a sweep that '
            f'judges only rewrites is vacuous here and must not report clean')
    print(f'\n=== MONITOR  {label} ===')
    print(f'  rewritten     {changed:7,} / {len(corpus):,}')
    print(f'  VIOLATIONS    {len(viol):7,}  (artifact-attributed)')
    print(f'  native-line   {len(native):7,}  (present with rules=[]; pre-existing engine behavior)')
    print(f'  unscored      {unscored:7,}  (rewrites the judge cannot read or decide)')
    for expr, out, detail in viol[:15]:
        print(f'    VIOL {" ".join(expr)}  ->  {" ".join(out)}   [{detail}]')
    for expr, out, detail in native[:5]:
        print(f'    (native) {" ".join(expr)}  ->  {" ".join(out)}   [{detail}]')
    return {
        'rules': len(rules),
        'corpus': len(corpus),
        'changed': changed,
        'violations': viol,
        'native': native,
        'unscored': unscored,
        'selftest_passed': selftest_passed,
    }


def _load_rules(rules):
    """Load rules from a JSON path (list of [lhs, rhs]) into (tuple, tuple) pairs, or pass a
    pre-loaded rule list through unchanged."""
    if isinstance(rules, (str, os.PathLike)):
        with open(rules) as fh:
            return [(tuple(l), tuple(r)) for l, r in json.load(fh)]
    return rules
