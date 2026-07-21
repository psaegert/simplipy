# mypy: ignore-errors
"""Redundancy refund: apply the ?->_ promotion, then prune rules the promoted set derives.

Mining at the ? (variable-leaf) quantifier makes the miner spell out composite-headed
instances that broader subtree rules would otherwise (unsoundly) subsume: `* (-1) abs ?0 ->
neg abs ?0` sits alongside the base rule `* (-1) ?0 -> neg ?0`. Once the base rule passes the
pointwise bar and is promoted to the `_` (arbitrary subtree) sort, every such instance is
derivable at deploy time and can be pruned.

Two prunes, both conservative (any mismatch keeps the rule):
  1. SYNTACTIC SUBSUMPTION (exact, sort-aware): rule B is redundant iff B = sigma(A) for
     another kept rule A, where sigma maps A's `_i` to any subpattern of B and A's `?i` only
     to B's `?j` leaf slots (a leaf slot may not receive a composite).
  2. SEQUENTIAL derivability probe (?-rules only): bind each `?i` to a fresh variable leaf,
     simplify the lhs under the engine WITHOUT this rule; redundant iff the result equals the
     rhs. Sequential so mutually-derivable siblings cannot both vanish.
"""
import re

from ..engine import SimpliPyEngine

Q = re.compile(r'^\?(\d+)$')


def refund(rules, operators, promote_set):
    """Apply ?->_ for rules in ``promote_set``, then prune the derivable instances.

    ``rules``: iterable of (lhs, rhs) token sequences (all ?-sorted from the mine).
    ``operators``: the engine operator config dict (``{name: {arity, realization, ...}}``),
    supplying operator arities for tree parsing and the rule-free probe engine.
    ``promote_set``: set of (lhs, rhs) tuples (the pointwise + const-bearing PROMOTE verdicts)
    to upgrade to the `_` subtree sort. Returns the kept, sorted-and-pruned rules.
    """
    OPS = operators
    rules = [(tuple(l), tuple(r)) for l, r in rules]
    up = lambda ts: tuple('_' + Q.match(t).group(1) if Q.match(t) else t for t in ts)
    sorted_rules = []
    for lhs, rhs in rules:
        if (lhs, rhs) in promote_set:
            sorted_rules.append((up(lhs), up(rhs)))
        else:
            sorted_rules.append((lhs, rhs))

    # PRUNE 1 -- SYNTACTIC SUBSUMPTION (exact, order-safe, sort-aware): rule B is redundant iff
    # B = sigma(A) for another kept rule A, where sigma maps A's `_i` to any subpattern of B and
    # A's `?i` only to B's `?j` slots (a leaf slot may not receive a composite). This refunds the
    # composite-headed instances the ?-native mine spells out once their base rule is promoted
    # (`* (-1) abs _0` = sigma(`* (-1) _0`), sigma: _0 -> abs _0).
    def parse(ts, ops_ar):
        def sub(i):
            t = ts[i]
            ar = ops_ar(t)
            if ar is None:
                return (t, ()), i + 1
            kids, j = [], i + 1
            for _ in range(ar):
                k, j = sub(j)
                kids.append(k)
            return (t, tuple(kids)), j
        tree, j = sub(0)
        assert j == len(ts)
        return tree
    ar_map = {}
    for name, props in OPS.items():
        ar_map[name] = 2 if props.get('arity', 1) == 2 else 1

    def ops_ar(t):
        a = ar_map.get(t)
        return a

    def subsumes(a, b, sig):
        ta, kids_a = a
        if ta.startswith('_') and ta[1:].isdigit():
            if ta in sig:
                return sig[ta] == b
            sig[ta] = b
            return True
        if ta.startswith('?') and ta[1:].isdigit():
            tb, kids_b = b
            if not (tb.startswith('?') and tb[1:].isdigit() and not kids_b):
                return False       # a leaf slot may not receive a composite or a subtree slot
            if ta in sig:
                return sig[ta] == b
            sig[ta] = b
            return True
        tb, kids_b = b
        if ta != tb or len(kids_a) != len(kids_b):
            return False
        return all(subsumes(x, y, sig) for x, y in zip(kids_a, kids_b))

    def sub_apply(a, sig):
        t, kids = a
        if (t.startswith('_') or t.startswith('?')) and t[1:].isdigit():
            return sig[t]
        return (t, tuple(sub_apply(k, sig) for k in kids))
    trees = [(parse(list(l), ops_ar), parse(list(r), ops_ar)) for l, r in sorted_rules]
    wc_rules = [i for i, (l, r) in enumerate(sorted_rules)
                if any((t.startswith('_') or t.startswith('?')) and t[1:].isdigit() for t in l)]
    by_root = {}
    for i in wc_rules:
        by_root.setdefault(trees[i][0][0], []).append(i)
    subsumed = set()
    for j in wc_rules:                       # is rule j an instance of some other rule i?
        lj, rj = trees[j]
        for i in by_root.get(lj[0], []):
            if i == j or i in subsumed:
                continue
            li, ri = trees[i]
            if len(sorted_rules[i][0]) > len(sorted_rules[j][0]):
                continue
            sig = {}
            if subsumes(li, lj, sig) and sub_apply(ri, sig) == rj:
                subsumed.add(j)
                break
    remaining = [rl for i, rl in enumerate(sorted_rules) if i not in subsumed]

    # PRUNE 2 -- SEQUENTIAL derivability probe, ?-rules only (a ?-rule claims ONLY variable
    # bindings, so a variable probe covers its whole claim; sequential so mutually-derivable
    # siblings cannot both vanish).
    keep, pruned = [], []
    current = list(remaining)
    fresh = lambda ts: [f'x{t[1:]}' if Q.match(t) else t for t in ts]
    for i in range(len(current)):
        lhs, rhs = current[i]
        if not any(Q.match(t) for t in lhs):
            keep.append((lhs, rhs))
            continue
        rest = [[list(l), list(r)] for j, (l, r) in enumerate(current) if j != i and (l, r) not in pruned]
        e2 = SimpliPyEngine(operators=OPS, rules=rest)
        try:
            got = e2.simplify(fresh(lhs), max_pattern_length=None, mask_elementary_literals=False)
        except Exception:
            keep.append((lhs, rhs))
            continue
        if list(got) == list(fresh(rhs)):
            pruned.append((lhs, rhs))
        else:
            keep.append((lhs, rhs))
    return keep
