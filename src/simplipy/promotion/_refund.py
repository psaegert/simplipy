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

Prune 2 is BATCHED (0.14.1): the reference loop built a full engine per candidate, and rule
translation alone costs seconds per build, so a real corpus (~13k ?-probes over ~163k rules)
ran for days. One probe engine now serves the whole stage, and "the engine WITHOUT this
rule" is produced per probe by SUPPRESSING artifact rows at the matcher's fire site
(``ac_simplify_suppressed``) -- provably the same walk an engine built without those rows
performs (per-rule translation, in-place bucket filtering), EXCEPT where translate's
cross-rule state binds. Those exceptions are enumerated by the core's SHADOW CENSUS
(``ac_shadow_census``): removing an OWNER row resurrects a victim rule's orientation twin,
which carries the owner's identical canonical rewrite. An owner in the removal set whose
victim survives it is therefore left UNSUPPRESSED as a stand-in: the live engine serves the
very rewrite the rebuild would serve through the resurrected twin, at a different bucket
priority only (decision-equivalence validated against per-candidate rebuilds). Every census
shape outside that argument -- dedup-entangled rules, non-``twin-shadow`` kinds, twin
owners -- falls back to the reference per-candidate rebuild, so each probe is either exact
by construction or runs the old construction itself. The keep/prune decision SEQUENCE and
the rule order are those of the reference loop, byte for byte.
"""
import re

from ..engine import DEFAULT_EFFORT, SimpliPyEngine, _TAGGED_DIALECT_TOKENS
from ..progress import Progress
from ..utils import _dedup_keyed_rules

Q = re.compile(r'^\?(\d+)$')


def refund(rules, operators, promote_set, progress=None):
    """Apply ?->_ for rules in ``promote_set``, then prune the derivable instances.

    ``rules``: iterable of (lhs, rhs) token sequences (all ?-sorted from the mine).
    ``operators``: the engine operator config dict (``{name: {arity, realization, ...}}``),
    supplying operator arities for tree parsing and the rule-free probe engine.
    ``promote_set``: set of (lhs, rhs) tuples (the pointwise + const-bearing PROMOTE verdicts)
    to upgrade to the `_` subtree sort. Returns the kept, sorted-and-pruned rules.
    ``progress``: an optional :class:`~simplipy.progress.Progress` reporter; the
    derivability probe reports done/total every 500 probes through it.
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
    # siblings cannot both vanish). Batched: ONE probe engine + per-probe row suppression;
    # see the module docstring for the equivalence argument and the fallback tiers.
    prog = progress if progress is not None else Progress(False)
    keep, pruned = [], []
    pruned_set = set()          # the same membership `(l, r) not in pruned` tests, O(1)
    current = list(remaining)
    fresh = lambda ts: [f'x{t[1:]}' if Q.match(t) else t for t in ts]
    n_probes = sum(1 for l, _ in current if any(Q.match(t) for t in l))

    def build_probe_state():
        # The engine over every rule the reference loop would still feed a candidate's
        # build (all of `current` minus the pruned VALUES), plus the maps that let a
        # probe stand for "the engine without one more rule": candidate index -> live
        # position -> artifact row (through the dedup fold the constructor itself runs),
        # and the shadow census in that row space. Rebuilt only when a prune lands
        # outside the plain suppress-this-row case.
        live_pos, live = {}, []
        for j, pair in enumerate(current):
            if pair not in pruned_set:
                live_pos[j] = len(live)
                live.append(pair)
        prog.stage('stage 4/5 probe engine build', live=len(live), candidates=n_probes)
        engine = SimpliPyEngine(operators=OPS, rules=[[list(l), list(r)] for l, r in live])
        keyed = _dedup_keyed_rules(live, [f'x{k}' for k in range(100)], engine)
        buckets = {}
        for p, (key, _) in enumerate(keyed):
            buckets.setdefault(key, []).append(p)
        # Replicate the constructor's dedup fold exactly: one slot per key at its first
        # occurrence, the WHOLE PAIR with the strictly shortest target wins it (first on
        # ties). Positions in non-singleton buckets are dedup-entangled: removing one can
        # promote a twin or move a slot, which suppression cannot spell -- fallback tier.
        row_of, expected, irregular = {}, [], set()
        for p, (key, _) in enumerate(keyed):
            b = buckets[key]
            if len(b) > 1:
                irregular.add(p)
            if b[0] != p:
                continue
            w = min(b, key=lambda q: (len(keyed[q][1][1]), b.index(q)))
            row_of[w] = len(expected)
            expected.append(keyed[w][1])
        if expected != list(engine.simplification_rules):
            # The fold above no longer mirrors the constructor: refuse every shortcut
            # rather than trust a stale replica (each probe then runs the reference
            # construction itself).
            return {'broken': True, 'engine': engine, 'rebuild': False}
        census = engine._core.ac_shadow_census()
        owner_map, owner_regular = {}, {}
        for victim, owner, was_twin, kind in census:
            owner_map.setdefault(owner, []).append(victim)
            ok = (kind == 'twin-shadow') and not was_twin
            owner_regular[owner] = owner_regular.get(owner, True) and ok
        return {'broken': False, 'engine': engine, 'rebuild': False,
                'live_pos': live_pos, 'row_of': row_of, 'irregular': irregular,
                'owner_map': owner_map, 'owner_regular': owner_regular,
                'pruned_rows': set()}

    state = None
    done = fallbacks = 0
    for i in range(len(current)):
        lhs, rhs = current[i]
        if not any(Q.match(t) for t in lhs):
            keep.append((lhs, rhs))
            continue
        if state is None or state['rebuild']:
            state = build_probe_state()
        done += 1
        if done % 500 == 0:
            prog.stage('stage 4/5 derivability probe', probes=f'{done}/{n_probes}',
                       pruned=len(pruned), fallbacks=fallbacks)

        pos = state['live_pos'].get(i) if not state['broken'] else None
        row_i = state['row_of'].get(pos) if pos is not None else None
        removal = set(state['pruned_rows']) if not state['broken'] else set()
        if row_i is not None:
            removal.add(row_i)
        use_fallback = state['broken'] or (pos is not None and pos in state['irregular'])
        suppress = []
        if not use_fallback:
            for r in removal:
                victims = state['owner_map'].get(r)
                if victims is None:
                    suppress.append(r)
                    continue
                if not state['owner_regular'][r]:
                    use_fallback = True
                    break
                if any(v not in removal for v in victims):
                    # A victim survives the removal set: the rebuild would resurrect its
                    # twin, which is this owner's own rewrite -- keep the owner as the
                    # stand-in instead of suppressing it.
                    continue
                suppress.append(r)

        if use_fallback:
            # The reference construction, verbatim -- the arbiter for every candidate the
            # batched fast path cannot serve exactly.
            fallbacks += 1
            rest = [[list(l), list(r)] for j, (l, r) in enumerate(current)
                    if j != i and (l, r) not in pruned_set]
            e2 = SimpliPyEngine(operators=OPS, rules=rest)
            try:
                got = list(e2.simplify(fresh(lhs)))
            except Exception:
                keep.append((lhs, rhs))
                continue
        else:
            probe = fresh(lhs)
            form = 'tagged' if _TAGGED_DIALECT_TOKENS.intersection(probe) else 'explicit'
            try:
                got = list(state['engine']._core.ac_simplify_suppressed(
                    probe, 48, form, DEFAULT_EFFORT, sorted(suppress)))
            except Exception:
                keep.append((lhs, rhs))
                continue

        if got == list(fresh(rhs)):
            pruned.append((lhs, rhs))
            pruned_set.add((lhs, rhs))
            if not state['broken']:
                if use_fallback or row_i is None:
                    # A prune the row bookkeeping cannot spell (dedup-entangled, or the
                    # candidate had no row of its own): rebuild before the next probe.
                    state['rebuild'] = True
                else:
                    state['pruned_rows'].add(row_i)
        else:
            keep.append((lhs, rhs))
    prog.stage('stage 4/5 derivability probe: done', probes=f'{done}/{n_probes}',
               pruned=len(pruned), fallbacks=fallbacks)
    return keep
