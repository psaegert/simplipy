# mypy: ignore-errors
"""Post-mine sort promotion.

The miner emits every rule at the conservative ``?`` (variable-leaf) sort. This pass
re-certifies each rule at the stronger sorts and ships it at the strongest SOUND one.
The SHIPPED artifact speaks the four-sort language ``?`` < ``$`` < ``!`` < ``_``
(binding-set nesting; see docs/rules.md): mined rules descend the ``_`` -> ``!`` -> ``?``
ladder, and the ``$`` sort ships through the SEEDED multiplicative-cancellation
identities (``$0/$0 -> 1`` and twins), certified at the finite-nonzero bar -- the
unminable cancellation family, entered explicitly rather than reached by demotion:

  ``_``  arbitrary-subtree (pointwise Dirac bar over the atom lattice, both directions),
  ``!``  certified-finite subtree (the finite-only bar, enforced by a match-time
         defined-and-finite-a.e. certificate),
  ``$``  certified-finite-AND-nonzero subtree (seeded identities only; match-time
         ``finite_nonzero_ae`` certificate -- the multiplicative group's regular domain),
  ``?``  variable-leaf (the fallback: what the mine already certified).

Five stages, matching the reference pipeline exactly:
  1. pointwise ``_``-bar on const-free wildcard rules,
  2. const-bearing witness-map ``_``-bar,
  3. exact-arbiter overturn of finite-draw demotions (f64 saturation is not authoritative),
  4. apply ``?``->``_`` for the promoted set and refund the now-derivable instances,
  5. the ``_``->``!``->``?`` descent ladder on the enriched atom lattice, with the
     moving-spike structural refusal, plus the ``$``-seeded cancellation identities
     (``judge_bang_mult``, the finite-nonzero atom lattice).

Promotion is fail-safe: a rule that cannot be certified at a stronger sort ships at ``?`` and
loses only composite-subtree recall, never soundness.
"""
import numpy as np

from . import _f64_eval, _pointwise, _const_bearing, _overturn, _refund, _ladder


def promote(rules, engine, *, run_positive_controls=True):
    """Assign each ``?``-sorted mined rule its strongest sound sort; prune derivable rules.

    ``rules``: iterable of (lhs, rhs) prefix-token sequences, all ``?``-sorted.
    ``engine``: the ``SimpliPyEngine`` whose operator realizations define the semantics.
    Returns ``(kept_rules, report)`` where ``kept_rules`` is a list of ``[lhs, rhs]`` lists.
    """
    _f64_eval.configure(engine)
    rules = [(tuple(l), tuple(r)) for l, r in rules]

    # SPECIAL-BEARING pure-literal ground rules (no wildcards, no `<constant>`, an
    # `np.pi`/`np.e` in the LHS) BYPASS promotion: the sorts classify WILDCARD
    # instantiation domains and the ladder's ground tier judges exactly the
    # one-`<constant>`-slot skeleton family -- a zero-slot literal rule like
    # `cos np.pi -> (-1)` has NO axis for any of the judges to quantify over (one
    # instantiation: itself, certified at mint + stage-2 confirmation + the hiprec
    # arbiter, which reads specials as their transcendental values). `judge_ground`
    # scored them UNSUPPORTED -- a scoping refusal, not a verdict -- and they silently
    # vanished (measured 2026-08-01: the owner-ruled exact collapses `cos np.pi -> (-1)`,
    # `sin np.pi -> 0`, `tan np.pi -> 0`, `log np.e -> 1` minted, confirmed, survived
    # the covered-prune, then dropped here). They pass through unchanged, appended after
    # the ladder's kept set.
    #
    # WIDENED to ALL pure-literal ground rules (fold unification, 2026-08-02): with
    # the engine's transcendental fold deleted, legitimate SPECIAL-FREE exact hits
    # (`cos 0 -> 1`, `exp 0 -> 1`, `log 1 -> 0`) now arrive as mined ground rules and
    # must pass through like their special-bearing siblings -- same argument: zero
    # wildcard slots, so no judge here has an axis to quantify over; certification is
    # the mint + stage-2 confirmation + the SYMBOLIC GATE (precision-stability, which
    # kills any respell or near-identity however small). The original special-only
    # scoping existed to contain the f64-respell family, which is now refused at the
    # mint three ways (strict-mu resolved criterion, the skeleton guard, the symbolic
    # gate), so the containment role is retired. `<constant>`-bearing ground rules
    # keep flowing to the ladder's ground tier (judge_ground), exactly as in the
    # reference.
    def _pure_literal_ground(l, r):
        return (not _pointwise.wildcards(list(l))
                and '<constant>' not in (list(l) + list(r)))

    ground = [(l, r) for l, r in rules if _pure_literal_ground(l, r)]
    rules = [(l, r) for l, r in rules if not _pure_literal_ground(l, r)]

    # STAGE 1 -- pointwise `_`-bar on const-free wildcard rules.
    rng1 = np.random.default_rng(_pointwise.SEED)
    scoped_cf = [(l, r) for l, r in rules
                 if _pointwise.wildcards(l) and _pointwise.is_const_free(list(l) + list(r))]
    pw = {'PROMOTE': [], 'DEMOTE': [], 'UNDECIDED': [], 'EVAL-ERR': []}
    for lhs, rhs in scoped_cf:
        ws = _pointwise.wildcards(list(lhs) + list(rhs))
        v, info = _pointwise.judge(list(lhs), list(rhs), _pointwise.valuations_for(ws, rng1))
        pw[v].append((lhs, rhs, info))

    # STAGE 2 -- const-bearing witness-map bar.
    rng2 = np.random.default_rng(_const_bearing.SEED)
    scoped_cb = [(l, r) for l, r in rules
                 if _pointwise.wildcards(list(l)) and '<constant>' in (list(l) + list(r))]
    cb = {'PROMOTE': [], 'DEMOTE': [], 'NO-WITNESS': [], 'EVAL-ERR': []}
    for lhs, rhs in scoped_cb:
        v, info = _const_bearing.certify_rule(lhs, rhs, rng2)
        cb[v].append((lhs, rhs, info))

    # STAGE 3 -- exact-arbiter overturn of finite-draw demotions. A kill at an ATOM valuation
    # is exact special-value algebra; a kill at a FINITE draw is f64 evidence that may be pure
    # saturation (atanh(tanh(37)) since tanh(37) == 1.0), so it is re-judged at high precision.
    rng3 = np.random.default_rng(_overturn.SEED)
    finite_kills = [(l, r, info) for l, r, info in pw['DEMOTE']
                    if not all(v in _pointwise.ATOMS for v in info[0].values())]
    overturned = []
    for lhs, rhs, info in finite_kills:
        ws = _pointwise.wildcards(list(lhs) + list(rhs))
        v, _kill = _overturn.judge_exact(list(lhs), list(rhs), _pointwise.valuations_for(ws, rng3))
        if v == 'PROMOTE':
            overturned.append((lhs, rhs, info))
    ot_keys = {(tuple(l), tuple(r)) for l, r, _ in overturned}
    pw['DEMOTE'] = [x for x in pw['DEMOTE'] if (tuple(x[0]), tuple(x[1])) not in ot_keys]
    pw['PROMOTE'] = pw['PROMOTE'] + [(l, r, 'overturned-by-exact-arbiter') for l, r, _ in overturned]

    # STAGE 4 -- apply ?->_ for the promoted set, then refund the derivable instances.
    promote_set = {(tuple(l), tuple(r)) for l, r, *_ in pw['PROMOTE']} \
        | {(tuple(l), tuple(r)) for l, r, *_ in cb['PROMOTE']}
    sorted_rules = _refund.refund(rules, engine._operators_config, promote_set)

    # STAGE 5 -- the _->!->? ladder on the enriched lattice + moving-spike refusal.
    kept, report = _ladder.promote_rules(sorted_rules, engine,
                                         run_positive_controls=run_positive_controls)
    kept = list(kept) + [[list(l), list(r)] for l, r in ground]
    report = dict(report)
    report['stage_counts'] = {
        'pointwise_promote': len(pw['PROMOTE']), 'pointwise_demote': len(pw['DEMOTE']),
        'cb_promote': len(cb['PROMOTE']), 'overturned': len(overturned),
        'ground_passthrough': len(ground),
        'after_refund': len(sorted_rules), 'final': len(kept),
    }
    return kept, report
