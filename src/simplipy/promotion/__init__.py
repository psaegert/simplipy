# mypy: ignore-errors
"""Post-mine sort promotion: each rung asks the judge.

The miner emits every rule at the conservative ``?`` (variable-leaf) sort. This pass respells
each rule at the stronger sorts and ships it at the strongest one the judge licenses in the
rule's mode. The shipped files speak the four-sort language ``?`` < ``$`` < ``!`` < ``_``
(binding-set nesting; see docs/rules.md):

  ``_``  any subtree;
  ``!``  a subtree certified finite almost everywhere (a match-time certificate);
  ``$``  a subtree certified finite and nonzero almost everywhere (the seeded cancellation
         identities only);
  ``?``  a variable leaf: what the mine certified.

A RUNG is the rule with every slot respelled at one sort, and it asks the judge:
:func:`simplipy.verify.licensed` in the mode, which reads the judged tier at that sort (the
sorts' quantifiers are the judge's) and the named policies (a ``_`` slot may not be
duplicated). Promotion has no evaluator or probe list of its own.

Three steps:

  1. the ``_`` rung of every rule with slots;
  2. the derivability refund (:mod:`._refund`): the rules the ``_``-promoted set derives are
     pruned;
  3. the ``!`` rung of every rule still at ``?``. A rule licensed at neither ships at ``?``, as
     mined: the mine's judgment and its final gate decide it there.

Then the seeded cancellation identities, which no carrier can mine (``- !0 !0 -> 0`` and its
twins, whose sources are cancelled at enumeration; ``* 0 !0 -> 0``, whose carrier the canon
orders literal-last; ``/ $0 $0 -> 1`` and its twins, ``/ 0 $0 -> 0``), join the set where the
judge licenses them and the set lacks them. A rule without slots has no rung and passes through.

Promotion is deterministic, as the judge is, and fail-safe: a rule the judge does not license
at a stronger sort keeps the sort it was certified at.
"""
import time

from . import _refund

# The declared surface: the entry point alone -- the steps are private.
__all__ = ['promote']

#: The additive cancellation seeds and the zero product: three spellings of one identity, AC
#: equal and textually distinct, deduplicated by their internal form (`_ac_key`).
ADD_SEEDS = ((('-', '!0', '!0'), ('0',)),
             (('+', '!0', 'neg', '!0'), ('0',)),
             (('+', 'neg', '!0', '!0'), ('0',)),
             (('*', '0', '!0'), ('0',)))

#: The multiplicative twins, at the finite-nonzero sort.
MULT_SEEDS = ((('/', '$0', '$0'), ('1',)),
              (('*', '$0', 'inv', '$0'), ('1',)),
              (('*', 'inv', '$0', '$0'), ('1',)),
              (('/', '0', '$0'), ('0',)))

#: The positive controls, asked of the rule judge itself (`verify.verify_rule`) before any rung:
#: `x + 1 -> x` holds in no mode at the `_` sort, and `1 * x -> x` holds in every mode. A judge
#: that answers either wrongly is miscalibrated, and promotion raises rather than promote by it.
#: (The rungs ask the caller's `judge`, which a test may replace.)
CONTROLS = ((('+', '_0', '1'), ('_0',), False),
            (('*', '1', '_0'), ('_0',), True))


def _is_slot(t):
    return t[:1] in ('?', '_', '!', '$') and t[1:].isdigit()


def _has_slots(rule):
    return any(_is_slot(t) for side in rule for t in side)


def _respell(tokens, sort):
    """Every slot of `tokens` at `sort`."""
    return tuple(sort + t[1:] if _is_slot(t) else t for t in tokens)


def _rungs(pairs, mode, engine, judge):
    """The licence of each pair in `mode` (:func:`simplipy.verify.licensed`), its tier from
    `judge` (one batch over the distinct pairs)."""
    from ..verify import licensed
    distinct = list(dict.fromkeys(pairs))
    if not distinct:
        return []
    tiers = judge([(list(l), list(r)) for l, r in distinct])
    out = {p: licensed(list(p[0]), list(p[1]), mode, engine, detail={'tier': tier})
           for p, tier in zip(distinct, tiers)}
    return [out[p] for p in pairs]


def _run_controls(mode, engine):
    from ..verify import effective_tier, verify_rule

    def judge(pairs):
        return [effective_tier(verify_rule(list(lhs), list(rhs))) for lhs, rhs in pairs]
    got = _rungs([(l, r) for l, r, _ in CONTROLS], mode, engine, judge)
    for (lhs, rhs, want), verdict in zip(CONTROLS, got):
        if verdict['holds'] != want:
            raise RuntimeError(
                f'positive control failed in {mode}: {" ".join(lhs)} -> {" ".join(rhs)} '
                f'judged {verdict["tier"]}, which {"does not hold" if want else "holds"}')


def promote(rules, engine, *, run_positive_controls=True, verbose=False, progress=None,
            rule_mode, judge=None):
    """Assign each ``?``-sorted mined rule the strongest sort the judge licenses; prune the
    rules the ``_``-promoted set derives.

    ``rules``: iterable of (lhs, rhs) prefix-token sequences, ``?``-sorted (a slot of another
    sort is read as ``?``: the rule climbs the rungs from there).
    ``engine``: the ``SimpliPyEngine`` whose operators define the rules (the refund's probe
    engine and the named policies read it).
    ``run_positive_controls``: ask the rule judge the controls (``CONTROLS``) first, raising on
    a miss.
    ``verbose`` reports each step; ``progress`` overrides the reporter (see
    :mod:`simplipy.progress`).
    ``rule_mode`` is the mode the rules serve in (``'f64'``, also ``'default'``, ``'real'``,
    ``'permissive'`` or a :class:`~simplipy.Mode` member): each rung is licensed in it, and the
    derivability refund asks that mode's engine.
    ``judge``: the judged tier of each pair of a batch, as the mine's judge gives it
    (:func:`simplipy.mining.judge_tiers` when None).
    Returns ``(kept_rules, report)`` where ``kept_rules`` is a list of ``[lhs, rhs]`` lists and
    ``report`` lists the rules each rung promoted (``promoted_any``, ``promoted_bang``, the
    seeds among the latter with the reason ``seed`` or ``seed-mult``) with ``stage_counts``.
    """
    from ..engine import _coerce_rule_mode
    from ..mining import judge_tiers
    from ..progress import Progress
    rule_mode = _coerce_rule_mode(rule_mode)
    mode = 'f64' if rule_mode == 'default' else rule_mode
    judge = judge if judge is not None else judge_tiers
    prog = progress if progress is not None else Progress(verbose, prefix="  [promote] ")
    # Every slot starts at `?`, the sort the mine certified: a rule offered at a stronger sort
    # climbs the same rungs, and one licensed at neither ships at `?`.
    rules = [(_respell(l, '?'), _respell(r, '?')) for l, r in rules]
    prog.stage("sort promotion starting", rules=len(rules))
    if run_positive_controls:
        _run_controls(mode, engine)

    # A pure-literal ground rule (no slot, no `<constant>`) has no rung and takes no part in
    # the refund: it passes through, after the promoted set.
    def _pure_literal_ground(rule):
        return not _has_slots(rule) and '<constant>' not in rule[0] + rule[1]

    ground = [p for p in rules if _pure_literal_ground(p)]
    rules = [p for p in rules if not _pure_literal_ground(p)]
    report = {'promoted_any': [], 'promoted_bang': []}

    # 1. the `_` rung
    started = time.monotonic()
    slotted = [p for p in rules if _has_slots(p)]
    verdicts = _rungs([(_respell(l, '_'), _respell(r, '_')) for l, r in slotted], mode, engine,
                      judge)
    promote_set = set()
    for (l, r), v in zip(slotted, verdicts):
        if v['licensed']:
            promote_set.add((l, r))
            report['promoted_any'].append((l, r, 'rung', v['tier']))
    prog.stage("rung _", rules=len(slotted), licensed=len(promote_set),
               seconds=round(time.monotonic() - started, 1))

    # 2. the derivability refund over the `_`-promoted set
    sorted_rules = _refund.refund(rules, engine._operators_config, promote_set, progress=prog,
                                  rule_mode=rule_mode)
    prog.stage("refund", kept=len(sorted_rules))

    # 3. the `!` rung of every rule still at `?`
    started = time.monotonic()
    at_q = [p for p in sorted_rules if any(t[:1] == '?' and _is_slot(t) for t in p[0] + p[1])]
    bang = {p: (_respell(p[0], '!'), _respell(p[1], '!')) for p in at_q}
    verdicts = dict(zip(at_q, _rungs([bang[p] for p in at_q], mode, engine, judge)))
    kept = []
    for p in sorted_rules:
        v = verdicts.get(p)
        if v is not None and v['licensed']:
            kept.append(bang[p])
            report['promoted_bang'].append((p[0], p[1], 'rung', v['tier']))
        else:
            kept.append(p)
    prog.stage("rung !", rules=len(at_q),
               licensed=sum(1 for v in verdicts.values() if v['licensed']),
               seconds=round(time.monotonic() - started, 1))

    # The seeded cancellation identities. THE RE-SEED GUARD IS KEYED ON THE INTERNAL FORM,
    # not the spelling: each family seeds three AC-equal spellings of one identity, and a
    # spelling key would land every twin as a duplicate row of the one before it. Batched
    # through the core in one call; a side the AC parser refuses keeps its spelling as the key.
    def _ac_key(sides):
        keys = engine._core.ac_canonical_keys([list(s) for s in sides], rule_mode=rule_mode)
        return [(True, tuple(k)) if k is not None else (False, tuple(s))
                for k, s in zip(keys, sides)]

    seen = set(_ac_key([l for l, _ in kept]))
    seeded = 0
    for seeds, reason in ((ADD_SEEDS, 'seed'), (MULT_SEEDS, 'seed-mult')):
        todo = [(s, k) for s, k in zip(seeds, _ac_key([l for l, _ in seeds])) if k not in seen]
        for (s, k), v in zip(todo, _rungs([s for s, _ in todo], mode, engine, judge)):
            if v['licensed'] and k not in seen:
                kept.append(s)
                report['promoted_bang'].append((s[0], s[1], reason, v['tier']))
                seen.add(k)
                seeded += 1

    kept = [[list(l), list(r)] for l, r in kept + ground]
    report['stage_counts'] = {
        'rules': len(rules) + len(ground), 'ground_passthrough': len(ground),
        'promoted_any': len(report['promoted_any']),
        'promoted_bang': len(report['promoted_bang']) - seeded, 'seeded': seeded,
        'after_refund': len(sorted_rules), 'final': len(kept),
    }
    prog.stage("sort promotion complete", final=len(kept), elapsed=prog.total_elapsed())
    return kept, report
