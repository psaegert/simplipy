"""THE DIFFERENTIAL TEST'S WORKLOAD: the specification's inputs (its section 7), generated, in
f64 and in permissive. `test_differential` holds every step to the judge on fixed examples; this
module runs the inputs the specification names and holds every step to the judge on them.

The inputs, fixed by `SEED`:

1. the scan: every (source, candidate) pair the length-4 scan considers for 500 sources drawn
   from the canonical sources of lengths 2 to 4 over the acj2-4 operators (the reduced
   alphabet: two variables, `<constant>`, (-1), 0 and 1; targets to length 3), with the verdict
   that decided each (`find_rule_lib_traced`): the pre-filter that rejected the candidate, or
   the acceptance's answer on the target it emits;
2. the completions: every overlap completion within the cell the engine proposes on the shipped
   rule file (acj-4-3's; acj2-4's ship with the release mine) and on the file of the scan's
   licensed picks, each installed as the mode's set;
3. the promotions: every respelling sort promotion tries on those picks (`?` to `_` and `!`,
   and the seeds);
4. the probe table: each operator at its poles, at +-0, +-inf and NaN, at the overflow and
   underflow thresholds and at the subnormals;
5. the examples of sections 1 to 4 (here those of section 1; `test_differential` holds the
   others).

The six assertions, each where its test says. Where the workload is narrowed, the test says how,
and why.
"""
import collections
import functools
import math
import os

import numpy as np
import pytest
import yaml
from mpmath import mp

from simplipy import SimpliPyEngine
from simplipy.verify import MODE_TIERS, effective_tier, licensed
from simplipy.verify import _contract

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ('f64', 'permissive')
RULE_MODE = {'f64': 'default', 'permissive': 'permissive'}
SEED = 20261101
N_SOURCES = 500
LEAVES = ('x0', 'x1', '<constant>', '(-1)', '0', '1')
MAX_SOURCE, MAX_TARGET = 4, 3
N_ROWS = 512
FIT = dict(challenges=8, retries=8, rtol=1e-11, atol=1e-12, min_informative=N_ROWS // 8)

#: The scan's verdicts (`worker::certify_candidate`, `Acceptor::license`, `Acceptor::confirm`),
#: by kind: a pre-filter of truth ('N', it may reject only what the judge rejects), a refusal
#: counted as undecided ('U', fail closed, 4.10), a named policy (the policy's name, 4.x), or a
#: bar of the mine's own search ('bar': below the engine's answer, the judge loop's exclusions).
VERDICTS = {
    'reach': 'N', 'finite_rows': 'N', 'rows': 'N', 'fit': 'N', 'domain_gate': 'N',
    'special_points': 'N', 'special_constants': 'N', 'dirac': 'N', 'resolved_rows': 'N',
    'evaluable': 'N', 'not_realised': 'N', 'unconfirmed': 'N', 'pole-class-change': 'N',
    'domain_undecided': 'U', 'blind': 'U',
    'variable_mask': 'well_formed', 'constant_count': 'constant_count',
    'resolution': 'constant_count', 'ground_composite': 'ground_composite',
    'collapse_licence': 'evidence', 'evidence': 'evidence', 'ordering': 'ordering',
    'engine_owned': 'engine_owned', 'engine-owned-law': 'engine_owned',
    'well_formed': 'well_formed', 'special_absorption': 'specials',
    'const_introduction': 'constant_count',
    'not_below': 'bar', 'not_structural': 'bar', 'excluded': 'bar', 'not_served': 'bar',
    'undecided': 'U',
    'certified': 'pass', 'licensed': 'pass', 'confirmed': 'pass',
}


@functools.lru_cache(maxsize=1)
def engine() -> SimpliPyEngine:
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


#: The operators the probe table reads beyond acj2-4's: every operator the package realizes.
PROBE_ONLY = {'erfc': 1, 'bessely': 2, 'besseli': 2, 'besselk': 2}


@functools.lru_cache(maxsize=1)
def probe_engine() -> SimpliPyEngine:
    """acj2-4's operators and `PROBE_ONLY`: the probe table's vocabulary (the scan's workload
    stays on acj2-4's, as the specification fixes it)."""
    ops = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4',
                                           'config.yaml')))['operators']
    for name, arity in PROBE_ONLY.items():
        ops[name] = {'realization': f'simplipy.operators.{name}', 'alias': [], 'arity': arity,
                     'precedence': 2, 'commutative': False}
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


@functools.lru_cache(maxsize=1)
def matrix() -> tuple:
    """The mine's rows and confirmation rows, drawn as a mine of `SEED` draws them."""
    rng = np.random.default_rng(SEED)
    data = _contract.mine_rows(N_ROWS, 2, rng)
    return data, _contract.mine_rows(2 * N_ROWS, 2, rng)


def _arity() -> dict:
    return dict(sorted(engine().operator_arity.items(), key=lambda x: x[1]))


@functools.lru_cache(maxsize=None)
def library(mode: str):
    from simplipy.utils import enumerate_expressions
    exprs = enumerate_expressions(list(LEAVES), _arity(), MAX_TARGET)
    candidates = [list(e) for length in sorted(exprs) for e in sorted(exprs[length])]
    data, _ = matrix()
    return engine()._core.build_candidate_library(
        candidates, ['x0', 'x1'], data.flatten(order='C').tolist(), data.shape[0],
        rule_mode=RULE_MODE[mode])


@functools.lru_cache(maxsize=None)
def sources(mode: str) -> tuple:
    """`N_SOURCES` of the mode's canonical sources of lengths 2 to `MAX_SOURCE` (one per class,
    as the mine reads them), drawn by `SEED`."""
    from simplipy.utils import enumerate_expressions
    exprs = enumerate_expressions(list(LEAVES), _arity(), MAX_SOURCE)
    pool = set()
    for length in range(2, MAX_SOURCE + 1):
        raw = [list(e) for e in sorted(exprs[length])]
        pool.update(tuple(s) for s in engine()._core.mine_sources(raw,
                                                                  rule_mode=RULE_MODE[mode]))
    pool = sorted(pool)
    pick = np.random.default_rng(SEED).choice(len(pool), N_SOURCES, replace=False)
    return tuple(pool[i] for i in sorted(pick))


@functools.lru_cache(maxsize=None)
def scans(mode: str) -> tuple:
    """Each source's scan: `(source, pick, steps)`, every step `(candidate, target, verdict)`."""
    from simplipy.mining import RuleMiner
    data, confirm = matrix()
    kwargs = RuleMiner._confirm_kwargs(confirm, data, FIT['min_informative'], SEED)
    lib, core = library(mode), engine()._core
    out = []
    for k, src in enumerate(sources(mode)):
        pick, steps = core.find_rule_lib_traced(list(src), lib, MAX_TARGET, seed=SEED + k,
                                                rule_mode=RULE_MODE[mode], **FIT, **kwargs)
        out.append((src, None if pick is None else tuple(pick),
                    tuple((tuple(c), tuple(t), v) for c, t, v in steps)))
    return tuple(out)


def pattern(lhs, rhs) -> tuple:
    """A pair as the mine writes it: each variable the slot `?<k>`."""
    def slot(t):
        return '?' + t[1:] if t in ('x0', 'x1') else t
    return tuple(map(slot, lhs)), tuple(map(slot, rhs))


_JUDGED: dict = {}


def judge(pairs) -> dict:
    """The judge's record of each pair (`verify_rule` under the mine's watchdog, in a pool of
    four, on the mine's rows): `{pair: {'verdict', 'tier', 'basis'}}`, memoised."""
    import multiprocessing
    from simplipy.mining import _watched_judgments
    todo = [p for p in dict.fromkeys(pairs) if p not in _JUDGED]
    if todo:
        size = -(-len(todo) // 4)
        chunks = [[(list(lhs), list(rhs)) for lhs, rhs in todo[i:i + size]]
                  for i in range(0, len(todo), size)]
        with multiprocessing.get_context('spawn').Pool(
                len(chunks), initializer=_contract.set_rows,
                initargs=(_contract.judge_rows(),)) as pool:
            parts = pool.map(_watched_judgments, chunks)
        for pair, d in zip(todo, [d for part in parts for d in part]):
            _JUDGED[pair] = {'verdict': d.get('verdict'), 'tier': effective_tier(d),
                             'basis': d.get('basis') or {}}
    return {p: _JUDGED[p] for p in pairs}


def holds(record: dict, mode: str) -> bool:
    return record['tier'] in MODE_TIERS[mode]


@pytest.fixture(autouse=True, scope='module')
def _the_judge_reads_the_mines_rows():
    data, confirm = matrix()
    before = _contract.set_rows(np.vstack([data, confirm]))
    yield
    _contract.set_rows(before)


#: What a refusal of the mint's hint arm (`judge_hint`) is, beside the scan's verdicts.
HINT = {'accepted': 'pass', 'uncertified': 'N', 'atomic_mark': 'bar', 'respelled': 'bar',
        'wildcard_multiplicity': 'duplication'}

#: The policies the mint applies to a pick (`Acceptor::license`, the scan's emit guard and its
#: constant count); `cell_bound` is the mine's, `duplication` its candidate selection's.
MINT_POLICIES = ('evidence', 'constant_count', 'ground_composite', 'ordering', 'specials',
                 'engine_owned', 'well_formed')

#: The pairs judged of each verdict's rejections: all where there are at most this many, a
#: sample drawn by `SEED` where there are more. The scan rejects about 100,000 pairs per mode,
#: nearly all on the rows; the judge takes 0.6 s (f64) to 0.8 s (permissive) a pair in each of
#: four processes, about five hours per mode for all of them.
SAMPLE = 120

#: What the assertions saw, for the gate's log: counts by assertion and mode.
REPORT: dict = {}


def _n_constants(tokens) -> int:
    return sum(t == '<constant>' for t in tokens)


def kind(src, cand, tgt, verdict) -> str:
    """The kind of a scan step's verdict (`VERDICTS`). A candidate carrying more constants than
    its source, refused before its constants were resolved to a literal (no target formed), is
    the constant count's (4.2), whatever check the scan ran on it: the only rule it could mint is
    a resolved literal, which it never formed."""
    if cand and not tgt and _n_constants(cand) > _n_constants(src):
        return 'constant_count'
    return VERDICTS.get(verdict, '?')


def sampled(pairs: list) -> list:
    if len(pairs) <= SAMPLE:
        return list(pairs)
    rng = np.random.default_rng(SEED)
    return [pairs[i] for i in sorted(rng.choice(len(pairs), SAMPLE, replace=False))]


@functools.lru_cache(maxsize=None)
def by_kind(mode: str) -> dict:
    """The distinct pairs of the scan's steps, by verdict, with the verdict's kind:
    `{verdict: (kind, [pair])}`, each pair as the mine writes it (the target the scan formed,
    else the candidate)."""
    by: dict = collections.defaultdict(set)
    kinds: dict = {}
    for src, _, steps in scans(mode):
        for cand, tgt, v in steps:
            if not (cand or tgt):
                continue
            k = kind(src, cand, tgt, v)
            key = v if k == VERDICTS.get(v) else f'{v} (constant count)'
            kinds[key] = k
            by[key].add(pattern(src, tgt or cand))
    return {v: (kinds[v], sorted(p)) for v, p in sorted(by.items())}


@pytest.mark.parametrize('mode', MODES)
def test_every_verdict_of_the_scan_is_named(mode) -> None:
    """Every step the scan records carries a verdict this module classifies."""
    unknown = {v for _, _, steps in scans(mode) for _, _, v in steps if v not in VERDICTS}
    assert not unknown, unknown
    assert sum(len(steps) for _, _, steps in scans(mode)) > 100_000


# ------------------------------------------------------------------------------------------
# Assertion 1: every pre-filter rejection is a judge rejection, except refusals counted as
# undecided (4.10), which are reported with their count.
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('mode', MODES)
def test_a_prefilter_rejection_is_a_judge_rejection(mode) -> None:
    from simplipy import _core
    todo = {v: sampled(pairs) for v, (k, pairs) in by_kind(mode).items() if k == 'N'}
    record = judge([p for pairs in todo.values() for p in pairs])
    admitted = [(v, ' '.join(lhs), ' '.join(rhs), record[(lhs, rhs)]['verdict'])
                for v, pairs in todo.items() for lhs, rhs in pairs
                if holds(record[(lhs, rhs)], mode)]
    undecided = collections.Counter(v for _, _, steps in scans(mode) for _, _, v in steps
                                    if VERDICTS.get(v) == 'U')
    # a source with evidence whose rows could not be drawn is refused undecided
    for src, _, steps in scans(mode):
        if any(v == 'evidence' for _, _, v in steps) and any(t in ('x0', 'x1') for t in src) \
                and _core.table_mode_evidence(list(src), dict(_contract.ARITY),
                                              RULE_MODE[mode]) is True:
            undecided['evidence rows'] += 1
    REPORT[('assertion 1', mode)] = {
        'rejections': {v: len(by_kind(mode)[v][1]) for v in todo},
        'judged': {v: len(pairs) for v, pairs in todo.items()},
        'undecided': dict(undecided)}
    assert not admitted, admitted


# ------------------------------------------------------------------------------------------
# Assertion 2: every rule the mint accepts is licensed.
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('mode', MODES)
def test_every_pick_of_the_mint_is_licensed(mode) -> None:
    pairs = list(dict.fromkeys(pattern(src, pick) for src, pick, _ in scans(mode)
                               if pick is not None))
    record = judge(pairs)
    refused = []
    for lhs, rhs in pairs:
        got = licensed(list(lhs), list(rhs), mode, engine(), max_source=MAX_SOURCE,
                       detail=record[(lhs, rhs)])
        if not got['licensed']:
            refused.append((' '.join(lhs), ' '.join(rhs), record[(lhs, rhs)]['verdict'],
                            got['refused_by']))
    REPORT[('assertion 2', mode)] = {'picks': len(pairs), 'refused': len(refused)}
    assert pairs and not refused, refused


# ------------------------------------------------------------------------------------------
# Assertion 3: each policy gives the same answer from every step that applies it.
# ------------------------------------------------------------------------------------------
def _policies(lhs, rhs, mode) -> list:
    """The policies the licence names for a pair (`licensed`'s `refused_by`, which reads no
    tier)."""
    return licensed(list(lhs), list(rhs), mode, engine(), max_source=MAX_SOURCE,
                    detail={'tier': 'reject'})['refused_by']


@pytest.mark.parametrize('mode', MODES)
def test_the_scans_policies_are_the_licences(mode) -> None:
    """A policy refusal of the scan (its constant count, emit guard and collapse licence, the
    acceptance's load predicate, evidence and registry) is one the licence makes of the same
    pair, or the pair holds in no reading of the mode (the judge refuses it, whatever names the
    scan's refusal), and a source the scan reads has evidence by the policy's function (4.1)
    where it has variables (a source of constants alone reads the collapse licence). The
    licence that reads no tier names the policies a pair meets whatever it judges; where it does
    not name the scan's, the licence on the judge's verdict decides: permissive's evidence is
    that of a reading the rule holds in, so `acosh(erf(tan(x0))) -> 0`, which permissive's own
    criterion holds vacuously (no value in the reals) and f64 refuses, has none."""
    from simplipy import _core
    wrong = []
    for src, _, steps in scans(mode):
        if not any(t in ('x0', 'x1') for t in src) or not any(c for c, _, _ in steps):
            continue
        if _core.table_mode_evidence(list(src), dict(_contract.ARITY),
                                     RULE_MODE[mode]) is not True:
            wrong.append(('read without evidence', ' '.join(src)))
    checked, unnamed = {}, []
    for v, (k, pairs) in by_kind(mode).items():
        if k not in MINT_POLICIES:
            continue
        checked[v] = len(sampled(pairs))
        unnamed += [(v, k, lhs, rhs) for lhs, rhs in sampled(pairs)
                    if k not in _policies(lhs, rhs, mode)]
    record = judge([(lhs, rhs) for _, _, lhs, rhs in unnamed])
    for v, k, lhs, rhs in unnamed:
        got = licensed(list(lhs), list(rhs), mode, engine(), max_source=MAX_SOURCE,
                       detail=record[(lhs, rhs)])
        if got['holds'] and k not in got['refused_by']:
            wrong.append((v, ' '.join(lhs), ' '.join(rhs), got['refused_by']))
    REPORT[('assertion 3, the scan', mode)] = dict(checked, judged=len(unnamed))
    assert checked and not wrong, wrong


@pytest.mark.parametrize('lhs, rhs', [('abs asinh log ?1', 'asinh abs ?0'),
                                      ('acosh ceil inv ?0', 'log atanh ?1')])
@pytest.mark.parametrize('mode', MODES)
def test_the_load_names_the_policy_before_f64s_pre_filter(lhs, rhs, mode) -> None:
    """A right side's slot the left side does not bind is ill-formed in every mode: the load
    names that policy first, before f64's pole entry, a pre-filter of truth, which refuses these
    pairs too and would hide the policy from the licence."""
    assert 'well_formed' in _policies(tuple(lhs.split()), tuple(rhs.split()), mode)


@functools.lru_cache(maxsize=None)
def picks_file(mode: str) -> tuple:
    """The scan's licensed picks: the workload's rule file of the mode."""
    pairs = list(dict.fromkeys(pattern(src, pick) for src, pick, _ in scans(mode)
                               if pick is not None))
    record = judge(pairs)
    return tuple(p for p in pairs
                 if licensed(list(p[0]), list(p[1]), mode, engine(), max_source=MAX_SOURCE,
                             detail=record[p])['licensed'])


#: The shipped rule file the completions are proposed on: acj-4-3's (the last file shipped, 5,319
#: rules; acj2-4's ship with the release mine).
SHIPPED = os.path.join(REPO, 'tests', 'fixtures', 'acj-4-3')


@functools.lru_cache(maxsize=None)
def completions(mode: str) -> tuple:
    """Every overlap completion the engine proposes within the cell on the shipped file, read as
    the mode's set, and on the picks' file: `(instance pair, pattern)`."""
    import json
    from simplipy.mining import RuleMiner, _completion_pattern, _in_cell, _unjoined_overlaps
    shipped_ops = yaml.safe_load(open(os.path.join(SHIPPED, 'config.yaml')))['operators']
    shipped = [(tuple(lhs), tuple(rhs))
               for lhs, rhs in json.load(open(os.path.join(SHIPPED, 'rules.json')))]
    out = []
    for ops, rules in ((shipped_ops, shipped), (engine()._operators_config, picks_file(mode))):
        eng = SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])
        RuleMiner(eng)._install(RULE_MODE[mode], list(rules))
        out += [(tuple(map(tuple, o['completion'])), _completion_pattern(o))
                for o in _unjoined_overlaps(eng, RULE_MODE[mode], list(rules))
                if _in_cell(o, MAX_SOURCE)]
    return tuple(dict.fromkeys(out))


@pytest.mark.parametrize('mode', MODES)
def test_a_completion_is_judged_as_the_mint_judges_it(mode) -> None:
    """Each completion the overlap gate asks the licence about, put to the mint's hint arm
    (`judge_hint`, the scan's acceptance): a policy refusal of the mint is the licence's, a
    completion the mint accepts no policy of the mint's refuses, and a numeric refusal of the
    mint is a judge rejection (else the gate would demand a completion the mint cannot mint)."""
    comps = [(inst, p) for inst, p in completions(mode)
             if set(inst[0] + inst[1]) & {f'x{k}' for k in range(2, 10)} == set()]
    record = judge([p for _, p in comps])
    wrong = []
    for (lhs, rhs), pair in comps:
        ok, why = engine()._core.judge_hint(list(lhs), list(rhs), library(mode), seed=SEED,
                                            rule_mode=RULE_MODE[mode], **FIT)
        k = HINT.get(why, VERDICTS.get(why, '?'))
        names = _policies(*pair, mode)
        if k in MINT_POLICIES and k not in names:
            wrong.append(('mint', why, pair, names))
        if ok and set(names) & set(MINT_POLICIES):
            wrong.append(('licence', why, pair, names))
        if k == 'N' and holds(record[pair], mode):
            wrong.append(('judge', why, pair, record[pair]['verdict']))
        if k == '?':
            wrong.append(('unnamed', why, pair))
    REPORT[('assertion 3, completions', mode)] = {'completions': len(completions(mode)),
                                                  'with two variables': len(comps)}
    assert not wrong, wrong


@pytest.mark.parametrize('mode', MODES)
def test_every_rung_of_promotion_is_the_licences(mode, monkeypatch) -> None:
    """Every respelling sort promotion tries on the picks (`?` to `_` and `!`, and the seeds) is
    licensed exactly where the licence licenses it, with the judge's tier."""
    import simplipy.promotion as promotion
    asked = []
    inner = promotion._rungs

    def recording(pairs, mode_, engine_, judge_):
        out = inner(pairs, mode_, engine_, judge_)
        asked.extend(zip([(tuple(lhs), tuple(rhs)) for lhs, rhs in pairs], out))
        return out

    monkeypatch.setattr(promotion, '_rungs', recording)
    promotion.promote([[list(lhs), list(rhs)] for lhs, rhs in picks_file(mode)], engine(),
                      rule_mode=RULE_MODE[mode], run_positive_controls=False)
    record = judge([p for p, _ in asked])
    wrong = []
    for pair, verdict in asked:
        got = licensed(list(pair[0]), list(pair[1]), mode, engine(), detail=record[pair])
        if (got['licensed'], got['tier']) != (verdict['licensed'], verdict['tier']):
            wrong.append((pair, verdict, got))
    REPORT[('assertion 3, promotions', mode)] = {'rungs': len(asked)}
    assert asked and not wrong, wrong


# ------------------------------------------------------------------------------------------
# Assertion 4: every consumer's value at a probe agrees with the table, and every interval
# enclosure contains the table's values.
# ------------------------------------------------------------------------------------------
UNARY_ARGS = ['0', '-0.0', '1', '(-1)', '2', '-2', '0.5', 'float("inf")', 'float("-inf")',
              'float("nan")', '5e-324', '2.2250738585072014e-308', '1.7976931348623157e308',
              '709.782712893384', '710', '-745.1332191019412', '-746', '/ np.pi 2', 'np.pi',
              'np.e', '3']
BINARY_ARGS = ['0', '-0.0', '1', '(-1)', '2', '0.5', '-2', 'float("inf")', 'float("-inf")',
               'float("nan")', '5e-324', '1.7976931348623157e308', '3']


def _pairs(orders: list, args: list) -> list:
    return [[a, b] for a in orders for b in args]


#: Each operator's own range thresholds and boundaries, beside the shared arguments: erfc's
#: plateau at 2 and its cut to 0 (subnormal from about 26.55); Y's and I's orders where the
#: limit at 0 changes (the negative half-integers, where Y has no value in the reals),
#: the order thresholds of the kernels (50 for I, 85.9 for K), and the arguments where float64
#: overflows (I past 713.99, Y near 0), cuts to 0 (K past 697.87 and 710 (1 + |nu|)) or turns NaN
#: (I below 5.6e-309).
OWN_PROBES = {
    'erfc': [[a] for a in ['-5.863584748755168', '-5.8635847487551676', '-6', '-1e300', '0.999',
                           '8', '26.55', '26.641747557046326', '26.64174755704633', '27',
                           '1e-300']],
    'bessely': _pairs(['-0.5', '-1.5', '-2.5', '0.25', '-0.25', '2.5', '50.5', '86', '1000',
                       '-1000'],
                      ['0', '-0.0', '5e-324', '1e-300', '1e-10', '700', '1e5', '1e10']),
    'besseli': _pairs(['-0.5', '-1.5', '-2.5', '50.5', '-50.5', '51', '-51', '1000'],
                      ['-0.0', '1e-310', '-1e-310', '1e-300', '700', '713.98', '714', '-714',
                       '1e160']),
    'besselk': _pairs(['0.25', '-0.5', '85.5', '86', '1000'],
                      ['1e-310', '1e-300', '697.87', '697.88', '698', '710', '711', '1e6']),
}


def probes() -> list:
    """Each operator at the probe arguments (`UNARY_ARGS`; a binary one at every pair of
    `BINARY_ARGS`, and its own thresholds, `OWN_PROBES`): its poles and boundaries (0, +-1,
    pi/2, the negative integers), +-0, +-inf, NaN, exp's overflow and underflow thresholds, the
    subnormals and the largest double. Every operator the package realizes (`probe_engine`)."""
    out = []
    for op, arity in sorted(probe_engine().operator_arity.items()):
        if arity == 1:
            out += [[op] + a.split() for a in UNARY_ARGS]
        else:
            out += [[op] + a.split() + b.split() for a in BINARY_ARGS for b in BINARY_ARGS]
        out += [[op] + [t for a in args for t in a.split()] for args in OWN_PROBES.get(op, [])]
    return out


def _class(x) -> str:
    """A value as `'nan'`, `'+inf'`, `'-inf'` or its double's repr (one zero)."""
    x = float(x)
    if x != x:
        return 'nan'
    if math.isinf(x):
        return '+inf' if x > 0 else '-inf'
    return repr(abs(x)) if x == 0 else repr(x)


def _bits(x: float) -> str:
    return 'nan' if x != x else repr(x)


def _literal(tokens) -> float | None:
    """A folded answer's value, where it is one literal."""
    if len(tokens) != 1:
        return None
    try:
        return float(_contract.literal_value(tokens[0]))
    except Exception:
        return None


def _contains(box, x) -> bool:
    """Does an enclosure `(has_fin, pinf, ninf, nan, lo, hi)` contain `x` (a double or the
    reals' value, past the doubles' range too)? None, no enclosure, contains nothing to check."""
    if box is None:
        return True
    has_fin, pinf, ninf, nan, lo, hi = box
    if x != x:
        return nan
    if x in (math.inf, -math.inf, mp.inf, -mp.inf):
        return pinf if x > 0 else ninf
    return has_fin and lo <= x <= hi


@functools.lru_cache(maxsize=1)
def probe_table() -> tuple:
    """Each probe's table values, `(tokens, real, (f64, in range))`; `real` None where the
    table cannot read it at 60 digits."""
    from simplipy import _core
    out = []
    for tokens in probes():
        expr = _core.TableExpr(tokens, [], dict(_contract.ARITY))
        spec, unresolved, *_ = expr.eval_real([], 60)
        old = mp.dps
        try:
            mp.dps = 60
            real = None if unresolved else _contract._from_spec(spec)
        finally:
            mp.dps = old
        out.append((tuple(tokens), real, expr.eval_f64([])))
    return tuple(out)


#: The defects this workload checks for, each in a step outside the miner, with the probes that
#: show it (`_defect_of`). A test of an open one (`OPEN`) is marked and fails today; the fix
#: removes it from `OPEN`, and the test keeps the defect from coming back.
DEFECTS = {
    'D1': "the engine's f64 fold of a numeric expression is not float64's value in written order "
          "(F8): `0.1 + 0.2` folds to 0.3, `0 * (-1)` to +0, `0.5/1.8e308` to 0, "
          "`besselj(inf, 3)` to 0",
    'I1': "the interval layer's f64 reading of a point zero's reciprocal must take the zero's "
          "sign: `1/(-0.0)` is -inf in float64",
    'I2': "the interval layer's f64 reading must read every double past 2^53 as an even "
          "integer exponent: `(-2)^1.8e308` is +inf in float64",
    'I3': "the interval layer's f64 reading of `besselj` at an infinite order must hold the "
          "kernel's nan: `besselj(inf, 3)`",
    'I4': "the interval layer's real enclosure of a value past the largest double must be open "
          "towards infinity, with no nan bound: `5e-324 + 1.8e308`",
    'I5': "the interval layer's f64 enclosure of an overflow must hold float64's infinity and the "
          "true value: `2 * 1.8e308`, `exp(710)`",
}
#: The defects still open.
OPEN = {'D1'}


def known(defect: str):
    assert defect in OPEN, defect
    return pytest.mark.xfail(strict=True, reason=f'{defect}: {DEFECTS[defect]}')


def _defect_of(consumer: str, tokens, got) -> str | None:
    """The known defect a disagreement of `consumer` at a probe (`got`, the consumer's answer)
    shows, or None."""
    big = '1.7976931348623157e308'
    if consumer == 'f64 fold':
        return 'D1'
    if consumer == 'f64 enclosure':
        if '-0.0' in tokens:
            return 'I1'
        if got is not None and got[0] and (got[4] == math.inf or got[5] == -math.inf):
            return 'I5'
        if tokens[0] == 'pow' and tokens[-1] == big and tokens[1] in ('-2', 'float("-inf")'):
            return 'I2'
        if tokens[0] == 'besselj' and tokens[1] in ('float("inf")', 'float("-inf")'):
            return 'I3'
    if consumer == 'real enclosure' and tokens[0] in ('+', '-') and '5e-324' in tokens \
            and big in tokens:
        return 'I4'
    return None


@functools.lru_cache(maxsize=1)
def disagreements() -> tuple:
    """Every consumer's disagreement with the table at a probe: `(consumer, tokens, got,
    the table's)`. Float64's consumers read bit for bit (F8): the judge (`d_eval`), the mint's
    evaluator (`evaluate_batch`), the engine's fold where it folds the probe to a literal; the
    interval layer's f64 enclosure contains the table's double, or at a range event (F5) the
    true value. The reals' consumers: permissive's fold writes the table's value as the double
    nearest to it where it folds the probe to a literal (P3, P5, one zero); the real enclosure
    contains it."""
    from simplipy import _core
    arity = dict(_contract.ARITY)
    readers = {
        'f64 judge': lambda t: float(_contract.d_eval(_contract.parse(list(t)), {})),
        'f64 mint': lambda t: probe_engine()._core.evaluate_batch(list(t), [], [], 1, [])[0],
        'f64 fold': lambda t: _literal(probe_engine().simplify(list(t), mode='f64'))}
    out = []
    for tokens, real, (f64, in_range) in probe_table():
        for name, read in readers.items():
            try:
                v = read(tokens)
            except Exception as error:
                out.append((name, tokens, repr(error), f64))
                continue
            if v is not None and (_bits(v) != _bits(f64) or (
                    v == 0 and math.copysign(1, v) != math.copysign(1, f64))):
                out.append((name, tokens, v, f64))
        box = _core.table_value_set(list(tokens), arity, True)
        # at a range event the true value is accepted (F5), where the table reads it, else an
        # unbounded finite part in the overflow's direction
        true_inside = not in_range and (
            _contains(box, real) if real is not None else
            box is not None and box[0] and math.isinf(f64)
            and (box[5] if f64 > 0 else -box[4]) == math.inf)
        if not (_contains(box, f64) or true_inside):
            out.append(('f64 enclosure', tokens, box, f64))
        if real is None:
            continue
        folded = _literal(probe_engine().simplify(list(tokens), mode='permissive'))
        if folded is not None and _class(folded) != _class(real):
            out.append(('permissive fold', tokens, folded, _class(real)))
        box = _core.table_value_set(list(tokens), arity, False)
        if not _contains(box, real):
            out.append(('real enclosure', tokens, box, _class(real)))
    return tuple(out)


def test_every_consumer_reads_the_tables_values() -> None:
    """At every probe, every consumer agrees with the table, but for the known defects
    (`DEFECTS`, each checked by its own test below)."""
    unknown = [(c, ' '.join(t), got, want) for c, t, got, want in disagreements()
               if _defect_of(c, t, got) is None]
    REPORT[('assertion 4', 'probes')] = {
        'probes': len(probe_table()),
        'unread by the table': sum(real is None for _, real, _ in probe_table()),
        'known': dict(collections.Counter(_defect_of(c, t, got)
                                          for c, t, got, _ in disagreements()))}
    assert not unknown, unknown


@pytest.mark.parametrize('defect', [pytest.param(d, marks=known(d)) if d in OPEN else d
                                    for d in DEFECTS])
def test_a_known_defect_is_gone(defect) -> None:
    left = [(c, ' '.join(t), got, want) for c, t, got, want in disagreements()
            if _defect_of(c, t, got) == defect]
    assert not left, left[:10]


@pytest.mark.parametrize('expr, infinities', [
    # a zero the layer signs: a literal, a product of two points, the reciprocal of an infinity
    ('inv -0.0', '-'), ('inv * -0.0 3', '-'), ('inv * 0 -1', '-'), ('inv * -0.0 -2', '+'),
    ('inv / 1 float("-inf")', '-'), ('inv / -1 float("-inf")', '+'),
    ('inv / -0.0 float("-inf")', '+'), ('inv inv float("-inf")', '-'), ('inv neg 0', '-'),
    ('pow -0.0 -1', '-'), ('pow -0.0 -2', '+'),
    # a zero of a node that does not compute its sign may carry either
    ('inv atan2 -0.0 1', '+-'), ('inv max 0 -0.0', '+-'), ('inv min -0.0 0', '+-'),
    ('inv pow -0.0 1', '+-'), ('inv besselj 1 -0.0', '+-'), ('inv * -0.0 0.5', '+-'),
])
def test_the_f64_enclosure_holds_the_sign_of_a_zero(expr, infinities) -> None:
    """The reciprocal of a point zero is, in float64's reading, the infinity of each sign the zero
    may carry: the layer signs the zero where it computes float64's sign, and reads either sign
    elsewhere, so the enclosure holds float64's value."""
    from simplipy import _core
    tokens = expr.split()
    value, _ = _core.TableExpr(tokens, [], dict(_contract.ARITY)).eval_f64([])
    box = _core.table_value_set(tokens, dict(_contract.ARITY), True)
    assert _contains(box, value), (box, value)
    assert ('+' if box[1] else '') + ('-' if box[2] else '') == infinities, box


@pytest.mark.parametrize('expr', [
    'inv atan2 -0.0 0', 'inv atan2 -0.0 5e-324', 'inv harmonic -0.0', '/ -1 harmonic -0.0',
    'inv mod -0.0 -1', 'inv mod -0.0 -0.5', 'inv mod 0 -1', 'inv mod 0 -0.5', 'inv mod -1 -1',
    'inv mod -1 -0.5', 'inv mod 1 -1', 'inv mod 1 -0.5', 'inv mod -0.5 -0.5', 'inv mod 2 -1',
    'inv mod 2 -0.5', 'inv mod 1.7976931348623157e308 -1', 'inv mod 1.7976931348623157e308 -0.5',
])
def test_the_f64_enclosure_holds_either_sign_of_a_hulls_zero(expr) -> None:
    """A zero the layer encloses as a hull (an interval that holds 0, not a point) carries no
    sign: the layer writes a hull's zero endpoint as +0.0. Float64 reads either sign there, so the
    reciprocal of such a zero holds both infinities, and the enclosure holds float64's -inf at
    `1/atan2(-0.0, 0)` and `1/mod(1, -1)`."""
    from simplipy import _core
    tokens = expr.split()
    value, _ = _core.TableExpr(tokens, [], dict(_contract.ARITY)).eval_f64([])
    box = _core.table_value_set(tokens, dict(_contract.ARITY), True)
    assert _contains(box, value), (box, value)
    assert box[1] and box[2], box


@pytest.mark.parametrize('expr, infinities', [
    # an even power of either zero is +0, so its negative power is +inf alone
    ('pow mod 1 -1 -2', '+'), ('pow mod 1 -1 -3', '+-'), ('pow atan2 -0.0 0 -1', '+-'),
])
def test_a_negative_power_of_a_hulls_zero_reads_its_parity(expr, infinities) -> None:
    """A negative integer power of a hull's zero: an odd one reads either sign, an even one +inf."""
    from simplipy import _core
    tokens = expr.split()
    value, _ = _core.TableExpr(tokens, [], dict(_contract.ARITY)).eval_f64([])
    box = _core.table_value_set(tokens, dict(_contract.ARITY), True)
    assert _contains(box, value), (box, value)
    assert ('+' if box[1] else '') + ('-' if box[2] else '') == infinities, box


# ------------------------------------------------------------------------------------------
# Assertion 5: the examples of section 1, each with the value the specification gives it
# (`test_differential` holds those of sections 2 to 4).
# ------------------------------------------------------------------------------------------
SECTION_1 = [
    # 1.1, real (V1-V8), which permissive's values are (P1)
    ('neg 0', 'real', '0.0'), ('+ 0.1 0.2', 'real', '0.3'),
    ('pow -8 / 1 3', 'real', 'nan'), ('rootn -8 3', 'real', '-2.0'), ('log -1', 'real', 'nan'),
    ('asin 2', 'real', 'nan'), ('acosh 0', 'real', 'nan'), ('log 0', 'real', '-inf'),
    ('atanh 1', 'real', '+inf'), ('atanh -1', 'real', '-inf'), ('/ 1 0', 'real', 'nan'),
    ('tan / np.pi 2', 'real', 'nan'), ('pow 0 -1', 'real', 'nan'),
    ('exp float("inf")', 'real', '+inf'), ('exp float("-inf")', 'real', '0.0'),
    ('tanh float("inf")', 'real', '1.0'), ('atan float("inf")', 'real', repr(math.pi / 2)),
    ('log float("inf")', 'real', '+inf'), ('/ 1 float("inf")', 'real', '0.0'),
    ('floor float("inf")', 'real', '+inf'), ('sin float("inf")', 'real', 'nan'),
    ('- float("inf") float("inf")', 'real', 'nan'), ('* 0 float("inf")', 'real', 'nan'),
    ('/ float("inf") float("inf")', 'real', 'nan'), ('/ 0 0', 'real', 'nan'),
    ('pow 1 float("inf")', 'real', 'nan'), ('pow 1 float("-inf")', 'real', 'nan'),
    ('pow float("inf") 0', 'real', 'nan'), ('pow float("-inf") 0', 'real', '1.0'),
    ('pow 0 0', 'real', '1.0'), ('mod 0 float("inf")', 'real', 'nan'),
    ('exp tan asin 1', 'real', 'nan'), ('atanh floor 1.5', 'real', '+inf'),
    ('pow 0 -2', 'real', 'nan'), ('pow 0 -0.5', 'real', 'nan'), ('mod 1 0', 'real', 'nan'),
    ('rootn 2 float("inf")', 'real', 'nan'),
    # 1.3: a fully numeric expression takes its table value (P3, P5)
    ('floor tanh 40', 'real', '0.0'), ('mod 0 atanh 1', 'real', 'nan'),
    ('- log -1 log -1', 'real', 'nan'),
    # 1.2, f64 (F2-F4, F7)
    ('0.1', 'f64', 0.1), ('np.pi', 'f64', 3.141592653589793),
    ('+ 0.1 0.2', 'f64', 0.30000000000000004), ('sin np.pi', 'f64', 1.2246467991473532e-16),
    ('max -0.0 0', 'f64', 0.0), ('max 0 -0.0', 'f64', 0.0),
    ('pow float("-inf") 0.5', 'f64', math.nan), ('/ 1 0', 'f64', math.inf),
    ('/ 1 -0.0', 'f64', -math.inf), ('log 0', 'f64', -math.inf), ('atanh 1', 'f64', math.inf),
    ('tan asin 1.0', 'f64', 1.633123935319537e16), ('tanh 20', 'f64', 1.0),
    ('atanh tanh 20', 'f64', math.inf), ('inv ceil -0.5', 'f64', -math.inf),
    ('inv + ceil -0.5 0', 'f64', math.inf),
]


@pytest.mark.parametrize('expr, mode, expected', SECTION_1)
def test_the_table_gives_section_1_its_values(expr, mode, expected) -> None:
    """The table's value in the mode; in the reals, permissive's fold too where it folds to a
    literal, as the double nearest to the value (P3, P5). (The f64 fold is `D1`'s.)"""
    from simplipy import _core
    expr_ = _core.TableExpr(expr.split(), [], dict(_contract.ARITY))
    if mode == 'real':
        spec, unresolved, *_ = expr_.eval_real([], 60)
        assert not unresolved
        assert _class(_contract._from_spec(spec)) == expected
        folded = _literal(engine().simplify(expr.split(), mode='permissive'))
        assert folded is None or _class(folded) == expected, folded
    else:
        value, _ = expr_.eval_f64([])
        assert _bits(value) == _bits(expected), value
        assert value != 0 or math.copysign(1, value) == math.copysign(1, expected), value


@pytest.mark.parametrize('a, b, agree', [(0.30000000000000004, 0.3, True),
                                         (1.2246467991473532e-16, 0.0, False),
                                         (0.0, -0.0, True), (math.nan, math.nan, True),
                                         (math.inf, math.inf, True), (math.inf, -math.inf, False)])
def test_f64_agreement_is_eight_units(a, b, agree) -> None:
    """F6: two doubles agree within 8 units in the last place, the two zeros as one value, both
    undefined, or the same infinity."""
    from simplipy import _core
    assert (_core.table_f64_agree(a, b) == 'eq') is agree, _core.table_f64_agree(a, b)


@pytest.mark.parametrize('lhs, rhs, mode, expected', [
    # F5: at x = 800 the evaluator's exp(x)/exp(x) is inf/inf, and 1 is the true value
    ('/ exp x0 exp x0', '1', 'f64', True),
    # P5: `mod(0, _0)` has no value only at 0 and +-inf, where permissive may give one
    ('mod 0 _0', '0', 'permissive', True),
])
def test_section_1s_rules(lhs, rhs, mode, expected) -> None:
    record = judge([(tuple(lhs.split()), tuple(rhs.split()))])
    assert holds(record[(tuple(lhs.split()), tuple(rhs.split()))], mode) is expected, record


def test_f64_keeps_the_zero_term_a_negative_zero_needs() -> None:
    """F7: at x = -0.5, `ceil(x)` is -0, so `1/ceil(x)` is -inf and `1/(ceil(x) + 0)` is +inf:
    f64 keeps `ceil(x) + 0`."""
    assert engine().simplify('+ ceil x0 0'.split(), mode='f64') == '+ ceil x0 0'.split()


@known('D1')
def test_f64_folds_a_numeric_expression_in_written_order() -> None:
    """F8: `0.1 + 0.2` becomes 0.30000000000000004, never 0.3."""
    assert engine().simplify('+ 0.1 0.2'.split(), mode='f64') == ['0.30000000000000004']


# ------------------------------------------------------------------------------------------
# Assertion 6: the judge gives every pair a definite answer, with how it was reached.
# ------------------------------------------------------------------------------------------
def test_the_judge_answers_every_pair_of_the_workload() -> None:
    """Every pair the workload put to the judge has, in each mode, an answer and its basis:
    proved, evidence or refused (3.4). Runs last, over every pair the tests above judged."""
    assert _JUDGED
    unanswered = [(pair, rec) for pair, rec in _JUDGED.items()
                  if rec['verdict'] is None or any(
                      rec['basis'].get(m) not in ('proved', 'evidence', 'refused')
                      for m in MODES)]
    REPORT[('assertion 6', 'both')] = {'pairs': len(_JUDGED),
                                       'by basis': {m: dict(collections.Counter(
                                           rec['basis'].get(m) for rec in _JUDGED.values()))
                                           for m in MODES}}
    report = os.environ.get('SIMPLIPY_WORKLOAD_REPORT')
    if report:
        import json
        with open(report, 'w') as fh:
            json.dump({f'{a} [{m}]': v for (a, m), v in REPORT.items()}, fh, indent=1)
    assert not unanswered, unanswered[:20]
