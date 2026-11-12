"""A candidate target's class is the mode's reading of it, each ground part written as the
engine reads it (`min(log(1), x0)` is `min(0, x0)`). The candidate library keeps one candidate
per class, written as that reading, and the judge loop's exclusion of a refused target excludes
its class, so a value the judge refuses costs one round, not one per spelling. Each case here is
a value with many spellings that the judge refuses."""
import json
import logging
import os

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.mining import RuleMiner, judge_tiers

_UNARY = {'neg': 'simplipy.operators.neg', 'log': 'np.log', 'sin': 'np.sin', 'tan': 'np.tan',
          'asin': 'np.arcsin', 'atan': 'np.arctan', 'sinh': 'np.sinh', 'tanh': 'np.tanh',
          'erfc': 'simplipy.operators.erfc', 'abs': 'np.abs'}
_OPERATORS = {
    'min': {'realization': 'simplipy.operators.min', 'alias': [], 'arity': 2, 'precedence': 3,
            'commutative': False},
    '-': {'realization': '-', 'alias': [], 'inverse': '+', 'arity': 2, 'precedence': 1,
          'commutative': False},
    '/': {'realization': '/', 'alias': [], 'inverse': '*', 'arity': 2, 'precedence': 2,
          'commutative': False},
    **{name: {'realization': realization, 'alias': [], 'arity': 1,
              'precedence': 2.5 if name == 'neg' else 3, 'commutative': False}
       for name, realization in _UNARY.items()}}
#: the manual proposal `(x0 - abs(x0))/2 -> min(x0, 0)`
_PROPOSAL = ['/', '-', 'x0', 'abs', 'x0', '2']
#: spellings of the order 0 of a Bessel function
_ZERO_ORDERS = [['0'], ['neg', '0'], ['log', '1'], ['sin', '0'], ['tan', '0'], ['acos', '1'],
                ['acosh', '1'], ['asin', '0'], ['atan', '0']]


def _bessel_engine() -> SimpliPyEngine:
    config = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4',
                          'config.yaml')
    operators = dict(yaml.safe_load(open(config))['operators'])
    operators['erfc'] = {'realization': 'simplipy.operators.erfc', 'alias': [], 'arity': 1,
                         'precedence': 2, 'commutative': False}
    operators['bessely'] = {'realization': 'simplipy.operators.bessely', 'alias': ['yv'],
                            'arity': 2, 'precedence': 2, 'commutative': False}
    return SimpliPyEngine(operators=operators, rules=[])


def _library(engine: SimpliPyEngine, candidates: list) -> object:
    rows = RuleMiner(engine)._mining_sample_x(512, 2, np.random.default_rng(7))
    return engine._core.build_candidate_library(
        sorted(candidates, key=lambda t: (len(t), t)), ['x0', 'x1'],
        rows.flatten(order='C').tolist(), rows.shape[0], rule_mode='default')


@pytest.mark.parametrize('mode', ['default', 'permissive'])
def test_a_candidates_class_reads_its_ground_parts(mode) -> None:
    core = _bessel_engine()._core

    def read(*tokens):
        return core.mine_candidate_class(list(tokens), rule_mode=mode)

    zero = read('bessely', '0', 'x1')
    assert zero == (['bessely', '0', 'x1'], ['bessely', '0', 'x1'])
    for order in _ZERO_ORDERS[2:]:
        assert read('bessely', *order, 'x1') == zero, order
    assert read('bessely', 'inv', '2', 'x1') == read('bessely', '/', '1', '2', 'x1')
    assert read('bessely', 'harmonic', '2', 'x1') == read('bessely', '/', '3', '2', 'x1')
    assert read('bessely', 'inv', '2', 'x1') != read('bessely', 'harmonic', '2', 'x1')
    # a value whose literal is dearer than its expression keeps the expression, and a
    # candidate without a ground part keeps its spelling
    assert read('bessely', 'erfc', '10', 'x1')[0] == ['bessely', 'erfc', '10', 'x1']
    assert read('-', 'x1', 'x0')[0] == ['-', 'x1', 'x0']


def test_the_library_keeps_one_candidate_per_class_written_as_its_reading() -> None:
    """The nine spellings of the order 0 are two classes in f64, 0 and -0 (strict reads the
    sign); a spelling alone is written as its reading."""
    engine = _bessel_engine()
    assert _library(engine, [['bessely', *order, 'x1'] for order in _ZERO_ORDERS]) \
        .n_candidates == 2
    library = _library(engine, [['bessely', 'log', '1', 'x1']])
    source = ['bessely', 'erfc', '10', 'x1']
    assert engine._core.mine_one_length([source], library, None, 16, 16, 7, 1e-11, 1e-12, None,
                                        True, rule_mode='default') == \
        [(0, source, ['bessely', '0', 'x1'])]


def test_a_refused_value_costs_the_judge_loop_one_round() -> None:
    """A length-4 f64 mine of the 0.15 vocabulary offered Y_0 in eight spellings to
    `bessely(erfc(10), x1)` and `bessely(erfc(8), x1)` (their order is 2.1e-45 and 2.2e-29), and
    the judge refused each: an exclusion now removes the class of the refused spelling, so each
    source is refused the two values the library holds and gets no rule."""
    engine = _bessel_engine()
    library = _library(engine, [['bessely', *order, 'x1'] for order in _ZERO_ORDERS])

    def mine(batch, indices, exclusions):
        return engine._core.mine_one_length(batch, library, None, 16, 16, 7, 1e-11, 1e-12, None,
                                            True, rule_mode='default', indices=indices,
                                            exclusions=exclusions)

    sources = [['bessely', 'erfc', '10', 'x1'], ['bessely', 'erfc', '8', 'x1']]
    assert [t for _, _, t in mine(sources[:1], None, {0: [['bessely', 'log', '1', 'x1']]})] \
        == [['bessely', '-0.0', 'x1']], 'an excluded spelling excludes its class'
    found, stats = RuleMiner(engine)._mine_length(
        sources, mine, 'default', lambda pairs: judge_tiers(pairs, processes=1), {},
        ['x0', 'x1'])
    assert found == [] and stats['given_up'] == []
    assert stats['excluded'] == [[source, ['bessely', zero, 'x1']] for source in sources
                                 for zero in ('-0.0', '0')]


def test_a_value_of_many_spellings_is_no_longer_given_up(tmp_path, caplog) -> None:
    """`min(x0, erfc(10))` reads as min(0, x0) on the mine's rows (erfc(10) is 2.1e-45), and the
    judge refuses min(0, x0) in f64; so it does for the proposal `(x0 - abs(x0))/2 -> min(x0, 0)`.
    Eight spellings of min(0, x0) used to exhaust the judge loop; now the source is refused 0
    and -0, the proposal 0 and the dearer `min(erfc(10), x0)`, and neither is given up."""
    out = str(tmp_path / 'mined.json')
    with caplog.at_level(logging.WARNING, logger='simplipy.mining'):
        SimpliPyEngine(operators=_OPERATORS, rules=[]).find_rules(
            max_source_pattern_length=4, max_target_pattern_length=4, dummy_variables=1,
            extra_internal_terms=['0', '1', '10'], X=256, seed=7, promote_sorts=False,
            output_file=out, modes=['f64'], operator_max_length={'-': 1, '/': 1, 'abs': 1},
            proposals=[{'source': _PROPOSAL, 'target': ['min', 'x0', '0']}])
    side = json.load(open(out + '.provenance.json'))
    assert not side.get('given_up') and side['per_length']['4']['given_up'] == 0
    assert not [r for r in caplog.records if 'gave up on' in r.getMessage()]
    source = ['min', 'x0', 'erfc', '10']
    assert [t for s, t in side['per_length']['4']['excluded'] if s == source] == \
        [['min', '-0.0', 'x0'], ['min', '0', 'x0']]
    assert [t for s, t in side['proposals']['judge_loop']['excluded'] if s == _PROPOSAL] == \
        [['min', '0', 'x0'], ['min', 'erfc', '10', 'x0']]
    assert [t['verdict'] for t in side['proposals']['trail']] == ['rejected']
    shipped = {tuple(lhs) for lhs, _ in json.load(open(out))}
    assert ('min', '?0', 'erfc', '10') not in shipped and tuple(_PROPOSAL) not in shipped
