"""The judge loop gives up on a source or a proposal whose picks the acceptance refuses
`_JUDGE_ROUNDS_MAX` times: it gets no rule, the mine goes on, and the incident is recorded (the
sidecar's `given_up`, one warning line). Each case here is a value with many spellings, every one
of which the judge refuses."""
import json
import logging
import os

import numpy as np
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
#: `min(0, x0)` and the spellings the length-4 library holds of it, in scan order: a ground part
#: the mode reads as 0 (`log(1)`, `sin(0)`, ...) is its own class to the scan
_ZERO_SPELLINGS = [['min', '0', 'x0'], ['min', 'neg', '0', 'x0'], ['min', 'log', '1', 'x0'],
                   ['min', 'sin', '0', 'x0'], ['min', 'tan', '0', 'x0'],
                   ['min', 'asin', '0', 'x0'], ['min', 'atan', '0', 'x0'],
                   ['min', 'sinh', '0', 'x0']]
#: the manual proposal its loop gave up on: `(x0 - abs(x0))/2 -> min(x0, 0)`
_PROPOSAL = ['/', '-', 'x0', 'abs', 'x0', '2']


def test_a_value_of_many_spellings_is_given_up_and_the_mine_goes_on(tmp_path, caplog) -> None:
    """`min(x0, erfc(10))` reads as min(0, x0) on the mine's rows (erfc(10) is 2.1e-45), and the
    judge refuses min(0, x0) in f64 in every spelling the scan offers; so does it for the
    proposal `(x0 - abs(x0))/2 -> min(x0, 0)`. Each is given up on after eight refusals and
    gets no rule; the mine finishes and records both."""
    out = str(tmp_path / 'mined.json')
    with caplog.at_level(logging.WARNING, logger='simplipy.mining'):
        SimpliPyEngine(operators=_OPERATORS, rules=[]).find_rules(
            max_source_pattern_length=4, max_target_pattern_length=4, dummy_variables=1,
            extra_internal_terms=['0', '1', '10'], X=256, seed=7, promote_sorts=False,
            output_file=out, modes=['f64'], operator_max_length={'-': 1, '/': 1, 'abs': 1},
            proposals=[{'source': _PROPOSAL, 'target': ['min', 'x0', '0']}])
    side = json.load(open(out + '.provenance.json'))
    scan, proposal = side['given_up']
    assert (scan['mode'], scan['length'], scan['stage'], scan['source']) == \
        ('f64', 4, 'scan', ['min', 'x0', 'erfc', '10'])
    assert [r['target'] for r in scan['refused']] == _ZERO_SPELLINGS
    assert (proposal['mode'], proposal['length'], proposal['stage'], proposal['source']) == \
        ('f64', 6, 'proposals', _PROPOSAL)
    assert len(proposal['refused']) == 8
    for refusal in scan['refused'] + proposal['refused']:
        assert refusal['by'] == 'judge' and refusal['reason'], refusal
    assert side['per_length']['4']['given_up'] == 1
    assert [t['verdict'] for t in side['proposals']['trail']] == ['rejected']
    shipped = {tuple(lhs) for lhs, _ in json.load(open(out))}
    assert ('min', '?0', 'erfc', '10') not in shipped and tuple(_PROPOSAL) not in shipped
    lines = [r.getMessage() for r in caplog.records if 'gave up on' in r.getMessage()]
    assert len(lines) == 2 and 'min x0 erfc 10 (f64, length 4, scan)' in lines[0], lines


def test_the_bessely_sources_of_a_length_4_mine_are_given_up() -> None:
    """A length-4 f64 mine of the 0.15 vocabulary gave up on `bessely(erfc(10), x1)` and
    `bessely(erfc(8), x1)`: their order is 2.1e-45 (2.2e-29), so the scan accepts Y_0 in eight
    spellings, and the judge refuses each. The loop gives each up, in that order."""
    config = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4',
                          'config.yaml')
    operators = dict(yaml.safe_load(open(config))['operators'])
    operators['erfc'] = {'realization': 'simplipy.operators.erfc', 'alias': [], 'arity': 1,
                         'precedence': 2, 'commutative': False}
    operators['bessely'] = {'realization': 'simplipy.operators.bessely', 'alias': ['yv'],
                            'arity': 2, 'precedence': 2, 'commutative': False}
    engine = SimpliPyEngine(operators=operators, rules=[])
    orders = [['0'], ['neg', '0'], ['log', '1'], ['sin', '0'], ['tan', '0'], ['acos', '1'],
              ['acosh', '1'], ['asin', '0'], ['atan', '0']]
    library_rows = RuleMiner(engine)._mining_sample_x(512, 2, np.random.default_rng(7))
    library = engine._core.build_candidate_library(
        sorted([['bessely', *order, 'x1'] for order in orders], key=lambda t: (len(t), t)),
        ['x0', 'x1'], library_rows.flatten(order='C').tolist(), library_rows.shape[0],
        rule_mode='default')

    def mine(batch, indices, exclusions):
        return engine._core.mine_one_length(batch, library, None, 16, 16, 7, 1e-11, 1e-12, None,
                                            True, rule_mode='default', indices=indices,
                                            exclusions=exclusions)

    sources = [['bessely', 'erfc', '10', 'x1'], ['bessely', 'erfc', '8', 'x1']]
    found, stats = RuleMiner(engine)._mine_length(
        sources, mine, 'default', lambda pairs: judge_tiers(pairs, processes=1), {},
        ['x0', 'x1'])
    assert found == []
    assert [g['source'] for g in stats['given_up']] == sources
    for given_up in stats['given_up']:
        assert [r['target'] for r in given_up['refused']] == \
            [['bessely', *order, 'x1'] for order in orders[:8]]
        assert {r['by'] for r in given_up['refused']} == {'judge'}, given_up
