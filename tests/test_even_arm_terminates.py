"""An even function reads its argument's sign-blind representative, and the arm that moves it there
terminates: it re-enters the constructor on a strictly cheaper argument, or on one it does not move
again. NaN is no negatable argument (a negation leaves it as it is), so an odd head of NaN is no
sign site, and a sign the arm drops from a product stays dropped. Every spelling to length 5 over
the operators' unary and binary heads around the special leaves builds in both modes, and
`abs(sin(nan)/(-inf))` and its kin in every entry.

The work runs in a child process, so a regression fails the test instead of the test run."""
import json
import os
import subprocess
import sys

import yaml

from test_new_heads import ACJ2_4
from test_scipy_heads import NEW

EVEN = ('abs', 'cos', 'cosh')
ODD = ('asin', 'asinh', 'atan', 'atanh', 'erf', 'sin', 'sinh', 'tan', 'tanh')
LEAVES = ('float("nan")', 'float("inf")', 'float("-inf")', '0', '-0.0', 'x0')

CHILD = r'''
import json, sys, warnings
warnings.filterwarnings('ignore')
from simplipy import SimpliPyEngine
spec = json.load(open(sys.argv[1]))
engine = SimpliPyEngine(operators=spec['operators'], rules=[], rules_permissive=[])
core = engine._core
arity = {name: op['arity'] for name, op in spec['operators'].items() if op['arity'] in (1, 2)}
leaves = spec['leaves']

def spellings(length, memo={}):
    """Every prefix spelling of `length` tokens over the heads and the leaves."""
    if length not in memo:
        if length == 1:
            out = [(t,) for t in leaves]
        else:
            out = []
            for head, k in arity.items():
                if k == 1:
                    out.extend((head,) + s for s in spellings(length - 1))
                else:
                    for i in range(1, length - 1):
                        for left in spellings(i):
                            out.extend((head,) + left + right for right in spellings(length - 1 - i))
        memo[length] = out
    return memo[length]

counts = {}
for toks in spec['family']:
    for mode, rule_mode in (('f64', 'default'), ('permissive', 'permissive')):
        engine.simplify(toks, mode=mode)
        core.mine_sources([toks], rule_mode=rule_mode)
        core.mine_mark(toks, rule_mode=rule_mode)
for length in range(1, spec['max_length'] + 1):
    batch = [list(s) for s in spellings(length)]
    for rule_mode in ('default', 'permissive'):
        for i in range(0, len(batch), 20000):
            core.mine_sources(batch[i:i + 20000], rule_mode=rule_mode)
    counts[length] = len(batch)
print(json.dumps(counts))
'''


def _operators() -> dict:
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    for name, arity in NEW.items():
        ops[name] = {'realization': f'simplipy.operators.{name}', 'alias': [], 'arity': arity,
                     'commutative': False}
    return ops


def _run(tmp_path, family: list, max_length: int) -> dict:
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'operators': _operators(), 'leaves': list(LEAVES),
                                'family': family, 'max_length': max_length}))
    done = subprocess.run([sys.executable, '-c', CHILD, str(spec)], capture_output=True,
                          text=True, env={**os.environ, 'PYTHONWARNINGS': 'ignore'})
    assert done.returncode == 0, (done.returncode, done.stderr[-2000:])
    return {int(k): v for k, v in json.loads(done.stdout.strip().splitlines()[-1]).items()}


def test_an_even_head_over_an_odd_head_of_nan_over_minus_inf_builds_in_every_entry(tmp_path) -> None:
    family = [[even, '/', odd, 'float("nan")', 'float("-inf")'] for even in EVEN for odd in ODD]
    assert len(family) == 27
    _run(tmp_path, family, 0)


def test_every_short_spelling_around_the_special_leaves_builds(tmp_path) -> None:
    counts = _run(tmp_path, [], 5)
    # 22 unary and 14 binary heads around 6 leaves
    assert counts == {1: 6, 2: 132, 3: 3408, 4: 97152, 5: 2953824}
