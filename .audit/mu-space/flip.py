import sys, json, time
import numpy as np
sys.path.insert(0,'/home/psaegert/Projects/simplipy/src')
from simplipy import SimpliPyEngine
from simplipy.mining import RuleMiner
eng = SimpliPyEngine.from_config('/home/psaegert/Projects/simplipy/assets/engines/acj-4-3/config.yaml')
miner = RuleMiner(eng); core = eng._core
core.set_rules([])
rng = np.random.default_rng(42); X = miner._mining_sample_x(1024, 2, rng)
xf = X.flatten(order='C').tolist()
allc = json.load(open('/tmp/mu/all_candidates.json'))
dummies={'x0','x1'}
libF = core.build_candidate_library(allc, ['x0','x1'], xf, X.shape[0], fold_filter=True)
old = [c for c in allc if len(c) < 2 or (dummies & set(c))]
libOld = core.build_candidate_library(old, ['x0','x1'], xf, X.shape[0], fold_filter=True)
print('libF', libF.n_candidates, 'libOld', libOld.n_candidates, flush=True)
probes = [
  ['+','1','*','10','pow','np.pi','0'],
  ['*','2','+','1','*','5','pow','np.pi','0'],
  ['+','np.pi','-','1','np.pi'],
  ['*','np.e','pow','np.pi','0'],
]
for s in probes:
    try: m = core.ac_simplify(s, 48)
    except Exception as e: print(s,'SKIP',e); continue
    try: mu = core.ac_complexity(m)
    except Exception: mu = None
    for nm, L in (('NEW(fold-filter)', libF), ('OLD(all var-free composites dropped)', libOld)):
        t=time.time()
        try: r = core.find_rule_lib(s, len(s), None, L, 16,16,42,1e-11,1e-12,None, None)
        except Exception as e: r = f'ERR {e}'
        print(f'{" ".join(s):34s} mark=None [{nm:36s}] -> {r} ({time.time()-t:.1f}s)', flush=True)
    for nm, L in (('NEW+mark', libF), ('OLD+mark', libOld)):
        try: r = core.find_rule_lib(s, len(s), None, L, 16,16,42,1e-11,1e-12,None, m)
        except Exception as e: r = f'ERR {e}'
        print(f'{" ".join(s):34s} mark={" ".join(m)} mu={mu} [{nm}] -> {r}', flush=True)
