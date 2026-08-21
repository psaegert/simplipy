import sys, json, time
import numpy as np
sys.path.insert(0,'/home/psaegert/Projects/simplipy/src')
from simplipy import SimpliPyEngine
from simplipy.mining import RuleMiner
eng = SimpliPyEngine.from_config('/home/psaegert/Projects/simplipy/assets/engines/acj-4-3/config.yaml')
miner = RuleMiner(eng); core = eng._core
rng = np.random.default_rng(42); X = miner._mining_sample_x(1024, 2, rng)
xf = X.flatten(order='C').tolist()
allc = json.load(open('/tmp/mu/all_candidates.json'))
libmu = {tuple(c): m for c, m in json.load(open('/tmp/mu/library_mu.json'))}
libA = core.build_candidate_library(allc, ['x0','x1'], xf, X.shape[0], fold_filter=True)

CAP = 24000
drop = {c for c, m in libmu.items() if '<constant>' in c and m > CAP and len(c) > 1}
print('dropping', len(drop), 'const-bearing candidates with mu >', CAP, flush=True)
red = [c for c in allc if tuple(c) not in drop]
libB = core.build_candidate_library(red, ['x0','x1'], xf, X.shape[0], fold_filter=True)
print('libA', libA.n_candidates, 'libB', libB.n_candidates, flush=True)

for s, m in [ (['*','np.pi','x0'], ['<mul>','np.pi','x0','</mul>']),
              (['tanh','tanh','x0'], ['tanh','tanh','x0']),
              (['pow','x0','2'], ['pow','x0','2']) ]:
    for name, L in (('A(full)', libA), ('B(bounded)', libB)):
        t=time.time(); r = core.find_rule_lib(s, len(s), None, L, 16,16,42,1e-11,1e-12,None, m); dt=time.time()-t
        print(f'{" ".join(s):18s} {name:11s} -> {r}  {dt*1000:8.1f} ms', flush=True)
