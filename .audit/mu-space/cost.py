import sys, json, time, random
import numpy as np
sys.path.insert(0,'/home/psaegert/Projects/simplipy/src')
from simplipy import SimpliPyEngine
from simplipy.mining import RuleMiner
from simplipy.utils import enumerate_expressions
eng = SimpliPyEngine.from_config('/home/psaegert/Projects/simplipy/assets/engines/acj-4-3/config.yaml')
miner = RuleMiner(eng); core = eng._core; core.set_rules([])
rng = np.random.default_rng(42); X = miner._mining_sample_x(1024, 2, rng)
xf = X.flatten(order='C').tolist()
allc = json.load(open('/tmp/mu/all_candidates.json'))
dummies={'x0','x1'}
libF = core.build_candidate_library(allc, ['x0','x1'], xf, X.shape[0], fold_filter=True)
old  = core.build_candidate_library([c for c in allc if len(c)<2 or (dummies & set(c))],
                                    ['x0','x1'], xf, X.shape[0], fold_filter=True)
extra = ['<constant>','(-10)','(-9)','(-8)','(-7)','(-6)','(-5)','(-4)','(-3)','(-2)','(-1)','0','1','2','3','4','5','6','7','8','9','10','np.e','np.pi','float("inf")','float("-inf")','float("nan")']
ex = enumerate_expressions(['x0','x1']+extra, dict(sorted(eng.operator_arity.items(), key=lambda x:x[1])), 3)
vf = [list(e) for e in sorted(ex[3]) if not (dummies & set(e))]
random.Random(0).shuffle(vf)
sample = vf[:60]
tot = {'new':0.0,'old':0.0}; diff=0
for s in sample:
    m = core.ac_simplify(s, 48)
    if len(m) == 1: continue
    res={}
    for nm,L in (('new',libF),('old',old)):
        t=time.time(); res[nm]=core.find_rule_lib(s,len(s),None,L,16,16,42,1e-11,1e-12,None,m); tot[nm]+=time.time()-t
    if res['new'] != res['old']:
        diff+=1
        print('DIFF', ' '.join(s), '| mark', ' '.join(m), '| new', res['new'], '| old', res['old'], flush=True)
print('var-free length-3 sources timed:', tot, 'differing:', diff, flush=True)
