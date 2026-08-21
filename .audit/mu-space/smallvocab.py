import sys, json
sys.path.insert(0,'/home/psaegert/Projects/simplipy/src')
from simplipy import SimpliPyEngine
from simplipy.utils import enumerate_expressions
eng = SimpliPyEngine.from_config('/home/psaegert/Projects/simplipy/assets/engines/acj-4-3/config.yaml')
core = eng._core
for extra, dummies, L in ((['0','1','<constant>'], ['x0'], 3),
                          (['0','1','<constant>'], ['x0'], 4)):
    leaves = dummies + extra
    nl = dict(sorted(eng.operator_arity.items(), key=lambda x:x[1]))
    ex = enumerate_expressions(leaves, nl, L)
    cands = [list(e) for k in sorted(ex) for e in sorted(ex[k])]
    keys = core.ac_canonical_keys(cands)
    dset = set(dummies)
    kept=[]
    for c,k in zip(cands,keys):
        vf = not (dset & set(c))
        if len(c)>=2 and vf and not (k is not None and len(k)==1):
            continue
        kept.append(c)
    mus = [core.ac_complexity(c) for c in kept]
    cb = [(c,m) for c,m in zip(kept,mus) if '<constant>' in c]
    print(f'vocab {extra} L={L}: universe {len(cands)}, library {len(kept)}, const-bearing {len(cb)}')
    print(f'   max mu overall = {max(mus)}   max_const_mu = {max(m for _,m in cb)}  -> break reachable? {max(mus) > max(m for _,m in cb)}')
