import sys, json, time, collections
sys.path.insert(0, '/home/psaegert/Projects/simplipy/src')
from simplipy import SimpliPyEngine
eng = SimpliPyEngine.from_config('/home/psaegert/Projects/simplipy/assets/engines/acj-4-3/config.yaml')
core = eng._core
allc = json.load(open('/tmp/mu/all_candidates.json'))
dummies = {'x0','x1'}
varfree = [c for c in allc if len(c) >= 2 and not (dummies & set(c))]
print('var-free len>=2:', len(varfree))
t = time.time()
keys = core.ac_canonical_keys(varfree)
print('keyed in', time.time()-t)
admitted = [(c,k) for c,k in zip(varfree, keys) if k is not None and len(k) == 1]
unparse = [c for c,k in zip(varfree, keys) if k is None]
print('admitted (fold to single leaf):', len(admitted))
print('unparseable:', len(unparse))
json.dump([{'tokens':c,'key':k} for c,k in admitted], open('/tmp/mu/admitted.json','w'))
byleaf = collections.Counter(k[0] for c,k in admitted)
print('distinct leaves:', len(byleaf))
for leaf, n in byleaf.most_common():
    print(f'  {leaf!r}: {n}')
bylen = collections.Counter(len(c) for c,k in admitted)
print('by length', dict(sorted(bylen.items())))
