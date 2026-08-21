import sys, json, collections
sys.path.insert(0,'/home/psaegert/Projects/simplipy/src')
from simplipy import SimpliPyEngine
eng = SimpliPyEngine.from_config('/home/psaegert/Projects/simplipy/assets/engines/acj-4-3/config.yaml')
core = eng._core
allc = [c for c in json.load(open('/tmp/mu/all_candidates.json')) if len(c) <= 3]
d = {'x0','x1'}
adm = {tuple(a['tokens']) for a in json.load(open('/tmp/mu/admitted.json'))}
kept_new = [c for c in allc if len(c) < 2 or (d & set(c)) or tuple(c) in adm]
kept_old = [c for c in allc if len(c) < 2 or (d & set(c))]
print('length<=3 universe:', len(allc))
print('  library under the shipped (old) filter:', len(kept_old))
print('  library under the new fold filter     :', len(kept_new),
      f'  (+{len(kept_new)-len(kept_old)} var-free composites)')
mus = [core.ac_complexity(c) for c in kept_new]
cb = [m for c,m in zip(kept_new,mus) if '<constant>' in c]
print('  max mu =', max(mus), ' max_const_mu =', max(cb), ' break reachable?', max(mus) > max(cb))
alpha = {'-1','-10','-2','-3','-4','-5','-6','-7','-8','-9','0','1','10','2','3','4','5','6','7','8','9','<constant>','float("-inf")','float("inf")','float("nan")','np.e','np.pi'}
cat = collections.Counter()
byval = collections.defaultdict(list)
for a in json.load(open('/tmp/mu/admitted_mu.json')):
    if len(a['tokens']) > 3: continue
    k = ' '.join(a['key'])
    if k == '<constant>': c='C'
    elif k in ('float("nan")','float("inf")','float("-inf")'): c='D'
    elif k in alpha: c='A'
    else: c='B'; byval[k].append(' '.join(a['tokens']))
    cat[c]+=1
print('  admitted (len<=3) by category:', dict(cat))
print('  category B (no single-token spelling) distinct values:', len(byval))
for k in sorted(byval)[:20]: print('    ', k, '<-', byval[k][:4])
