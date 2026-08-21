import sys, json, collections
sys.path.insert(0,'/home/psaegert/Projects/simplipy/src')
from simplipy import SimpliPyEngine
eng = SimpliPyEngine.from_config('/home/psaegert/Projects/simplipy/assets/engines/acj-4-3/config.yaml')
core = eng._core
alphabet = {'<constant>','(-10)','(-9)','(-8)','(-7)','(-6)','(-5)','(-4)','(-3)','(-2)','(-1)','0','1','2','3','4','5','6','7','8','9','10','np.e','np.pi','float("inf")','float("-inf")','float("nan")'}
# canonical keys of the alphabet leaves
leafkeys = {}
for a in sorted(alphabet):
    k = core.ac_canonical_keys([[a]])[0]
    leafkeys[' '.join(k)] = a
print('alphabet leaf keys:', leafkeys)
adm = json.load(open('/tmp/mu/admitted_mu.json'))
inA, notA = [], []
for r in adm:
    kk = ' '.join(r['key'])
    (inA if kk in leafkeys else notA).append(r)
print('admitted total', len(adm))
print('  folding to an ALPHABET leaf:', len(inA))
print('  folding to a leaf NOT in the alphabet:', len(notA))
byk = collections.Counter(' '.join(r['key']) for r in notA)
print('  distinct non-alphabet leaves:', len(byk))
ex = {}
for r in notA:
    k=' '.join(r['key'])
    ex.setdefault(k, []).append((r['mu'], ' '.join(r['tokens'])))
for k in sorted(ex, key=lambda k:-len(ex[k]))[:25]:
    print(f"   {k:>16}  n={len(ex[k]):4d}  e.g. {ex[k][0]}")
json.dump(notA, open('/tmp/mu/notalpha.json','w'))
# the <constant>-folding ones
cst = [r for r in adm if r['key']==['<constant>']]
print('\nfold to <constant>:', len(cst))
for r in cst[:25]: print('   ', r['mu'], ' '.join(r['tokens']))
