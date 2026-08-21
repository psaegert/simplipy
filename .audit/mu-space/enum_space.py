import sys, json, time
sys.path.insert(0, '/home/psaegert/Projects/simplipy/src')
import simplipy
from simplipy.utils import enumerate_expressions, count_expressions
from simplipy import SimpliPyEngine

eng = SimpliPyEngine.from_config('/home/psaegert/Projects/simplipy/assets/engines/acj-4-3/config.yaml')
print('engine ok', type(eng))
core = eng._core
extra = ['<constant>', '(-10)', '(-9)', '(-8)', '(-7)', '(-6)', '(-5)', '(-4)', '(-3)', '(-2)', '(-1)', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
         'np.e', 'np.pi', 'float("inf")', 'float("-inf")', 'float("nan")']
dummies = ['x0','x1']
leaf_nodes = dummies + extra
non_leaf = dict(sorted(eng.operator_arity.items(), key=lambda x: x[1]))
print('n leaves', len(leaf_nodes), 'ops', non_leaf)
counts = count_expressions(len(leaf_nodes), non_leaf, 4)
print('counts', counts)
t=time.time()
exprs = enumerate_expressions(leaf_nodes, non_leaf, 4)
print('enumerated', {k: len(v) for k,v in exprs.items()}, time.time()-t)
allc = [list(e) for L in sorted(exprs) for e in sorted(exprs[L])]
json.dump(allc, open('/tmp/mu/all_candidates.json','w'))
print('total', len(allc))
