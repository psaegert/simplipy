"""REPRODUCTION: rules.json (the f64 set) ships 16 rows that the deployed f64
evaluator does not realise at ordinary constant bindings.

Run:  .venv/bin/python .audit/REPRO_f64_unrealised_constants.py
"""
import json
import warnings

import numpy as np

warnings.filterwarnings('ignore')
from simplipy.engine import SimpliPyEngine                     # noqa: E402
from simplipy.verify import MODE_TIERS                          # noqa: E402
from simplipy.verify._contract import judge_rule, judge_cl_battery  # noqa: E402

CFG = 'assets/engines/acj-4-3/config.yaml'
ROWS = json.load(open('assets/engines/acj-4-3/rules.json'))
e = SimpliPyEngine.from_config(CFG)

# (row index in rules.json, the smallest-magnitude constant that breaks it)
CASES = [(192, '(-19)'), (204, '(-19)'), (139, '(-20)'), (229, '(-20)'), (275, '(-20)'),
         (142, '(-50)'), (232, '(-50)'), (278, '(-50)'), (116, '(-750)'), (312, '(-750)'),
         (140, '1e17'), (198, '1e17'), (200, '1e17'), (203, '1e17'), (230, '1e17'),
         (276, '1e17')]

print('judge cl battery reaches only |c| <=',
      max(abs(float(b())) for b in judge_cl_battery()))
print("rules.json licenses tiers", sorted(MODE_TIERS['f64']), '\n')

bad = 0
for idx, c in CASES:
    lhs, rhs = ROWS[idx]
    assert lhs.count('<constant>') == 1 and '<constant>' not in rhs

    # 1. the shipped ROW judges as licensed for the f64 file
    row = judge_rule(list(lhs), list(rhs))
    # 2. an INSTANCE of the same row does not
    inst = [c if t == '<constant>' else t for t in lhs]
    ins = judge_rule(list(inst), list(rhs))
    # 3. the engine really performs the rewrite, and the deployed values differ in CLASS
    fired = e.simplify(list(lhs))
    dl, dr = e.as_callable(inst)(), e.as_callable(rhs)()

    ok = (row['tier'] in MODE_TIERS['f64']
          and ins.get('tier') not in MODE_TIERS['f64']
          and fired == list(rhs)
          and not (dl == dr or (np.isnan(dl) and np.isnan(dr))))
    bad += ok
    print(f"row {idx:4d}  {' '.join(lhs):28s} -> {' '.join(rhs):13s}"
          f"  row={row['verdict']}/{row['tier']}/realised={row['realised']}"
          f"  | {' '.join(inst):26s} -> {ins['verdict']}/{ins.get('tier')}"
          f"/realised={ins.get('realised')}  deployed {dl!r} vs {dr!r}   {'CONFIRMED' if ok else 'no'}")

print(f'\n{bad}/{len(CASES)} rows confirmed: licensed as a row, unrealised as an instance.')
