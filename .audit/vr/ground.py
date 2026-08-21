import json, sys
from mpmath import mp, mpf, log10, isnan, isinf
from simplipy.verify._contract import parse, c_eval, slots_of

path = sys.argv[1]
rows = json.load(open(path))

def varfree(toks):
    return not slots_of(toks) and '<constant>' not in toks

cand = [(i, r) for i, r in enumerate(rows) if varfree(r[0]) and varfree(r[1])]
print('rows', len(rows), 'both-sides var-free', len(cand))

def gaps(row, dpss):
    out = []
    for d in dpss:
        mp.dps = d
        try:
            a = c_eval(parse(row[0]), {})
            b = c_eval(parse(row[1]), {})
        except Exception as e:
            return ('ERR', repr(e))
        if isnan(a) or isnan(b) or isinf(a) or isinf(b):
            return ('SPECIAL', str(a), str(b))
        if a == b:
            out.append(None)
        else:
            m = max(abs(a), abs(b))
            out.append(float(log10(abs(a-b)/m)) if m != 0 else float('-inf'))
    return ('OK', out)

stable, collapsing, exact, other = [], [], [], []
for i, r in cand:
    res = gaps(r, [50, 100, 400, 1500])
    if res[0] != 'OK':
        other.append((i, r, res)); continue
    g = res[1]
    if all(x is None for x in g):
        exact.append((i, r)); continue
    # look at dps 400 vs 1500: stable => real gap
    g4, g15 = g[2], g[3]
    if g4 is None or g15 is None:
        collapsing.append((i, r, g)); continue
    if abs(g4 - g15) < 0.5:
        stable.append((i, r, g))
    else:
        collapsing.append((i, r, g))

print('exact-equal   :', len(exact))
print('collapsing    :', len(collapsing))
print('STABLE GAP    :', len(stable))
print('special/err   :', len(other))
json.dump([[i, r, g] for i, r, g in stable], open('/home/psaegert/Projects/simplipy/.audit/vr/stable_%s.json' % path.split('/')[-1], 'w'), indent=0)
for i, r, g in stable[:25]:
    print(i, ' '.join(r[0]), '->', ' '.join(r[1]), ['%.2f' % x if x is not None else None for x in g])
