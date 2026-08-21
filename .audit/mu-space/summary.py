import sys, json, collections
adm = json.load(open('/tmp/mu/admitted_mu.json'))
alphabet_keys = {'-1','-10','-2','-3','-4','-5','-6','-7','-8','-9','0','1','10','2','3','4','5','6','7','8','9','<constant>','float("-inf")','float("inf")','float("nan")','np.e','np.pi'}
cat = collections.Counter(); bylen = collections.Counter()
rows = []
for r in adm:
    k = ' '.join(r['key'])
    if k == '<constant>': c = 'C. folds to <constant> (free-constant respelling)'
    elif k in ('float("nan")','float("inf")','float("-inf")'): c = 'D. folds to a non-finite leaf'
    elif k in alphabet_keys: c = 'A. folds to a leaf that IS in the alphabet'
    else: c = 'B. folds to a leaf NOT in the alphabet (composite is the only spelling)'
    cat[c]+=1; bylen[(c,len(r['tokens']))]+=1
    rows.append((c,k,r['mu'],' '.join(r['tokens'])))
print('TOTAL ADMITTED VAR-FREE COMPOSITES:', len(adm))
for c,n in sorted(cat.items()):
    print(f'  {c}: {n}   (len2={bylen[(c,2)]}, len3={bylen[(c,3)]}, len4={bylen[(c,4)]})')
with open('/tmp/mu/ADMITTED_FULL.txt','w') as f:
    for c,k,mu,t in sorted(rows):
        f.write(f'{c[0]}\t{mu}\t{t}\t->\t{k}\n')
print('full list written to /tmp/mu/ADMITTED_FULL.txt')
# category B distinct values
B = sorted({k for c,k,mu,t in rows if c.startswith('B')})
print('\ncategory B distinct folded values:', len(B))
print(', '.join(B[:80]))
