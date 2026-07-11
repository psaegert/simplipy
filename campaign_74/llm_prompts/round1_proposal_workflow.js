export const meta = {
  name: 'llm-rule-proposals',
  description: 'Family-diversified LLM proposals of simplifiable source expressions for the 7-4 mine',
  phases: [{ title: 'Propose', detail: '6 identity-family agents emit candidate sources' }],
}

const GRAMMAR = `
You are proposing candidate SOURCE expressions for a symbolic simplification rule miner. The miner will itself derive and numerically certify the minimal equivalent target, so your job is ONLY to point at expressions LIKELY to be reducible -- being wrong costs ~1 CPU-second, being right adds a rule that uniform sampling (0.03% coverage at length 7) would almost surely miss.

GRAMMAR (prefix notation, one token per array element):
- UNARY ops: abs acos acosh asin asinh atan atanh cos cosh div2 div3 div4 div5 exp inv log mult2 mult3 mult4 mult5 neg pow1_2 pow1_3 pow1_4 pow1_5 pow2 pow3 pow4 pow5 sin sinh tan tanh
  (multK = multiply by K; divK = divide by K; powK = K-th power; pow1_K = K-th root; inv = 1/x; neg = -x)
- BINARY ops: + - * / pow
- LEAVES: variables x0 x1 x2 x3; constants 0 1 (-1) np.e np.pi float("inf") float("-inf") float("nan"); the wildcard <constant> (matches ANY fitted constant; use it for parameterized identities).
- SOURCE length: 5 to 7 tokens (shorter lengths are already exhaustively mined). The certified TARGET must be SHORTER than the source (at most 4 tokens), so propose expressions whose simplified form is SHORT.
- Semantics are float64 numerics; domain-extension is allowed (x/x -> 1 style), but identities that are wrong at any FINITE point the heavy-tailed sampler hits (corners 0, +-1, +-pi...) will be rejected.

EXAMPLES of the shape wanted (these may already be known -- propose DIFFERENT ones):
["+", "pow2", "sin", "x0", "pow2", "cos", "x0"]   (sin^2+cos^2 -> 1, length 7)
["-", "pow2", "cosh", "x0", "pow2", "sinh", "x0"] (cosh^2-sinh^2 -> 1, length 7)
["mult2", "*", "sin", "x0", "cos", "x0"]          (2 sin cos -> sin(2x), length 6)
["log", "*", "exp", "x0", "exp", "x1"]            (log(e^a e^b) -> a+b... target len 3, length 6)

Think systematically through your ASSIGNED FAMILY; enumerate variations (operand orders, neg/inv wrappings, <constant> parameterizations, x0 vs two-variable forms). Aim for 40-80 proposals. Verify each token list: every token from the grammar above, correct prefix arity (unary consumes 1 subtree, binary 2), length 5-7. Return ONLY valid arrays.`

const FAMILIES = [
  'TRIGONOMETRIC identities: Pythagorean variants, double/half angle, sum-to-product shapes expressible in <=7 tokens, tan = sin/cos compositions, phase shifts with pi, parity compositions (sin(neg x) wrappings that survive lengths 5-7), arc-function compositions (atan(tan), sin(asin) domain-extension pairs, asin+acos = pi/2 shapes).',
  'HYPERBOLIC identities: cosh^2-sinh^2, tanh = sinh/cosh, double-argument forms, exp-based collapses (cosh x + sinh x -> exp x!), inverse-hyperbolic compositions (asinh(sinh), tanh(atanh)), parity wrappings.',
  'EXP/LOG laws: log(a*b), log(a/b), log(exp), exp(log), log(pow_k x), exp(a+b) shapes, log(1/x), powers of e (np.e leaf!), log(np.e), <constant>-parameterized: log(C*x), exp(x+C), C^x via pow.',
  'ALGEBRAIC/RATIONAL cancellations: (x*y)/x shapes, x/(x*y), (x+y)-x, nested inv/neg collapses at depth 5-7, div-of-div, mult/div constant-ladder collapses (mult2 div2, mult4 = mult2 mult2, div4 mult2 -> div2), <constant> absorption shapes (C*x + C*x, C*(x+C)).',
  'POWERS and ROOTS: pow2(pow1_2), pow1_2(pow2) (abs!), pow composition ladders (pow2 pow3 = pow... 6 not in vocab -- think which ARE reducible), pow(x, <constant>) shapes, sqrt of products, pow2(neg x), abs wrappings (abs(pow2), abs(neg), abs(abs)), inv(pow_k) vs pow_k(inv).',
  'SPECIAL VALUES and constant-folding at depth: expressions over np.pi/np.e/0/1/(-1) leaves that collapse (sin(pi), log(e), cos(0) INSIDE larger expressions so total length is 5-7), inf/nan propagation shapes that certify under domain extension, <constant>-only towers with one variable multiplied in, x + 0*y shapes, x * pow(y, 0) style dead subtrees.',
]

const SCHEMA = {
  type: 'object', required: ['proposals'],
  properties: { proposals: { type: 'array', items: {
    type: 'object', required: ['source', 'why'],
    properties: {
      source: { type: 'array', items: { type: 'string' } },
      why: { type: 'string', description: 'the identity, one line' },
      expected_target: { type: 'array', items: { type: 'string' } },
    } } } },
}

phase('Propose')
const res = await parallel(FAMILIES.map((fam, i) => () =>
  agent(GRAMMAR + `\nYOUR ASSIGNED FAMILY: ${fam}`,
    { label: `family:${i}`, phase: 'Propose', schema: SCHEMA, effort: 'high' })))

const all = res.filter(Boolean).flatMap((r, i) => (r.proposals || []).map(p => ({ ...p, family: i })))
log(`${all.length} raw proposals from ${res.filter(Boolean).length}/6 families`)
return { n: all.length, proposals: all }