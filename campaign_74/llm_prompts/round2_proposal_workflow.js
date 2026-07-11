export const meta = {
  name: 'llm-rules-fullscale',
  description: 'Full-scale LLM identity harvest for the 7-4+ rule packs (length-uncapped)',
  phases: [{ title: 'Propose', detail: '10 family agents, uncapped source length' }],
}

const GRAMMAR = `
You are proposing candidate SOURCE expressions for a certifying simplification-rule miner (it derives and numerically certifies the minimal target itself; your target hints are recorded but not trusted). Wrong proposals cost ~1 CPU-second; right ones add rules uniform sampling would never find. GATHER EVERY IDENTITY YOU CAN THINK OF in your family -- aim for 80-150 proposals, systematic enumeration over cleverness.

GRAMMAR (prefix notation, one token per array element):
- UNARY: abs acos acosh asin asinh atan atanh cos cosh div2 div3 div4 div5 exp inv log mult2 mult3 mult4 mult5 neg pow1_2 pow1_3 pow1_4 pow1_5 pow2 pow3 pow4 pow5 sin sinh tan tanh  (multK/divK = *K, /K; powK = ^K; pow1_K = K-th root; inv = 1/x)
- BINARY: + - * / pow
- LEAVES: x0 x1 x2 x3; constants 0 1 (-1) np.e np.pi float("inf") float("-inf") float("nan"); wildcard <constant> (matches any fitted constant -- use for parameterized identities).
- SOURCE length: 5 to 13 tokens (NO upper limit at 7 anymore -- binomial-class sources of 9-13 tokens are wanted). The certified target must be SHORTER, ideally <= 4 tokens (longer minimal forms are rejected for now -- still propose borderline cases).
- float64 semantics; domain extension allowed (x/x -> 1); identities false at any finite sampled point (corners 0, +-1, +-pi, huge magnitudes) will be rejected -- do not propose approximations.
- OPERAND-ORDER VARIANTS: for commutative nodes, propose the 2-3 most distinct orderings/nestings as separate entries (the engine matches trees syntactically), and factored/expanded algebraic variants as separate entries too.
Verify every token list: known tokens only, exact prefix arity, length 5-13.`

const FAMILIES = [
  'BINOMIAL and polynomial: x^2+2xy+y^2 -> (x+y)^2 (10 tokens: ["+","pow2","x0","+","mult2","*","x0","x1","pow2","x1"]), x^2-2xy+y^2, difference of squares (x^2-y^2)/(x+y) and /(x-y), sum/difference of cubes over their factors, x^3+3x^2y+3xy^2+y^3 if it fits 13, partial-factored variants x(x+2y)+y^2, and nesting/order variants of each.',
  'TRIGONOMETRIC advanced: sum-to-product and product-to-sum shapes with <=4-token targets, sin(x+y)sin(x-y) = sin^2x - sin^2y direction that REDUCES, tan sum formulas, triple angles reducing, cofunction/phase compositions at depth, arc-function algebra (atan(x)+atan(1/x), asin+acos), tan(x/2) Weierstrass shapes.',
  'HYPERBOLIC advanced: exp-collapses at depth (cosh+sinh)^k -> exp(kx), tanh addition shapes, asinh(x) = log(x+sqrt(x^2+1)) direction that reduces (12 tokens!), acosh/atanh log forms, half-argument identities.',
  'EXP/LOG at depth: log(x^k)/k -> log-ladder collapses, log(sqrt(x)) forms, exp(log(x)+log(y)), log(e^x * e^y), change-of-base with np.e, log(1/x^k), nested exp(log(exp(...))), <constant>-parameterized log(C x)-log(C y) -> log(x/y) variants.',
  'RATIONAL-FUNCTION collapses: (x^2+xy)/(x+y) -> x, (xy+xz)/x -> y+z... careful target lengths, 1/(1/x+1/y) -> xy/(x+y) direction that reduces, nested fraction flattening (a/b)/(c/d) shapes, (x - x/y*y)-style zeros at depth.',
  'POWER/ROOT towers: sqrt(x^2 y^2) -> |xy|, roots of roots (pow1_2 pow1_2 = pow1_4), k-th root of x^k with parity/abs care, x^a x^b ladders via powK ops, (x^k)^(1/k) variants for k=2..5, sqrt(x)^2 vs sqrt(x^2), inv of pow towers.',
  'NEG/ABS/INV algebra at depth: nested neg/inv/abs towers 5-9 deep with mixed ops, abs(x)*abs(y) -> abs(xy), abs(x^2 y) collapses, sign cancellations through odd functions at depth (neg sin neg tan neg x0 chains), inv(neg(inv(neg x))).',
  'CONSTANT-LADDER arithmetic: multK/divK/powK compositions that collapse (mult2 mult3 -> mult... only if target expressible: mult2 mult2 = mult4!, div2 div2 = div4, mult4 div2 = mult2, pow2 pow2 = pow4), mixed with <constant> absorption (C*mult2(x) family), and with literals 2=1+1-ish shapes over 0/1/(-1).',
  'MULTI-VARIABLE symmetry: (x+y)-(y+x)-style zeros at depth, (x+y)^2-(x-y)^2 -> 4xy (13 tokens), x/(x+y) + y/(x+y) -> 1, cyclic 3-variable collapses that fit 13 tokens, differences of symmetric forms.',
  'SPECIAL-VALUE and dead-subtree: expressions with pi/e/0/1/(-1) subtrees inside larger 5-9-token forms (sin(pi)*f(x), log(e)*x, x^(pi/pi)), 0*f(x)+g(x) shapes, pow(f(x),0)-style collapses, x*1-tower peeling, and inf/nan propagation that certifies under domain extension.',
]

const SCHEMA = { type: 'object', required: ['proposals'], properties: { proposals: { type: 'array', items: {
  type: 'object', required: ['source', 'why'], properties: {
    source: { type: 'array', items: { type: 'string' } },
    why: { type: 'string' }, expected_target: { type: 'array', items: { type: 'string' } } } } } } }

phase('Propose')
const res = await parallel(FAMILIES.map((fam, i) => () =>
  agent(GRAMMAR + `\nYOUR FAMILY: ${fam}`, { label: `fam${i}`, phase: 'Propose', schema: SCHEMA, effort: 'high' })))
const all = res.filter(Boolean).flatMap((r, i) => (r.proposals || []).map(p => ({ ...p, family: i, round: 2 })))
log(`${all.length} proposals from ${res.filter(Boolean).length}/10 families`)
return { n: all.length, proposals: all }