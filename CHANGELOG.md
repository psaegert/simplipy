# Changelog

## 0.14.0 (unreleased)

### Changed — SOUNDNESS IS AN AXIS, NOT A LADDER (owner rulings, 2026-08-19/20)

`Mode` is now `f64` (the default), `real` and `corpus`. It is a plain `Enum`, so `<`
between modes raises `TypeError`.

The old `IntEnum` encoded a premise — `EXACT ≤ SOUND ≤ AE ≤ LOSSY` — that measurement
refuted. "True over ℝ" and "realised in f64" are *incomparable*: `atanh(tanh t) → t` is
true for every real `t` and gives `inf` in f64 past 18.990341103219276, while
`asin(1e-8) → 1e-8` is bit-identical in f64 and wrong by the cubic term. Neither is more
sound than the other.

- **`Mode.SOUND` and `Mode.LOSSY` are removed**, as are the strings `'sound'` and
  `'lossy'`. Use `Mode.f64` and `Mode.corpus`. A retired spelling now raises rather than
  resolving quietly, because the old names asserted a single soundness ordering that does
  not exist.
- **`Mode.f64` no longer folds `sin(np.pi)` to `0`.** Exactly `0` in mathematics,
  `1.2246467991473532e-16` in f64 — the rewrite changes what the deployed evaluator
  computes. It serves `real` and `corpus` instead. 102 rules move, nearly all of the same
  symbolic-cancellation family.
- **`Mode.real` fails closed** on an artifact with no `rules_real.json` rather than
  serving it the f64 set, which would answer a request for mathematical soundness with
  rules that are f64-exact and mathematically false.
- `f64` mode preserves what the deployed evaluator computes for every rewrite it
  applies, **not** your evaluation order: the canonical form re-associates, and IEEE-754
  addition commutes but does not associate. Deterministic, and value-preserving for
  well-conditioned expressions.

### Upgrading from 0.13.x

**What you get back changes, and mostly it gets better.** Measured on the 400-row
benchmark corpus, comparing 0.13.1's `SOUND` against each 0.14.0 mode by cost under the
serve ordering:

| mode | vs 0.13.1 `SOUND` |
|---|---|
| `real` | identical on all 400 rows |
| `f64` (the new default) | identical on 397, more complex on 3 |
| `corpus` | simpler on 53, never worse |

The three rows where the default gives up ground are rewrites that are true over ℝ but
not realised in f64 — `asinh(sinh t) → t`, `cos(asin(sin x)) → |cos x|`,
`pow(inf, x) → exp(inf·x)`. They did not disappear; they moved to `real` and `corpus`,
which is the whole point of the split. If you were relying on them, ask for
`mode='real'` and you are back where you were, exactly.

**Expressions come back in a different dialect.** `simplify` now answers in the form it
was given (see the conversion section below); 0.13.x returned the tagged form
regardless. The expression is the same expression — cost is unchanged on every row above
— but the tokens differ, so any test pinning literal output needs regenerating.

**Everything deprecated is removed in this release.** There is no alias cycle: a retired
name raises rather than warning, so the migration is mechanical and complete.

| removed | use instead |
|---|---|
| `Mode.SOUND`, `Mode.LOSSY`, `'sound'`, `'lossy'` | `Mode.f64`, `Mode.corpus` |
| `SimpliPyEngine.parse` | `read_infix` |
| `simplify(..., form=…)` | convert first: `simplify(to_tagged(x))` |
| `simplify(..., node_budget=)` | `max_passes=` |
| `SimpliPyEngine.load(path=)` | `load(engine=)` |
| `normalize_skeleton`, `normalize_expression` | `to_skeleton`, `to_expression` (now require `engine=`) |
| `masking.mask_values_keep_structure` | `masking.mask_fittable` |
| `utils.substitude_constants` | `utils.substitute_constants` |
| `utils.numbers_to_constant` | `explicit_constant_placeholders` |

**BREAKING for downstream packages: this is a hard cut.** New `flash-ansr` and `srbf`
releases require simplipy >= 0.14.0. There is no compatibility shim, as with the last
artifact-format break: the rule sets, the artifact layout and the returned dialect all
move together, and a shim would have to lie about at least one of them. Pin the pair.

### Removed — the superseded `acj-2-1` and `acj-3-2` engines

Installs no longer bundle them; `acj-4-3` is the shipped engine. Both remain on the
Hugging Face assets repository and still load by name, so nothing published is
withdrawn — `simplipy install acj-3-2` fetches it as before.

### Changed — the literal floor is one bit, not two

Every numeric codeword was clamped at two bits, which swallowed `0`, `1`, `2` and `3`
alike — so `mu` could not tell `x**2` from `x**3` from `x**4`. The clamp is now one bit,
where it binds only at `0` and `±1`, values whose codeword could not be shorter anyway.
The exponents order again, and the sign on `±1` becomes visible, which is what the rest
of the measure already promised: a description length must tell a number from its
negation.

No rule of the 5,451 published changes direction and no printed literal spelling moves —
the serializer compares the same two codeword totals the measure does, and the floor
shifts both. `SimpliPyEngine.complexity` returns different numbers; pinned values
downstream must be re-earned.

### Changed — every grammar symbol carries its own price

The simplicity measure `mu` charged a flat 8 bits for every grammar symbol: a bag, a
`Pow`, a function head, a variable leaf and a named constant alike. A description length
is not flat, and the flatness had a visible cost — `pi - 2` and `asin(sin 2)` priced
identically, so the miner preferred the composition and shipped a whole parametrised
family of them. Those rules are true and certified; they are simply not simplifications.

`mu` now reads each symbol off a table: a variable leaf 6 bits, `Add`/`Mul`/`Pow` 3,
`pi` and `e` 4, an elementary head (`exp log abs sin cos tan rootn`) 6, an
inverse/hyperbolic head 8, an infinity or an unnamed head 8. Literal pricing is
untouched. The entries are read against the same 1/8 unit and scale with it, so
`SIMPLIPY_MU_SYM` keeps its meaning.

Two entries are fixed by rules a frequency count cannot supply: **a named constant must
cost less than the cheapest expression denoting it** (`acos(-1)` is 11 bits, `exp(1)`
is 9), and **a leaf must also name which variable it is** — the reason it is 6 and not
the 1.65 bits the census reads for the class.

Measured across the change: no rule of the shipped 5,451 changes direction, no corpus
output changes in any mode, and no literal spelling moves. What does change is the
number `SimpliPyEngine.complexity` returns — it is a different measure, so any pinned
`mu` value downstream must be re-earned. Published rule sets mined under the previous
measure still load and serve; they warn (D25/R6) that their minimality and ordering
claims were certified against the old prices, until the next mine replaces them.

### Changed — the artifact is a TRIPLE

A published asset is now six files, not four: `rules.json` (the f64 set, keeping its
name so older configs load unchanged), `rules_real.json` and `rules_corpus.json`, plus
`config.yaml`, `mine.yaml` and the provenance sidecar. One distinct, complete rule set
per mode — no base plus overlays, so what is loaded is what is served. The triple is the
unit of mining, pinning and distribution; rules licensed in no mode are recorded in the
drop census rather than silently absent.

### Added

- `SimpliPyEngine.evaluate_constants(expression)` — the explicit door to numeric folding.
  Folds maximal variable-free, slot-free subtrees; refuses a subtree carrying a
  `<constant>` (a fitted degree of freedom, not a value) and refuses a non-finite result.
  Never reached by `simplify`.
- `verify_ruleset(..., mode=...)` and `verify_triple(...)` — cleanliness is now **per
  mode**. `atanh(tanh t) → t` is exactly what belongs in `rules_real.json` and a defect in
  `rules.json`; a single bucket count cannot say that. Omitting `mode` keeps the
  pre-triple meaning.
- `simplify(..., max_passes=)` replaces `node_budget=`, which never counted nodes. **The
  old name is removed**; the bound is the number of outer rewrite passes, which is what
  the new name says.
- **`SimpliPyEngine.to_realization(expr)` — the fourth notation.** The same expression
  with each operator spelled as the callable it runs (`sin` → `simplipy.operators.sin`),
  and reversible like the other three: `to_prefix`, `to_infix` and `to_tagged` all accept
  realization-spelled input. Operators realized as themselves (`+`, `-`, `*`) are
  identical in both spellings, so only dotted realizations ever need detecting.
  - Reading realization notation back requires the engine's realization map to be
    injective. Two operators may legally share one realization, and then the spelling
    names both; that refuses loudly rather than picking one. Writing never refuses.
- **`SimpliPyEngine.as_code(expr)` and `as_callable(expr)` — the compile pipeline in one
  step.** `as_callable` replaces the `prefix_to_infix(realization=True)` → `codify` →
  `code_to_lambda` dance for the common case, binding the result to this engine's own
  namespace. `expression_variables(expr)` exposes the signature it binds, in order of
  first appearance.
  - Named `as_`, not `to_`, and deliberately: every `to_*` in this API is a notation that
    converts back, and a code object has no syntax to recover. `to_lambda` would advertise
    a round-trip that cannot exist. Compiling also runs through `compile()`, so the trust
    rules apply where the notation family's do not.
- `simplipy.DEFAULT_ENGINE` and `simplipy.DEFAULT_ENGINE_REVISION` — the artifact this
  version was built and tested against. `SimpliPyEngine.load()` with no argument resolves
  it and says which it took. The pin lives in the package rather than the hosted manifest,
  so what you get by default is answerable offline and does not change under you without a
  simplipy release. The revision is what separates one name across versions: `acj-4` mined
  under two different judges is legitimately `acj-4` both times, and only (name, revision)
  names one artifact.
- `SimpliPyEngine.load(engine=...)` replaces `load(path=...)`, which named neither a path
  nor a version. **The old keyword is removed.**
- **`simplipy.utils.substitude_constants` is removed** (the historic misspelling, warning
  since 0.13.0); use `substitute_constants`.

### Security — the trust model covers the whole dotted path

A realization is resolved by attribute traversal, and a module's attributes include
every module it imported. The trust check decided only the leading component, so a
config that named a trusted root could walk past it: `simplipy.engine.os.system` loaded
without an `UntrustedModuleError` and called `os.system` at first evaluation. The reach
was general rather than incidental — a walk of the attribute graph from the four default
roots lands on 53 distinct top-level packages within three hops, `os`, `ctypes`,
`importlib` and `builtins` among them.

The allowlist now applies to every hop. Any component that resolves to a module is held
to the same trusted set as the root, dunder components are refused, and a realization
that names a module rather than a callable is refused. `math`, `np`, `scipy` and
`simplipy` remain trusted by default; `trusted_modules=` and `SIMPLIPY_TRUSTED_MODULES`
remain the only ways to widen that, and they widen the whole path.

A realization spelled as a **bare name** is now an allowlist too. It imports nothing,
which is why it used to pass on shape alone, but it is still interpolated into generated
source and resolved against Python's own builtins — where `eval`, `exec`, `compile`,
`open` and `__import__` sit beside `abs`. The accepted names are `abs`, `round`, `min`,
`max`, `sum`, `pow`, `divmod`, `float`, `int`, `bool` and `len`, alongside the bare
operator symbols; anything else refuses at load.

This affects any deployment that builds an engine from a config it did not write — a
Hugging Face asset loaded by name, a colleague's file, a cloned repository. Shipped
configs, the legacy vocabulary and every realization the deployed evaluation path uses
are unchanged; upgrade to 0.14.0 to get the check.

### Fixed

- The infix reader bound a minus after a power operator outside the power: `2^-3`
  parsed as `-(2^3)` and returned **-8** where the value is 0.125, `sin(x0)^-1` lost
  its reciprocal entirely, and every `x^-c` collapsed to `-(x^c)`. The minus after
  `^`/`**` is the exponent's sign -- positional, not precedential, exactly Python's
  own grammar (`-x**2` is still `-(x**2)`) -- and the reader now binds it to the
  exponent, including chained signs (`x^--2`) and exponent power chains (`x^-2**3`
  is `x^(-(2**3))`). Only expressions read from infix with a negative exponent were
  affected; prefix input and every mined artifact are untouched.

- The `atanh(tanh(t))` → `t` collapse carried a band of |t| ≤ 18, derived from where
  f64 `tanh` attains ±1 (18.99). But `tanh` fails by **compression** long before it
  saturates -- it loses ~2|t|/ln2 bits of the argument approaching ±1 -- and under the
  same realised 8-ULP criterion every rule is held to, the roundtrip first breaches at
  |t| ≈ 2.7 (libm-dependent; by |t| = 18 the drift is ~0.03-0.06 *absolute*). The band
  is now 2, the last magnitude ceiling on the safe side of every measured breach. The
  overflow-derived bands (`log∘exp` 709, `sinh`/`cosh` pairs 710) are a different
  failure mode and keep their attainment thresholds.

- The same re-derivation exposed the sibling holes at the other end: the collapse
  bands guarded only against **overflow**, but compression near a flat point of the
  inner function eats the low end -- `log(exp(t))` breaches the realised bar for all
  |t| below ~0.0625 and `acosh(cosh(t))` below ~0.25 (`exp` rounds a small argument
  into the neighbourhood of 1; `cosh` is quadratically flat at 0). Both collapses now
  require a *proven* magnitude floor of 1 beside their ceiling, where the measured
  worst is 0 and 1 ULP; `asinh(sinh(t))` measured clean over its whole band and is
  unchanged. The two shipped f64 rules of this shape (`log exp <constant>`,
  `acosh cosh <constant>`) re-tier to `real` with the 0.14.0 artifact.

- The literal codeword floor (one bit, ruled 2026-08-22) reached `mu_rat` but not the
  beyond-i128 leaf pricer, whose clamp stayed at two bits -- so `1e-39` priced *above*
  `3e-38`: the simpler mantissa at the adjacent magnitude, upcharged for crossing the
  representation line. The floor is now one constant shared by both sides, the seam
  (`1e-40` = `1/1⟨40 zeros⟩`) stays closed, and the measure fingerprint gained a
  beyond-i128 probe so this class of drift can never again leave the digest unchanged.
  Measured over all 16,018 shipped rules: zero prices move, zero directions flip.

- The judge quantified a source-side `<constant>` over |c| ≤ 5 while the miner draws its
  constants from [1e-3, 1e3] and sweeps magnitudes to 500 — so `forall c over the reals`
  was checked on one decade. The battery now spans ±1e-3 to ±1e4. Reaching there needed
  two fixes to the exists-witness search, both cases of answering at a point where the
  answer cannot be read: a witness fitted where the deployed left-hand side has
  **saturated** solves an *interval*, not a point, and its edge is not a witness (an
  identity was convicted on the difference); and the fixed search grid thins out exactly
  where the deployed algebra overflows, so the constant under test is now tried directly.
  Rules true over the reals but unrealised at large constants move from `f64` to `real`,
  where they belong — 107 of the 5,451 currently published, with no rule newly rejected.

- The judge's realisation axis was measured with a `1e-9` tolerance carrying an absolute
  floor — roughly 10⁷ ULP — so a rule the contract convicts could be stamped "realised"
  and admitted to the default rule set. It now uses a bound derived from this platform's
  measured libm error of the platform's own math library, and the two questions it used
  to conflate —
  "does f64 compute this?" and "has the deployed algebra diverged?" — have separate
  comparisons.
- The contract lane's second precision rung was a fixed dps 120, which confirms nothing
  when intermediates are 10²¹⁷: both rungs are swamped alike and agree on a manufactured
  verdict. It is now sized from the largest intermediate actually seen. This was killing
  `log(cosh(25t) + sinh(25t)) = 25t`, which is exactly true.
- The contract's finite comparison measured a RELATIVE separation, normalised by the
  larger of the two sides. Against a side that is exactly zero that normaliser is the
  other side, so the reading was `1.0` at every precision — it could not respond to
  precision at all, which is the only question the comparison asks, and an identity whose
  two sides agree exactly was convicted for its own arithmetic residue. The separation is
  now measured absolutely, against a single scale held fixed across the precision rungs.
  This was killing true identities of the form `log(eᵃ · e⁻ᵃ) = 0`, and the two judges
  could disagree about them, because the verdict turned on whether the arbitrary-precision
  library happened to return an exact zero at the upper rung.
- The gate had no verdict at all for the saturation family — a bounded function claimed to
  reach its limit. `tanh(cosh(10))` differs from 1 by 2e-9566, far inside the precision
  band at any working precision anyone can afford, so a numeric comparison can only
  abstain however deep it is taken, and 174 shipped rules were carried without the gate
  ever forming an opinion on them. A bounded function does not attain its limit at a finite
  argument, and the comparison now settles those rules from that fact rather than from a
  measurement — confirming every one of them into the tier it already ships in. The gate
  now returns a verdict for every rule in the artifact.
- The judge could not check the constants the miner explores. Its source-constant battery
  reached `|c| ≤ 5` plus π and e while the miner draws from `[10⁻³, 10³]`, and widened to
  ±10⁴ it convicted 97 of the 343 constant-carrying rules in the shipped set — every one
  of them true by construction, and every one an artifact of the instrument at a magnitude
  it could not resolve. Six distinct causes, each fixed:
  - The exists-witness search accepted `0` as the witness for a target of `4.4e-589`. Its
    noise floor is an absolute band, there to stop a cancellation zero from being chased
    as a witness; a value that is simply small — one power and no subtraction — fell
    under it too. The floor is now consulted only where the target actually shrinks as
    the working precision rises, which is what separates a residue from a value.
  - That search also stopped eight decades short of the f64 ceiling, so a witness of
    `7.6e300` had no bracket and the rule was reported as having none.
  - A precision rung that read one side as exactly zero, while the other was not, was
    still used to date the gap's decay. It cannot: the reading is pinned at the
    instrument's rail. The comparison now measures the rate between two rungs that both
    saw something, which can only ever move a conviction to a stricter pair of rungs.
  - The rungs were sized only from the LARGEST intermediate. A small one costs the same:
    `cos x = 1 - x²/2 + …` at `x = e⁻³⁰⁰` puts the whole content of the answer 261 digits
    below the 1 it sits beside, so three rungs in a row read exactly `1.0` and `acos` of
    that is `0`. Precision is now sized from both ends.
  - The evaluator's own symbolic-cancellation floor was absolute, so `sin(e⁻³⁰⁰)` — which
    is exactly the size its argument predicts, with nothing cancelled — read as `0` at two
    precisions and as `5.1e-131` at a third. The floor is now relative to what the
    argument predicts, and still fires for every cancellation it was written for.
  - Past its own precision ceiling the judge convicted rather than abstaining, which is
    the opposite of what that ceiling documents. Running out of evidence is a statement
    about the judge, so the conviction is withheld — except against a value the rule
    itself spells, which is exact at every precision and cannot be a casualty of running
    short of it.
- Two shipped rules of the form `exp(exp(sinh c)) = 1` were certified as true over ℝ. They
  are not: at `c = -6` the left side differs from 1 by `2.5e-88`. They are exact in f64 and
  stay in the same rule set; what changes is that the gate now says which of the two it is.
- The miner asked one question where the judge asks two. Its row gate skips the contract
  entirely wherever the deployed f64 sides "agree", and the bar for that was the band that
  answers a different question — whether the deployed algebra has *diverged*, which is
  deliberately loose because a few ULP of rounding is not a divergence. As a pre-screen it
  is roughly 10⁷ ULP, and below 1 a pure absolute floor: it reads `e^sinh(-5) = 0` as an
  equality when the sides are `5.9e-33` and `0.0`, so a rule that is false could be mined
  without its truth ever being asked. The two questions now have the two comparisons the
  judge already used — realisation in ULP for the pre-screen, the structural band for
  divergence. Measured: 8 of 102 `real`-tier rules in a reference mine have a point that
  now reaches the contract, up to 4.9 million ULP apart, and every mined artifact is
  byte-for-byte what it was.
- The D15 diagonal lane never received the precision coupling its commit claimed, and fed
  the same measure that convicts.


### Changed — CONVERSION and SIMPLIFICATION are separate (owner rulings, 2026-08-18)

- **`to_infix` / `to_prefix` / `to_tagged` are now PURE, SYNTACTIC conversions.** They
  re-notate and do nothing else: no canonical state is built, no rewrite rule fires,
  nothing is collected, folded, reordered or re-spelled, and the answer no longer
  depends on the engine ARTIFACT. `to_prefix(['+','x0','x0'])` is `['+','x0','x0']`
  (it was `['*','2','x0']`), and `to_prefix(['pow','abs','x0','2'])` keeps the `abs`
  under every ruleset (it was dropped under acj-4-3 and kept under acj-2-1 — a
  *conversion* whose answer depended on which artifact was loaded).
  - Why: `to_prefix(x)` was EXACTLY `simplify(x, form='explicit')` on every probe, so
    the trio added no capability over `simplify(form=…)`, and there was no
    spelling-preserving prefix↔tagged path at all. Measured before the change, on 400
    raw corpus rows: **not one** of the nine in/out directions preserved the input's
    spelling on more than 1/400 rows, and no 2-cycle or 3-cycle was an identity.
  - What is guaranteed now, per direction (full table in the design note): `P→P`,
    `T→T`, `I→I` and `T→P→T` are **byte identity** (400/400); `P→T→P` is identity **up
    to associativity** of same-polarity `+`/`*` runs, with bag member ORDER preserved;
    `P→I→P` is identity **up to the infix language's missing atoms** (`inv X ↦ / 1 X`,
    `p/q ↦ / p q`). Every direction and every cycle preserves the canonical state
    exactly: `simplify(to_X(y))` is byte-identical whichever dialect `y` arrived in
    (400/400).
- **`simplify` is DIALECT-PRESERVING: it answers in the form it was given.** `str` in →
  `str` out; explicit binary prefix in → explicit binary prefix out; tagged in → tagged
  out. Container mirroring (`tuple`/`ndarray`) is unchanged, and the simplification
  itself — every canonical answer — is unchanged.
  - **This closes the tagged leak.** `simplify(['*','2','atan','x0'])` returned
    `['<mul>','2','atan','x0','</mul>']`; it now returns `['*','2','atan','x0']`. A
    tag-free token input can never come back carrying tags.
  - Dialect detection: a token sequence is TAGGED iff it carries a bag delimiter or an
    inverse-section marker (`<add> </add> <mul> </mul> <sub> <div>`), else EXPLICIT.
- **`simplify(..., form=…)` is REMOVED.** `simplify` answers in the form it was given;
  choosing the notation is the conversion layer's job, not a simplification argument.
  - The migration is **convert first, then simplify**: `simplify(x, form='tagged')` →
    `simplify(to_tagged(x))`, `form='explicit'` → `simplify(to_prefix(x))`,
    `form='infix'` → `simplify(to_infix(x))`. All three are byte-identical to the
    parameter on every corpus row measured (400/400).
  - The reverse composition `to_tagged(simplify(x))` is **not** the migration: the
    tagged and explicit canonical emitters carry different sign/inverse/literal
    doctrine, so it agrees with `form='tagged'` on only 186/400 rows.
- **`prefix_to_infix` renders two things correctly that it rendered ambiguously.** A
  compound numeric leaf (`7/3`, `-1/3`) now carries the precedence its spelling embeds,
  so `* 7/3 7/3` renders `7/3 * (7/3)` instead of `7/3 * 7/3` (which re-read as
  `((7/3)*7)/3`); and a config-declared binary operator outside `+ - * / **` renders as
  a function call, like `rootn` and `pow`, instead of `x0 hypot2 x1` — a spelling this
  engine's own reader could not read back.
- **`read_infix` is unchanged**, and its exclusive capability is now clearly the
  vocabulary TOLERANCE: `read_infix('sqrt(x0)')` is `['sqrt','x0']` where every
  conversion refuses. The canonicalising contrast it used to draw with `to_prefix` is
  now with `simplify`.
- **`to_skeleton` / `to_expression` are unchanged in contract**; internally they now ask
  for canonicalisation explicitly (`simplify(to_prefix(x))`) instead of relying on
  `to_prefix` to canonicalise. `masking.mask`'s collect stage likewise dropped its
  `form=` projection — dialect preservation is `simplify`'s own contract now.

### Changed — `normalize_skeleton`/`normalize_expression` are now `to_skeleton`/`to_expression`
- **The name states the OUTPUT form**, matching the engine's
  `to_infix`/`to_prefix`/`to_tagged`. The old names hid what the skeleton form
  actually does: it masks **every** numeric literal, `pow` exponents and `rootn`
  indices included, so `pow(x0, 3)` becomes `pow(x0, <constant>)` — a family no
  constant optimizer can fit. (`engine.mask(expr, 'fittable')` is the kind that
  keeps the structural literals.) **The old spellings are removed**, from both
  `simplipy.normalization` and the package root. They are out of the declared `__all__`, the same
  treatment `masking.mask_values_keep_structure` got.
- **`normalize_variable_token` keeps its name.** It is a single-TOKEN helper that
  answers `(token, is_variable)`; it produces no form, reads no dialect and needs
  no engine, so the `to_*` family naming would misdescribe it.

### Changed — both forms accept all three dialects, and the answer no longer depends on which
- **They take an infix `str`, an explicit binary-prefix token sequence or a tagged
  token sequence, and return the dialect they were given** (`str`→`str`,
  prefix→prefix, tagged→tagged), detected exactly as the conversion API detects it
  (type first, then a liberal read of the tokens). Masking names an abstraction
  level, not an output dialect, so neither function may silently re-spell.
- **The CONTENT is now dialect-invariant, and that is a correctness fix, not a
  convenience.** Masking the caller's own spelling one token at a time counted a
  different number of `<constant>` slots for the same expression depending on the
  dialect it arrived in, in both directions — `-x0*x1` is `neg * x0 x1` (0
  constants) but `<mul> -1 x0 x1 </mul>` (1), and `-1/x0` is `/ -1 x0` (1) but
  `neg inv x0` (0). Measured at **35/400 rows (8.8%)** of the repo corpus, and at
  the same 35/400 for a role-aware `mask(mask_all)` applied per dialect — role
  awareness does not help, because the disagreement is about what the SITES ARE.
  A skeleton is what downstream holdout and decontamination key on, so a
  dialect-dependent answer let a tagged-era pipeline silently fail to match
  explicit-era keys and report clean. Every input is now canonicalised into the
  internal AC state first, abstracted on the canonical **explicit binary prefix**
  (where a sign and a reciprocal stay structure instead of being folded into a
  literal), and only then rendered into the caller's dialect.
- **BREAKING: both now require an `engine`.** That is the cost of the guarantee;
  there is deliberately no engine-free path and no fall-back to the old positional
  pass, because silently returning a dialect-dependent key is the failure the
  change exists to prevent. Callers that passed a token list now pass
  `to_skeleton(tokens, engine)`.
- Consequences of canonicalising: the expression form carries the state's **exact
  rationals** (`2.5` re-spells as `5/2`; the skeleton is unaffected, both halves
  being one degree of freedom); undeclared vocabulary and reserved numeric
  spellings (`inf`, `1_000`) now raise `ValueError` at the core's token grammar
  instead of passing through; `<c>` folds to `<constant>` in **both** forms.

### Changed — `to_skeleton` is not a third masking path
- It is documented and implemented as exactly `to_expression` followed by the
  ratified front door `engine.mask(..., policy='all')`. Two consequences are now
  on the record: the collect stage re-runs the engine, so simplification **rules
  fire** and a skeleton depends on the engine ARTIFACT and not only on the
  expression; and `engine.mask(to_expression(e, engine), 'all', collect=False)` is
  the escape for a rules-free, strictly positional abstraction of the canonical
  spelling.
### Removed — `simplipy.utils.numbers_to_constant`

- **The standalone helper is gone** (owner ruling 2026-08-18: the removal lands
  in 0.14.0 and the downstream packages adapt). It warned from 0.12.0 and the
  0.13.0 changelog already named 0.14.0 as its removal version; until now it was
  still present, so the two surfaces disagreed. Replacement:
  `simplipy.masking.mask(tokens, engine, policy)` or the `engine.mask` front
  door, with the policy that states the intent — `mask_all` for the legacy
  mask-everything behaviour, `mask_fittable` for what a constant optimizer can
  actually fit; `collect=False` gives the positional 1:1 substitution the helper
  approximated.
- **The replacement is deliberately not a drop-in**, because the helper was
  wrong in ways a rename would have preserved. It classified by a bare `float()`
  probe, so it minted a finite-by-doctrine `<constant>` for the reserved
  spellings `inf`/`nan` (any case or sign) and `1_000` — masking a non-finite
  literal into a placeholder that is finite by contract. Those now raise
  `ValueError` at the masking boundary. The same probe raised on
  `np.pi`/`np.e` and on the AC core's exact fraction `1/3`, so the helper walked
  past all three; they are masking sites and `mask_all` masks them. And the
  helper was role-blind and structure-blind: it never looked at the expression,
  so a malformed sequence was rewritten positionally instead of refused.
- **Two names that share a word with it did NOT move.**
  `explicit_constant_placeholders(..., convert_numbers_to_constant=)` is a
  keyword on a mechanical code-generation helper, still accepted with both
  values, and shipped downstream call sites pass it explicitly.
  `read_infix(..., mask_numbers=True)` keeps
  masking through the internal Rust port, which is not reachable by name from
  Python and is unchanged.
### Changed — rule dedup compares the INTERNAL FORM, not the spelling
- **`simplipy.utils.deduplicate_rules` now keys on the engine's internal form**
  (owner ruling 2026-08-18: *"instead of comparing spelling, compare internal
  form"*). Two rules are one rule when the loader translates their sources into the
  same pattern. Keyed on the token spelling, `* (-1) asin _0` and `* asin _0 (-1)`
  shipped as two entries for one pattern, and the second could never fire — the
  first already owns every subject it matches.
- **`engine=` is a REQUIRED, keyword-only argument.** There is no engine-free
  reading of "the same rule": the internal form is defined by the operator table.
  It is keyword-only on purpose — a positional third argument would silently have
  swallowed an existing caller's positional `verbose`. Existing calls fail loudly
  with a `TypeError` naming the argument.
- **The key is deliberately not `to_prefix`/`to_tagged`.** Those run the rewrite
  pass under the *loaded ruleset* and the full certificate context: on an engine
  holding acj-4-3, `to_prefix(['*','(-1)','asin','_0'])` returns `asin neg _0` —
  that rule's own target — while on a rules-less engine over the same vocabulary it
  returns `neg asin _0`. A dedup key that depends on the ruleset being
  deduplicated is circular. The new key is a pure function of the operator table.
- **Measured effect on the shipped artifacts, at LOAD (the files on disk are
  untouched and byte-identical):** acj-2-1 26 → 24 served (−2, 7.7%), acj-3-2
  106 → 96 (−10, 9.4%), acj-4-3 6,594 → 5,545 (−1,049, 15.9%). Every collision is
  DEAD — same lhs *and* same rhs internal form, zero shadowed — so first-match-wins
  meant the dropped copies could never fire.
- **Behaviour is unmoved**, measured four ways per artifact: zero byte-diffs over
  the 443 corpus rows the operator set accepts (`benchmarks/corpus/raw_skeletons*`,
  800 rows), zero diffs on a direct probe of every dropped rule's own LHS (raw and
  variable-instantiated), the corpus-invariance gate's complexity, idempotence,
  permutation and twin counts all identical (acj-4-3: 294,730,680 / 0 / 0 / 8), and a
  differential fuzz of
  the served engine against one force-fed the raw artifact over **65,536** random
  expressions (lengths 3–11 over the mine alphabet): **0 differences, 0 raised**.
- **Pins that move.** The translation census counts served patterns, so it drops
  with the duplicates: 6602/0/0 → **5553/0/0** (rows, complexity and twins unmoved).
  The gate and the loader move together, so no commit in between leaves the gate
  green against a number the code no longer produces.
  The in-suite twin of that pin (`tests/test_licence_registry.py`) is updated with
  the evidence.

### Changed — `parse` is now `read_infix` (the name states the contract)
- **`SimpliPyEngine.parse` is renamed to `SimpliPyEngine.read_infix`, and `parse` is
  removed.** The rename is not cosmetic: this reader is the only public entry that is
  **tolerant of unknown vocabulary** (an undeclared `sqrt(...)` survives as a bare
  leaf, which `to_prefix`/`to_infix`/`to_tagged`/`simplify` all refuse) and
  **preserves the input's spelling** — it never enters the canonical state, so
  `read_infix('x0+x0')` is `+ x0 x0` where `to_prefix('x0+x0')` is `* 2 x0`. Use
  `read_infix` when the input's own spelling is the data, `to_prefix` when the
  canonical spelling is. `mask_numbers=` and `convert_expression=` are unchanged
  on both names.

### Changed — the masking redundancy is cleared (one policy, one name)
- **`simplipy.masking.mask_values_keep_structure` is removed**; use `mask_fittable`. They were never two policies: the older name was this module's
  original spelling, `mask_fittable` landed as a rename, and the two have been the
  same function object ever since. There is no third semantics — "mask the values,
  keep the structure" partitions `Role` into `{COEFFICIENT, ADDEND, VALUE}` and
  `{EXPONENT, ROOT_INDEX}`, exactly what `mask_fittable` applies; the names state
  one criterion from two sides (what is kept, and why it must be). Because they were
  one object, nothing about the behaviour changes — only the name you call it by.
- **`simplipy.masking.__all__` now declares only the live surface** (`Role`,
  `literal_sites`, `mask`, `mask_all`, `mask_fittable`). The deprecated spelling is
  no longer advertised or bound by `from simplipy.masking import *`.

### Added — `engine.mask(..., collect=False)`
- **The front door reaches the toolkit's full behaviour.** `engine.mask` had no
  `collect` escape while `masking.mask` did, so the raw positional substitution was
  unreachable from the engine. The default (`collect=True`) is unchanged and stays
  byte-identical. The docstring now states what the collect stage does and costs:
  it re-runs the engine, which is what enforces one `<constant>` per degree of
  freedom (`2*x0/3` is one free value, not two), and therefore fires rules and
  re-imposes canonical order — terms may be re-ordered relative to the input
  spelling.

### Changed — `is_valid` accepts and verifies all three forms
- **`is_valid` now reads infix strings and the tagged dialect**, not only explicit
  binary prefix, detecting the form exactly as the conversion API does (type first,
  then dialect). This closes a real seam rather than adding convenience: the tagged
  form is the engine's own serialization *and* the v24 target format, and `is_valid`
  answered `False` for every well-formed tagged sequence — a wrong verdict, not a
  refusal. An infix `str` fared worse: the old call walked the string's
  **characters**, so `is_valid('x0+x0')` was `False`.
- Explicit binary-prefix verdicts are **unchanged**, including the sharp edges
  (`[]` is `False`; undeclared vocabulary is `False`; `tuple` input keeps working).
  Malformed input in any form is still a `False` verdict, never an exception; only a
  non-expression type raises `TypeError`. The verbose diagnostic that used to say
  "is_valid reads the explicit binary-prefix dialect" now explains a *malformed
  tagged* sequence instead, which is the only case that still reaches it.
- Note: `is_valid` verifies against the engine's vocabulary while `read_infix`
  tolerates it, so `read_infix('sqrt(x0)')` reads and `is_valid('sqrt(x0)')` is
  `False` when no `sqrt` is declared.

### Changed — an engine that ends up with no rules says so
- **`from_config` (and `load`) now warn on the resulting STATE, not just on a
  missing file.** An engine with zero simplification rules is fully functional — it
  parses, evaluates and returns canonical output — and simply never rewrites
  anything, so the old silence turned a broken setup into merely disappointing
  numbers. All three causes now warn: an unresolvable configured rules path, a
  config with no `rules` key, and a rules file that loads and contains nothing.
  The warning is **non-fatal**, as ruled.
- When the cause is an unresolvable path the message names **both** the literal
  config value and the resolved absolute path (previously only an unnormalized
  join, so `./missing.json` printed as `/dir/./missing.json`).
- Direct `SimpliPyEngine(operators=..., rules=[])` construction stays silent: the
  caller asked for a bare engine explicitly, and it is the sanctioned idiom for
  mining, pickling and testing.

## 0.13.1 — 2026-08-17

### Added — the fair-benchmark results
- The pre-registered three-corpus benchmark results are published: headline
  table and full ECDF panels in the simplify guide, summary in the README.
  Across 131,600 scored rows the sound engine never inflates an expression
  (0.00% made-bigger); paired median speedup vs SymPy's `simplify` is
  ~600–800× at 92–130 µs per row.

## 0.13.0 — 2026-08-17

### Added — the declared public API (`__all__`) and the compatibility policy
- **The public surface is now declared**: `__all__` at the package root and in
  `simplipy.utils` / `simplipy.io` / `simplipy.asset_manager` /
  `simplipy.engine` / `simplipy.mining`. What is declared is what the
  compatibility policy stabilizes; reachable-but-undeclared names carry no
  promise (being rendered by the docs generator never was one, and the API
  reference now renders only the declared surface). `from simplipy import *`
  no longer injects eight submodules into the caller's namespace — one of
  which (`simplipy.io`) shadowed the stdlib's `io`.
- **`simplipy.mining.RuleMiner`** — mining/certification now lives in its own
  module as an explicit object over an engine (`RuleMiner(engine)`).
  `SimpliPyEngine.find_rules` / `certify_rules` remain as delegators with
  identical signatures and semantics: the split is architectural (gated
  byte-identical mines), not a surface change. One miner per process, as
  before.
- **Deprecated**: `substitude_constants` (the historic misspelling) now warns;
  use `substitute_constants`. Removal not before 0.15.0. `numbers_to_constant`
  (warning since 0.12.0) is removed in 0.14.0; use
  `explicit_constant_placeholders`.

### Changed — clean-cut config rulings (D12, owner-ratified)
- **`pow1`/`pow_1` join `RETIRED_OPERATOR_TOKENS`.** A config declaring them was
  accepted while the parser unconditionally rewrote their tokens away — one engine,
  two answers for one expression. Artifact loading refuses them like the rest of the
  retired family; the parser contract (legacy *input* rewriting) is untouched.
- **The `inverse` operator-spec key is optional.** Nothing has read it since the
  relic layer's deletion — a required key that was pure ceremony made every wild
  config fail construction. Declared values are still accepted silently; the shipped
  acj configs no longer carry the key (their digests updated with the republish).
- **The torch shim stays** — its deletion was gated on a published-source grep, and
  the grep refused it: flash-ansr compiles simplipy expressions into torch-callable
  functions and feeds tensors through the operator realizations
  (`flash_ansr/utils/metrics.py`). The `[torch]` extra remains a documented opt-in.

### Removed — Python 3.11 support (floor moves to 3.12)
- **`requires-python` is `>=3.12`** (D35, owner-resolved). numpy 2.5 and scipy 1.18 both
  declare `>=3.12`, so a 3.11 environment silently resolved an *older* numeric stack
  than every developer and every other CI job — a real difference in the f64 algebra
  the equivalence contract is stated against, and it was invisible until CI began
  printing the resolved stack per job. The floor now matches the dependency reality;
  the abi3 wheel tag moves to `cp312`. Standing version policy recorded with the
  ruling: new-but-stable versions, reasoned floors, no unmeasured ceilings.

### Fixed — SOUND `simplify` no longer fabricates values it cannot certify
- **Fabricated `nan` for finite contract values.** With a `±1` base and an exponent whose
  enclosure reaches an infinity, the interval magnitude-step arms omitted the genuinely
  attained value 1 (`pow(t, ±inf) = 1` at `|t| = 1`), so `x0 + (-1)**(3/sin(np.pi))`
  returned `nan` where the contract value is `x0 + 1` — and the nan escaped into any
  enclosing expression. Measured 78 fabricating rows on the 61,200-ground judged sweep;
  now 0 in both SOUND and LOSSY.
- **Fabricated definite infinities.** A base the intervals could only bracket
  (`cos(np.pi)` encloses `[-1, -0.9999999999999991]`) drove the same arms to assert a
  definite `+inf` where the true value is 1: `x0 + pow(cos(np.pi), float("-inf"))`
  returned `inf`, annihilating `x0`. 46 rows measured; now 0 across the full
  1,560-row family sweep.
- **The `A − A → 0` certificate looked only inside a bounded box.** `finite_ae` proved
  finiteness over `[-R, R]^n` (R ≥ 64) and nothing looked outside, so
  `log(1 - 0.001*x1) - log(1 - 0.001*x1)` simplified to `0` — an expression that is nan
  for every `x1 > 1000`. A tail arm now discharges the complement (the claim covers the
  horizon failure class only; certificates lost to interval dependency loss remain out
  of scope, deliberately).
- **The interval box for odd negative exponents came back reflected** (Rust `%`
  truncates, so `k % 2 == 1` is false for negative odd k): `pow(x, -1)` over a range
  straddling zero missed its lower pole, and cancellations fired on nan-bearing
  expressions. An enclosure fuzz (every unary operator, dense grids, degenerate boxes)
  now pins the property.
- **`atanh` folds and boxes disagreed by up to 1.5e13 ulp.** The interval arm evaluated
  endpoints with Rust std's own `atanh` composition while the fold ships the system
  libm's; near ±1 the gap reaches the fourth significant digit, so the box could exclude
  the fold's own value. Both now call the same libm symbol.
- **Net recall cost, accepted deliberately:** a handful of grounds that used to fold by
  numerical luck now stay symbolic (`pow((-1), exp(40))`, `exp(-100000)`); an engine may
  over-refuse, an instrument may not convict.

### Changed — one owner for the odd-function literal sign (canonical spellings move)
- `-5 * sin(2)` and the same term built through a sum used to reach two different
  fixpoints (`-5·sin(2)` vs `5·sin(-2)`) — one value, two canonical states, decided by
  construction history. The sign placement is now priced by one shared orbit on every
  route: μ picks the cheaper spelling, exact ties resolve to the non-negative
  coefficient (`-1·sin(2)` files as `sin(-2)`; `-1·x0·sin(2)` keeps its free `-1`).
- The collector also learned the inverse: `sin(-2) + sin(2)` cancels to `0` in one pass
  (it previously needed a render/re-parse cycle).
- **Effect:** canonical spellings move on odd-function-of-literal shapes; the
  400-skeleton reference corpus is row-identical, and 30 of the previous artifact's
  rules become redundant respells (see Published artifacts).

### Changed — the complexity measure saturates instead of overflowing
- μ's linear literal schedule exhausts u64 around scale 5.5e15; both saturation
  entrances collided every larger literal at one price (`u64::MAX`), and a poisoned
  leaf overflowed the tree sums — a debug panic, or a release-mode wrap that priced a
  composite below its own parts (the direction that licenses wrong rewrites). Literals
  now price on a two-regime schedule (exact up to scale 2^32 — nine orders beyond any
  f64-derived literal — then ordered by the exponent's own digits), and accumulation
  saturates at the top, which can only refuse a rewrite, never license one.

### Changed — certification gates refuse what their judge cannot pass
- **`certify_rules` runs the same symbolic gate as the mine.** It certified
  `tanh(exp(e)) → 1` — false by a stable 1.37e-13, below any numeric tolerance — while
  the mining path killed the same pair. Accepted pairs are now re-judged and fatal
  verdicts refused, so "certified output is exactly as sound as mined rules" is earned,
  not asserted.
- **Every non-sound verdict is fatal at the gates, not only KILL.** A rule the judge
  cannot evaluate (`NO-WITNESS`, `UNSUPPORTED-SHAPE`, `JUDGE-TIMEOUT`) or cannot
  reconcile (`ENGINE-MISALIGN`, `UNRESOLVED-COVERAGE`) no longer ships with a
  certified rule's standing; the provenance sidecar records the full bucket census
  (every bucket, present even when empty). One scoped exemption: the multi-`<constant>`
  family the judge explicitly declares outside its jurisdiction (its soundness
  authority is the constant-fitting chain), recorded under its own sidecar key.
- **scipy's absence is a hard error in the miner, never a silent degradation** — it
  used to change mined artifacts with no record (a certifiable promotion read
  NO-WITNESS instead of PROMOTE purely on scipy's presence).

### Added — artifact identity: recorded environment, digests, and read-back checks
- **Every mine's provenance sidecar records its environment**: python, platform, libc,
  numpy/scipy/mpmath versions, and a `libm_fingerprint` — a digest of a fixed probe
  battery evaluated through the deployed folding path. The transcendental folds go
  through the system libm, which resolves on the running machine: measured, glibc 2.43
  is 1 ulp wrong on `cosh(acosh(2))` and mints a rule other hosts do not, so
  "byte-deterministically reproducible" is now stated — and checkable — *at the
  recorded environment*.
- **The asset manifest carries per-file `sha256` and a pinned `revision`, and both are
  enforced**: installs download at the pinned revision and verify what landed;
  resolution re-verifies the cache, so corrupting a byte of a cached `rules.json`
  raises instead of silently serving. (Manifest entries without digests remain
  permissive, so older manifests keep working.)
- **The measure fingerprint is read back at load**: an engine loading a ruleset mined
  under a different μ now warns loudly (the rules stay sound; their minimality claims
  are what a measure change voids).

### Fixed — interval and certificate hardening (mining instrumentation)
- Mixed-infinity products propagate their sign flags per-sign; a `b_mul` collapse could
  previously flip a definite sign class on expressions reaching both infinities.
- The certificate algebra widened (denominator-clearing certificates, unconditional
  coefficient splits), moving a measured set of trapped towers to cheaper forms.
- Masking policy left the mechanical helpers: `explicit_constant_placeholders` no longer
  converts digit-only tokens by default (`convert_numbers_to_constant` flipped
  `True → False` and is deprecated — masking decisions belong to `simplipy.masking`,
  the helpers stay mechanical).

### Published artifacts
- **acj-4-3 re-mined and republished** under the 0.13.0-train engine: 6,671 → 6,594
  rules. Every removed row is attributed: 60 odd-function sign-spelling twins made
  redundant by the shared sign owner (their sources still reduce natively to the same
  values), and 17 rules the widened symbolic gate refuses (`exp(pow((-10), k)) → 0`
  shipped an f64 underflow-rounding as an exact rewrite; verdict UNRESOLVED-COVERAGE).
  Minted on a correctly-rounding libm host — the previous interim mines on the dev box
  carried a `cosh(acosh(2)) → 2` rule that exists only because its glibc is 1 ulp
  wrong there — ×3 byte-identical, with the environment and `libm_fingerprint`
  recorded in the sidecar and per-file `sha256` + pinned `revision` in the manifest.
  The 400-skeleton reference corpus is row-identical under the new engine + artifact;
  `μ`-non-increase and idempotence hold 400/400.
- Interim (2026-08-11, superseded by the above): acj-4-3 6,661 → 6,671 (+10 adopted
  post-F83 rules, each judge-verified with zero pointwise exceptional-point
  disagreements).

### Docs
- **The "never longer than the input" guarantee was false and is gone.** `simplify`
  guarantees μ-non-increase, soundness, and idempotence — never an output-token bound:
  77 of 400 reference outputs are token-longer in the explicit form (71 exact μ ties,
  6 strictly μ-cheaper yet longer), and the tagged form adds bag delimiters on top
  (264/400). The pipeline page now states the μ-guarantee with the measured caveat,
  and a test pins it 400/400.
- The independent monitor's verdict vocabulary is API-visible and documented:
  `UNSCORED` (an output the judge cannot read never silently passes as OK) and the
  singular-scoped violation clause.

### Known performance
- The serve path carries a measured throughput regression (−32%/−43.7% on the
  benchmark corpora) inherited with the certificate widening: the `ac_zsn` witness
  is re-computed where the neighbouring `finite_ae` is doubly cached (4.78 ms vs
  0.19 ms). Disclosed here deliberately and scheduled for a cache, not silently
  carried.


### Security — the realization trust model
- **A config file is no longer executable input.** Operator realizations name Python
  modules, and the engine imported every root it found — which runs that module's
  top-level code, before any expression is evaluated, on every path including
  `SimpliPyEngine.load` (which fetches a config from Hugging Face) and unpickling.
  Realization module roots are now checked against an allowlist BEFORE anything is
  imported: `math`, `np`, `scipy`, `simplipy`. Anything else refuses at construction,
  naming the operator that asked for it.
- **One spelling per module: `np`, not `numpy`.** `np.pi` and `np.e` are normative token
  grammar, so `np` is the canonical form; a realization written `numpy.sin` is refused
  with a hint pointing at `np.sin`.
- **Trust is granted from outside the config, never by the config** (a `trusted_modules:`
  key in the YAML would let a hostile file authorize itself, and is ignored). Two
  surfaces: `SimpliPyEngine.from_config(path, trusted_modules=[...])` (also on `load`
  and the constructor) and the `SIMPLIPY_TRUSTED_MODULES` environment variable. The
  resolved trust travels with the engine through pickling, so a spawn worker imports
  exactly what the parent was allowed to import.
- **Each engine evaluates in its own namespace.** `code_to_lambda` bound expressions to
  `simplipy.engine`'s module globals, so one engine's imports were reachable from another
  engine's expressions, as were simplipy's own imports (`os`, `importlib`, `hashlib`, …).
  Expressions now see only the modules their engine needed plus `np`.
  **API note:** `code_to_lambda` is an instance method instead of a `staticmethod`.
  Instance calls (`engine.code_to_lambda(code)` — the documented usage) are unaffected;
  a call on the class now needs an instance.
- **Rule files are data, never code.** `verify_ruleset`'s literal reader evaluated
  rule-file tokens with `eval()`, so a hostile `rules.json` could execute Python at
  verification time. Literals now go through a total, eval-free acceptor (numbers,
  `(-N)`, `float("inf"/"-inf"/"nan")`, `np.pi`/`np.e`, `p/q`); anything else refuses
  as `UNSUPPORTED-SHAPE`. Measured against the shipped artifact: 0.00% of 18,238
  literal occurrences refused.
- **Inline `lambda` realizations are refused at load.** A config could smuggle
  arbitrary Python source into generated code through a `realization: 'lambda x: ...'`
  string; realizations must now name a callable in a trusted module.
- Scope, stated plainly: this makes a config safe to LOAD and scopes what a compiled
  expression can SEE. It does not make `codify`/`code_to_lambda` safe against a hostile
  EXPRESSION; evaluating attacker-supplied source is unsafe by construction in Python.

### Removed — `simplify(inplace=...)`
- **The `inplace` parameter of `SimpliPyEngine.simplify` is gone.** It had three
  different behaviors behind one flag (mutate the input list, raise for ndarray input,
  silently do nothing for `form='infix'`), no callers anywhere, and no performance
  content — the compiled core produces a fresh result regardless, so `inplace=True` was
  sugar for copying that result back into the caller's list. Passing it now raises
  `TypeError` like any unknown keyword. The `inplace` parameters of the utils helpers
  (`numbers_to_constant`, `substitude_constants`, `explicit_constant_placeholders`) are
  unaffected — those are genuine in-place list rewrites and downstream code uses them.

### Fixed — the independent monitor no longer fabricates verdicts on extreme-scale expressions
- **`simplipy.verify` judges honestly at every magnitude the token grammar reaches.**
  The independent monitor's judge (the end-to-end soundness gate) carried five
  instrument defects that each fabricated violation verdicts against correct
  rewrites once expressions carried extreme literals (`1e309`, `5e-324`,
  `2^127`-scale rationals): a refused evaluation was converted into a fabricated
  nan instead of a skipped probe; magnitudes beyond 10^(1e50) were refused although
  they evaluate exactly in milliseconds (the cap is now 10^(1e2000), with measured
  cost bounds); the literal zero-snap tested only outermost constant subtrees at two
  hardcoded precisions, so a true nonzero difference spanning 272 digits read as an
  exact zero — the judge then rewrote the input to `0` and convicted the correct
  output — while inner exact zeros (`tan(pi)`) were never recognized; the working
  precision was a flat 50 digits, so `cos(1e-320)` read as exactly 1; and values
  that ROUND to a decision boundary at any precision (`tanh(9e15)` is 10^-8e15
  below 1) fabricated the boundary value instead of refusing. The judge now folds
  exact-rational subtrees in unbounded rational arithmetic, adapts its working
  precision to each pair's own literal exponent span, resolves constant subtrees to
  provable zeros/integers on an escalating precision ladder, and refuses honestly
  (probe skipped, refusal recorded) where no finite precision can decide.
  Small-literal expressions — everything a typical deployment sees — are judged at
  the same precision and cost as before. On the ordinary 1M random-expression
  corpus no row convicts that did not convict before, and 96 of the 98 previous
  convictions plus 104 previous refusals now judge OK; on a 1M extreme-literal
  stress stream, 164 of 229 convictions were measured to be instrument
  fabrications and now judge OK, and 43 more become recorded refusals instead of
  convictions.

### Fixed — the tagged serialization could read back a different value on extreme states
- **A sign could vanish in the tagged rendering of an overflow-partition state.** When
  an expression's exact rational content is split across several members (folding it
  would overflow the exact arithmetic — only reachable with extreme literals around
  10^19-squared and beyond), the tagged emitter pooled their reciprocals into the
  `<div>` section and tracked the sign with a per-member flag that a second signed
  member silently overwrote. The rendering then re-parsed to a DIFFERENT value
  (measured: six of one million random extreme expressions; e.g. an expression worth
  −2/3 serialized to a spelling worth +2/3), while the engine's internal state and the
  explicit/infix renders stayed correct. Partition members now serialize
  self-contained in every dialect, the round-trip is the identity on them, and a new
  fuzz battery permanently screens the tagged rendering's value against the input.
  Ordinary expressions are unaffected (zero changed renders across 1M+64k
  random-expression corpora; the shipped rule artifact re-mines byte-identical).

### Changed — sign decisions are arrival-invariant (canonical forms move slightly)
- **Two ways the canonical form could depend on HOW an expression arrived are gone.**
  (1) When a sign placement was fully μ-tied (mirror pairings like
  `-1 · (a-b)(c-d)` vs `(b-a)(d-c)` price identically), the tie broke on token
  interning order — an artifact of which literal the input mentioned first — so the
  same expression could canonize differently depending on operand order, and the same
  config could canonize differently across vocabulary orderings. The tie now breaks on
  the canonical content order (string-based). (2) When a product's exact rational
  content is split across several members because folding it would overflow the exact
  arithmetic, the minus sign stayed on whichever member carried it on arrival; the
  split and the sign host are now functions of the value alone.
- **Effect — the full upgrade differential, re-measured at publish time** (the
  published 0.12.0 wheel + its artifact vs this release + the re-mined artifact,
  split by cause under the current measure):
  - *400-row reference corpus* (`benchmarks/corpus/raw_skeletons_nv.json`):
    **140/400 = 35%** of outputs differ — 109 equal-μ mirror respells, 21 strictly
    simplifying, 10 μ-costlier than the 0.12.0 output; 12 outputs are token-LONGER.
  - *Frozen fuzz lane* (50,000 rows, the `fuzz_properties.py` generator, seeds fixed
    since 2026-08-03): **1,926/50,000 = 3.85%** differ — 1,205 mirror respells,
    575 strictly simplifying, 146 μ-costlier; 221 token-longer.
  - Stated plainly: **some outputs get longer or less reduced.** The μ-costlier rows
    are the deliberate soundness price — folds and cancellations this release refuses
    to certify (fabricated-value fixes, the widened gates) leave those expressions
    symbolic where 0.12.0 collapsed them. Within one engine the guarantee
    `complexity(simplify(e)) ≤ complexity(e)` holds (400/400, pinned); the μ-costlier
    rows above compare across releases, against outputs the old engine should not
    have produced.
  - A training corpus generated under 0.12.0 is NOT reproducible under 0.13.0 —
    regenerate rather than mix.
  - *Existing 0.12.0 installs are unaffected by the artifact republish alone*:
    the 0.12.0 wheel loads the re-mined artifact loss-free (6,604 translations,
    0 subsumed, 0 dropped) and its reference-corpus outputs are byte-identical
    (0/400 moved) — the engine changes above require the new wheel.

### Changed — pickling refuses a diverged rule state
- **An engine whose `simplification_rules` and compiled core disagree no longer
  pickles silently.** The documented contract is mutate-the-list-then-`compile_rules()`;
  nothing verified it, and a pickle rebuilds workers from the list — so a diverged
  parent could spawn workers running different rules with no warning (measured: 50 of
  400 corpus expressions simplified differently between a parent and its own unpickled
  worker). `pickle`/`deepcopy` now raise a `ValueError` naming the fix
  (`compile_rules()`); engines that follow the documented contract are unaffected.

### Changed — operator specs are validated loudly at load
- **A spec missing a required key is a config error naming the operator and the key(s)**
  (`realization`, `alias`, `inverse`, `arity`, `commutative`). It used to die as a bare
  `KeyError: 'alias'` pointing at nothing. `precedence` is documented as optional for
  non-core operators: they render as function calls, which never consult it.
- **`commutative: true` on anything but `+`/`*` is refused at load.** The engine consumed
  the flag for nothing else — only `+` and `*` canonicalize as AC bags — so `f(a, b)` and
  `f(b, a)` stayed two distinct canonical states while the config claimed one value. A
  declaration the engine silently ignores is worse than a refusal. (No known config
  declares one: measured across every config in the repo and downstream.)
- **A core serialization token declared without `precedence:` now gets the core's own
  value** instead of slipping past the conflicting-precedence guard, which only compared
  values that were present. A conflicting declared value still refuses, as before.

## 0.12.0 — 2026-08-08

### Removed — the legacy kernel (the clean release)
- **The binary simplify kernel is deleted; the AC engine IS the engine.** `simplify_ac`
  becomes `SimpliPyEngine.simplify` (same signature: `form` ∈ tagged/infix/explicit,
  `mode`, `node_budget`); the old best-first tree-search kernel, `cancel_terms`, the
  operand-sort pass (except inside `mask`), the old pattern matcher, `apply_rules`,
  `cancel_only`, `sort_only`, `prune_redundant_rules`/`prune-rules` (the wildcard-shadow
  prune), the old-kernel counters and the `free_coefficients` axis are all gone. The
  hyper-operator realization functions left `simplipy.operators` with them, so LEGACY
  CONFIGS NO LONGER LOAD: cross-version comparison runs against a pip-pinned previous
  release in a separate environment, never against embedded legacy code. Masking survives
  as a current feature applied downstream of simplify, rebuilt as the `simplipy.masking`
  module (see below). The supported pairing is the AC-JUDGED artifact family
  (`acj-2-1`/`acj-3-2`/`acj-4-3`, published on the Hugging Face assets repository); the
  test suites run on `acj-4-3`.

### Removed — the hyper-operator vocabulary
- **`mult2..5`, `div2..5`, `pow2..5`, `pow1_2..pow1_5` are deleted from the vocabulary, once
  and for all.** The base operator config drops from 38 to 23 operators; coefficients and
  exponents are explicit exact literals (`* 6 x`, `pow x 3`), and the general
  signed root `rootn(x, n)` — an engine BUILT-IN, independent of any config — replaces the
  odd-root family (`rootn x 3`, with arbitrary odd indices now expressible). Every layer
  understands `rootn` natively: the interval certificates, the f64 evaluator, and the
  high-precision battery all gained a first-class binary arm (odd integer index, everything
  else fail-closed NaN), so the former `rootn -> pow1_3/pow1_5` re-sugaring in the explicit
  projection is deleted along with the **entire bounded projection** (`form="bounded"`),
  whose only purpose was hyper-operator re-sugaring.

  STRICTLY CLEAN: legacy spellings have NO config-independent reading. They parse only
  under a legacy CONFIG that still declares them (dying with the legacy kernel in the
  clean release), and legacy corpora convert ONCE at the boundary instead —
  `benchmarks/corpus/convert_nv.py` is the pure respeller (`mult3 t -> * 3 t`,
  `pow5 t -> pow t 5`, `pow1_3 t -> rootn t 3`; no simplification, every raw redex kept),
  and `raw_skeletons_nv.json` is the converted benchmark corpus. A new-vocabulary engine
  REFUSES legacy input outright. Downstream data and model-output migration is
  flash-ansr's side of the boundary. Re-mining under the new vocabulary confirmed the
  economics: most of the legacy ruleset was vocabulary artifact — the shipped counts are
  under "Published artifacts" below.

### Added — the AC engine: n-ary associative-commutative simplification
- **`+` and `*` become flat, sorted n-ary bags with EXACT rational coefficients** composed
  explicitly (`* 7 x`, `pow x 3`); `-`, `/`, `neg`, `inv` and the hyper-operator family
  desugar at the boundary and do not exist in the core. Rules are widened to SUB-MULTISET
  matching with the unmatched remainder preserved, so a rule fires wherever the algebra
  permits, independent of operand order and bracketing:

  * both axes of the binary engine's commutative-order invariance defect are removed at the
    representation level — `* cos A tan A` and `* tan A cos A` both give `sin A`, and
    `x3 + (x8 + x3/5)` collects the two `x3` terms across the bracketing;
  * `simplify` output is invariant under permutation of commutative operands and idempotent
    by construction within the exact-coefficient boundary (verified corpus-wide; see "Known
    boundary" below for the one documented exception class at i128-overflow literal scales);
  * coefficient arithmetic is exact rational computation instead of mined rules, so the
    arithmetic-identity rule families of earlier artifacts are simply derived;
  * like-term/like-factor collection (the AC form of term cancellation) runs inside the
    canonical constructors under the SAME soundness certificates as the rule matcher
    (finite-a.e. for sign-cancelling addition, finite-and-nonzero-a.e. for exponent-cancelling
    multiplication), each gate documented with its counterexample — including the branch-cut
    gate (`x^(1/2) * x^(1/2)` does not merge to `x`) and the sign-erasure gate
    (`(x^2)^(1/2)` does not become `x`);
  * pole identities resolve exactly under the supported pairing — `inf - inf` is `nan`,
    `inf / inf` is `nan` on every spelling — via the certificate sorts plus the exact fold;
    ground (variable- and constant-free) expressions are additionally never licensed by the
    a.e. collection gates (an expression with no measure space has nothing for "almost
    everywhere" to quantify over), keeping ground soundness independent of certificate edge
    behavior;
  * the native prefix serialization is the STRICT TAGGED form (default for token inputs):
    n-ary bags are delimited (`<add> ... </add>`, `<mul> ... </mul>`) and carry their group
    inverse as a SECTION — terms after `<sub>` subtract, factors after `<div>` divide, so
    `(2*x1)/(x2*x3)` is `<mul> 2 x1 <div> x2 x3 </mul>` and bags contain no negative
    literals and no inverse operators; `pow` and the unary functions stay plain prefix;
    `neg`/`inv` remain only as the standalone unary spellings (`tan neg x0`, `inv x0`).
    Integer literals are one token each; a non-unit in-vocabulary fraction spells
    STRUCTURALLY through the bag's divide section (`2/3 * x0` is
    `<mul> 2 x0 <div> 3 </mul>`), so a tokenized consumer never needs fraction tokens,
    and out-of-vocabulary rationals fall back to the cheapest atomic spelling
    (`0.12345`); the structural-fraction bound is the declared integer vocabulary
    (default 10, `SIMPLIPY_TAGGED_FRACTION_MAX`). Tagged output is accepted back
    as input (one shared, liberal parser: in-bag `neg`, negative literals and `pow x -1`
    spellings all parse to the same canonical state). Two further
    projections of the same canonical answer: `form='infix'` — a PRETTY human-readable
    rendering (`x8 + 1.2*x3`, `-x0/3`, `(x0 + 1)^2`; the default for `str` inputs);
    `form='explicit'` — the binary-chain diagnostic form with literal coefficients.
    `Mode.LOSSY` relaxes every certificate exactly as in the strict mode.

  The internal canonical form is UNIQUE: bags order by STRIPPED keys (Add terms by their
  coefficient-stripped structural key with constant-like terms last — `x + 1`, the polynomial
  convention; Mul factors by exponent-stripped base), and every canonical sum is PRIMITIVE —
  positive lead coefficient, no unanimous coefficient magnitude, the extracted content
  wrapping the sum (`-a - b` is `-1 * (a + b)`; `a/3 + b/3` is `1/3 * (a + b)`; sums with no
  unanimous content, like `x8 + 1.2*x3`, are untouched, so extraction never inflates tokens).
  Factored sums entering an addition unfactor, so the form is independent of association
  order; sign and grouping spellings of one function converge in the constructors themselves,
  and parse -> canonicalize -> serialize is the identity on every canonical state (asserted
  corpus-wide in debug builds).

  **Supported pairing.** The engine is supported with SORT-PROMOTED rulesets: their
  cancellation rules carry the certificate sorts, so pole identities resolve exactly
  (`inf - inf -> nan`) under the AC matcher. In this release that means the published
  `acj-2-1`/`acj-3-2`/`acj-4-3` artifacts (see "Published artifacts" below); the pre-0.12
  artifacts (`2-1`/`3-2`/`4-3`, `dev_*`) use the retired hyper-operator vocabulary and no
  longer load at all (see the artifact-generation gate below).

### Changed — the reduction ordering is a description length (μ)
- **`SimpliPyEngine.complexity()` and the search objective price the canonical state by μ,
  its coding cost in milli-bits under the token grammar.** A variable costs 8.000 bits, a
  magnitude-1 coefficient is free, and a rational prices as its fraction code
  (`1/2` = 2.585 bits, `355/113` = 15.309 bits); `<constant>` prices at the free-constant
  bound. The decimal code is PRINT-only — it never enters the measure — and the printer
  follows the cheaper code, so `0.5*x0` emits `x0/2` and `1/2` emits `/ 1 2` in the explicit
  dialect, while `0.2` and `0.12345` keep their decimal spelling. Every rewrite must
  strictly descend in (μ, tie-break) — a Knuth–Bendix orientation that makes termination a
  theorem and `μ(simplify(e)) <= μ(e)` a corpus-verified property. Loaded rules are
  re-oriented under μ at load time, so rules mined under any earlier ordering stay aligned
  with the ordering the engine actually fires under.

### Changed — the mining alphabet is the deployed literal vocabulary
- The mine's source AND target alphabet is now the integers −10..10, `np.e`, `np.pi` and
  the IEEE specials — the same literal vocabulary downstream tokenized consumers declare —
  instead of a nine-term development alphabet. Adopted over a measured cost, deliberately:
  the full alphabet costs about 10% simplify throughput for a 7x larger ruleset, and
  completeness over the deployed vocabulary is the goal, not the margin. The complete
  mining universes at the shipped tier are 13,427 length-3 and 399,823 length-4 sources.

### Added — `simplipy.masking`: role-aware literal masking as a module
- **`SimpliPyEngine.mask` is deleted; masking is representation, applied downstream of
  simplify, and lives in `simplipy.masking`.** The module provides `literal_sites` — a
  role-aware walk over both prefix dialects classifying every literal position
  (COEFFICIENT / ADDEND / EXPONENT / ROOT_INDEX / VALUE, with `neg`/`inv` transparent) —
  plus `mask(tokens, engine, policy)`, `mask_all`, and `mask_values_keep_structure`.
  Policies decide per (value, role); the engine's operator table is read directly, never
  through a silent default.

### Added — the artifact-generation gate: generation-1 artifacts refuse at load
- The 0.12 compatibility claim ("legacy configs no longer load") is ENFORCED, not
  aspirational (`simplipy.compat`): artifacts carry an explicit `engine_generation` pin in
  `config.yaml` (generation 2 = the AC engine's clean 23-operator vocabulary; the acj-*
  family and `base` are pinned), the package carries the allowlist
  (`SUPPORTED_ENGINE_GENERATIONS = {2}`), and the refusal is MUTUAL with a actionable
  message in both directions (an older artifact points at `pip install "simplipy<0.12"`,
  a newer one at upgrading simplipy). Un-pinned configs are classified by vocabulary —
  any retired hyper-operator token means generation 1 — so every already-published legacy
  artifact refuses without republishing.

### Added — soundness and measure provenance ships with every artifact
- Every mined `rules.json` is accompanied by `rules.json.provenance.json` carrying the mine
  parameters, the core build stamp (version + git revision), the soundness state — the
  certificate kill-switch states, every artifact-affecting environment override recorded
  verbatim (a registered, single-place switch list), and the interval fail-closed miss
  counters — and the measure fingerprint (unit, the μ constants, probe values, digest), so
  artifacts mined under different orderings are distinguishable from provenance alone.

### Fixed — soundness hardening across the numeric stack
- **The interval layer is rigorous in rounding.** Every computed endpoint steps outward
  (IEEE arithmetic by 1 ulp, libm calls by 8, integer powers exponent-scaled); exact values
  never step (semantic constants, exact libm hits, a-priori range bounds applied as
  post-widening clamps). This retired a class of f64-precision verdicts the engine itself
  used to ship: exact-zero compositions such as `acosh(acos(cos 1))` no longer fold to
  `nan`, and the rendered-overflow pole rules that overflow at f64's `tan(f64(π/2))` are
  gone from the published family.
- **Ground expressions classify exactly.** Integer literals of any magnitude read sign and
  parity from the spelling (`(-inf)^1e40` is `+inf`; a finite negative base at an integer
  exponent refuses instead of asserting `nan`); a non-finite f64 verdict inside the miner
  must be corroborated by the honest interval class before it may mint.
- **Special constants never round into bits.** A ground source that denotes π or e — by
  spelling or by computed VALUE (`acos(-1)`) — stays symbolic; exact collapses ship as
  certified rules, never as f64 decimals.
- **Every token-taking entry is guarded.** The recursion cap covers all FFI entries and the
  deep-infix parser (interpreter aborts on ~200k-deep chains became `ValueError`s); mode
  strings validate strictly; malformed tagged or `rootn` input raises instead of passing
  through unchanged; `load_config` returns values VERBATIM (path resolution moved to each
  key's consumer — a small public API break).
- **The verify judges speak the full 0.12 language** (tagged bags, rationals, `rootn`) and
  fail closed on anything they cannot read; the shipping family carries zero killed
  verdicts under the ruleset gate.

### Changed — rule families become derivations
- The constructors now derive natively what earlier artifacts said as rule families, each
  landed only after the contract judge certified the family and each verified by triple
  byte-identical re-mines with unchanged scale-gate outcomes: sign-blindness as a
  propagating CONTEXT (an even head plants it, an odd function carries it inward, `abs` is
  a no-op inside it, an integer power passes it through — one recursion replacing thirty
  rules, reaching depths the mine cannot); powers of `exp` compose
  (`(e^a)^b = e^(a*b)` wherever the product folds, reached from both `pow` and `rootn`,
  including `exp(1) = e`); the inverse pairs `f(g(t)) = t` per a certified table; the
  reciprocal-base power arms; and the half-period shift `f(t ± π)` for sin/cos/tan at
  coefficient exactly ±1. Across the arc the 4-3 artifact went from 7,153 rules at the
  alphabet expansion to 6,661 shipped — every drop a rule the engine now simply derives.

### Published artifacts
- **`acj-2-1` (26 rules), `acj-3-2` (106), `acj-4-3` (6,661)** on the Hugging Face assets
  repository — the complete AC-judged mines of the clean 23-operator vocabulary at their
  tiers, covered-pruned and sort-promoted, byte-deterministic from the co-located
  `mine.yaml` with one command (three independent mines byte-identical per cell). Each
  ships `config.yaml` (generation-pinned), `rules.json`, the provenance sidecar and
  `mine.yaml`. The hosted family was republished 2026-08-08, superseding an earlier upload
  that predated the interval-rounding hardening and still contained 13 rules since retired
  as uncertifiable or unsound. `base` is the bare 23-operator starting config for fresh
  mining. All pair with simplipy >= 0.12; `load(name, install=True)` fetches and verifies
  by manifest.

### Verified
- Structural gates at 65,536 and 1,000,000 expressions: 0 errors, 0 idempotence failures,
  0 permutation failures, every numeric-divergence flag adjudicated sound; the tracked
  400-row reference corpus pinned exactly; 129 Rust tests (release and debug profiles) and
  650 Python tests green; clippy clean.
- **Known boundary (documented, unchanged in practice):** when two literals *both* refuse
  to fold because their exact sum or product would overflow i128, their relative bag order
  can depend on construction history; randomized fuzzing at ≥ 1e19-scale literal pairs
  shows 2 idempotence violations in 50,000 calls (none reachable from mining, corpus, or
  deployment literal scales; the cure — arbitrary-precision coefficients — is a design
  decision deferred until it matters).

### Docs
- `docs/formal.md` — the formal specification of the engine as a normalized term-rewriting
  system modulo AC: the canonical form, the μ ordering, the licence ledger with each
  claim's status (theorem / by-construction / enforced / empirical), and a change-impact
  map. README and quickstarts run on the acj family with executed outputs.

## 0.11.0 — 2026-07-26

### Fixed
- **The mine no longer certifies rules that disagree with their source at a constant binding.**
  A `<constant>` slot is chosen by the fitter, not drawn from a continuum, so a single
  disagreeing value carries the full data measure: where the source is defined, the target must
  be defined and exactly equal, with no measure tolerance. Four defects let violations through,
  all now closed:
  - **Stage-2 confirmation was vacuous for constant-bearing sources.** It asked "does a target
    exist?" instead of "is it the target we mined?", because the search's all-constant
    short-circuit returns a class literal without reading the candidate list. It now compares
    the returned target against the one under confirmation.
  - **A refuter for literal targets.** Variable-free constant-bearing sources are evaluated at
    exact witnesses and must agree in the extended reals (NaN with NaN, sign-exact infinities,
    exact finite values) before a literal target can be minted.
  - **A collapse licence.** A source may collapse to a single term only where its behaviour is
    unambiguous: the candidate must preserve the source's value class, and a bare `<constant>`
    target additionally requires the source to be finite on a set of positive measure. This
    replaces a narrower special-case guard.
  - **Witness closure under inverse images.** Some sources are finite only at the *preimage* of
    a corner point (`acosh(sin(C/3))` only where `sin(C/3) >= 1`, i.e. `C = 3pi/2 + 6k·pi`),
    which no amount of dense sampling reaches. Witnesses are now closed by pulling operator
    corners back through each unary in the source, with period shifts for periodic inverses.

  Measured on a complete enumeration of every variable-free `<constant>`-bearing source at
  length <= 4 (45,560 sources): **0 rules disagree with their source**, against 24 in the
  previously published rule set. Freshly mined cells are correspondingly cleaner; deployed
  artifacts change only when re-mined.

### Added
- **Ladder re-use: `find_rules(snapshot_at={length: path})`.** The cells of one
  `max_target_pattern_length` form a prefix chain -- mining `(7,4)` already does all the work of
  `(5,4)` and `(6,4)` on the way up, because lengths are mined shortest-first, each source's
  seed is indexed by its length alone rather than by how far the climb goes, the master seeds
  are drawn before the universe is built, and enumeration consumes no randomness. A climb can
  now emit each shorter cell as it passes through: the full post-pass (proposals, prune,
  promotion) runs on a copy of the mine, that cell's artifact and sidecar are written, and the
  raw un-pruned state is restored so the climb continues unaffected. Each snapshot's sidecar
  carries a `ladder_snapshot` record naming the climb it came from, and its parameters and
  universe census describe the cell rather than the climb. Verified byte-identical to one-shot
  mines of the same cells, including with a sampled length above the snapshot. Also exposed as
  the `snapshot_at:` key in the find-rules config.
- **Peak memory during mining is lower.** Each length's source universe is released as soon as
  it has been linearised; at the top lengths that set is the largest object in the process.
- **Ladder seed `* 0 !0 -> 0`** (the multiplicative zero absorber), certified through the same
  `judge_bang` bar as the other seeds. Commutative operands are canonically ordered
  variable < number < composite, so the enumerable carrier `* 0 x0` canonicalises to `* x0 0`
  and the mine only ever mints the right-hand form. But "zero times a *composite*" is
  canonically literal-first, so the left form is the shape that actually occurs, and the only
  carriers a cell can enumerate for it (`* 0 <unary> <leaf>`) mint per-operator rules that miss
  every binary operand. Without the seed, 213 of 65,536 corpus expressions simplified less --
  one 27-token expression reaching 17 tokens instead of 1. This restores it: mean length ratio
  0.8999 -> 0.8989, with 213 regressed rows falling to 18. Same situation as the additive-cancel
  seeds, whose carriers are pre-cancelled rather than canonicalised away.

### Changed
- **`simplify()` is ~4% faster: the hot-path observability is now a compile-time feature, off by
  default.** `simplify_counters()`'s counters sat in the innermost loops of the released wheel --
  `pattern_attempts` alone is one relaxed `fetch_add` per candidate rule tried, ~2.1k per
  expression on the published 4-3 rule set -- and the `nanos_*` phase timers cost two clock reads
  per bracket. Two independent measurements, both on the published 4-3 rule set: min-of-30 over 6
  interleaved rounds at two corpus sizes gave 99.4 -> 94.7 us/expr at n=8192 and 99.4 -> 94.5 at
  n=2048 (**-4.6%**, faster in 6/6 rounds at both sizes, counters ~4.0 us and timers ~0.5 us of
  that); an independent min-of-7 on n=6000 gave 99.80 -> 95.75 us/expr (**-4.1%**, median
  103.41 -> 100.67). Take the effect as **4-5%**: the interleaved protocol is the better one, and
  the weaker protocol lands slightly lower, as expected. Output is token-identical -- one sha256
  over 5,000 corpus rows agrees across HEAD, the default build and the `profiling` build, while
  `simplify_counters()` reads all zeros in the default build and non-zero under `profiling`.
  `simplify_counters()` therefore reads all zeros in a released wheel; build the profiling
  extension to read it: `maturin develop --release --features profiling` (or the individual
  `stats-counters` / `stats-timers`). The mining-progress counters are untouched -- they are off
  the simplify hot path.

### Fixed
- **`nanos_rules` is recorded again.** The counter was declared and exported but had no recording
  site since the tree-search refactor deleted the block its timer lived in, so
  `simplify_counters()` reported the rule-application pass as 0 ns while every other phase
  reported honestly, and share-of-runtime readings silently attributed the whole rules pass to the
  unaccounted remainder. It is now bracketed at the same site as `nanos_cancel`, which makes the
  two edge generators directly comparable: on the published 4-3 rule set the rules pass is 55.2 of
  95.2 us/expr (58%), cancellation 24.5 (26%), mask+sort 9.0 (9%), leaving 6.5 us (7%)
  unaccounted; that works out to ~27 ns per candidate-rule attempt, and the same ~27 ns on the
  much larger dev 7-3 rule set at 10x the attempts per expression.

## 0.10.1 — 2026-07-26

### Added
- **Per-candidate verdict trail for the proposal channel.** `find_rules(proposals=...)` now
  writes `proposals.trail` into the provenance sidecar: one entry per proposal, in file order,
  as `{source, target, verdict, stage, certificate}`. The aggregate tally it accompanies cannot
  be audited on its own -- "93 rejected" never says which candidate died at which gate, so a
  reviewer cannot separate a correctly killed hallucination from a wrongly killed identity.
  `stage` names the deciding gate: `vocabulary` (token outside the alphabet, or malformed),
  `covered` (the mined rules already shorten the source), `search` (no library target and no
  verifiable hint), `confirm` (failed independent stage-2 re-verification), `merge` (certified
  but folded away as a duplicate), or `accepted`. No certification semantics change.

## 0.10.0 — 2026-07-26

### Added
- **The certificate algebra: structural null-measure certificates.** A new predicate family in
  the interval engine — `zero_set_null` (identity-theorem witness for entire-analytic
  compositions plus structural recursion), `nonfinite_null` (pole-set tracking through the
  domain-restricted unaries), `finite_nonzero_ae` (= the two combined: the regular domain of
  multiplication), and `positive_ae` (v1) — closing the class of binding-dependent side
  conditions. `finite_ae` now certifies through the structural path as well as box subdivision.
- **The `$` wildcard sort: certified multiplicative cancellation.** `$N` binds composite
  subtrees gated by `finite_nonzero_ae`, licensing `A/A -> 1`, `A·inv(A) -> 1`, and
  `0/A -> 0` exactly where they are sound (`cosh(x)/cosh(x) -> 1`; `asin(x)/asin(x)` and
  `0/0`-bearing trees refuse). Ships with `judge_bang_mult` (the promotion bar on the
  finite-nonzero atom lattice) and four ladder seeds; deployed artifacts are unchanged until
  the next production mine.

### Fixed
- **Promotion oracle non-termination (B4).** `_literal_spans` could loop forever with unbounded
  memory when the operator table was empty or contained unknown ops (unvisited `end[]` entries
  sent the scan backwards). A termination guard plus a fail-fast `RuntimeError` in `prefold`
  when the oracle is unconfigured close both the hang and the silent misconfiguration path.

### Removed
- **BREAKING: the `max_pattern_length` parameter of `simplify` (and `apply_rules`) is gone.**
  Rule application always considers every pattern in the loaded artifact; the scan window is
  the ruleset's own longest pattern, an intrinsic property rather than a caller knob. All
  deployed callers already passed `None`; passing the keyword now raises `TypeError`.

## 0.9.1 — 2026-07-25

### Added
- **Within-tier mining progress.** `find_rules(verbose=True)` now reports progress WHILE a length
  tier is mining, not only at the end. The Rust miner publishes a `(sources_done, sources_total)`
  counter (`Engine.mining_progress()` on the compiled core) that a daemon monitor polls, printing
  `Length L: X/N sources (P%) | R src/s | ETA T | elapsed E | RSS M` at 20 / 60 / 180 s and then
  every `SIMPLIPY_MINE_PROGRESS_INTERVAL` seconds (default 600). A long tier -- the length-5+
  combinatorial wall where a mine can sit for days -- is no longer an opaque wait: its rate, ETA,
  and memory are visible as it runs. Zero effect when `verbose=False`.

## 0.9.0 — 2026-07-24

A new simplify kernel. `simplify` is now a best-first cancellation SEARCH instead of a fixed
cancellation order; masking (numeric literals -> `<constant>`) is separated from the
equivalence loop so representation is no longer entangled with rewriting; and the constant-fold
fallback is brought under the same finite-a.e. certificate the rules and cancellation already
enforce.

### Added
- **Cancellation search** (`Engine::simplify_search`). `simplify` no longer commits to one
  cancellation order. Cancel is non-confluent -- taking candidate A can destroy candidate B,
  and which choice ends shortest depends on what the ruleset can fold -- so the kernel now
  SEARCHES a small move graph instead of guessing. State = an expression; the moves are a flat
  choice set (apply the rules pass, or cancel any one qualifying candidate); the answer is the
  shortest state visited. Every state is a.e.-equivalent to the input and the input is state
  zero, so a result can never be longer than its own input. Best-first by
  length with a visited-set, pre-filled with the greedy result so a truncated budget can never
  do worse than the plain fixpoint. The node budget (`SIMPLIPY_SEARCH_BUDGET`, default 24;
  0 restores the plain greedy fixpoint and its speed exactly) bounds a rare heavy tail -- the
  median expression expands 2 nodes and 53% have no cancellation candidate at all, but the p99
  is ~186. The default is the measured elbow of the returns curve on the 64k v23.0 prior: below
  24 an extra microsecond buys ~14 output tokens, above it ~2.7, and by 256 only 0.7. On that
  corpus the search shortens ~2860/1000/350 expressions (2-1/3-2/4-3) that no previous version
  could reach, at ~2.3x the simplify wall time; raise the budget for offline corpus
  canonicalisation, lower it for latency.
- **Symmetric `neg`/`inv` cancellation** (BEHAVIOUR CHANGE). The cancellation unit already
  EMITTED the class inverses but never CONSUMED them: a leaf under its own class inverse was
  shielded, so `x * inv(x)` and `x + neg(x)` did not cancel through the inverse. Each inverse is
  now region-continuing in its own class (`neg` additive, `inv` multiplicative), symmetric with
  the emit path, and there is exactly one region shape -- the asymmetry is not preserved behind
  a flag. Consequence on sparse rule sets: a handful of expressions come out ONE token longer
  than in 0.7.x, because the old opaque treatment happened to leave a sign arrangement those
  rule sets can re-fold and the connected treatment does not (6 of 65536 on the 2-1/3-2-class
  `3-2` set; none on `4-3` or richer, and never longer than the input). Cancelling through the
  class inverse is worth far more than it costs: the same corpus gets 1005 expressions SHORTER
  on `3-2` and 350 on `4-3`.
- **Soundness `Mode` (`simplify(expr, mode=Mode.SOUND)`).** A single ordinal soundness axis,
  exposed as `simplipy.Mode`. `Mode.SOUND` (the default) is equivalence-preserving and
  idempotent -- the deployed inference/scoring path, byte-identical to the historical default.
  `Mode.LOSSY` trades soundness for recall: every rule placeholder binds any subtree (the
  `!`-sort finite-a.e. certificate is skipped) AND the constant-fold's finiteness gate is
  relaxed, so a non-finite-a.e. subtree such as `<constant>/0` collapses to `<constant>` too.
  `Mode.LOSSY` is for training-corpus canonicalisation ONLY -- the training data is generated
  FROM the simplified form (target == data) -- and must never run on an inference or scoring
  path. The decided full ordering is `EXACT <= SOUND <= AE <= LOSSY`; only `SOUND` and `LOSSY`
  are implemented.

### Changed
- **BREAKING: masking is separated from `simplify`.** Masking numeric literals to the generic
  `<constant>` placeholder is a REPRESENTATION step for downstream models that cannot consume
  literals, not an equivalence-preserving rewrite. `simplify` no longer masks; it is now the
  equivalence loop only (search + sort to a fixpoint), sound and idempotent by construction. The
  `mask_elementary_literals` parameter of `simplify` is removed; use the new terminal
  `Engine.mask()` (relabel literals + one sort, no re-simplify) on `simplify`'s output when
  placeholders are needed, and do NOT re-`simplify` a masked expression. Entangling masking with
  the search fixpoint was also the sole cause of the former non-idempotence: masking mints a free
  `<constant>` from a structural literal (`x - x -> 0 -> <constant>`), and re-searching could
  fold that constant into a denominator, dropping the reachable `C = 0` the input required.

### Fixed
- **Constant folding respects the finiteness certificate.** The constant-fold fallback collapsed
  any subtree whose operands are all `<constant>` or finite literals to a free `<constant>`, on a
  syntactic test that assumed finite operands compose to a finite result. That is false at a pole
  (`<constant> / 0` is +-inf/nan for every constant), so a structural zero reaching a denominator
  could be folded to a finite free constant -- unsound (it revives a structurally zeroed term).
  This was the one search edge that skipped the finiteness certification the rule matcher and the
  cancellation already enforce. The fold now consults the same value-set analysis and collapses to
  `<constant>` only when the subtree has a positive-measure finite part, keeping
  unbounded-but-finite-a.e. folds (`1/C`, `tan C`, `cosh(C + 5)`) and refusing non-finite-a.e.
  ones (`C / 0`, `C * inv(0)`).

## 0.8.0 — 2026-07-22

Makes the public package produce the BEST rule sets from one command: mining now natively
promotes every rule to the strongest sound sort, and a public verification API can
independently gate + monitor any rule set.

### Added
- **Native sort promotion** (`simplipy.promotion`; `find_rules(promote_sorts=True)`, CLI
  config key `promote_sorts`). After mining and pruning, every rule (mined + proposed) is
  re-certified at the stronger sorts and shipped at the strongest SOUND one: `_` (arbitrary
  subtree), `!` (certified-finite subtree, enforced by a match-time defined-and-finite-a.e.
  certificate), or `?` (variable-leaf, the fallback). Five stages: a pointwise exact bar on
  an atom lattice, a const-bearing witness-map bar, an exact-arbiter overturn of finite-draw
  demotions, a subsumption/derivability refund, and the `_`→`!`→`?` ladder with a
  moving-spike structural refusal. Promotion is fail-safe: an uncertifiable rule stays `?`
  and loses only composite-subtree recall, never soundness. This recovers the simplification
  power that conservative `?`-only rulesets leave unrealized (a mined ruleset promoted this
  way matches a far larger legacy engine's reduction at a fraction of the rules), because the
  limiting factor was sort generality, not the number of rules.
- **Independent verification API** (`simplipy.verify`): `verify_ruleset` gates a rule set by
  judging every rule at its own symbolic trigger points under an arbitrary-precision contract
  evaluator (eight-bucket classification, 100% coverage by construction); `monitor_ruleset`
  runs the deployed engine over an adversarial+sampled corpus and attributes any
  deployed-value violation to the responsible rule under an independent high-precision
  evaluator; `verify_rule` gives a single-rule verdict. Deliberately implemented
  independently of the compiled core so it cross-checks the miner rather than echoing it.
  Both carry poison self-tests.

### Changed
- `mpmath` and `scipy` are now runtime dependencies (offline mining + verification only; the
  inline simplify path remains the compiled core).

## 0.7.1 — 2026-07-21

### Fixed
- **CLI `find-rules` honors the full config**: the `prune` and `relaxed_kruskal` keys in a
  find-rules YAML were silently ignored (`prune` fell back to the `--prune` CLI flag,
  i.e. `False`; `relaxed_kruskal` to its default), so a config declaring
  `prune: covered` produced an unpruned artifact that did not match its own claims.
  Both keys are now forwarded (a config `prune:` key takes precedence over `--prune`),
  and the CLI rejects unknown config keys fail-closed, naming the offending key — a
  mis-spelled or unsupported key is an error, never a silent no-op (`confirm_rules:`
  was such a silent no-op; the honored key is `confirm:`).
- The provenance sidecar now records `relaxed_kruskal` alongside the other mine
  parameters.

## 0.7.0 — 2026-07-21

Real-semantics pow alignment and the removal of the faithful dev_7-3 reproduction line:
the package now ships ONE engine line.

### Added
- **LLM-proposal channel in the miner**: `find_rules(..., proposals=...)` accepts a path
  to a proposals JSON (consolidated `{"proposals": [...]}` artifact or a bare list of
  `{source, target?}` objects) or the equivalent in-memory object. After the mining
  length loop and before the optional prune, every proposal runs the exact
  `certify_rules` chain against the just-mined rule state with the mine's evaluation
  matrices, tolerances and master-seed-derived, content-derived per-proposal seeds;
  certified proposals join the ruleset through the same `deduplicate_rules` path. The
  find-rules YAML forwards a `proposals:` key, and the provenance sidecar records the
  proposals file, its sha256, and per-outcome counts
  (`certified` / `already_covered` / `rejected` / `duplicate`). This makes a
  mined-plus-proposed ruleset reproducible from one config and one command.
- **Special-point certification phase in the miner** (`rust/battery.rs`): rule
  certification now additionally checks every accepted (source, candidate) pair at a
  battery of symbolic special points (0, ±1/2, ±1, ..., ±π/2, π, e) per variable, sweeps a
  battery of special source-constant values, and snaps fitted witness constants to nearby
  integers / half-integers before the domain-preservation gate. This closes three gaps
  where a rule could certify on random evaluation matrices yet be unsound at points
  deployed expressions actually reach: a fitted exponent like `2.9999999999999996` is NaN
  on the negative half-line and hides the positive-measure domain extension its snapped
  witness `3.0` creates (`exp(log(pow3 x)) → pow(x, C)`); a rule can be false exactly at a
  symbolic coincidence (`pow(sin x, inf) → 0` is 1 at x = π/2); and a pattern-bound
  source constant reaches special values in deployment (`pow(cos C, inf) → 0` at C = 0).
  Rows where deployed f64 evaluation diverges from contract semantics are re-judged at
  three fixed precision rungs (50/120/250 decimal digits) with symbolic coordinates
  rendered at each precision, so stable limit completions (`x/x → 1`) keep certifying
  while precision-dependent cancellation seams are rejected fail-closed.

### Changed
- **Stricter mining certification (see the special-point phase above)**: `find_rules`
  rejects rule families that earlier versions could mine. Rulesets mined with
  `simplipy <= 0.6.0` may therefore contain rules that 0.7.0 refuses to re-mine;
  re-mining under 0.7.0 is the recommended migration.
- **Pow alignment (real semantics)**: `pow` at a `-inf` base with a finite non-integer
  exponent now evaluates to NaN (real semantics) across the constant folder, the operator
  realizations, and the interval engine. Magnitude-step cells at infinite exponents are
  unchanged.
- **`engine_id` now reports the package version** (e.g. `simplipy-0.7.0`) instead of the
  frozen reference id `dev_7-3`, so provenance records identify the exact engine build.

### Removed
- **BREAKING: the faithful dev_7-3 reproduction line is removed.** The `fold` parameter of
  `_core.Engine.simplify` / `apply_rules` / `prune_explicit` is gone (the numeric
  constant-folding behavior is now the only one), the legacy quirk-preserving conversion
  variants are gone (the corrected conversions are now exposed under the plain
  `prefix_to_infix` / `infix_to_prefix` / `convert_expression` / `parse` names, which is
  what the Python API always routed), and the frozen byte-identical-to-0.2.15 parity
  fixtures and reference constants (`FAITHFUL_ENGINE_ID`, `REFERENCE_SIMPLIPY_VERSION`,
  `REFERENCE_SIMPLIPY_COMMIT`) are gone. The parity regression test is re-baselined
  against the 0.7.0 aligned engine line. To reproduce v23.0/dev_7-3-era behavior
  byte-for-byte, install `simplipy<=0.6.0`.

## 0.6.0

A performance overhaul of the simplify hot path (byte-identical outputs), sorted rule
placeholders with match-time certificates, ruleset pruning and observability tools, and
a round of rule-mining improvements and constant-folding coherence fixes. The shipped
`dev_7-3` asset is unchanged.

### Added
- **Sorted rule placeholders** (`_` / `?` / `!`): every placeholder in a rule carries a
  *sort* — the binding claim its sigil encodes, enforced by the matcher at apply time.
  `_i` binds any subtree; `?i` binds only a bare variable leaf; `!i` binds a variable
  leaf freely, but a composite subtree only when a match-time certificate proves it
  defined and finite almost everywhere (an adaptive interval analysis over the reals;
  fail-closed — what cannot be certified is not bound). Sorts let a rule that is
  value-sound only for certified operands make the wider claim without giving up
  soundness. Newly mined rulesets emit sorted patterns; existing assets (explicit and
  `_`-wildcard rules) behave exactly as before. See the rule-authoring guide
  (`docs/rules.md`) for details.
- **Covered-rule pruning** (`SimpliPyEngine.prune_covered_rules`, CLI
  `simplipy prune-covered-rules`): removes any rule — wildcard-pattern rules included —
  whose effect the remaining rules already achieve compositionally. A rule is covered
  only if every instantiation variant of its source (distinct variable leaves, literal
  `<constant>`, composite probes for wide placeholder slots) still simplifies to at
  most the length of the corresponding target without it. Deterministic batch
  remove-and-repair, longest sources first; greedy (the result is valid, not
  necessarily minimal). Complementary to `prune_redundant_rules`, which removes only
  explicit rules shadowed by pattern rules under an equality criterion. `find_rules`
  gains `prune='covered'` to run this prune after discovery instead of the
  redundant-rule prune.
- **Simplify observability**: `SimpliPyEngine.simplify_counters()` /
  `reset_simplify_counters()` expose the process-global simplify hot-path counters
  (calls, iterations, pattern attempts/fires, certificate calls/hits, and coarse
  per-stage nanosecond accounting) for profiling a batch.
- **Formal mining-algorithm documentation**: a formally typeset specification of
  `find_rules` (the discovery loop and the equivalence certification), with PDF/LaTeX
  downloads, an algorithm-line-to-configuration reading guide, and the determinism
  contract (`docs/algorithm.md`).
- **Relaxed Kruskal search** (`mine_one_length(..., relaxed_kruskal=...)`): with
  `True`, a source the current rules already shorten is still searched, with the
  candidate bound tightened to its simplified length (only targets strictly shorter
  than what `simplify` reaches are accepted). Relaxed rules are one-step shortcuts that
  are capability-redundant wherever the reduct's own rule exists; for sampled
  universes, re-mining the simplify-fixpoints of skipped sources generalizes better (a
  fixpoint rule covers every source that reduces to it). The Rust-level parameter
  defaults to the strict skip, but `find_rules(relaxed_kruskal=True)` is the
  Python-side production default (a capability comparison across four mined corpora,
  ~780k expressions total, found the relaxed mine strictly shorter on 735 of them and
  never longer, at ~+6% mine cost).
- **High-precision rescue for const-free equivalence** (`astro-float`, pure Rust, 1024-bit):
  f64 remains the fast pre-filter, but a const-free candidate that fails the tolerance on
  only a small fraction of its binding rows (<= 25%, <= 256 rows) is re-evaluated on
  exactly those rows at 1024 bits and re-judged under the same generic-equivalence
  semantics. True identities whose f64 round-trip loses precision now certify --
  `atanh(tanh(x)) -> x` (f64 error reaches 2e-1 for |x| in [10, 19.6]) and, via domain
  extension, `tanh(atanh(x)) -> x`, both previously unminable at the strict rtol=1e-9.
  Every f64 accept is unchanged; plateau falsities keep failing their near-corner rows and
  never reach the escalation. Applies to the mine scan and (through the same code path)
  stage-2 confirmation and `certify_rules`.
- **Candidate library minimization** (`find_rules(candidate_fold_filter=True)`, default on):
  variable-free candidates of length >= 2 are excluded from the mining candidate library.
  This is provably behavior-preserving (any source they could match is already matched by
  the length-1 `<constant>` candidate, which is scanned first) and removes the bulk of the
  constant-fitting work: measured 55-88x faster per source on the dev configuration at
  identical mining decisions. The library reports `n_candidates`/`n_filtered`.
- **Provenance sidecar**: `find_rules(output_file=...)` now writes
  `<output>.provenance.json` recording all parameters, derived seeds, the evaluation-matrix
  specification, and per-length universe coverage, so a mined ruleset is reproducible from
  its artifact alone.
- **Per-run sampler validation**: sampled sources are checked for universe membership
  (length, vocabulary, well-formedness) on every run.

### Changed
- **NaN literals propagate through constant folding**: a subtree with a
  `float("nan")` operand now folds directly to `float("nan")` (every operator maps a
  NaN operand to NaN; the IEEE `pow` edge cases whose value does not depend on the NaN
  operand are excluded), and `inf`/`nan` literals are never absorbed into a fittable
  `<constant>` — a `<constant>` claims a finite value exists to fit, and NaN is not
  one. This is operator-table knowledge, not sampling: it also keeps the miner from
  certifying rules that rewrite an everywhere-NaN expression, for which no finite
  evidence can exist.
- **Constant-fit restart seeds are order-independent** (a pure function of the source seed,
  candidate, and challenge instance). Mines remain byte-reproducible for a fixed seed; the
  output of a given seed may differ from 0.5.0 on marginal nonlinear fits.

### Performance
- **Certificate economics**: `!`-sort certificates are evaluated only after a
  *completed* syntactic match (instead of during every candidate attempt), memoized
  per call and in a generational per-engine cache that never stops memoizing. This
  replaces the old hard-capped certificate cache, which silently stopped memoizing
  once full and degraded at scale (measured on a 65,536-expression training-prior
  benchmark: hit rate falling from 99.3% to 36.6% and per-expression engine cost from
  ~600 µs to ~7,400 µs once the cap was hit). Certificate verdicts are unchanged.
- **Fixpoint memoization**: the simplify fixpoint memoizes whole cancel/rules passes
  and rule-normal subtrees per call, so the convergence-confirming iteration and
  unchanged subexpressions cost hash lookups instead of re-scans.
- **Interned token representation**: the hot path runs on interned token ids with a
  per-engine property table (and a per-call overlay for unknown tokens), reducing
  allocations per simplify call by ~20×.
- Net effect on a 65,536-expression training-prior benchmark: large
  certificate-bearing rulesets simplify ~59× faster than 0.5.0, with byte-identical
  outputs; certificate-free rulesets run ~2.3× faster.
- **Mining**: the source expression is evaluated once per source (per challenge
  instance) and shared across the whole candidate scan, instead of per
  (candidate, challenge, sign-combo); a const-free source collapses to a single
  challenge instance (identical targets add no evidence and only multiplied fit
  flakiness).

### Removed
- The `scipy` runtime dependency: dead since the 0.5.0 native constant-fit cutover
  (no `scipy` import remained anywhere in the package).
- 14 dead `SimpliPyEngine` attributes left over from the removed pure-Python simplify
  engine, each verified to have no remaining consumers: `operator_inverses`,
  `inverse_base`, `inverse_unary`, `inverse_binary`, `unary_mult_div_operators`,
  `commutative_operators`, `realization_to_operator`, `operator_precedence_compat`,
  `max_power`, `max_fractional_power`, `connection_classes`, `operator_to_class`,
  `connection_classes_inverse`, `connection_classes_hyper`,
  `binary_connectable_operators`.
- `simplipy.utils.leaf_value`: unused Python duplicate of the engine's single
  leaf-value table.
- Two undocumented, dev-only methods on the compiled core (`eval_bench_resident`,
  `classify_linearity`) and, on the Rust side, the unused `match_pattern` wrapper and
  the unused `criterion` dev-dependency.

### Fixed
- **Term cancellation respects non-finite algebra**: with an empty ruleset (or rule
  application disabled), `0/0`, `nan/nan`, and `inf/inf` previously canceled to `1`,
  and `inf - inf` / `nan - nan` to `0` — all wrong answers (the true value is NaN).
  Cancellation assumes the group axioms, so a leaf now registers as cancellable only
  in the connection classes where they hold: nonzero finite literals in both, `0` in
  the additive class only, `nan`/`±inf` in neither. Variables and nonzero-finite
  literals cancel exactly as before.
- **One leaf-value table everywhere**: constant folding (`_try_fold_constants` /
  `evaluate_constant_subtree`), the miner's all-constant short-circuit, and the offline
  tape evaluator now share a single leaf resolution (Rust `numeric::leaf_value`):
  numeric literals, `np.pi`, `np.e`, parenthesized literals such
  as `(-1)`, and the `float("...")` tokens. Previously the folder and short-circuit used
  the numeric-string predicate only, so e.g. `cosh(np.pi)` never folded exactly and
  `(-1)/cosh(np.pi)` simplified no further than `(-1)/<constant>` even though every leaf
  has a known value. The `<constant>` absorption collapse now admits any FINITE-valued
  leaf (still never inf/nan literals: non-finite algebra remains the explicit rules'
  domain).
- **Generic-equivalence vacuity in the constant-bearing fit arm**: an equivalence-check
  instance whose source evaluates nowhere finite (e.g. the sign-combo `C = 0`
  instantiation of a constant-denominator source, `y = k/0`) binds no rows and is now
  treated as domain-extendable — matching the declared semantics and the const-free arm —
  instead of falling into the underdetermined-fit bail that returned `False` and vetoed
  the candidate. This single instance previously killed every constant-denominator rule:
  `x0*<constant>/<constant> -> x0*<constant>` was unminable, and (with the short-circuit
  gap above) so was the whole `k/<constant> -> <constant>` absorption family. Evidence
  accounting is unchanged: vacuous instances contribute zero rows toward
  `min_informative`.
- **Affine constant-fit recall with fast-growing bases**: fits of the form `C0*f(x)+C1`
  and `f(x)+C1` for `f` in `exp`/`cosh`/`sinh`/`pow3`-`pow5` were spuriously rejected at
  exactly-true constants. The closed-form solver now uses row weights matched to the
  acceptance tolerance, overflow-safe column equilibration, iterative refinement, an exact
  feasibility decision for the bare-`<constant>` candidate, and integer snapping for
  exact-cancellation cases. The acceptance gate itself is unchanged (recall-only fix;
  non-equivalences are still rejected).
- Fold-filter variable detection is correct for configurations with more than 32 variables.
- `min_informative` defaults are floored at 1 for very small evaluation matrices.

### Docs
- 0.5.0 entry corrected: the mine evaluation matrix's log-uniform tier is 1e-4..1e3, not
  1e-6..1e6.

## 0.5.0 — 2026-07-11 — sound rule-mine checker + Python mining mirror removed

The pre-0.5.0 checker shipped ~8.3% defective
rules in the dev_7-3 sample (52/840), including ~5,125 vacuous all-NaN wildcard rules such as
`asin(cosh(_0)) -> nan` (false at 0, where the source is pi/2).

### Fixed (checker soundness)
- **Vacuous equal_nan acceptance closed.** Certification now requires at least `min_informative`
  (default `n_rows / 8`) SOURCE-FINITE evidence rows, accumulated across challenge instances; a
  precomputed candidate-finite count serves as a fast necessary-condition gate for const-free
  candidates.
- **Tolerances tightened** `rtol` 1e-5 -> 1e-9, `atol` 1e-8 -> 1e-12 (0 borderline rules in an
  840-rule review sample; tanh/exp saturation towers are no longer "equal" to constants).
- **Heavy-tailed, seeded evaluation matrix** (`_mining_sample_x`): 40% N(0,5) + 25% U(-50,50) +
  25% signed log-uniform magnitudes 1e-4..1e3 + 10% exact corner points ({+-0.0, +-0.1, +-0.5,
  +-1, +-2, +-e, +-pi, +-10}). (This entry mis-stated the tier as 1e-6..1e6 until 2026-07-11;
  the code always shipped 1e-4..1e3 -- the wider tail was tried and reverted for fit
  conditioning, see the `_mining_sample_x` docstring.)
- **GENERIC-EQUIVALENCE semantics with domain extension** (`allclose_extends`, the single accept
  gate everywhere: const-free compare, affine/log-linear/LM fit accepts). Rows where the SOURCE is
  finite bind: the replacement must be finite and equal within tolerance -- the exact corner points
  refute wrong-VALUE identities (`asin(cosh(_0)) -> nan` is false AT 0, where the source is pi/2).
  Rows where the source is NaN/inf are extendable: the replacement may complete them with the
  generic/limit value (`div(_0, _0) -> 1` certifies; `log(exp(_0)) -> _0` certifies under f64
  overflow). The evidence gate (`min_informative`) counts SOURCE-FINITE rows accumulated across
  challenge instances, so an (almost-)nowhere-defined source can never be rewritten from its
  corner rows alone.
- **Stage-2 confirmation** (`confirm=True`): every mined pair is re-verified on an independent,
  twice-as-wide X with fresh constant draws and seeds before it can enter the Kruskal cascade.
- **IEEE inv/div semantics** aligned across the Rust kernels, the scalar Python operators and the
  constant folder (`1/±0 = ±inf`, `x/0` sign-correct): the mine now certifies exactly the semantics
  the deployed numpy engine executes.
- **Complete Phase-1 enumeration.** The old pass-based closure stopped once the maximum length was
  REACHED, not SATURATED: the dev_7-3 mine saw 444,865 of the 9.0e9 length<=7 expressions and missed
  e.g. ALL 179,685 triple-unary chains at length 4. Phase 1 is now a bottom-up DP per length,
  complete by construction and cross-checked against an exact count DP every run.
- **Universe policy for infeasible lengths** (`source_sample_per_length`): lengths whose complete
  universe explodes (dev set: 2.4e8 at length 6, 8.8e9 at length 7) are drawn uniformly from the
  complete universe via a count-weighted top-down sampler; coverage is always logged, and sampling
  inside the candidate range warns (the library then no longer certifies minimality).
- **Full determinism**: seeded master RNG (`seed=42`), sorted set iteration for sources and the
  candidate library, derived per-length seed blocks. Same seed => identical rule set.
- **Fit-path completeness**: `constants_fit_challenges` / `constants_fit_retries` defaults raised
  5 -> 16 end to end (the old find_rules passed 5 against an FFI default of 16; measured fit-path
  completeness at 5/5 was ~24%).

- **Affine-fit conditioning fix**: the whole `C0*f(x)+C1` family silently REJECTED before. Two
  causes, both introduced by earlier hardening of the same path: a GLOBAL trace-scaled Tikhonov ridge biased the
  intercept, and the normal-equations solve squared the condition number on the wide-magnitude X
  so an exact affine relationship's intercept came out ~5e-9 off (past rtol). Now the affine path
  solves by Householder-QR least-squares (no `A^T A`, no ridge -- works at cond(A), not its square),
  and the mine X's log-uniform tier is capped at 1e3 (from 1e6): 1e3 still exercises every
  saturation and f64 overflow the tier is FOR, but 1e6 wrecked the fit conditioning. Confirmed by
  probe: 0/30 -> 64/72 affine cases certify, 0 false-accepts; the residual misses have constants
  spanning >1e6.
- **Log-linear recall fix**: for `pow(<constant>, g)` candidates, only a closed-form log-space
  ACCEPT short-circuits; an imprecise `Some(false)` solve (its ~1e-10 base error amplified past
  rtol by large exponents on the heavy-tailed X) now SEEDS the LM restart instead of rejecting,
  restoring the declared const-bearing policy ("accept iff constants EXIST that fit"). Surfaced by
  extensive adversarial re-verification of the generic-equivalence change (which found NO soundness
  defects).

### Changed
- **Perf**: the source expression is now evaluated once per source (per challenge instance) and
  shared across the whole candidate scan, instead of per (candidate, challenge, sign-combo); a
  const-free source collapses to a single challenge instance (identical targets add no evidence
  and only multiplied fit flakiness).

### Removed (BREAKING)
- **The pure-Python mining mirror is gone**: `find_rule_worker`, `exist_constants_that_fit` and the
  fork/SharedMemory pool (with the `n_workers` parameter). It duplicated `rust/worker.rs`/`fit.rs`
  and repeatedly desynced from them (e.g. diverging IEEE inv/div semantics). `find_rules` now requires the
  compiled core and raises `RuntimeError` on a bare engine; parallelism is rayon's
  (`RAYON_NUM_THREADS`).

### Compatibility
- Shipped rule assets (`dev_7-3` etc.) are unchanged by this release; they remain the frozen
  reference that downstream training pipelines pin for reproducibility. A re-mine with the
  hardened checker is a separate, explicitly-versioned artifact.


## 0.4.1 — 2026-07-02 — find_rules works with the Rust core + safe concurrent asset installs

### Fixed
- **`find_rules` now mines natively on the Rust core.** With `_core` attached (any engine from
  `load()` / `from_config()`), the fork-based Python pool mined **0 rules**: the workers mutate
  Python-side rule state while `simplify` runs on the immutable core — the same class of bug as the
  0.4.0 `prune_redundant_rules` fix. Phase 2 now delegates to the native mine when the core is
  attached (candidate library + per-length `mine_one_length` + `set_rules` between lengths,
  rayon-parallel; cap with `RAYON_NUM_THREADS`; `n_workers` only applies to the pure-Python
  fallback). Mined rules are now always canonicalized to the wildcard (`_j`) form.
- **`find_rules(X=<ndarray>)`** no longer raises `NameError` (the documented array form was never
  assigned to the internal data variable).
- **Concurrent cold-cache asset installs are now safe.** Installs serialize per asset via a
  `FileLock`, and an asset only counts as installed when **every** file in its manifest is present,
  so a partially-downloaded asset is correctly treated as not installed. New dependency: `filelock`.

### Changed
- Internal offline-mining scaffolding (mine grid and validation drivers) moved out of the
  released repo.

## 0.4.0 — 2026-07-01 — native offline rule-miner + simplify fixes + CLI fix

### Added
- **Native (Rust-core) offline rule mining.** An all-cores native mine driver (with a grid timing
  harness) supersedes the pure-Python `find_rules` for offline discovery against the Rust `_core`.
- **Closed-form `pow(C, x)` / `pow(x, C)` log-linearization** in the offline pipeline.

### Fixed
- **`sort_operands` is now idempotent** — canonical operand order is a fixpoint (rotation iterated to
  convergence; mask-before-sort), so simplifying a simplified expression no longer changes it. This can
  change the canonical operand ordering of some expressions vs 0.3.x (verified not to affect the
  downstream symbolic-data / flash-ansr / srbf test goldens).
- **`prune_redundant_rules` corrected against the Rust core** (the pure-Python prune produced 0 rules
  with `_core`); offline grid mining caps sources at length ≤ i and chunks the per-config budget.
- **`simplipy install <name>` / `simplipy remove <name>` now work.** They previously passed the asset
  *type* as the asset name (and the name as `force`/`quiet`), so every invocation raised
  `ValueError: Unknown asset: 'engine'`. They now take just an asset name with clean error handling
  (message + exit 1, no traceback); the vestigial `--type` flag is dropped from `install`/`remove`
  (`list` keeps it).

### Docs
- Documented the normalization helpers in the API reference; added a CLI reference for
  `list` / `prune-rules` / `resolve-rules` / `install` / `remove`; qualified `SimpliPyEngine`
  methods in the component overview; named `symbolic-data` as the direct downstream in the family DAG.

## 0.3.1 — 2026-06-28 — expression-token normalization helpers

Adds pure-Python `normalize_skeleton`, `normalize_expression`, and `normalize_variable_token`
to the package root (also available as `simplipy.normalization`). These canonicalize a prefix
token sequence -- variable tokens (`v1`/`x1`, case-insensitive) to a stable `x{n}`, and numeric
literals to a `<constant>` placeholder (skeleton form) or kept as-is (expression form) -- so two
expressions that are "the same" up to variable renaming / constant values compare equal.

Relocated from flash-ansr (behavior-identical) so the canonicalizer lives at the shared
expression-engine leaf that downstream packages (symbolic-data, flash-ansr, srbf) all depend on,
keeping holdout-matching and symbolic-recovery scoring consistent by construction. No change to
the Rust inline backend (`simplipy._core`) or any existing API; purely additive.

## 0.3.0 — 2026-06-21 — Rust inline backend + the "numeric" engine line (MAJOR behavior change)

This release rewrites SimpliPy's **inline phase** (`simplify`, the prefix/infix conversions, and
validation) as a compiled Rust extension (`simplipy._core`) and makes the improved **numeric** engine
line the default. The **offline phase** (rule mining + `curve_fit`) stays pure Python.

### ⚠ Breaking — behavior changes vs 0.2.x

The default `simplify`/conversion behavior is now the corrected ("numeric") line. The engine-id is
**unchanged** (`dev_7-3` still identifies the rule-mining parameters: max source length 7, max target
length 3; the `rules.json` asset is byte-identical). What changed is the engine **code**:

1. **Numeric constant folding** is now applied during simplification. All-numeric subtrees evaluate to
   their `f64` result (e.g. `1/0 → float("inf")`, `sqrt(-1) → float("nan")`); the folding fires as a
   fallback after rule matching, so a rule that applies to an all-`<constant>` subtree is tried before
   the subtree is collapsed.
2. **Six conversion bug fixes**, most notably:
   a fractional-power child (`pow1_M`) is no longer silently dropped by `convert_expression`; left/right
   associativity in `infix_to_prefix`/`prefix_to_infix` is corrected (and round-trip-preserving);
   `x**0`, `--5`, `^`-vs-`**` unary-minus, and raw `powN` are handled correctly.
3. **`float("inf")` / `float("-inf")` / `float("nan")` are atomic tokens** in the conversion tokenizer,
   so a folded constant round-trips through prefix↔infix instead of fragmenting.

> Reproducibility: to reproduce results generated with the pre-0.3 engine (e.g. models trained against
> `dev_7-3` under SimpliPy 0.2.x), pin `simplipy<0.3`. The 0.2.x behavior is the frozen anchor; 0.3 is a
> deliberate, documented quality improvement, not a silent drift.

### Robustness

- **ndarray return no longer truncates folded tokens.** `simplify(np.array([...]))` re-infers the result
  string width (keeping the input dtype *kind*); previously a fold that emitted a token wider than the
  input tokens (e.g. `1/0 -> float("inf")` from a `<U1` input) was silently truncated. Affected both the
  Rust-routed and pure-Python paths.
- **Malformed / pathological input raises a clean `ValueError`** instead of an uncatchable abort. A
  malformed prefix (an operator with too few operands) and an excessively long expression
  (`> 4096` tokens, which would recurse deep enough to overflow the stack) are now rejected up front.
  Empty input is still valid (`simplify([]) == []`). Valid inputs are unaffected (verified: 0 diffs vs
  the pre-fix engine across 112,040 corpus comparisons).
- **`simplipy._core` load failures now warn** (`RuntimeWarning`) rather than silently degrading to the
  slower pure-Python path.
- The pow-chain exponent product is computed in `i128` (Python's `prod` is arbitrary-precision), pushing
  the divergence boundary past any reachable exponent.

### Notes

- **Offline mining now uses the Rust inline methods internally** (e.g. `find_rules` calls the improved
  `simplify`/conversions). For Phase-A this is intended; it does not re-mine or change shipped assets,
  but mining a *new* asset under 0.3 would inherit the numeric-line behavior.
- **Folding precision:** the f64 evaluator folds through the platform's system `libm` (not NumPy). On a
  given platform it is byte-identical between the Rust and the pure-Python fallback, but folded constants
  can differ **sub-ULP** from the previous NumPy-based folding. No bit-identity vs 0.2.x folded constants
  is promised (or expected).
- **Pure-Python fallback:** if the compiled `simplipy._core` extension is unavailable, the engine
  transparently falls back to pure-Python implementations (correct, slower). `import simplipy` never
  depends on the extension or on `ctypes`/`libm` being present.

### Build / packaging

- Build backend switched from setuptools to **maturin** (mixed layout: Python at `src/simplipy/`,
  Rust crate under `rust/`, compiled module `simplipy._core`). One abi3 wheel per platform/arch for
  CPython ≥ 3.11.
- **New runtime dependency:** `platformdirs` (resolves the per-user asset cache directory).
