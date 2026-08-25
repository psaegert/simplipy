# Simplifying expressions

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("acj-4", install=True)
```

## The pipeline

```text
function simplify(expr, max_passes=48, mode=f64, effort=DEFAULT_EFFORT):
    state = canon(parse(expr))            # infix→prefix or validate, then the CANONICAL
                                          # form: flat AC bags, exact rationals, like
                                          # terms collected — this is state ZERO

    loop:                                 # a DETERMINISTIC chain, no search
        next = canon(rewrite_pass(state, mode))   # one rules pass, re-canonicalized
        if next == state: break           # fixpoint
        state = next
    return state                          # μ(state) ≤ μ(state zero): every changed pass
                                          # output strictly descends the well-founded
                                          # reduction ordering, so termination is a
                                          # THEOREM; max_passes and the step cap are
                                          # defence-in-depth, not the mechanism
```

Every step preserves the function almost everywhere: like-term collection inside the
canonical constructors, rule application, and the exact fold. The result is therefore
sound, never costlier than the input under the engine's description-length measure μ
as `complexity()` prices it — the instrument parses through the same route the chain
descends from, mode for mode — and idempotent at any fixpoint run. Two *different
spellings* of the same value may still settle at different fixpoints; each obeys its
own bound.

The chain itself does not search: **cancellation IS canonicalization** — like-term
collection in flat bags, computed by one deterministic function, so inside a pass there
is nothing to branch over. Every changed pass output strictly descends the reduction
ordering (μ, then the canonical order), and that ordering is well-founded, so the chain
never revisits a state and reaches its fixpoint in finitely many passes as a theorem;
the `max_passes` parameter (default 48) and the internal step cap are defence-in-depth
against ordering-invariant bugs, not part of the argument. `max_passes` bounds the
number of times that `loop` body runs — **passes, not nodes**: on the 400-skeleton
reference corpus the chain converges in 2–4 passes, so the default of 48 is never
reached and the knob is not a performance dial. What strict descent alone cannot do —
pass through a state that prices *higher* on the way to a cheaper one — is the job of
the [search budget](#the-search-budget) below.
The full ledger of what is theorem, what is enforced, and what is empirical is
[formal.md](../formal.md).

**Masking is part of the same machinery.** Relabelling numeric literals to the generic
`<constant>` placeholder is a *representation* step for a downstream model that cannot
consume literals, and which literals to abstract is the caller's policy, so it is its own
call, `mask()`, never a side effect of `simplify` (see the [Masking guide](masking.md)).
It is not outside the engine, though: `<constant>` is a first-class token inside
`simplify` — rules bind it, and the exact fold collapses constant subtrees into it under
the positive-measure licence — and `mask()`'s collect stage is itself a `simplify` call
over the substituted tokens, which is what enforces one `<constant>` per degree of
freedom (`2*x0/3` is one free value, not two).

The same call as one procedure, with the memo state each stage touches noted where
it applies:

1. **Intern** the input tokens to token ids (the token store: the engine table plus a
   per-call overlay).
2. **Canonicalize** into state zero: flat AC bags, exact rationals, like terms
   collected. Cancellation lives here.
3. **Loop** — one rewrite pass over the state, then re-canonicalize; stop at the first
   pass that changes nothing (the fixpoint). Every changed pass output is strictly
   below its predecessor in the reduction ordering. Inside the pass, per subtree,
   top-down:
    1. exact rule lookup; on a miss, the pattern scan, first match wins;
    2. a match that binds wide slots must pass the bind-time `!`/`$` certificates
       (skipped in corpus mode); verdicts are memoized per call and in a
       generational per-engine cache;
    3. no match: the exact fold with the `<constant>` collapse (positive-measure
       licence; relaxed in corpus mode);
    4. still nothing: recurse into the operands and re-check the rebuilt node.
4. **Serialize** back out: μ ≤ input, sound, idempotent. Callers that need
   placeholders then run `mask()` — the separate call above.

The compiled core implements one engine line: numeric constant folding (including
non-finite results such as `1/0 -> float("inf")`) and real-semantics power evaluation.

## The search budget

Strict descent cannot cross a μ-hill. `x2*(x2 + (x1+1)/x2)` reaches the strictly
cheaper `1 + x1 + x2**2` only by distributing first — which prices *higher* at that
node — and recollecting after, so the chain above, which only ever takes descending
steps, never finds that valley. The search budget exists for exactly this class.

With a budget, the chain first runs unchanged to its fixpoint. A bounded exploration
phase then proposes expansion moves the descent refuses — distributing a product over
its sums, expanding an integer power of a sum — runs every candidate through the same
certified constructors and the same descent loop, and replaces the incumbent only when
the candidate's endpoint lands strictly below it in the engine's one reduction
ordering. Acceptance is that ordering test and nothing else; no new measure or
tolerance enters.

The budget is the `effort` parameter: `simplify(expr, effort=64)` explores with 64
candidate descents, and `effort=0` never enters the phase — byte-identical to the
chain alone. The default is `simplipy.DEFAULT_EFFORT` = 4, set from the acceptance
benchmark's explore-budget arms (65,536 rows): budget 4 strictly improves 3.18% of
rows with zero regressions and captures every win a 16x larger budget finds, at +18%
median per-row cost. Pass `effort=0` on throughput-critical paths.
Every guarantee above survives any budget: candidates are built under the same
certificates (soundness), the incumbent is only ever replaced by something strictly
below it (the result is never worse than the fixpoint, hence never costlier than the
input), the frontier only grows on strict descent of a well-founded ordering
(termination, independent of the budget), and the walk order is deterministic
(idempotence and reproducibility).

## Soundness modes

`simplify(expr, mode=...)` selects a point on an **axis**, not a rung on a ladder.
`simplipy.Mode` has three members: `f64` (the default), `real` and `corpus`.

The axis exists because soundness is two *incomparable* notions, not one ordering. A rewrite
can be true in mathematics and not reproduced by floating point, or reproduced exactly by
floating point and mathematically false:

| rewrite | true over ℝ? | what f64 computes |
|---|---|---|
| `atanh(tanh t) → t` | yes, for every real `t` | `inf` once `tanh t` rounds to exactly `1.0` — from `t ≈ 19.06` on the release host; the exact threshold is libm-dependent |
| `asin(1e-8) → 1e-8` | no — wrong by the cubic term, `1.667e-25` | bit-identical |

Neither rule is "more sound" than the other, so there is no rung to put them on, and `<`
between modes raises `TypeError`.

- **`Mode.f64`** (the default) is sound as the deployed f64 evaluator computes. Use it
  whenever the output will be evaluated in floating point.

- **`Mode.real`** is sound as mathematics defines, independent of any float format. Use it
  when a rewrite must hold symbolically.

- **`Mode.corpus`** is the permissive superset, for training-corpus canonicalisation ("beautification").
  Every rule placeholder binds any subtree (the `!`-certificate is skipped), cancellation drops
  its group-axiom gate, and the constant-fold drops its finiteness gate. It is *not*
  equivalence-preserving. Do not use it on an inference or scoring path: the training data is
  generated *from* the simplified form, so the target equals the data and there is no external
  function to violate.

Each mode names one **distinct, complete** rule set — `rules_f64.json` / `rules_real.json` /
`rules_corpus.json` (artifacts published before the rename, acj-4 among them, call the f64
file `rules.json`) — so selecting a mode selects a file, and what is loaded is what is served.

```python
from simplipy import Mode

# log(C) is undefined for C <= 0. The default is strict about it; only the permissive
# mode collapses it.
engine.simplify('exp(log(<constant>))', mode=Mode.f64)     # -> 'exp(log(<constant>))'
engine.simplify('exp(log(<constant>))', mode=Mode.corpus)  # -> '<constant>'

# A finite-a.e. subtree (pole at a single measure-zero constant) folds in every mode:
engine.simplify('1/<constant>')                            # -> '<constant>'
```

`Mode.real` needs a `rules_real.json`, and **fails closed** on an artifact without one
rather than quietly serving it the f64 set. Against the shipped triple it shows the two
soundnesses disagreeing on a single expression:

```python
engine.simplify('atanh(tanh(30))', mode=Mode.f64)   # -> 'inf'   what f64 computes
engine.simplify('atanh(tanh(30))', mode=Mode.real)  # -> '30'    what is true
```


### What `f64` mode does and does not promise

It promises that **every rule it applies** has been checked against the deployed evaluator
itself. Both sides of the rewrite are evaluated in floating point across the verifier's
battery and grid, and a rule enters the `f64` set only where they agree to within 8 ULP. That bound is derived rather than chosen: a
rewrite has two sides, each a composition of up to four implementation-defined library
calls whose errors are independent and can oppose.

It does **not** promise that a *sequence* of rewrites carries the same bound. Each rule
carries it individually; simplification composes rules, and floating-point error composes
with them.

It does **not** promise to preserve your evaluation order. The canonical form flattens sums
and products into bags and re-emits them, and IEEE-754 addition commutes but does not
associate:

```python
engine.simplify(['+', '+', 'x0', 'x1', 'x2'])   # -> ['+', 'x0', '+', 'x1', 'x2']
# at x0=1e16, x1=-1e16, x2=1 the input evaluates to 1.0 and the output to 0.0
```

The association is deterministic — the same input always gives the same output — and for
expressions that are well-conditioned in f64 the value is preserved. For ill-conditioned ones
it may not be. This is the standard position for any AC-normalising simplifier, and it is
stated here rather than left to be discovered.

**Why two modes, and how flash-ansr uses them.** The downstream trainer
([flash-ansr](https://github.com/psaegert/flash-ansr), a transformer for symbolic regression)
uses each mode on a different side of its pipeline:

- **Training-data generation uses `Mode.corpus`.** A skeleton is corpus-simplified and the numeric
  data is then generated *from that simplified form* — so the target the model learns and the data
  it is trained on are the *same* expression (`target == data`). There is no external ground-truth
  function for `corpus` to violate, so the aggressive reductions are safe here and they give the model
  the shortest, most canonical target. (An `exp(log(<constant>))` that survives cancellation, for
  instance, becomes a plain `<constant>`, which is what the generated data reflects.)

- **Inference and recovery scoring use `Mode.f64`.** At test time the data comes from an unknown
  true function; the predicted skeleton must be simplified *without* changing what it computes, or
  the fit and the score would drift, and the data is evaluated in floating point -- which is
  exactly the soundness `f64` preserves.

That split is the whole reason the permissive mode exists: `corpus` maximizes canonicalization where the
simplified form *defines* the data, and `f64` preserves what the evaluator computes where
it must not change.

## Key components

- **Parsing & normalization** – `SimpliPyEngine.read_infix` and
	`SimpliPyEngine.convert_expression` convert infix input, harmonize power
	operators, and propagate unary negation without losing prefix fidelity.
- **Canonicalization (cancellation lives here)** – the canonical constructors keep
	every associative-commutative operator as a flat bag with exact rational
	arithmetic and collect like terms as they build, so opposite-parity subtrees and
	redundant factors cancel *by construction*, before and between rule passes, and
	identical expressions share identical layouts under the stable canonical order.
- **Rule execution** – `SimpliPyEngine.compile_rules` syncs machine-discovered or
	human-authored simplifications into the compiled core, which performs fast
	top-down, first-match-wins rewriting in each pass.
- **Rule discovery workflow** – `SimpliPyEngine.find_rules` explores expression
	space natively on the compiled core (parallelized across all cores via rayon),
	confirms identities with numeric sampling, and writes back deduplicated
	rulesets that future engines can load instantly.

## Normalization helpers

Besides the engine, SimpliPy exports the two canonical expression FORMS at the
package root: `to_skeleton`, `to_expression`, and the single-token helper
`normalize_variable_token` (also available as `simplipy.normalization`). They
canonicalize an expression so that two expressions that are "the same" up to
variable renaming / constant values compare equal, giving downstream consumers
(holdout matching, symbolic-recovery scoring) identical behavior by construction.
Each takes all three forms (infix `str`, explicit prefix, tagged) and returns the
one it was given; the canonicalization runs through the engine's internal state,
so the answer does not depend on the dialect you passed.

```python
import simplipy as sp

# Skeleton form: variables -> x{n}, EVERY numeric literal -> <constant>
sp.to_skeleton(['+', 'v1', '2.5'], engine)
# -> ['+', 'x1', '<constant>']

# Expression form: variables canonicalized, numeric values kept
sp.to_expression(['+', 'V1', '3'], engine)
# -> ['+', 'x1', '3']

# Classify / canonicalize a single token -> (normalized_token, is_variable)
sp.normalize_variable_token('X3')
# -> ('x3', True)
sp.normalize_variable_token('sin')
# -> ('sin', False)
```

See the [Normalization](../api.md#normalization) API reference for details.

## Performance

The inline phase (`simplify`, conversions, validation) runs in a compiled Rust extension
(`simplipy._core`) on interned token ids. `!`-sort certificates are memoized —
per call, and in a generational per-engine cache — so repeated match attempts cost
nothing, and the fixpoint loop memoizes whole passes and rule-normal subtrees. There is one compiled
engine line; the published ruleset artifacts are the distinguishing factor between engines,
and rule application always considers every pattern in the loaded artifact.

The published fair benchmark is pre-registered: serial
single-core for every arm (one pinned core of an otherwise idle AMD Ryzen 5 2600,
BLAS thread caps at 1), paired per-row scoring against SymPy 1.14.0
(1 s cap, censoring stated on the panel), three corpora — an SR training
prior (n = 65,536), its raw-masked transform (n = 65,536), and an
external neutral problem set (n = 528). Scored in the deployment space
under the MDL measure, with bootstrap 95% CIs; ratio = output/input, lower
is better; "made bigger" = the fraction of rows an arm inflated. Measured
on the 0.14.0 release: `f64` is the shipped default (`effort=4`), `corpus`
is `Mode.corpus` at its default.

| corpus | arm | mean ratio | wins | made bigger |
|---|---|---|---|---|
| SR prior, unmasked | simplipy f64 (default) | **0.966** | 11.7% | **0.0%** |
| | simplipy corpus | **0.940** | **29.5%** | **0.0%** |
| | sympy simplify | 1.078 | 15.1% | 40.1% |
| SR prior, masked raw | simplipy f64 (default) | **0.993** | 7.0% | **0.0%** |
| | simplipy corpus | **0.961** | **28.4%** | **0.0%** |
| | sympy simplify | 1.059 | 14.9% | 38.0% |
| external set | simplipy f64 (default) | 0.995 | 4.0% | **0.0%** |
| | simplipy corpus | 0.995 | 4.5% | **0.0%** |
| | sympy simplify | 1.045 | 15.0% | 22.7% |

No simplipy arm inflated a single row of 131,600, in either mode: refusal
semantics mean an unprovable rewrite returns the input unchanged, and the
serve construction never returns a costlier form than the input under its
own mode's measure. SymPy's `simplify` inflates roughly four rows in ten
on SR-shaped corpora and hits its 1 s timeout on ~20% of them (censored
rows score ratio 1, the charitable choice). Paired wall-clock on the same
rows: median speedup **~650–780×** across corpora, medians at 211–265 µs
per row against SymPy's ~171 ms. On the external set both systems are
near the fixpoint; SymPy's wins there are dominated by number-respelling
(floats rewritten as exact rationals), not structural simplification.

![ECDF, masked raw corpus](../assets/benchmarks/ecdf_masked_raw.png)

Full panels: [unmasked](../assets/benchmarks/ecdf_unmasked.png) ·
[masked raw](../assets/benchmarks/ecdf_masked_raw.png) ·
[external](../assets/benchmarks/ecdf_external.png)
