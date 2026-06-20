# Conversion-function quirks: analysis, impact, and fixes

Six surprising/incorrect behaviors in the `prefix_to_infix` / `infix_to_prefix` / `convert_expression`
chain, found while building the faithful Rust re-port (simplipy-rust). They are present in BOTH the
deployed tag `c84741f` (engine-id `dev_7-3`) and HEAD `1fe9b7e` (the 4 numeric commits do not touch
these functions). This branch (`fix/conversion-quirks`, off HEAD) corrects them as a NEW improvement
version; the tag / `dev_7-3` stays frozen so v23.0 remains reproducible.

Impact measured 2026-06-20 with the faithful port (= tag behavior) vs this branch, on the real
distributions: 27,030 skeleton-derived `convert_expression` inputs + 49,572 real PySR
`predicted_expression` strings from the eval pickles. "Skeleton-path" approximates training-data
canonicalization (skeleton_pool.py builds pools via SymPy-simplify -> `parse`); "PySR" is the eval /
baseline ingestion path (model_adapters.py `infix_to_prefix(str(best['equation']))`).

| # | quirk | example | impact (measured) | severity |
|---|-------|---------|-------------------|----------|
| 1 | fractional power silently DROPPED | `pow2(pow1_3(x))` → `pow2(x)` | **~2% of all expr; 3.8% skeleton / 8.8% PySR of root-bearing expr** | **serious** |
| 2 | `pow0`, an undefined operator token | `['**','x1','0']` → `['pow0','x1']` | 0 skeleton / 18 PySR | minor (eval) |
| 3 | double negation → literal `--5` | `['neg','-5']` → `['--5']` | 0 skeleton / 11 PySR | minor (eval) |
| 4 | `^` vs `**` parse unary-minus differently | `x1 ^ - x2` → binary `-` | up to 907 PySR (`^ -` trigger) | edge (eval) |
| 5 | left-assoc operators chain right | `1/2*m*v**2` → `1/(2*m*v**2)` | 68.8% fastsrb GT structural; **6.5% VALUE-changing (incl ½mv²)**; 0% PySR | **fixed (coordinated parse+render)** |
| 6 | raw `powN` token crashes (asymmetric) | `convert_expression(['pow7','x1'])` → KeyError | 0 (out-of-contract via `parse`) | edge |

## #1 — fractional power silently dropped (`convert_expression` pass-2, the serious one)
**Mechanism.** The pass-2 pow-chain combiner picks `r'pow\d+'` (NO negative lookahead) to extend an
INTEGER-power chain. That regex matches `pow1_3` (sees the `pow1` prefix), so when a `pow{N}` node's
operand is a `pow1_M`, the chain absorbs it, multiplies the exponent by `re.match(r'pow(\d+)','pow1_3')
.group(1)` = `1`, and DROPS the `_M` denominator. `pow2(pow1_3(x))` → `pow2(x)`. Asymmetric: the
fractional chain uses `r'pow1_\d+'`, which won't match `pow2`, so `pow1_3(pow2(x))` is preserved.
**Impact.** Fires on **499/27,030 (1.85%)** skeleton-derived + **1,145/49,572 (2.31%)** PySR inputs;
**3.8% / 8.8%** of root-bearing expressions are corrupted. Real, frequent, semantic (the canonical form
denotes a different function). Reaches the training-data canonicalization path (skeleton_pool SymPy
round-trip), so v23.0's skeleton pools likely contain a ~2% subset with a dropped root.
**Correct behavior / fix.** Give the integer-chain extension the same `(?!_)` lookahead as the outer
dispatch: `operator_patterns = [r'pow1_\d+', r'pow\d+(?!_)']` (engine.py:838). Then a `pow1_M` child is
NOT absorbed into an integer chain; it is combined on its own fractional chain and preserved. **DONE on
this branch.**

## #2 — `pow0`, an undefined token (`convert_expression`)
**Mechanism.** A zero exponent (`['**','x1','0']`, or a float rounding to 0 under `limit_denominator`,
or a `0/N` integer-fraction) builds `f'pow{0}'='pow0'`, which is not a configured operator. It survives
because `factorize_to_at_most(0)` raises (`p<1`) and the except-fallback keeps the constructed node.
**Impact.** 0 skeleton (SymPy pre-simplifies `x**0`), 18 PySR. **Correct behavior:** `x**0 → 1` (the
multiplicative identity; masked to `<constant>` downstream). **Fix:** at each `pow0`-producing site,
emit `['1']` instead of `['pow0', [base]]`. **DONE on this branch.**

## #3 — double negation → literal `--5` (`convert_expression`)
**Mechanism.** The neg branch prepends `-` to a numeric leaf; the `elif` meant to STRIP a leading minus
has a guard identical to the `if` (dead code), so negating `-5` yields `--5`.
**Impact.** 0 skeleton, 11 PySR. **Correct behavior:** strip when already negative (`-5 → 5`), prepend
otherwise. **Fix:** make the `elif` live with guard `startswith('-')` -> `s[1:]`. **DONE on this branch.**

## #4 — `^` and `**` disagree on unary minus (`infix_to_prefix`)
**Mechanism.** Unary-minus detection reads the RAW left-neighbor; `^`→`**` normalization is applied only
to the current token, never the lookahead, and `^` is not in `operator_precedence_compat`. So `x1 ** -
x2` → `neg`, but `x1 ^ - x2` → binary `-`.
**Impact.** `^` reaches `infix_to_prefix` raw on the PySR eval path (model_adapters does NOT pre-replace
`^`); 907 PySR strings carry the `^ -` trigger. **Correct behavior:** treat `^` like `**` for the
lookahead. **Fix:** normalize the lookahead token (or add `^` to the membership set). **DONE on this branch.**

## #5 — left-associative operators chain right (`infix_to_prefix`) — the second serious one
**Mechanism.** Right-to-left shunting-yard popping on `>=` precedence yields right-leaning trees for
left-assoc ops: `a-b-c → a-(b-c)`, `x1/x2/x3 → x1/(x2/x3)`, and (mixed `*`/`/` at equal precedence)
`1/2 * m * v**2 → 1/(2*m*v**2)`. Value-changing whenever the chain is non-associative (`-`, multiple
`/`, or a leading `1/2 *`).
**Impact (the "self-consistent" hold was WRONG -- measured).** 0% on PySR (sympy fully-parenthesizes
its output), which is why it looked benign at first. But on UN-parenthesized external infix it is
common: on fastsrb GT (parsed via `data_sources.py:985`) **68.8% structurally misparse, 6.5% (30/465)
change VALUE** -- including the textbook `1/2 * m * v**2` (kinetic energy) and `3/2 * P * V`, which the
engine parsed as `1/(2*m*v**2)`. So ~6.5% of fastsrb benchmark GT targets were mis-parsed. (Self-
consistency held only for the engine round-tripping its OWN fully-disambiguated render.)
**Correct behavior / fix (COORDINATED parse+render).** The right-leaning parse was COUPLED to
`prefix_to_infix`'s flattening render: the render flattens equal-precedence right operands of `+`/`*`
(`a*(b*c) → a * b * c`) and the right-leaning parse reconstructs that exact prefix, giving prefix↔infix
round-trip identity (`tests/test_prefix_infix.py::test_prefix_to_infix_complex_stress`). A standalone
parse change breaks that round-trip. So the fix is TWO coordinated halves: (1) **parse half** --
`infix_to_prefix` pops on strict `>` for left-assoc operators (`>=` only for `**`/`pow`); (2) **render
half** -- `prefix_to_infix`'s `right_allows_flatten` is disabled, so an equal-precedence right operand
keeps its parens. Together the right-operand paren rule becomes `<=` for left-assoc, which is exactly
the inverse of the left-assoc parse: round-trip identity is preserved AND external un-parenthesized
infix parses correctly (`1/2*m*v**2 → (1/2)*m*v**2`; `a-b-c → (a-b)-c`; `a**b**c` stays `a**(b**c)`).
Cost: right-NESTED associative chains render with explicit parens (`x1 * (x2 * x3)`); left-nested stay
flat (`x1 * x2 * x3`) -- 6 `test_prefix_to_infix_expected_output` cases updated to the (correct) parens.
**DONE on this branch.** (First attempt was the parse-only change -- it broke the round-trip stress
test and was reverted before this coordinated version; lesson: a parse quirk coupled to the render must
be fixed on both sides.)

## #6 — same `powN`, opposite fate (`convert_expression`) — edge
**Mechanism.** `['**','x1','7']` builds `pow7` as a node (fine), but the literal token `['pow7','x1']`
crashes (`operator_arity_compat[op]` hard index in pass-1; pass-2 uses `.get`). **Impact.** Never via
`parse` (`infix_to_prefix` emits `**`, not `powN`). **Fix:** `.get(op, 1)` in pass-1 (or validate). **DONE on this branch.**

## v23.0 implications + the "frozen dev_7-3" precision
#1 reaches the training-data canonicalization path (~2% of v23.0 skeleton-pool expressions likely have
a dropped root); #5 mis-parsed ~6.5% of fastsrb benchmark GT targets (eval-side). **Decision (user,
2026-06-20): NO retrain / no GT-regeneration -- accept as a known noise floor; just fix the engine going
forward.** The faithful `dev_7-3` port + this fixed branch make the exact affected set enumerable on
demand if that ever changes.

**Precision on "frozen `dev_7-3`":** there is NO engine-id gating in `engine.py` -- this branch is an
UNCONDITIONAL code change, i.e. a package **version** bump that changes behavior for every engine-id
(including `dev_7-3`) once released/installed. `dev_7-3`-as-trained therefore stays reproducible only by
PINNING the old package version / tag `c84741f` (what v23.0 used), not by anything in this branch. The
Rust port is the one place the old and new behaviors coexist in a single build, gated by the `fixed`
flag (faithful `dev_7-3` = default; corrected = `*_fixed`). Name/release the fixed version as a new
engine-id before any pipeline switches to it.
