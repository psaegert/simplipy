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
| 5 | left-assoc operators chain right | `a - b - c - d` → `a-(b-(c-d))` | pervasive but self-consistent | **HELD — confirm intentional** |
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

## #5 — left-associative operators chain right (`infix_to_prefix`) — HELD
**Mechanism.** Right-to-left shunting-yard popping on `>=` precedence yields right-leaning trees for
left-assoc ops: `a-b-c-d → a-(b-(c-d))`, `x1/x2/x3 → x1/(x2/x3)`. Value-changing for `-`/`/`.
**Status.** Pervasive but the engine round-trips self-consistently. **HELD pending confirmation that it
is unintentional** — fixing it (pop on `>` for left-assoc) is a large, distribution-wide behavior change.

## #6 — same `powN`, opposite fate (`convert_expression`) — edge
**Mechanism.** `['**','x1','7']` builds `pow7` as a node (fine), but the literal token `['pow7','x1']`
crashes (`operator_arity_compat[op]` hard index in pass-1; pass-2 uses `.get`). **Impact.** Never via
`parse` (`infix_to_prefix` emits `**`, not `powN`). **Fix:** `.get(op, 1)` in pass-1 (or validate). **DONE on this branch.**

## v23.0 implications (downstream decision)
#1 reaches the training-data canonicalization path, so ~2% of v23.0's skeleton-pool expressions
likely have a dropped root -> a small but real label-corruption subset. Whether that warrants
regenerating the affected skeleton pools + retraining (a large commitment) vs. accepting it as a known
~2% noise floor is a separate decision, informed by these numbers. The faithful `dev_7-3` port + the
fixed branch together make the exact corrupted set enumerable on demand.
