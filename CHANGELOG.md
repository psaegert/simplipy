# Changelog

## 0.3.0 — Rust inline backend + the "numeric" engine line (MAJOR behavior change)

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
2. **Six conversion bug fixes** (previously documented in `docs/CONVERSION_QUIRKS.md`), most notably:
   a fractional-power child (`pow1_M`) is no longer silently dropped by `convert_expression`; left/right
   associativity in `infix_to_prefix`/`prefix_to_infix` is corrected (and round-trip-preserving);
   `x**0`, `--5`, `^`-vs-`**` unary-minus, and raw `powN` are handled correctly.
3. **`float("inf")` / `float("-inf")` / `float("nan")` are atomic tokens** in the conversion tokenizer,
   so a folded constant round-trips through prefix↔infix instead of fragmenting.

> Reproducibility: to reproduce results generated with the pre-0.3 engine (e.g. models trained against
> `dev_7-3` under SimpliPy 0.2.x), pin `simplipy<0.3`. The 0.2.x behavior is the frozen anchor; 0.3 is a
> deliberate, documented quality improvement, not a silent drift.

### Robustness (review-driven)

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
