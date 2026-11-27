# Upgrading to 0.15

This page is for users of SimpliPy 0.14. It lists what changes in 0.15, in the order you are
likely to meet it, and what to do about each change. The [compatibility policy](compatibility.md)
has the full tables of removed and renamed names, and the CHANGELOG has every change in detail.

**In short:** install 0.15 with the engine published for it, rename `f64` to `strict`, and
regenerate anything that stores answers or prices (test expectations, caches, datasets). Every
mode's answers change, and so do the prices `complexity` returns.

```python
import simplipy as sp
from simplipy import Mode

engine = sp.SimpliPyEngine.load("acj-4-llm_0.15", install=True)
```

## 1. Rule files from 0.14 do not load

0.15 changed the canonical forms, the price of an expression and the rule judge, so a rule set
mined by 0.14 is no longer minimal, correctly ordered, or, for some rules, true under 0.15's
criterion. Loading one raises:

- a rule file whose provenance sidecar (`rules.json.provenance.json`) records a simplipy version
  before 0.15 raises `IncompatibleArtifactError`, naming the version that mined it;
- a config that names a `rules_real:` set raises `ValueError`, since the `real` mode is retired.
  `acj-5-4-llm` and `acj-4` as published for 0.14 name one, but their rule files' version is
  checked first, so they raise the `IncompatibleArtifactError` above.

What to do: load the engines published for 0.15 (`acj-4-llm_0.15`, the default, and
`acj-5-4-llm_0.15`), or pin `simplipy<0.15` to keep serving a 0.14 artifact. A rule file without
a sidecar is treated as hand-written and still loads. A 0.15 artifact holds two rule files,
`rules_strict.json` and `rules_permissive.json`, and its `config.yaml` names them; a config may
still call its strict file `rules.json`.

## 2. The `f64` mode is now `strict`

The default mode has a new name and a precise promise.

| 0.14 | 0.15 |
|---|---|
| `Mode.f64`, `mode='f64'`, `mode='default'` | `Mode.strict`, `mode='strict'` (the old names still work in 0.15, with a `DeprecationWarning`, and are removed in 0.16) |
| `Mode.real`, `mode='real'` | removed: raises `ValueError` |
| `Mode.permissive` | unchanged name |

```python
Mode('strict') is Mode.strict
# -> True
```

**What `strict` promises.** Every step it takes is either reproduced by float64 itself (the two
sides agree within 8 units in the last place) or true in the real numbers with float64 giving
both sides the same kind of value at every point: both finite, the same infinity, both
undefined, and a zero's sign wherever it changes a value on a region. Where an operation
overflows or underflows, a step may give the true value instead of float64's. The promise is
made for each step, not for the whole answer: where float64's evaluation of your input is
ill-conditioned, the answer can differ from it by much more than rounding, never in kind.
[What `strict` mode does and does not promise](guides/simplify.md#what-strict-mode-does-and-does-not-promise)
explains this with examples.

**What changed for `permissive`.** It includes `strict`: every rule `strict` holds, `permissive`
holds too, so `permissive` simplifies at least as far as `strict` wherever a strict rule applies. Its own promise is per
rule as well: each rule agrees with the reals' values up to float resolution, or holds in
`strict`; its named simplifications (such as `sqrt(x^2)` becoming `x`) may drop a domain or a
sign. Use it only to canonicalize training data, never on an inference or scoring path.

## 3. Numbers are exact

A decimal is the rational number it spells, and `+`, `-`, `*`, `/` and integer powers between
numbers are computed exactly, in both modes. Other functions of a number are float64's function
of the nearest double in `strict`, and the real function in `permissive`.

```python
engine.simplify('0.1 + 0.2')          # -> '0.3'
engine.simplify('1e16 + 1 - 1e16')    # -> '1'
```

numpy gives `0.30000000000000004` and `0` for the same sums.

A fully numeric part is written as the cheapest number the mode admits. In `strict` that is
float64's value, or the real value where float64 agrees with it in kind; in `permissive`, the
real value. A part with no real value is NaN in `permissive`; `strict` keeps float64's value.
`floor(tanh(100))` is 0 in both modes, though float64 rounds `tanh(100)` to 1.

```python
engine.simplify('sin(np.pi)')                     # -> '0'
engine.simplify('exp(log(3))')                    # -> '3'
engine.simplify('exp(sin(np.pi))')                # -> '1'
engine.simplify('1/0')                            # -> 'inf'
engine.simplify('1/0', mode=Mode.permissive)      # -> 'nan'
```

A literal that float64 reads as exactly the double of a short symbolic number is read as that
number: `3.141592653589793` is `pi`, `0.3333333333333333` is `1/3`, `0.6931471805599453` is
`log(2)` (the full list is under
[How numbers are printed](guides/simplify.md#how-numbers-are-printed)).

```python
engine.simplify('0.3333333333333333*x0')   # -> 'x0/3'
```

## 4. Canonical forms changed

Both modes write a number that several terms of a sum share once, and write polynomials
expanded rather than nested:

```python
engine.simplify('3.71649*x1 + 3.71649*x2 + x3')   # -> 'x3 + 3.71649*(x1 + x2)'
engine.simplify('x1*(x1*(0.4*x1 + 2.7) + 1.3)')    # -> '1.3*x1 + 2.7*x1^2 + 0.4*x1^3'
```

So a masked answer holds one `<constant>` for a shared number where it held one per term.
In `permissive`, a zero factor absorbs the whole product, defined or not, with or without
rules loaded:

```python
engine.simplify('0*log(x2)', mode=Mode.permissive)   # -> '0'
```

The search for a cheaper form now runs until a round finds nothing: `simplipy.DEFAULT_EFFORT`
is `None` (it was 4), so a second call returns the answer unchanged. Pass `effort=4` for the old
cap, or `effort=0` on throughput-critical paths.

## 5. Prices changed, and there is a second price

`complexity(e, mode=m)` prices `e` as mode `m` reads it, and every number costs its own exact
spelling, so prices change in both modes and are not comparable across modes. The `canon=`
argument is removed; `simplipy.mining.cross_mode_complexity` compares answers of different
modes on one scale.

`written_complexity(e)` is new. It prices an expression exactly as it is written, with the same
code book and no simplification, so a redundant spelling costs more:

```python
engine.written_complexity('x0 + 0')   # -> 11000
engine.complexity('x0 + 0')           # -> 6000
```

Both return milli-bits. The [written price](guides/simplify.md#the-written-price) section
explains the difference.

## 6. What the engine reads

- **Named constants.** `<constant_NAME>` (NAME letters and digits) is one unknown shared by every
  occurrence of the name; `<constant>` stays one independent constant per occurrence. 0.14 read
  `<constant_a>` as an ordinary variable. See [Named constants](guides/simplify.md#named-constants).
- **More operators.** The engines published for 0.15 declare thirteen operators beyond 0.14's
  23: `floor`, `ceil`, `max`, `min`, `mod`, `atan2`, `erf`, `erfc`, `harmonic` and the Bessel
  functions `besselj`, `bessely`, `besseli` and `besselk`. Their float64 values are the package's own kernels or the platform's libm, listed
  in [the float64 realizations](guides/simplify.md#what-strict-mode-does-and-does-not-promise)
  table. Where they have no value: `harmonic` at the negative integers; `bessely` and `besselk`
  for `x <= 0` (infinite at 0); `besselj` and `besseli` for `x < 0` at a non-integer order;
  `mod` by zero.
- **Supported but not mined.** `erfc` and `harmonic` are read, evaluated, priced and written,
  but no mined rule mentions them, so no answer introduces them: `1 - erf(x0)` stays as written.
- **Slow operators.** The four Bessel functions are slow to evaluate and form one family: a rule
  may bring one into an answer only where the input already holds one of the family. See
  [Slow operators](rules.md#slow-operators).
- **Defined functions.** `sqrt`, `cot`, `sec`, `csc`, `acot`, `asec` and `acsc` are defined over
  the operators (`sqrt(u)` is `rootn(u, 2)`, `cot(u)` is `1/tan(u)`, and so on, with SymPy's
  conventions). The readers keep them as written; `simplify` expands them, so
  `simplify('cot(x0)')` is `'1/tan(x0)'`. See [Defined functions](guides/artifacts.md#defined-functions).
- **Aliases.** `Abs(x)` reads as `abs(x)` and `div(a, b)` as `a/b`.
- **Variable names.** An expression over `v1`, `x_1` or `alpha` now gets the answer it gets over
  `x1`, renamed. In 0.14 some simplifications did not happen over such names.

## 7. Everything else

- `load` compiles every rule set it is asked for before it returns; `load(..., modes=('strict',))`
  builds only the strict set, and `engine.unload_mode(mode)` frees one.
- Messages go through the `simplipy` logger instead of being printed. Silence them with
  `logging.getLogger('simplipy').setLevel(logging.ERROR)`.
- An installed asset loads without the network.
- `simplify(..., return_status=True)` returns `(answer, status)`, and the status says whether a
  resource cap stopped the call.
- Mining: `prune_covered_rules`, `deduplicate_rules` and `promote` require `rule_mode=`, and a
  mine writes one file per mode. See [Creating rulesets](rules.md) and the
  [compatibility tables](compatibility.md).

## A checklist

1. Install 0.15 and load the engine published for it.
2. Replace `f64` and `'default'` with `strict`, and remove any use of `real`.
3. Regenerate stored answers, prices and masked datasets.
4. If you mine your own rules, mine them again with 0.15.
5. If you compared `DEFAULT_EFFORT` with an integer, it is now `None`.
