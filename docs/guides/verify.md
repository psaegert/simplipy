# Verifying rulesets

`simplipy.verify` is the **rule judge**: it decides whether a rule holds in a mode,
and the miner's own certification is a pre-filter that may only refuse what the judge
refuses. Every value it compares comes from one table, defined once in the compiled
core: read in the reals, a literal is the rational it spells and a pole takes its limit
where every direction gives the same one (`log(0)` is `-inf`, `tan(pi/2)` has no value), and
a division by zero has no value however it is written (`1/0`, `pow(0, -2)`, `mod(1, 0)`), so
`x^-2` and `1/x^2` agree everywhere; read in `f64`, an operation is the engine's own float64
kernel. `permissive` reads the reals'
values. The judge's own part is the
procedure: the precision ladder, the fitted constants and the sample points. It is a
power-user surface: verdicts are environment-qualified (see
[environment qualification](../method/environment-qualification.md)), and the judge
refuses what it cannot evaluate rather than guessing.

## The gate: `verify_ruleset`

Judges every rule at its own symbolic trigger points, classifying each into nine
buckets. Coverage is 100% of the rule set by construction.

A rule holds in the reals when, wherever its left side has a value, an infinity included, its
right side has the same value, and its right side gives a value where the left side has none
only on a set of measure zero, decided by certified cells (`x0/x0 -> 1` holds; `exp(log(x0)) ->
x0` does not, and neither does `abs(log(cos(x0))) -> log(1/cos(x0))`, whose left side is `+inf`
at pi/2, where the right side has no value). In `f64` the same cells are read in float64's
values, where a saturation plateau is a cell like any other (`tanh(x)` is 1.0 past 19.06), and
an extension at a double counts as measure zero only where its neighbouring doubles are no
extension (`x0/x0 -> 1` at 0). A rule without variables has one point and no tolerance; a `_`
slot is checked at `+inf`, `-inf` and NaN; each `<constant>` is its own symbol, and nothing is
owed at a constant where the left side has no finite value.

```python
from simplipy.verify import verify_ruleset

report = verify_ruleset([
    [['*', 'x0', '1'], ['x0']],          # sound: multiplicative identity
    [['exp', 'log', 'x0'], ['x0']],      # NOT sound: undefined for x0 <= 0
])
report['is_clean']
# -> False
report['buckets']['CERTIFIED']
# -> [0]
report['buckets']['KILL']
# -> [1]
```

The second rule is the classic trap this package exists to catch:
`exp(log(x0))` equals `x0` only on `x0 > 0` — rewriting it changes the
function on a positive-measure set, so the judge kills it.

A rule the judge cannot evaluate — `NO-WITNESS`, `UNSUPPORTED-SHAPE`,
`JUDGE-TIMEOUT`, `UNRESOLVED-COVERAGE` — is unshippable in every mode, and one whose
measure the cells cannot decide (`UNDECIDED-MEASURE`) is refused in `f64`: an
undecided check never reads as a pass.

`KILL` and `ENGINE-MISALIGN` are different, and this is what the modes are for.
Each rule also gets a **tier**: the modes it holds in, from two independent questions. Does
the deployed f64 evaluator reproduce it? Does it hold in `permissive`, whose criterion is
the reals' values up to float resolution, and agreement alone where the input has no value
(a permissive rule may give a value where its input has none, over real-valued bindings)?
`permissive` binds every slot to any subexpression, so its question is asked at the `_`
sort: no disagreement at an isolated point is excused there. The tiers are one registry,
`simplipy.verify.TIERS`, and this table is rendered from it (`simplipy.verify.tier_table()`):

<!-- tier-table: begin (rendered by simplipy.verify.tier_table(); a test checks it) -->
| tier | `f64` | `permissive` | meaning |
|---|---|---|---|
| `core` | yes | yes | holds in f64 (the deployed evaluator reproduces it) and in permissive |
| `f64-only` | yes | no | float64's own value, off the reals beyond float resolution (`floor(tanh(40)) -> 1`), or true by an allowance a slot guards, which permissive's any-subexpression binding does not give |
| `permissive` | no | yes | true in the reals and not reproduced by float64 (`atanh(tanh(x0)) -> x0`), or an extension (`exp(log(x0)) -> x0`) |
| `reject` | no | no | holds in no mode; recorded in the drop census |
<!-- tier-table: end -->

"f64-realised" means the deployed evaluator computes the rewrite, to within 8 ULP, at every
battery and grid point where no operation of either side overflows or underflows. At a point
where one does, the rewrite may give the true value instead, as the canonical constructors do,
so `rootn(exp(x0), 2) → exp(x0/2)` is `core` although `exp` overflows past 709.78. A pole
(`atanh(1)`, `log(0)`) is not an overflow and saturation inside the range is not one either,
so `atanh(tanh(x0)) → x0` stays `permissive`. The verdict's `out_of_range` counts the points read
this way.

## Cleanliness is per mode

`verify_ruleset(rules, mode=...)` asks whether every rule's tier is one that mode
licenses, and whether the load admits it: the load's gates are policies (the constant count,
a ground composite, the ordering, what the engine does itself, well-formedness), read through
one predicate, `load_policy_refusal`, which the load, `licensed` and `verify_ruleset` share.
`asin(1) -> acos(0)` holds in every mode and is clean in none (`refused_by:
['ground_composite']`). The same rule can be clean for one file and a defect in another,
which is the whole point and which a bucket count cannot express:

```python
rule = [['atanh', 'tanh', 'x0'], ['x0']]     # true over R; f64 saturates to inf at large t

verify_ruleset([rule], mode='permissive')['is_clean']  # -> True   belongs there
verify_ruleset([rule], mode='f64')['is_clean']         # -> False  f64 contradicts it
```

`report['offenders']` names the rules that fail, with their tier. Omitting `mode`
keeps the mode-less meaning — only `CERTIFIED`/`TOLERATED` are clean — because that
is what every caller written before the split means by the word; it is **not** the
right gate for a shipped artifact.

For a whole artifact use `verify_artifact`, which sweeps each file against its own
mode's contract **and** checks the relationship between them. A routing bug that
moved a rule between sets leaves every individual sweep clean:

<!-- docs-example: skip: reads a published artifact off disk; the paths are per-artifact -->
```python
from simplipy.verify import verify_artifact
report = verify_artifact('rules.json', 'rules_permissive.json')
report['is_clean'], report['relationships']
```

With `engine_config=` it also checks that `permissive` is at least as capable as `f64` on
every expression of a corpus: `corpus_rows`, or by default the 400-expression corpus that
ships with the package (`simplipy.verify.default_corpus()`). The check runs on the config's
operators and the two rule sets passed in, prices both modes' answers on one shared scale,
and counts a row on which a mode raises as a violation. `verify_triple`, the gate of the
earlier three-file artifact, raises: its middle file was the retired real mode's.

The same tier routing gates the miner's own output, so the two authorities cannot
drift apart.

Single rules can be judged directly:

```python
from simplipy.verify import verify_rule

verify_rule(['*', 'x0', '1'], ['x0'])['verdict']
# -> 'CERTIFIED'
```

On two fragments the judge decides a rule exactly, over its whole domain, and reports how
(`verdict['exact']`):

- **Piecewise-affine rules of one variable.** Sides built from one variable, rational literals,
  `+`, `-`, `neg`, products and quotients by a constant, `abs`, `max` and `min` are continuous
  piecewise-affine functions, compared cell by cell with exact rational breakpoints. The rule is
  proved, or refuted at a rational witness: `max(0, 1e-200 - abs(x0 - 0.123456789)) -> 0` is
  false on an interval no sample point reaches, and is refused in every mode. In `f64` such a
  proof carries over only where every operation is exact on doubles (`neg`, `abs`, `max`,
  `min`). Rules of several variables, and functions past 256 breakpoints or 4,096-bit
  coefficients, are left to the sampled procedure.
- **A left side that is one literal everywhere.** Where the interval layer encloses the left
  side, over every finite binding of its variables and constants, by the right side's literal
  and nothing else, the rule is proved (`0*exp(x0) -> 0`).

What a mode's file may hold is `licensed`: the rule holds in the mode, and no named policy
refuses it — evidence (the left side finite on a set of positive measure), the constant
count, ground composites, the mine's cell, the serve ordering, special constants,
duplication of a `_` slot, and what the engine does itself (`policy_refusals`):

<!-- docs-example: skip: needs an engine; see tests/test_differential.py -->
```python
from simplipy.verify import licensed
licensed(['asin', '1'], ['acos', '0'], 'f64', engine)
# -> {'licensed': False, 'holds': True, 'tier': 'core', 'refused_by': ['ground_composite']}
```

## The monitor: `monitor_ruleset`

The gate's complement: instead of a per-rule scan, `monitor_ruleset` runs the
*deployed* engine over an adversarial-plus-sampled corpus and judges every
input→output rewrite as the judge judges a rule, in the mode it was made in,
attributing any violation to the rule set or, where the rules-empty engine
makes it too, to the engine. It has no evaluator of its own: every value it
compares is the core's. Both the gate and the monitor carry poison
self-tests (`selftest`) that must pass before their verdicts are trusted.

## Promotion: `simplipy.promotion.promote`

The sort promotion: each mined rule, certified at the variable-leaf sort
`?`, is respelled at the stronger sorts (`_`, then `!`; see
[Creating rulesets](../rules.md)), and each respelling asks the judge:
`licensed` in the rule's mode. A rule ships at the strongest sort the judge
licenses, and at `?` where it licenses neither. Promotion has no evaluator or
probe list of its own. Its internals are private; `promote(rules, engine,
rule_mode=...)` is the entry point. Mining calls it for you
(`find_rules(promote_sorts=True)`).
