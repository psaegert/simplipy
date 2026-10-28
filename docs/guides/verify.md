# Verifying rulesets

`simplipy.verify` is the **rule judge**: it decides whether a rule holds in a mode,
and the miner's own certification is a pre-filter that may only refuse what the judge
refuses. Every value it compares comes from one table per mode, defined once in the
compiled core: in `real` a literal is the rational it spells and a pole takes its limit
where every direction gives the same one (`pow(0, -2)` is `+inf`, `1/0` has no value), in
`f64` an operation is the engine's own float64 kernel. The judge's own part is the
procedure: the precision ladder, the fitted constants and the sample points. It is a
power-user surface: verdicts are environment-qualified (see
[environment qualification](../method/environment-qualification.md)), and the judge
refuses what it cannot evaluate rather than guessing.

## The gate: `verify_ruleset`

Judges every rule at its own symbolic trigger points, classifying each into nine
buckets. Coverage is 100% of the rule set by construction.

A rule holds in `real` when, wherever its left side has a value, its right side has the
same value, and its right side gives a value where the left side has none only on a set
of measure zero, decided by certified cells (`x0/x0 -> 1` holds; `exp(log(x0)) -> x0`
does not). A rule without variables has one point and no tolerance; a `_` slot is
checked at `+inf`, `-inf` and NaN; each `<constant>` is its own symbol, and nothing is
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
measure the cells cannot decide (`UNDECIDED-MEASURE`) is refused in `real`: an
undecided check never reads as a pass.

`KILL` and `ENGINE-MISALIGN` are different, and this is what the modes are for.
Each rule also gets a **tier**, from three independent questions: does it hold over ℝ,
does the deployed f64 evaluator reproduce it, and does it hold in `permissive`, whose
criterion is agreement alone (a permissive rule may give a value where its input has
none, over real-valued bindings)?

| tier | holds over ℝ | f64-realised | serves |
|---|---|---|---|
| `core` | yes | yes | every mode |
| `real` | yes | no | `real`, `permissive` |
| `real-only` | yes, a `$` slot owing nothing at 0 | no | `real` |
| `f64` | no | yes | `f64`, `permissive` |
| `permissive` | by agreement alone (`exp(log(x0)) -> x0`) | no | `permissive` |
| `reject` | no | no | nothing — recorded in the drop census |

"f64-realised" means the deployed evaluator computes the rewrite, to within 8 ULP, at every
battery and grid point where no operation of either side overflows or underflows. At a point
where one does, the rewrite may give the true value instead, as the canonical constructors do,
so `rootn(exp(x0), 2) → exp(x0/2)` is `core` although `exp` overflows past 709.78. A pole
(`atanh(1)`, `log(0)`) is not an overflow and saturation inside the range is not one either,
so `atanh(tanh(x0)) → x0` stays `real`. The verdict's `out_of_range` counts the points read
this way.

## Cleanliness is per mode

`verify_ruleset(rules, mode=...)` asks whether every rule's tier is one that mode
licenses. The same rule can be clean for one file and a defect in another, which is
the whole point and which a bucket count cannot express:

```python
rule = [['atanh', 'tanh', 'x0'], ['x0']]     # true over R; f64 saturates to inf at large t

verify_ruleset([rule], mode='real')['is_clean']    # -> True   belongs there
verify_ruleset([rule], mode='permissive')['is_clean']  # -> True
verify_ruleset([rule], mode='f64')['is_clean']     # -> False  f64 contradicts it
```

`report['offenders']` names the rules that fail, with their tier. Omitting `mode`
keeps the pre-triple meaning — only `CERTIFIED`/`TOLERATED` are clean — because that
is what every caller written before the split means by the word; it is **not** the
right gate for a shipped artifact.

For a whole artifact use `verify_triple`, which sweeps each file against its own
mode's contract **and** checks the relationships between them. A routing bug that
moved a rule between sets leaves every individual sweep clean:

<!-- docs-example: skip: reads a published triple off disk; the paths are per-artifact -->
```python
from simplipy.verify import verify_triple
report = verify_triple('rules.json', 'rules_real.json', 'rules_permissive.json')
report['is_clean'], report['relationships']
```

With `engine_config=` it also checks that `permissive` is at least as capable as
either other mode on every expression of a corpus: `corpus_rows`, or by default the
400-expression corpus that ships with the package (`simplipy.verify.default_corpus()`).
The check runs on the config's operators and the three rule sets passed in, prices
every mode's answer on one shared scale, and counts a row on which a mode raises as a
violation.

The same tier routing gates the miner's own output, so the two authorities cannot
drift apart.

Single rules can be judged directly:

```python
from simplipy.verify import verify_rule

verify_rule(['*', 'x0', '1'], ['x0'])['verdict']
# -> 'CERTIFIED'
```

What a mode's file may hold is `licensed`: the rule holds in the mode, and no named policy
refuses it — evidence (the left side finite on a set of positive measure), the constant
count, ground composites, the mine's cell, the serve ordering, special constants,
duplication of a `_` slot, and what the engine does itself (`policy_refusals`):

<!-- docs-example: skip: needs an engine; see tests/test_differential.py -->
```python
from simplipy.verify import licensed
licensed(['asin', '1'], ['acos', '0'], 'real', engine)
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
