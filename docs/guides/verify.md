# Verifying rulesets

`simplipy.verify` is a **second, independent soundness authority** alongside
the miner's own certification. Where the miner certifies each rule as it is
discovered (via the compiled core), this package re-judges a *finished* rule
set — deliberately implemented independently of the compiled core, so it
cross-checks the engine rather than echoing it. It is a power-user surface:
verdicts are environment-qualified (see
[environment qualification](../method/environment-qualification.md)), and the
judge refuses what it cannot evaluate rather than guessing.

## The gate: `verify_ruleset`

Judges every rule at its own symbolic trigger points under an
arbitrary-precision contract evaluator, classifying each into eight buckets.
Coverage is 100% of the rule set by construction.

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
`JUDGE-TIMEOUT`, `UNRESOLVED-COVERAGE` — is unshippable in every mode: the
second authority never reached an answer, and "confirmed by two independent
authorities" is the claim a shipped rule makes.

`KILL` and `ENGINE-MISALIGN` are different, and this is what the modes are for.
Each rule also gets a **tier**, from two independent questions: is it true over
ℝ, and does the deployed f64 evaluator reproduce it?

| tier | true over ℝ | f64-realised | serves |
|---|---|---|---|
| `core` | yes | yes | every mode |
| `real` | yes | no | `real`, `corpus` |
| `f64` | no | yes | `f64`, `corpus` |
| `reject` | no | no | nothing — recorded in the drop census |

## Cleanliness is per mode

`verify_ruleset(rules, mode=...)` asks whether every rule's tier is one that mode
licenses. The same rule can be clean for one file and a defect in another, which is
the whole point and which a bucket count cannot express:

```python
rule = [['atanh', 'tanh', 'x0'], ['x0']]     # true over R; f64 gives inf past 18.99

verify_ruleset([rule], mode='real')['is_clean']    # -> True   belongs there
verify_ruleset([rule], mode='corpus')['is_clean']  # -> True
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
report = verify_triple('rules.json', 'rules_real.json', 'rules_corpus.json')
report['is_clean'], report['relationships']
```

The same tier routing gates the miner's own output, so the two authorities cannot
drift apart.

Single rules can be judged directly:

```python
from simplipy.verify import verify_rule

verify_rule(['*', 'x0', '1'], ['x0'])['verdict']
# -> 'CERTIFIED'
```

## The monitor: `monitor_ruleset`

The gate's complement: instead of a per-rule scan, `monitor_ruleset` runs the
*deployed* engine over an adversarial-plus-sampled corpus and re-judges every
input→output rewrite under an independent high-precision evaluator,
attributing any deployed-value violation to the responsible rule. Both the
gate and the monitor carry poison self-tests (`selftest`) that must pass
before their verdicts are trusted.

## Promotion: `simplipy.promotion.promote`

The sort-promotion certifier: it re-derives each pattern rule's binding sort
(`_`/`?`/`!`/`$` — see [Creating rulesets](../rules.md)) from its own
numeric evidence, promoting a rule to a wider sort only on a certified
witness. Its internals are a verbatim port of an external ratified certifier
and are private; `promote(rules, engine)` is the entry point. Mining calls it
for you (`find_rules(promote_sorts=True)`).
