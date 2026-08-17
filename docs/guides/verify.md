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

**Every non-CERTIFIED/TOLERATED bucket is fatal** (`simplipy.verify.FATAL_BUCKETS`):
a rule the judge cannot evaluate (`NO-WITNESS`, `UNSUPPORTED-SHAPE`,
`JUDGE-TIMEOUT`) or cannot reconcile (`ENGINE-MISALIGN`,
`UNRESOLVED-COVERAGE`) is exactly as unshippable as a `KILL` — the second
authority never passed it. The same fatal set gates the miner's own output,
so the two authorities cannot drift apart.

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
