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
owed at a constant where the left side has no finite value. Where it has one, a witness for the
right side's constants is owed, and a search that finds none refuses the rule (`NO-WITNESS`):
`tanh(x0 + log(C)) -> tanh(x0 + C)` is refused, its left side being -1 everywhere at C = 0.

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

Each mode is judged on its own values. `f64` holds a rule with variables where float64's two
sides agree (to 8 ULP, the true value accepted where an operation overflows or underflows), or
where the rule is true in the reals and float64's two sides agree in kind -- both finite -- at
every point the judge reads where the reals give both sides a value (there their difference is
rounding), and to 8 ULP at every other point. The second admits the identities float64 only
rounds, however far a jump or a pole magnifies the rounding, and nothing false in the reals:
`sin(x0 + pi) -> -sin(x0)` (1.2e-16 against -0 at x0 = 0), `x0 + 1 - x0 -> 1`,
`floor(x0 + 1) -> floor(x0) + 1` (1 against 0 at x0 = -1e-17) and `floor(tanh(abs(x0))) -> 0`
hold in `f64`; `atanh(tanh(x0)) -> x0` (+inf against x0 past 19.06), `exp(log(x0)) -> x0` (NaN
against x0 for x0 < 0) and `tanh(1/(abs(x0) - x0)) -> tanh(-1/(x0 - abs(x0)))` (1 against -1 on
x0 > 0, where the reals give neither side a value) do not. In both readings `f64` reads a
zero's sign where it changes a value on a region: float64 may give the two sides zeros of
opposite sign, and the reading in kind may admit a difference at which one side's value turns on
a zero's sign (a divisor, `pow` or `rootn` at an odd negative exponent, `atan2`) and the other's
does not, only at isolated points. `0*asin(x0) -> 0*acos(x0)` (-0 against +0 for x0 < 0) and
`atan2(0*x0, -1) -> pi` (-pi against pi for x0 < 0) do not hold in `f64`; `log(exp(x0)) -> x0`
(+0 against -0 at x0 = -0 alone) does, and `log(exp(_0)) -> _0` does not, since a slot that
matches any subexpression can bind one that is -0 on a region. A zero's sign is `f64`'s alone: `permissive`, which reads one zero, includes `f64` with the two zeros one value, so those rules hold in `permissive`, with float64's evidence where `f64` so read holds them (`neg(acosh(erf(x0))) -> acosh(erf(x0))`). `permissive` is judged on the reals' values at float resolution.
The bucket names (`CERTIFIED`, `KILL`, `UNRESOLVED-COVERAGE`, ...) are the reals' outcome, a
diagnostic that decides no mode by itself; `f64` reads it for its second reading. A mode whose reading reaches no answer
refuses the rule, with uncertainty: a witness search that finds none where the mode's left side
is finite (`NO-WITNESS`), a token the judge cannot read (`UNSUPPORTED-SHAPE`; a variable is read
whatever it is called, `v1`, `x7` or `alpha`, as a data slot), a judgment past its
time (`JUDGE-TIMEOUT`), or a point the mode owes a reading at and leaves unread. The owed points
are the ones the judge takes because they are exceptional: every slot at each of its sort's
exact atoms (zero, the units, the infinities, pi/2 and the rest), at each generic or special
constant value the mode owes a witness at, and a ground rule's one point; the verdict names an
unread one (`unread_owed`). The grid, the rows of the shipped mines' matrix (one matrix, the
same for every caller, so a verdict is the rule's), the preimages of jumps, the slots'
magnitude ladder and the constants' span and saturation magnitudes (`|c|` from 0.001 to 0.1
and from 10 to 1e4, and 19 to 1e17) are samples: a point among them that cannot be read counts neither way. A rule with
constants and no slots reads one point per constant value, and at least a quarter of them must
be read.
`f64` reads float64's values: `x^2 + 1 + 1e-3000 -> x^2 + 1` holds in
`f64`, where `1e-3000` is the double 0.0 and the two sides are bit-identical. `permissive`'s
own criterion, agreement at float resolution, needs no exact equality: where the reals' readings
run out of precision (`UNRESOLVED-COVERAGE`, the 6,050 digits `1e-3000` beside 1 demands), the
interval layer's enclosure of both sides at the point decides that every double either side can
round to is the same, so `x + 1e-3000 -> x` holds by it too. (At the atom e, an approximation,
the enclosure of `x^2 + 1` is wider than that, so `permissive`'s own reading of the first rule
leaves an owed point unread; `permissive` holds it through `f64`.) Running out of precision
never reads as equality: `((x + 1e-3000) - x) * 1e3000 -> 0`, whose left side is 1, encloses
unbounded and is refused. A measure the cells cannot decide is refused where it is owed: `f64`
owes it in float64's values, so an undecided float64 measure refuses `f64`;
`UNDECIDED-MEASURE` labels the reals' measure, which `permissive` does not owe. An undecided
check never reads as a pass.

`KILL` and `ENGINE-MISALIGN` are different, and this is what the modes are for.
Each rule gets a **tier**: the modes it holds in, from two questions. Does it hold in `f64`
(the deployed evaluator reproduces it, or it is true in the reals and float64 keeps every
value's kind)? Does it hold by `permissive`'s own criterion, the reals' values up to
float resolution, and agreement alone where the input has no value (a permissive rule may give
a value where its input has none, over real-valued bindings)? `permissive` binds every slot to
any subexpression, so its own question is asked at the `_` sort: no disagreement at an isolated
point is excused there. **Permissive includes f64**: a rule holds in `permissive` where it holds
in `f64` or by permissive's own criterion, and a step permissive takes by an f64 rule promises
what `f64` promises (float64's values agree wherever the left side has one). So
`floor(tanh(40 + abs(x0))) -> 1`, float64's own value and off the reals, and
`ceil(abs(cos(x0))) -> 1`,
false at pi/2 in the reals and true at every double, hold in both modes; `ceil(x0) - floor(x0)
-> 1` holds in neither. A verdict carries each mode's reading: `realised` for `f64` (from its two
readings of float64, `realised_f6` and `realised_kinds`, and the reals' outcome) and `permissive`
for `permissive`, which includes `f64`'s: `permissive_own` is its own criterion's reading alone
(`2*tanh(x0)/(1 + tanh(x0)^2) -> 2/(tanh(x0) + 1/tanh(x0))` holds in `f64`, float64's `1/0` being
`inf`, and so in `permissive`, though its own criterion, which reads `1/0` as no value, refuses
it at x0 = 0). The tiers are one registry, `simplipy.verify.TIERS`, and this table is rendered
from it (`simplipy.verify.tier_table()`):

<!-- tier-table: begin (rendered by simplipy.verify.tier_table(); a test checks it) -->
| tier | `f64` | `permissive` | meaning |
|---|---|---|---|
| `core` | yes | yes | holds in f64 (the deployed evaluator reproduces it, or it is true in the reals and float64 keeps every value's kind: `sin(x0 + pi) -> -sin(x0)`), and so in permissive, which includes every rule f64 holds |
| `permissive` | no | yes | holds in permissive alone: true in the reals at float resolution and refused by f64 (`atanh(tanh(x0)) -> x0`, where float64 gives +inf against x0), an extension (`exp(log(x0)) -> x0`), or held by f64 with the two zeros one value and refused by strict for a zero's sign (`0*asin(x0) -> 0*acos(x0)`) |
| `reject` | no | no | holds in no mode; recorded in the drop census |
<!-- tier-table: end -->

`f64`'s first reading means the deployed evaluator computes the rewrite, to within 8 ULP, at
every battery and grid point, and at every slot magnitude from the smallest normal double to the
largest, where no operation of either side overflows or underflows; its second, that the rule is
true in the reals and both sides have one kind at all those points (`sinh(asinh(x0)) → x0` is 66
ULP off at 1e100 and holds by the second). At a point
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

The converse does not happen: permissive includes f64, so a rule that holds in `f64` holds in
`permissive` too (a policy can still tell the files apart, the ordering being each mode's own).

`report['offenders']` names the rules that fail, with their tier. Omitting `mode`
keeps the mode-less meaning — only the reals' `CERTIFIED`/`TOLERATED` are clean — a
diagnostic that decides no mode; it is **not** the right gate for a shipped artifact.

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

Equality is undecidable for the language rules are spelled in (Richardson, 1968), so the judge
cannot decide every rule exactly. It gives every rule a definite answer and says what the answer
rests on, per mode (`verdict['basis']`):

- **`proved`**: an exact route decides the rule over its whole domain (below);
- **`evidence`**: the rule holds at every point the judge read (its sort batteries, a grid, the
  mine's rows, the preimages of jumps, at a precision that is climbed until the reading settles)
  and the measure of clause (b) is decided by certified cells; no counterexample was found, which
  is no proof;
- **`refused`**: a counterexample, or a check that could not decide (an undecided check never
  reads as a pass). Two readings that agree only because the precision ran out are no equality:
  a point whose operands demand more than 2,000 digits is undecided
  (`x^2 + 1 + 1e-3000 -> x^2 + 1` is refused in `permissive`, which reads the reals; at 1e-100
  the readings reach the separation).

A mine records the basis of every rule of each file in its provenance sidecar (`basis`), and
`verify-mine` reports how many rules each file holds on evidence alone.

The exact routes (`verdict['exact']`):

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
- **A rule without variables or constants** has one point, read by the ground evaluator the
  engine's fold and `verify-mine` read too. In `f64` it holds where its two sides are the same
  double, or the reals give them one value and float64's two values agree in kind (both finite,
  two zeros of one sign, the same infinity, both undefined, or a range event's true value):
  `exp(sin(pi)) -> 1` holds (the left side is 1.0000000000000002), `cos(pi) -> -1` and
  `floor(tanh(100)) -> 0` too, while `sin(-0.0) -> 0` does not (the zero's sign). In
  `permissive` it holds where the table's two values agree at float resolution, each read to as
  many digits as it needs: `log(-1) -> 0` is refused, the left side having no value. Each mode
  reads its own value of a fully numeric expression, so here permissive does not include
  `f64`: `0*tan(asin(1)) -> 0` holds in `f64` (float64's 1.633e16 times 0) and not in
  `permissive` (tan(pi/2) has no value), and its tier is `reject`, since `f64`'s fold writes
  every fully numeric part as the cheapest value it admits and no file needs such a rule.
- **`L(C) -> C` without variables** holds in every mode: the witness is the left side's own
  value wherever it is finite (`acos(C) -> C`).

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
