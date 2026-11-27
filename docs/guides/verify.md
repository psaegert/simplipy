# Verifying rulesets

`simplipy.verify` is the **rule judge**: it decides whether a rule holds in a mode,
and the miner's own certification is a pre-filter that may only refuse what the judge
refuses. Every value it compares comes from one table, defined once in the compiled
core: read in the reals, a literal is the rational it spells and a pole takes its limit
where every direction gives the same one (`log(0)` is `-inf`, `tan(pi/2)` has no value), and
a division by zero has no value however it is written (`1/0`, `pow(0, -2)`, `mod(1, 0)`), so
`x^-2` and `1/x^2` agree everywhere; read in `strict`, an operation is the engine's own float64
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
at pi/2, where the right side has no value). In `strict` the same cells are read in float64's
values, where a saturation plateau is a cell like any other (`tanh(x)` is 1.0 past 19.06), and
an extension at a double counts as measure zero only where its neighbouring doubles are no
extension (`x0/x0 -> 1` at 0). A rule without variables has one point and no tolerance; a `_`
slot is checked at `+inf`, `-inf` and NaN; each `<constant>` is its own symbol, and nothing is
owed at a constant where the left side has no finite value. Where it has one, a witness for the
right side's constants is owed, and a search that finds none refuses the rule (`NO-WITNESS`:
`floor(C*x0) -> C*floor(x0)`). A collapse, a right side that is its left side with
variable-free parts holding a constant replaced by constants, takes each part's own value as
its witness, and a constant at which a part has no finite value is no member of the family (a
fitted constant is finite): `tanh(x0 + log(C)) -> tanh(x0 + C)` holds, its left side at C = 0
(-1 everywhere) being no `tanh(x0 + D)` of a finite D. Where the right side does not read a
constant at all, every value is a witness, and the left constant's own one is taken.

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

Each mode is judged on its own values. `strict` holds a rule with variables where float64's two
sides agree (to 8 ULP, the true value accepted where an operation overflows or underflows), or
where the rule is true in the reals at every point where they give both sides a value and
float64's two sides agree in kind -- both finite -- at every such point the judge reads (there
their difference is rounding), and to 8 ULP at every other point: where the reals give one side
no value, float64 decides (`1/3^t -> (1/3)^t` holds at `_`, where at t = -inf the left side has
no real value and float64 gives +inf on both sides). An isolated pole's infinity is a value: the
judge solves for the zeros of a divisor and reads float64 at the doubles nearest them, where a
change of kind is refused (`1/sin(x0)^2 -> 1/sin(x0 + 2*pi)^2` is +inf against 1.7e31 at
x0 = 0, `1/(x0 - 3)^2 -> 0` +inf against 0 at x0 = 3). The sign of that infinity
is not promised: where a part is exactly zero at an
isolated point and an operation reads it as a pole, +inf and -inf agree, as the two zeros do, so
`-1/(x0 - 3.5) -> 1/(3.5 - x0)` (-inf against +inf at x0 = 3.5 alone) and
`1/(-x1 - 3) -> -1/(x1 + 3)` hold in `strict`, while `1/(0*x0) -> -inf/x0` (+inf against -inf for
every x0 > 0) does not. Where the reals give the left side no value at an isolated point, its
real limit decides what float64 owes there: with no limit (a two-signed pole), an answer may take
any value, so `1/(x0 - 3) + 1/(3 - x0) -> 0` holds in `strict` (+inf against 0 at x0 = 3); with an
infinity of one sign, an infinity of either sign or the left side's value, so `1/(x0 - 3)^2 -> 0`
is refused; with a finite limit, the left side's value or the limit, or NaN where the left side is
an infinity, so `exp((x0 - 1)/x0^2) -> exp(1/x0 - 1/x0^2)` is refused (0 against NaN at +0, the
limit 0). A point may be a run of doubles: where float64 rounds a part to an exact zero on
consecutive doubles around an isolated zero of the reals, the run is that zero's point; an
infinity's and a zero's sign are open on all of it, and elsewhere on it an answer agrees with
float64's left side or with the reals' value there, so `1/(-sin(x0) - 1) -> -1/(sin(x0) + 1)` holds
(float64's sin(x0) is exactly -1 on about 10^8 doubles around -pi/2), and so does
`1/(sin(x0) - 1) -> -(1 + sin(x0))/cos(x0)^2` (the reals' -5.3e32 where float64's left side is
+inf); a run where the reals have no zero is a region, so
`1/(tanh(x0) - 1) -> -1/(1 - tanh(x0))` is refused (float64's tanh(x0) is exactly 1 past 19.06).
The reals place a run's zero by a sign change across it, or by its least |value| falling with the
precision; a run they cannot place is no point. A kernel's NaN is
float64's value, not a range event (`besseli(1, x)` is NaN for 0 < |x| <= 2.2e-308). The
second reading admits the identities float64 only rounds, however far a jump or a pole
magnifies the rounding, and nothing false in the reals:
`sin(x0 + pi) -> -sin(x0)` (1.2e-16 against -0 at x0 = 0), `x0 + 1 - x0 -> 1`,
`floor(x0 + 1) -> floor(x0) + 1` (1 against 0 at x0 = -1e-17) and `floor(tanh(abs(x0))) -> 0`
hold in `strict`; `atanh(tanh(x0)) -> x0` (+inf against x0 past 19.06), `exp(log(x0)) -> x0` (NaN
against x0 for x0 < 0) and `tanh(1/(abs(x0) - x0)) -> tanh(-1/(x0 - abs(x0)))` (1 against -1 on
x0 > 0, where the reals give neither side a value) do not. In both readings `strict` reads a
zero's sign where it changes a value on a region: float64 may give the two sides zeros of
opposite sign, and the reading in kind may admit a difference at which one side's value turns on
a zero's sign (a divisor, `pow` or `rootn` at an odd negative exponent, `atan2`) and the other's
does not, only at isolated points. `0*asin(x0) -> 0*acos(x0)` (-0 against +0 for x0 < 0) and
`atan2(0*x0, -1) -> pi` (-pi against pi for x0 < 0) do not hold in `strict`; `log(exp(x0)) -> x0`
(+0 against -0 at x0 = -0 alone) does, and `log(exp(_0)) -> _0` does not, since a slot that
matches any subexpression can bind one that is -0 on a region. A zero's sign is `strict`'s alone: `permissive`, which reads one zero, includes `strict` with the two zeros one value, so those rules hold in `permissive`, with float64's evidence where `strict` so read holds them (`neg(acosh(erf(x0))) -> acosh(erf(x0))`). `permissive` is judged on the reals' values at float resolution.
The bucket names (`CERTIFIED`, `KILL`, `UNRESOLVED-COVERAGE`, ...) are the reals' outcome, a
diagnostic that decides no mode by itself; `strict`'s second reading reads the reals'
counterexamples at the points where they give both sides a value, from the precision ladder and,
where it runs out, at float resolution (`ceil(atan(atan(x0))) -> ceil(erf(x0))`, 2 against 1 for
x0 > 74.7, is `KILL` with clause `a-float-resolution`). A mode whose reading reaches no answer
refuses the rule, with uncertainty: a witness search that finds none where the mode's left side
is finite (`NO-WITNESS`), a token the judge cannot read (`UNSUPPORTED-SHAPE`; a variable is read
whatever it is called, `v1`, `x7` or `alpha`, as a data slot), a judgment past its
time (`JUDGE-TIMEOUT`), or a point the mode owes a reading at and leaves unread. The owed points
are the ones the judge takes because they are exceptional: every slot at each of its sort's
exact atoms (zero, the units, the infinities, pi/2 and the rest), at each generic or special
constant value the mode owes a witness at, and a ground rule's one point; the verdict names an
unread one (`unread_owed`). At a `!` slot's +-inf and a `$` slot's 0 and +-inf, the null set
the slot's certificate leaves it, a value the left side has is read, and its having none, or a
zero's sign, is not: `tan(atan(!0)) -> !0` (1.633123935319537e16 against +inf at +inf) is
refused in `strict` and holds at `?`, while `erf(!0) - 1 -> -erfc(!0)` (+0 against -0 at +inf) and
`!0 - !0 -> 0` hold. The grid, the rows of the shipped mines' matrix (one matrix, the
same for every caller, so a verdict is the rule's), the preimages of jumps, the slots'
magnitude ladder and the constants' span and saturation magnitudes (`|c|` from 0.001 to 0.1
and from 10 to 1e4, and 19 to 1e17) are samples: a point among them that cannot be read counts neither way. The
slots' magnitude ladder runs each slot from the smallest normal double to the largest, both signs;
float64's lane reads every rung, and past the grid's 1e4 the reals read the rungs as they read the
grid: `min(10, log(x0)) -> log(x0)`, false past e^10 = 22,026, is refused in both modes (10
against 11.5 at 1e5). A rule with
constants and no slots reads one point per constant value, and at least a quarter of them must
be read.
`strict` reads float64's values: `x^2 + 1 + 1e-3000 -> x^2 + 1` holds in
`strict`, where `1e-3000` is the double 0.0 and the two sides are bit-identical. `permissive`'s
own criterion, agreement at float resolution, needs no exact equality: where the reals' readings
run out of precision (`UNRESOLVED-COVERAGE`, the 6,050 digits `1e-3000` beside 1 demands), the
interval layer's enclosure of both sides at the point decides that every double either side can
round to is the same, so `x + 1e-3000 -> x` holds by it too. (At the atoms e and the multiples
of pi, which the enclosure rounds before `pow` reads them, the enclosure of `x^2 + 1` is wider
than that, so `permissive`'s own reading of the first rule leaves owed points unread;
`permissive` holds it through `strict`.) Running out of precision
never reads as equality: `((x + 1e-3000) - x) * 1e3000 -> 0`, whose left side is 1, encloses
unbounded and is refused. A measure the cells cannot decide is refused where it is owed: `strict`
owes it in float64's values, so an undecided float64 measure refuses `strict`;
`UNDECIDED-MEASURE` labels the reals' measure, which `permissive` does not owe. An undecided
check never reads as a pass.

`KILL` and `ENGINE-MISALIGN` are different, and this is what the modes are for.
Each rule gets a **tier**: the modes it holds in, from two questions. Does it hold in `strict`
(the deployed evaluator reproduces it, or it is true in the reals and float64 keeps every
value's kind)? Does it hold by `permissive`'s own criterion, the reals' values up to
float resolution, and agreement alone where the input has no value (a permissive rule may give
a value where its input has none, over real-valued bindings)? `permissive` binds every slot to
any subexpression, so its own question is asked at the `_` sort: no disagreement at an isolated
point is excused there. **Permissive includes strict**: a rule holds in `permissive` where it holds
in `strict` or by permissive's own criterion, and a step permissive takes by a strict rule promises
what `strict` promises (float64's values agree wherever the left side has one). So
`floor(tanh(40 + abs(x0))) -> 1`, float64's own value and off the reals, and
`ceil(abs(cos(x0))) -> 1`,
false at pi/2 in the reals and true at every double, hold in both modes; `ceil(x0) - floor(x0)
-> 1` holds in neither. A verdict carries each mode's reading: `realised` for `strict` (from its two
readings of float64, `realised_f6` and `realised_kinds`, and the reals' outcome) and `permissive`
for `permissive`, which includes `strict`'s: `permissive_own` is its own criterion's reading alone
(`2*tanh(x0)/(1 + tanh(x0)^2) -> 2/(tanh(x0) + 1/tanh(x0))` holds in `strict`, float64's `1/0` being
`inf`, and so in `permissive`, though its own criterion, which reads `1/0` as no value, refuses
it at x0 = 0). A rule that is an instance of the steps `permissive` takes inside its
constructors, its named simplifications, holds in `permissive` with the steps' names as its reason
(`permissive_by`): the core exports each step's schemas from the tables the constructors read
(`_core.permissive_schemas()`: the inverse pairs, the powers' composition, the like-term laws), the
judge rewrites both sides by them, and the rewritten rule (`permissive_by_rule`) is read by
permissive's own criterion. `asin(sin(x0)) -> x0`, false in the reals at x0 = 2 (pi - 2 against
2), holds so by `inverse pairs cancel`, and `(x0^2)^(1/4) -> x0^(1/2)` by `powers compose`; both
stay refused in `strict`. The tiers are one registry, `simplipy.verify.TIERS`, and this table is rendered
from it (`simplipy.verify.tier_table()`):

<!-- tier-table: begin (rendered by simplipy.verify.tier_table(); a test checks it) -->
| tier | `strict` | `permissive` | meaning |
|---|---|---|---|
| `core` | yes | yes | holds in strict (the deployed evaluator reproduces it, or it is true in the reals and float64 keeps every value's kind: `sin(x0 + pi) -> -sin(x0)`), and so in permissive, which includes every rule strict holds |
| `permissive` | no | yes | holds in permissive alone: true in the reals at float resolution and refused by strict (`atanh(tanh(x0)) -> x0`, where float64 gives +inf against x0), an extension (`exp(log(x0)) -> x0`), held by strict's readings with the two zeros one value and refused by strict for a zero's sign (`0*asin(x0) -> 0*acos(x0)`), or an instance of permissive's named steps (`asin(sin(x0)) -> x0`) |
| `reject` | no | no | holds in no mode; recorded in the drop census |
<!-- tier-table: end -->

`strict`'s first reading means the deployed evaluator computes the rewrite, to within 8 ULP, at
every battery and grid point, and at every slot magnitude from the smallest normal double to the
largest, where no operation of either side overflows or underflows; its second, that the rule is
true in the reals and both sides have one kind at all those points (`sinh(asinh(x0)) → x0` is 66
ULP off at 1e100 and holds by the second). At a point
where one does, the rewrite may give the true value instead, as the canonical constructors do,
so `rootn(exp(x0), 2) → exp(x0/2)` is `core` although `exp` overflows past 709.78. Where no
rung of the precision ladder reads the true values there, they are read at float resolution, as
`permissive` reads a point, and where that reads nothing and one side alone left the range, that
side's true value is read against the other side's double: `besselk(1, x0) - besselk(-1, x0) →
besselj(exp(10), x0)` is refused at x0 = 1e5, where the left side's Bessel functions underflow and
its value, 0 as computed and in truth, is not the right side's -0.0022. A pole
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
rule = [['atanh', 'tanh', 'x0'], ['x0']]     # true over R; float64 saturates to inf at large t

verify_ruleset([rule], mode='permissive')['is_clean']  # -> True   belongs there
verify_ruleset([rule], mode='strict')['is_clean']      # -> False  float64 contradicts it
```

The converse does not happen: permissive includes strict, so a rule that holds in `strict` holds in
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

With `engine_config=` it also checks that `permissive` is at least as capable as `strict` on
every expression of a corpus: `corpus_rows`, or by default the 400-expression corpus that
ships with the package (`simplipy.verify.default_corpus()`). The check runs on the config's
operators and the two rule sets passed in and asks, row by row, that permissive's answer cost
no more than `strict`'s answer as permissive reads it, both priced on one shared scale (`strict`'s
canon). `strict`'s answer is read by permissive's constructors rather than taken as `strict` spells
it: a product `strict` keeps for a premise of its own, such as a zero's sign it reads, is the
expanded polynomial permissive writes, which costs more and is no simplification permissive
misses. A permissive answer without a value (NaN: the expression has no value in the reals, such
as a division by zero however it is written) is not priced: `strict`'s answer then holds float64's
value, an infinity from the division, which permissive does not share. A row on which a mode
raises counts as a violation. `verify_triple`, the gate of the
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
  and the measure of clause (b) is decided by certified cells or by a structural certificate (in
  the reals, a left side that is a rational function of its atoms has no value only where an atom
  has none, which the right side must hold too, where an atom is infinite, or where a divisor is
  zero, a null set where the divisor's numerator is a nonzero polynomial of atoms that are
  analytic, nonconstant and read disjoint variables, or a nonconstant polynomial of entire atoms:
  `(x1*log(x3)/sin(x2) - x1)*sin(x2) -> x1*(log(x3) - sin(x2))`); no counterexample was found,
  which is no proof;
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
  false on an interval no sample point reaches, and is refused in every mode. In `strict` such a
  proof carries over only where every operation is exact on doubles (`neg`, `abs`, `max`,
  `min`). Rules of several variables, and functions past 256 breakpoints or 4,096-bit
  coefficients, are left to the sampled procedure.
- **A left side that is one literal everywhere.** Where the interval layer encloses the left
  side, over every finite binding of its variables and constants, by the right side's literal
  and nothing else, the rule is proved (`0*exp(x0) -> 0`).
- **A rule without variables or constants** has one point, read by the ground evaluator the
  engine's fold and `verify-mine` read too. In `strict` it holds where its two sides are the same
  double, or the reals give them one value and float64's two values agree in kind (both finite,
  two zeros of one sign, the same infinity, both undefined, or a range event's true value):
  `exp(sin(pi)) -> 1` holds (the left side is 1.0000000000000002), `cos(pi) -> -1` and
  `floor(tanh(100)) -> 0` too, while `sin(-0.0) -> 0` does not (the zero's sign). In
  `permissive` it holds where the table's two values agree at float resolution, each read to as
  many digits as it needs: `log(-1) -> 0` is refused, the left side having no value. Each mode
  reads its own value of a fully numeric expression, so here permissive does not include
  `strict`: `0*tan(asin(1)) -> 0` holds in `strict` (float64's 1.633e16 times 0) and not in
  `permissive` (tan(pi/2) has no value), and its tier is `reject`, since `strict`'s fold writes
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
licensed(['asin', '1'], ['acos', '0'], 'strict', engine)
# -> {'licensed': False, 'holds': True, 'tier': 'core', 'refused_by': ['ground_composite']}
```

## The monitor: `monitor_ruleset`

The gate's complement: instead of a per-rule scan, `monitor_ruleset` runs the
*deployed* engine over an adversarial-plus-sampled corpus and judges every
input→output rewrite as the judge judges a rule, in the mode it was made in,
attributing any violation to the rule set or, where the rules-empty engine
makes it too, to the engine. It has no evaluator of its own: every value it
compares is the core's. An answer is held per step: every rule step to the mode, and the
engine's own algebra to the reals with float64's kinds kept at the step's own points. Around the
smallest part of the input the answer changed that holds as a step, only a zero's sign on a
region is read, so `1/sin(np.pi - x0) -> 1/sin(x0)` passes (the step `sin(np.pi - x0) ->
sin(x0)` gives 0 against 1.2e-16 at x0 = np.pi, which `1/` makes +inf against 8.2e15) and
`atan2(0*x0, -1) -> pi` does not (-pi against pi for every x0 < 0). Where the answer respells a
product or a quotient, the factors it changed, less those the two sides share, are a part too,
read under the reciprocal where one divides: `x/(a*x^2 + b*x) -> 1/(a*x + b)` passes where
float64's divisor of the answer vanishes a double away from the input's, since
`(a*x^2 + b*x)/x -> a*x + b` holds as a step. A number the engine folds
from literals (a product's coefficient, a sum's constant term, a long literal written at its
double: `_core.ac_simplify_trace`'s folds) is a step too: each fold is judged as the ground rule
it makes, and where every fold holds in `strict`, the answer is read with each folded number as the
numbers it was made of, so `27.940055475362794*(x1 + 0.2856797429345068*x2) ->
27.940055475362794*x1 + 7.981907865777503*x2` passes although the reals refute the answer as
written by the fold's rounding. Both the gate and the monitor carry poison
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
