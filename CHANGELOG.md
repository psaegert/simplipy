# Changelog

## Unreleased (0.15.0)

0.15.0 changes what `simplify` returns and what `complexity` charges, in every mode, and a rule
file is minimal and correctly ordered only under the canonical forms it was mined with, so the
rule files are mined again for this release. The section starts with what to do when upgrading,
organised by who you are, then the highlights, then every change in detail.

### Upgrading from 0.14.x

**Everyone**

- **Answers and prices change.** The search owns the distributive law, poles are read by their
  limits, numbers stay exact beyond 128 bits, and `complexity` prices each mode in its own
  reading, a literal costing at most what its float costs. Regenerate every test, cache or
  dataset that pins answers or prices.
- **What a variable is called no longer matters.** Over names other than `x<k>` (`x_1`, `v1`,
  `alpha`) every certificate failed, so collections and orientations the certificates license
  did not happen there: `2*sin(1/x_1) - sin(1/x_1)` stayed as written, where
  `2*sin(1/x1) - sin(1/x1)` is `sin(1/x1)`. An expression now gets the answer it gets over
  `x<k>`, renamed.
- **The search runs until a round finds nothing.** `simplipy.DEFAULT_EFFORT` is `None` (it was
  4), so a second call has nothing left to continue. Code that compares `DEFAULT_EFFORT` with an
  int now sees `None`; pass `effort=4` for the old cap.
- **Loading does more up front.** Every rule set named at load is compiled before `load`
  returns, so the first call in each mode answers at once. With `acj-5-4-llm` the process peaks
  at about 0.9 GB with all three sets, about 0.7 GB with `modes=('f64', 'permissive')` and about
  0.45 GB with `f64` alone; a set left out builds on its mode's first use, in about 3–4 s and
  about 100 MB.
- **Messages go through `logging`.** The default-engine announcement and the first-use build of
  a deferred set are WARNING records on the `simplipy` logger, install progress is INFO and an
  install failure ERROR; the library itself prints nothing (`list_assets` excepted, whose output
  is its purpose). Silence them with `logging.getLogger('simplipy').setLevel(logging.ERROR)`.
- **An installed asset loads without the network.** `load` and `get_path` resolve an installed
  asset from the local copy of the manifest, verifying the files' digests, and go online only
  for an asset that is not installed.
- **`packaging` is a declared dependency** (it was already installed through `huggingface_hub`).

**`f64` and `real` users**

- **What `f64` promises.** `f64` promises the deployed evaluator's value wherever no operation
  overflows or underflows. Where one does, a rewrite may give the true value instead, as the
  constructors always have (`exp(x0)/exp(x0)` is `1`), and the rule judge now reads f64 rules by
  the same standard.
- **Factoring is a move of the search.** No rule states the distributive law, so factored and
  expanded answers change. `effort=0`, the chain alone, does not factor (in `real` it did,
  through four common-factor rules).
- **One orientation per class.** A sum and its negation, and odd functions and odd roots of
  them, are written in one orientation with the sign in the coefficient: `-1/(x1 - 3)` and
  `1/(3 - x1)` are one answer, and at a pole the answer keeps the limit.
- **Signed zeros.** `rootn(-0, 3)` is `-0` and `rootn(-0, -3)` is `-inf`, `(-0)^(1/4)` is `+0`,
  the scalar `abs(-0.0)` is `+0.0`, and an integer spelling of zero (`-0`) reads as `+0`. Where
  an exact zero is inverted, `f64` (and `permissive`) keep the sign float64 gives it:
  `1/(x1*(x2 - x2))` is `inf/x1`, which is `-inf` wherever `x1 < 0` (it was `inf`); `real` reads
  one zero and keeps `inf`.
- **`f64` writes the negative zero.** A spelling of zero that float64 reads as -0 (`-0.0`; the
  integer `-0` is Python's +0) keeps its sign in `f64` and is printed `-0.0`, priced as the zero
  and one sign bit; `real` and `permissive` read it as 0. An `f64` ground fold writes its float
  exactly, -0 included, and the floats of pi and e as `np.pi` and `np.e`: `1/(-0.0)` is `-inf`
  (it was `inf`), `atan2(0, -0.0)` is `np.pi` (it was `0`), `x + -0.0` is `x`. A written
  product or negation of numbers that is zero takes float64's sign: `-1*0`, `0*(-2)` and `-(0)`
  are `-0.0` (they were `0`), `-1*-0.0` is `0`; a written difference subtracts by adding the
  negation, as float64 does, so `-0.0 - 0` and `0*(-2) - 0` are `-0.0` and `1/(0*(-2) - 0)` is
  `-inf` (it was `inf`), while `-0.0 + 0` is `0`. A written product of a zero with ground
  factors takes the sign of their float64 values (`0*sin(-1)` and `sinh(-0.0)*0` are `-0.0`,
  `ceil(-0.0)*(-0)` is `0`), and a sign moved into an odd function of the zero keeps it:
  `-sin(0)`, `-2*atan(0)` and `-erf(0)` are `-0.0` (they were `0`). In a ground expression the
  reciprocal of `-inf` is float64's -0 too: `1/float("-inf")`, `1/(-float("inf"))` and
  `(-inf)^-3` are `-0.0` (they were `0`), so `1/(1/float("-inf"))` is `-inf` (it was `inf`);
  `(-inf)^-2` is `0`. A zero the search derives keeps the one zero (`x1*0` and
  `x1/float("-inf")` are `0`). A nonzero literal that underflows to -0 stays the leaf it is
  written as.
- **Known `f64` limit: a zero term at -0.** `x + 0` is `x`, and `x1 + sin(0)` is `x1`: at the
  single point `x = -0` the answer is -0 where float64's sum is +0 (`-0 + 0 = +0`). A point is
  a null set, whose zero sign `f64` does not keep. A term that is -0 on a whole region keeps the
  zero: `ceil(x1) + 0` and `2*ceil(x1) - ceil(x1)` are `ceil(x1) + 0`, since `ceil(x1)` is -0 on
  all of (-1, 0). `x + -0.0` is `x` exactly, since float64's `x + -0` is `x` at every `x`, -0
  included.
- **Long exact coefficients.** An answer carries every digit of an exact number
  (`2.566969966535569956e-38*x1`), and the search accepts expansions whose coefficients are
  long, because a literal costs at most what its float costs. Show answers to a reader with
  `engine.prettify(expr)`, which prints shortest floats.
- **Float notation.** Numbers print positionally from 0.0001 up to 10^16 and as
  `<mantissa>e<exponent>` outside that range where that is shorter (`1e-5*x1`, was
  `0.00001*x1`).
- **Throughput.** The uncapped search does more work on long inputs. Where time matters, bound a
  call with `effort=k` or `effort=0` (the chain alone).
- **New errors.** `simplify(max_passes=0)` raises `ValueError` (it ran one pass; one pass
  reaches the fixpoint, so pass 1). `complexity(..., canon=)` and
  `evaluate_constants(..., form=)` raise `TypeError` (see Removed).
  An unknown `mode` raises `ValueError` in `complexity` as in `simplify` (it raised `KeyError`).
  A malformed numeral token (`1/0`, `--5`, `2x`) raises `ValueError` at every boundary. In
  infix text, two operands separated only by whitespace raise `ValueError` (`x0 x1`: write the
  operator), and a call of an undeclared function is refused by name (`unknown function 'sqrt'`:
  write `rootn(x, 2)`).

**`permissive` users**

- **One walk.** `permissive` runs its chain and then the search, like every mode, so its answer
  never prices above the input and a second call returns it unchanged. Its answers change
  accordingly.
- **Literals are rounded once**, to their float, where they land (`3*(1/3 + 1e-20)` is `1`), and
  land on whichever of the two floats around them prints shorter.
- **The ground fold is a step of the chain**, kept only where it is cheaper (`asin(1e-8)` is
  `1e-8`).
- **The sign of an infinity is kept**: distributing over an infinity needs certified
  finiteness, as in `f64` and `real` (`1 + tanh(inf*(x1 - 1/2))` is no longer `0`).
- **The lean profile.** A worker that needs inference and corpus canonicalisation only loads
  `modes=('f64', 'permissive')`, about 0.7 GB with `acj-5-4-llm`; `real` then builds on its
  first use.

**Publishers of rule files**

- A config may carry `requires_simplipy`, a PEP 440 specifier; a simplipy version outside it
  refuses the artifact. The miner records the specifier for its own minor series
  (`>=0.15,<0.16`).
- Each rule file has its own provenance sidecar, and loading compares each sidecar present
  probe by probe.
- No environment variable changes what the core mines, certifies or serves:
  `simplipy.engine.ARTIFACT_ENV_SWITCHES` is empty, and a sidecar records every `SIMPLIPY_*`
  variable set during the mine.
- The load refuses a rule that states the distributive law (a rule one factoring turns from its
  left side into its right; drop reason `engine-owned-law`). A file mined before 0.15 loses such
  rows in `real` and `permissive`.
- Check a freshly mined file with `simplipy verify-mine <file> --mode <m> -e <engine>`.
- `verify_triple(..., engine_config=)` checks the three rule sets it is given (the config
  contributes its operators only) and sweeps `simplipy.verify.default_corpus()`, which ships in
  the wheel, when `corpus_rows` is not given.
- <!-- PENDING the re-mine: name the new artifacts and the pinned revision here. -->The rule
  files are mined again for 0.15 and published under new names; `simplipy.DEFAULT_ENGINE` and
  `simplipy.DEFAULT_ENGINE_REVISION` name the artifact and revision this version loads by
  default.

**Users of the miner API**

- `find_rules` mines `rules_f64.json`, `rules_real.json` and `rules_permissive.json` each in its
  own mode, and writes a checkpoint after every completed length, which a rerun of the same mine
  resumes from; the rerun takes a mode whose file is already final as it is.
- New keywords: `find_rules(interval_node_budget=)` (it replaces `SIMPLIPY_IVL_NODE_BUDGET`),
  `find_rules(judge=, judge_processes=)` and `find_rules(offer_final_rules=)`. `save_every=`
  (CLI `-s/--save-every`) is removed: the rule file is written once, when the mode's mine ends
  (see Removed).
- `prune_covered_rules`, `utils.deduplicate_rules` and `promotion.promote` take a required
  keyword `rule_mode` (`'f64'` or its alias `'default'`, `'real'`, `'permissive'`), and
  deduplication keeps the target lower in that mode's serve ordering, not the shorter one.
- The sidecar records `gates`, `per_length`, `acceptance` and `code_digest`; the `triple` record,
  `soundness.kill_switches` and `soundness.node_budget_env` are gone.
- The mode-less entries of the compiled core are gone: every `_core` entry that reads a mode takes
  it explicitly (see Removed).
- `N_PTS` and `THRESH` are no longer read; `utils.remove_pow1` and
  `explicit_constant_placeholders(convert_numbers_to_constant=)` are removed.

### Highlights

- **`prettify()`** displays an answer for a reader, with shortest floats and readable fractions.
- **Exact numbers beyond 128 bits.** A literal is an exact rational of up to 1,100 bits on each
  side, printed in Python's float notation.
- **Poles are read by their limits**, and a sum and its negation are one class in every mode.
- **The search owns the distributive law**, in every mode alike, with whole-state acceptance.
- **One complexity measure per mode**, with a literal costing at most what its float costs.
- **`permissive` is one walk**, like every mode: never above its input, and idempotent.
- **Rule files per mode.** Each file is mined in its own mode, `verify-mine` checks it, and an
  artifact names the simplipy versions that serve it.
- **Infix text** reads implicit multiplication, and whitespace separates tokens.

### In detail

The measurements below use three sets: the 129,490 simplify inputs of flash-ansr's
training-data generator (the training-data inputs), srbf's 6,531 ground truths and its 125,127
model predictions.

- **`prettify()` writes an expression for a reader.** `simplify` keeps every number exact;
  `engine.prettify(expr, digits=3)` is the display of that canonical form, as infix text or
  explicit or tagged tokens. The budget counts significant digits, never an exponent's
  (`2.5e307` counts 2), and each number is written in the first form that fits it: exactly, as
  `simplify` writes it (`0.3*x1`, `x1/4.2`, `22*x1/7`, `x2 + 1/2`); as a short exact sum, its
  first digits and the rest to its value (`x1 + 1 + 3e-21` for `x1 + 1.000000000000000000003`,
  `(2.5e30 + 1e15)*x1`, `(1e20 + 1)*x1`, `(0.3 + 4e-17)*x1` for `0.30000000000000004*x1`), for
  a number float64 does not write exactly in 15 digits (`1905` and `1.0001` stay as they are);
  or as its float64 in shortest form (`9.765625e-4*x1`, `2.1458314277221002*x1` for a 33-digit
  literal), with `k` significant digits at `precision=k` (`2.15*x1`), written as that number is
  so that the display reads back as itself (and no short sum: `0.043*x1` at `precision=3` for
  `0.043000000000000003*x1`, `(0.043 + 3e-18)*x1` in full). A number shown as a float is read
  as a float.
  A multiple of the literal `3.141592653589793` is written with pi factored out as the symbol
  `pi` where the rest fits (`1/(2*pi)`; `np.pi` in the token forms, the float the literal is),
  at every precision and never shortened, and with the literal where the symbol would read back
  as another expression (`sin(x1 + 3.141592653589793)`) or the expression holds the symbol
  itself; even roots read `rootn(x, n)`, and every divisor of a
  product joins one denominator, functions and sums included (`x1/(x2*log(x3))`,
  `x1/(sin(x3)*(x2 + 1))`). A function of numbers that would all be written as floats is one
  float, computed as the evaluator computes it and rounded to the precision
  (`1.8488479557720672e-83*x1` for `x1*exp(-381/2)`), and never 0, inf or a subnormal; a function
  of a number written exactly stays (`sin(2)`), and so does one whose float would let reading the
  display distribute it over a sum or collect a like term (`x1 + atan(1.2345678)*(x2 + x3)`,
  `x1 + x2 + x1/log(0.7377233226141932)`). Numbers a product keeps apart for their size are one
  float, the product of the floats they are written as (`x1 + 0.11030062109689767*x2` in `real`
  for a 330-digit literal times `0.893435038202104*x2`). It changes how a number is written, never the
  structure: a number whose spelling would let parts collect or cancel, drop a multiplier or an
  exponent of 1, or make a non-integer exponent an integer is written as that spelling and the
  rest to its value (`(1 + 3.1e-21)*x1` for `1.0000000000000000000031*x1`,
  `sin((1 + 1e-4)*x1) - sin(x1)` for `sin(1.0001*x1) - sin(x1)` at `precision=3`).
  At the default `precision=None` a number is written exactly, as a short exact sum or as its
  float64, so the display reads back to the same float64 values; at every precision it
  reads back to the structure, and
  displaying a display gives it back. `keep_grouping=True` (off by default) writes every sum and
  product with the grouping of SimpliPy's explicit tokens, the order in which it evaluates the
  canonical form (`2*x2 + (2*x3 + exp(x2 + x3)) - x1`, `2*(x1*x2)/(x3*(x4*x5))`), so a display
  read back evaluates every sum and product in that order, where by default reading groups a
  flat sum from the left and terms that nearly cancel give another float64 value. Nothing inside
  SimpliPy calls it; see the guide "Displaying expressions".
<!-- TO BE MEASURED AFTER THE RE-MINE: the speed of 0.15.0 against 0.14.7, per mode, on the
     simplify inputs of flash-ansr's training-data generator, srbf's ground truths and its model
     predictions. The entry that stood here ("Fast in every mode") no longer holds: on the
     400-row corpus f64 takes 26% more time than 0.14.7, real 14% more, permissive 52% less. -->
- **Faster canonical forms.** Placing the signs of a nested sum asked the same flip four times per
  nesting level, so an expression nested `d` deep cost `4^d` flips: even with the faster products
  below, `x1*(x1 + (N + 1)/x1)` nested 10 deep took 23.5 s in `f64` with the search off, and a
  73-token nested input 25 s in `permissive` (0.14.7: over 4 minutes at 8 deep). The flip is now
  computed once per call (0.006 s and 0.11 s), and the matcher no longer copies the expressions it
  compares. Answers are byte-identical.
- **A pole is read by its limits.** Where the two sides of a pole disagree (`1/(x1 - 3)` at
  `x1 = 3`), the value there is not part of the function; where they agree, the answer has that
  limit. So the factors over one sum (the sum, its negation and its constant multiples, and odd
  functions, odd roots and odd powers of them, read through products) form one class, written in
  one orientation with the sign in the coefficient, in every mode: `-1/(x1 - 3)` and `1/(3 - x1)`
  are one answer, `1/(3 - x1)`, and so are `-atan(1/(x1 - 3))` and `atan(1/(3 - x1))` (0.14.7
  gives two answers for each pair). `1/((x1 - 3)*(3 - x1))` is `-1/(x1 - 3)^2`, which is `-inf`
  at `x1 = 3` like its limit (0.14.7 and the input give `+inf` there), and
  `1/(-(x1 - 3)*(2*x1 - 6))` keeps `-inf` (0.14.7: `1/(x1 - 3)/(6 - 2*x1)`, `+inf`). A sum that
  holds the class's sum as an exact polynomial factor is read in the class too, odd or even in it
  by the multiplicity of that factor, where the class's sum is linear in one of its variables:
  `9 - x1^2` is `-(x1 - 3)*(x1 + 3)`, so `1/((9 - x1^2)*(x1 - 3))`, `1/((x1^2 - 9)*(3 - x1))` and
  `-1/((x1^2 - 9)*(x1 - 3))` are one answer, `-1/((x1 - 3)*(x1^2 - 9))`, `-inf` at `x1 = 3` like
  their limit (the first two inputs give `+inf` there, and so did their answer). Only a factor
  proven by exact polynomial division counts, and its cofactor's sign at the zero is read where
  it is a constant there (`x1 + 3` is 6 at `x1 = 3`). A class
  trades its sign only as a whole, so no sign moves through a pole whose two sides agree. A member
  defined on one side of the zero only (a log, an even root, a fractional or symbolic power) fixes
  the class's orientation, positive on that side, so the value at the zero is that side's limit:
  `log(x1 - 3)/(3 - x1)`, `-log(x1 - 3)/(x1 - 3)` and `log(x1 - 3)*(3 - x1)/(x1 - 3)^2` are one
  answer, `-log(x1 - 3)/(x1 - 3)`, `+inf` at `x1 = 3` like the limit from `x1 > 3`, and
  `(3 - x1)^(-1/2)/(x1 - 3)` and `-(3 - x1)^(-1/2)/(3 - x1)` are `-1/(3 - x1)^(3/2)`, `-inf` there
  like the limit from `x1 < 3` (0.14.7 answers the first `1/(x1 - 3)/(3 - x1)^(1/2)`, `+inf`
  there). `permissive` keeps no value at a pole in general and gives its cheapest orientation.
  Like terms collect up to
  sign and a pattern variable that occurs twice matches a sum or its negation:
  `x1/(x2 - x3) + x4/(x3 - x2)` is `(x1 - x4)/(x2 - x3)` and `x1 + sin(x3 - x2) - sin(x3 - x2)`
  is `x1` (0.14.7 keeps the first as written and gives `x1 + sin(x2 - x3) + sin(x3 - x2)` for
  the second). An odd function or odd root of a literal multiple and of its negation are one class
  as well, a bare argument and a negated one included: `sin(2*x1)/sin(-2*x1)` and
  `sin(x1)/sin(-x1)` are `-1`, and `sin(2*x1)*sin(-2*x1)` and `sin(2)*sin(-2)` are
  `-sin(2*x1)^2` and `-sin(2)^2`, the answers of `-sin(2*x1)^2` and `-sin(2)^2`; an even power over
  a negative literal files the positive one (`sin(-2*x1)^2` is `sin(2*x1)^2`). One reading of a
  class serves the product, the like terms and the search's shared factors, so
  `x3*sin(x1) - sin(x1)` is `sin(x1)*(x3 - 1)` as `x3*sin(x1 - x2) + sin(x2 - x1)` is
  `sin(x1 - x2)*(x3 - 1)`, and `<constant>*sin(x1) - sin(x1)` is `<constant>*sin(x1)`.
  An odd negative power distributes over a negative coefficient as over a positive
  one, since signed zeros carry the sign through: `1/(-3*x1^2)` and `-1/(3*x1^2)` are one
  answer, `-1/3/x1^2`, and `1/(-x1/3)` is `-3/x1`, all `-inf` at the zero like their inputs.
  Where two singularities meet, the float value at that point is the one the
  canonical spelling gives (most such points have no limit at all).
  Known limits: where a member's value at the zero is not read, the class takes its cheapest
  orientation, which can leave that value off the limit (`log(x2*(x1 - 3))/(3 - x1)`, a class
  scaled by a factor of unknown sign, is `+inf` at `x1 = 3, x2 = -1` where its limit from
  `x1 < 3` is `-inf`, and so can `1/((x2^2 - x3^2)*(x2 - x3))`, whose cofactor `x2 + x3` has no
  one sign at `x2 = x3`); a polynomial multiple of a class whose sum is linear in none of its
  variables (`x1^2 - 2`) is not read; one class split across the terms of a sum with a dominant
  even term can give NaN at its zero where the limit is infinite, as the input does at other such
  points: `x5/abs(x3 - x2) - 2*x1/tan(x2 - x3)` at `x2 = x3` has the limit `+inf` where
  `x5 > 2*abs(x1)`, and its answer `2*x1/tan(x3 - x2) + x5*abs(1/(x2 - x3))` is `+inf` there for
  `x1 > 0` and NaN for `x1 < 0`, the input the other way round; the sign of the odd term's
  infinity at the zero follows `x1`, so no spelling of that cost is right at every such point; and
  a sum that vanishes at a class's zero only through a term that vanishes faster there
  (`(3 - x1) + x2*exp(-1/(x1 - 3)^2)`) is read as a class of its own, so its sign can move alone
  and flip the infinity at that zero (also in 0.14.7).
- **Fewest signs, then outermost.** Where two spellings cost the same, the answer is the one with
  fewer signs, then the one whose signs stand further out, wherever such a tie is decided: the sign
  placement, the product table, a sum's orientation and the order the rules fire under (a mined
  rule whose two sides cost the same loads, and fires only where it descends). A sign is a negative coefficient (a negation is `-1`) or a negative literal argument; an
  exponent and an infinity carry none, and a sign inside a function's constant argument belongs to
  that constant. An odd function of a literal multiple, a bare or negated argument included, trades
  its sign wherever it stands, alone too. So `sin(-x1)` is `-sin(x1)`, `x1 + sin(-x1)` is
  `x1 - sin(x1)` and `-sin(x1 - x2)` is `sin(x2 - x1)`, while `-sin(2)` is `sin(-2)`, and
  `x2*sin(-x1)` is `-x2*sin(x1)`. On srbf's
  ground truths, every spelling that writes an odd function `f(u)` as `-f(-u)` gets the
  original's answer (2,660 such spellings of 1,780 ground truths, in every mode). Mined rules that
  only move a sign at an odd function have one canonical state on both sides and are refused at
  load as subsumed.
- **A sign costs one bit wherever it sits.** A negative number paid one bit for its sign (`-2`,
  `-3*x1`, the exponent of `1/x1^2`), while the coefficient -1 of a product beside other factors
  was free (`-x1*tanh(3)` cost what `x1*tanh(3)` does) and a negation standing alone paid a whole
  product node, three bits (`-x1`, the `-x2` of `x1 - x2`). Each now costs one bit: a negation is
  a sign, not a product, so `-x1` costs `x1` and one bit and `x1 - x2` one bit more than
  `x1 + x2`; `+1` is no sign, and a reciprocal is division, not a sign. Where a sign stands is
  then never a matter of price, and the rule above decides it: `-x1*tanh(3)` and `x1*tanh(-3)`
  cost the same, and the answer is `x1*tanh(-3)`. `exp(-x1)` is now two bits cheaper than
  `1/exp(x1)` (they cost the same before, and the reciprocal had fewer signs). Against the same
  code with the old prices, at the default effort: on srbf's ground truths 675 answers change in
  `f64` (143 cheaper, 12 dearer, the rest the same price), 703 in `real` (166, 13) and 905 in
  `permissive` (257, 49); on a tenth of the training-data inputs 2,540 in `f64` (666, 16), 2,562
  in `real` (670, 17) and 2,858 in `permissive` (908, 31); on a tenth of the model predictions
  2,111 in `f64` (166, 15), 2,113 in `real` (169, 16) and 2,720 in `permissive` (332, 33). No
  answer changes on a second call. The dearer answers are the search ending at another local
  minimum, or a cancellation the shipped rule files reach only through the old spelling
  (`exp(u)/exp(u)` cancels, `exp(u)*exp(-u)` does not), which the re-mine under these prices
  addresses. The staged `acj-4` asset's f64 file loses 2 rows to the load gate (their sides now
  climb: `-exp(-x)` is cheaper than `-1/exp(x)`). The permissive answer on the corpus row where it
  priced a bit above `f64`'s now matches `f64`'s price. The measure fingerprint gains three sign
  probes (`-2`, `neg x0`, `* neg x0 x1`); sidecars that predate them do not record them, so
  loading stays silent.
- **The engine folds the NaN facts the rule files carried.** A root whose index is a ground
  constant that is no integer is NaN at every radicand (`rootn(-5, asinh(tanh(4)))`,
  `rootn(x1, log(8))`), and a family of `<constant>` values that is NaN at every constant is NaN
  (`acosh(acos(cosh(c)))`; the value-set fold used to read such a family as a free constant,
  `acosh(cos(c)/5)` as `acosh(c)`, which is defined). Both now fold in every mode, the index
  decided by its rigorous
  interval: where that holds an integer the root stays (`rootn(x1, acos(cos(1)))`). The interval
  also reads a positive base under a non-integer exponent by its corner powers, so
  `acosh(1/pi^(1/6))` is NaN. The published `acj-5-4-llm` files hold 110,122 rules with a NaN
  answer in `f64` (110,135 in each other mode), 106,130 of them about roots. With every one of
  them removed, the engine alone answers 108,323 of them NaN in `f64` (108,365 in `real`,
  108,470 in `permissive`); the rest are roots whose index is exactly an integer that the float
  evaluator rounds off it, or families the interval cannot bound. Answers do not change on
  srbf's ground truths and on a tenth of the training-data inputs, in any mode, with or without
  those rules, nor on a tenth of the model predictions; without them the engine holds 215 MB
  after loading every mode, where the published files take 839 MB.
- **`f64` folds a ground constant as a step of the walk.** The constructor folded a function of
  one literal to its float, bottom-up, so an enclosing node never saw the expression it came from:
  `rootn(exp(19.11787581094798), 2)` was `rootn(200811218.03870487, 2)` (66,912 milli-bits). The
  fold is now the pass's step, as in `permissive`: tried after the rules at each node, enclosing
  nodes first, and kept only where it is cheaper, so that expression is `14170.787488305119`
  (58,354). The value folded is the deployed evaluator's, each literal read at the float nearest
  to it, and it now reaches every ground node, not only a function of one literal: `cos(pi)` is
  `-1` and `acos(0)/pi` is `1/2` by the walk, so an `f64` mine no longer needs them as rules, as a
  `permissive` mine did not. Every reading prices a ground constant as written now:
  `complexity(['asin', '1e-08'], mode='f64')` is 13,170 (was 5,170), and `simplify` still answers
  `1e-8`; the readings the mine's sources and the rule load use keep `cos(0)` too. The measure
  fingerprint gains the probe `atanh tanh x0`, which `f64` reads apart from the other modes. On
  srbf's ground truths 22 `f64` answers change, all cheaper; on a tenth of the training-data
  inputs 241 (164 cheaper, 6 dearer: a whole ground node folded where folding its parts first
  reached a cheaper exact form) and on a tenth of the predictions 51 (50 cheaper). `real` and
  `permissive` answers do not change, and none changes on a second call.
- **Permissive lands a literal on the shorter of its two floats.** Permissive promises float
  precision, not the nearest float, so a literal that lands as an operand moves to whichever of the
  two floats around its exact value prints shorter (the nearest on a tie): `4/3*3.141592653589793`
  is `4.188790204786391`, not the nearest float's `4.1887902047863905`. Against the nearest float,
  on srbf's ground truths 142 permissive answers change (103 cheaper, none dearer), on a tenth of
  the training-data inputs 724 (610 cheaper, 9 dearer) and on a tenth of the predictions 7,621
  (5,890 cheaper, 270 dearer); the dearer ones are literals combined from several landed literals,
  each landing moving less than an ulp, which can end farther from the single rounding. No answer
  changes on a second call, and `f64` and `real` do not change.
- **An overflow cuts no `f64` collapse band.** Where an operation overflows or underflows, an `f64`
  answer may hold the evaluator's value there or the true one, so the inverse pairs collapse past
  the overflow too: `log(exp(800))` is `800` and `asinh(sinh(x0))` is `x0` (0.14.7 kept both),
  and `log(exp(t))` and `acosh(cosh(t))` collapse for every `t` proven at least 1 in magnitude,
  `log(exp(cosh(x0)))` included. The bands that rounding sets stay: `log∘exp` and `acosh∘cosh`
  keep their floor of 1 (`log(exp(t))` is 19 ulps from `t` at `t = 0.0137`, where nothing
  overflows), `sinh(asinh(t))` keeps its ceiling of 710 (past 8 ulps from about `t = 4.4e6`,
  where nothing overflows either), and `atanh(tanh(t))` keeps its ceiling of 2 (saturation is not
  overflow). On a tenth of the training-data inputs 2 `f64` answers change, each now collapsing an
  `asinh(sinh(.))` (16,000 and 19,000 milli-bits cheaper); srbf's ground truths and predictions do
  not change, `real` and `permissive` do not change, and no answer changes on a second call.
- **Powers of e collect by their exponents.** `exp` is a power of e, so the product constructor
  merges `exp(a)*exp(b)` into `exp(a + b)` in every mode, as it merges `x^a*x^b` into
  `x^(a + b)`: `e*exp(x1)` is `exp(x1 + 1)`, `exp(x1)/exp(x2)` is `exp(x1 - x2)`, and
  `exp(u)*exp(-u)` is `1` wherever `u` is finite, as `exp(u)/exp(u)` is. Since a sign costs one
  bit, `exp(-u)` is the cheaper spelling of `1/exp(u)`, and nothing cancelled it against
  `exp(u)`. The law is exact on the extended reals; under the f64 evaluator the two sides differ
  where an `exp` overflows or underflows (`exp(800)*exp(-800)` is `inf*0 = nan`, `exp(0)` is
  1), where `f64` accepts the true value, and by the rounding of the summed exponent
  (`exp(x1)*exp(1.7)` against `exp(x1 + 1.7)`, measured on 40,002 points per band: at most 8 ulps
  for `|x1| <= 12.3`, 26 up to 100, 152 up to 300 and 410 up to 700). A rule stating the law, in
  either direction, has one
  canonical state on both sides and the load refuses it as subsumed. The published
  `acj-5-4-llm` files serve 16 rows fewer in `f64` (2 subsumed, 14 now duplicating the left side
  of another row), 57 in `real` and 55 in `permissive` (all subsumed). Ground truth 2255 rests at
  77,000 milli-bits (94,000 before) and
  1299 at 64,000 (75,000). Against the previous build, at the default effort: on srbf's ground
  truths 38 answers change in `f64` (19 cheaper, 1 dearer, 18 rereading to the same state), 38
  in `real` (19, 1) and 69 in `permissive` (49, 1); on a tenth of the training-data inputs 3, 4
  and 3, none cheaper or dearer; on a tenth of the predictions 14, 14 and 13 (`permissive`: 2
  cheaper, 2 dearer; the other modes 1 dearer). The dearer answers are the search ending
  elsewhere: `2*exp(x1)*exp(exp(x1)) + exp(exp(x1))` was `exp(exp(x1))*(2*exp(x1) + 1)`, and the
  merged `exp(x1 + exp(x1))` no longer shows the factor `exp(exp(x1))` that FACTOR takes out.
  No answer changes on a second call.
- **A `<constant>` absorbs the orientation sign of every class beside it.** A `<constant>`
  absorbs every constant factor of its product (`<constant>*5` is `<constant>`), and a class's
  orientation sign is the constant factor -1, so beside a bare `<constant>` the orientation of
  every class is free and the sign key's is taken (fewest signs, then outermost; a tie by the
  class's own sum). A sum holding `<constant>`s of its own is such a class too: `-(x1 + x2^c)/c`,
  `c*-(x1 + x2^c)` and `c*(-x1 - x2^c)` are all `c*(x1 + x2^c)` (the first and the last kept
  `c*(-x1 - x2^c)`), and `c*sin(c - x1)` is `c*sin(x1 + c)`. The skeletons of a model's
  training data change accordingly: on the release corpus (400 skeletons) 18 change, from the
  explicit and the tagged spelling alike, 12 of them cheaper and 6 at an equal price, and the
  two spellings now give one skeleton for every row (one row gave two). srbf's ground truths,
  predictions and the training-data inputs hold no `<constant>`, and none of their answers
  changes in any mode.
- **A zero keeps its IEEE sign where it is inverted, in `f64` and `permissive`.** Their evaluator
  is float64, whose zero carries the sign of what formed it: `x1*(x2 - x2)` is `-0` wherever
  `x1 < 0`, and its inverse `-inf`. The inverse of a product that holds an exact zero now
  distributes over its members, the zero giving `+inf` and each other member its own power, which
  is exact in IEEE arithmetic for every member, zeros, infinities and NaN included:
  `1/(x1*(x2 - x2))` is `inf/x1` (it was `inf`), `1/-(x1 - x1)` is `-inf`, `1/(0/tanh(-2))` and
  `1/(0/(-2))` are `-inf`, and `permissive` keeps `S` in `x/(0/S)` (`inf*x*S`). `real` reasons
  over the reals, whose one zero inverts to `+inf`, and keeps its answers (`inf`, `inf`,
  `inf*x`). The canonical form holds one exact zero, so a sign a derived zero takes through an
  odd function or a sum is not kept: `1/sin(0*x1)` and `1/(0*x1 + 0*x2)` are `+inf`, where
  float64 gives `-inf` for `x1 < 0`, resp. `x1, x2 < 0` (a written signed zero keeps float64's
  sign, see "`f64` writes the negative zero"). On a tenth of the training-data inputs 85 `f64` answers change (14
  cheaper, 64 dearer, 7 at an equal price) and 81 `permissive` ones (14, 61, 6); evaluated beside
  their inputs on 256 points, the `f64` answers that give an infinity of the other sign fall from
  50 rows to 1, and the `permissive` ones from 48 to 13. srbf's ground truths and predictions do
  not change in any mode, `real` does not change, and no answer changes on a second call. The
  rule judge's deployed lane already reads float64's signed zeros; the end-to-end monitor
  (`monitor_ruleset`, `judge_pair(..., mode=)`) now reads a zero in the lane of the mode a rewrite
  was made in, the signed zeros of `f64` and `permissive` or the one zero of `real`. On the
  release corpus `permissive` no longer prices at or below `real` on one row, whose `real` answer
  takes the one zero's `+inf` where float64 gives `-inf`.
- **A class takes one orientation wherever it stands.** An odd power or odd root of a sum, and an
  odd function of one, can be written over the sum or over its negation (`S^3` or `-(-S)^3`).
  Inside a product the sign placement chooses (the cheaper, then fewest signs, then the outer
  ones); a site standing alone kept the orientation it was written in. It now takes the one it
  takes as a product's one factor, so `S^3` and `1*S^3` are one answer, and so are `1/S^3` and
  `1/(1*S^3)`: srbf ground truth 6397, `1/(x3*(x3 - x1) - x2)^3`, is
  `-1/(x2 + x3*(x1 - x3))^3`, as it was inside a product (the same price, two signs either way,
  the second's further out), and `sin(x1 - x2 - x3)` is `-sin(x2 + x3 - x1)`. A power over a
  negated site reads its class, `(-F)^s` being the flip of `F` to the `s`, so `1/S^3` and
  `(S^3)^(1/2)` are read over one class whichever orientation `S^3` stands in, and a factor of a
  product is still oriented by its product. On srbf's ground truths 3 answers change in `f64`, 3
  in `real` and 4 in `permissive`, on a tenth of the training-data inputs 10 in each mode, and on
  a tenth of the predictions 2, 2 and 12, every one at an equal price and none changing on a
  second call; respelling each odd function of srbf's 1,780 ground truths that hold one as
  `-f(-u)` gives the original answer in every mode. Where the sum under such a site vanishes on a
  region and is inverted, `f64` keeps the sign of its zero (see "a sum that vanishes on a region
  keeps its zero's sign", below). The decision costs about 5% of `permissive`'s time on a tenth of
  the predictions.
- **The certificates read a variable whatever it is called.** The interval layer that certifies
  finiteness, a null zero set and the other a.e. properties took a variable to be a dummy, `x<k>`
  or `_<k>`; any other name (`x_1`, `v1`, `alpha`) was a token it could not read, and every
  certificate over it failed closed. So over such names the collections and sign placements the
  certificates license did not happen: `2*sin(1/x_1) - sin(1/x_1)` stayed as written where
  `2*sin(1/x1) - sin(1/x1)` is `sin(1/x1)`, a sum such as `x_1 - sin(x_2) - 3` kept the orientation
  it was written in, and `interval_finite_ae(['sin', 'x_1'])` was `False`. A variable is now what
  the engine's vocabulary says it is (no operator, no literal, no `<constant>`, no placeholder),
  and the layer reads every other name as a dummy of its own, past every dummy the expression
  holds and never on the axis a free `<constant>` shares, so an expression over `x_1, x_2` is
  analysed as it is over `x1, x2`. srbf's ground truths and the training-data inputs, spelled
  over `x<k>`, do not change in any mode. srbf's model predictions are spelled over `v<k>` and
  `x_<k>`: of all 125,127, 44,103 `f64` answers change (15,024 cheaper, 1,520 dearer, 27,559 at
  an equal price), 44,061 in `real` (14,968, 1,525, 27,568) and 5,387 in `permissive` (1, 1,
  5,385). Every changed answer checked (each dearer one and 2,000 random rows per mode) is the
  answer of the same expression spelled over `x<k>`, renamed, and no answer changes on a second
  call. Evaluated beside their inputs on 256 points, 2 `f64` and 9 `real` answers no longer show
  the input's infinities and NaNs where the old answers did, each where a power or an exact
  literal leaves float64's range (`v1^158`, a coefficient of `5.3e320`).
- **A zero set is read on the expression's domain.** The certificate of a null zero set that
  reading A, the distribution licences and the orientation of a pole read applied the identity
  theorem only to expressions analytic on all of R^n, so a sum over `log`, a root, a fractional
  power, `asin`, `acos`, `atanh`, `acosh`, a quotient or `abs` was never shown null: under an
  inverse `log(x2) - x2` kept the sign it was written with, `-1/(x1 - log(x1))` stayed as written
  in `f64` and `real` where `permissive` wrote `1/(log(x1) - x1)`, and the inverse of
  `x3*(log(x2) - x2)` stayed joined. A zero is a value, so only where an expression is defined
  can it have one, and a NaN region holds no zero. The certificate now reads the zero set on the
  domain: every operator is analytic where its argument is positive, in (-1, 1), above 1 or
  nonzero; where each such argument is affine in the variables (or its condition holds over the
  whole space) the domain is a union of convex pieces bounded by hyperplanes, and the zero set is
  null when interval evaluation finds in every piece a cell on which the expression is certainly
  nonzero. `log(x2) - x2`, `sqrt(x2) - x2`, `log(x3) - log(x2)`, `asin(x1) - x1`, `acosh(x1) - x1`
  and `sin(x3)/x1 - sqrt(x2)` are now null; `x1 - |x1|`, `asin(sin(x1)) - x1` and
  `log(exp(x1)) - x1`, which vanish on a region of their domain, are not, nor is a sum over `tan`
  (its poles cut the line into infinitely many pieces). The plain certificate behind the
  `$`-cancel keeps its strength. On srbf's ground truths 321 `f64` answers change (14 cheaper, 1
  dearer, 306 at an equal price), 323 in `real` (15, 4, 304) and 38 in `permissive` (all at an
  equal price); on a tenth of the training-data inputs 458 (23, 3, 432), 461 (23, 4, 434) and
  156; on a tenth of the predictions 256 (29, 12, 215), 255 (31, 11, 213) and 35. No answer
  changes on a second call. The dearer answers are inverses the stronger licence now distributes
  over a product (`1/(a*b)` as `1/a * 1/b`), which can price above the joined product, and the
  searches that start from those forms. One `real` training-data answer no longer shows the
  input's float64 infinity: a denominator `x2*tanh(u) - x2`, which float64 rounds to exactly zero
  where `tanh` saturates to 1 and the real `tanh` never does (`f64` reads float64's `tanh`, below).
- **A zero set is read on a slice.** The certificate of a null zero set that reading A, the
  distribution licences and the orientation of a pole read applied the identity theorem on the
  expression's domain only where every restricted argument is affine. So a sum over
  `log(x1*x2)`, `sqrt(x1 - x2^2)`, `log(log(x1))` or a root of `cos(x1)` was never shown null;
  nor was a sum whose terms share a factor that moves too fast for interval evaluation
  (`cos(500*x1*x2)*(2 - x3)` written out), nor an entire function whose literals put the search
  horizon past its largest box (`0.41*x1^5 + 182*x1^4 + ... + 1.9e9`). The certificate now also
  reads an expression one variable at a time. Where, for almost every value of the other
  variables, it is constant on no interval of its domain in that variable (its strictly
  monotone parts add up in one direction, or its derivative's zero set is null), it vanishes at
  countably many points of each line, and its zero set is null. A sum is also read through a
  factor every term shares (`g*a + g*b` vanishes only where `g` or `a + b` does), and an entire
  function's witness is searched in the largest box. `log(x1*x2) - x1`, `sqrt(x1 - x2^2) - x3`,
  `log(log(x1)) - x2`, `x1 + log(cos(x1))` and `x1 - log(sqrt(x1))` are now null. `x1 - |x1|` and
  `x1*|1/x1| + 1`, which vanish on a region, are not, nor is `x1 + 1/log(x1)`, whose derivative
  needs a second derivative to read. The plain certificate behind the `$`-cancel keeps its
  strength. On srbf's ground truths 25 `f64` answers change (1 cheaper, 24 at an equal price), 52
  in `real` (3 cheaper, 1 dearer, 48 at an equal price) and 10 in `permissive` (all at an equal
  price). On a tenth of the training-data inputs 470 (21, 4, 445), 906 (40, 10, 856) and 246
  change; on a tenth of the predictions 501 (66, 19, 416), 744 (87, 32, 625) and 131. No answer
  changes on a second call, and none loses the input's infinities and NaNs on 256 points. The
  dearer answers are inverses the stronger licence distributes over a product, and the searches
  that start from them. The time per call does not change.
- **`f64` reads `tanh` and `atan` on their float64 range.** float64's `tanh(u)` is exactly ±1 for
  every |u| ≥ 18.99 and its `atan(u)` exactly ±fl(π/2) for every |u| ≥ 5.8e15, in the deployed
  evaluator as in the engine's. No other operator is constant on a tail: measured on each out to
  the largest float, the others keep changing, overflow or underflow (which the overflow reading
  covers), or are NaN off their domains. So in `f64` a sum that cancels such a limit is `+0` on a
  whole region: `x2*tanh(x1) - x2` vanishes wherever x1 ≥ 18.99. The zero-set certificate the core
  reads took the real functions, which never reach their limits, and let a sign move through such
  a sum under an inverse: `-1/(x2*tanh(x1) - x2)` became `1/((1 - tanh(x1))*x2)`, `+inf` where
  float64 gives `-inf`. In `f64` the certificate now reads the expression as the real one off the
  tails and, on each tail, as the expression with the function replaced by its limit, whose zero
  set must be null too. `-1/(x2*tanh(x1) - x2)` stays as written, and so, under an inverse, do
  `tanh(x1 - x3) + 1`, `atan(x1) - fl(π/2)` and `tanh(tanh(x1)) - tanh(1)`. `x2*tanh(x1) - x1` and
  `atan(x1) - x1`, whose tails do not cancel, are null as before. `real` and `permissive` read the
  real functions. srbf's ground truths do not change in any mode. On a tenth of the
  training-data inputs 2 `f64` answers change (1 cheaper, 1 at an equal price), and the
  `x2*tanh(u) - x2` answer above keeps the input's infinity in `f64`. On a tenth of the
  predictions 5 change (3 dearer, 2 at an equal price). `real` and `permissive` do not change, and
  no answer changes on a second call. The dearer answers keep an inverse joined where the
  certificate cannot read a tail and so fails closed: the limit leaves a constant past its horizon
  (`atan(u)^64` is fl(π/2)^64 = 3.6e12 there), or it sits under `cos`, where `cos(fl(π/2))` is
  6e-17 but the literal's enclosure holds π/2.
- **The `!` and `$` certificates read float64's tails in `f64` too.** The certificates behind the
  cancellations are "defined and finite almost everywhere" (`!`: `2t - t` as `t`, `0*t` as `0`)
  and "defined, finite and nonzero almost everywhere" (`$`: `t/t` as `1`, `t^2/t` as `t`). They
  are consulted on the core's terms and on a rule's pieces. They took the real `tanh` and `atan`,
  so in `f64` `(tanh(x1) - 1)/(tanh(x1) - 1)` became `1` and `2/(tanh(x1) - 1) - 1/(tanh(x1) - 1)`
  became `1/(tanh(x1) - 1)`: a value where float64 gives NaN, wherever x1 ≥ 18.99. They now read
  an expression as the zero-set certificate does, as the real one off the tails and, on each tail,
  with the function replaced by its limit; each reading keeps its own certificate memo. `real` and
  `permissive` do not change. srbf's ground truths and a tenth of the training-data inputs do not
  change in any mode. On a tenth of the predictions one `f64` answer is 2.8% dearer: there a tail
  leaves a constant past the certificate's horizon (`480586*atan(v2)` is 754906 on it), and the
  certificate fails closed. No answer changes on a second call.
- **The symbolic-exponent merge reads float64's tails in `f64`.** `x^y * x^y` merges into
  `x^(2y)` where the level sets of `y` off the integers are null, which the certificate drew from
  `y` being a nonconstant analytic function. In float64 a `tanh` or `atan` exponent is constant on
  whole tails, so its level set there is the whole tail. No merge the certificate licensed was
  wrong: a `tanh` tail sits at an integer, harmless to the merge, and an `atan` tail at a float
  none of whose small multiples is an integer. But the claim it rested on was false. In `f64` the
  certificate now reads `y` as the zero-set certificate does and accepts a tail only where `y` is
  an integer there: `x2^tanh(x1) * x2^tanh(x1)` still merges, and
  `x2^(3*atan(x1)) * x2^(3*atan(x1))` stays as written. `real` and `permissive` do not change, and
  no answer of srbf's ground truths, a tenth of the training-data inputs or a tenth of the
  predictions changes in any mode.
- **In `f64` a sum that vanishes on a region keeps its zero's sign where it is inverted.** float64
  adds to `+0` wherever a sum cancels, however the sum is written: `|x1| - x1` is `+0` for every
  x1 ≥ 0, so `-(|x1| - x1)` is `-0` there and `1/-(|x1| - x1)` is `-inf`. The canonical form holds a
  sum and its negation as one sum in one orientation, and `f64` wrote that inverse as
  `1/(x1 - |x1|)`, `+inf` there. Where the zero-set certificate cannot show a sum's zero set null
  and the sum is inverted, `f64` now keeps the sign with the inverse: `1/-(|x1| - x1)` is
  `-1/(|x1| - x1)`. The same holds through an odd power, an odd function or an odd root of such a
  sum, and in a product holding one, whose inverse distributes over its members:
  `1/(x2*-(|x1| - x1))` is `-(1/(|x1| - x1))/x2`. Under an inverse the search no longer factors a
  content out of such a sum unless that content is certainly positive. A ground sum, and a sum
  holding a `<constant>` or an infinity, are read as before. `real` and `permissive` do not change.
  On srbf's ground truths 50 `f64` answers change (2 cheaper, 3 dearer, 45 at an equal price), on
  a tenth of the training-data inputs 596 (28, 32, 536) and on a tenth of the predictions 488 (70,
  63, 355), most of them at an equal price. No answer changes on a second call, and none loses the
  input's infinities and NaNs on 256 points.
  `prettify` builds its display with the same `f64` constructors, so a display can move a sign
  accordingly, also where the answer is unchanged: an answer that displayed as
  `-sin(x1 + pi/6 - x1^3)` now displays as `sin(x1^3 - x1 - pi/6)`.
  Respelling each odd function of srbf's 1,780 ground truths that hold one as `-f(-u)` still gives
  the original answer in every mode. Respelling each sum as its negation turned around still does
  in `real` and `permissive` (5,986 of 5,986), and does for 5,897 in `f64`. The other 89 (44 over
  `log`, 37 over a root, 8 others) hold a sum whose zero set the certificate cannot show null, so
  there the orientation written decides which infinity the answer gives. A sum turned around
  before it is inverted can still lose its zero's sign: `1/(-x1 - |x1|)` is `+inf` for every
  x1 ≤ 0, but the sum is held as `-(x1 + |x1|)` when it is built, and the answer
  `-1/(x1 + |x1|)` is `-inf` there. Keeping every sum the zero-set test cannot clear as written
  would reach it, but even with the slice reading ("a zero set is read on a slice", above) it
  makes 594 of the `f64` answers on a tenth of the predictions dearer, all but 3 over sums whose
  zero set is null and unread (univariate sums over `log`, roots, `abs` or `tan` that mix
  directions), so it is not done.
- **In `f64` a sum keeps a `+0` term where its other terms may all be -0 on a region.** float64's
  sum of zeros is -0 only where every term is -0 (`-0 + 0` is `+0`), and some terms are -0 on a
  whole region: `ceil(x1)` on (-1, 0), `-floor(x1)` on [0, 1), `min(x1, -0.0)` for x1 ≥ 0,
  `mod(floor(x1), -1)` everywhere, `x2*ceil(x1)` wherever x2 > 0 there, `sin`, `atan2` or an odd
  root of such a term. The sum constructor dropped a zero term (`ceil(x1) + 0` was `ceil(x1)`)
  and collected like terms of opposite signs (`2*ceil(x1) - ceil(x1)` was `ceil(x1)`), so
  `1/(ceil(x1) + 0)` and `1/(2*ceil(x1) - ceil(x1))` were `-inf` on (-1, 0), where float64 gives
  `+inf`. Now both are `ceil(x1) + 0` and their inverses `1/(ceil(x1) + 0)`, `+inf` there;
  `0 - floor(x1)` and `x2*ceil(x1) + 0` stay as written. A zero term goes, and a pair collects,
  where one remaining term is shown not to be -0 on a region, its zero set null or its zero `+0`:
  `x1 + 0` is `x1`, `0 - ceil(x1)` is `-ceil(x1)` (`+0` on (-1, 0)), `|ceil(x1)| + 0` is
  `|ceil(x1)|`, `ceil(|x1|) + 0` is `ceil(|x1|)`. Where the certificate cannot show it, the zero
  term stays. Two terms of one class written in its two orientations no longer collect where
  their key may vanish on a region, since both orientations of `x1 - |x1|` are `+0` for x1 ≥ 0.
  The zero-set certificate reads `exp(g)` as vanishing where `g` is -inf (`exp(1/ceil(x1))` is 0
  on (-1, 0), and `exp(-1/(x1 - |x1|))` for x1 ≥ 0), and `ceil(g)` and `floor(g)` over a range that
  meets their zero interval only at `g = 0` as zero where `g` is (`ceil(|x1|)` vanishes only at
  x1 = 0). `real` and `permissive` do not change. On srbf's ground truths and a tenth of its
  predictions no answer changes in any mode; on a tenth of the training-data inputs 14 `f64`
  answers change, 3 cheaper and 11 dearer by a kept `+0` term where the certificate cannot show
  the remaining product's zero set null (`0 - x13*(x13 + 1/(2*log(x13)))` after an `x13 - x13`,
  where it was `-x13*(x13 + 1/(2*log(x13)))`). No answer changes on a second call, and none
  loses the input's infinities and NaNs on 256 points. A sum that cancels on a region
  (`ceil(x1) + ceil(x2)` is `1 - 1 = +0` on (0, 1) × (-2, -1)), or a product holding one, is
  still turned around when it is negated, as described above.
- **`rootn` keeps the sign of zero.** An odd root is odd at zero as everywhere else, as IEEE-754
  `rootn` is: `rootn(-0, 3)` is `-0` and `rootn(-0, -3)` is `-inf` (0.14.7: `+0` and `+inf`, while
  `rootn(-0, 1)`, `sinh(-0)` and `(-0)^3` were already `-0`), in numpy, on scalars, in the compiled
  evaluator and in the high-precision judge; `rootn` also takes torch tensors. A canonical form
  that moves a negation under an odd root therefore keeps the infinity beside an odd factor of the
  same zero: `1/(rootn((3 - x1)*x2, 3)*sinh(x2*(3 - x1)))`,
  `1/(rootn(x2*(3 - x1), 3)*(9 - x1^2))` and `1/(rootn((3 - x1)*x2, 3)*(3*x2 - x1*x2))` are
  `+inf` at `x1 = 3, x2 = 1`, like their inputs and their limits, in every mode (0.14.7 gives
  `-inf` for the first). The scalar `abs(-0.0)` is `+0.0`, as on
  arrays, so `1/abs(x)` is `+inf` at `-0.0` on every input path.
- **The search owns the distributive law, in every mode alike.** Factoring and distributing are
  moves of the search, accepted only where the whole expression ends cheaper, and no rule states
  them. FACTOR takes what some terms of a sum share out of exactly those terms, a bare term being
  its own product with 1, and a term holding a sum whose terms, or their negations, stand beside
  it factors through that sum; DISTRIBUTE also moves one factor into a sum factor, or the other
  factors over one term of it. So `x1*x2 + x1*x3 + x4` is `x4 + x1*(x2 + x3)`,
  `x2*x3 + x2 + x3 + 1` is `(x2 + 1)*(x3 + 1)`, `sin(x1)*x2 + sin(x1)*x3 + x4` is
  `x4 + sin(x1)*(x2 + x3)`, `x3 - x2 + (x2 - x3)*x4` is `(x2 - x3)*(x4 - 1)` and
  `(x1 + x2 + x3)/x2` is `(x1 + x3)/x2 + 1`, in `f64`, `real` and `permissive`, and
  `sin(x1)*sin(rootn(sin(x1), 2))/(x1*sin(x1) + sin(x1))` is `sin(rootn(sin(x1), 2))/(x1 + 1)`
  (0.14.7: `sin(x1)*sin(rootn(sin(x1), 2))/(sin(x1) + x1*sin(x1))`). On 400 random sums of 2 to
  5 monomials, which mean the same in every mode, the three modes give one answer for each. The
  rule load refuses a rule that one factoring turns from its left side into its right, judged by
  that law and not by spelling: on the `real` and `permissive` files it refuses four
  common-factor rules and `<constant>*x - x -> <constant>*x`. Factoring a sum under a division
  takes the licence distributing the division needs, so in `f64` and `real` a sum that vanishes
  on a region keeps its spelling. Against the factoring this replaces, at the default effort,
  answers are cheaper on 653 of srbf's 6,531 ground truths in `f64` and 576 in `real`, none
  costlier; on its 125,127 predictions 33,401 are cheaper and 76 costlier in `f64` (`real`:
  33,194 and 149), and on the 129,490 simplify inputs of flash-ansr's training-data generator
  21,403 and 25 (`real`: 19,047 and 78).
  <!-- TO BE RE-MEASURED: the permissive figures, measured against the removed three-arm
       selection. --> No answer changes on a second call in `f64` or `real`. The search does
  more work for it: in `f64` 56% more matcher steps and constructor calls on the ground truths,
  48% on the predictions and 138% on the training-data inputs (`real`: 8%, 15% and 73% more).
  `effort=0`, the chain alone, no longer factors in `real` and `permissive`.
- **A fractional power of `-0` is `+0`.** The evaluator computed `(-0)^(1/2)` with numpy's
  `sqrt(-0) = -0`, so `1/x^(1/2)` was `-inf` at `x = -0` and `+inf` at `x = 0`; it now follows
  IEEE `pow` (`+0`, and `+inf` for a negative exponent) on numpy arrays, torch tensors and plain
  numbers. Integer powers keep their sign (`(-0)^3 = -0`).
- **Answers read back as themselves, whatever the brackets.** Three canonical forms depended on
  the route that built them, so an answer could change on a second call: a product with more than
  six factors whose sign can move kept the orientation it arrived in (the cheapest orientation is
  now found exactly for any number of them, one closed form over whole classes with the values
  kept at their zeros: `(x2 - x1)*sin(x1 - x2)*(x3 - x4)*...*(x13 - x14)` and its mirror are one
  answer), two sums that are each other's negation stayed apart
  where they met in one product, and a term whose coefficient became 1 kept its factors' signs. A
  product is now built once from all of its factors, however brackets, `/`, `neg` or the tagged
  `<mul>`/`<div>` spell it, and each class of factors over a sum and its negation, or over an odd
  function or odd root of one (`tanh(x1 - x2)` and `tanh(x2 - x1)`), is decided once from its
  exponent totals, every placement priced as the whole product. Positive powers and even negative
  powers move between the two orientations (an odd amount moves a sign to the coefficient, or to a
  bare constant or infinity, which carries it); the fractional part of a power and symbolic powers
  stay; and negative odd powers move with their class as a whole (above). So
  `(x1 - x2)*(x2 - x1)` is `-(x1 - x2)^2`, `(x1 - x2)^2/(x2 - x1)/(x1 - x2)` is `-1`, and
  `tanh(x1 - x2)*(tanh(x1 - x2)*(-1))` is `-tanh(x1 - x2)^2`, as written flat (0.14.7:
  `(x1 - x2)*(x2 - x1)`, `(x1 - x2)/(x2 - x1)` and `tanh(x1 - x2)*tanh(x2 - x1)`). An odd function of a negative literal multiple has one spelling
  in a product: `x2*atan(-2*x1)` and `-x2*atan(2*x1)` reach one answer, so
  `x2*atan(-2*x1) + x2*atan(2*x1)` is `0` (0.14.7 keeps both terms). On 1,653 generated products
  over opposite sums, odd functions and odd roots, each in up to eleven spellings (four
  bracketings and the tagged bag for all; two division spellings, the inverse of a product, a
  chain of divisions and the tagged bag with `<div>` where the product has a divisor; a negation
  for 72), no product in `f64` and `real` reaches two answers (0.14.7: 616), and none changes on
  a second call at the same effort.
  <!-- TO BE RE-MEASURED: the 1,653 generated products in permissive (0.14.7: 687 reach two
       answers), measured before permissive became one walk. --> In `f64` and `real` no answer
  changes on a second call on any of the three sets (0.14.7: 59 and 57 of the training-data
  inputs, 4 ground truths, 31 predictions). The debug-build check that every intermediate state
  reads back as itself printed in a spelling no caller receives; it now prints as the callers do
  and passes on all ground truths, predictions and training-data inputs in every mode.
- **A fitted constant beside a sum keeps its family.** A product of a sum and a sum holding
  `<constant>` let the constant's sum take the product's sign as if the constant absorbed it:
  `(x2 - x1)*(x1 - <constant>)` became `(x1 + <constant>)*(x1 - x2)`, which no value of the
  constant makes equal to the input (also in 0.14.7). The sign now goes to the sum's other terms:
  `(x1 + <constant>)*(x2 - x1)`.
- **`pi^C` is a fitted constant.** `pow(np.pi, <constant>)` simplifies to `<constant>`, as
  `pow(2, <constant>)`, `pow(np.e, <constant>)` and `exp(<constant>)` do: every positive base
  gives the same family of values, which the fitted constant represents.
- **One f64 standard for the rule judge and the constructors.** The f64 constructors fill a NaN
  hole on a null set (`x1/x1` is `1`), and the rule judge now files a rule that does only that,
  at points the contract tolerates as null events, as realised in f64: `/ $0 $0 -> 1` is `core`
  (served in every mode), not `real`. A NaN hole on positive measure, or at a point the contract
  does not tolerate, still keeps a rule out of f64.
  Overflow and underflow are the same kind of special case: f64 promises the evaluator's value
  wherever nothing overflows or underflows. Where an operation overflows or underflows, the
  constructors give the true value (`exp(x1)/exp(x1)` is `1`, `rootn(exp(800.5), 2)` is
  `exp(1601/4)`), and the judge now accepts the true value at such a point as well as the
  evaluator's: `rootn(exp(x0), 2) -> exp(x0/2)` and `exp(x0)/exp(x0) -> 1` are `core`, not
  `real`. A pole is not an overflow (`atanh(1)` is infinite exactly) and an exact zero is not an
  underflow (`log(1)`). Every other point is compared as before, so a rule that gives neither
  value at a range event (`exp(1000) -> 5`) or disagrees where nothing leaves the range stays
  out of f64: `atanh(tanh(x0)) -> x0`, since `tanh` saturates inside the range, and
  `log(exp(x0)) -> x0`, 34 ULP apart at `x0 = 0.0137421`, where `exp` rounds its argument into
  the neighbourhood of 1. The verdict counts the points read this way (`out_of_range`). The
  miner's own numeric check in f64 applies the same standard: a row of its matrix where an
  operation of either side overflows or underflows is judged at high precision, where the
  two sides' true values must agree, and every other row is compared as before. Both read
  one definition of the event, the core's (`simplipy._core.range_event`), so an f64 mine
  mints `rootn(exp(x0), 2) -> exp(x0/2)`. `log(exp(x0)) -> x0` passes the numeric check,
  whose tolerance is wider than the 34 ULP, and the judge refuses it.
- **Permissive keeps the sign of an infinity.** Its search distributed a product over a sum under
  permissive's "finite almost everywhere" licence even when a factor was an infinity:
  `-inf*(x1 - 1/2)` became `-inf*x1 + inf`, undefined wherever `x1 > 0`, and then `inf`, wrong for
  `x1 > 1/2` (`1 + tanh(inf*(x1 - 1/2))` became `0`). Distributing over an infinity now needs
  certified finiteness, as in `f64` and `real`.
  <!-- TO BE RE-MEASURED: wrong answers on the 3,008 fuzzed expressions (389 -> 0 when it
       landed) and the changed training-data inputs whose answer holds an infinity, both
       measured against the removed three-arm selection. -->
- **The tagged form writes every fraction as a bag.** `simplify` answers in the notation it was
  given and tells the tagged one by its bag delimiters. An answer whose bags all collapsed carried
  none, and a one-token fraction in it (`-25/4`, `pow x10 17/4`) re-read as the explicit notation
  on a second call (`/ -25 4`). Every fraction is now `<mul> p <div> q </mul>` in the tagged form
  (or splits into the product it is a coefficient of); decimals keep their spelling (as in 0.14.7:
  a small one is a bag, `0.2` is `<mul> 1 <div> 5 </mul>`, the others one token). So a tagged
  answer that holds a fraction always carries a delimiter. A tagged answer without a bag is
  spelled as the explicit notation spells it, and `-1/x1` keeps its bag
  (`<mul> -1 <div> x1 </mul>`, like `-2/x1`; 0.14.7 answered `neg inv x1`, which re-read as
  `/ -1 x1`), so every tagged answer reads back as itself. Explicit input is unchanged,
  `exp 1/2` included.
- **The tagged form writes a reciprocal and a negative power as a quotient.** `simplify` answered
  `1/x0` as `inv x0` and `x0^-3` as `pow x0 -3` in the tagged form, while `to_tagged` writes their
  explicit spellings `inv x0` and `inv pow x0 3` as the bags `<mul> 1 <div> x0 </mul>` and
  `<mul> 1 <div> pow x0 3 </mul>`, so one value had two tagged spellings. The core now writes
  these bags wherever the quotient stands (`sin <mul> 1 <div> x0 </mul>`), the same member a
  product's `<div>` section holds, and a negated one keeps its coefficient
  (`exp <mul> -1 <div> x0 </mul>`, as `to_tagged` writes `exp / -1 x0`). Explicit and infix
  answers do not change. From tagged inputs (`to_tagged` of each row), the tagged answers that
  hold an `inv` fall to none (srbf's ground truths: 965 in `f64`, 964 in `real` and 1,205 in
  `permissive`; a tenth of the training-data inputs: 1,970, 1,983 and 2,314; a tenth of the
  predictions: 1,011, 1,012 and 4,967), every one respelled at its price, and the tagged answers
  that are `to_tagged` of the explicit answer rise from 4,317 to 5,307 of the 6,531 ground truths
  in `f64` (`real`: 4,296 to 5,291, `permissive`: 4,126 to 5,394). No answer changes on a second
  call, in either notation.
- **Permissive is one walk.** `simplify(.., mode='permissive')` ran three arms (its two fold
  disciplines and the `f64` chain) and chose between them by the price of internal states it did
  not return: 0.14.7 returns 769 of the training-data inputs, 3,326 of srbf's predictions and 12
  of its ground truths costlier than they came in, in permissive's own measure. It now runs one
  walk, like every mode: its chain, then the search, in its own canon and rules. Its constructors never fold a
  transcendental, so a constructor identity sees the shape it reads (`atanh(tanh(30))` is `30`); its
  ground fold is a step of the chain, tried after the rules at each node and kept only where it is
  cheaper (`asin(1e-8)` is `1e-8`, `rootn(exp(19.11787581094798), 2)` is `14170.787488305119`); its
  literals are exact and rounded once, where they land as an operand (`3*(1/3 + 1e-20)` is `1`, a
  64-digit coefficient becomes its 17-digit float); and its search joins reciprocal sums (`c/(A*B)`,
  held as `c * A^-1 * B^-1`, is also tried as `(A*B)^-1` and `(A/c)^-1` with the inside
  distributed). So, as in every mode, its answer never prices above the input or above its answer
  at a smaller `effort`, and a second call returns it unchanged: on srbf's 6,531 ground truths,
  a tenth of the training-data inputs (12,949) and of srbf's predictions (12,513) none prices above
  its input and none changes on a second call. Against the three-arm selection it replaces, in
  permissive's own measure, 323 ground truths get cheaper and 18 costlier, 790 training-data
  inputs and 54, and 5,368 predictions and 552; it spends 3.3, 2.6 and 2.9 times `f64`'s work
  units on these sets (the selection: 8.2, 5.9 and 10.6). Where `f64`'s answer prices below
  permissive's in permissive's reading (14, 49 and 399 rows), the causes are a literal rounded
  twice (rounded where it landed, then combined again, so one ulp off: on the predictions 454
  answers print such a literal one digit longer and 634 one digit shorter than the selection's
  single rounding), sign placement, `inf` beside a signed zero, and the search's walk order.
- **One complexity measure per mode.** `complexity(e, mode=m)` prices `e` as mode `m` reads it:
  the modes accept different simplifications as true (`f64` follows the float evaluator, so
  `atanh(tanh(x))` is not `x`; `real` follows exact arithmetic, so it is; `permissive` also what
  holds only up to sign or domain, so `(x^2)^(1/2)` is `x`, where `f64` and `real` give `abs(x)`),
  and each mode's `simplify` descends its own reading. So
  `complexity(simplify(e, mode=m), mode=m) <= complexity(e, mode=m)` in every mode (no answer
  prices above its input on srbf's ground truths, its predictions and the training-data inputs in
  `f64` and `real`, nor in `permissive` on the ground truths and a tenth of each other set), and
  prices of different modes are not comparable. Before, `complexity()` read every expression the `f64` way, under which a `real` or
  `permissive` answer could price above its input although its own mode made it cheaper. `f64` is
  unchanged.
- **The search runs until a round finds nothing.** `simplify()`'s default `effort` is now `None`
  (`simplipy.DEFAULT_EFFORT`; it was 4): the exploration phase stops when a whole round of
  expansions finds nothing cheaper, so a second call has nothing left to continue. A cap counts
  every candidate tried over the whole expression, refused ones included, so the cap a search
  needs grows with the expression (one srbf prediction needs 5 candidates, a sum of 2 copies of it
  10 and of 6 copies 30). Capped at 4 in this release, 1,632 of srbf's 125,127 model predictions
  change on a second call in `f64` and in `real`. Uncapped, none does in `f64` and `real` (with
  the canonical forms that read back as themselves, above). On the predictions no answer is
  costlier than at cap 4, and 1,633 are cheaper, in `f64` and in `real`. The search is one walk: it tries the current
  answer's candidates cheapest whole expression first, steps to the first that ends cheaper, and
  stops when a whole round finds nothing. Each candidate's whole state is rebuilt through the
  canonical constructors and is not re-canonicalized in full before its descent. On srbf's
  predictions the uncapped search spends 15% more matcher steps and constructor calls than
  `effort=4` in `f64` (16% in `real`). Every
  internal caller that passes no `effort` (normalization, masking, mining, the verification
  monitor, the promotion refund) follows the default. `effort=k` caps the search at `k`
  candidates (a cap beyond 2^63 - 1 is none), the first `k` of that one walk; `effort=0` never enters the search, so the search change
  leaves its answers byte-identical (the float cap below changes some: 269 of srbf's 125,127
  predictions in `f64`<!-- TO BE RE-MEASURED: the permissive count, measured against the
  removed three-arm selection -->; so do permissive's one walk, the canonical-form fixes above
  and, in `real` and `permissive`, the refused common-factor rules). Code that
  compares `DEFAULT_EFFORT` with an int now sees `None`.
- **Faster products.** Building a product decides where its sign goes by pricing every
  orientation of its sign-carrying factors. It now negates each such factor once instead of once
  per orientation (each negation builds products of its own, so the waste compounded with
  nesting), sums each orientation's price from per-factor prices, and remembers the prices of
  numbers beyond 128 bits. Answers are byte-identical on every simplify input of flash-ansr's
  training-data generator, srbf's ground truths and srbf's model predictions, in every mode. On
  srbf's predictions `simplify` takes 40-45% less time than without this in `f64`.
  <!-- TO BE RE-MEASURED: the permissive speed-up (about half the time, the slowest inputs about
       5 times faster), measured against the removed three-arm selection. -->
- **A literal costs at most what its float costs.** The evaluator reads every literal as its
  nearest float64, so the measure (`complexity`, and the ordering every simplification follows)
  caps a literal's price at the price of that float's shortest decimal (at most 17 significant
  digits), however a form holds the literal: `f64` and `permissive` admit a number only where its
  numerator and denominator are at most 2^1022 and hold any other value in pieces
  (`1.2345678901234567e-305` as written, its quotient by `9.876543210987654e-10` as that literal
  times a number), which cost at most what `real` prices the one number at (59,436 milli-bits
  for that quotient, its float's price, where its pieces cost 122,955; no answer on srbf's ground
  truths or a tenth of the training-data inputs changes with it); only a literal beyond the
  1,100-bit cap keeps its own price. A number carrying more digits than its float needs -- a product or power of
  long constants folded exactly, or a constant printed with 17 digits where 16 suffice -- now
  costs that decimal, not its exact digits; every literal that is its own float's shortest decimal
  keeps its price (every constant drawn for flash-ansr's training data among them), and no price
  rises. The cap is a decimal, never a fraction that reads as the same float: `0.3333333333333333`
  keeps its price, `1/3` costs less. Answers still carry and print the exact value. Because long
  exact numbers no longer count against a form, the search accepts more expansions in `f64` and
  `real`: on srbf's model predictions 21% of the `f64` answers change (26,440 of 125,127; 0.2% of
  the simplify inputs of flash-ansr's training-data generator), almost all (98%) to expanded forms (a product of sums
  multiplied out, a constant distributed over a sum) with about as many tokens (0.7% fewer) but
  long exact coefficients, so they print 25% longer (all answers 7%).
  <!-- TO BE RE-MEASURED: the permissive share of changed answers and their printed length
       (34% of the predictions, 7% shorter; 2.6% of the training-data inputs, 4% shorter),
       measured against the removed three-arm selection. --> The measure fingerprint gains two probes for the cap, and compound probes priced in each mode's own
  reading. Every rule file gets its own provenance sidecar, and loading compares each sidecar
  present probe by probe: it warns, naming the probes, when a recorded probe now prices
  differently, and a probe the sidecar does not record is not a mismatch. The published
  rulesets record no probe that moved, so they load without a warning.
- **Exact numbers beyond 128 bits.** A literal is an exact rational whose numerator and
  denominator have at most 1,100 bits each (every float64's shortest spelling fits); before, a
  number left the exact form at 128 bits and became an opaque symbol. Literal arithmetic now folds
  where it used to stay apart: `x1*1e-40*3` is one coefficient, `3e-40*x1 - x1*1e-40*3` is `0`,
  and the Larmor denominator `6*3.1415926535897*8.854e-12*(2.99792458e8)^3` is one number. A
  result beyond the cap keeps its pieces (`2^1100*x1` stays a power). Answers carry these numbers
  exactly, every significant digit included (`x1*1.602176634e-19*1.602176634e-19` is
  `2.566969966535569956e-38*x1`), in `f64` and `real`; `permissive` rounds a literal once, to its
  float, where it lands (`2.56696996653557e-38*x1`). A number prints positionally from 0.0001 up to 10^16 and as
  `<mantissa>e<exponent>` outside that range where that is shorter, Python's float notation
  applied to exact values, in the list, tagged and text answers alike: `x1*1e-300` comes back as
  written, `x1*1e-150*1e-150` as `1e-300*x1`, `x1*1e10*1e10` as `1e20*x1` and `x1*1e-5` as
  `1e-5*x1` (0.14.7: `1e-150^2*x1`, `100000000000000000000*x1` and `0.00001*x1`), while `1000`,
  `0.001` and `12345678901234567890` keep their digits. The measure prices such a number at most
  like the float it reads as (above).
- **f64 mode holds a number only where float64 can.** In `mode='f64'` (and `permissive`) a
  number's numerator and denominator must both be at most 2^1022, so the number and its reciprocal
  are normal float64 values; any other literal stays as written (`1e308`, `1e400`, the subnormal
  `5e-324`), as before. `real` mode holds everything within the cap: `x1 + 1e308 + 1e308` is
  `x1 + 2e308` there and `2*1e308 + x1` in f64 mode. Two more f64 readings follow the evaluator:
  an integer beyond 2^53 has unknown parity (`(-1)^9007199254740993` stays, and `x1^(2^53)*x1`
  is not merged: the evaluator reads 2^53 + 1 as 2^53), and a literal the evaluator reads as inf
  or as 0 is neither certainly finite nor certainly nonzero (`0*1e400` and `1e400/1e400` stay;
  the evaluator gives NaN). `real` keeps exact parity and finiteness.
- **Literal bags fold to a canonical partition.** The literals of a product or sum, the
  coefficients of one like term and the exponents of one like base fold pairwise until no two of
  them fold, so each such group re-reads to itself where a fold is refused. The old one-pass fold left
  pieces the next call merged (also in 0.14.7: `10/((250000000/1571)/(pi*pi*x3))` with pi written
  out needed two calls). `<constant>` and the infinities absorb the pieces a refused fold keeps,
  and a fraction beyond 128 bits stays together (`p/q` in the text answer,
  `<mul> p <div> q </mul>` in the tagged list answer, its own `/ p q` in the explicit list answer)
  instead of being split across a product (the evaluator multiplied a 300-digit numerator in
  before dividing and overflowed).
- **A term's coefficient is all of its literals.** Where a fold is refused (`1e300*1e300` has
  more than 1,100 bits, and is beyond float64 in `f64`), a product keeps both literals, and the
  coefficient of the term is now both of them. Like terms of one such coefficient collect and
  cancel (`1e300*1e300*x1 - 1e300*1e300*x1` is `0`, and `2*1e300*1e300*x1 - 1e300*1e300*x1` is
  `1e300*1e300*x1`; both stayed as written), a sum scales by it as by one number
  (`1e300*1e300*(x1 + x2) + x3` is `1e300*1e300*x1 + 1e300*1e300*x2 + x3`), and no literal of it
  is a factor it shares with a term of one literal. Only the first literal was the coefficient,
  the others part of the term, so which terms collected depended on which ones a construction
  saw together, and an answer could change on a second call (also in 0.14.7, at 128 bits):
  `-1e300*1e300*x1 - 1e300*1e300*x2` gave `-1e300*(1e300*x1 + 1e300*x2)`, whose second call
  gave `-1*1e300*1e300*(x1 + x2)`; the answer is now `-1e300*1e300*(x1 + x2)`. A negative such
  coefficient writes its sign on its smallest literal, as a single number carries its own sign,
  in every notation: `-1e300*1e300*x1` in the text answer, `* -1e300 * 1e300 x1` in the explicit
  list and `<mul> -1e300 1e300 x1 </mul>` in the tagged list (it was a separate `-1`, `neg` in
  the explicit list, priced as a sign either way). On 20,000 random sums of products of
  literals at the cap and float64 boundaries no answer changes on a second call in any mode
  with the search off (before: 80 in `f64`, 39 in `real`, 81 in `permissive`), nor on 6,000 at
  the default effort (before: 2, 4, 3). srbf's ground truths, a tenth of its predictions and a
  tenth of the T8.1 training inputs keep their answers in every mode, except one prediction in
  `real` whose terms change order at the same price, and none changes on a second call.
- **What the wider range moves.** Against the same code at 128 bits: none of the 400 corpus
  rows; 23 of srbf's 6,531 ground truths, all by literal arithmetic the 128-bit limit refused
  (two of them are coefficient sums that then allow a distribution), their values equal to 1e-15. srbf's judge sees a few more
  recoveries and loses none: in a sample of 6,000 result rows whose answers move, 6 verdicts go
  from not recovered to recovered, where a truth's or a prediction's coefficients now fold to the
  same form (`3.097*x2 + 0.966*14.68*x2` becomes one coefficient). The `real` and `permissive` rule
  sets of acj-5-4-llm serve 42 fewer rules (`(+-0.1)^n -> 1e-n` for n >= 40): the constructor now
  computes them, and the load drops a rule whose two sides the constructors already make equal. A mode's rules are
  read in that mode's number domain, whichever mode asks for them first. The verification
  instruments fold only within the engine's cap and refuse oversized spellings before building
  them.
- **The compiled evaluator reads every literal as its nearest float64.** A compiled expression
  is Python source, and Python read an integer literal as an exact `int`: numpy's ufuncs reject
  one above int64 (`sin(100000000000000000000)` raised TypeError, and the engine printed that
  spelling for `sin(1e20)`) and float arithmetic rejects one above float64's range
  (OverflowError). `as_code`/`as_callable` now spell an integer beyond 2^53 as its nearest
  float64 and one beyond the range as `inf`, and a fraction beyond the range as `inf`
  (`utils.evaluator_literal`). Integers up to 2^53 and fractions inside the range keep their
  values; a `+`, leading zeros and `-0` are respelled the way Python reads them (`007` was a
  SyntaxError). Arithmetic between two integer literals up to 2^53 is still Python's exact
  integer arithmetic, and arithmetic between literals beyond it is float64 arithmetic, as in
  the offline evaluator: `10^400 / 10^399` written as a division is NaN (inf/inf) and
  `9007199254740993 - 9007199254740992` is 0. A single fraction token `10^400/10^399` is 10.
- **Literal readers are exact at any size.** The offline evaluator reads a fraction token whose
  components exceed 128 bits as its correctly rounded float64 (`10^400/10^399` is 10, not NaN),
  and the interval kernel certifies such a fraction like a decimal and certifies an integer
  literal beyond 2^127 as a point when its float64 is exactly that integer (`2^200` written out;
  `1e40` still brackets, as its nearest float64 is not 10^40). The mining oracles read a fraction token instead
  of failing on it (`utils.literal_float`), and the contract judge refuses an oversized spelling
  (`1e999999999`, more than 4,300 digits) before building it and reads a value beyond float64's
  range as `inf` in its deployed lane (it skipped the deployed check for such a rule).
  Canonical forms move only where a literal is beyond i128, because the kernel now
  certifies such literals: a fraction token behaves like its decimal spelling. The contract
  judge kills none of the moved forms; the underflow folds among them (`log(-1/10^400)` is
  NaN) get the verdict the decimal `log(-1e-400)` already gets. Some moves undo a wrong fold:
  `pow(-2, (3*2^200)/3)` is (-2)^(2^200), which is finite, and used to fold to NaN.
- **Integer questions at any size.** The number type holds big integers on num-bigint (MIT OR
  Apache-2.0). Integer questions ("is this an integer, odd, negative?") no longer go through a
  128-bit value, which would read an integer beyond 128 bits as "not an integer" at four sites (a
  `rootn` index, the pow-of-pow and power-through-root licences, and the guard that keeps a
  negative coefficient from distributing over a factor that may vanish under a negative odd
  power).
- **Exact literals are read as their nearest float.** The deployed evaluator reads a literal
  correctly rounded (Python's `float()` of a decimal, `int / int` for a fraction), but two
  engine paths computed `float(p) / float(q)`, which is an ulp or two off once `p` or `q` leaves
  53 bits. The f64 fold of a ground function evaluated there, and the function amplified the
  slip: `sin(91618929858.17857)` folded to `-0.9807961394411915` instead of the evaluator's
  `-0.9807991153359534`. The fold now reads the nearest float, through the same system libm
  kernels as the evaluator (Rust's own `atanh` drifts from libm near the domain edges). The
  numeric leaf reader did the same for a one-token fraction, and the interval kernel bracketed
  that value by one ulp, so its enclosure could miss the fraction it encloses
  (`673107593011939307760027002528/810572757194796821120128085049` sat 1.5 ulps away). The leaf
  now reads the certified nearest float; a fraction with a component beyond 128 bits keeps the
  old value and a four-ulp bracket. The nearest float is certified by exact midpoint tests at
  every magnitude, tiny values included. And the interval kernel takes a point enclosure only
  for a literal that denotes an integer: a non-integer whose nearest float is an integer
  (`10182333997706870151/10000`, `2311319199101329.25`) used to become that point, missing its
  own value.
- **One numeral grammar, refused when malformed.** A numeral is a decimal
  `[+-]? digits [. digits] [e|E [+-] digits]` or an exact fraction `[+-]? digits / digits` with a
  nonzero denominator, and every boundary checks tokens against that one definition, so only
  numerals reach the number readers. The masking predicate
  `is_numeric_string` used to disagree with the readers both ways: `1E6`, `+5` and `2.5E-4` were
  read as numbers but never masked, while `e5`, `1e-5e` and `1/0` were masked as numbers no reader
  evaluates. Folding `neg` into a literal turns a leading `+` into `-` (`neg +5` is `-5`; it was
  `-+5`). A token that starts like a number but is not a numeral (`1/0`, `1/6.28`, `0.5/3`,
  `--5`, `-+5`, `1e5.5`, `1/-3`, `2x`) is now refused at every boundary, like the reserved
  numeric spellings (`inf`, `1_000`, `0x10`). Before, it became a free symbol (`1/0 - 1/0` simplified to `0`), or was
  read as something else (`-+5` as -5, `1/-3` as -1/3), and the tagged form, which skips the
  arity check, split `--5` into `-1 -5`. These refusals are for token lists, which are expected
  well formed; infix text is read as written (`1/6.28` is a division there).
- **Infix reads implicit multiplication.** After a number or a closing parenthesis, a name, an
  opening parenthesis or (after a parenthesis) a number multiplies: `2x0` is `2*x0`, `0x10` is
  `0*x10`, `2(x0 + 1)` is `2*(x0 + 1)`, `(a)(b)` is `(a)*(b)`, `2pi` is `2*pi`. The product has
  the precedence of `*`, so `1/2x0` is `(1/2)*x0` and `2^3x0` is `(2^3)*x0`. Before, these
  inputs were malformed. A name before a parenthesis stays a call, known or not
  (`read_infix('sqrt(x0)')` is still `sqrt x0`); a number before a name that starts with `_` is Python's digit grouping, not a
  product (`1_000` stays malformed); and a character the tokenizer drops never becomes a product
  (`x1 $ x2` stays malformed).
- **Infix: whitespace separates tokens.** Spaces used to be removed before reading, so
  `sin x0` became the name `sinx0`, `x0 x1` the name `x0x1` and `2 3` the number 23; a tab
  instead kept tokens apart. Now every whitespace character separates tokens, and a product
  forms across it as across touching tokens (`2 x0` is `2*x0`, `2\tx0` too).
  - A declared one-argument function without parentheses applies to the operand after it.
    Its argument takes powers and signs; products, quotients and sums apply to its result:
    `sin x0^2` is `sin(x0^2)`, `log x0 / 2` is `log(x0)/2`, `sin -x0` is `sin(-x0)`, and
    `sin cos x0` is `sin(cos(x0))`. A minus sign right after a function also starts its
    argument, so `sin-x0` is `sin(-x0)`; it used to read `-sin(x0)`. A name before a
    parenthesis stays a call (`sin (x0)^2` is `sin(x0)^2`).
  - Two joins remain, because the text can mean only one thing: `x0 * * 2` is `x0**2`, and
    an exponent part after a number joins it (`1 e-5` is `1e-5`). `1e` is no numeral, so
    `1e -5` is `1*e - 5`.
  - Any other two operands with only whitespace between them raise `ValueError`: `x0 x1`,
    `x 1`, `2 3`, `1 000`, `sin x0 x1`, and an undeclared function such as `sqrt x0`. So does
    an implicit product inside the argument of a function without parentheses, which is
    ambiguous: `sin 2x0` could be `sin(2*x0)` or `sin(2)*x0`. Every entry point that reads
    infix text raises (`read_infix`, `infix_to_prefix`, `simplify`, `to_prefix`, `to_infix`,
    `to_tagged`, `complexity`, `mask`), and `is_valid` returns `False`. Before,
    `to_prefix('x0 x1')` returned the name `x0x1`.

  No real string has whitespace in a position these rules change: none of 12.9 million
  strings (srbf's ground truths, every method's predictions and candidates, symbolic-data's
  catalogs and upstream formulas) reads differently. That shows ordinary spacing around
  operators is unaffected; the new rules are covered by the tests.
- **An interrupted mine is not final.** On SIGINT the miner stops after the running length and
  writes what it has, with the proposal channel, the symbolic gate and sort promotion skipped,
  and its sidecar said `"final": true`. It now says `"final": false` and `"interrupted": true`,
  and keeps the last completed source length.
- **In the infix answer, a fraction with no finite decimal can print as an integer over a
  decimal.** Every literal is its exact rational value, so `1/(2*3.141592653589793)` folds to
  500000000000000 / 3141592653589793 and printed as exactly that: correct, but nothing a reader
  recognises. A decimal is as exact as an integer, so the infix text now moves the numerator's
  factors 2 and 5 into the denominator, which then terminates, and prints
  `1/6.283185307179586` (Feynman's `3/(20*3.141592653589793)`: `3/62.83185307179586`). The spelling needs a
  numerator with a factor 2 or 5 and is used where it is strictly shorter than the two
  integers, outside exponents: `1/2`, `2/3`, `5/8` and `1/3` keep their fractions, an exponent
  keeps the fraction (`x0^(500000000000000/3141592653589793)`), and the first value that moves
  is 100/101 (`1/1.01`). The state, `complexity()`,
  the measure and every rewrite are unchanged, and the text re-parses to the same state. The
  token answers (explicit prefix and tagged) do not use this spelling: the engine
  prints states in the explicit form and reads them back (certificates, folds, served rules,
  the mining judge), and callers mask and compare token answers, so a spelling chosen for a
  reader stays out of them. Measured when it landed: the infix spelling changes no token answer,
  and 57 of the infix answers for srbf's 6,531 ground truths and the 400-row corpus change (the
  entries above change token answers for their own reasons).
- **An artifact names the versions that serve it.** A config may carry `requires_simplipy`, a
  PEP 440 specifier; a simplipy version outside it refuses to load the artifact with
  `IncompatibleArtifactError` and names the pin to install. The miner records the specifier for
  its own version's minor series (`>=0.15,<0.16`) in the provenance sidecar, and
  `simplipy.compat.requires_simplipy_specifier()` derives it. A pre-release counts as the release
  it precedes; a config without the key loads as before. `packaging` is now a declared
  dependency (it was already installed through `huggingface_hub`).

- **Each rule file is mined in its own mode.** `find_rules` mines `rules_f64.json`,
  `rules_real.json` and `rules_permissive.json` each in its own mode: the mode's reading of
  a source, its own answer as the mark to beat, its numerics, its serve ordering for the
  covered prune and the deduplication, and its licence for the judge's tier. The contract
  judge decides per length, inside the scan: a pick whose tier the mode does not license is
  excluded by class and its source scanned again, so a target another mode would keep never
  hides one this mode licenses, and a refused rule never prunes a longer source. Proposals
  run through the same loop. Rows the mode's load mints anyway are dropped per length, and
  each file passes a final gate (every file spelling and served entry licensed, no ground
  rule the mode's constructor performs) before it is written. In `permissive` the covered
  prune also asks the non-finite instances every slot binds there, so the file keeps
  `$0 / $0 → 1` (`inf/inf` is `1`, not `nan`).
- **The permissive mine is offered the f64 and real files.** Permissive promises less than
  either mode, so their rules hold there too, and its answer must never cost more than theirs;
  a per-mode mine finds only what its own scan reaches. The permissive file is mined last, and
  every rule of the finished f64 and real files runs through its proposal lane: certified by
  permissive's own acceptance, it enters only where permissive's answer is not already as
  low. The permissive sidecar pins both files (`offered`: name, sha256 and count, the lane's
  outcomes, and every offered rule that did not read `already_covered`), and `verify-mine`
  checks the pins (gate `offered`). `find_rules(offer_final_rules=False)` (config key
  `offer_final_rules`) declines the offer.
- **A mine resumes from its checkpoint.** `find_rules` writes `<file>.checkpoint.json` after
  every completed length and a rerun of the same mine resumes after the last one; a
  checkpoint written by another mine — other parameters, mode, universe, starting rules, or
  code (the compiled core or the package's sources) — refuses to start, and is never
  overwritten. The rule file and its sidecar are written once, when the mode's mine ends. A
  rerun of the same mine does not mine a finished file again: the sidecar records the mine's
  key and the file's sha256 (`resume`), and an unchanged file is taken as it is, its ladder
  snapshots included, so a crash in the permissive mine costs only the permissive mine.
- **`simplipy verify-mine <file> --mode m -e <engine>`** (and `simplipy.mining.verify_mine`)
  checks a mined file against the re-mine gates in its own mode: no twin-shadowed row, no
  engine-owned law, every spelling licensed, nothing preempted (no ground fact the rule-less
  engine folds, such as `log (-3) -> nan`), every rule live (`dead`, below), a translation
  that subsumes and drops nothing, a clean measure fingerprint and no `SIMPLIPY_*` switch; it
  exits non-zero when one fails. `-e` names the engine (an official name or a config path)
  whose operators read the file.
- **The sidecar records each file's own mine**: `acceptance` (tolerances, whether and at
  which gate a near miss is re-judged, the licensed tiers, the judge's digest), `per_length`
  (classes, picks, judge rounds, exclusions, twin-shadowed rows), `gates` and
  `code_digest`, beside the existing parameters and `requires_simplipy`.
  `find_rules(interval_node_budget=...)` (config key `interval_node_budget`) sets the
  interval layer's node budget for the mine and records it in the parameters.
- **`find_rules(judge=, judge_processes=)`.** The per-length judge loop asks `judge(pairs)` for
  the judged tier of each served pattern; the default (`None`) is the contract judge, run on
  `judge_processes` worker processes (default 4). Verdicts do not depend on the number of
  processes. `simplipy verify-mine` takes `--judge-processes`.
- **A constant collapse owes a witness only where its source is defined.** A `<constant>` is
  fitted, and a value that makes the source NaN or infinite is never the fit, so the judge
  licenses `acos(c) -> c_t` from |c| <= 1 alone. Sort promotion refused a collapse at the first
  probed constant where its source was NaN or infinite, so licensed rules never reached a file:
  `atanh(cos(c)) -> c_t` (infinite where cos(c) = +-1), `log(abs(c)) -> c_t`,
  `x*log(c) -> c_t*x` and `max(x, acos(c)) -> max(c_t, x)`, and with them `f64` left
  `atan(atanh(tanh(c)))` at `atan(atanh(c))`; it now reduces it to `c_t`. On the reduced
  length-4 mine the `f64` and `real` files gain 139 rules each and `permissive` 24. The judge,
  the sort ladder's ground tier and its witness bar now read one definition of "the source is
  defined" (finite, the rule the evidence gate counts rows by); a source defined at no constant
  (`log(-cosh(c))`) still gets no rule.
- **Sort promotion is deterministic.** A rule's sort depended on random draws from one
  stream shared by the whole pass, so on where the stream stood when the rule's turn came,
  and on whether a draw fell where float64 rounds too coarsely to decide: `atan(sinh(x)) ->
  asin(tanh(x))` passed the `!` bar on 4 of 12 seeds and shipped at `?` otherwise. Its exact
  arbiter re-judged every valuation, and the `1e20` probe needs 10^20 digits, so it never
  overturned anything. Every bar now judges fixed valuations (the atoms, a fixed set of generic
  points that includes float64's hard points: `tanh` within or at its rounding of 1, at
  `x = 17.5, 18.5, 19.5`, and `exp` either side of overflow, at `x = 709.5, 710.5`), and the
  arbiter decides exactly the finite valuations float64 left open, in the `_` and `!` bars
  alike. So `atan(sinh(x)) -> asin(tanh(x))` and `sin(acos(tanh(x))) -> 1/cosh(x)` ship at `_`
  in `real` and `permissive`, and a mine's rules promoted in any order get the same sorts (they
  got four different files in four orders).
- **A source's evidence is decided from its domain, not from a count of sampled rows.** A mine
  certifies a rewrite on the rows where the source is finite, so a source finite only on a null
  set (`asin(cosh(x))`, finite at 0 alone) gets no rule. Whether a source had evidence was a
  count of the rows of the mine's random matrix where it was finite, held against the floor
  (`min_informative`, an eighth of the rows), and a source finite on a narrow band fell on
  either side of it by seed: `atanh(floor(x))` is finite on [0, 1) alone, about 13% of the
  mixture, so `asinh(atanh(floor(x))) -> atanh(floor(x))` and three siblings were mined at seed
  42 (79 rows of 512) and not at seed 7 (63). A source now has evidence iff the interval layer
  certifies a cell of its box on which it is finite everywhere, in the mode's own semantics
  (below); an undecided search certifies nothing. Where the matrix holds fewer rows than the floor inside that set, the
  missing rows are drawn from the certified cells, by a stream the seed fixes, and judged with
  the rest. `abs(atanh(acos(x))) -> atanh(acos(x))`, finite for x in (cos 1, 1] (15 rows of
  512), is mined now.
- **A high-precision refutation needs converged values.** `real` and `permissive` re-judge an
  f64 near miss at 1024 and 2048 bits. A row whose values still moved between the two was taken
  as a refutation: `tanh(x)` is exactly -1 at both precisions once x < -710, so
  `sin(acos(tanh(x)))` read the residue of `sin(pi)` against `1/cosh(x)` ~ 1e-407 at x = -937,
  and `sin(acos(tanh(x))) -> 1/cosh(x)` was mined only when the matrices held no row below
  -700 (one variable at seed 42, none at seed 7). Such a row is now judged again at 4096 and
  8192 bits, the precision the mine's range (|x| <= 1e3) needs, by the same rule; a converged
  false row (`exp(sinh(-5)) -> 0`) is refuted at 2048 bits as before. With these two changes the
  reduced length-4 mines at seeds 42 and 7 write byte-identical files in all three modes, and
  the files gain 16 `f64` rules, 7 `real` and 6 `permissive` (`cos(asin(tanh(x))) ->
  1/cosh(x)` among them) and lose none that is not served by the rest.
- **A source without evidence is not scanned.** The evidence decision is the source's, the
  same for every candidate, but it was read at the end of each candidate's judgment, after its
  constant fits, domain gates and high-precision rescues, so a source finite nowhere
  (`log(-cosh(x))`) was judged against the whole library to be refused by every candidate. It
  is now read once, before the source's context is built, and such a source returns at once;
  the files are unchanged. On the reduced length-4 mine about 14% of the scanned sources have
  no evidence and held a quarter of the length-4 scan's time; the mine ran in 3232 s instead
  of 4258 s (4 threads).
- **A constant collapse is minted where the judge licenses it.** The judge licenses
  `acos(c) -> c_t` from |c| <= 1 alone (a fitted constant owes a witness only where its source
  is finite), but the mint kept a stricter rule of its own for a source without variables: its
  value class over the constant axis had to equal the target's, and a source finite on part of
  the axis and NaN on the rest is mixed, never the finite class of `<constant>`. So
  `atan(atanh(c)) -> c_t`, `acos(c) -> c_t` and `exp(log(c)) -> c_t` were never mined, three
  tokens or fewer though they are. A collapse to `<constant>` now asks the judge's question
  alone: the interval layer certifies a cell of constants on which the source is finite. A
  literal target keeps the class guard, a source finite only at the integers (`(-1)^c`) has no
  such cell, and one infinite on a band (`0^c`) is still out of a finite constant's reach. The
  engine's own fold reads the same criterion (below), so these collapses are the engine's.
- **The most specific rule fires first.** At a node the engine fired the first rule of the
  node's head that matched, in file order, so a rule mined for a longer source never fired
  where a shorter rule that also matches stood before it. The `real` and `permissive` files
  of the reduced length-4 mine hold `sin(acos(_0)) -> cos(asin(_0))` and, after it,
  `sin(acos(tanh(_0))) -> 1/cosh(_0)`: `sin(acos(tanh(x)))` took the first and stopped at
  `cos(asin(tanh(x)))` (28,000 milli-bits), which no rule continues. The rules of a head now
  fire most specific first, by the nodes a match must find beyond one per placeholder, exact
  rules before patterns and file order among equals, as the engines before 0.12 tried the
  longest pattern first; `sin(acos(tanh(x)))` is `1/cosh(x)` (17,000). No answer changes on
  the training-data inputs, the ground truths or the predictions (a tenth of each large set,
  all three modes), with `acj-5-4-llm` or with the reduced mine's files, and simplify takes
  the same time on them (within 0.3%).
  `cos(asin(tanh(x)))` itself is `1/cosh(x)` through the completion below.
- **`verify-mine` and the mine check that every rule fires** (`dead`). On every generic
  instance of a rule's left side (each placeholder a fresh variable, a wide one also a sum
  with a variable of its own) the engine holding the file must reach the rule's right side or something
  cheaper; a rule another rule takes from its own instance, ending dearer, ships and never
  decides an answer. The mine lists such rules in the sidecar (`gates.dead`) and on its
  final-gate line, and `verify-mine` fails on them. Under file-order matching the gate lists
  `sin(acos(tanh(_0))) -> 1/cosh(_0)` in the reduced mine's `real` and `permissive` files
  (the only rule it lists); it lists none now, and mined again the reduced mine writes the
  same three files.
- **Each mode reads a source's evidence in its own semantics.** `f64` certifies the cell where
  float64 finds the source finite, its exact tails included; `real` and `permissive` judge over
  the reals, and certify there. `acosh(erf(x))` is finite only where float64 rounds erf to 1
  (x >= 5.92) and NaN everywhere over the reals, so `real` and `permissive` no longer mine
  rules on it or on `acosh(tanh(x))`: `acosh(tanh(x)) -> acosh(erf(x))` and
  `atan(acosh(erf(x))) -> acosh(erf(x))` stay in `f64` alone.
- **The engine collapses a constant-only subtree wherever the mint would.** The engine's own
  fold collapsed a `<constant>`-bearing subtree only when its value class was finite, while
  the mint certifies the collapse by the judge's criterion: the fitted constant owes a value
  only where the source is finite. Both now read one function, `interval::constant_collapse`:
  a cell of the free constants on which the subtree is finite, and no infinity it provably
  reaches. So `log(c)`, `acos(c)`, `exp(log(c))` and `atan(atanh(c))` are `<constant>` in every
  mode, `x * acos(c)` is `c * x`, and the mine has none of them to write; `(-1)^c`, finite only
  at the integers, and `0^c`, infinite for c < 0, keep their form outside `permissive`. A
  special in the exponent is a fixed real like a literal, as it is in the base (`pi^c` was
  already `<constant>`): `c^pi`, `c^e` and `c^2.5`, finite for c >= 0, are `<constant>`.
  With `acj-5-4-llm` no answer changes on the srbf ground truths or on a tenth of its
  predictions or of the training-data inputs, in any mode; with `acj-4-3` the 400-row corpus
  walk rests 141,510 milli-bits cheaper on 11 rows, none dearer.
- **The covered prune and the liveness gate read every alphabet.** Their probe instance for a
  wide placeholder wrapped it in `sin(x)`, which an engine without `sin` cannot read; it is
  now `1/(x + x')`, with a variable of its own, which every engine reads (the core language)
  and every wide sort binds. It is no bag, so a narrower placeholder cannot take it apart: a
  bare sum `x + x'` would read `sin(_0 + 1) -> sin(_0)` as covered by `?0 + 1 -> ?0`, which
  rewrites `sin(x + x' + 1)` with `?0` bound to `x` alone.
- **The mine completes the overlaps of its rules.** Where two rules overlap at an instance
  (one's left side unifies with a subterm of the other's) and the two paths end in different
  states, the rule between the two, from the dearer state to the cheaper, is offered to the
  mine as a proposal and certified by its own scan, confirmation and judge, then pruned and
  promoted with the rest, up to three rounds. `sin(acos(_0)) -> cos(asin(_0))` and
  `sin(acos(tanh(_0))) -> 1/cosh(_0)` overlap on `sin(acos(tanh(x)))`, and the mine now writes
  `cos(asin(tanh(_0))) -> 1/cosh(_0)`; `acos(cos(asin(_0)))` and `cos(asin(sin(_0)))` overlap
  where the slot is `sin(x)`. Only completions within the mine's cell are offered, their
  source no longer than the mine's sources: beyond it a completion overlaps the rules again at
  a longer instance, and completion breeds without end (on the reduced length-4 mine's f64
  file, offering every overlap took 543 rules to 1,022 in one round, and the next round
  offered 13,772). The mine's final-gate line and sidecar (`gates.overlaps`) count the
  completions within the cell that the judge licenses and the file still lacks, list those
  it refuses (`gates.overlaps_refused`) with its tier and verdict, and count the overlaps
  beyond the cell (`gates.overlaps_beyond_cell`); `verify-mine` fails on the first
  (`overlaps`), reading the cell from the sidecar. The step prints its progress
  (`[complete]`, `[gate]`).
- **`simplipy find-rules --modes f64,real,permissive`** (and `find_rules(modes=...)`) mines
  only the listed modes, all three by default, and leaves the other files as they are; a mode
  mined beside an existing file reads it (`real` and `permissive` mine with the f64 file in
  the default slot, and `permissive` is offered the f64 and real files). Each mined file
  keeps its own sidecar and final-gate line.
- **`prettify(keep_grouping=True)` writes every number as the expression holds it.** Its
  promise is a display that evaluates as the expression does, but at full precision it wrote
  a quotient the explicit tokens divide by as the quotient's float, one rounding apart, and a
  function that amplifies its argument shows that rounding: two srbf ground truths,
  `sin(x1*x2/(6.626e-34/(2*3.1415926535897)))^2`, read back from
  `sin(9.482621954692726e33*(x1*x2))^2` with values up to 32 times off. At full precision the
  grouped display now writes each number exactly, as the tokens do,
  `sin(3.1415926535897e37*(x1*x2)/3313)^2`, and reads back bit for bit. The default display
  still writes the float.

### Removed

Each removal is a hard cut: the removed keyword or name raises instead of warning.

- **`find_rules(..., save_every=)` and the CLI's `find-rules -s/--save-every`.** The rule file
  and its sidecar are written once, when the mode's mine ends, and progress survives in the
  per-length checkpoint, so nothing reads the interval. Passing the keyword raises `TypeError`
  and the CLI refuses the flag; drop the argument.
- **`evaluate_constants(..., form=)`.** `evaluate_constants` answers in the input's dialect, as
  `simplify` does; passing the keyword raises `TypeError`. For another notation, convert the
  answer with `to_infix`, `to_tagged` or `to_prefix`.
- **`complexity(..., canon=)`.** `complexity()` prices each mode in its own reading, and the
  keyword that priced a mode's parse in `f64`'s canon is gone (passing it raises `TypeError`). To
  compare answers of different modes on one scale, use `simplipy.mining.cross_mode_complexity`.
- **`explicit_constant_placeholders(..., convert_numbers_to_constant=)`.** The helper renames
  placeholder slots and never numerals; which literals become fittable constants is decided
  upstream with `simplipy.masking`. Passing the keyword raises `TypeError`; drop it (every
  `convert_numbers_to_constant=False` call keeps its behaviour).
- **Eleven environment switches.** `SIMPLIPY_AC_ABSORB_FIRST`, `SIMPLIPY_MU_SYM`,
  `SIMPLIPY_MU_FREE`, `SIMPLIPY_IVL_GATE`, `SIMPLIPY_IVL_CLASS`, `SIMPLIPY_IVL_REACH`,
  `SIMPLIPY_SPECIAL_BATTERY`, `SIMPLIPY_POLE_GRID`, `SIMPLIPY_ZERO_SIGN`,
  `SIMPLIPY_EXACT_INDEX` and `SIMPLIPY_HIPREC_FRAC` are no longer read; the engine always
  behaves as with each unset. None of the four soundness layers of the miner can be switched
  off, so the provenance sidecar no longer records `soundness.kill_switches`.
  `SIMPLIPY_IVL_NODE_BUDGET` is now `find_rules(interval_node_budget=...)`, and
  `SIMPLIPY_TAGGED_FRACTION_MAX` is gone: the tagged form spells a decimal as a fraction bag
  up to numerator and denominator 10. No environment variable changes what the core mines,
  certifies or serves, so `simplipy.engine.ARTIFACT_ENV_SWITCHES` (declared by the compiled
  core) is empty; a mine's sidecar records every `SIMPLIPY_*` variable set during it. The
  sort promotion no longer reads the unprefixed `N_PTS` and `THRESH` at import.
- **The mine's triple router.** The sidecar's `triple` record (counts, rejects, preempted)
  is replaced by each file's `gates` and `per_length`. The sidecar key
  `soundness.node_budget_env` is gone with the variable it recorded; the node budget is in the
  mine's parameters (`interval_node_budget`).
- **`utils.remove_pow1`**, a cleanup of the generation-1 `pow1`/`pow_1` operators that nothing
  called. Generation-2 configs that declare those names as operators still evaluate them.
- **The `fast-assets` cargo feature**, which gated nothing. Builds that passed it drop the flag.
- **Entries of the compiled core.** `simplipy._core` is internal and carries no stability promise;
  for code that called it directly: the mode-less `Engine` entries `ac_simplify`,
  `ac_simplify_infix`, `ac_ordered_below`, `ac_rules_info`, `ac_served_rules`, `set_rules` and
  `ac_judge` are gone (use the `*_in_mode` entries, `ac_ordered_below_in`,
  `set_mode_rules('default', ...)` and `mine_mark`, which take the mode explicitly, keyword-only
  and without a default), and so are the probes `ac_odd_neg_carriers`,
  `interval_entire_analytic_composition`, `interval_zero_set_null_generic_const` and
  `registry_pole_refusals`. New on `Engine`: `ac_ordered_below_in`, `ac_translate_preview`,
  `judge_hint`, `judge_pair`, `law_instance`, `mine_mark`, `mine_sources`, `prettify` and
  `step_cap_binds`; new in the module: `ARTIFACT_ENV_SWITCHES`,
  `NEAR_MISS_FRAC`, `is_numeric_string`, `range_event`, `reserved_numeric_spelling` and
  `set_interval_node_budget`. The simplify entries default `explore_budget` to the public effort
  (`None`), not 0.

### Changed

<!-- PENDING: CHANGELOG lines from the miner/rules and the core/concurrency fix rounds go here
     (and under Removed, if any). -->
- **The core's tests are lint-clean.** `cargo clippy --all-targets -- -D warnings` passes, the
  test targets included (it stopped at four lints in the tests of the interval layer and the
  high-precision `besselj`).
- **`masking.Role.ORDER`** is the role of `besselj`'s order (its first operand). Like a `pow`
  exponent it decides the domain (a non-integer order is NaN for every negative argument), so
  it types its whole subtree and `mask_fittable` keeps it; `mask_all` masks it.
- **A call takes its operator's arity in arguments.** The infix readers refuse a call whose
  argument count is not its operator's (`sin(x0, x1)`: "sin takes 1 argument, got 2") and a
  comma outside a call's argument list, where they used to emit a malformed prefix list.
  `max` and `min` take any number of at least two, nested to the right, as sympy prints them:
  `Max(x0, x1, x2)` reads as `max(x0, max(x1, x2))`, which the engine reads as one bag, and the
  infix printer writes a right-nested chain as one call again (`max(x0, x1, x2)`; a
  realization stays binary). A call of an undeclared name still reads tolerantly in
  `read_infix`.
- **Every realization is the engine's own kernel.** Each function in `simplipy.operators`
  evaluates, on real operands, the float64 kernel the engine's evaluator uses for its token
  (`simplipy._core`), so the deployed value, the constant fold, the miner and the rule judge
  compute the same bits on every host. Before, numpy's own SIMD routines could decide them: on
  AVX2 and AVX-512 hosts numpy's `tanh` is its own (up to 3 ulp from the platform libm's, and 1.0
  from 18.990341103219276), and on AVX-512 hosts numpy routes `exp`, `log`, `tan`, the inverse
  trigonometric and hyperbolic functions, `sinh`, `cosh`, `power` and `arctan2` through Intel's
  SVML. The transcendentals are the platform libm's, through one module of the core. So `tanh`
  is exactly 1.0 from 19.061547465398498 (glibc), where float64's tails are now read in `f64`
  (`atan`'s stays 5805358775541310). Scalars give numpy float64 scalars (`abs` keeps a Python
  scalar's own type and `rootn` gives a float, as before); arrays of 32,768 elements or more
  are split over threads (`SIMPLIPY_KERNEL_THREADS` caps the count; 1 keeps every call on the
  calling thread), the GIL released. A complex operand of a function with a complex
  continuation takes numpy's, and a torch tensor takes torch's. New realizations name the
  arithmetic, `add`, `sub` and `mul`, for configs that spell every operator in
  `simplipy.operators`. `pow` takes numpy's fast paths, as numpy's `array ** scalar` does:
  the exponent 2 is `x*x`, -1 is `1/x`, 0.5 is `sqrt(x)`, 0 is `1` and 1 is `x`, bit for bit
  numpy's, in the realization, the constant fold and the evaluator alike. So `(-0.0)**0.5` is
  `-0.0` (C's `pow` gives `+0.0`), and the other cells are unchanged (`(-inf)**0.5` is NaN, as
  every negative base under a non-integer exponent). `pow(b, 1/2)` and `rootn(b, 2)` are then
  two functions at `b = -0`, which `f64` keeps apart where `b` may be a zero with a sign of its
  own: `pow(-0.0, 1/2)` is `-0.0` and `rootn(-0.0, 2)` is `0`; over a base that vanishes with a
  sign on a region, `pow(ceil(x0), 1/2)` stays the call it is, so `1/pow(ceil(x0), 1/2)` is
  `-inf` and `1/rootn(ceil(x0), 2)` is `inf` on (-1, 0), and `ceil(x0)^(-1/2)` is
  `1/rootn(ceil(x0), 2)`. Elsewhere `pow(x0, 1/2)` is `rootn(x0, 2)`, as before.
- **Nine new operators and the `acj2-4` engine config.** `simplipy.operators` gains `floor`,
  `ceil`, `max`, `min` (IEEE 754-2019 at the zero tie: `max(-0, +0) = +0`, `min(+0, -0) = -0`),
  `mod` (numpy's `np.mod`, the divisor's sign), `harmonic` (H(x) = digamma(x + 1) +
  euler_gamma, the core's own kernel: within 0.53 ulp on x > -1, `H(-0) = -0`, NaN at every
  negative integer), `erf` and `atan2(y, x)` (the platform libm's), and `besselj(nu, x)`, the
  Bessel function of the first kind of real order: scipy 1.18.1's `jv` carried in the core
  (AMOS and Cephes, BSD-3-Clause, notices in `LICENSES_bundled.txt`), identical to
  `scipy.special.jv` bit for bit, and so with scipy's accuracy. Each is a core kernel, and the
  evaluator dispatches it, so the folds, the miner, simplify and the realization compute the
  same bits. `max(a, b, c)` reads as one bag of three members (and `min` alike).
  `assets/engines/acj2-4/config.yaml` declares acj-4's operators plus the nine, every
  realization in `simplipy.operators`, with no rule sets yet: an engine loaded from it
  simplifies with the constructors alone. The mine kits are `acj2-4/mine.yaml` and
  `acj2-5-4-llm/`.
- **The nine heads in every layer.**
  - *Folds.* `floor`, `ceil` and `mod` of literals fold exactly in `real` (`mod(a, 0)` is NaN)
    and to their float in `f64` and `permissive`, where a literal reads as its float
    (`floor(0.99999999999999999)` is 1 there, 0 in `real`). A function of literals whose float
    is dearer than it stays written (`erf(1)`), as for the other heads. `erf` is odd, so
    `erf(-x)` is `-erf(x)`.
  - *Exact ground folds in `real`.* Where a certified enclosure (the interval layer's value
    set, sound for the real function) decides it, a ground node of these heads folds to the
    exact number it denotes: `floor` and `ceil` of an argument between two integers or exactly
    one (`floor(tanh(1))` is `0`, `ceil(cos(0))` is `1`, `floor(atan2(1, 0))` is `1`), `erf` at
    0 and +-inf, `mod` where the floor of the quotient is decided (`mod(1, sinh(1))` is `1`,
    `mod(cosh(1), 1)` is `cosh(1) - 1`; under an infinite divisor, a dividend of the divisor's
    sign, `mod(-1, log(0))` is `-1`), `besselj(nu, 0)` at an order certainly positive, zero or
    a negative integer (`besselj(sin(1), 0)` is `0`) and `atan2(0, x)` at an `x` certainly
    positive (`atan2(0, cosh(1))` is `0`). An undecided enclosure leaves the node as written:
    `floor(tanh(40))` and `ceil(exp(-800))` stay, where float64 gives `1` and `0` and the reals
    `0` and `1`. `f64` and `permissive` keep their float folds. The `real` rule file no longer
    states these values one spelling at a time (the length-4 mine drops 816 such facts).
  - *A ground fold returns a member.* A ground node whose value is one of its own arguments'
    folds to that argument, never to a newly written float: in `f64` and `permissive` where
    float64 gives the argument's bits, in `real` where an enclosure proves it.
    `mod(sin(1), 123456.78912345678)` and `mod(sin(1), 2)` are `sin(1)` in every mode (`f64` and
    `permissive` wrote the float `0.8414709848078965`, or kept the call), `abs(sin(1))` and
    `abs(-sin(1))` are `sin(1)`, and in `f64` and `permissive` `sin(1) + 1e-30` is `sin(1)`,
    float64's value of the sum (`real` keeps it). Where several arguments carry the value the
    cheapest is taken; a literal argument is its float (`floor(0.99999999999999999)` is `1` in
    `f64`).
  - *`max` and `min` are bags,* as sums and products are: `max(x2, max(x0, x1))`,
    `max(max(x1, x2), x0)` and `max(x0, x1, x2)` are one answer, written `max x0 max x1 x2`
    in the explicit form (the right-nested chain of the members in the canonical order),
    `<max> x0 x1 x2 </max>` in the tagged form (a two-member bag too, as a two-term sum is
    bagged; `<min> ... </min>` alike, read only where the config declares the head) and
    `max(x0, x1, x2)` in infix. A bag costs one head whatever its size, as a sum does. Its laws
    hold bit for bit in float64, whose kernels are IEEE 754-2019's `maximum` and `minimum`
    (NaN-propagating, `-0` below `+0`; like `np.maximum` they return the first NaN operand,
    whose payload the engine does not read), and each was put to the judge: equal members
    collapse (`max(a, a)` is `a`; two `<constant>`s stay apart), the literal members fold to
    their extreme (`max(-0.0, 0)` is `0`, `min(-0.0, 0)` is `-0.0` in `f64`; in `real` and
    `permissive` `pi` and `e` fold with them, each comparison decided by a certified enclosure:
    `max(3, pi)` is `pi`, `min(e, pi, 4)` is `e`; so does every ground member whose value set
    certainly is a finite real, `max(sin(1), 1/2)` is `sin(1)` and `min(-1, asinh(1))` is `-1`,
    and two members whose enclosures overlap both stay; `f64`'s ground fold reads them all as
    their floats), `-inf` drops from
    a `max` and `+inf` from a `min`, and a NaN member makes the bag NaN. A ground bag folds in
    `f64` and `permissive` to the member whose float64 value is the bag's, never to a newly
    written float, and members of one value (the same float64 bits; `+0` and `-0` are two) are
    decided by price: `max(sin(1), 0.8414709848078965)` is `sin(1)` (it was the literal, 7 times
    the price), `max(np.pi, 3.1415926535897933)` is `np.pi`, `min(sin(-0.0), 0)` is `-0.0`. `real`
    orders two ground members whose enclosures overlap at high precision, each literal its exact
    number, up to 1024 bits (an order two adjacent precisions agree on), and keeps both where
    none decides: `max(sin(1), 0.8414709848078965)` is `sin(1)` (6.7e-18 apart; it kept both),
    `max(cosh(acosh(2)), 2)` keeps both. The readings can pick differently:
    `max(cosh(acosh(3)), 3.0000000000000001)` is the literal in `real` (3 is below it) and
    `cosh(acosh(3))` in `f64` and `permissive`, whose kernels give 3.0000000000000004 and read
    the literal as 3.0. The absorber and the
    dual bag need their other members to be no NaN almost everywhere (the finite-a.e. licence;
    `max(+inf, nan)` is `nan`): `max(+inf, x0)` is `+inf` and `max(x0, min(x0, x1))` is `x0`,
    while `max(+inf, log(x0))` and `max(x0, min(x0, log(x1)))` stay; a number of the dual bag
    at most a number of the bag absorbs it too (`max(3, min(2, x0))` is `3`). A rule matches any
    sub-bag of a bag, in any nesting, the other members kept: `max(|x|, x) -> |x|` turns
    `max(max(|x0|, x1), x0)` into `max(x1, |x0|)`.
  - *Enclosures.* `floor`, `ceil`, `erf`, `max` and `min` map their operands' bounds (`erf`
    reaches exactly +-1 from |x| = 5.921587195794507); `atan2` lies in [-pi, pi];
    `besselj(nu, x)` in [-1, 1] for an order nu >= 0, within 1/sqrt(2) at an integer order
    n != 0, between partial sums of its series at an integer order on 0 < |x| <= 2, exactly 0
    or 1 at x = 0, NaN where a non-integer order meets a negative argument; `harmonic` is
    increasing on (-1, inf) and between any two of its poles, NaN at -inf and at every negative
    integer; `mod(a, b)` lies between 0 and `b`, NaN at `b = 0` and at an infinite `a`, and is
    numpy's under an infinite `b` (`mod(1, -inf)` is `-inf`, `mod(-1, inf)` is `inf`,
    `mod(-1, -inf)` is `-1`); `atan2(0, x)` is 0 at x > 0 and `atan2(y, 0)` is +-pi/2 at
    y != 0. So a ground NaN or infinity these decide folds in every mode:
    `harmonic(atanh(ceil(-1)))`, `acos(harmonic(sinh(-1)))`, `acosh(besselj(-1, 1))` and
    `log(besselj(-1, 1))` are NaN, `mod(1, log(0))` is `-inf`. `harmonic` of an argument that
    may reach a pole among other values, `besselj` of an order that may be a negative
    non-integer at an argument that may reach 0, `mod` by a divisor that may be zero, and every
    head without an arm are unanalyzable: every certificate refuses them, and the miner accepts
    no candidate for a variable-free source it cannot analyze. The tail certificate's table of
    pole-free operators names the operators that are, so an unnamed head is pole-bearing.
  - *Certificates.* The finite-almost-everywhere, nonzero-almost-everywhere and zero-set
    certificates read each of the nine by its argument, so a cancellation or a zero product
    over one is licensed where it was refused: `x1*erf(x2)/erf(x2)`,
    `x1*besselj(0, x2)/besselj(0, x2)`, `max(x1, x2)/max(x1, x2)` and
    `x3*mod(x1, x2)/mod(x1, x2)` are `x1`, `1` and `x3`, and `mod(x1, x2)*0` is `0`, in `f64`
    and `real`. `erf` vanishes only at 0; `besselj` at a literal order has isolated zeros in x;
    `harmonic` is finite off its poles at the negative integers and vanishes once between each
    two; `max` and `min` vanish only where a member does; `mod(x, y)` vanishes on countably many
    hyperplanes x = k y and is NaN only at y = 0; `atan2(y, x)` vanishes only where y = 0. They
    stay refused where the head vanishes, is NaN or has a pole on a region: `floor(x2)` (0 on
    [0, 1)), `ceil(x2)`, `besselj(3/2, x2)` (NaN below 0), `besselj(x1, x2)` (an order that is
    no literal), and `mod(x1, 2)`, which float64 makes 0 at every |x1| >= 2^53. A step head of
    an argument that certainly misses the interval is nonzero (`floor(exp(x2) + 2)`). In `f64`,
    `harmonic(x2)*0` and `x1*harmonic(x2)/harmonic(x2)` stay: float64's `harmonic` is NaN at
    every float at or below -2^52, each a negative integer; `real` gives `0` and `x1`.
  - *Signed zeros.* `ceil(-0.5)`, `mod(3, -3)` and their kin are -0 in float64, and `atan2`
    reads the sign of a zero in both arguments, as an inverse does (`atan2(-0, -1)` is -pi,
    `atan2(+0, -0)` is pi). In `f64` such a ground folds to `-0.0` and a ground infinity takes
    float64's sign (`1/ceil(-0.5)` and `1/mod(3, -3)` are `-inf`). Every `atan2` node is built
    from its arguments' product members, however it arises (the parse, a rule's right-hand
    side, the pass's rebuild): a negative coefficient of the first leaves the call,
    `atan2(-S, x) = -atan2(S, x)` (exact in float64, not in `real`), and no argument's sign
    moves into a sum's orientation, so a sum that vanishes on a region keeps the sign of its
    zero in either argument (printed as the negation it is). The search moves inside an
    `atan2`'s arguments as inside any other head's (`atan2((x0 + 1)^2 - x0^2 - 1, x1)` is
    `atan2(2*x0, x1)`), and in `f64` takes no move where the product member it rewrites, or what
    it writes, vanishes with a sign of its own, the bar an inverse keeps:
    `atan2(x0*x1 - x0*|x1|, -1)` stays, since the factored `x0*(x1 - |x1|)` is -0 for every
    x1 >= 0 where x0 < 0 and the call would read -pi for pi. The step and lattice heads count as
    region zeros wherever an inverse reads one.
  - *Discontinuities.* The jumps of `floor`, `ceil` and `mod`, the poles of `harmonic` and the
    cut of `atan2` are read the way poles are. Over the reals a rule must hold at every jump: the
    contract evaluates the step heads exactly, a value inside the working-precision band of a
    jump, a pole or the cut is unresolved rather than guessed, and the licence registry probes
    the jumps of an argument affine in one variable, so `ceil(3x) -> floor(3x) + 1` is refused
    at x = 1/3. In float64 a rule spelled with one of the nine must also agree bit for bit with
    the float64 kernels, a zero's sign included, at each variable's battery, +-inf, NaN, -0, the
    float neighbours of the integers, tiny and huge magnitudes and the negative integers, every
    0.05 over [-25, 25] and on the diagonal: one function in the core (`f64_bar`), which the
    contract judge ANDs into a rule's realisation and the `f64` mine applies to every pick. So
    `floor(x + 1) -> floor(x) + 1`, `mod(x + 1, 1) -> mod(x, 1)` and `floor(mod(x, 1)) -> 0`,
    true over the reals and false in float64 just below an integer, are minted in `real` and
    `permissive` and refused in `f64`, and `ceil(x) -> floor(x) + 1`, false at every integer,
    is refused everywhere. A source `<constant>` is probed as a finite variable; a fitted target
    constant beside a step head or `mod` cannot be judged bit for bit, so such a rule is not
    served in `f64`. A source constant of such a source is also swept over zero and the small
    integers and half-integers, and the refuter's witnesses read the new heads (integer
    neighbours for `floor`, `ceil` and `harmonic`; preimages through `atan2`, `mod`, `max`
    and `min`).
  - *High precision and the judge.* All nine have high-precision and contract kernels
    (`besselj`'s below): `floor`, `ceil`, `max`, `min` and `mod` exact on the representation, `atan2` from `atan`,
    `erf` by a series without cancellation (`2a/sqrt(pi) e^(-a^2) sum (2a^2)^n / (2n+1)!!`) and
    `harmonic` without the sum `digamma(x + 1) + gamma` (a finite sum and a digamma difference
    at two large arguments, the Bernoulli numbers from the tangent numbers), each within the
    working precision of mpmath's value at 400 digits, `H(1e-300)` included (mpmath's own
    `harmonic` gives -4e-84 there at 80 digits), and unresolved past the judge's 2000-digit
    ceiling, where it costs seconds a call. So the contract judges a rule with them like
    any other, and such rules enter all three files: `floor(floor(x)) -> floor(x)` is `core`,
    `floor(x + 1) -> floor(x) + 1` is `real`, `floor(<constant>) -> <constant>` and
    `harmonic(<constant>) -> <constant>` are `core`. An operator the high-precision
    evaluators do not know (none of the nine) is unevaluable, even at a NaN operand: the
    contract point verdict is none (it read as equal), the licence registry refuses a pattern
    with exceptional points that holds one, the judge refuses its rule (`UNSUPPORTED-SHAPE`)
    and the mine refuses it too. The corpus monitor evaluates the nine with the contract's
    kernels and float64's zero signs, and reads no zero sign for an operator its table does not
    name; the high-precision oracle reads a complex value as NaN.
  - *Sort promotion.* Sort promotion's high-precision oracle evaluates the nine heads with the
    contract's kernels, so its contract atom bar and its exact-arbiter overturn read their
    contract values (the atom bar fell back to float64 for them), and `floor(floor(x)) ->
    floor(x)` and `mod(x, 1) -> x - floor(x)` ship at `_` in every mode. Beside the `pow`
    integer-preimage probe a step-preimage probe solves where a `floor`, `ceil` or `harmonic`
    argument, or a `mod` dividend by a literal divisor, crosses its jump or pole at a point no
    atom holds: `ceil(sinh(x) + 1/2) - floor(sinh(x) + 1/2) -> 1`, true at every atom, is
    killed at x = asinh(-5.5). Its ground tier skips a probe a contract kernel cannot decide
    (`erf(1e20)`, inside the band of its asymptote) instead of dropping the rule as unevaluable,
    so licensed constant collapses such as `acos(erf(c)) -> c_t` and
    `harmonic(exp(c)) -> c_t` reach the `f64` and `real` files (`permissive` folds a function
    of a constant itself).
  - *The judge's constant witness.* The witness of a target `<constant>` is polished at the
    generic point the fit determined it at (it was always the first one, where a lattice head
    can make the target flat in the witness: `max(x, acos(c)) -> max(c_t, x)` at x = 1.7), and
    the measure and diagonal lanes re-round it at each rung as the battery lane does (frozen at
    50 digits it left a fixed gap against an exact `<constant>` at every higher rung). Both
    convicted identities at every head: `x + log(c) -> c_t + x`, `max(x, log(c)) -> max(c_t, x)`
    and `atan2(x, acosh(c)) -> atan2(x, c_t)` were KILLs and are certified. The float64 fit
    keeps the witness it found: a snap to a round value (0, +-1, an integer) is taken only where
    it reproduces the source at least as well, so `asinh(exp(c)) -> c_t` keeps 3.7e-44 at
    c = -100 (it snapped to 0) and `erf(c) -> c_t` keeps 0.9999999999984626 at c = 5 (it
    snapped to 1), and both are realised (`core`; they were `real`). A fitted witness is no
    written bound for the saturation test either, so `tanh(c) -> c_t` and `erf(c) -> c_t` are
    certified (they were KILLs at c = 30, where no float64 carries the witness 1 - 1e-393);
    `tanh(c) -> 1` is still refused.
  - *Fingerprints.* The libm fingerprint probes `erf`, `atan2` and `besselj` (whose port calls
    libm in each of its regimes), and the measure fingerprint prices `atan2(x0, x1)`, so every
    host's fingerprint moves.
  - *besselj* accepts torch tensors: the core kernel on the host, returned in the tensor's dtype
    and device (not differentiable). The port carries no branch a real argument cannot reach
    (AMOS's overflow rescaling and its `kscl`); its bits stay scipy's.
- **`besselj` is complete at every real order.**
  - *Identities in the engine, in every mode.* The canonical order is non-negative:
    `besselj(-3, x)` is `-besselj(3, x)` and `besselj(-2, x)` is `besselj(2, x)`. An even
    integer order is an even function, so `besselj(2, -x)`, `besselj(2, |x|)` and
    `besselj(2, sin(-x))` are `besselj(2, x)` and `besselj(0, x2 - x1)` is
    `besselj(0, x1 - x2)`; an odd one carries a sign out, so `besselj(3, -2*x)` is
    `-besselj(3, 2*x)`. An odd order is an odd head, as `sin` is: its sign trades into a sum
    argument's orientation and the class owners place it (`-besselj(1, x0 - x1)` is
    `besselj(1, x1 - x0)`, `besselj(1, -2) + besselj(1, 2)` is `0`), and an even order's sum
    takes one orientation (`besselj(2, x1 - x0)` is `besselj(2, x0 - x1)`). The reflection and
    the parity hold bit for bit in float64, a zero's sign included; a sum's orientation keeps
    the sign of its zero exactly where `sin`'s does (where the zero set may hold a region). A
    literal order at 0 folds exactly: `besselj(0, 0)` is `1`, `besselj(5/2, 0)` and
    `besselj(-2, 0)` are `0` (in `f64` the kernel's zero, whose sign varies with the order:
    `besselj(-1, 0)` and `besselj(1.5, 0)` are `-0.0`, so `1/besselj(1.5, 0)` is `-inf`, as in
    scipy), and a negative non-integer order gives its pole's one-sided limit, the sign of
    `1/Gamma(nu + 1)`: `besselj(-1/2, 0)` is `inf`, `besselj(-3/2, 0)` is `-inf`, as scipy
    has them; an infinite order gives 0 at a literal `x >= 0` and NaN below
    (`besselj(atanh(1), 2)` is `0`).
  - *The kernel.* An integer order reads its reflection and its parity, so both hold bit for
    bit: `besselj(1, -0.0)` and `besselj(3, -1e-300)` are `-0.0` (scipy gives `0.0`),
    `besselj(2**31, -1.0)` is `0.0` (scipy gives NaN: its integer test is a C `int` cast).
    Where scipy's routines give up under orders of about 3e17 and more, a value a bound puts
    below the subnormals is `0.0` (`besselj(1e300, 100)`; scipy gives NaN). Everywhere else the
    value is `scipy.special.jv`'s bit for bit, over 2.5 million (order, argument) pairs with
    orders to 1e308. scipy's other NaN cells stay: the oscillating region of orders beyond
    3e17, and negative non-integer orders below about -470 where the value nears the float
    range's end.
  - *High precision and the judge.* `besselj` has a high-precision kernel in the core (the
    Hankel expansion where its bounded error reaches the target, the power series elsewhere at
    a precision that carries the cancellation, Gamma by Stirling's series), within the last
    place of mpmath's `besselj` at 169 to 3,325 bits over 980 points, and the contract judge
    evaluates it with mpmath. Both read scipy's cells and refuse an order beyond 1e3 or an
    argument beyond 1e12. So a `besselj` rule is judged like any other: `besselj(1, -x) ->
    -besselj(1, x)` is `core`, `besselj(1/2, x) -> sqrt(2/(pi*x))*sin(x)` is `real` (equal over
    the reals, a few ulp apart in float64), and `besselj` rules enter the `real` file and the
    `<constant>` lanes. The corpus monitor reads `besselj`, with float64's zero signs.
  - *Enclosures.* A negative integer order lies in [-1, 1] like a non-negative one; an order
    that may be a negative non-integer is bounded away from its pole at 0 by the recurrence
    `J_nu = (2 (nu + 1)/x) J_{nu+1} - J_{nu+2}` (`besselj(-3/2, x)` over [2, 3] within +-3.5),
    and refused where the argument may reach 0 among other values. A non-integer order on the
    negative axis is NaN alone and its pole at 0 has its sign, so ground cases fold in every
    mode: `besselj(1/2, -1)` is `nan`, `besselj(sin(-1), 0)` is `inf`, `besselj(tan(-1), 0)`
    is `-inf`.
- **A rule file is read in the canon it was certified in.** Since the per-mode mines, every set
  was read in its serving mode's canon, so a file mined before them could read as rules nothing
  judged: `real` read the f64 fact `rootn 2 atanh tanh 3 -> nan` as `rootn 2 3 -> nan`, and
  `rootn(2, 3)`, `x1 + rootn(2, 3)` and `x1*rootn(10, 5)` simplified to `nan` in `real` (212
  such rows served on acj-5-4-llm). `from_config` now reads each file's canon off its
  provenance sidecar: a recorded `mode` is that mode's canon; a file whose sidecar records none
  (simplipy 0.14 and earlier mined all three files in one canon), or that has no sidecar, is
  read in that one canon (`legacy`). Numerals are read in the serving mode's number domain
  either way. In-memory sets are read in their own mode. New keyword
  `SimpliPyEngine(rule_canons=...)`; on the core, `from_strs(.., canon)`,
  `set_mode_rules(.., canon=)`, `ac_canonical_keys(.., canon=)` and `mode_rules_canon`.
  On acj-5-4-llm, `real` now serves 133,552 rules (was 132,754; 122 subsumed at load, was 838)
  and the f64 file keeps 141 more rows through the construction dedup.
- **The engine-owned-law gate takes the search's own FACTOR move.** The load and the miner
  refused a distributive-law rule only over plain sums and products, so
  `sin(x)*a + sin(x)*b -> sin(x)*(a + b)` loaded and was mined although the search performs it
  in every mode. The gate now judges the rule on an instance the search meets, with the search's
  licence and `factor_moves`. On acj-5-4-llm it drops 2 (f64), 10 (real) and 13 (permissive)
  rows at load, was 0, 5 and 5.
- **A mine without `--reset-rules` starts each mode from its own set.** The real and
  permissive mines were seeded from the f64 set.
- **No judge timeout inside a mine.** The judge's 30 s wall-clock timeout filed a slow
  judgment as a rejection, so a mined file depended on machine load. A mine now judges with no
  timeout under a watchdog: a judgment past `judge_warn_s` (60 s) is logged, one past
  `judge_fail_s` (600 s) is interrupted in the thread that runs it, whichever thread that is,
  and fails the mine with the rule's name. Both are new `find_rules` parameters, recorded in
  the sidecar. A judgment with no timeout leaves the process's own alarm and its handler
  alone. Outside a mine, `verify_ruleset` keeps its 30 s timeout.
- **Checkpoints resume only their own mine.** The key now also holds the operator table, the
  content of X and of the confirm matrix, the numeric environment (python, numpy, scipy,
  mpmath, libc, the libm fingerprint) and the `SIMPLIPY_*` switches. A resumed mine keeps its
  original `created` and `host` and records each resume under `resumed`.
- **The judge loop's rounds cap is per source.** One stubborn source no longer fails a whole
  length after the scan with nothing saved: it is given up on after 8 refusals while the others
  converge, the checkpoint (with every verdict reached) is written, and the mine fails naming
  the source and its excluded targets. The proposals' loop caps per proposal.
- **The post-pass judges before it prunes.** The final gate now runs before each covered-rule
  prune and once more on the set that ships, so a prune never leans on a rule the gate drops.
- **`verify-mine` passes nothing vacuously.** New gates `sidecar` (a final, uninterrupted mine
  of the mode asked) and `rules` (the file is not empty); `fingerprint` fails on a probe the
  record lacks (an empty record) and on a record of another mode. The `engine_owned_law` gate
  catches a sign pull (`neg sin _0 -> sin neg _0`): the core places that sign itself, so both
  sides are one canonical state.
- **`requires_simplipy`**: an empty specifier is malformed, and an exact pin of a pre-release
  (`==0.15.0.dev2`) admits that pre-release. Other specifiers still judge a pre-release as the
  release it precedes.
- **Rule pushes**: the core's pushed-rules digest is stored only after the push compiles, and a
  push re-translates only the set it replaces (a default push used to discard and rebuild the
  real and permissive translations).
- **The promotion refund reads the constructor's dedup** instead of a copy of it ("shortest
  target"), which disagreed with the serve-ordering tie-break and sent every probe to the slow
  rebuild.
- **The search keeps a one-signed pole joined** in `f64` and `real`, where a pole is read by its
  limit. Distributing `(1 + x3 - x1 + 5/3*exp(x2))/abs(x3)` gave
  `5/3*exp(x2)*abs(1/x3) + (x3 + 1 - x1)*abs(1/x3)`, which is `inf - inf` at `x3 = 0` wherever
  `1 < x1 < 1 + 5/3*exp(x2)`, where the expression is `+inf`. A factor is now distributed only
  where at most one of the terms it lands in keeps its pole, where its poles are two-signed
  (`1/x`, `1/sin(x)`), or where the terms it multiplies keep one sign. Answers change on 4 of
  srbf's 6,531 ground truths and 11 of a tenth of the T8.1 inputs in each of the two modes, none
  priced lower; `permissive` keeps no value at a pole and is unchanged.
- **`mode='real'` is never served the f64 set, also while another thread unloads it.** The
  fail-closed check runs inside the core call, under the read guard the answer is computed
  under. A call that finds the set unloaded rebuilds it and answers, where it used to raise
  `ValueError` or, rarely, answer from the f64 set.
- **No call waits for the engine with the GIL held.** A call queued behind a rule push
  (`set_mode_rules`, a deferred set's first build) froze every other Python thread until the push
  landed (14.9 s measured); every entry now waits with the GIL released.
- **Permissive answers holding tiny numbers are fixpoints.** `x1 + 5/1.7976931348623157e308`
  folded to `2.781342323134002e-308`, a literal the reader keeps as a symbol, so a second call
  reordered the sum. A folded float whose shortest decimal is outside the f64 number range stays
  unfolded, as the written decimal does.
- **`to_tagged` bags every quotient.** `to_tagged(['exp', '1/2'])` is
  `['exp', '<mul>', '1', '<div>', '2', '</mul>']` and `to_tagged(['inv', 'x0'])` is
  `['<mul>', '1', '<div>', 'x0', '</mul>']`. Such a sequence carried no bag, so `simplify` read it
  as explicit prefix and answered in explicit prefix. A lone negation stays `neg x0`.
- **A token that is neither a string nor a number is refused** with `ValueError`:
  `simplify(['+', 'x1', True])` read `True` as the symbol `'True'`, and `None` as `'None'`.
- **`read_infix` reads `pow1` and `pow_1` as the legacy operators only where they are called.**
  `read_infix('pow_1(x0)')` is still `['inv', 'x0']`, and a bare `pow_1` is a symbol, as `pow_2`
  is: `read_infix('x0*pow_1')` returned the malformed `['*', 'x0', 'inv']`.
- **`import simplipy` succeeds without the compiled core**, and constructing an engine raises the
  `ImportError` that names the failure; an unguarded import in `simplipy.utils` failed the import
  itself. For code that calls `simplipy._core` directly: the simplify and infix entries take a
  keyword `own_set`, which raises the new `simplipy._core.MissingModeSet` where the mode holds no
  set of its own.
- **The mining calls name their mode.** `prune_covered_rules`, `deduplicate_rules` and
  `promote` take a required keyword `rule_mode`: `'f64'` (also spelled `'default'`, the core's
  name for the f64 set), `'real'` or `'permissive'`. The CLI's `prune-covered-rules` prunes the
  f64 set. Every entry of the compiled core that reads a mode takes it explicitly, and its
  mode-less twins are gone. Migration: add `rule_mode='f64'` where the call relied on the f64
  set.
- **One mode vocabulary.** Every entry that takes a mode (`mode=`, `modes=`, `rule_mode=`,
  `unload_mode`, `verify_ruleset`, `verify_mine`, `cross_mode_complexity` and the CLI's
  `verify-mine --mode`) accepts `'f64'` and `'default'` as the same mode, besides `'real'`,
  `'permissive'` (in any case) and `simplipy.Mode` members. Before, `rule_mode` took
  `'default'` and refused `'f64'` (`prune_covered_rules` raised a bare `KeyError`), while
  `simplify`, `--mode` and `verify_mine` took `'f64'` and refused `'default'`. An unknown name
  raises `ValueError` naming both spellings, and a mode that is neither a string nor a `Mode`
  raises `TypeError` (`verify_ruleset` raised `ValueError`).
- **`simplipy.mining` declares `verify_mine` and `cross_mode_complexity`** in `__all__`, so both
  are declared surface, in the power-user tier of the compatibility policy.
- **Deduplication keeps the target lower in the mode's serve ordering**, not the one with
  fewer tokens.
- **`complexity(mode=...)` refuses an unknown mode with `ValueError`**, as `simplify` does (it
  raised `KeyError`).
- **Infix text calling an undeclared function is refused by name**
  (`unknown function 'sqrt' in 'sqrt(x1^2)'`) in `simplify`, `complexity`, `evaluate_constants`,
  `mask` and the conversions, instead of "invalid or malformed prefix expression"; `read_infix`
  still keeps such a call as written. Write an even root as `rootn(x, 2)`.
- **`simplify(max_passes=0)` raises `ValueError`** ("max_passes must be at least 1"); it used to
  run one pass. One pass reaches the fixpoint, so `max_passes=1` gives the same answers as the
  default; the cap guards against a bug and does not bind on real input.
- **One reading of an expression argument.** `simplify`, `complexity`, `prettify`, the
  conversions and `simplipy.normalization` read an argument the same way. Realization notation
  is accepted by each of them (`simplify(['simplipy.operators.neg', 'x1'])` reads like
  `to_prefix`; it raised), but only through an injective realization map: a config in which two
  operators share a realization (`abs` and `absolute` both `np.abs`) refuses such input, naming
  both, instead of silently picking one. A numpy array must be one-dimensional with string-like
  tokens, or `ValueError` says which (`complexity(np.array([['x1']]))` raised a raw `TypeError`,
  and `simplipy.normalization` flattened it); a bytes array is decoded.
- **Rule sets are compiled when they load.** `load` returns with every set it was asked for
  translated and ready, so no mode's first call pays for it (before, the first call per mode
  did, for seconds). Peak memory with `acj-5-4-llm`: about 0.9 GB for all three sets, about
  0.7 GB for `modes=('f64', 'permissive')`, about 0.45 GB for `f64` alone; a set left out builds
  on its mode's first use in about 3–4 s and adds about 100 MB. Answers do not change.
- **Library messages go through `logging`.** The default-engine announcement and the first-use
  build of a deferred set are WARNING records on the `simplipy` logger (visible with no logging
  setup), install progress is INFO and an install failure ERROR; nothing in the library prints
  except `list_assets`, whose output is its purpose. To silence them:
  `logging.getLogger('simplipy').setLevel(logging.ERROR)`. The CLI shows INFO records on
  stderr.
- **An installed asset loads without the network.** `get_path` (and so `load`) fetched the
  manifest from the hub on every call, even for an installed asset. It now resolves an installed
  asset from the local copy of the manifest, with the files present and their digests verified,
  and goes online only when the asset is not installed or no local manifest exists.
- **`install_asset` returns `False` on every download failure**, an offline or connection error
  included (only an HTTP error did; the others escaped raw and left a partial download), logs the
  cause and removes the partial download. A digest mismatch still raises `RuntimeError`.
- **`evaluate_constants` knows a variable as the engine does.** It used its own definition
  (`x<digits>`), so a subtree holding `y` was tried as a constant and left alone only by a
  blanket `except`; it now asks `expression_variables` and catches only the evaluator's
  `ValueError` for an operator it has no kernel for.
- **An integer spelling of zero reads `+0`.** The core's literal reader (the offline evaluator,
  the interval kernel, `permissive`'s ground fold) read `-0`, `-00` and `(-0)` as `-0.0`, while
  the deployed evaluator reads an integer and an integer has one zero; they now read `+0.0`. A
  float spelling keeps its sign (`-0.0`, `-0e0`), as Python's float literal does. No mined or
  served token is affected.
- **`verify_triple` checks the triple it is given.** With `engine_config`, permissive dominance
  is checked on an engine built from the config's operators and the three rule sets passed in,
  not from the rule files the config names, and every answer is priced by
  `simplipy.mining.cross_mode_complexity`. Without `corpus_rows` it sweeps
  `simplipy.verify.default_corpus()`, which ships in the wheel (a wheel install used to sweep
  nothing and warn). A row on which a mode raises is a violation, reported with its error.
- **The load census names its reasons anew.** The rows the load drops are counted by reason
  (`_core`'s `ac_rules_drop_census` and `ac_shadow_census`): `engine-owned-law` is new (a rule
  stating the distributive law), and `inverse-of-earlier-rule` and `twin-inverse` are gone,
  their rows dropped as `not-ordered-below`, the reason they always also had. Served rules do
  not change.

## 0.14.7 (2026-10-01)

- **A floating-point warning inside an expression is a warning again.** An expression's inline
  arithmetic (`x1 + x2`, `x1 * x2`) runs in its own frame, so numpy raises its floating-point
  warnings there (`inf - inf`, an overflow in a product), and Python's warning machinery looks
  for `__builtins__` in that frame's globals. The evaluation namespace did not carry it, so such
  an evaluation raised `KeyError('__builtins__')` instead of returning `nan` with a
  `RuntimeWarning` whenever numpy's error policy was `warn` (its default). The namespace now
  binds `__builtins__` explicitly. Bare names resolved through the interpreter's builtins before
  and still do, so what an expression can reach is unchanged.

## 0.14.6 (2026-09-03)

- **The permissive tier folds an expensive exact literal to its float.** The core keeps every
  literal as an exact rational and prints the cheaper of its two exact codewords, so a value
  with no finite decimal expansion -- the quotient of two 16-digit decimals, a coefficient
  cleared into `p/q`, a 30-digit integer -- came out as a fraction of two long integers:
  `4 / (2.7167019109484434 * 3.1415926535897)` printed `200000000000000000000000000000 /
  426738538271436458205631863649`, 196.8 bits and three tokens for one degree of freedom, on
  1-5% of training skeletons. The strict tiers must do that: a spelling denotes the state's
  exact value. The permissive tier is licensed to move values, so there such a literal now
  folds to the f64 nearest to it (printed as that float's exact decimal) WHEN mu' prices that spelling strictly
  cheaper (`0.46867105279529636`, 57.1 bits). The gate is the same self-limiting mu comparison
  the f64 transcendental fold uses: `1/2`, `15/37` and `4366/8875` keep their fractions, a
  drawn constant is already its own shortest spelling and never moves, and a moved literal
  re-simplifies with its neighbours so the endpoint stays the chain's fixpoint. The same
  licence folds an all-literal arithmetic subtree the exact fold refused because its value left
  `i128` (`74.22 - 92.13 * pow(9.44, 4)`, four operators whose value is one number) by evaluating
  it in f64, children first. `Mode.f64` and `Mode.real` are unchanged.

## 0.14.5 (2026-09-02)

- **`literal_sites` types every literal inside an exponent or root-index subtree.** The
  canonical spells a rational exponent structurally -- `pow(x, 3/2)` is
  `pow x <mul> 3 <div> 2 </mul>` -- and the role walk entered that `<mul>` with its own
  COEFFICIENT role, so `mask_fittable` masked both literals and the collect stage folded
  them into one free exponent: exactly the real-valued exponent the policy exists to keep
  (`pow(x, c)` is undefined on half the axis for almost every `c`). Openers and the
  arithmetic operators now pass EXPONENT / ROOT_INDEX down through the whole typed
  subtree (tagged and explicit spellings alike); the base of a `pow` stays a value
  position, so its coefficients remain fittable. Measured on the 29-catalog holdout pool
  (6,660 laws), the structural spelling occurs in 17% of them (20% of erbench-syneq, 5%
  of the rest; `1/2`, `3/2` and `-1/2` carry most of it), and in every one of them the
  fittable skeleton used to lose the exponent.

## 0.14.4 (2026-08-31)

- **The mimalloc wheels actually publish.** 0.14.3 never reached PyPI: the manylinux
  aarch64 cross toolchain cannot compile libmimalloc-sys (its gcc rejects the
  `-Wdate-time` probe), so the dependency is now scoped away from exactly that target --
  linux-aarch64 wheels keep glibc malloc; every other platform ships the allocator as
  0.14.3 intended. No other change.

## 0.14.3 (2026-08-31, tagged but not published)

- **Wheels ship with the mimalloc allocator** (`mimalloc-allocator` is now in the maturin
  feature set; the cargo feature stays opt-in for source builds). Measured on the v25
  data-generation workload: **-8.6% per `simplify` call** on real draw inputs -- the AC
  rewrite loop is allocation-heavy (~25% of simplify self-time is clone/alloc/free), and
  mimalloc's per-thread heaps also address the long-standing glibc RSS growth under the
  miner's allocation storm (the reason the feature exists). No output changes: an
  allocator swap moves time and memory, never a result.

## 0.14.2 (2026-08-30)

- **Per-mode lazy rule loading: `modes=` on `load`/`from_config`/`__init__`, plus the
  operational drop `unload_mode`** (task #88). The default `modes='all'` builds every
  set the artifact names, byte-identical to the historical behavior — lean is opted
  into, never inherited. A lean selection (e.g. the worker profile
  `modes=('f64', 'permissive')`) defers any other config-named set: its file is not
  even read at load; it builds on the mode's first `simplify`/`complexity` use —
  once, lock-guarded against double-building (with a fairness gate so the install
  lands under concurrent traffic), announced with one loud log line on the same
  channel as the default-engine announcement. The machinery is additive-only:
  nothing is ever evicted implicitly. `unload_mode(mode)` is the paired public RAM
  knob: it drops the mode's built structures (for config-named sets the parsed list
  too) and the next use lazily rebuilds from the recorded recipe; the default f64
  set is the core's construction substrate and loudly refuses to unload. Measured on
  acj-5-4-llm (fresh process, RSS after `load`): `'all'` 569 MB,
  `('f64', 'permissive')` 469 MB, `('f64',)` 338 MB, with a deferred set costing
  ~4 s and ~130 MB on first use. Eager, lazy-on-first-use, and reloaded-after-unload
  engines produce byte-identical per-mode outputs (pinned by the identity suite in
  `tests/test_lazy_modes.py`).

## 0.14.1 (2026-08-30)

- **The triple router judges the spellings as they will be WRITTEN** (task #83).
  `mining._route_triple` used to tier every rule by judging its served-canon form
  (`ac_served_rules`) — the in-memory respelling the load canon mints, with ground
  subterms folded to exact rationals — while release verification judges the FILES,
  which serialize `engine.simplification_rules` verbatim. For 13 constant-fold rules
  of the run-5 mine (`+ tanh log 9 _0 → + inv 1.025 _0` and family) the folded served
  form judged f64-realised and the router put them into `rules_f64.json`, which the
  file-level gate — agreeing with the deployed f64 behaviour, measured at 1–6 ulp
  drift — then rejected as `real`-tier. The router now judges exactly the pairs the
  writer dumps, so the judged spelling and the written spelling coincide by
  construction and the router cannot disagree with release verification. The served
  set still decides EXISTENCE (a rule translation drops is licensed nowhere), and the
  served twins' soundness remains the upstream symbolic gate's jurisdiction.
- **`complexity()` stays Default-canon-pinned as THE public measure, and gains the
  explicit diagnostic `canon='mode'`** (task #87, owner ruling: SHIP BOTH).
  `canon='default'` (the default) is byte-identical to the previous behaviour: one
  Default-canon yardstick for every mode's output, with `mode` routing only the parse.
  The docs and the docstring now state this plainly — the descent theorem
  `μ(simplify(e)) ≤ μ(e)` is a theorem of the public pricing for the default `f64`
  mode; a `real`/`permissive` chain descends its own mode's canon measure (an internal
  descent), and its fixpoint may price above its input on the public yardstick.
  `canon='mode'` routes the CANON through the requested mode as well — the
  engine-internal diagnostic that prices in the measure that mode's chain actually
  descends, making the per-mode serve guarantee checkable from the outside. Validated
  on the 25 run-5 inflated fixpoints (real/permissive arms whose outputs priced above
  input under the public measure): under `canon='mode'` with the arm's mode, 0 of 25
  price above input (14 strictly below, 11 equal).
- **Promotion stage 4 (the derivability refund) probes under one engine, not one
  engine per rule.** The reference loop rebuilt a full `SimpliPyEngine` — ~5 s of
  construction plus ~7 s of lazy rule translation — inside its serial per-`?`-rule
  probe, which is ~49 hours over a real 163k-rule mine. "The engine WITHOUT this
  rule" is now produced per probe by suppressing the rule's artifact row (and the
  orientation twins minted from it) at the matcher's single fire site: behaviorally
  the rebuild, because translation is per-rule and bucket order filters in place.
  The one place that argument fails is translate's cross-rule state — removing a
  rule can resurrect a sibling's shadowed orientation twin, which carries the
  removed rule's identical rewrite — and the new SHADOW CENSUS enumerates exactly
  those rows: the refund leaves such an owner unsuppressed as the stand-in for the
  twin the rebuild would serve, and anything outside that argument (dedup-entangled
  rules, inverse-pair drops) falls back to the reference per-candidate rebuild.
  The keep/prune decision sequence and rule order are the reference loop's: on a
  3,000-rule slice of the run5 corpus the old and new implementations return
  identical keep lists (64.1 s vs 0.4 s); the full 163,002-rule snapshot runs its
  13,061 probes in 40 s (1,401 pruned, zero fallbacks). The stage now reports
  probes done/total every 500 through the `[promote]` reporter, so a mine's
  longest silent stage is attributable while it runs.
  Core: `Engine.ac_simplify_suppressed` (default-mode simplify with artifact rows
  suppressed at the fire site) and `Engine.ac_shadow_census` (the translate-time
  cross-rule dependency table) on `simplipy._core`.
- **`verify_triple` no longer reports an unrunnable dominance check as a pass.** With
  `engine_config` given and a corpus of 0 rows (every wheel install lacks
  `benchmarks/corpus/raw_skeletons_nv.json`), `permissive_dominance` used to come back
  `[]` — indistinguishable from a genuine zero-violation sweep, which it did twice
  during the 0.14.0 release verification. It is now an explicit sentinel
  `{'checked': False, 'reason': …}` plus a `UserWarning`; a sweep that actually ran
  returns `{'checked': True, 'violations': […]}`. `is_clean` is unaffected by an
  unchecked dominance: unchecked means unknown, not passing.

## 0.14.0 (2026-08-30)

### Changed — SOUNDNESS IS AN AXIS, NOT A LADDER (owner rulings, 2026-08-19/20)

The third mode is spelled **`Mode.permissive`** (renamed from the pre-release
`corpus` spelling before it ever shipped, 2026-08-25): the name states the
contract — certificates skipped, gates dropped, not equivalence-preserving —
instead of the application it was built for. The pre-release string raises
with a pointer; artifact configs that declare `rules_corpus:` (acj-4 among
them) load unchanged via a reader-side fallback.

`Mode` is now `f64` (the default), `real` and `permissive`. It is a plain `Enum`, so `<`
between modes raises `TypeError`.

The old `IntEnum` encoded a premise — `EXACT ≤ SOUND ≤ AE ≤ LOSSY` — that measurement
refuted. "True over ℝ" and "realised in f64" are *incomparable*: `atanh(tanh t) → t` is
true for every real `t` and gives `inf` in f64 past 18.990341103219276 on the
release host (the exact threshold is libm-dependent), while
`asin(1e-8) → 1e-8` is bit-identical in f64 and wrong by the cubic term. Neither is more
sound than the other.

- **`Mode.SOUND` and `Mode.LOSSY` are removed**, as are the strings `'sound'` and
  `'lossy'`. Use `Mode.f64` and `Mode.permissive`. A retired spelling now raises rather than
  resolving quietly, because the old names asserted a single soundness ordering that does
  not exist.
- **`Mode.f64` no longer folds `sin(np.pi)` to `0`.** Exactly `0` in mathematics,
  `1.2246467991473532e-16` in f64 — the rewrite changes what the deployed evaluator
  computes. It serves `real` and `permissive` instead. 102 rules move, nearly all of the same
  symbolic-cancellation family.
- **`Mode.real` fails closed** on an artifact with no `rules_real.json` rather than
  serving it the f64 set, which would answer a request for mathematical soundness with
  rules that are f64-exact and mathematically false.
- `f64` mode preserves what the deployed evaluator computes for every rewrite it
  applies, **not** your evaluation order: the canonical form re-associates, and IEEE-754
  addition commutes but does not associate. Deterministic, and value-preserving for
  well-conditioned expressions.

### Upgrading from 0.13.x

**What you get back changes, and mostly it gets better.** Measured on the 400-row
benchmark corpus, comparing 0.13.1's `SOUND` against each 0.14.0 mode by cost under the
serve ordering:

| mode | vs 0.13.1 `SOUND` |
|---|---|
| `real` | identical on all 400 rows |
| `f64` (the new default) | identical on 397, more complex on 3 |
| `permissive` | simpler on 53, never worse |

The three rows where the default gives up ground are rewrites that are true over ℝ but
not realised in f64 — `asinh(sinh t) → t`, `cos(asin(sin x)) → |cos x|`,
`pow(inf, x) → exp(inf·x)`. They did not disappear; they moved to `real` and `permissive`,
which is the whole point of the split. If you were relying on them, ask for
`mode='real'` and you are back where you were, exactly.

**Expressions come back in a different dialect.** `simplify` now answers in the form it
was given (see the conversion section below); 0.13.x returned the tagged form
regardless. The expression is the same expression — cost is unchanged on every row above
— but the tokens differ, so any test pinning literal output needs regenerating.

**Everything deprecated is removed in this release.** There is no alias cycle: a retired
name raises rather than warning, so the migration is mechanical and complete.

| removed | use instead |
|---|---|
| `Mode.SOUND`, `Mode.LOSSY`, `'sound'`, `'lossy'` | `Mode.f64`, `Mode.permissive` |
| `SimpliPyEngine.parse` | `read_infix` |
| `simplify(..., form=…)` | convert first: `simplify(to_tagged(x))` |
| `simplify(..., node_budget=)` | `max_passes=` |
| `SimpliPyEngine.load(path=)` | `load(engine=)` |
| `normalize_skeleton`, `normalize_expression` | `to_skeleton`, `to_expression` (now require `engine=`) |
| `masking.mask_values_keep_structure` | `masking.mask_fittable` |
| `utils.substitude_constants` | `utils.substitute_constants` |
| `utils.numbers_to_constant` | `explicit_constant_placeholders` |

**BREAKING for downstream packages: this is a hard cut.** New `flash-ansr` and `srbf`
releases require simplipy >= 0.14.0. There is no compatibility shim, as with the last
artifact-format break: the rule sets, the artifact layout and the returned dialect all
move together, and a shim would have to lie about at least one of them. Pin the pair.

### Removed — the superseded `acj-2-1` and `acj-3-2` engines

Installs no longer bundle them; `acj-4-3` is the shipped engine. Both remain on the
Hugging Face assets repository and still load by name, so nothing published is
withdrawn — `simplipy install acj-3-2` fetches it as before.

### Removed — mining source sampling (`source_sample_per_length`)

`find_rules` no longer accepts `source_sample_per_length`, and mine configs naming it
are rejected. The option represented a length by a uniform sample of its expression
universe when complete enumeration was infeasible — but a sampled length cannot certify
"no shorter equivalent exists", and no published artifact ever used it: every shipped
mine is a complete enumeration. Phase 1 now always saturates every length, and the
provenance sidecar drops the `source_sample_per_length` parameter and the per-length
`sampled` flag (coverage is always complete or vacuous). The seeded uniform sampler
itself, `simplipy.utils.sample_expression`, remains — the verification monitor draws
its probe corpus with it.

### Changed — the literal floor is one bit, not two

Every numeric codeword was clamped at two bits, which swallowed `0`, `1`, `2` and `3`
alike — so `mu` could not tell `x**2` from `x**3` from `x**4`. The clamp is now one bit,
where it binds only at `0` and `±1`, values whose codeword could not be shorter anyway.
The exponents order again, and the sign on `±1` becomes visible, which is what the rest
of the measure already promised: a description length must tell a number from its
negation.

No rule of the 5,451 published changes direction and no printed literal spelling moves —
the serializer compares the same two codeword totals the measure does, and the floor
shifts both. `SimpliPyEngine.complexity` returns different numbers; pinned values
downstream must be re-earned.

### Changed — every grammar symbol carries its own price

The simplicity measure `mu` charged a flat 8 bits for every grammar symbol: a bag, a
`Pow`, a function head, a variable leaf and a named constant alike. A description length
is not flat, and the flatness had a visible cost — `pi - 2` and `asin(sin 2)` priced
identically, so the miner preferred the composition and shipped a whole parametrised
family of them. Those rules are true and certified; they are simply not simplifications.

`mu` now reads each symbol off a table: a variable leaf 6 bits, `Add`/`Mul`/`Pow` 3,
`pi` and `e` 4, an elementary head (`exp log abs sin cos tan rootn`) 6, an
inverse/hyperbolic head 8, an infinity or an unnamed head 8. Literal pricing is
untouched. The entries are read against the same 1/8 unit and scale with it, so
`SIMPLIPY_MU_SYM` keeps its meaning.

Two entries are fixed by rules a frequency count cannot supply: **a named constant must
cost less than the cheapest expression denoting it** (`acos(-1)` is 11 bits, `exp(1)`
is 9), and **a leaf must also name which variable it is** — the reason it is 6 and not
the 1.65 bits the census reads for the class.

Measured across the change: no rule of the shipped 5,451 changes direction, no corpus
output changes in any mode, and no literal spelling moves. What does change is the
number `SimpliPyEngine.complexity` returns — it is a different measure, so any pinned
`mu` value downstream must be re-earned. Published rule sets mined under the previous
measure still load and serve; they warn (D25/R6) that their minimality and ordering
claims were certified against the old prices, until the next mine replaces them.

### Changed — the artifact is a TRIPLE

A published asset is now six files, not four: `rules_f64.json` (the f64 set; artifacts
published before the naming ruling call it `rules.json`, and configs name their files,
so they load unchanged), `rules_real.json` and `rules_permissive.json`, plus
`config.yaml`, `mine.yaml` and the provenance sidecar. One distinct, complete rule set
per mode — no base plus overlays, so what is loaded is what is served. The triple is the
unit of mining, pinning and distribution; rules licensed in no mode are recorded in the
drop census rather than silently absent. `find-rules` derives the sibling file names
from `-o` with the `_f64` marker replaced, so `-o rules_f64.json` writes
`rules_real.json` / `rules_permissive.json` beside it and `-o rules.json` keeps its
historic siblings.

### Published artifacts — the flagship is `acj-5-4-llm`

- **`acj-5-4-llm`** (2026-08-30, `psaegert/simplipy-assets` manifest): complete
  sources through length 5, targets through length 4, plus the LLM-augmented
  proposal lane, certified by the same gates as the mined rules. Mined as a
  triple at build 17dbf35 under the growth-argument range guard — zero judge
  timeouts across the 163,649-rule certification — and independently
  gate-verified: 138,651 rules in `rules_f64.json`, 133,294 in
  `rules_real.json`, 139,299 in `rules_permissive.json`, all three modes clean,
  permissive dominance at zero violations on the 400-row reference corpus.
  13 real-tier constant folds are relocated out of the f64 set by the gate,
  each present in the real and permissive sets. The exact mine configuration
  ships with the asset as `mine.yaml`. Pairs with simplipy >= 0.14.
- **The published benchmark is regenerated on the shipped artifact** — four
  arms (`f64`, `real`, `permissive`, SymPy 1.14.0 under a 1 s cap) on three
  declared corpora: the v25 SR prior (n = 65,536, seed 20260830), its
  mask-policy-`'all'` transform, and the SOOSE external set (n = 600). The
  made-bigger claim is scoped to what the shipped `complexity()` instrument
  measures: `f64` inflates nothing on any corpus; `real` and `permissive`
  minimize their own mode's ordering and price above the input on 23/2
  (unmasked) and 2/0 (masked) of 65,536 rows, never via an uphill step. All
  ECDF figures and the README headline panel are regenerated; the simplify
  guide carries the protocol and full tables.

### Added

- **`simplify(effort=...)`** — the search budget (ledger D39). The deterministic chain
  runs unchanged to its fixpoint; a bounded exploration phase then tries expansion moves
  strict descent refuses (distribute, pow-expand), each candidate re-descended through
  the same certified machinery, accepted only strictly below the incumbent in the serve
  ordering. `effort=0` is byte-identical to the plain chain; soundness, never-worse,
  termination and idempotence hold at every budget, in all three modes. The default is
  `simplipy.DEFAULT_EFFORT` = 4, measured on the 65,536-row acceptance panel: budget 4
  strictly improves 3.18% of rows with zero regressions and captures every win budget
  64 finds, at +18% median per-row cost.

- `complexity(..., mode=...)` / FFI `rule_mode=` — the pricing instruments are
  **route-parameterized**. `complexity()` now parses its argument through the same
  route the named mode's chain descends from (fold-at-parse for `f64`/`permissive`,
  fold-free for `real`) instead of a single fold-free route for all modes. Pricing a
  chain's output with another route's instrument is how the benchmark's ten "μ-ascent"
  rows arose; the theorem μ(simplify(e)) ≤ μ(e) is stated — and now measured — with
  each mode priced by its own instrument.

- `SimpliPyEngine.evaluate_constants(expression)` — the explicit door to numeric folding.
  Folds maximal variable-free, slot-free subtrees; refuses a subtree carrying a
  `<constant>` (a fitted degree of freedom, not a value) and refuses a non-finite result.
  Never reached by `simplify`.
- `verify_ruleset(..., mode=...)` and `verify_triple(...)` — cleanliness is now **per
  mode**. `atanh(tanh t) → t` is exactly what belongs in `rules_real.json` and a defect in
  `rules.json`; a single bucket count cannot say that. Omitting `mode` keeps the
  pre-triple meaning.
- `simplify(..., max_passes=)` replaces `node_budget=`, which never counted nodes. **The
  old name is removed**; the bound is the number of outer rewrite passes, which is what
  the new name says.
- **`SimpliPyEngine.to_realization(expr)` — the fourth notation.** The same expression
  with each operator spelled as the callable it runs (`sin` → `simplipy.operators.sin`),
  and reversible like the other three: `to_prefix`, `to_infix` and `to_tagged` all accept
  realization-spelled input. Operators realized as themselves (`+`, `-`, `*`) are
  identical in both spellings, so only dotted realizations ever need detecting.
  - Reading realization notation back requires the engine's realization map to be
    injective. Two operators may legally share one realization, and then the spelling
    names both; that refuses loudly rather than picking one. Writing never refuses.
- **`SimpliPyEngine.as_code(expr)` and `as_callable(expr)` — the compile pipeline in one
  step.** `as_callable` replaces the `prefix_to_infix(realization=True)` → `codify` →
  `code_to_lambda` dance for the common case, binding the result to this engine's own
  namespace. `expression_variables(expr)` exposes the signature it binds, in order of
  first appearance.
  - Named `as_`, not `to_`, and deliberately: every `to_*` in this API is a notation that
    converts back, and a code object has no syntax to recover. `to_lambda` would advertise
    a round-trip that cannot exist. Compiling also runs through `compile()`, so the trust
    rules apply where the notation family's do not.
- `simplipy.DEFAULT_ENGINE` and `simplipy.DEFAULT_ENGINE_REVISION` — the artifact this
  version was built and tested against. `SimpliPyEngine.load()` with no argument resolves
  it and says which it took. The pin lives in the package rather than the hosted manifest,
  so what you get by default is answerable offline and does not change under you without a
  simplipy release. The revision is what separates one name across versions: `acj-4` mined
  under two different judges is legitimately `acj-4` both times, and only (name, revision)
  names one artifact.
- `SimpliPyEngine.load(engine=...)` replaces `load(path=...)`, which named neither a path
  nor a version. **The old keyword is removed.**
- **`simplipy.utils.substitude_constants` is removed** (the historic misspelling, warning
  since 0.13.0); use `substitute_constants`.

### Security — the trust model covers the whole dotted path

A realization is resolved by attribute traversal, and a module's attributes include
every module it imported. The trust check decided only the leading component, so a
config that named a trusted root could walk past it: `simplipy.engine.os.system` loaded
without an `UntrustedModuleError` and called `os.system` at first evaluation. The reach
was general rather than incidental — a walk of the attribute graph from the four default
roots lands on 53 distinct top-level packages within three hops, `os`, `ctypes`,
`importlib` and `builtins` among them.

The allowlist now applies to every hop. Any component that resolves to a module is held
to the same trusted set as the root, dunder components are refused, and a realization
that names a module rather than a callable is refused. `math`, `np`, `scipy` and
`simplipy` remain trusted by default; `trusted_modules=` and `SIMPLIPY_TRUSTED_MODULES`
remain the only ways to widen that, and they widen the whole path.

A realization spelled as a **bare name** is now an allowlist too. It imports nothing,
which is why it used to pass on shape alone, but it is still interpolated into generated
source and resolved against Python's own builtins — where `eval`, `exec`, `compile`,
`open` and `__import__` sit beside `abs`. The accepted names are `abs`, `round`, `min`,
`max`, `sum`, `pow`, `divmod`, `float`, `int`, `bool` and `len`, alongside the bare
operator symbols; anything else refuses at load.

This affects any deployment that builds an engine from a config it did not write — a
Hugging Face asset loaded by name, a colleague's file, a cloned repository. Shipped
configs, the legacy vocabulary and every realization the deployed evaluation path uses
are unchanged; upgrade to 0.14.0 to get the check.

### Fixed

- Sort promotion could sit on a single rule indefinitely. The high-precision evaluator
  runs on mpf, which carries no exponent limit, so an intermediate like `cosh(2**1.4e10)`
  is represented exactly and any later operation on it is unbounded — one observed mine
  spent 26 h inside `cos` of an argument with a binary exponent of 1.4e20, computing pi
  to that many places. `sin`/`cos`/`tan` and now `exp`/`cosh`/`sinh` refuse an ARGUMENT
  whose binary exponent exceeds `PERIODIC_ARG_EXP_LIMIT` (4096). The check is on the
  argument, not the result: magnitude alone is cheap, and bounding the result would
  refuse `exp <constant>` and `cosh <constant>`, both sound.

  The refusal is fail-closed — the evaluator raises, the caller reads EVAL-ERR, and a
  rule is never certified on an evaluation that did not finish. Over the 31,315 rules of
  a 163,002-rule corpus that reach the ladder: **0 timeouts at a 15 s per-rule deadline**
  (whole set, 52.1 min; 66 rules take 1–15 s, the slowest 5.71 s), and the guard fires on
  1,225 rules — 3.91% of those reaching the ladder, 0.752% of the corpus — spread over
  all six functions. Their shape is a nested growth chain (`asin exp cosh <constant>`,
  `sin acos cos cosh ?0`), 77.5% ground tier and 22.5% `?cf` on a 5,000-rule sample.

  `PeriodicRangeRefusal` remains as an alias of the new `RangeRefusal`.

- The nonzero-a.e. certificate for cleared quotients dropped a side condition: the
  clearing identity Z(t) = Z(numerator) presumes the cleared **denominator** is finite
  almost everywhere. Without it, `inv(inf + h)` — identically zero in f64 — certified
  as nonzero-a.e., licensing a per-factor sign move that flips the sign of infinity on
  a positive-measure set (one judge VIOLATION in 65,536 benchmark rows). The
  denominator now goes through the existing finiteness authority (`isn`); both
  regression probes refuse.

- `Mode.permissive` at nonzero `effort` could return a costlier serve form than
  `Mode.f64` on the same input: exploration is mode-dependent, so the permissive
  arbitration's two candidates (folded/unfolded permissive constructions) no longer
  contained every f64 endpoint once `DEFAULT_EFFORT` became 4. The arbitration now
  admits the default-mode result as a third candidate, so permissive-dominates-f64 is a
  property of the construction rather than an empirical gate observation. Ties keep
  the permissive-own winner.

- The infix reader bound a minus after a power operator outside the power: `2^-3`
  parsed as `-(2^3)` and returned **-8** where the value is 0.125, `sin(x0)^-1` lost
  its reciprocal entirely, and every `x^-c` collapsed to `-(x^c)`. The minus after
  `^`/`**` is the exponent's sign -- positional, not precedential, exactly Python's
  own grammar (`-x**2` is still `-(x**2)`) -- and the reader now binds it to the
  exponent, including chained signs (`x^--2`) and exponent power chains (`x^-2**3`
  is `x^(-(2**3))`). Only expressions read from infix with a negative exponent were
  affected; prefix input and every mined artifact are untouched.

- The `atanh(tanh(t))` → `t` collapse carried a band of |t| ≤ 18, derived from where
  f64 `tanh` attains ±1 (18.99). But `tanh` fails by **compression** long before it
  saturates -- it loses ~2|t|/ln2 bits of the argument approaching ±1 -- and under the
  same realised 8-ULP criterion every rule is held to, the roundtrip first breaches at
  |t| ≈ 2.7 (libm-dependent; by |t| = 18 the drift is ~0.03-0.06 *absolute*). The band
  is now 2, the last magnitude ceiling on the safe side of every measured breach. The
  overflow-derived bands (`log∘exp` 709, `sinh`/`cosh` pairs 710) are a different
  failure mode and keep their attainment thresholds.

- The same re-derivation exposed the sibling holes at the other end: the collapse
  bands guarded only against **overflow**, but compression near a flat point of the
  inner function eats the low end -- `log(exp(t))` breaches the realised bar for all
  |t| below ~0.0625 and `acosh(cosh(t))` below ~0.25 (`exp` rounds a small argument
  into the neighbourhood of 1; `cosh` is quadratically flat at 0). Both collapses now
  require a *proven* magnitude floor of 1 beside their ceiling, where the measured
  worst is 0 and 1 ULP; `asinh(sinh(t))` measured clean over its whole band and is
  unchanged. The two shipped f64 rules of this shape (`log exp <constant>`,
  `acosh cosh <constant>`) re-tier to `real` with the 0.14.0 artifact.

- The literal codeword floor (one bit, ruled 2026-08-22) reached `mu_rat` but not the
  beyond-i128 leaf pricer, whose clamp stayed at two bits -- so `1e-39` priced *above*
  `3e-38`: the simpler mantissa at the adjacent magnitude, upcharged for crossing the
  representation line. The floor is now one constant shared by both sides, the seam
  (`1e-40` = `1/1⟨40 zeros⟩`) stays closed, and the measure fingerprint gained a
  beyond-i128 probe so this class of drift can never again leave the digest unchanged.
  Measured over all 16,018 shipped rules: zero prices move, zero directions flip.

- The judge quantified a source-side `<constant>` over |c| ≤ 5 while the miner draws its
  constants from [1e-3, 1e3] and sweeps magnitudes to 500 — so `forall c over the reals`
  was checked on one decade. The battery now spans ±1e-3 to ±1e4. Reaching there needed
  two fixes to the exists-witness search, both cases of answering at a point where the
  answer cannot be read: a witness fitted where the deployed left-hand side has
  **saturated** solves an *interval*, not a point, and its edge is not a witness (an
  identity was convicted on the difference); and the fixed search grid thins out exactly
  where the deployed algebra overflows, so the constant under test is now tried directly.
  Rules true over the reals but unrealised at large constants move from `f64` to `real`,
  where they belong — 107 of the 5,451 currently published, with no rule newly rejected.

- The judge's realisation axis was measured with a `1e-9` tolerance carrying an absolute
  floor — roughly 10⁷ ULP — so a rule the contract convicts could be stamped "realised"
  and admitted to the default rule set. It now uses a bound derived from this platform's
  measured libm error of the platform's own math library, and the two questions it used
  to conflate —
  "does f64 compute this?" and "has the deployed algebra diverged?" — have separate
  comparisons.
- The contract lane's second precision rung was a fixed dps 120, which confirms nothing
  when intermediates are 10²¹⁷: both rungs are swamped alike and agree on a manufactured
  verdict. It is now sized from the largest intermediate actually seen. This was killing
  `log(cosh(25t) + sinh(25t)) = 25t`, which is exactly true.
- The contract's finite comparison measured a RELATIVE separation, normalised by the
  larger of the two sides. Against a side that is exactly zero that normaliser is the
  other side, so the reading was `1.0` at every precision — it could not respond to
  precision at all, which is the only question the comparison asks, and an identity whose
  two sides agree exactly was convicted for its own arithmetic residue. The separation is
  now measured absolutely, against a single scale held fixed across the precision rungs.
  This was killing true identities of the form `log(eᵃ · e⁻ᵃ) = 0`, and the two judges
  could disagree about them, because the verdict turned on whether the arbitrary-precision
  library happened to return an exact zero at the upper rung.
- The gate had no verdict at all for the saturation family — a bounded function claimed to
  reach its limit. `tanh(cosh(10))` differs from 1 by 2e-9566, far inside the precision
  band at any working precision anyone can afford, so a numeric comparison can only
  abstain however deep it is taken, and 174 shipped rules were carried without the gate
  ever forming an opinion on them. A bounded function does not attain its limit at a finite
  argument, and the comparison now settles those rules from that fact rather than from a
  measurement — confirming every one of them into the tier it already ships in. The gate
  now returns a verdict for every rule in the artifact.
- The judge could not check the constants the miner explores. Its source-constant battery
  reached `|c| ≤ 5` plus π and e while the miner draws from `[10⁻³, 10³]`, and widened to
  ±10⁴ it convicted 97 of the 343 constant-carrying rules in the shipped set — every one
  of them true by construction, and every one an artifact of the instrument at a magnitude
  it could not resolve. Six distinct causes, each fixed:
  - The exists-witness search accepted `0` as the witness for a target of `4.4e-589`. Its
    noise floor is an absolute band, there to stop a cancellation zero from being chased
    as a witness; a value that is simply small — one power and no subtraction — fell
    under it too. The floor is now consulted only where the target actually shrinks as
    the working precision rises, which is what separates a residue from a value.
  - That search also stopped eight decades short of the f64 ceiling, so a witness of
    `7.6e300` had no bracket and the rule was reported as having none.
  - A precision rung that read one side as exactly zero, while the other was not, was
    still used to date the gap's decay. It cannot: the reading is pinned at the
    instrument's rail. The comparison now measures the rate between two rungs that both
    saw something, which can only ever move a conviction to a stricter pair of rungs.
  - The rungs were sized only from the LARGEST intermediate. A small one costs the same:
    `cos x = 1 - x²/2 + …` at `x = e⁻³⁰⁰` puts the whole content of the answer 261 digits
    below the 1 it sits beside, so three rungs in a row read exactly `1.0` and `acos` of
    that is `0`. Precision is now sized from both ends.
  - The evaluator's own symbolic-cancellation floor was absolute, so `sin(e⁻³⁰⁰)` — which
    is exactly the size its argument predicts, with nothing cancelled — read as `0` at two
    precisions and as `5.1e-131` at a third. The floor is now relative to what the
    argument predicts, and still fires for every cancellation it was written for.
  - Past its own precision ceiling the judge convicted rather than abstaining, which is
    the opposite of what that ceiling documents. Running out of evidence is a statement
    about the judge, so the conviction is withheld — except against a value the rule
    itself spells, which is exact at every precision and cannot be a casualty of running
    short of it.
- Two shipped rules of the form `exp(exp(sinh c)) = 1` were certified as true over ℝ. They
  are not: at `c = -6` the left side differs from 1 by `2.5e-88`. They are exact in f64 and
  stay in the same rule set; what changes is that the gate now says which of the two it is.
- The miner asked one question where the judge asks two. Its row gate skips the contract
  entirely wherever the deployed f64 sides "agree", and the bar for that was the band that
  answers a different question — whether the deployed algebra has *diverged*, which is
  deliberately loose because a few ULP of rounding is not a divergence. As a pre-screen it
  is roughly 10⁷ ULP, and below 1 a pure absolute floor: it reads `e^sinh(-5) = 0` as an
  equality when the sides are `5.9e-33` and `0.0`, so a rule that is false could be mined
  without its truth ever being asked. The two questions now have the two comparisons the
  judge already used — realisation in ULP for the pre-screen, the structural band for
  divergence. Measured: 8 of 102 `real`-tier rules in a reference mine have a point that
  now reaches the contract, up to 4.9 million ULP apart, and every mined artifact is
  byte-for-byte what it was.
- The D15 diagonal lane never received the precision coupling its commit claimed, and fed
  the same measure that convicts.


### Changed — CONVERSION and SIMPLIFICATION are separate (owner rulings, 2026-08-18)

- **`to_infix` / `to_prefix` / `to_tagged` are now PURE, SYNTACTIC conversions.** They
  re-notate and do nothing else: no canonical state is built, no rewrite rule fires,
  nothing is collected, folded, reordered or re-spelled, and the answer no longer
  depends on the engine ARTIFACT. `to_prefix(['+','x0','x0'])` is `['+','x0','x0']`
  (it was `['*','2','x0']`), and `to_prefix(['pow','abs','x0','2'])` keeps the `abs`
  under every ruleset (it was dropped under acj-4-3 and kept under acj-2-1 — a
  *conversion* whose answer depended on which artifact was loaded).
  - Why: `to_prefix(x)` was EXACTLY `simplify(x, form='explicit')` on every probe, so
    the trio added no capability over `simplify(form=…)`, and there was no
    spelling-preserving prefix↔tagged path at all. Measured before the change, on 400
    raw corpus rows: **not one** of the nine in/out directions preserved the input's
    spelling on more than 1/400 rows, and no 2-cycle or 3-cycle was an identity.
  - What is guaranteed now, per direction (full table in the design note): `P→P`,
    `T→T`, `I→I` and `T→P→T` are **byte identity** (400/400); `P→T→P` is identity **up
    to associativity** of same-polarity `+`/`*` runs, with bag member ORDER preserved;
    `P→I→P` is identity **up to the infix language's missing atoms** (`inv X ↦ / 1 X`,
    `p/q ↦ / p q`). Every direction and every cycle preserves the canonical state
    exactly: `simplify(to_X(y))` is byte-identical whichever dialect `y` arrived in
    (400/400).
- **`simplify` is DIALECT-PRESERVING: it answers in the form it was given.** `str` in →
  `str` out; explicit binary prefix in → explicit binary prefix out; tagged in → tagged
  out. Container mirroring (`tuple`/`ndarray`) is unchanged, and the simplification
  itself — every canonical answer — is unchanged.
  - **This closes the tagged leak.** `simplify(['*','2','atan','x0'])` returned
    `['<mul>','2','atan','x0','</mul>']`; it now returns `['*','2','atan','x0']`. A
    tag-free token input can never come back carrying tags.
  - Dialect detection: a token sequence is TAGGED iff it carries a bag delimiter or an
    inverse-section marker (`<add> </add> <mul> </mul> <sub> <div>`), else EXPLICIT.
- **`simplify(..., form=…)` is REMOVED.** `simplify` answers in the form it was given;
  choosing the notation is the conversion layer's job, not a simplification argument.
  - The migration is **convert first, then simplify**: `simplify(x, form='tagged')` →
    `simplify(to_tagged(x))`, `form='explicit'` → `simplify(to_prefix(x))`,
    `form='infix'` → `simplify(to_infix(x))`. All three are byte-identical to the
    parameter on every corpus row measured (400/400).
  - The reverse composition `to_tagged(simplify(x))` is **not** the migration: the
    tagged and explicit canonical emitters carry different sign/inverse/literal
    doctrine, so it agrees with `form='tagged'` on only 186/400 rows.
- **`prefix_to_infix` renders two things correctly that it rendered ambiguously.** A
  compound numeric leaf (`7/3`, `-1/3`) now carries the precedence its spelling embeds,
  so `* 7/3 7/3` renders `7/3 * (7/3)` instead of `7/3 * 7/3` (which re-read as
  `((7/3)*7)/3`); and a config-declared binary operator outside `+ - * / **` renders as
  a function call, like `rootn` and `pow`, instead of `x0 hypot2 x1` — a spelling this
  engine's own reader could not read back.
- **`read_infix` is unchanged**, and its exclusive capability is now clearly the
  vocabulary TOLERANCE: `read_infix('sqrt(x0)')` is `['sqrt','x0']` where every
  conversion refuses. The canonicalising contrast it used to draw with `to_prefix` is
  now with `simplify`.
- **`to_skeleton` / `to_expression` are unchanged in contract**; internally they now ask
  for canonicalisation explicitly (`simplify(to_prefix(x))`) instead of relying on
  `to_prefix` to canonicalise. `masking.mask`'s collect stage likewise dropped its
  `form=` projection — dialect preservation is `simplify`'s own contract now.

### Changed — `normalize_skeleton`/`normalize_expression` are now `to_skeleton`/`to_expression`
- **The name states the OUTPUT form**, matching the engine's
  `to_infix`/`to_prefix`/`to_tagged`. The old names hid what the skeleton form
  actually does: it masks **every** numeric literal, `pow` exponents and `rootn`
  indices included, so `pow(x0, 3)` becomes `pow(x0, <constant>)` — a family no
  constant optimizer can fit. (`engine.mask(expr, 'fittable')` is the kind that
  keeps the structural literals.) **The old spellings are removed**, from both
  `simplipy.normalization` and the package root. They are out of the declared `__all__`, the same
  treatment `masking.mask_values_keep_structure` got.
- **`normalize_variable_token` keeps its name.** It is a single-TOKEN helper that
  answers `(token, is_variable)`; it produces no form, reads no dialect and needs
  no engine, so the `to_*` family naming would misdescribe it.

### Changed — both forms accept all three dialects, and the answer no longer depends on which
- **They take an infix `str`, an explicit binary-prefix token sequence or a tagged
  token sequence, and return the dialect they were given** (`str`→`str`,
  prefix→prefix, tagged→tagged), detected exactly as the conversion API detects it
  (type first, then a liberal read of the tokens). Masking names an abstraction
  level, not an output dialect, so neither function may silently re-spell.
- **The CONTENT is now dialect-invariant, and that is a correctness fix, not a
  convenience.** Masking the caller's own spelling one token at a time counted a
  different number of `<constant>` slots for the same expression depending on the
  dialect it arrived in, in both directions — `-x0*x1` is `neg * x0 x1` (0
  constants) but `<mul> -1 x0 x1 </mul>` (1), and `-1/x0` is `/ -1 x0` (1) but
  `neg inv x0` (0). Measured at **35/400 rows (8.8%)** of the repo corpus, and at
  the same 35/400 for a role-aware `mask(mask_all)` applied per dialect — role
  awareness does not help, because the disagreement is about what the SITES ARE.
  A skeleton is what downstream holdout and decontamination key on, so a
  dialect-dependent answer let a tagged-era pipeline silently fail to match
  explicit-era keys and report clean. Every input is now canonicalised into the
  internal AC state first, abstracted on the canonical **explicit binary prefix**
  (where a sign and a reciprocal stay structure instead of being folded into a
  literal), and only then rendered into the caller's dialect.
- **BREAKING: both now require an `engine`.** That is the cost of the guarantee;
  there is deliberately no engine-free path and no fall-back to the old positional
  pass, because silently returning a dialect-dependent key is the failure the
  change exists to prevent. Callers that passed a token list now pass
  `to_skeleton(tokens, engine)`.
- Consequences of canonicalising: the expression form carries the state's **exact
  rationals** (`2.5` re-spells as `5/2`; the skeleton is unaffected, both halves
  being one degree of freedom); undeclared vocabulary and reserved numeric
  spellings (`inf`, `1_000`) now raise `ValueError` at the core's token grammar
  instead of passing through; `<c>` folds to `<constant>` in **both** forms.

### Changed — `to_skeleton` is not a third masking path
- It is documented and implemented as exactly `to_expression` followed by the
  ratified front door `engine.mask(..., policy='all')`. Two consequences are now
  on the record: the collect stage re-runs the engine, so simplification **rules
  fire** and a skeleton depends on the engine ARTIFACT and not only on the
  expression; and `engine.mask(to_expression(e, engine), 'all', collect=False)` is
  the escape for a rules-free, strictly positional abstraction of the canonical
  spelling.
### Removed — `simplipy.utils.numbers_to_constant`

- **The standalone helper is gone** (owner ruling 2026-08-18: the removal lands
  in 0.14.0 and the downstream packages adapt). It warned from 0.12.0 and the
  0.13.0 changelog already named 0.14.0 as its removal version; until now it was
  still present, so the two surfaces disagreed. Replacement:
  `simplipy.masking.mask(tokens, engine, policy)` or the `engine.mask` front
  door, with the policy that states the intent — `mask_all` for the legacy
  mask-everything behaviour, `mask_fittable` for what a constant optimizer can
  actually fit; `collect=False` gives the positional 1:1 substitution the helper
  approximated.
- **The replacement is deliberately not a drop-in**, because the helper was
  wrong in ways a rename would have preserved. It classified by a bare `float()`
  probe, so it minted a finite-by-doctrine `<constant>` for the reserved
  spellings `inf`/`nan` (any case or sign) and `1_000` — masking a non-finite
  literal into a placeholder that is finite by contract. Those now raise
  `ValueError` at the masking boundary. The same probe raised on
  `np.pi`/`np.e` and on the AC core's exact fraction `1/3`, so the helper walked
  past all three; they are masking sites and `mask_all` masks them. And the
  helper was role-blind and structure-blind: it never looked at the expression,
  so a malformed sequence was rewritten positionally instead of refused.
- **Two names that share a word with it did NOT move.**
  `explicit_constant_placeholders(..., convert_numbers_to_constant=)` is a
  keyword on a mechanical code-generation helper, still accepted with both
  values, and shipped downstream call sites pass it explicitly.
  `read_infix(..., mask_numbers=True)` keeps
  masking through the internal Rust port, which is not reachable by name from
  Python and is unchanged.
### Changed — rule dedup compares the INTERNAL FORM, not the spelling
- **`simplipy.utils.deduplicate_rules` now keys on the engine's internal form**
  (owner ruling 2026-08-18: *"instead of comparing spelling, compare internal
  form"*). Two rules are one rule when the loader translates their sources into the
  same pattern. Keyed on the token spelling, `* (-1) asin _0` and `* asin _0 (-1)`
  shipped as two entries for one pattern, and the second could never fire — the
  first already owns every subject it matches.
- **`engine=` is a REQUIRED, keyword-only argument.** There is no engine-free
  reading of "the same rule": the internal form is defined by the operator table.
  It is keyword-only on purpose — a positional third argument would silently have
  swallowed an existing caller's positional `verbose`. Existing calls fail loudly
  with a `TypeError` naming the argument.
- **The key is deliberately not `to_prefix`/`to_tagged`.** Those run the rewrite
  pass under the *loaded ruleset* and the full certificate context: on an engine
  holding acj-4-3, `to_prefix(['*','(-1)','asin','_0'])` returns `asin neg _0` —
  that rule's own target — while on a rules-less engine over the same vocabulary it
  returns `neg asin _0`. A dedup key that depends on the ruleset being
  deduplicated is circular. The new key is a pure function of the operator table.
- **Measured effect on the shipped artifacts, at LOAD (the files on disk are
  untouched and byte-identical):** acj-2-1 26 → 24 served (−2, 7.7%), acj-3-2
  106 → 96 (−10, 9.4%), acj-4-3 6,594 → 5,545 (−1,049, 15.9%). Every collision is
  DEAD — same lhs *and* same rhs internal form, zero shadowed — so first-match-wins
  meant the dropped copies could never fire.
- **Behaviour is unmoved**, measured four ways per artifact: zero byte-diffs over
  the 443 corpus rows the operator set accepts (`benchmarks/corpus/raw_skeletons*`,
  800 rows), zero diffs on a direct probe of every dropped rule's own LHS (raw and
  variable-instantiated), the corpus-invariance gate's complexity, idempotence,
  permutation and twin counts all identical (acj-4-3: 294,730,680 / 0 / 0 / 8), and a
  differential fuzz of
  the served engine against one force-fed the raw artifact over **65,536** random
  expressions (lengths 3–11 over the mine alphabet): **0 differences, 0 raised**.
- **Pins that move.** The translation census counts served patterns, so it drops
  with the duplicates: 6602/0/0 → **5553/0/0** (rows, complexity and twins unmoved).
  The gate and the loader move together, so no commit in between leaves the gate
  green against a number the code no longer produces.
  The in-suite twin of that pin (`tests/test_licence_registry.py`) is updated with
  the evidence.

### Changed — `parse` is now `read_infix` (the name states the contract)
- **`SimpliPyEngine.parse` is renamed to `SimpliPyEngine.read_infix`, and `parse` is
  removed.** The rename is not cosmetic: this reader is the only public entry that is
  **tolerant of unknown vocabulary** (an undeclared `sqrt(...)` survives as a bare
  leaf, which `to_prefix`/`to_infix`/`to_tagged`/`simplify` all refuse) and
  **preserves the input's spelling** — it never enters the canonical state, so
  `read_infix('x0+x0')` is `+ x0 x0` where `to_prefix('x0+x0')` is `* 2 x0`. Use
  `read_infix` when the input's own spelling is the data, `to_prefix` when the
  canonical spelling is. `mask_numbers=` and `convert_expression=` are unchanged
  on both names.

### Changed — the masking redundancy is cleared (one policy, one name)
- **`simplipy.masking.mask_values_keep_structure` is removed**; use `mask_fittable`. They were never two policies: the older name was this module's
  original spelling, `mask_fittable` landed as a rename, and the two have been the
  same function object ever since. There is no third semantics — "mask the values,
  keep the structure" partitions `Role` into `{COEFFICIENT, ADDEND, VALUE}` and
  `{EXPONENT, ROOT_INDEX}`, exactly what `mask_fittable` applies; the names state
  one criterion from two sides (what is kept, and why it must be). Because they were
  one object, nothing about the behaviour changes — only the name you call it by.
- **`simplipy.masking.__all__` now declares only the live surface** (`Role`,
  `literal_sites`, `mask`, `mask_all`, `mask_fittable`). The deprecated spelling is
  no longer advertised or bound by `from simplipy.masking import *`.

### Added — `engine.mask(..., collect=False)`
- **The front door reaches the toolkit's full behaviour.** `engine.mask` had no
  `collect` escape while `masking.mask` did, so the raw positional substitution was
  unreachable from the engine. The default (`collect=True`) is unchanged and stays
  byte-identical. The docstring now states what the collect stage does and costs:
  it re-runs the engine, which is what enforces one `<constant>` per degree of
  freedom (`2*x0/3` is one free value, not two), and therefore fires rules and
  re-imposes canonical order — terms may be re-ordered relative to the input
  spelling.

### Changed — `is_valid` accepts and verifies all three forms
- **`is_valid` now reads infix strings and the tagged dialect**, not only explicit
  binary prefix, detecting the form exactly as the conversion API does (type first,
  then dialect). This closes a real seam rather than adding convenience: the tagged
  form is the engine's own serialization *and* the v24 target format, and `is_valid`
  answered `False` for every well-formed tagged sequence — a wrong verdict, not a
  refusal. An infix `str` fared worse: the old call walked the string's
  **characters**, so `is_valid('x0+x0')` was `False`.
- Explicit binary-prefix verdicts are **unchanged**, including the sharp edges
  (`[]` is `False`; undeclared vocabulary is `False`; `tuple` input keeps working).
  Malformed input in any form is still a `False` verdict, never an exception; only a
  non-expression type raises `TypeError`. The verbose diagnostic that used to say
  "is_valid reads the explicit binary-prefix dialect" now explains a *malformed
  tagged* sequence instead, which is the only case that still reaches it.
- Note: `is_valid` verifies against the engine's vocabulary while `read_infix`
  tolerates it, so `read_infix('sqrt(x0)')` reads and `is_valid('sqrt(x0)')` is
  `False` when no `sqrt` is declared.

### Changed — an engine that ends up with no rules says so
- **`from_config` (and `load`) now warn on the resulting STATE, not just on a
  missing file.** An engine with zero simplification rules is fully functional — it
  parses, evaluates and returns canonical output — and simply never rewrites
  anything, so the old silence turned a broken setup into merely disappointing
  numbers. All three causes now warn: an unresolvable configured rules path, a
  config with no `rules` key, and a rules file that loads and contains nothing.
  The warning is **non-fatal**, as ruled.
- When the cause is an unresolvable path the message names **both** the literal
  config value and the resolved absolute path (previously only an unnormalized
  join, so `./missing.json` printed as `/dir/./missing.json`).
- Direct `SimpliPyEngine(operators=..., rules=[])` construction stays silent: the
  caller asked for a bare engine explicitly, and it is the sanctioned idiom for
  mining, pickling and testing.

## 0.13.1 — 2026-08-17

### Added — the fair-benchmark results
- The pre-registered three-corpus benchmark results are published: headline
  table and full ECDF panels in the simplify guide, summary in the README.
  Across 131,600 scored rows the sound engine never inflates an expression
  (0.00% made-bigger); paired median speedup vs SymPy's `simplify` is
  ~600–800× at 92–130 µs per row.

## 0.13.0 — 2026-08-17

### Added — the declared public API (`__all__`) and the compatibility policy
- **The public surface is now declared**: `__all__` at the package root and in
  `simplipy.utils` / `simplipy.io` / `simplipy.asset_manager` /
  `simplipy.engine` / `simplipy.mining`. What is declared is what the
  compatibility policy stabilizes; reachable-but-undeclared names carry no
  promise (being rendered by the docs generator never was one, and the API
  reference now renders only the declared surface). `from simplipy import *`
  no longer injects eight submodules into the caller's namespace — one of
  which (`simplipy.io`) shadowed the stdlib's `io`.
- **`simplipy.mining.RuleMiner`** — mining/certification now lives in its own
  module as an explicit object over an engine (`RuleMiner(engine)`).
  `SimpliPyEngine.find_rules` / `certify_rules` remain as delegators with
  identical signatures and semantics: the split is architectural (gated
  byte-identical mines), not a surface change. One miner per process, as
  before.
- **Deprecated**: `substitude_constants` (the historic misspelling) now warns;
  use `substitute_constants`. Removal not before 0.15.0. `numbers_to_constant`
  (warning since 0.12.0) is removed in 0.14.0; use
  `explicit_constant_placeholders`.

### Changed — clean-cut config rulings (D12, owner-ratified)
- **`pow1`/`pow_1` join `RETIRED_OPERATOR_TOKENS`.** A config declaring them was
  accepted while the parser unconditionally rewrote their tokens away — one engine,
  two answers for one expression. Artifact loading refuses them like the rest of the
  retired family; the parser contract (legacy *input* rewriting) is untouched.
- **The `inverse` operator-spec key is optional.** Nothing has read it since the
  relic layer's deletion — a required key that was pure ceremony made every wild
  config fail construction. Declared values are still accepted silently; the shipped
  acj configs no longer carry the key (their digests updated with the republish).
- **The torch shim stays** — its deletion was gated on a published-source grep, and
  the grep refused it: flash-ansr compiles simplipy expressions into torch-callable
  functions and feeds tensors through the operator realizations
  (`flash_ansr/utils/metrics.py`). The `[torch]` extra remains a documented opt-in.

### Removed — Python 3.11 support (floor moves to 3.12)
- **`requires-python` is `>=3.12`** (D35, owner-resolved). numpy 2.5 and scipy 1.18 both
  declare `>=3.12`, so a 3.11 environment silently resolved an *older* numeric stack
  than every developer and every other CI job — a real difference in the f64 algebra
  the equivalence contract is stated against, and it was invisible until CI began
  printing the resolved stack per job. The floor now matches the dependency reality;
  the abi3 wheel tag moves to `cp312`. Standing version policy recorded with the
  ruling: new-but-stable versions, reasoned floors, no unmeasured ceilings.

### Fixed — SOUND `simplify` no longer fabricates values it cannot certify
- **Fabricated `nan` for finite contract values.** With a `±1` base and an exponent whose
  enclosure reaches an infinity, the interval magnitude-step arms omitted the genuinely
  attained value 1 (`pow(t, ±inf) = 1` at `|t| = 1`), so `x0 + (-1)**(3/sin(np.pi))`
  returned `nan` where the contract value is `x0 + 1` — and the nan escaped into any
  enclosing expression. Measured 78 fabricating rows on the 61,200-ground judged sweep;
  now 0 in both SOUND and LOSSY.
- **Fabricated definite infinities.** A base the intervals could only bracket
  (`cos(np.pi)` encloses `[-1, -0.9999999999999991]`) drove the same arms to assert a
  definite `+inf` where the true value is 1: `x0 + pow(cos(np.pi), float("-inf"))`
  returned `inf`, annihilating `x0`. 46 rows measured; now 0 across the full
  1,560-row family sweep.
- **The `A − A → 0` certificate looked only inside a bounded box.** `finite_ae` proved
  finiteness over `[-R, R]^n` (R ≥ 64) and nothing looked outside, so
  `log(1 - 0.001*x1) - log(1 - 0.001*x1)` simplified to `0` — an expression that is nan
  for every `x1 > 1000`. A tail arm now discharges the complement (the claim covers the
  horizon failure class only; certificates lost to interval dependency loss remain out
  of scope, deliberately).
- **The interval box for odd negative exponents came back reflected** (Rust `%`
  truncates, so `k % 2 == 1` is false for negative odd k): `pow(x, -1)` over a range
  straddling zero missed its lower pole, and cancellations fired on nan-bearing
  expressions. An enclosure fuzz (every unary operator, dense grids, degenerate boxes)
  now pins the property.
- **`atanh` folds and boxes disagreed by up to 1.5e13 ulp.** The interval arm evaluated
  endpoints with Rust std's own `atanh` composition while the fold ships the system
  libm's; near ±1 the gap reaches the fourth significant digit, so the box could exclude
  the fold's own value. Both now call the same libm symbol.
- **Net recall cost, accepted deliberately:** a handful of grounds that used to fold by
  numerical luck now stay symbolic (`pow((-1), exp(40))`, `exp(-100000)`); an engine may
  over-refuse, an instrument may not convict.

### Changed — one owner for the odd-function literal sign (canonical spellings move)
- `-5 * sin(2)` and the same term built through a sum used to reach two different
  fixpoints (`-5·sin(2)` vs `5·sin(-2)`) — one value, two canonical states, decided by
  construction history. The sign placement is now priced by one shared orbit on every
  route: μ picks the cheaper spelling, exact ties resolve to the non-negative
  coefficient (`-1·sin(2)` files as `sin(-2)`; `-1·x0·sin(2)` keeps its free `-1`).
- The collector also learned the inverse: `sin(-2) + sin(2)` cancels to `0` in one pass
  (it previously needed a render/re-parse cycle).
- **Effect:** canonical spellings move on odd-function-of-literal shapes; the
  400-skeleton reference corpus is row-identical, and 30 of the previous artifact's
  rules become redundant respells (see Published artifacts).

### Changed — the complexity measure saturates instead of overflowing
- μ's linear literal schedule exhausts u64 around scale 5.5e15; both saturation
  entrances collided every larger literal at one price (`u64::MAX`), and a poisoned
  leaf overflowed the tree sums — a debug panic, or a release-mode wrap that priced a
  composite below its own parts (the direction that licenses wrong rewrites). Literals
  now price on a two-regime schedule (exact up to scale 2^32 — nine orders beyond any
  f64-derived literal — then ordered by the exponent's own digits), and accumulation
  saturates at the top, which can only refuse a rewrite, never license one.

### Changed — certification gates refuse what their judge cannot pass
- **`certify_rules` runs the same symbolic gate as the mine.** It certified
  `tanh(exp(e)) → 1` — false by a stable 1.37e-13, below any numeric tolerance — while
  the mining path killed the same pair. Accepted pairs are now re-judged and fatal
  verdicts refused, so "certified output is exactly as sound as mined rules" is earned,
  not asserted.
- **Every non-sound verdict is fatal at the gates, not only KILL.** A rule the judge
  cannot evaluate (`NO-WITNESS`, `UNSUPPORTED-SHAPE`, `JUDGE-TIMEOUT`) or cannot
  reconcile (`ENGINE-MISALIGN`, `UNRESOLVED-COVERAGE`) no longer ships with a
  certified rule's standing; the provenance sidecar records the full bucket census
  (every bucket, present even when empty). One scoped exemption: the multi-`<constant>`
  family the judge explicitly declares outside its jurisdiction (its soundness
  authority is the constant-fitting chain), recorded under its own sidecar key.
- **scipy's absence is a hard error in the miner, never a silent degradation** — it
  used to change mined artifacts with no record (a certifiable promotion read
  NO-WITNESS instead of PROMOTE purely on scipy's presence).

### Added — artifact identity: recorded environment, digests, and read-back checks
- **Every mine's provenance sidecar records its environment**: python, platform, libc,
  numpy/scipy/mpmath versions, and a `libm_fingerprint` — a digest of a fixed probe
  battery evaluated through the deployed folding path. The transcendental folds go
  through the system libm, which resolves on the running machine: measured, glibc 2.43
  is 1 ulp wrong on `cosh(acosh(2))` and mints a rule other hosts do not, so
  "byte-deterministically reproducible" is now stated — and checkable — *at the
  recorded environment*.
- **The asset manifest carries per-file `sha256` and a pinned `revision`, and both are
  enforced**: installs download at the pinned revision and verify what landed;
  resolution re-verifies the cache, so corrupting a byte of a cached `rules.json`
  raises instead of silently serving. (Manifest entries without digests remain
  permissive, so older manifests keep working.)
- **The measure fingerprint is read back at load**: an engine loading a ruleset mined
  under a different μ now warns loudly (the rules stay sound; their minimality claims
  are what a measure change voids).

### Fixed — interval and certificate hardening (mining instrumentation)
- Mixed-infinity products propagate their sign flags per-sign; a `b_mul` collapse could
  previously flip a definite sign class on expressions reaching both infinities.
- The certificate algebra widened (denominator-clearing certificates, unconditional
  coefficient splits), moving a measured set of trapped towers to cheaper forms.
- Masking policy left the mechanical helpers: `explicit_constant_placeholders` no longer
  converts digit-only tokens by default (`convert_numbers_to_constant` flipped
  `True → False` and is deprecated — masking decisions belong to `simplipy.masking`,
  the helpers stay mechanical).

### Published artifacts
- **acj-4-3 re-mined and republished** under the 0.13.0-train engine: 6,671 → 6,594
  rules. Every removed row is attributed: 60 odd-function sign-spelling twins made
  redundant by the shared sign owner (their sources still reduce natively to the same
  values), and 17 rules the widened symbolic gate refuses (`exp(pow((-10), k)) → 0`
  shipped an f64 underflow-rounding as an exact rewrite; verdict UNRESOLVED-COVERAGE).
  Minted on a correctly-rounding libm host — the previous interim mines on the dev box
  carried a `cosh(acosh(2)) → 2` rule that exists only because its glibc is 1 ulp
  wrong there — ×3 byte-identical, with the environment and `libm_fingerprint`
  recorded in the sidecar and per-file `sha256` + pinned `revision` in the manifest.
  The 400-skeleton reference corpus is row-identical under the new engine + artifact;
  `μ`-non-increase and idempotence hold 400/400.
- Interim (2026-08-11, superseded by the above): acj-4-3 6,661 → 6,671 (+10 adopted
  post-F83 rules, each judge-verified with zero pointwise exceptional-point
  disagreements).

### Docs
- **The "never longer than the input" guarantee was false and is gone.** `simplify`
  guarantees μ-non-increase, soundness, and idempotence — never an output-token bound:
  77 of 400 reference outputs are token-longer in the explicit form (71 exact μ ties,
  6 strictly μ-cheaper yet longer), and the tagged form adds bag delimiters on top
  (264/400). The pipeline page now states the μ-guarantee with the measured caveat,
  and a test pins it 400/400.
- The independent monitor's verdict vocabulary is API-visible and documented:
  `UNSCORED` (an output the judge cannot read never silently passes as OK) and the
  singular-scoped violation clause.

### Known performance
- The serve path carries a measured throughput regression (−32%/−43.7% on the
  benchmark corpora) inherited with the certificate widening: the `ac_zsn` witness
  is re-computed where the neighbouring `finite_ae` is doubly cached (4.78 ms vs
  0.19 ms). Disclosed here deliberately and scheduled for a cache, not silently
  carried.


### Security — the realization trust model
- **A config file is no longer executable input.** Operator realizations name Python
  modules, and the engine imported every root it found — which runs that module's
  top-level code, before any expression is evaluated, on every path including
  `SimpliPyEngine.load` (which fetches a config from Hugging Face) and unpickling.
  Realization module roots are now checked against an allowlist BEFORE anything is
  imported: `math`, `np`, `scipy`, `simplipy`. Anything else refuses at construction,
  naming the operator that asked for it.
- **One spelling per module: `np`, not `numpy`.** `np.pi` and `np.e` are normative token
  grammar, so `np` is the canonical form; a realization written `numpy.sin` is refused
  with a hint pointing at `np.sin`.
- **Trust is granted from outside the config, never by the config** (a `trusted_modules:`
  key in the YAML would let a hostile file authorize itself, and is ignored). Two
  surfaces: `SimpliPyEngine.from_config(path, trusted_modules=[...])` (also on `load`
  and the constructor) and the `SIMPLIPY_TRUSTED_MODULES` environment variable. The
  resolved trust travels with the engine through pickling, so a spawn worker imports
  exactly what the parent was allowed to import.
- **Each engine evaluates in its own namespace.** `code_to_lambda` bound expressions to
  `simplipy.engine`'s module globals, so one engine's imports were reachable from another
  engine's expressions, as were simplipy's own imports (`os`, `importlib`, `hashlib`, …).
  Expressions now see only the modules their engine needed plus `np`.
  **API note:** `code_to_lambda` is an instance method instead of a `staticmethod`.
  Instance calls (`engine.code_to_lambda(code)` — the documented usage) are unaffected;
  a call on the class now needs an instance.
- **Rule files are data, never code.** `verify_ruleset`'s literal reader evaluated
  rule-file tokens with `eval()`, so a hostile `rules.json` could execute Python at
  verification time. Literals now go through a total, eval-free acceptor (numbers,
  `(-N)`, `float("inf"/"-inf"/"nan")`, `np.pi`/`np.e`, `p/q`); anything else refuses
  as `UNSUPPORTED-SHAPE`. Measured against the shipped artifact: 0.00% of 18,238
  literal occurrences refused.
- **Inline `lambda` realizations are refused at load.** A config could smuggle
  arbitrary Python source into generated code through a `realization: 'lambda x: ...'`
  string; realizations must now name a callable in a trusted module.
- Scope, stated plainly: this makes a config safe to LOAD and scopes what a compiled
  expression can SEE. It does not make `codify`/`code_to_lambda` safe against a hostile
  EXPRESSION; evaluating attacker-supplied source is unsafe by construction in Python.

### Removed — `simplify(inplace=...)`
- **The `inplace` parameter of `SimpliPyEngine.simplify` is gone.** It had three
  different behaviors behind one flag (mutate the input list, raise for ndarray input,
  silently do nothing for `form='infix'`), no callers anywhere, and no performance
  content — the compiled core produces a fresh result regardless, so `inplace=True` was
  sugar for copying that result back into the caller's list. Passing it now raises
  `TypeError` like any unknown keyword. The `inplace` parameters of the utils helpers
  (`numbers_to_constant`, `substitude_constants`, `explicit_constant_placeholders`) are
  unaffected — those are genuine in-place list rewrites and downstream code uses them.

### Fixed — the independent monitor no longer fabricates verdicts on extreme-scale expressions
- **`simplipy.verify` judges honestly at every magnitude the token grammar reaches.**
  The independent monitor's judge (the end-to-end soundness gate) carried five
  instrument defects that each fabricated violation verdicts against correct
  rewrites once expressions carried extreme literals (`1e309`, `5e-324`,
  `2^127`-scale rationals): a refused evaluation was converted into a fabricated
  nan instead of a skipped probe; magnitudes beyond 10^(1e50) were refused although
  they evaluate exactly in milliseconds (the cap is now 10^(1e2000), with measured
  cost bounds); the literal zero-snap tested only outermost constant subtrees at two
  hardcoded precisions, so a true nonzero difference spanning 272 digits read as an
  exact zero — the judge then rewrote the input to `0` and convicted the correct
  output — while inner exact zeros (`tan(pi)`) were never recognized; the working
  precision was a flat 50 digits, so `cos(1e-320)` read as exactly 1; and values
  that ROUND to a decision boundary at any precision (`tanh(9e15)` is 10^-8e15
  below 1) fabricated the boundary value instead of refusing. The judge now folds
  exact-rational subtrees in unbounded rational arithmetic, adapts its working
  precision to each pair's own literal exponent span, resolves constant subtrees to
  provable zeros/integers on an escalating precision ladder, and refuses honestly
  (probe skipped, refusal recorded) where no finite precision can decide.
  Small-literal expressions — everything a typical deployment sees — are judged at
  the same precision and cost as before. On the ordinary 1M random-expression
  corpus no row convicts that did not convict before, and 96 of the 98 previous
  convictions plus 104 previous refusals now judge OK; on a 1M extreme-literal
  stress stream, 164 of 229 convictions were measured to be instrument
  fabrications and now judge OK, and 43 more become recorded refusals instead of
  convictions.

### Fixed — the tagged serialization could read back a different value on extreme states
- **A sign could vanish in the tagged rendering of an overflow-partition state.** When
  an expression's exact rational content is split across several members (folding it
  would overflow the exact arithmetic — only reachable with extreme literals around
  10^19-squared and beyond), the tagged emitter pooled their reciprocals into the
  `<div>` section and tracked the sign with a per-member flag that a second signed
  member silently overwrote. The rendering then re-parsed to a DIFFERENT value
  (measured: six of one million random extreme expressions; e.g. an expression worth
  −2/3 serialized to a spelling worth +2/3), while the engine's internal state and the
  explicit/infix renders stayed correct. Partition members now serialize
  self-contained in every dialect, the round-trip is the identity on them, and a new
  fuzz battery permanently screens the tagged rendering's value against the input.
  Ordinary expressions are unaffected (zero changed renders across 1M+64k
  random-expression corpora; the shipped rule artifact re-mines byte-identical).

### Changed — sign decisions are arrival-invariant (canonical forms move slightly)
- **Two ways the canonical form could depend on HOW an expression arrived are gone.**
  (1) When a sign placement was fully μ-tied (mirror pairings like
  `-1 · (a-b)(c-d)` vs `(b-a)(d-c)` price identically), the tie broke on token
  interning order — an artifact of which literal the input mentioned first — so the
  same expression could canonize differently depending on operand order, and the same
  config could canonize differently across vocabulary orderings. The tie now breaks on
  the canonical content order (string-based). (2) When a product's exact rational
  content is split across several members because folding it would overflow the exact
  arithmetic, the minus sign stayed on whichever member carried it on arrival; the
  split and the sign host are now functions of the value alone.
- **Effect — the full upgrade differential, re-measured at publish time** (the
  published 0.12.0 wheel + its artifact vs this release + the re-mined artifact,
  split by cause under the current measure):
  - *400-row reference corpus* (`benchmarks/corpus/raw_skeletons_nv.json`):
    **140/400 = 35%** of outputs differ — 109 equal-μ mirror respells, 21 strictly
    simplifying, 10 μ-costlier than the 0.12.0 output; 12 outputs are token-LONGER.
  - *Frozen fuzz lane* (50,000 rows, the `fuzz_properties.py` generator, seeds fixed
    since 2026-08-03): **1,926/50,000 = 3.85%** differ — 1,205 mirror respells,
    575 strictly simplifying, 146 μ-costlier; 221 token-longer.
  - Stated plainly: **some outputs get longer or less reduced.** The μ-costlier rows
    are the deliberate soundness price — folds and cancellations this release refuses
    to certify (fabricated-value fixes, the widened gates) leave those expressions
    symbolic where 0.12.0 collapsed them. Within one engine the guarantee
    `complexity(simplify(e)) ≤ complexity(e)` holds (400/400, pinned); the μ-costlier
    rows above compare across releases, against outputs the old engine should not
    have produced.
  - A training corpus generated under 0.12.0 is NOT reproducible under 0.13.0 —
    regenerate rather than mix.
  - *Existing 0.12.0 installs are unaffected by the artifact republish alone*:
    the 0.12.0 wheel loads the re-mined artifact loss-free (6,604 translations,
    0 subsumed, 0 dropped) and its reference-corpus outputs are byte-identical
    (0/400 moved) — the engine changes above require the new wheel.

### Changed — pickling refuses a diverged rule state
- **An engine whose `simplification_rules` and compiled core disagree no longer
  pickles silently.** The documented contract is mutate-the-list-then-`compile_rules()`;
  nothing verified it, and a pickle rebuilds workers from the list — so a diverged
  parent could spawn workers running different rules with no warning (measured: 50 of
  400 corpus expressions simplified differently between a parent and its own unpickled
  worker). `pickle`/`deepcopy` now raise a `ValueError` naming the fix
  (`compile_rules()`); engines that follow the documented contract are unaffected.

### Changed — operator specs are validated loudly at load
- **A spec missing a required key is a config error naming the operator and the key(s)**
  (`realization`, `alias`, `inverse`, `arity`, `commutative`). It used to die as a bare
  `KeyError: 'alias'` pointing at nothing. `precedence` is documented as optional for
  non-core operators: they render as function calls, which never consult it.
- **`commutative: true` on anything but `+`/`*` is refused at load.** The engine consumed
  the flag for nothing else — only `+` and `*` canonicalize as AC bags — so `f(a, b)` and
  `f(b, a)` stayed two distinct canonical states while the config claimed one value. A
  declaration the engine silently ignores is worse than a refusal. (No known config
  declares one: measured across every config in the repo and downstream.)
- **A core serialization token declared without `precedence:` now gets the core's own
  value** instead of slipping past the conflicting-precedence guard, which only compared
  values that were present. A conflicting declared value still refuses, as before.

## 0.12.0 — 2026-08-08

### Removed — the legacy kernel (the clean release)
- **The binary simplify kernel is deleted; the AC engine IS the engine.** `simplify_ac`
  becomes `SimpliPyEngine.simplify` (same signature: `form` ∈ tagged/infix/explicit,
  `mode`, `node_budget`); the old best-first tree-search kernel, `cancel_terms`, the
  operand-sort pass (except inside `mask`), the old pattern matcher, `apply_rules`,
  `cancel_only`, `sort_only`, `prune_redundant_rules`/`prune-rules` (the wildcard-shadow
  prune), the old-kernel counters and the `free_coefficients` axis are all gone. The
  hyper-operator realization functions left `simplipy.operators` with them, so LEGACY
  CONFIGS NO LONGER LOAD: cross-version comparison runs against a pip-pinned previous
  release in a separate environment, never against embedded legacy code. Masking survives
  as a current feature applied downstream of simplify, rebuilt as the `simplipy.masking`
  module (see below). The supported pairing is the AC-JUDGED artifact family
  (`acj-2-1`/`acj-3-2`/`acj-4-3`, published on the Hugging Face assets repository); the
  test suites run on `acj-4-3`.

### Removed — the hyper-operator vocabulary
- **`mult2..5`, `div2..5`, `pow2..5`, `pow1_2..pow1_5` are deleted from the vocabulary, once
  and for all.** The base operator config drops from 38 to 23 operators; coefficients and
  exponents are explicit exact literals (`* 6 x`, `pow x 3`), and the general
  signed root `rootn(x, n)` — an engine BUILT-IN, independent of any config — replaces the
  odd-root family (`rootn x 3`, with arbitrary odd indices now expressible). Every layer
  understands `rootn` natively: the interval certificates, the f64 evaluator, and the
  high-precision battery all gained a first-class binary arm (odd integer index, everything
  else fail-closed NaN), so the former `rootn -> pow1_3/pow1_5` re-sugaring in the explicit
  projection is deleted along with the **entire bounded projection** (`form="bounded"`),
  whose only purpose was hyper-operator re-sugaring.

  STRICTLY CLEAN: legacy spellings have NO config-independent reading. They parse only
  under a legacy CONFIG that still declares them (dying with the legacy kernel in the
  clean release), and legacy corpora convert ONCE at the boundary instead —
  `benchmarks/corpus/convert_nv.py` is the pure respeller (`mult3 t -> * 3 t`,
  `pow5 t -> pow t 5`, `pow1_3 t -> rootn t 3`; no simplification, every raw redex kept),
  and `raw_skeletons_nv.json` is the converted benchmark corpus. A new-vocabulary engine
  REFUSES legacy input outright. Downstream data and model-output migration is
  flash-ansr's side of the boundary. Re-mining under the new vocabulary confirmed the
  economics: most of the legacy ruleset was vocabulary artifact — the shipped counts are
  under "Published artifacts" below.

### Added — the AC engine: n-ary associative-commutative simplification
- **`+` and `*` become flat, sorted n-ary bags with EXACT rational coefficients** composed
  explicitly (`* 7 x`, `pow x 3`); `-`, `/`, `neg`, `inv` and the hyper-operator family
  desugar at the boundary and do not exist in the core. Rules are widened to SUB-MULTISET
  matching with the unmatched remainder preserved, so a rule fires wherever the algebra
  permits, independent of operand order and bracketing:

  * both axes of the binary engine's commutative-order invariance defect are removed at the
    representation level — `* cos A tan A` and `* tan A cos A` both give `sin A`, and
    `x3 + (x8 + x3/5)` collects the two `x3` terms across the bracketing;
  * `simplify` output is invariant under permutation of commutative operands and idempotent
    by construction within the exact-coefficient boundary (verified corpus-wide; see "Known
    boundary" below for the one documented exception class at i128-overflow literal scales);
  * coefficient arithmetic is exact rational computation instead of mined rules, so the
    arithmetic-identity rule families of earlier artifacts are simply derived;
  * like-term/like-factor collection (the AC form of term cancellation) runs inside the
    canonical constructors under the SAME soundness certificates as the rule matcher
    (finite-a.e. for sign-cancelling addition, finite-and-nonzero-a.e. for exponent-cancelling
    multiplication), each gate documented with its counterexample — including the branch-cut
    gate (`x^(1/2) * x^(1/2)` does not merge to `x`) and the sign-erasure gate
    (`(x^2)^(1/2)` does not become `x`);
  * pole identities resolve exactly under the supported pairing — `inf - inf` is `nan`,
    `inf / inf` is `nan` on every spelling — via the certificate sorts plus the exact fold;
    ground (variable- and constant-free) expressions are additionally never licensed by the
    a.e. collection gates (an expression with no measure space has nothing for "almost
    everywhere" to quantify over), keeping ground soundness independent of certificate edge
    behavior;
  * the native prefix serialization is the STRICT TAGGED form (default for token inputs):
    n-ary bags are delimited (`<add> ... </add>`, `<mul> ... </mul>`) and carry their group
    inverse as a SECTION — terms after `<sub>` subtract, factors after `<div>` divide, so
    `(2*x1)/(x2*x3)` is `<mul> 2 x1 <div> x2 x3 </mul>` and bags contain no negative
    literals and no inverse operators; `pow` and the unary functions stay plain prefix;
    `neg`/`inv` remain only as the standalone unary spellings (`tan neg x0`, `inv x0`).
    Integer literals are one token each; a non-unit in-vocabulary fraction spells
    STRUCTURALLY through the bag's divide section (`2/3 * x0` is
    `<mul> 2 x0 <div> 3 </mul>`), so a tokenized consumer never needs fraction tokens,
    and out-of-vocabulary rationals fall back to the cheapest atomic spelling
    (`0.12345`); the structural-fraction bound is the declared integer vocabulary
    (default 10, `SIMPLIPY_TAGGED_FRACTION_MAX`). Tagged output is accepted back
    as input (one shared, liberal parser: in-bag `neg`, negative literals and `pow x -1`
    spellings all parse to the same canonical state). Two further
    projections of the same canonical answer: `form='infix'` — a PRETTY human-readable
    rendering (`x8 + 1.2*x3`, `-x0/3`, `(x0 + 1)^2`; the default for `str` inputs);
    `form='explicit'` — the binary-chain diagnostic form with literal coefficients.
    `Mode.LOSSY` relaxes every certificate exactly as in the strict mode.

  The internal canonical form is UNIQUE: bags order by STRIPPED keys (Add terms by their
  coefficient-stripped structural key with constant-like terms last — `x + 1`, the polynomial
  convention; Mul factors by exponent-stripped base), and every canonical sum is PRIMITIVE —
  positive lead coefficient, no unanimous coefficient magnitude, the extracted content
  wrapping the sum (`-a - b` is `-1 * (a + b)`; `a/3 + b/3` is `1/3 * (a + b)`; sums with no
  unanimous content, like `x8 + 1.2*x3`, are untouched, so extraction never inflates tokens).
  Factored sums entering an addition unfactor, so the form is independent of association
  order; sign and grouping spellings of one function converge in the constructors themselves,
  and parse -> canonicalize -> serialize is the identity on every canonical state (asserted
  corpus-wide in debug builds).

  **Supported pairing.** The engine is supported with SORT-PROMOTED rulesets: their
  cancellation rules carry the certificate sorts, so pole identities resolve exactly
  (`inf - inf -> nan`) under the AC matcher. In this release that means the published
  `acj-2-1`/`acj-3-2`/`acj-4-3` artifacts (see "Published artifacts" below); the pre-0.12
  artifacts (`2-1`/`3-2`/`4-3`, `dev_*`) use the retired hyper-operator vocabulary and no
  longer load at all (see the artifact-generation gate below).

### Changed — the reduction ordering is a description length (μ)
- **`SimpliPyEngine.complexity()` and the search objective price the canonical state by μ,
  its coding cost in milli-bits under the token grammar.** A variable costs 8.000 bits, a
  magnitude-1 coefficient is free, and a rational prices as its fraction code
  (`1/2` = 2.585 bits, `355/113` = 15.309 bits); `<constant>` prices at the free-constant
  bound. The decimal code is PRINT-only — it never enters the measure — and the printer
  follows the cheaper code, so `0.5*x0` emits `x0/2` and `1/2` emits `/ 1 2` in the explicit
  dialect, while `0.2` and `0.12345` keep their decimal spelling. Every rewrite must
  strictly descend in (μ, tie-break) — a Knuth–Bendix orientation that makes termination a
  theorem and `μ(simplify(e)) <= μ(e)` a corpus-verified property. Loaded rules are
  re-oriented under μ at load time, so rules mined under any earlier ordering stay aligned
  with the ordering the engine actually fires under.

### Changed — the mining alphabet is the deployed literal vocabulary
- The mine's source AND target alphabet is now the integers −10..10, `np.e`, `np.pi` and
  the IEEE specials — the same literal vocabulary downstream tokenized consumers declare —
  instead of a nine-term development alphabet. Adopted over a measured cost, deliberately:
  the full alphabet costs about 10% simplify throughput for a 7x larger ruleset, and
  completeness over the deployed vocabulary is the goal, not the margin. The complete
  mining universes at the shipped tier are 13,427 length-3 and 399,823 length-4 sources.

### Added — `simplipy.masking`: role-aware literal masking as a module
- **`SimpliPyEngine.mask` is deleted; masking is representation, applied downstream of
  simplify, and lives in `simplipy.masking`.** The module provides `literal_sites` — a
  role-aware walk over both prefix dialects classifying every literal position
  (COEFFICIENT / ADDEND / EXPONENT / ROOT_INDEX / VALUE, with `neg`/`inv` transparent) —
  plus `mask(tokens, engine, policy)`, `mask_all`, and `mask_values_keep_structure`.
  Policies decide per (value, role); the engine's operator table is read directly, never
  through a silent default.

### Added — the artifact-generation gate: generation-1 artifacts refuse at load
- The 0.12 compatibility claim ("legacy configs no longer load") is ENFORCED, not
  aspirational (`simplipy.compat`): artifacts carry an explicit `engine_generation` pin in
  `config.yaml` (generation 2 = the AC engine's clean 23-operator vocabulary; the acj-*
  family and `base` are pinned), the package carries the allowlist
  (`SUPPORTED_ENGINE_GENERATIONS = {2}`), and the refusal is MUTUAL with a actionable
  message in both directions (an older artifact points at `pip install "simplipy<0.12"`,
  a newer one at upgrading simplipy). Un-pinned configs are classified by vocabulary —
  any retired hyper-operator token means generation 1 — so every already-published legacy
  artifact refuses without republishing.

### Added — soundness and measure provenance ships with every artifact
- Every mined `rules.json` is accompanied by `rules.json.provenance.json` carrying the mine
  parameters, the core build stamp (version + git revision), the soundness state — the
  certificate kill-switch states, every artifact-affecting environment override recorded
  verbatim (a registered, single-place switch list), and the interval fail-closed miss
  counters — and the measure fingerprint (unit, the μ constants, probe values, digest), so
  artifacts mined under different orderings are distinguishable from provenance alone.

### Fixed — soundness hardening across the numeric stack
- **The interval layer is rigorous in rounding.** Every computed endpoint steps outward
  (IEEE arithmetic by 1 ulp, libm calls by 8, integer powers exponent-scaled); exact values
  never step (semantic constants, exact libm hits, a-priori range bounds applied as
  post-widening clamps). This retired a class of f64-precision verdicts the engine itself
  used to ship: exact-zero compositions such as `acosh(acos(cos 1))` no longer fold to
  `nan`, and the rendered-overflow pole rules that overflow at f64's `tan(f64(π/2))` are
  gone from the published family.
- **Ground expressions classify exactly.** Integer literals of any magnitude read sign and
  parity from the spelling (`(-inf)^1e40` is `+inf`; a finite negative base at an integer
  exponent refuses instead of asserting `nan`); a non-finite f64 verdict inside the miner
  must be corroborated by the honest interval class before it may mint.
- **Special constants never round into bits.** A ground source that denotes π or e — by
  spelling or by computed VALUE (`acos(-1)`) — stays symbolic; exact collapses ship as
  certified rules, never as f64 decimals.
- **Every token-taking entry is guarded.** The recursion cap covers all FFI entries and the
  deep-infix parser (interpreter aborts on ~200k-deep chains became `ValueError`s); mode
  strings validate strictly; malformed tagged or `rootn` input raises instead of passing
  through unchanged; `load_config` returns values VERBATIM (path resolution moved to each
  key's consumer — a small public API break).
- **The verify judges speak the full 0.12 language** (tagged bags, rationals, `rootn`) and
  fail closed on anything they cannot read; the shipping family carries zero killed
  verdicts under the ruleset gate.

### Changed — rule families become derivations
- The constructors now derive natively what earlier artifacts said as rule families, each
  landed only after the contract judge certified the family and each verified by triple
  byte-identical re-mines with unchanged scale-gate outcomes: sign-blindness as a
  propagating CONTEXT (an even head plants it, an odd function carries it inward, `abs` is
  a no-op inside it, an integer power passes it through — one recursion replacing thirty
  rules, reaching depths the mine cannot); powers of `exp` compose
  (`(e^a)^b = e^(a*b)` wherever the product folds, reached from both `pow` and `rootn`,
  including `exp(1) = e`); the inverse pairs `f(g(t)) = t` per a certified table; the
  reciprocal-base power arms; and the half-period shift `f(t ± π)` for sin/cos/tan at
  coefficient exactly ±1. Across the arc the 4-3 artifact went from 7,153 rules at the
  alphabet expansion to 6,661 shipped — every drop a rule the engine now simply derives.

### Published artifacts
- **`acj-2-1` (26 rules), `acj-3-2` (106), `acj-4-3` (6,661)** on the Hugging Face assets
  repository — the complete AC-judged mines of the clean 23-operator vocabulary at their
  tiers, covered-pruned and sort-promoted, byte-deterministic from the co-located
  `mine.yaml` with one command (three independent mines byte-identical per cell). Each
  ships `config.yaml` (generation-pinned), `rules.json`, the provenance sidecar and
  `mine.yaml`. The hosted family was republished 2026-08-08, superseding an earlier upload
  that predated the interval-rounding hardening and still contained 13 rules since retired
  as uncertifiable or unsound. `base` is the bare 23-operator starting config for fresh
  mining. All pair with simplipy >= 0.12; `load(name, install=True)` fetches and verifies
  by manifest.

### Verified
- Structural gates at 65,536 and 1,000,000 expressions: 0 errors, 0 idempotence failures,
  0 permutation failures, every numeric-divergence flag adjudicated sound; the tracked
  400-row reference corpus pinned exactly; 129 Rust tests (release and debug profiles) and
  650 Python tests green; clippy clean.
- **Known boundary (documented, unchanged in practice):** when two literals *both* refuse
  to fold because their exact sum or product would overflow i128, their relative bag order
  can depend on construction history; randomized fuzzing at ≥ 1e19-scale literal pairs
  shows 2 idempotence violations in 50,000 calls (none reachable from mining, corpus, or
  deployment literal scales; the cure — arbitrary-precision coefficients — is a design
  decision deferred until it matters).

### Docs
- `docs/formal.md` — the formal specification of the engine as a normalized term-rewriting
  system modulo AC: the canonical form, the μ ordering, the licence ledger with each
  claim's status (theorem / by-construction / enforced / empirical), and a change-impact
  map. README and quickstarts run on the acj family with executed outputs.

## 0.11.0 — 2026-07-26

### Fixed
- **The mine no longer certifies rules that disagree with their source at a constant binding.**
  A `<constant>` slot is chosen by the fitter, not drawn from a continuum, so a single
  disagreeing value carries the full data measure: where the source is defined, the target must
  be defined and exactly equal, with no measure tolerance. Four defects let violations through,
  all now closed:
  - **Stage-2 confirmation was vacuous for constant-bearing sources.** It asked "does a target
    exist?" instead of "is it the target we mined?", because the search's all-constant
    short-circuit returns a class literal without reading the candidate list. It now compares
    the returned target against the one under confirmation.
  - **A refuter for literal targets.** Variable-free constant-bearing sources are evaluated at
    exact witnesses and must agree in the extended reals (NaN with NaN, sign-exact infinities,
    exact finite values) before a literal target can be minted.
  - **A collapse licence.** A source may collapse to a single term only where its behaviour is
    unambiguous: the candidate must preserve the source's value class, and a bare `<constant>`
    target additionally requires the source to be finite on a set of positive measure. This
    replaces a narrower special-case guard.
  - **Witness closure under inverse images.** Some sources are finite only at the *preimage* of
    a corner point (`acosh(sin(C/3))` only where `sin(C/3) >= 1`, i.e. `C = 3pi/2 + 6k·pi`),
    which no amount of dense sampling reaches. Witnesses are now closed by pulling operator
    corners back through each unary in the source, with period shifts for periodic inverses.

  Measured on a complete enumeration of every variable-free `<constant>`-bearing source at
  length <= 4 (45,560 sources): **0 rules disagree with their source**, against 24 in the
  previously published rule set. Freshly mined cells are correspondingly cleaner; deployed
  artifacts change only when re-mined.

### Added
- **Ladder re-use: `find_rules(snapshot_at={length: path})`.** The cells of one
  `max_target_pattern_length` form a prefix chain -- mining `(7,4)` already does all the work of
  `(5,4)` and `(6,4)` on the way up, because lengths are mined shortest-first, each source's
  seed is indexed by its length alone rather than by how far the climb goes, the master seeds
  are drawn before the universe is built, and enumeration consumes no randomness. A climb can
  now emit each shorter cell as it passes through: the full post-pass (proposals, prune,
  promotion) runs on a copy of the mine, that cell's artifact and sidecar are written, and the
  raw un-pruned state is restored so the climb continues unaffected. Each snapshot's sidecar
  carries a `ladder_snapshot` record naming the climb it came from, and its parameters and
  universe census describe the cell rather than the climb. Verified byte-identical to one-shot
  mines of the same cells, including with a sampled length above the snapshot. Also exposed as
  the `snapshot_at:` key in the find-rules config.
- **Peak memory during mining is lower.** Each length's source universe is released as soon as
  it has been linearised; at the top lengths that set is the largest object in the process.
- **Ladder seed `* 0 !0 -> 0`** (the multiplicative zero absorber), certified through the same
  `judge_bang` bar as the other seeds. Commutative operands are canonically ordered
  variable < number < composite, so the enumerable carrier `* 0 x0` canonicalises to `* x0 0`
  and the mine only ever mints the right-hand form. But "zero times a *composite*" is
  canonically literal-first, so the left form is the shape that actually occurs, and the only
  carriers a cell can enumerate for it (`* 0 <unary> <leaf>`) mint per-operator rules that miss
  every binary operand. Without the seed, 213 of 65,536 corpus expressions simplified less --
  one 27-token expression reaching 17 tokens instead of 1. This restores it: mean length ratio
  0.8999 -> 0.8989, with 213 regressed rows falling to 18. Same situation as the additive-cancel
  seeds, whose carriers are pre-cancelled rather than canonicalised away.

### Changed
- **`simplify()` is ~4% faster: the hot-path observability is now a compile-time feature, off by
  default.** `simplify_counters()`'s counters sat in the innermost loops of the released wheel --
  `pattern_attempts` alone is one relaxed `fetch_add` per candidate rule tried, ~2.1k per
  expression on the published 4-3 rule set -- and the `nanos_*` phase timers cost two clock reads
  per bracket. Two independent measurements, both on the published 4-3 rule set: min-of-30 over 6
  interleaved rounds at two corpus sizes gave 99.4 -> 94.7 us/expr at n=8192 and 99.4 -> 94.5 at
  n=2048 (**-4.6%**, faster in 6/6 rounds at both sizes, counters ~4.0 us and timers ~0.5 us of
  that); an independent min-of-7 on n=6000 gave 99.80 -> 95.75 us/expr (**-4.1%**, median
  103.41 -> 100.67). Take the effect as **4-5%**: the interleaved protocol is the better one, and
  the weaker protocol lands slightly lower, as expected. Output is token-identical -- one sha256
  over 5,000 corpus rows agrees across HEAD, the default build and the `profiling` build, while
  `simplify_counters()` reads all zeros in the default build and non-zero under `profiling`.
  `simplify_counters()` therefore reads all zeros in a released wheel; build the profiling
  extension to read it: `maturin develop --release --features profiling` (or the individual
  `stats-counters` / `stats-timers`). The mining-progress counters are untouched -- they are off
  the simplify hot path.

### Fixed
- **`nanos_rules` is recorded again.** The counter was declared and exported but had no recording
  site since the tree-search refactor deleted the block its timer lived in, so
  `simplify_counters()` reported the rule-application pass as 0 ns while every other phase
  reported honestly, and share-of-runtime readings silently attributed the whole rules pass to the
  unaccounted remainder. It is now bracketed at the same site as `nanos_cancel`, which makes the
  two edge generators directly comparable: on the published 4-3 rule set the rules pass is 55.2 of
  95.2 us/expr (58%), cancellation 24.5 (26%), mask+sort 9.0 (9%), leaving 6.5 us (7%)
  unaccounted; that works out to ~27 ns per candidate-rule attempt, and the same ~27 ns on the
  much larger dev 7-3 rule set at 10x the attempts per expression.

## 0.10.1 — 2026-07-26

### Added
- **Per-candidate verdict trail for the proposal channel.** `find_rules(proposals=...)` now
  writes `proposals.trail` into the provenance sidecar: one entry per proposal, in file order,
  as `{source, target, verdict, stage, certificate}`. The aggregate tally it accompanies cannot
  be audited on its own -- "93 rejected" never says which candidate died at which gate, so a
  reviewer cannot separate a correctly killed hallucination from a wrongly killed identity.
  `stage` names the deciding gate: `vocabulary` (token outside the alphabet, or malformed),
  `covered` (the mined rules already shorten the source), `search` (no library target and no
  verifiable hint), `confirm` (failed independent stage-2 re-verification), `merge` (certified
  but folded away as a duplicate), or `accepted`. No certification semantics change.

## 0.10.0 — 2026-07-26

### Added
- **The certificate algebra: structural null-measure certificates.** A new predicate family in
  the interval engine — `zero_set_null` (identity-theorem witness for entire-analytic
  compositions plus structural recursion), `nonfinite_null` (pole-set tracking through the
  domain-restricted unaries), `finite_nonzero_ae` (= the two combined: the regular domain of
  multiplication), and `positive_ae` (v1) — closing the class of binding-dependent side
  conditions. `finite_ae` now certifies through the structural path as well as box subdivision.
- **The `$` wildcard sort: certified multiplicative cancellation.** `$N` binds composite
  subtrees gated by `finite_nonzero_ae`, licensing `A/A -> 1`, `A·inv(A) -> 1`, and
  `0/A -> 0` exactly where they are sound (`cosh(x)/cosh(x) -> 1`; `asin(x)/asin(x)` and
  `0/0`-bearing trees refuse). Ships with `judge_bang_mult` (the promotion bar on the
  finite-nonzero atom lattice) and four ladder seeds; deployed artifacts are unchanged until
  the next production mine.

### Fixed
- **Promotion oracle non-termination (B4).** `_literal_spans` could loop forever with unbounded
  memory when the operator table was empty or contained unknown ops (unvisited `end[]` entries
  sent the scan backwards). A termination guard plus a fail-fast `RuntimeError` in `prefold`
  when the oracle is unconfigured close both the hang and the silent misconfiguration path.

### Removed
- **BREAKING: the `max_pattern_length` parameter of `simplify` (and `apply_rules`) is gone.**
  Rule application always considers every pattern in the loaded artifact; the scan window is
  the ruleset's own longest pattern, an intrinsic property rather than a caller knob. All
  deployed callers already passed `None`; passing the keyword now raises `TypeError`.

## 0.9.1 — 2026-07-25

### Added
- **Within-tier mining progress.** `find_rules(verbose=True)` now reports progress WHILE a length
  tier is mining, not only at the end. The Rust miner publishes a `(sources_done, sources_total)`
  counter (`Engine.mining_progress()` on the compiled core) that a daemon monitor polls, printing
  `Length L: X/N sources (P%) | R src/s | ETA T | elapsed E | RSS M` at 20 / 60 / 180 s and then
  every `SIMPLIPY_MINE_PROGRESS_INTERVAL` seconds (default 600). A long tier -- the length-5+
  combinatorial wall where a mine can sit for days -- is no longer an opaque wait: its rate, ETA,
  and memory are visible as it runs. Zero effect when `verbose=False`.

## 0.9.0 — 2026-07-24

A new simplify kernel. `simplify` is now a best-first cancellation SEARCH instead of a fixed
cancellation order; masking (numeric literals -> `<constant>`) is separated from the
equivalence loop so representation is no longer entangled with rewriting; and the constant-fold
fallback is brought under the same finite-a.e. certificate the rules and cancellation already
enforce.

### Added
- **Cancellation search** (`Engine::simplify_search`). `simplify` no longer commits to one
  cancellation order. Cancel is non-confluent -- taking candidate A can destroy candidate B,
  and which choice ends shortest depends on what the ruleset can fold -- so the kernel now
  SEARCHES a small move graph instead of guessing. State = an expression; the moves are a flat
  choice set (apply the rules pass, or cancel any one qualifying candidate); the answer is the
  shortest state visited. Every state is a.e.-equivalent to the input and the input is state
  zero, so a result can never be longer than its own input. Best-first by
  length with a visited-set, pre-filled with the greedy result so a truncated budget can never
  do worse than the plain fixpoint. The node budget (`SIMPLIPY_SEARCH_BUDGET`, default 24;
  0 restores the plain greedy fixpoint and its speed exactly) bounds a rare heavy tail -- the
  median expression expands 2 nodes and 53% have no cancellation candidate at all, but the p99
  is ~186. The default is the measured elbow of the returns curve on the 64k v23.0 prior: below
  24 an extra microsecond buys ~14 output tokens, above it ~2.7, and by 256 only 0.7. On that
  corpus the search shortens ~2860/1000/350 expressions (2-1/3-2/4-3) that no previous version
  could reach, at ~2.3x the simplify wall time; raise the budget for offline corpus
  canonicalisation, lower it for latency.
- **Symmetric `neg`/`inv` cancellation** (BEHAVIOUR CHANGE). The cancellation unit already
  EMITTED the class inverses but never CONSUMED them: a leaf under its own class inverse was
  shielded, so `x * inv(x)` and `x + neg(x)` did not cancel through the inverse. Each inverse is
  now region-continuing in its own class (`neg` additive, `inv` multiplicative), symmetric with
  the emit path, and there is exactly one region shape -- the asymmetry is not preserved behind
  a flag. Consequence on sparse rule sets: a handful of expressions come out ONE token longer
  than in 0.7.x, because the old opaque treatment happened to leave a sign arrangement those
  rule sets can re-fold and the connected treatment does not (6 of 65536 on the 2-1/3-2-class
  `3-2` set; none on `4-3` or richer, and never longer than the input). Cancelling through the
  class inverse is worth far more than it costs: the same corpus gets 1005 expressions SHORTER
  on `3-2` and 350 on `4-3`.
- **Soundness `Mode` (`simplify(expr, mode=Mode.SOUND)`).** A single ordinal soundness axis,
  exposed as `simplipy.Mode`. `Mode.SOUND` (the default) is equivalence-preserving and
  idempotent -- the deployed inference/scoring path, byte-identical to the historical default.
  `Mode.LOSSY` trades soundness for recall: every rule placeholder binds any subtree (the
  `!`-sort finite-a.e. certificate is skipped) AND the constant-fold's finiteness gate is
  relaxed, so a non-finite-a.e. subtree such as `<constant>/0` collapses to `<constant>` too.
  `Mode.LOSSY` is for training-corpus canonicalisation ONLY -- the training data is generated
  FROM the simplified form (target == data) -- and must never run on an inference or scoring
  path. The decided full ordering is `EXACT <= SOUND <= AE <= LOSSY`; only `SOUND` and `LOSSY`
  are implemented.

### Changed
- **BREAKING: masking is separated from `simplify`.** Masking numeric literals to the generic
  `<constant>` placeholder is a REPRESENTATION step for downstream models that cannot consume
  literals, not an equivalence-preserving rewrite. `simplify` no longer masks; it is now the
  equivalence loop only (search + sort to a fixpoint), sound and idempotent by construction. The
  `mask_elementary_literals` parameter of `simplify` is removed; use the new terminal
  `Engine.mask()` (relabel literals + one sort, no re-simplify) on `simplify`'s output when
  placeholders are needed, and do NOT re-`simplify` a masked expression. Entangling masking with
  the search fixpoint was also the sole cause of the former non-idempotence: masking mints a free
  `<constant>` from a structural literal (`x - x -> 0 -> <constant>`), and re-searching could
  fold that constant into a denominator, dropping the reachable `C = 0` the input required.

### Fixed
- **Constant folding respects the finiteness certificate.** The constant-fold fallback collapsed
  any subtree whose operands are all `<constant>` or finite literals to a free `<constant>`, on a
  syntactic test that assumed finite operands compose to a finite result. That is false at a pole
  (`<constant> / 0` is +-inf/nan for every constant), so a structural zero reaching a denominator
  could be folded to a finite free constant -- unsound (it revives a structurally zeroed term).
  This was the one search edge that skipped the finiteness certification the rule matcher and the
  cancellation already enforce. The fold now consults the same value-set analysis and collapses to
  `<constant>` only when the subtree has a positive-measure finite part, keeping
  unbounded-but-finite-a.e. folds (`1/C`, `tan C`, `cosh(C + 5)`) and refusing non-finite-a.e.
  ones (`C / 0`, `C * inv(0)`).

## 0.8.0 — 2026-07-22

Makes the public package produce the BEST rule sets from one command: mining now natively
promotes every rule to the strongest sound sort, and a public verification API can
independently gate + monitor any rule set.

### Added
- **Native sort promotion** (`simplipy.promotion`; `find_rules(promote_sorts=True)`, CLI
  config key `promote_sorts`). After mining and pruning, every rule (mined + proposed) is
  re-certified at the stronger sorts and shipped at the strongest SOUND one: `_` (arbitrary
  subtree), `!` (certified-finite subtree, enforced by a match-time defined-and-finite-a.e.
  certificate), or `?` (variable-leaf, the fallback). Five stages: a pointwise exact bar on
  an atom lattice, a const-bearing witness-map bar, an exact-arbiter overturn of finite-draw
  demotions, a subsumption/derivability refund, and the `_`→`!`→`?` ladder with a
  moving-spike structural refusal. Promotion is fail-safe: an uncertifiable rule stays `?`
  and loses only composite-subtree recall, never soundness. This recovers the simplification
  power that conservative `?`-only rulesets leave unrealized (a mined ruleset promoted this
  way matches a far larger legacy engine's reduction at a fraction of the rules), because the
  limiting factor was sort generality, not the number of rules.
- **Independent verification API** (`simplipy.verify`): `verify_ruleset` gates a rule set by
  judging every rule at its own symbolic trigger points under an arbitrary-precision contract
  evaluator (eight-bucket classification, 100% coverage by construction); `monitor_ruleset`
  runs the deployed engine over an adversarial+sampled corpus and attributes any
  deployed-value violation to the responsible rule under an independent high-precision
  evaluator; `verify_rule` gives a single-rule verdict. Deliberately implemented
  independently of the compiled core so it cross-checks the miner rather than echoing it.
  Both carry poison self-tests.

### Changed
- `mpmath` and `scipy` are now runtime dependencies (offline mining + verification only; the
  inline simplify path remains the compiled core).

## 0.7.1 — 2026-07-21

### Fixed
- **CLI `find-rules` honors the full config**: the `prune` and `relaxed_kruskal` keys in a
  find-rules YAML were silently ignored (`prune` fell back to the `--prune` CLI flag,
  i.e. `False`; `relaxed_kruskal` to its default), so a config declaring
  `prune: covered` produced an unpruned artifact that did not match its own claims.
  Both keys are now forwarded (a config `prune:` key takes precedence over `--prune`),
  and the CLI rejects unknown config keys fail-closed, naming the offending key — a
  mis-spelled or unsupported key is an error, never a silent no-op (`confirm_rules:`
  was such a silent no-op; the honored key is `confirm:`).
- The provenance sidecar now records `relaxed_kruskal` alongside the other mine
  parameters.

## 0.7.0 — 2026-07-21

Real-semantics pow alignment and the removal of the faithful dev_7-3 reproduction line:
the package now ships ONE engine line.

### Added
- **LLM-proposal channel in the miner**: `find_rules(..., proposals=...)` accepts a path
  to a proposals JSON (consolidated `{"proposals": [...]}` artifact or a bare list of
  `{source, target?}` objects) or the equivalent in-memory object. After the mining
  length loop and before the optional prune, every proposal runs the exact
  `certify_rules` chain against the just-mined rule state with the mine's evaluation
  matrices, tolerances and master-seed-derived, content-derived per-proposal seeds;
  certified proposals join the ruleset through the same `deduplicate_rules` path. The
  find-rules YAML forwards a `proposals:` key, and the provenance sidecar records the
  proposals file, its sha256, and per-outcome counts
  (`certified` / `already_covered` / `rejected` / `duplicate`). This makes a
  mined-plus-proposed ruleset reproducible from one config and one command.
- **Special-point certification phase in the miner** (`rust/battery.rs`): rule
  certification now additionally checks every accepted (source, candidate) pair at a
  battery of symbolic special points (0, ±1/2, ±1, ..., ±π/2, π, e) per variable, sweeps a
  battery of special source-constant values, and snaps fitted witness constants to nearby
  integers / half-integers before the domain-preservation gate. This closes three gaps
  where a rule could certify on random evaluation matrices yet be unsound at points
  deployed expressions actually reach: a fitted exponent like `2.9999999999999996` is NaN
  on the negative half-line and hides the positive-measure domain extension its snapped
  witness `3.0` creates (`exp(log(pow3 x)) → pow(x, C)`); a rule can be false exactly at a
  symbolic coincidence (`pow(sin x, inf) → 0` is 1 at x = π/2); and a pattern-bound
  source constant reaches special values in deployment (`pow(cos C, inf) → 0` at C = 0).
  Rows where deployed f64 evaluation diverges from contract semantics are re-judged at
  three fixed precision rungs (50/120/250 decimal digits) with symbolic coordinates
  rendered at each precision, so stable limit completions (`x/x → 1`) keep certifying
  while precision-dependent cancellation seams are rejected fail-closed.

### Changed
- **Stricter mining certification (see the special-point phase above)**: `find_rules`
  rejects rule families that earlier versions could mine. Rulesets mined with
  `simplipy <= 0.6.0` may therefore contain rules that 0.7.0 refuses to re-mine;
  re-mining under 0.7.0 is the recommended migration.
- **Pow alignment (real semantics)**: `pow` at a `-inf` base with a finite non-integer
  exponent now evaluates to NaN (real semantics) across the constant folder, the operator
  realizations, and the interval engine. Magnitude-step cells at infinite exponents are
  unchanged.
- **`engine_id` now reports the package version** (e.g. `simplipy-0.7.0`) instead of the
  frozen reference id `dev_7-3`, so provenance records identify the exact engine build.

### Removed
- **BREAKING: the faithful dev_7-3 reproduction line is removed.** The `fold` parameter of
  `_core.Engine.simplify` / `apply_rules` / `prune_explicit` is gone (the numeric
  constant-folding behavior is now the only one), the legacy quirk-preserving conversion
  variants are gone (the corrected conversions are now exposed under the plain
  `prefix_to_infix` / `infix_to_prefix` / `convert_expression` / `parse` names, which is
  what the Python API always routed), and the frozen byte-identical-to-0.2.15 parity
  fixtures and reference constants (`FAITHFUL_ENGINE_ID`, `REFERENCE_SIMPLIPY_VERSION`,
  `REFERENCE_SIMPLIPY_COMMIT`) are gone. The parity regression test is re-baselined
  against the 0.7.0 aligned engine line. To reproduce v23.0/dev_7-3-era behavior
  byte-for-byte, install `simplipy<=0.6.0`.

## 0.6.0

A performance overhaul of the simplify hot path (byte-identical outputs), sorted rule
placeholders with match-time certificates, ruleset pruning and observability tools, and
a round of rule-mining improvements and constant-folding coherence fixes. The shipped
`dev_7-3` asset is unchanged.

### Added
- **Sorted rule placeholders** (`_` / `?` / `!`): every placeholder in a rule carries a
  *sort* — the binding claim its sigil encodes, enforced by the matcher at apply time.
  `_i` binds any subtree; `?i` binds only a bare variable leaf; `!i` binds a variable
  leaf freely, but a composite subtree only when a match-time certificate proves it
  defined and finite almost everywhere (an adaptive interval analysis over the reals;
  fail-closed — what cannot be certified is not bound). Sorts let a rule that is
  value-sound only for certified operands make the wider claim without giving up
  soundness. Newly mined rulesets emit sorted patterns; existing assets (explicit and
  `_`-wildcard rules) behave exactly as before. See the rule-authoring guide
  (`docs/rules.md`) for details.
- **Covered-rule pruning** (`SimpliPyEngine.prune_covered_rules`, CLI
  `simplipy prune-covered-rules`): removes any rule — wildcard-pattern rules included —
  whose effect the remaining rules already achieve compositionally. A rule is covered
  only if every instantiation variant of its source (distinct variable leaves, literal
  `<constant>`, composite probes for wide placeholder slots) still simplifies to at
  most the length of the corresponding target without it. Deterministic batch
  remove-and-repair, longest sources first; greedy (the result is valid, not
  necessarily minimal). Complementary to `prune_redundant_rules`, which removes only
  explicit rules shadowed by pattern rules under an equality criterion. `find_rules`
  gains `prune='covered'` to run this prune after discovery instead of the
  redundant-rule prune.
- **Simplify observability**: `SimpliPyEngine.simplify_counters()` /
  `reset_simplify_counters()` expose the process-global simplify hot-path counters
  (calls, iterations, pattern attempts/fires, certificate calls/hits, and coarse
  per-stage nanosecond accounting) for profiling a batch.
- **Formal mining-algorithm documentation**: a formally typeset specification of
  `find_rules` (the discovery loop and the equivalence certification), with PDF/LaTeX
  downloads, an algorithm-line-to-configuration reading guide, and the determinism
  contract (`docs/algorithm.md`).
- **Relaxed Kruskal search** (`mine_one_length(..., relaxed_kruskal=...)`): with
  `True`, a source the current rules already shorten is still searched, with the
  candidate bound tightened to its simplified length (only targets strictly shorter
  than what `simplify` reaches are accepted). Relaxed rules are one-step shortcuts that
  are capability-redundant wherever the reduct's own rule exists; for sampled
  universes, re-mining the simplify-fixpoints of skipped sources generalizes better (a
  fixpoint rule covers every source that reduces to it). The Rust-level parameter
  defaults to the strict skip, but `find_rules(relaxed_kruskal=True)` is the
  Python-side production default (a capability comparison across four mined corpora,
  ~780k expressions total, found the relaxed mine strictly shorter on 735 of them and
  never longer, at ~+6% mine cost).
- **High-precision rescue for const-free equivalence** (`astro-float`, pure Rust, 1024-bit):
  f64 remains the fast pre-filter, but a const-free candidate that fails the tolerance on
  only a small fraction of its binding rows (<= 25%, <= 256 rows) is re-evaluated on
  exactly those rows at 1024 bits and re-judged under the same generic-equivalence
  semantics. True identities whose f64 round-trip loses precision now certify --
  `atanh(tanh(x)) -> x` (f64 error reaches 2e-1 for |x| in [10, 19.6]) and, via domain
  extension, `tanh(atanh(x)) -> x`, both previously unminable at the strict rtol=1e-9.
  Every f64 accept is unchanged; plateau falsities keep failing their near-corner rows and
  never reach the escalation. Applies to the mine scan and (through the same code path)
  stage-2 confirmation and `certify_rules`.
- **Candidate library minimization** (`find_rules(candidate_fold_filter=True)`, default on):
  variable-free candidates of length >= 2 are excluded from the mining candidate library.
  This is provably behavior-preserving (any source they could match is already matched by
  the length-1 `<constant>` candidate, which is scanned first) and removes the bulk of the
  constant-fitting work: measured 55-88x faster per source on the dev configuration at
  identical mining decisions. The library reports `n_candidates`/`n_filtered`.
- **Provenance sidecar**: `find_rules(output_file=...)` now writes
  `<output>.provenance.json` recording all parameters, derived seeds, the evaluation-matrix
  specification, and per-length universe coverage, so a mined ruleset is reproducible from
  its artifact alone.
- **Per-run sampler validation**: sampled sources are checked for universe membership
  (length, vocabulary, well-formedness) on every run.

### Changed
- **NaN literals propagate through constant folding**: a subtree with a
  `float("nan")` operand now folds directly to `float("nan")` (every operator maps a
  NaN operand to NaN; the IEEE `pow` edge cases whose value does not depend on the NaN
  operand are excluded), and `inf`/`nan` literals are never absorbed into a fittable
  `<constant>` — a `<constant>` claims a finite value exists to fit, and NaN is not
  one. This is operator-table knowledge, not sampling: it also keeps the miner from
  certifying rules that rewrite an everywhere-NaN expression, for which no finite
  evidence can exist.
- **Constant-fit restart seeds are order-independent** (a pure function of the source seed,
  candidate, and challenge instance). Mines remain byte-reproducible for a fixed seed; the
  output of a given seed may differ from 0.5.0 on marginal nonlinear fits.

### Performance
- **Certificate economics**: `!`-sort certificates are evaluated only after a
  *completed* syntactic match (instead of during every candidate attempt), memoized
  per call and in a generational per-engine cache that never stops memoizing. This
  replaces the old hard-capped certificate cache, which silently stopped memoizing
  once full and degraded at scale (measured on a 65,536-expression training-prior
  benchmark: hit rate falling from 99.3% to 36.6% and per-expression engine cost from
  ~600 µs to ~7,400 µs once the cap was hit). Certificate verdicts are unchanged.
- **Fixpoint memoization**: the simplify fixpoint memoizes whole cancel/rules passes
  and rule-normal subtrees per call, so the convergence-confirming iteration and
  unchanged subexpressions cost hash lookups instead of re-scans.
- **Interned token representation**: the hot path runs on interned token ids with a
  per-engine property table (and a per-call overlay for unknown tokens), reducing
  allocations per simplify call by ~20×.
- Net effect on a 65,536-expression training-prior benchmark: large
  certificate-bearing rulesets simplify ~59× faster than 0.5.0, with byte-identical
  outputs; certificate-free rulesets run ~2.3× faster.
- **Mining**: the source expression is evaluated once per source (per challenge
  instance) and shared across the whole candidate scan, instead of per
  (candidate, challenge, sign-combo); a const-free source collapses to a single
  challenge instance (identical targets add no evidence and only multiplied fit
  flakiness).

### Removed
- The `scipy` runtime dependency: dead since the 0.5.0 native constant-fit cutover
  (no `scipy` import remained anywhere in the package).
- 14 dead `SimpliPyEngine` attributes left over from the removed pure-Python simplify
  engine, each verified to have no remaining consumers: `operator_inverses`,
  `inverse_base`, `inverse_unary`, `inverse_binary`, `unary_mult_div_operators`,
  `commutative_operators`, `realization_to_operator`, `operator_precedence_compat`,
  `max_power`, `max_fractional_power`, `connection_classes`, `operator_to_class`,
  `connection_classes_inverse`, `connection_classes_hyper`,
  `binary_connectable_operators`.
- `simplipy.utils.leaf_value`: unused Python duplicate of the engine's single
  leaf-value table.
- Two undocumented, dev-only methods on the compiled core (`eval_bench_resident`,
  `classify_linearity`) and, on the Rust side, the unused `match_pattern` wrapper and
  the unused `criterion` dev-dependency.

### Fixed
- **Term cancellation respects non-finite algebra**: with an empty ruleset (or rule
  application disabled), `0/0`, `nan/nan`, and `inf/inf` previously canceled to `1`,
  and `inf - inf` / `nan - nan` to `0` — all wrong answers (the true value is NaN).
  Cancellation assumes the group axioms, so a leaf now registers as cancellable only
  in the connection classes where they hold: nonzero finite literals in both, `0` in
  the additive class only, `nan`/`±inf` in neither. Variables and nonzero-finite
  literals cancel exactly as before.
- **One leaf-value table everywhere**: constant folding (`_try_fold_constants` /
  `evaluate_constant_subtree`), the miner's all-constant short-circuit, and the offline
  tape evaluator now share a single leaf resolution (Rust `numeric::leaf_value`):
  numeric literals, `np.pi`, `np.e`, parenthesized literals such
  as `(-1)`, and the `float("...")` tokens. Previously the folder and short-circuit used
  the numeric-string predicate only, so e.g. `cosh(np.pi)` never folded exactly and
  `(-1)/cosh(np.pi)` simplified no further than `(-1)/<constant>` even though every leaf
  has a known value. The `<constant>` absorption collapse now admits any FINITE-valued
  leaf (still never inf/nan literals: non-finite algebra remains the explicit rules'
  domain).
- **Generic-equivalence vacuity in the constant-bearing fit arm**: an equivalence-check
  instance whose source evaluates nowhere finite (e.g. the sign-combo `C = 0`
  instantiation of a constant-denominator source, `y = k/0`) binds no rows and is now
  treated as domain-extendable — matching the declared semantics and the const-free arm —
  instead of falling into the underdetermined-fit bail that returned `False` and vetoed
  the candidate. This single instance previously killed every constant-denominator rule:
  `x0*<constant>/<constant> -> x0*<constant>` was unminable, and (with the short-circuit
  gap above) so was the whole `k/<constant> -> <constant>` absorption family. Evidence
  accounting is unchanged: vacuous instances contribute zero rows toward
  `min_informative`.
- **Affine constant-fit recall with fast-growing bases**: fits of the form `C0*f(x)+C1`
  and `f(x)+C1` for `f` in `exp`/`cosh`/`sinh`/`pow3`-`pow5` were spuriously rejected at
  exactly-true constants. The closed-form solver now uses row weights matched to the
  acceptance tolerance, overflow-safe column equilibration, iterative refinement, an exact
  feasibility decision for the bare-`<constant>` candidate, and integer snapping for
  exact-cancellation cases. The acceptance gate itself is unchanged (recall-only fix;
  non-equivalences are still rejected).
- Fold-filter variable detection is correct for configurations with more than 32 variables.
- `min_informative` defaults are floored at 1 for very small evaluation matrices.

### Docs
- 0.5.0 entry corrected: the mine evaluation matrix's log-uniform tier is 1e-4..1e3, not
  1e-6..1e6.

## 0.5.0 — 2026-07-11 — sound rule-mine checker + Python mining mirror removed

The pre-0.5.0 checker shipped ~8.3% defective
rules in the dev_7-3 sample (52/840), including ~5,125 vacuous all-NaN wildcard rules such as
`asin(cosh(_0)) -> nan` (false at 0, where the source is pi/2).

### Fixed (checker soundness)
- **Vacuous equal_nan acceptance closed.** Certification now requires at least `min_informative`
  (default `n_rows / 8`) SOURCE-FINITE evidence rows, accumulated across challenge instances; a
  precomputed candidate-finite count serves as a fast necessary-condition gate for const-free
  candidates.
- **Tolerances tightened** `rtol` 1e-5 -> 1e-9, `atol` 1e-8 -> 1e-12 (0 borderline rules in an
  840-rule review sample; tanh/exp saturation towers are no longer "equal" to constants).
- **Heavy-tailed, seeded evaluation matrix** (`_mining_sample_x`): 40% N(0,5) + 25% U(-50,50) +
  25% signed log-uniform magnitudes 1e-4..1e3 + 10% exact corner points ({+-0.0, +-0.1, +-0.5,
  +-1, +-2, +-e, +-pi, +-10}). (This entry mis-stated the tier as 1e-6..1e6 until 2026-07-11;
  the code always shipped 1e-4..1e3 -- the wider tail was tried and reverted for fit
  conditioning, see the `_mining_sample_x` docstring.)
- **GENERIC-EQUIVALENCE semantics with domain extension** (`allclose_extends`, the single accept
  gate everywhere: const-free compare, affine/log-linear/LM fit accepts). Rows where the SOURCE is
  finite bind: the replacement must be finite and equal within tolerance -- the exact corner points
  refute wrong-VALUE identities (`asin(cosh(_0)) -> nan` is false AT 0, where the source is pi/2).
  Rows where the source is NaN/inf are extendable: the replacement may complete them with the
  generic/limit value (`div(_0, _0) -> 1` certifies; `log(exp(_0)) -> _0` certifies under f64
  overflow). The evidence gate (`min_informative`) counts SOURCE-FINITE rows accumulated across
  challenge instances, so an (almost-)nowhere-defined source can never be rewritten from its
  corner rows alone.
- **Stage-2 confirmation** (`confirm=True`): every mined pair is re-verified on an independent,
  twice-as-wide X with fresh constant draws and seeds before it can enter the Kruskal cascade.
- **IEEE inv/div semantics** aligned across the Rust kernels, the scalar Python operators and the
  constant folder (`1/±0 = ±inf`, `x/0` sign-correct): the mine now certifies exactly the semantics
  the deployed numpy engine executes.
- **Complete Phase-1 enumeration.** The old pass-based closure stopped once the maximum length was
  REACHED, not SATURATED: the dev_7-3 mine saw 444,865 of the 9.0e9 length<=7 expressions and missed
  e.g. ALL 179,685 triple-unary chains at length 4. Phase 1 is now a bottom-up DP per length,
  complete by construction and cross-checked against an exact count DP every run.
- **Universe policy for infeasible lengths** (`source_sample_per_length`): lengths whose complete
  universe explodes (dev set: 2.4e8 at length 6, 8.8e9 at length 7) are drawn uniformly from the
  complete universe via a count-weighted top-down sampler; coverage is always logged, and sampling
  inside the candidate range warns (the library then no longer certifies minimality).
- **Full determinism**: seeded master RNG (`seed=42`), sorted set iteration for sources and the
  candidate library, derived per-length seed blocks. Same seed => identical rule set.
- **Fit-path completeness**: `constants_fit_challenges` / `constants_fit_retries` defaults raised
  5 -> 16 end to end (the old find_rules passed 5 against an FFI default of 16; measured fit-path
  completeness at 5/5 was ~24%).

- **Affine-fit conditioning fix**: the whole `C0*f(x)+C1` family silently REJECTED before. Two
  causes, both introduced by earlier hardening of the same path: a GLOBAL trace-scaled Tikhonov ridge biased the
  intercept, and the normal-equations solve squared the condition number on the wide-magnitude X
  so an exact affine relationship's intercept came out ~5e-9 off (past rtol). Now the affine path
  solves by Householder-QR least-squares (no `A^T A`, no ridge -- works at cond(A), not its square),
  and the mine X's log-uniform tier is capped at 1e3 (from 1e6): 1e3 still exercises every
  saturation and f64 overflow the tier is FOR, but 1e6 wrecked the fit conditioning. Confirmed by
  probe: 0/30 -> 64/72 affine cases certify, 0 false-accepts; the residual misses have constants
  spanning >1e6.
- **Log-linear recall fix**: for `pow(<constant>, g)` candidates, only a closed-form log-space
  ACCEPT short-circuits; an imprecise `Some(false)` solve (its ~1e-10 base error amplified past
  rtol by large exponents on the heavy-tailed X) now SEEDS the LM restart instead of rejecting,
  restoring the declared const-bearing policy ("accept iff constants EXIST that fit"). Surfaced by
  extensive adversarial re-verification of the generic-equivalence change (which found NO soundness
  defects).

### Changed
- **Perf**: the source expression is now evaluated once per source (per challenge instance) and
  shared across the whole candidate scan, instead of per (candidate, challenge, sign-combo); a
  const-free source collapses to a single challenge instance (identical targets add no evidence
  and only multiplied fit flakiness).

### Removed (BREAKING)
- **The pure-Python mining mirror is gone**: `find_rule_worker`, `exist_constants_that_fit` and the
  fork/SharedMemory pool (with the `n_workers` parameter). It duplicated `rust/worker.rs`/`fit.rs`
  and repeatedly desynced from them (e.g. diverging IEEE inv/div semantics). `find_rules` now requires the
  compiled core and raises `RuntimeError` on a bare engine; parallelism is rayon's
  (`RAYON_NUM_THREADS`).

### Compatibility
- Shipped rule assets (`dev_7-3` etc.) are unchanged by this release; they remain the frozen
  reference that downstream training pipelines pin for reproducibility. A re-mine with the
  hardened checker is a separate, explicitly-versioned artifact.


## 0.4.1 — 2026-07-02 — find_rules works with the Rust core + safe concurrent asset installs

### Fixed
- **`find_rules` now mines natively on the Rust core.** With `_core` attached (any engine from
  `load()` / `from_config()`), the fork-based Python pool mined **0 rules**: the workers mutate
  Python-side rule state while `simplify` runs on the immutable core — the same class of bug as the
  0.4.0 `prune_redundant_rules` fix. Phase 2 now delegates to the native mine when the core is
  attached (candidate library + per-length `mine_one_length` + `set_rules` between lengths,
  rayon-parallel; cap with `RAYON_NUM_THREADS`; `n_workers` only applies to the pure-Python
  fallback). Mined rules are now always canonicalized to the wildcard (`_j`) form.
- **`find_rules(X=<ndarray>)`** no longer raises `NameError` (the documented array form was never
  assigned to the internal data variable).
- **Concurrent cold-cache asset installs are now safe.** Installs serialize per asset via a
  `FileLock`, and an asset only counts as installed when **every** file in its manifest is present,
  so a partially-downloaded asset is correctly treated as not installed. New dependency: `filelock`.

### Changed
- Internal offline-mining scaffolding (mine grid and validation drivers) moved out of the
  released repo.

## 0.4.0 — 2026-07-01 — native offline rule-miner + simplify fixes + CLI fix

### Added
- **Native (Rust-core) offline rule mining.** An all-cores native mine driver (with a grid timing
  harness) supersedes the pure-Python `find_rules` for offline discovery against the Rust `_core`.
- **Closed-form `pow(C, x)` / `pow(x, C)` log-linearization** in the offline pipeline.

### Fixed
- **`sort_operands` is now idempotent** — canonical operand order is a fixpoint (rotation iterated to
  convergence; mask-before-sort), so simplifying a simplified expression no longer changes it. This can
  change the canonical operand ordering of some expressions vs 0.3.x (verified not to affect the
  downstream symbolic-data / flash-ansr / srbf test goldens).
- **`prune_redundant_rules` corrected against the Rust core** (the pure-Python prune produced 0 rules
  with `_core`); offline grid mining caps sources at length ≤ i and chunks the per-config budget.
- **`simplipy install <name>` / `simplipy remove <name>` now work.** They previously passed the asset
  *type* as the asset name (and the name as `force`/`quiet`), so every invocation raised
  `ValueError: Unknown asset: 'engine'`. They now take just an asset name with clean error handling
  (message + exit 1, no traceback); the vestigial `--type` flag is dropped from `install`/`remove`
  (`list` keeps it).

### Docs
- Documented the normalization helpers in the API reference; added a CLI reference for
  `list` / `prune-rules` / `resolve-rules` / `install` / `remove`; qualified `SimpliPyEngine`
  methods in the component overview; named `symbolic-data` as the direct downstream in the family DAG.

## 0.3.1 — 2026-06-28 — expression-token normalization helpers

Adds pure-Python `normalize_skeleton`, `normalize_expression`, and `normalize_variable_token`
to the package root (also available as `simplipy.normalization`). These canonicalize a prefix
token sequence -- variable tokens (`v1`/`x1`, case-insensitive) to a stable `x{n}`, and numeric
literals to a `<constant>` placeholder (skeleton form) or kept as-is (expression form) -- so two
expressions that are "the same" up to variable renaming / constant values compare equal.

Relocated from flash-ansr (behavior-identical) so the canonicalizer lives at the shared
expression-engine leaf that downstream packages (symbolic-data, flash-ansr, srbf) all depend on,
keeping holdout-matching and symbolic-recovery scoring consistent by construction. No change to
the Rust inline backend (`simplipy._core`) or any existing API; purely additive.

## 0.3.0 — 2026-06-21 — Rust inline backend + the "numeric" engine line (MAJOR behavior change)

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
2. **Six conversion bug fixes**, most notably:
   a fractional-power child (`pow1_M`) is no longer silently dropped by `convert_expression`; left/right
   associativity in `infix_to_prefix`/`prefix_to_infix` is corrected (and round-trip-preserving);
   `x**0`, `--5`, `^`-vs-`**` unary-minus, and raw `powN` are handled correctly.
3. **`float("inf")` / `float("-inf")` / `float("nan")` are atomic tokens** in the conversion tokenizer,
   so a folded constant round-trips through prefix↔infix instead of fragmenting.

> Reproducibility: to reproduce results generated with the pre-0.3 engine (e.g. models trained against
> `dev_7-3` under SimpliPy 0.2.x), pin `simplipy<0.3`. The 0.2.x behavior is the frozen anchor; 0.3 is a
> deliberate, documented quality improvement, not a silent drift.

### Robustness

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
- **New runtime dependency:** `platformdirs` (resolves the per-user asset cache directory).
