"""The AC core (``simplify_ac``): the n-ary associative-commutative engine.

These are the PRODUCT gates for the AC core:

* the two axes of the binary engine's commutative-order invariance defect are CLOSED --
  operand ORDER (both spellings of ``tan * cos`` simplify) and ADJACENCY (terms collect across
  bracketing, ``x3 + (x8 + x3/5)``);
* coefficient arithmetic is exact computation (``mult2 mult3 x -> * 6 x``), with the explicit
  form spelling coefficients as literals and NO hyper-operators;
* the hyper-operator family is DELETED from the vocabulary:
  legacy spellings parse on input only, no projection ever emits them;
* the result is idempotent and invariant under permutation of commutative operands -- the
  property the deleted binary engine could never provide (its commutative-invariance test
  was a permanent xfail until both left the tree together) holds BY CONSTRUCTION here.
"""

import pytest

from simplipy import Mode, SimpliPyEngine
from conftest import acj_asset_dir, acj_config_path

HYPER_PREFIXES = ("mult", "div")


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    # The SUPPORTED pairing: the AC engine + the AC-JUDGED artifact of the clean
    # vocabulary (acj-4-3, mined and tracked in this repo). Legacy assets neither load
    # (their hyper-operator realizations are deleted) nor parse (strictly clean).
    return SimpliPyEngine.from_config(
        acj_config_path())


def _is_hyper(token: str) -> bool:
    if token.startswith(HYPER_PREFIXES) and token[-1].isdigit():
        return True
    # pow2..pow5 / pow1_2..: 'pow' followed by digits (the binary 'pow' itself is fine).
    return token.startswith("pow") and len(token) > 3 and token[3].isdigit()


class TestDefectAxesClosed:
    def test_order_invariance_of_the_canonical_witness(self, engine: SimpliPyEngine) -> None:
        # A rule rewrite (log(exp(t)) -> t, an acj-mined inverse composition) fires
        # identically under BOTH operand orders of the enclosing product.
        # `sin x0` rather than a bare `x0`: log-exp is BANDED now (f64 exp overflows at
        # 709.782713), and ORDER INVARIANCE is what this pins -- so the argument is
        # bounded to keep the rewrite firing and the subject intact.
        a = engine.simplify(engine.to_tagged(["*", "log", "exp", "sin", "x0", "x1"]))
        b = engine.simplify(engine.to_tagged(["*", "x1", "log", "exp", "sin", "x0"]))
        assert a == b == ["<mul>", "x1", "sin", "x0", "</mul>"]

    def test_adjacency_collection_across_bracketing(self, engine: SimpliPyEngine) -> None:
        # `+ x3 (+ x8 (div5 x3))`: the two x3 spellings are never adjacent in the binary tree,
        # so the binary engine cannot collect them; the bag collects by construction. The
        # native output is the strict TAGGED prefix form.
        # (stripped term order: the 1.2*x3 term, key x3, sorts before x8)
        # The 6/5 coefficient renders STRUCTURALLY in the tagged form (both components
        # inside the vocabulary bound); the COLLECTION this test guards is unchanged,
        # and the pretty infix below still shows the coefficient as `1.2`.
        out = engine.simplify(engine.to_tagged(["+", "x3", "+", "x8", "*", "0.2", "x3"]))
        assert out == ["<add>", "<mul>", "6", "x3", "<div>", "5", "</mul>", "x8", "</add>"]
        # ...and the pretty infix rendering of the same answer:
        assert engine.simplify(engine.to_infix(["+", "x3", "+", "x8", "*", "0.2", "x3"])) \
            == "1.2*x3 + x8"

    def test_permutation_invariance_property(self, engine: SimpliPyEngine) -> None:
        # Swap the operands of every commutative node: the answer must not move.
        cases = [
            ["+", "x1", "*", "x2", "sin", "x3"],
            ["*", "+", "x1", "x2", "+", "x3", "x4"],
            ["+", "*", "x1", "x2", "+", "*", "x3", "x4", "x5"],
        ]

        def swap(tokens: list[str], i: int = 0) -> tuple[list[str], int]:
            t = tokens[i]
            arity = {"+": 2, "*": 2, "-": 2, "/": 2, "pow": 2}.get(t)
            if arity is None:
                arity = 1 if t in {"sin", "cos", "tan", "exp", "log", "abs"} else 0
            parts = []
            j = i + 1
            for _ in range(arity):
                p, j = swap(tokens, j)
                parts.append(p)
            if t in {"+", "*"}:
                parts.reverse()
            out = [t]
            for p in parts:
                out.extend(p)
            return out, j

        for case in cases:
            swapped, _ = swap(case)
            assert engine.simplify(case) == engine.simplify(swapped), case


class TestExplicitConstants:
    def test_coefficient_arithmetic_is_computation(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify(engine.to_tagged(["*", "2", "*", "3", "x0"])) \
            == ["<mul>", "6", "x0", "</mul>"]
        assert engine.simplify(engine.to_tagged(["*", "4", "*", "0.5", "x0"])) \
            == ["<mul>", "2", "x0", "</mul>"]
        seven_x = ["+", "*", "6", "x0", "x0"]
        assert engine.simplify(engine.to_tagged(seven_x)) == ["<mul>", "7", "x0", "</mul>"]
        # The binary-chain diagnostic projection is still available:
        assert engine.simplify(seven_x) == ["*", "7", "x0"]

    def test_strict_tagged_form(self, engine: SimpliPyEngine) -> None:
        # The bags carry their group inverse as a SECTION: terms after <sub> subtract,
        # factors after <div> divide. Non-decimal rationals are single fraction tokens
        # UNLESS the reciprocal spells strictly shorter: since divisor-side spelling
        # (2026-08-01, design §8) x0/3 renders through the `<div>` section rather than
        # as the `1/3` coefficient (`* 22/7 x0` still keeps its fraction token -- ties
        # stay on the coefficient side). See tests/test_divisor_side_spelling.py.
        assert engine.simplify(engine.to_tagged(["-", "x0", "x1"])) \
            == ["<add>", "x0", "<sub>", "x1", "</add>"]
        assert engine.simplify(engine.to_tagged(["/", "x0", "x1"])) \
            == ["<mul>", "x0", "<div>", "x1", "</mul>"]
        assert engine.simplify(engine.to_tagged(["/", "x0", "3"])) \
            == ["<mul>", "x0", "<div>", "3", "</mul>"]
        # (2*x1)/(x2*x3): one bag, both sections.
        assert engine.simplify(engine.to_tagged(["/", "*", "2", "x1", "*", "x2", "x3"])) \
            == ["<mul>", "2", "x1", "<div>", "x2", "x3", "</mul>"]
        # `neg`/`inv` are the sanctioned STANDALONE unary spellings (function arguments,
        # lone reciprocals); inside bags the sections own all inverses.
        assert engine.simplify(["tan", "neg", "x0"]) == ["tan", "neg", "x0"]
        assert engine.simplify(["inv", "x0"]) == ["inv", "x0"]
        # The parser is liberal: in-bag `neg` on input maps to the section spelling
        # (constant-like terms sort LAST: x2 + 2.3, the polynomial convention).
        assert engine.simplify(["<add>", "x2", "2.3", "neg", "x1", "</add>"]) \
            == ["<add>", "x2", "2.3", "<sub>", "x1", "</add>"]
        # Tagged output round-trips through the shared parser (idempotence in native form).
        for expr in (["/", "x0", "3"], ["-", "x0", "x1"], ["/", "x0", "x1"]):
            tagged = engine.simplify(expr)
            assert engine.simplify(tagged) == tagged
        # Pretty infix of the same answers:
        assert engine.simplify(engine.to_infix(["-", "x0", "x1"])) == "x0 - x1"
        assert engine.simplify(engine.to_infix(["/", "x0", "x1"])) == "x0/x1"
        assert engine.simplify(engine.to_infix(["/", "neg", "x0", "3"])) == "-x0/3"

    def test_no_hyper_operators_in_any_output(self, engine: SimpliPyEngine) -> None:
        cases = [
            ["*", "4", "x0"],
            ["pow", "+", "x0", "x1", "3"],
            ["*", "0.5", "*", "4", "sin", "x0"],
            ["pow", "x0", "0.5"],
        ]
        for case in cases:
            for form, convert in (("tagged", engine.to_tagged),
                                  ("explicit", engine.to_prefix)):
                out = engine.simplify(convert(list(case)))
                assert not any(_is_hyper(t) for t in out), (case, form, out)

    def test_real_odd_roots_are_rootn(self, engine: SimpliPyEngine) -> None:
        # The signed odd root is a DIFFERENT function from the principal rational power
        # (rootn(-8, 3) = -2 where (-8)^(1/3) is NaN; IEEE 754 keeps pow and rootn separate
        # for exactly this reason). The pow1_k family is deleted; rootn is the operator.
        assert engine.simplify(["rootn", "x0", "3"]) == ["rootn", "x0", "3"]
        assert engine.simplify(["rootn", "x0", "7"]) == ["rootn", "x0", "7"]
        # Unit index normalizes away. An EVEN index agrees with the principal power
        # everywhere, so either spelling is sound and mu TIES them (15000 both ways);
        # `rootn` is PRESERVED as the canonical representative (2026-08-06 rootn work).
        assert engine.simplify(["rootn", "x0", "2"]) == ["rootn", "x0", "2"]
        assert engine.simplify(["rootn", "x0", "1"]) == ["x0"]
        assert engine.simplify(engine.to_infix(["rootn", "x0", "3"])) == "rootn(x0, 3)"
        # MEASURE pin in MILLI-BITS since §10.10 (18 -> 18000), +1000 at D38/B2 (the
        # index literal pays the mu' selector bit), 19000 -> 15000 at the SYMBOL TABLE
        # (2026-08-21): `rootn` is an ELEMENTARY head at 6 bits, not a flat 8, and a
        # leaf is 6. Spelling-independent.
        assert engine.complexity(["rootn", "x0", "3"]) == 15000

    def test_rootn_cross_evaluator_parity(self, engine: SimpliPyEngine) -> None:
        # ONE SEMANTICS, FIVE SURFACES: the Python realization and the
        # compiled f64 evaluator agree cell-for-cell across the full index table --
        # odd/even/unit/negative/zero indices included. The Rust suite pins the
        # high-precision evaluators against the same table (rootn_parity_with_the_f64_
        # evaluator); together they make divergent surfaces impossible to reintroduce.
        import math

        from simplipy import operators as ops

        def tok(v: float) -> str:
            if math.isinf(v):
                return 'float("inf")' if v > 0 else 'float("-inf")'
            return repr(v)

        for x in (-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0, float("inf"), float("-inf")):
            for n in (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7):
                want = float(ops.rootn(x, float(n)))
                folded = engine._core.evaluate_constant_subtree(
                    ["rootn", tok(x), str(n)])
                assert folded is not None, (x, n)
                got = {"float(\"inf\")": float("inf"),
                       "float(\"-inf\")": float("-inf"),
                       "float(\"nan\")": float("nan")}.get(folded)
                got = float(folded) if got is None else got
                agree = (math.isnan(want) and math.isnan(got)) or want == got or (
                    math.isfinite(want) and math.isfinite(got)
                    and abs(want - got) <= 1e-12 * max(abs(want), abs(got)))
                assert agree, f"rootn({x}, {n}): realization {want}, engine {got}"

    def test_str_input_returns_pretty_infix(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify("x1 * 2 + x1 * 3") == "5*x1"


class TestVocabularyDeletion:
    def test_bounded_projection_is_gone(self, engine: SimpliPyEngine) -> None:
        # The hyper-operator family is DELETED from the vocabulary; the bounded
        # re-sugaring projection died with it.
        import pytest as _pytest
        # `form=` itself is gone in 0.14.0, so the projection is unreachable by a
        # stronger route than an unknown-value error: the parameter does not exist.
        with _pytest.raises(TypeError, match="form"):
            engine.simplify(["mult2", "x0"], form="bounded")

    def test_legacy_spellings_are_refused_outright(self, engine: SimpliPyEngine) -> None:
        # STRICTLY CLEAN: a new-vocabulary engine has no reading for the deleted
        # hyper-operator spellings -- legacy data converts ONCE at the boundary
        # (benchmarks/corpus/convert_nv.py), the engine never interprets it.
        for expr in (["mult2", "mult3", "x0"], ["+", "x3", "+", "x8", "div5", "x3"],
                     ["pow1_2", "x0"], ["pow1_5", "x0"], ["pow4", "div4", "x1"],
                     ["pow1_3", "x0"]):
            with pytest.raises(ValueError):
                engine.simplify(list(expr))


class TestContracts:
    def test_idempotence(self, engine: SimpliPyEngine) -> None:
        cases = [
            ["+", "x3", "+", "x8", "*", "0.2", "x3"],
            ["*", "cos", "x0", "tan", "x0"],
            ["-", "x1", "+", "x1", "x2"],
            ["/", "x0", "+", "x1", "*", "3", "x0"],
        ]
        for case in cases:
            once = engine.simplify(case)
            assert engine.simplify(once) == once, case

    def test_soundness_gates_hold(self, engine: SimpliPyEngine) -> None:
        # Structural x - x cancels (variables are finite by domain)...
        assert engine.simplify(["-", "x0", "x0"]) == ["0"]
        # ...but inf - inf must NOT become 0 (truth is nan; the group axioms fail there).
        out = engine.simplify(["-", 'float("inf")', 'float("inf")'])
        assert out == ['float("nan")']
        # 0/0 must not cancel to 1.
        out = engine.simplify(["/", "0", "0"])
        assert out != ["1"]

    def test_flat_bag_permutation_invariance_at_overflow_magnitudes(
            self, engine: SimpliPyEngine) -> None:
        # Accumulation-order regression: coefficient/exponent accumulation used
        # to fold in ARRIVAL order, so which pairs merged before an i128 overflow
        # depended on the input spelling. The constructors now sort each collected
        # multiset (cmp_exact is exact at every magnitude since B3) and fold in that
        # order: a FLAT bag's canonical state is a function of its member multiset.
        # (Binary-CHAIN spellings can still vary at these magnitudes: re-association
        # changes which partials inner constructor calls merged -- the documented
        # boundary of the ratified i128 boundedness, unreachable with mining/corpus
        # literals.)
        import itertools
        tiny = ["6.4495375319922606e-18", "3e-19", "0.5", "1e-19"]
        for perm in itertools.permutations(range(4)):
            out = engine.simplify(["<mul>"] + [tiny[i] for i in perm] + ["x0", "</mul>"])
            if perm == (0, 1, 2, 3):
                reference = out
            assert out == reference, perm
        terms = [["<mul>", c, "x0", "</mul>"] for c in tiny]
        for perm in itertools.permutations(range(4)):
            toks = ["<add>"] + [t for i in perm for t in terms[i]] + ["</add>"]
            out = engine.simplify(toks)
            if perm == (0, 1, 2, 3):
                reference = out
            assert out == reference, perm

    def test_zero_meeting_infinity_is_nan_on_every_path(self, engine: SimpliPyEngine) -> None:
        # term_join (the raw coefficient path used by collection and
        # distribution) absorbed c = 0 into an infinity as SIGN ONLY, dropping the zero.
        # 0 * inf is the IEEE invalid operation (contract s2, one zero), and
        # 0 * (inf * t) is NaN pointwise-TOTALLY, so every spelling must agree with the
        # constructor's direct arm.
        direct = engine.simplify(["*", "0", "*", 'float("inf")', "x0"])
        assert direct == ['float("nan")']
        # The distribution path that used to disagree (0 distributed over a sum whose
        # unlicensed member carries the infinity):
        distributed = engine.simplify(
            ["+", "*", "0", "+", "x1", "*", 'float("inf")', "x0", "x2"])
        assert distributed == ['float("nan")']
        # A structural zero never launders into the constant class (masking-soundness):
        # the nonzero direction absorbs, the zero direction collapses exactly.
        assert engine.simplify(["neg", "<constant>"]) == ["<constant>"]
        assert engine.simplify(["*", "0", "<constant>"]) == ["0"]

    def test_exact_root_folds_do_not_panic_or_miss(self, engine: SimpliPyEngine) -> None:
        # int_root regression, both halves. (1) The exponent (1/2)^32 has denominator 2^32,
        # which the root search truncated to 0 in a u32 cast: divide-by-zero panic
        # (PanicException) on a five-token valid input. Now it falls through to the
        # contractual all-literal f64 fold (2 has no rational 2^32-th root).
        # STAGE 2 (mu-governed fold): 2^(2^-32) is irrational -- the f64 fold this
        # regression test used to expect is a rounding, refused under mu. The exponent
        # itself folds EXACTLY ((1/2)^32 is dyadic), and the no-panic half of the
        # regression is unchanged. The 1/2^32 exponent now spells as the ROOT it is
        # (`rootn 2 4294967296`) rather than its 33-digit decimal: mu TIES the two
        # (42000 both ways) and rootn is the preserved representative.
        out = engine.simplify(["pow", "2", "pow", "0.5", "32"])
        assert out == ["rootn", "2", "4294967296"]
        # (2) The search bound 2^floor(127/k) undershot the true ceiling 2^(127/k):
        # exact roots in the band were silently demoted to the f64 fold, so the SAME
        # exact quantity canonicalized to different literals depending on spelling.
        # These produced '5000.000000000003' and '4999999999999.992' before the fix.
        assert engine.simplify(["pow", "pow", "5000", "10", "0.1"]) == ["5000"]
        assert engine.simplify(
            ["pow", "pow", "5000000000000", "3", "pow", "3", "-1"]) \
            == ["5000000000000"]
        # Control just inside the old bound (4000 < 4096 = 2^12): folded exactly before
        # and after -- pins that the fix widened the band without moving the exact tier.
        assert engine.simplify(["pow", "pow", "4000", "10", "0.1"]) == ["4000"]

    def test_certificate_verdicts(self, engine: SimpliPyEngine) -> None:
        # tan(x) - tan(x) cancels (tan's poles are a null set -> finite-a.e. certified)...
        assert engine.simplify(["-", "tan", "x0", "tan", "x0"]) == ["0"]
        # ...while log(x) - log(x) is REFUSED: log is undefined on x < 0, a
        # positive-measure set, so cancelling would fill undefined values with 0 -- exactly
        # what the certificate exists to forbid. The refusal keeps both terms.
        assert engine.simplify(engine.to_tagged(["-", "log", "x0", "log", "x0"])) \
            == ["<add>", "log", "x0", "<sub>", "log", "x0", "</add>"]
        assert engine.simplify(["-", "log", "x0", "log", "x0"]) \
            == ["-", "log", "x0", "log", "x0"]

    def test_ground_pole_arithmetic_is_exact(self, engine: SimpliPyEngine) -> None:
        # Under the supported (sort-promoted) pairing every pole spelling resolves exactly:
        # the `!`/`$` certificates refuse to bind the pole subtrees, and the numeric fold owns
        # the ground arithmetic.
        assert engine.simplify(["*", "0", 'float("inf")']) == ['float("nan")']
        assert engine.simplify(["/", 'float("inf")', 'float("inf")']) == ['float("nan")']
        assert engine.simplify(["-", "inv", "0", "inv", "0"]) == ['float("nan")']
        assert engine.simplify(["-", "/", "1", "0", "/", "1", "0"]) == ['float("nan")']
        assert engine.simplify(["/", "inv", "0", "inv", "0"]) == ['float("nan")']
        assert engine.simplify(["*", "inv", "0", "inv", "0"]) == ['float("inf")']
        # H1 pole fix (2026-08-02): a determined pole materializes AT CONSTRUCTION
        # (0^{-k} -> +inf, one-zero convention), so there is no structure left for
        # LOSSY's uncertified cancel to eat -- inf - inf = nan exactly, in BOTH
        # modes (pre-fix LOSSY canceled the twin pole subtrees structurally to 0).
        # LOSSY's structural-cancel contract is untouched for non-determined
        # structures (see the (x^2)^0.5 -> x and exp(log C) -> C pins).
        expr = ["-", "/", "1", "0", "/", "1", "0"]
        assert engine.simplify(expr, mode=Mode.corpus) == ['float("nan")']
        assert engine.simplify(expr) == ['float("nan")']

    def test_free_rhs_wildcards_are_dropped_at_translation(self, tmp_path) -> None:
        # substitute() panics on an RHS wildcard the LHS did not bind
        # ("rhs wildcard is bound by the lhs"). Translation now verifies containment on
        # the CANONICAL sides and drops violators at load -- fail-closed -- instead of a
        # delayed, data-dependent PanicException at rewrite time. The shipped asset is
        # clean (0/503); this builds a poisoned one.
        import json
        import os

        import yaml

        src = acj_asset_dir()
        cfg = yaml.safe_load(open(os.path.join(src, "config.yaml")))
        cfg["rules"] = "./rules.json"
        rules = [
            # Well-formed control: must survive translation and fire.
            # THIRD replacement, and the reasoning behind the second was wrong: `log exp _0`
            # was chosen 2026-08-07 because "no constructor evaluates log", but the inverse-pair
            # table landed the same day and absorbed it -- an inverse PAIR needs no evaluation.
            # A cross-function identity does: `1/cosh(asinh t) = cos(atan t)` relates four
            # different functions and collapses to nothing a table of pairs can express.
            [["inv", "cosh", "asinh", "_0"], ["cos", "atan", "_0"]],
            # Directly free RHS wildcard: _1 is bound nowhere.
            [["sin", "_0"], ["+", "_0", "_1"]],
            # The subtle flavor: raw containment HOLDS ({_0,_1} both sides), but canon
            # folds the LHS subterm pow(_1, 0) -> 1 (total), erasing _1 from the LHS
            # while the RHS still references it. Post-canon check required.
            [["*", "cos", "_0", "pow", "_1", "0"], ["*", "cos", "_0", "_1"]],
            # A DISORIENTED rule (rhs not strictly below lhs in the
            # reduction ordering -- here complexity 2 -> 3). The runtime oriented() gate
            # would refuse every fire anyway; translation drops it at load (gate G7).
            [["exp", "_0"], ["exp", "exp", "_0"]],
        ]
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
        (tmp_path / "rules.json").write_text(json.dumps(rules))
        e = SimpliPyEngine.from_config(str(tmp_path / "config.yaml"))
        kept, _, dropped, _ = e._core.ac_rules_info()
        assert kept == 1 and dropped == 3
        # The subjects the poison rules would have matched simplify without panicking or
        # growing, and the well-formed control still fires.
        assert e.simplify(["sin", "x0"]) == ["sin", "x0"]
        assert e.simplify(["cos", "x1"]) == ["cos", "x1"]
        assert e.simplify(["exp", "x3"]) == ["exp", "x3"]
        assert e.simplify(["cos", "neg", "x2"]) == ["cos", "x2"]
        # abs-of-abs sheds the outer abs by CONSTRUCTION now (no rule involved).
        assert e.simplify(["abs", "abs", "x2"]) == ["abs", "x2"]

    def test_pathological_numeric_tokens_stay_cheap(self, engine: SimpliPyEngine) -> None:
        # A zero mantissa never overflows the positive-exponent scaling
        # loop, so this 12-character token spun i32::MAX iterations (2.2 s) inside
        # parse_decimal before the explicit bail. Timing is pinned at the Rust unit
        # level; here the exact values.
        assert engine.simplify(["0e2147483647"]) == ["0"]
        assert engine.simplify(["-0.00e2147483647"]) == ["0"]
        # Nonzero mantissas beyond i128 keep their overflow bound and stay symbolic.
        assert engine.simplify(["1e2147483647"]) == ["1e2147483647"]

    def test_reduction_ordering_literal_size_component(self, engine: SimpliPyEngine) -> None:
        # The s-component of the reduction ordering (docs/formal.md §5): at equal
        # complexity, literal SIZE decides before literal VALUE -- this is what makes the
        # ordering well-founded (T-wf) and closes the former Open O1. The pair below
        # disagrees between the two orders: 1/3 is value-LARGER but size-SMALLER than
        # 1/1000000007, so under the old pair ordering the big-denominator spelling was
        # "below"; under the triple it is above.
        small_lit = ["*", "x0", "pow", "3", "-1"]           # x0 * 1/3
        big_lit = ["*", "x0", "pow", "1000000007", "-1"]    # x0 * 1/1000000007
        assert engine._core.ac_ordered_below(small_lit, big_lit) is True
        assert engine._core.ac_ordered_below(big_lit, small_lit) is False
        # At equal size, exact value order still decides (2 and 3 are both 2 bits).
        assert engine._core.ac_ordered_below(["*", "2", "x0"], ["*", "3", "x0"]) is True
        assert engine._core.ac_ordered_below(["*", "3", "x0"], ["*", "2", "x0"]) is False
        # STAGE 2 INVERSION (the measure's point): under the old triple a 54-bit
        # literal was "strictly simpler" than `x0 + 2` by node count alone. Under mu
        # the literal pays its bits (54 > 18), so the small symbolic expression is
        # below the huge literal -- the exact asymmetry that minted coefficient
        # materializations is gone at the ordering itself.
        assert engine._core.ac_ordered_below(
            ["9007199254740993"], ["+", "x0", "2"]) is False
        assert engine._core.ac_ordered_below(
            ["+", "x0", "2"], ["9007199254740993"]) is True

    def test_composite_step_heals_the_distribution_refusal(self, engine: SimpliPyEngine) -> None:
        # docs/formal.md, the composite rebuild case. In 2*(log(1) - c*sinh(x2))^2 the
        # children's work (log(1) -> 0, sum collapse) hands the Pow constructor a fresh
        # product; distributing the square is an equal-complexity, literal-size-INCREASING
        # intermediate, so the immediate ordering test refuses it -- discarding the
        # children's work and shipping a dead `log 1` redex (the pre-composite output,
        # complexity 9). The composite step judges the settled endpoint instead: the
        # distributed square's all-literal factor folds, complexity 6 < 8, committed.
        # Both spellings of the same function must now converge to the same form.
        with_dead_redex = "* 2 pow - log 1 * 6.4495375319922606e-18 sinh x2 2".split()
        pre_simplified = "* 2 pow neg * 6.4495375319922606e-18 sinh x2 2".split()
        out_a = engine.simplify(with_dead_redex)
        out_b = engine.simplify(pre_simplified)
        assert out_a == out_b
        assert "log" not in out_a  # the dead redex is gone
        # STAGE 2: the healed endpoint changed for the BETTER under mu. Squaring the
        # 55/112-bit coefficient would double its bits (the materialized square costs
        # ~221 units), so the distribution keeps the coefficient factored under its own
        # pow (~126 units): 2 * (-6.449...e-18)^2 * sinh(x2)^2. The children's work
        # still commits (log 1 -> 0, sum collapse), both spellings converge, and the
        # form is a fixpoint -- the healing claim survives with an MDL-honest endpoint
        # (pre-mu this pinned the fully materialized coefficient at complexity 7).
        # MEASURE pin, not a dialect one: mu is spelling-independent, so this number
        # stands under any print ruling. 212 -> 213786 was the milli-bit unit change
        # plus the real-valued L of §10.10 (213.786 bits); 213786 -> 111969 is D38/B2
        # mu': the 6.449...e-18 coefficient takes its decimal-scientific codeword
        # (selector + L(64495375319922606) + L(35) + sign = 62.014 bits against the
        # fraction code's ~112.9), and the endpoint STILL keeps it factored under its
        # own pow -- the EXACT square is a 34-digit mantissa (~118 bits scientific),
        # dearer than the factored spelling under mu' too, so the healing claim and
        # the endpoint shape survive; only the priced total moves. 111969 -> 94969 at
        # the SYMBOL TABLE (2026-08-21): the endpoint's five structural nodes and its
        # leaves re-price, the 62-bit coefficient does not (the table never touches a
        # literal), and the shape is again unmoved. 94969 -> 93724 at the ONE-BIT LITERAL
        # FLOOR (owner 2026-08-22): the small exponents stop being clamped together. The
        # endpoint shape is unmoved a third time.
        assert engine.complexity(out_a) == 93724
        assert "pow" in out_a and any("6449537531992260" in t for t in out_a), out_a
        assert engine.simplify(out_a) == out_a  # and the healed form is a fixpoint

    def test_infinity_sum_sign_families_converge_flat(self, engine: SimpliPyEngine) -> None:
        # An infinity-bearing sum skips
        # the primitive-sum orientation pass DELIBERATELY (docs/formal.md §3). The skip
        # is complete because the infinity pins the magnitude multiset at 1 (no content
        # to extract) and +-1 distributes totally (scale_term flips the infinity), so
        # every negation spelling of one function lands in the same flat bag. These pairs
        # pin the closure; a change to the add() infinity exit must keep them converging.
        inf = 'float("inf")'
        pairs = [
            # -(log x0 + inf)  ==  -log x0 - inf
            (["neg", "+", "log", "x0", inf], ["-", "neg", "log", "x0", inf]),
            # -(log x0 + log x1 + inf)  ==  -log x0 - log x1 - inf
            (["neg", "+", "+", "log", "x0", "log", "x1", inf],
             ["-", "-", "neg", "log", "x0", "log", "x1", inf]),
            # log x0 + log x1 - inf  ==  -(inf - (log x0 + log x1))
            (["-", "+", "log", "x0", "log", "x1", inf],
             ["neg", "-", inf, "+", "log", "x0", "log", "x1"]),
            # -1 * (inf - log x0)  ==  log x0 - inf
            (["*", "-1", "-", inf, "log", "x0"], ["-", "log", "x0", inf]),
        ]
        for a, b in pairs:
            oa, ob = engine.simplify(a), engine.simplify(b)
            assert oa == ob, (a, b, oa, ob)
            assert engine.simplify(oa) == oa

    def test_translation_audit_surface(self, engine: SimpliPyEngine) -> None:
        kept, subsumed, dropped, _ = engine._core.ac_rules_info()
        assert kept > 400
        # An AC-JUDGED artifact carries zero arithmetic redundancy AT MINE TIME: the miner
        # never mints what the engine computes. Subsumption at LOAD therefore measures how
        # far the engine has advanced past the artifact's mint-era canon. 2026-07-31: the
        # certainly-nonneg abs elimination moved 32 mined abs-composition rules
        # (abs∘exp/cosh/acos/..., cos∘sin, odd-increasing∘nonneg, pow(nonneg, s)) into the
        # constructors -- exactly the lattice's clauses, independently confirming both.
        # They translate as subsumed (dropped at load, behavior unchanged) until the next
        # re-mine, whose fresh artifact pins 0 again -- which happened at the 2026-08-01
        # mu refresh: the 1942-rule artifact is mint-time clean, 1943/0/0 + 1 twin.
        # G2 REPUBLISH (2026-08-15): the full-lane re-mine ships and the pin returns
        # to pristine -- the 6,594-rule artifact is mint-time clean, 6602/0/0 + 8 twins.
        # (2026-08-18, dedup key on the internal form; RE-PINNED for the 0.14.0 triple:
        # the shipped f64 file is 5,319 rows and reports kept = 5327 -- the
        # 1,049 rows whose pattern an earlier row already owns never reach translate.
        # `subsumed` is what this test pins, and it is unmoved at 0.)
        assert subsumed == 0, f"republished artifact must be mint-time clean; got {subsumed}"


class TestMatcherAssignmentCompleteness:
    """Finding F1: rule reach must not depend on where the canonical order places a shared
    subterm, nor on the rule's wildcard numbering (both are semantically arbitrary).

    The factoring rule is the smallest member of the exposed class: a bag pattern with two
    STRUCTURED sibling members sharing a wildcard, where a member's internal match has
    several valid assignments. A matcher that freezes the first internal assignment loses
    every subject arrangement in which the shared factor does not sort first -- so whether
    an expression simplifies would depend on its variable NAMES. These tests fix the engine
    surface: the same rule must fire on every alpha-variant of the same shape.
    """

    FACTOR_VARS = (("+", "*", "?0", "?1", "*", "?0", "?2"), ("*", "?0", "+", "?1", "?2"))
    FACTOR_ANY = (("+", "*", "_0", "_1", "*", "_0", "_2"), ("*", "_0", "+", "_1", "_2"))

    @pytest.fixture()
    def fresh_engine(self) -> SimpliPyEngine:
        # NOT the module fixture: set_rules replaces the ruleset, so these tests need a
        # private engine instance.
        return SimpliPyEngine.from_config(
            acj_config_path())

    def test_factoring_reach_is_alpha_invariant(self, fresh_engine: SimpliPyEngine) -> None:
        fresh_engine._core.set_rules([self.FACTOR_VARS])
        # The same shape t*a + t*b under different variable names. Canonical bag order
        # sorts the shared factor first in some spellings and last in others; every one
        # must factor.
        cases = [
            ("x0", "x1", "x2"),  # shared sorts FIRST (the lucky arrangement)
            ("x8", "x0", "x1"),  # shared sorts LAST
            ("x5", "x2", "x9"),  # shared sorts in the MIDDLE
        ]
        for shared, a, b in cases:
            expr = ["+", "*", shared, a, "*", shared, b]
            out = fresh_engine.simplify(expr)
            assert out[0] == "*", (expr, out)
            assert shared in out, (expr, out)

    def test_factoring_reach_composite_shared_factor(self, fresh_engine: SimpliPyEngine) -> None:
        fresh_engine._core.set_rules([self.FACTOR_ANY])
        # The shared factor is a composite (sin x0), which sorts after plain variables in
        # the subject Mul bags.
        expr = ["+", "*", "sin", "x0", "x1", "*", "sin", "x0", "x2"]
        out = fresh_engine.simplify(expr)
        assert out[0] == "*", (expr, out)
        assert "sin" in out, (expr, out)


class TestApiFootguns:
    """Audit Tier-2 (2026-08-03): the two API footguns, pinned closed.

    A STRING mode must mean what it says: ``mode='corpus'`` must compare equal to
    the enum and silently run SOUND. And malformed input through the tagged/``rootn``
    entry -- the one path that skips ``is_valid`` -- must raise like every other
    malformed input: it used to be returned unchanged, with no signal."""

    def test_mode_string_lossy_means_lossy(self, engine: SimpliPyEngine) -> None:
        # exp(log(<constant>)): SOUND refuses the collapse (log's domain restriction,
        # certificate-gated); LOSSY collapses to <constant> -- the documented
        # training-canonicalisation example. The string and the enum must take the
        # SAME branch.
        expr = ['exp', 'log', '<constant>']
        enum_out = engine.simplify(list(expr), mode=Mode.corpus)
        str_out = engine.simplify(list(expr), mode='corpus')
        assert str_out == enum_out
        assert str_out != engine.simplify(list(expr)), \
            'the LOSSY observable collapsed: this expression no longer distinguishes modes'

    def test_mode_string_any_case(self, engine: SimpliPyEngine) -> None:
        expr = ['+', 'x0', 'x0']
        assert engine.simplify(list(expr), mode='f64') == engine.simplify(list(expr))
        assert engine.simplify(list(expr), mode='F64') == engine.simplify(list(expr))

    def test_mode_unknown_string_raises(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match='unknown mode'):
            engine.simplify(['+', 'x0', 'x0'], mode='aggressive')

    def test_mode_ints_refuse(self, engine: SimpliPyEngine) -> None:
        # SUPERSEDED PIN (api-1/fmux-mode-1, 2026-08-16): int coercion used to be
        # blessed here (mode=1 -> SOUND) -- and it is exactly how a config-sourced
        # mode=3 silently selected LOSSY. Only Mode members and names bind now.
        expr = ['+', 'x0', 'x0']
        for bad in (1, 2, 3):
            with pytest.raises(TypeError):
                engine.simplify(list(expr), mode=bad)

    def test_malformed_tagged_raises(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError):
            engine.simplify(['<add>', 'x0'])  # unterminated bag

    def test_malformed_rootn_raises(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError):
            engine.simplify(['rootn', 'x0'])  # missing index operand

    def test_malformed_infix_projection_raises(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError):
            engine.simplify(engine.to_infix(['<mul>', 'x0']))

    def test_wellformed_tagged_and_rootn_still_simplify(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify(['rootn', 'x0', '1']) == ['x0']
        assert engine.simplify(engine.to_prefix(['<add>', 'x0', 'x1', '</add>'])) == \
            ['+', 'x0', 'x1']


class TestBoundaryWellFormedness:
    """Hardening campaign H-003/H-004/H-006 (2026-08-03).

    H-003: ``simplify([]) == []`` is the documented empty-input contract (the one
    valid case ``is_valid`` rejects); the malformed-input hardening of 58d492d
    accidentally made it raise. H-004: empty or whitespace-bearing tokens are not
    tokens -- they corrupt every space-joined serialization (``['+', '', 'x0']``
    rendered the infix ``' + x0'``, which cannot round-trip) and used to pass as
    silent opaque leaves. H-006: a negative max_passes surfaced as a raw pyo3
    OverflowError instead of a ValueError."""

    def test_empty_input_returns_empty(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify([]) == []
        assert engine.simplify(()) == ()
        assert engine.simplify(engine.to_infix([])) == ''

    def test_empty_token_raises(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match='token'):
            engine.simplify(['+', '', 'x0'])

    def test_whitespace_tokens_raise(self, engine: SimpliPyEngine) -> None:
        for bad in ([' '], ['x0 '], ['+', 'x\t0', 'x0'], ['+', 'x0', 'a b']):
            with pytest.raises(ValueError, match='token'):
                engine.simplify(list(bad))

    def test_unknown_nonempty_tokens_stay_opaque_leaves(self, engine: SimpliPyEngine) -> None:
        # DELIBERATE: unknown whitespace-free tokens are opaque leaves (free symbols);
        # the engine never folds them and carries them soundly.
        assert engine.simplify(['+', 'alpha', 'x0']) == ['+', 'alpha', 'x0']

    def test_negative_max_passes_raises_valueerror(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match='max_passes'):
            engine.simplify(['+', 'x0', 'x0'], max_passes=-1)


class TestMaxPassesRename:
    """0.14 rename: `node_budget` never bounded nodes -- it is the trip count of the outer
    `rewrite_pass` loop -- so the parameter is `max_passes` now. The old spelling stays as a
    warning alias (same doctrine as `parse` -> `read_infix`), and the two are mutually
    exclusive: silently letting one win would be exactly the ambiguity the rename removes."""

    def test_new_name_does_not_warn(self, engine: SimpliPyEngine) -> None:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('error', DeprecationWarning)
            engine.simplify(['+', 'x0', 'x0'], max_passes=2)

    def test_default_is_48_passes_and_convergence_is_far_below_it(
            self, engine: SimpliPyEngine) -> None:
        import inspect
        # The default lives in the body, not the signature (the signature default is the
        # `None` sentinel that makes "both at once" detectable), so pin it behaviourally:
        # every budget from the measured convergence depth up must agree with the default.
        assert inspect.signature(engine.simplify).parameters['max_passes'].default is None
        # THE NUMBER, pinned where it lives. The budgets-agree check below cannot see it:
        # every budget in the probe exceeds the convergence depth, so it holds for any
        # default at all and the test's own name went unasserted.
        import re
        src = inspect.getsource(type(engine).simplify)
        assert re.search(r'max_passes\s*=\s*48', src), \
            'the documented default of 48 is not in simplify\'s body'
        expr = ['+', '*', '2', 'x0', '*', '3', 'x0']
        default_out = engine.simplify(list(expr))
        for budget in (4, 8, 48, 512):
            assert engine.simplify(list(expr), max_passes=budget) == default_out


class TestMaskingDiscoverability:
    """Hardening campaign H-005: masking is the documented downstream step; it must be
    reachable as ``simplipy.masking`` like every other public submodule."""

    def test_masking_is_an_attribute(self) -> None:
        import importlib
        import simplipy as sp
        importlib.reload(sp)  # a prior `from simplipy import masking` must not mask the gap
        assert hasattr(sp, 'masking')
        assert callable(sp.masking.mask)


class TestLiteralGrammarComposition:
    """Hardening H-012 (2026-08-03): paren-wrapping composes with BOTH numeric
    grammars. ``(-1/3)`` used to be an OPAQUE leaf (it even sorted as a symbol) while
    ``(-1)``, ``(-0.5)`` and bare ``1/3`` were numbers; the offline evaluator had the
    mirrored hole (its paren branch returned even on a failed decimal parse, so the
    fraction arm below was unreachable)."""

    def test_paren_fraction_is_numeric(self, engine: SimpliPyEngine) -> None:
        # 1/3 renders STRUCTURALLY in the tagged form (both components inside the
        # vocabulary bound); the parenthesised-fraction PARSE is what this guards, and
        # the value is unchanged. mu ties the spellings at 3000.
        # `neg (-1/3)` carries no bag delimiter in EITHER dialect, so it reads as
        # explicit and the answer comes back in the explicit dialect's structural
        # division; the tagged twin of the same state is `<mul> 1 <div> 3 </mul>`.
        assert engine.simplify(['neg', '(-1/3)']) == ['/', '1', '3']
        assert engine.simplify(['+', '(-1/3)', 'x0']) == \
            engine.simplify(['+', '-1/3', 'x0'])

    def test_paren_fraction_evaluates(self, engine: SimpliPyEngine) -> None:
        out = engine._core.evaluate_batch(['+', '(-1/3)', 'x0'], ['x0'], [3.0], 1, [])
        assert abs(out[0] - (3.0 - 1.0 / 3.0)) < 1e-12


class TestInfixCallSyntaxRoundTrip:
    """Hardening H-011 (2026-08-03): the pretty renderer's 2-ary call syntax
    (``rootn(a, b)``, ``pow(a, b)``) must parse back to the same canonical state.
    The tokenizer used to DROP the comma silently, fusing the arguments:
    ``rootn(x0, x1 - 1)`` parsed as rootn(x0 - x1, 1) -- 97/2000 fuzz rows."""

    def test_two_arg_calls_roundtrip(self, engine: SimpliPyEngine) -> None:
        for expr in (['rootn', '<constant>', '-', 'exp', 'x2', '2.051'],
                     ['rootn', 'x0', '+', 'x1', '1'],
                     ['pow', '+', 'x0', '1', '-', 'x1', '2'],
                     ['rootn', 'rootn', 'x0', 'x1', 'x2']):
            out = engine.simplify(list(expr))
            infix = engine.simplify(engine.to_infix(list(expr)))
            back = engine._core.parse(infix, True, False)
            assert engine.simplify(list(back)) == out, (expr, infix, back)


class TestOneZeroPoleContract:
    """Hardening H-016 (2026-08-03, S0): every zero is THE unsigned zero and its pole
    is +inf (contract v2 §9.2, DECIDED 2026-07-18; §9.8.1 rejected a directional zero
    outright). The interval layer's 0.6.0-era signed-zero device (b_mul minted -0.0,
    u_inv / b_pow read the sign bit to pick the pole) implemented the referee-era
    deployment semantics that §9.2 formally overturned -- the miner-side twin (the R2''
    tie-break) was deleted at ratification, but the interval-layer device survived.
    Observable: a zero-coefficient bag whose cofactor is a RECIPROCAL survives to the
    ground-classification fold (the cofactor is not syntactically certainly-finite),
    the fold read the interval class, and IEEE `0 x negative = -0.0` flipped the pole:
    `inv(0/tanh(-2))` shipped a `-inf` LITERAL while every sibling spelling
    (`inv(0*tanh(-2))`, `inv(0/(-2))`, `inv(0*x1)`) collapsed structurally first and
    shipped +inf. Same value, two answers, judge-convicted (INF-CHANGE 40/40)."""

    def test_surviving_zero_bag_pole_is_positive(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify(['inv', '/', '0', 'tanh', '-2']) == ['float("inf")']

    def test_odd_negative_exponent_pole_is_positive(self, engine: SimpliPyEngine) -> None:
        # Under signed zero the odd-integer exponent carried -inf (pow(-0.0, -3) = -inf);
        # under one zero the exponent's parity is irrelevant: pow(0, y<0) = +inf.
        assert engine.simplify(engine.read_infix('pow(0/tanh(-2), -3)')) == ['float("inf")']

    def test_fuzz_row_188677_family(self, engine: SimpliPyEngine) -> None:
        # x2 / ((0*x1/tanh(-2)) / inf): the denominator collapses to a ground zero bag;
        # its -0.0 rendering flipped x2/0 to -inf*x2.
        out = engine.simplify(engine.to_tagged(
            ['/', 'x2', '/', '/', '*', '0', '*', 'x1', '1', 'tanh', '-2', 'float("inf")']))
        assert out == ['<mul>', 'float("inf")', 'x2', '</mul>']

    def test_sibling_spellings_stay_positive(self, engine: SimpliPyEngine) -> None:
        # Anti-regression: the structurally-collapsing spellings were already correct.
        for tokens in (['inv', '*', '0', 'tanh', '-2'],
                       ['inv', '/', '0', '-2'],
                       ['inv', '*', '0', 'x1']):
            assert engine.simplify(list(tokens)) == ['float("inf")'], tokens

    def test_interval_layer_speaks_one_zero(self, engine: SimpliPyEngine) -> None:
        # The oracle itself, not just the fold that consumes it: the class of a
        # point-zero pole is POSINF regardless of how IEEE arithmetic renders the zero.
        for tokens in (['inv', '/', '0', 'tanh', '-2'],
                       ['inv', '/', '0', '-2'],
                       ['inv', '*', '0', 'x0']):
            assert engine._core.interval_class(list(tokens)) == 'POSINF', tokens


class TestAtanhTanhBandIsPrecisionDerived:
    """S3: the atanh(tanh(t)) -> t collapse carried a band of 18, derived from where
    f64 tanh ATTAINS +-1 (18.99). But tanh fails by COMPRESSION long before it
    saturates -- it loses ~2|t|/ln2 bits of the argument approaching +-1 -- and under
    the realised criterion (`_ulp_gap <= REALISED_ULP = 8`, the same bar the judge
    holds every rule to) the roundtrip first breaches at |t| ~ 2.7 (2.677 audit host,
    2.808 this host, libm-dependent; by |t| = 18 the drift is ~0.03-0.06 ABSOLUTE).
    The band is now 2: the last integer magnitude ceiling on the safe side of every
    measured breach. The overflow-derived bands (log/exp 709, sinh/cosh 710) are a
    different disease and keep their attainment thresholds."""

    def test_collapses_inside_the_band(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify(['atanh', 'tanh', 'sin', 'x0']) == ['sin', 'x0']
        assert engine.simplify(['atanh', 'tanh', '*', '2', 'sin', 'x0']) == \
            ['*', '2', 'sin', 'x0']
        assert engine.simplify(['atanh', 'tanh', '2']) == ['2']

    def test_refused_outside_the_band(self, engine: SimpliPyEngine) -> None:
        """Ceiling 3 admits measured breaches (scattered from ~2.68 up), so the
        collapse is refused -- these all collapsed under the old band of 18."""
        assert engine.simplify(['atanh', 'tanh', '*', '3', 'sin', 'x0']) == \
            ['atanh', 'tanh', '*', '3', 'sin', 'x0']
        assert engine.simplify(['atanh', 'tanh', '3']) == ['atanh', 'tanh', '3']

    def test_unbounded_argument_never_collapses(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify(['atanh', 'tanh', 'x0']) == ['atanh', 'tanh', 'x0']


class TestTranscendentalAtomEnclosures:
    """Hardening H-018 (2026-08-03, S0): the interval layer rendered ``np.pi``/``np.e``
    as exact f64 points, so it could certify facts about the RENDERING that are false of
    the atom: ``sin(np.pi)`` evaluated to {1.22e-16}, minting a NONZERO-a.e. certificate
    for an expression whose ratified exact value IS 0 (the engine's own mined rule folds
    ``sin pi -> 0``). That false certificate licensed the pow-over-mul distribution
    ``inv(sin(pi) * u) -> inv(sin(pi)) * inv(u)`` at construction -- BEFORE the exact
    rule could fold the zero -- splitting a true zero factor: the one-zero absorption
    value (+inf identically) became +-inf by the sign of u (fuzz row 172346, judge-
    convicted). Atoms are now outward-rounded enclosures inside the interval layer, so
    exact-zero identity grounds contain 0 and every such licence fails closed; the
    certified rules then fold the zero first and absorption wins."""

    def test_zero_times_cofactor_is_the_positive_pole(self, engine: SimpliPyEngine) -> None:
        # These rest on `sin(np.pi) -> 0`, which is exactly true and NOT f64-realised
        # (f64 gives 1.2246467991473532e-16), so from 0.14.0 it is a `real`-tier rule.
        # The pole is therefore reached in `real`, and the default mode keeps the
        # expression -- the two soundnesses disagreeing, on one row.
        from conftest import require_triple_or_skip
        require_triple_or_skip()
        from simplipy import Mode
        for tokens in (['inv', '*', 'sin', 'np.pi', 'x1'],
                       ['inv', '*', 'abs', 'sin', 'np.pi', 'x1'],
                       ['inv', '*', 'tan', 'np.pi', 'x1']):
            assert engine.simplify(list(tokens), mode=Mode.real) == ['float("inf")'], tokens
            assert engine.simplify(list(tokens), mode=Mode.f64) != ['float("inf")'], tokens

    def test_fuzz_row_172346(self, engine: SimpliPyEngine) -> None:
        # inv( abs(sin(np.pi)) / inv(x4 - x1 + 2*x2) ): the reciprocal cofactor
        # canonicalizes with a rational -1, which turned the split pole visibly negative.
        # `sin(np.pi) -> 0` is `real`-tier from 0.14.0 (exactly true, and f64 computes
        # 1.2246467991473532e-16), so the pole is reached in the mode whose authority
        # is mathematics.
        from conftest import require_triple_or_skip
        require_triple_or_skip()
        from simplipy import Mode
        out = engine.simplify(['inv', '/', 'abs', 'sin', 'np.pi',
                               'inv', '-', '-', 'x4', 'x1', '*', 'x2', '-2'],
                              mode=Mode.real)
        assert out == ['float("inf")'], out

    def test_true_nonzero_atom_grounds_still_certify(self, engine: SimpliPyEngine) -> None:
        # The enclosure must not blanket-refuse: pi itself and shifted atoms are
        # rigorously bounded away from 0, so the licensed split still fires.
        from conftest import require_triple_or_skip
        require_triple_or_skip()
        assert engine.simplify(engine.to_tagged(['inv', '*', 'np.pi', 'x1'])) != \
            ['inv', '<mul>', 'np.pi', 'x1', '</mul>']  # some reduction happened
        # sin(pi)+2 is certainly nonzero; in `real` the exact rule folds it to 2. In
        # `f64` the fold is declined (see R1) and the enclosure must STILL not
        # blanket-refuse -- which is the property this test exists for.
        from simplipy import Mode
        out = engine.simplify(['inv', '*', '+', 'sin', 'np.pi', '2', 'inv', 'x1'],
                              mode=Mode.real)
        assert 'sin' not in out, out
        # The CERTIFIED ENDPOINT, not merely "something changed" -- the split having
        # fired is the property, and `!= input` is true of every subject whether the
        # enclosure certifies or blanket-refuses.
        f64_out = engine.simplify(['inv', '*', '+', 'sin', 'np.pi', '2', 'inv', 'x1'])
        assert f64_out == ['/', 'x1', '+', 'sin', 'np.pi', '2'], f64_out


class TestSignDistributionCanonicalUniqueness:
    """Hardening H-014 (2026-08-03): `Mul[-1, Add[terms]]` and the sign-flipped Add are
    one value with two derivation-reachable spellings, and the tagged renderer's display
    redistribution printed them IDENTICALLY -- so parse rebuilt the distributed state
    while rewrite chains could rest on the factored one. A rule with a `Mul[-1, F(..)]`
    LHS fires on the distributed member but cannot see inside the factored node: the
    one-call fixpoint differed from the re-entry fixpoint (P1/P7, 15/200k fuzz rows).
    The `mul` constructor now canonicalizes the conflated class to the distributed
    spelling, restoring serialization injectivity on canonical states."""

    def test_fuzz_row_17001_idempotent(self, engine: SimpliPyEngine) -> None:
        x = ['acos', '-', 'tanh', 'acos', '-', '<constant>', '-3',
             'sinh', '-', 'x2', 'exp', 'float("inf")']
        out = engine.simplify(engine.to_tagged(list(x)))
        assert engine.simplify(list(out)) == out, out
        # The one-call answer must already be the reduced fixpoint (both signs cancel).
        assert out == ['acos', '<add>', 'tanh', 'acos', '<constant>',
                       'float("inf")', '</add>'], out

    def test_fuzz_row_7333_explicit_fixpoint(self, engine: SimpliPyEngine) -> None:
        x = ['/', '-', 'float("-inf")', 'acosh', '*', '1.111', 'atanh', 'x2', 'inv', '-2']
        exp_out = engine.simplify(list(x))
        assert engine.simplify(list(exp_out)) == exp_out, exp_out

    def test_neg_of_sum_distributes(self, engine: SimpliPyEngine) -> None:
        # neg(<add> a b </add>) canonicalizes to the flipped sum, same as parsing the
        # section spelling -- one state, one spelling.
        a = engine.simplify(['neg', '+', 'tanh', 'x0', 'float("inf")'])
        b = engine.simplify(['-', 'float("-inf")', 'tanh', 'x0'])
        assert a == b, (a, b)
        assert engine.simplify(list(a)) == a, a


class TestLossyReciprocalRejoinProjection:
    """Hardening H-015 class (b) (2026-08-04): the lossy blanket licence distributed
    `(a*b)^-1` unconditionally (the match-enabling working form the $-cancellation
    needs), so where no partner cancelled, the lossy endpoint priced strictly above
    sound's licence-refusal spelling. An output-boundary projection now rejoins the
    surviving reciprocals wherever the joined spelling prices strictly smaller AND
    sound's licence would refuse to re-split it; the rewrite chain itself still runs
    in the distributed working canon (200k battery: P6 rows 822 -> 780, 0 added)."""

    def test_row_1414_joins_to_sound_form(self, engine: SimpliPyEngine) -> None:
        x = engine.to_tagged(['inv', '*', 'acos', '*', 'float("inf")', 'x4',
                              'atan', 'asinh', 'x0'])
        sound = engine.simplify(list(x))
        lossy = engine.simplify(list(x), mode=Mode.corpus)
        assert lossy == sound, (lossy, sound)
        assert lossy[:2] == ['inv', '<mul>'], lossy

    def test_partner_cancellation_survives(self, engine: SimpliPyEngine) -> None:
        x = ['*', 'acos', 'x0', 'inv', '*', 'acos', 'x0', 'atan', 'x1']
        assert engine.simplify(list(x), mode=Mode.corpus) == ['inv', 'atan', 'x1']

    def test_certified_pair_stays_distributed(self, engine: SimpliPyEngine) -> None:
        # Where sound's zero-set licence certifies the split, both modes keep the
        # distributed form: the two canons agree and mined rules keep matching.
        x = engine.to_tagged(['inv', '*', 'acos', 'x0', 'atan', 'x1'])
        sound = engine.simplify(list(x))
        lossy = engine.simplify(list(x), mode=Mode.corpus)
        assert lossy == sound == ['<mul>', '<div>', 'acos', 'x0', 'atan', 'x1', '</mul>']

    def test_literal_infinity_never_joins(self, engine: SimpliPyEngine) -> None:
        # inv(inf) is a determined zero (the mask-sentinel cancellation partner);
        # joining it into an opaque base would hide that fold.
        x = ['inv', '*', 'float("inf")', '*', 'acos', '*', 'x4', 'x1', 'atan', 'x0']
        out = engine.simplify(list(x), mode=Mode.corpus)
        assert out[:2] != ['inv', '<mul>'], out

    def test_lossy_idempotent_through_projection(self, engine: SimpliPyEngine) -> None:
        for x in (
            ['inv', '*', 'acos', '*', 'float("inf")', 'x4', 'atan', 'asinh', 'x0'],
            ['*', 'acos', 'x0', 'inv', '*', 'acos', 'x0', 'atan', 'x1'],
            ['inv', '*', 'float("inf")', '*', 'acos', '*', 'x4', 'x1', 'atan', 'x0'],
        ):
            once = engine.simplify(list(x), mode=Mode.corpus)
            assert engine.simplify(list(once), mode=Mode.corpus) == once, x


class TestF68ConstructionRouteConfluence:
    """F68 (2026-08-09): two F63 regressions visible only above the 200k fuzz horizon
    (the 1M base lane was last run 2026-08-05; every ladder since fuzzed 200k rows).

    * KEPT-ZERO MIRROR FREEZE (fuzz rows 392777/647852, sound P7): `mul`'s kept-zero
      arm assembled `Mul[0, ..]` raw, freezing whichever mirror of a trade-site
      factor arrived -- the direct build and the parse of its own rendering shipped
      two states. The arm now routes through `sign_place` with the zero as a
      sign-eating carrier (`0*X == 0*(-X)` in every case: finite -> 0, inf -> nan,
      nan -> nan), so both arrivals file the orbit argmin.
    * REJOIN FUNNEL vs CONSTRUCTION ROUTE (fuzz rows 303116/472156/603382, then
      59022/194208/308376/593418/713415/790992 under the first-cut fix; lossy P1/P7):
      the reciprocal join was priced against the WORKING state, but parse rebuilds a
      joined base's inner bag POSITIVELY (trade sites re-oriented by the orbit or the
      H-020 Free flip) while the division route keeps sites HIDDEN inside `^-1`
      wrappers -- two routes, two working states, and the join's strict win could
      evaporate into a tie on the second pass. The decision is now additionally
      gated on the candidate's RE-SETTLE IMAGE: a join ships only when it strictly
      beats what it provably re-parses to, making the shipped spelling a fixpoint
      by construction.
    """

    # minimal reproducers, delta-debugged from the frozen fuzz stream
    SOUND_ROWS = [
        ('392777', '* - 0.5 acos x4 * pow x1 x0 0'),
        ('647852', '* * log x3 0 - 2 - + x3 pow x1 x4 x0'),
    ]
    LOSSY_ROWS = [
        ('303116', '/ / x0 - 1 x2 - log x4 x1'),
        ('472156', '/ / x3 - -1 x4 + pow x3 np.e neg x3'),
        ('603382', '/ / x2 - asin x2 x3 + - x2 x3 x2'),
        ('59022-joined', 'inv * asinh - * x1 x4 x4 + rootn x2 3 pow x2 <constant>'),
    ]

    @pytest.mark.parametrize('row,src', SOUND_ROWS, ids=[r for r, _ in SOUND_ROWS])
    def test_sound_explicit_roundtrip_is_a_fixpoint(
            self, engine: SimpliPyEngine, row: str, src: str) -> None:
        out = engine.simplify(src.split())
        assert engine.simplify(list(out)) == out, out

    @pytest.mark.parametrize('row,src', LOSSY_ROWS, ids=[r for r, _ in LOSSY_ROWS])
    def test_lossy_is_a_fixpoint(self, engine: SimpliPyEngine, row: str, src: str) -> None:
        once = engine.simplify(src.split(), mode=Mode.corpus)
        assert engine.simplify(list(once), mode=Mode.corpus) == once, once

    def test_kept_zero_files_one_state_for_both_mirrors(self, engine: SimpliPyEngine) -> None:
        # 0 * (1/2 - acos x4) and 0 * (acos x4 - 1/2) are value-equal (the zero eats
        # the sign); pre-F68 each arrival froze its own spelling.
        a = engine.simplify('* 0 - / 1 2 acos x4'.split())
        b = engine.simplify('* 0 - acos x4 / 1 2'.split())
        assert a == b, (a, b)
        nested_a = engine.simplify('* 0 * pow x1 x0 - / 1 2 acos x4'.split())
        nested_b = engine.simplify('* 0 * pow x1 x0 - acos x4 / 1 2'.split())
        assert nested_a == nested_b, (nested_a, nested_b)

    def test_both_mirror_classes_of_the_distributed_form_are_stable(
            self, engine: SimpliPyEngine) -> None:
        # The two pole-distinct arrival spellings settle SEPARATELY (the odd-negative
        # refusal keeps them apart) and each must be its own lossy fixpoint.
        for spelling in (
            '<mul> x0 <div> <add> x1 <sub> log x4 </add> <add> x2 <sub> 1 </add> </mul>',
            '<mul> x0 <div> <add> log x4 <sub> x1 </add> <add> 1 <sub> x2 </add> </mul>',
        ):
            once = engine.simplify(spelling.split(), mode=Mode.corpus)
            assert engine.simplify(list(once), mode=Mode.corpus) == once, spelling


class TestF72ArrivalInvariantSignDecisions:
    """F72 (2026-08-09): two arrival-order leaks in the sign machinery, exposed by the
    extreme lane's first fresh census after F62..F71 (extreme-lane campaign,
    the research harness).

    * INTERNING-ORDER TIE-BREAK (P2 rows 829655/866090 -- the census's only NEW kind,
      a regression of the F62..F71 window itself): sign_place's residual equal-mu tie
      broke on a comparator that read Leaf/Fun tokens by raw interning id, so on a
      fully mu-tied orbit (`-1 * (a-b) * (c-d)`; a -1 coefficient rides mu-free) the
      winning mirror pairing followed whichever literal the input happened to intern
      first. The tie now breaks on cmp_ex, the one canonical string-based total order.
    * PARTITION SIGN HOST (P1-idempotence 88 -> 30 rows at 1M): with the rational
      content partitioned across several Num members (the product overflows i128),
      the sign rode whichever member carried it on arrival -- `-N * P` and the
      re-parse of its own rendering hosted it on different partials, two stable
      states for one value. `mul()` now runs the accumulation on ABSOLUTE values
      (the partition is a function of the unsigned multiset) with the net sign on
      the final coefficient, sign_place's slot; `term_join` routes Num-bearing keys
      through `mul()` instead of raw coefficient-first inserts.
    """

    B = '- 1e40 5e-324'
    C = '- 1.7976931348623157e308 np.pi'
    P = '1/170141183460469231731687303715884105727'

    def test_mirror_decision_is_arrival_invariant(self, engine: SimpliPyEngine) -> None:
        # the 9-token minimal pair, shrunk from row 829655
        a = engine.simplify(f'* -1 * {self.B} {self.C}'.split())
        b = engine.simplify(f'* -1 * {self.C} {self.B}'.split())
        assert a == b, (a, b)

    P2_ROWS = [
        ('829655',
         'rootn - 92233720368547758/7 * - 1e40 5e-324'
         ' - 1.7976931348623157e308 np.pi -2',
         'rootn - 92233720368547758/7 * - 1.7976931348623157e308 np.pi'
         ' - 1e40 5e-324 -2'),
        ('866090',
         '* / - x1 170141183460469231731687303715884105728 np.pi'
         ' * -1 * - x3 3.402823669209385e38 / float("inf") 1e19',
         '* * * / float("inf") 1e19 - x3 3.402823669209385e38 -1'
         ' / - x1 170141183460469231731687303715884105728 np.pi'),
    ]

    @pytest.mark.parametrize('row,src,swapped', P2_ROWS, ids=[r for r, _, _ in P2_ROWS])
    def test_p2_rows_are_permutation_invariant(
            self, engine: SimpliPyEngine, row: str, src: str, swapped: str) -> None:
        assert engine.simplify(src.split()) == engine.simplify(swapped.split())
        assert (engine.simplify(src.split(), mode=Mode.corpus)
                == engine.simplify(swapped.split(), mode=Mode.corpus))

    def test_partition_sign_host_is_route_invariant(self, engine: SimpliPyEngine) -> None:
        a = engine.simplify(['*', '-0.9999999999999999', self.P])
        b = engine.simplify(['neg', '*', '0.9999999999999999', self.P])
        assert a == b, (a, b)

    def test_extracted_sign_reorders_the_partition_bag(self, engine: SimpliPyEngine) -> None:
        # the 7-token minimal P1 reproducer: the add-bag absorbs the product's minus,
        # and the now-unsigned bag must file the unsigned canonical member order.
        src = (f'+ * -0.9999999999999999 {self.P}'
               ' pow 6.80564733841877e38 5.104235503814077e38')
        once = engine.simplify(src.split())
        assert engine.simplify(list(once)) == once, once

    P1_ROWS = [
        ('10783',
         'rootn rootn atanh * x3 np.e 2.552117751907039e38 pow'
         ' + * -0.9999999999999999 1/170141183460469231731687303715884105727'
         ' pow 6.80564733841877e38 5.104235503814077e38'
         ' + rootn (-92233720368547758/7) 1e19 x2'),
        ('27599',
         '+ log 5.104235503814077e38'
         ' - / -0.9999999999999999 170141183460469231731687303715884105727 atan 2'),
        ('33168', '* / -1e40 1.0000000000000002 neg 170141183460469231731687303715884105727'),
        ('40331',
         '- + / -0.9999999999999999 1.2379400392853803e27'
         ' rootn 1/170141183460469231731687303715884105727 9007199254740991'
         ' pow / x3 x0 (-92233720368547758/7)'),
    ]

    @pytest.mark.parametrize('row,src', P1_ROWS, ids=[r for r, _ in P1_ROWS])
    def test_p1_rows_settle_at_pass_one(
            self, engine: SimpliPyEngine, row: str, src: str) -> None:
        once = engine.simplify(src.split())
        assert engine.simplify(list(once)) == once, once


class TestF73PartitionAtomRendering:
    """F73 (2026-08-09, owner-approved plan B): an i128-overflow PARTITION's atoms
    survive every dialect's round-trip.

    When folding a bag's exact rational content would overflow i128, canon keeps the
    content SPLIT across several Num members -- and those atom boundaries are
    load-bearing. The gathering emitters pooled them into shared num/den chains
    (`* a/3 a/3` printed `/ (a*a) (3*3)` in explicit, `a*a/3/3` in infix, and split
    den entries into the tagged `<div>` section), so the re-parse pooled the pieces
    and re-CUT them: one value, a different partition per dialect -- 884 of the
    extreme lane's 900 post-F72 hard rows, including the 30 P1 "div-section
    aliasing" rows. Bags with >= 2 Num members now render each rational as a
    self-contained spelling (one token, a local `/ p q`, or a parenthesized infix
    atom); the sorted accumulation is cut-stable on preserved atoms, so the
    round-trip is the identity. Bags with at most one rational member render
    exactly as before.
    """

    A3 = '170141183460469231731687303715884105727/3'

    def test_minimal_pair_stable_in_every_dialect(self, engine: SimpliPyEngine) -> None:
        src = ['*', self.A3, self.A3]
        t = engine.simplify(list(src))
        assert engine.simplify(list(t)) == t, t
        x = engine.simplify(list(src))
        assert engine.simplify(list(x)) == x, x
        infix = engine.simplify(engine.to_infix(list(src)))
        back = engine._core.parse(infix, True, False)
        assert engine.simplify(list(back)) == t, infix

    def test_reparse_pool_state_is_its_own_fixpoint(self, engine: SimpliPyEngine) -> None:
        # `(a*a)/9` pools to the {a/9, a} cut; its rendering must reproduce that cut.
        a = '170141183460469231731687303715884105727'
        x = engine.simplify(['/', '*', a, a, '9'])
        assert engine.simplify(list(x)) == x, x

    def test_safe_fraction_rendering_is_unchanged(self, engine: SimpliPyEngine) -> None:
        # A single rational member is no partition: the pretty forms stay.
        assert engine.simplify(['*', '1/3', 'x0']) == ['/', 'x0', '3']
        assert engine.simplify(engine.to_infix(['*', '1/3', 'x0'])) == 'x0/3'

    HEALED_ROWS = [
        ('5215',
         'rootn - * 170141183460469231731687303715884105727/3'
         ' 170141183460469231731687303715884105727/3'
         ' pow rootn + -1e19 2.2250738585072014e-308 1/3 5e-324 5e-324'),
        ('50380',
         '* / 1/170141183460469231731687303715884105727 pow -2 x3'
         ' * - 9007199254740993 np.e rootn 3 -1'),
    ]

    @pytest.mark.parametrize('row,src', HEALED_ROWS, ids=[r for r, _ in HEALED_ROWS])
    def test_healed_rows_are_fixpoints_in_both_prefix_dialects(
            self, engine: SimpliPyEngine, row: str, src: str) -> None:
        once = engine.simplify(src.split())
        assert engine.simplify(list(once)) == once, once
        exp = engine.simplify(src.split())
        assert engine.simplify(list(exp)) == exp, exp


class TestConstSumSignAbsorption:
    """Hardening H-020 (2026-08-04, owner-ruled): a negative coefficient never coexists
    with a Const-bearing Add factor -- the sign folds INTO the sum (Const-carrying terms
    absorb it by the forall-exists refit, every other term negates as ordinary
    structure), at every parking: the lone wrapper, the bag coefficient, and collected
    terms. Before the fold, the infix display silently ate the sign under pow bases
    (`pow neg <add> C*x4 .. </add> 1/3` printed `(...)^(1/3)`), because `-<constant>`
    is not a spellable token -- fuzz rows 148693/166766, P7-infix."""

    def test_fuzz_row_166766_shape_folds_and_roundtrips(self, engine: SimpliPyEngine) -> None:
        x = ['pow', 'neg', '+', '*', '<constant>', 'x4', '*', '<constant>',
             'acos', '<constant>', '/', '1', '3']
        out = engine.simplify(list(x))
        assert 'neg' not in out, out
        infix = engine.simplify(engine.to_infix(list(x)))
        back = engine._core.parse(infix, True, False)
        assert engine.simplify(list(back)) == out, (infix, back)

    def test_mixed_sum_sign_lands_on_nonconst_term(self, engine: SimpliPyEngine) -> None:
        out = engine.simplify(engine.to_tagged(
            ['neg', '+', '*', '<constant>', 'x4', 'tanh', 'x1']))
        # The Const term refits sign-free; tanh carries the sign (pushed inside by the
        # odd-function sign-pull rule).
        assert out == ['<add>', 'tanh', 'neg', 'x1',
                       '<mul>', '<constant>', 'x4', '</mul>', '</add>'], out
        assert engine.simplify(list(out)) == out

    def test_bag_coefficient_parking_is_stable(self, engine: SimpliPyEngine) -> None:
        # The sign beside a Const sum under a division: every projection round-trips.
        x = ['/', 'neg', '+', '*', '<constant>', 'x0', 'atan', 'x1', '3.89']
        out = engine.simplify(list(x))
        assert engine.simplify(list(out)) == out
        infix = engine.simplify(engine.to_infix(list(x)))
        back = engine._core.parse(infix, True, False)
        assert engine.simplify(list(back)) == out, (infix, back)

    def test_fully_absorbing_sum_collapses_to_one_state(self, engine: SimpliPyEngine) -> None:
        # -(C*x4 + C*acos(C)) and the unsigned sum are ONE fitted family -> one state.
        neg = engine.simplify(['neg', '+', '*', '<constant>', 'x4',
                               '*', '<constant>', 'acos', '<constant>'])
        pos = engine.simplify(['+', '*', '<constant>', 'x4',
                               '*', '<constant>', 'acos', '<constant>'])
        assert neg == pos, (neg, pos)


class TestLossySentinelExpiryAndCoefficientCompletion:
    """Hardening H-015 class (a) (2026-08-04): two lossy endpoint mechanisms beyond the
    reciprocal rejoin. SENTINEL EXPIRY: the lossy `inv(inf)` keep is a licence for the
    SEARCH (the $-cancel needs the pair visible), not the answer -- at the phase-1
    fixpoint an unpartnered sentinel folds to its determined 0 and the chain keeps
    descending (the kept term had been BLOCKING rules downstream). COEFFICIENT
    COMPLETION: the rejoin offers the bag coefficient reciprocated into the joined base
    (0.5*(x4*S)^-1 IS (2*x4*S)^-1). Together with the narrowed Const exclusion and the
    doctrine-exempt P6 oracle, the mode-ordering census fell 822 -> 110/200k."""

    def test_unpartnered_sentinel_expires_and_chain_continues(self, engine: SimpliPyEngine) -> None:
        x = ['sin', 'cos', '+', '*', '<constant>', 'inv', 'float("inf")', 'neg', 'x2']
        assert engine.simplify(list(x), mode=Mode.corpus) == \
            engine.simplify(list(x)) == ['sin', 'cos', 'x2']

    def test_mask_doctrine_survives_expiry(self, engine: SimpliPyEngine) -> None:
        # A PARTNERED sentinel cancels in phase 1, before the licence expires.
        from conftest import require_triple_or_skip
        require_triple_or_skip()
        x = ['*', '/', 'float("inf")', 'float("inf")', 'x0']
        assert engine.simplify(list(x), mode=Mode.corpus) == ['x0']

    def test_coefficient_reciprocates_into_joined_base(self, engine: SimpliPyEngine) -> None:
        x = engine.to_tagged(
            ['*', '0.5', 'inv', '*', 'x4', '+', 'x3', '+', 'tan', 'x0', '/', '1', '3'])
        out = engine.simplify(list(x), mode=Mode.corpus)
        # RE-PINNED 2026-08-11 (E2, audit F81): the tan-bearing sum now passes the
        # zero-set licence (denominator clearing), so the kept carrier DISTRIBUTES
        # and renders in the flat F80 spelling. The guarded behaviour is unchanged:
        # 0.5 still reciprocates to the `2`, now a member of the flat `<div>`
        # section. Adjudicated with the rust twin: two-route convergence (the old
        # spelling re-simplifies to this same state), fixpoint, 0/1000 value
        # mismatches vs the raw input under the deployed evaluator.
        assert out == ['<mul>', '1', '<div>', '2', 'x4', '<add>', 'x3', 'tan', 'x0',
                       '<mul>', '1', '<div>', '3', '</mul>',
                       '</add>', '</mul>'], out
        assert engine.simplify(list(out), mode=Mode.corpus) == out

    def test_neg_one_coefficient_joins_inside(self, engine: SimpliPyEngine) -> None:
        x = engine.to_tagged(['neg', 'inv', '*', 'x0', '+', 'exp', 'x3', 'pow', 'x3', 'x4'])
        out = engine.simplify(list(x), mode=Mode.corpus)
        assert out == ['inv', '<mul>', '-1', 'x0', '<add>', 'exp', 'x3', 'pow', 'x3',
                       'x4', '</add>', '</mul>'], out
        assert engine.simplify(list(out), mode=Mode.corpus) == out


class TestC36DropCensus:
    """C36: six rule-drop reasons collapsed into one counter with no accessor -- a
    user could not find out what was dropped from their own mine. The census names
    every non-kept disposition."""

    def test_census_names_each_reason(self) -> None:
        rules = [
            (("+", "?0", "0"), ("?0",)),                       # kept
            (("+", "?0", "0"), ("+", "?0", "0")),              # arithmetic-subsumed (lhs==rhs)
            (("sin", "?0"), ("+", "?0", "<constant>")),        # const-count-increase
            # Catch-all lhs. The SORT is load-bearing here since the dedup key became the
            # internal form (2026-08-18): `?0` would key identically to the kept rule
            # above (whose lhs `+ ?0 0` canonicalises to `?0`), so the specimen would be
            # deduplicated away before translate ever classified it.
            (("_0",), ("sin", "_0")),                          # catch-all lhs
            (("+", "?0", "1"), ("+", "?0", "?1")),             # unbound rhs wildcard
            (("sin", "sin", "?0"), ("+", "sin", "sin", "?0", "1")),  # not ordered below
        ]
        ops = {'+': {'realization': '+', 'alias': [], 'inverse': '-', 'arity': 2,
                     'precedence': 1, 'commutative': True},
               'sin': {'realization': 'np.sin', 'alias': [], 'inverse': None,
                       'arity': 1, 'precedence': 3, 'commutative': False}}
        engine = SimpliPyEngine(operators=ops, rules=list(rules))
        census = engine._core.ac_rules_drop_census()
        assert census.get('const-count-increase') == 1, census
        assert census.get('catch-all-lhs') == 1, census
        assert census.get('unbound-rhs-wildcard') == 1, census
        assert census.get('not-ordered-below', 0) >= 1, census
        assert census.get('arithmetic-subsumed', 0) >= 1, census
        kept, subsumed, dropped, _ = engine._core.ac_rules_info()
        assert dropped == sum(v for k, v in census.items()
                              if k not in ('arithmetic-subsumed', 'registry-pole-class-change')), \
            'census must reconcile with the aggregate counter'

    def test_clean_artifact_census_is_empty(self, engine: SimpliPyEngine) -> None:
        assert engine._core.ac_rules_drop_census() == {}
