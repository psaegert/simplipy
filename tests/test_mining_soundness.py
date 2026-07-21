"""Regression tests for the rule-mine's numerical equivalence certification.

Each test encodes one known failure mode of the equivalence checker; together they
gate the checker against regressions of:

1. vacuous equal_nan acceptance (an (almost-)everywhere-NaN pair "agreeing" on NaN rows),
2. saturation false-accepts at loose tolerances (tanh/exp towers within 1e-5 of a constant),
3. corner blindness (wrong-VALUE identities like asin(cosh(_0)) -> nan are false AT 0; the
   exact corner points in the mixture X refute them -- while domain EXTENSION at points where
   the SOURCE is undefined stays allowed: div(_0,_0) -> 1, the generic-equivalence policy),
4. Phase-1 non-exhaustiveness (enumeration stopping at max length REACHED, not SATURATED),
5. non-reproducibility (unseeded X / hash-order iteration),
6. end-to-end: a small mine on operators that CAN express the vacuous pair must not ship it.
"""
import json
import os

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.utils import compositions, count_expressions, enumerate_expressions, sample_expression

# Arithmetic + the transcendental operators needed to express the known defect cases
# (asin(cosh(_0)) is NaN except at 0; tanh(exp(exp(_0))) saturates to 1).
_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "inv": {"realization": "simplipy.operators.inv", "alias": ["inverse"], "inverse": "inv", "arity": 1, "precedence": 4, "commutative": False},
    "/": {"realization": "simplipy.operators.div", "alias": [], "inverse": "*", "arity": 2, "precedence": 2, "commutative": False},
    "pow2": {"realization": "simplipy.operators.pow2", "alias": [], "inverse": "sqrt", "arity": 1, "precedence": 3, "commutative": False},
    "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
    "tanh": {"realization": "np.tanh", "alias": [], "inverse": "atanh", "arity": 1, "precedence": 3, "commutative": False},
    "asin": {"realization": "np.arcsin", "alias": [], "inverse": "sin", "arity": 1, "precedence": 3, "commutative": False},
    "cosh": {"realization": "np.cosh", "alias": [], "inverse": "acosh", "arity": 1, "precedence": 3, "commutative": False},
}


@pytest.fixture()
def engine(tmp_path) -> SimpliPyEngine:
    (tmp_path / "rules.json").write_text(json.dumps([]))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"operators": _OPERATORS, "rules": "rules.json"}))
    eng = SimpliPyEngine.from_config(str(config))
    assert eng._core is not None, "compiled core failed to attach"
    return eng


@pytest.fixture()
def mining_x(engine):
    X = engine._mining_sample_x(1024, 1, np.random.default_rng(0))
    return X.flatten(order='C').tolist(), X.shape[0]


class TestCheckerSoundness:
    """Direct probes of the core equivalence checker on the known failure modes."""

    def test_vacuous_nan_pair_rejected(self, engine, mining_x) -> None:
        """asin(cosh(x0)) vs log(neg(pow2(x0))): both NaN almost everywhere, DIFFERENT
        at 0 (pi/2 vs -inf). A checker comparing with equal_nan alone accepts this
        vacuously; the informativeness gate must reject it (the dev_7-3 ruleset
        contained ~5,125 such rules)."""
        x_flat, n = mining_x
        assert engine._core.equivalent_no_const(
            ['asin', 'cosh', 'x0'], ['log', 'neg', 'pow2', 'x0'], ['x0'], x_flat, n) is False

    def test_real_identity_accepted(self, engine, mining_x) -> None:
        x_flat, n = mining_x
        assert engine._core.equivalent_no_const(['+', 'x0', '0'], ['x0'], ['x0'], x_flat, n) is True
        assert engine._core.equivalent_no_const(['neg', 'neg', 'x0'], ['x0'], ['x0'], x_flat, n) is True

    def test_saturation_tower_rejected(self, engine, mining_x) -> None:
        """tanh(exp(exp(x0))) == 1 held within the OLD rtol=1e-5 on Gaussian-bulk
        samples; strict tolerances + heavy-tailed X must reject it."""
        x_flat, n = mining_x
        assert engine._core.equivalent_no_const(
            ['tanh', 'exp', 'exp', 'x0'], ['1'], ['x0'], x_flat, n) is False

    def test_domain_extension_accepted(self, engine, mining_x) -> None:
        """GENERIC-EQUIVALENCE POLICY: where the SOURCE is undefined
        (0/0 at exactly 0), the replacement may complete it with the limit value --
        x/x -> 1 and x*inv(x) -> 1 certify despite the corner rows in the mixture X."""
        x_flat, n = mining_x
        assert engine._core.equivalent_no_const(
            ['/', 'x0', 'x0'], ['1'], ['x0'], x_flat, n) is True
        assert engine._core.equivalent_no_const(
            ['*', 'x0', 'inv', 'x0'], ['1'], ['x0'], x_flat, n) is True

    def test_overflow_extension_accepted(self, engine, mining_x) -> None:
        """log(exp(x0)) == x0 must certify although exp overflows to +inf on the
        mixture's heavy tail (source-nonfinite rows are extendable f64 artifacts)."""
        x_flat, n = mining_x
        assert engine._core.equivalent_no_const(
            ['log', 'exp', 'x0'], ['x0'], ['x0'], x_flat, n) is True

    def test_extension_is_asymmetric(self, engine, mining_x) -> None:
        """The reverse direction stays rejected: a replacement that is NaN where the
        source is FINITE loses a defined value."""
        x_flat, n = mining_x
        # source finite everywhere; candidate NaN almost everywhere
        assert engine._core.equivalent_no_const(
            ['+', 'x0', '0'], ['log', 'neg', 'pow2', 'x0'], ['x0'], x_flat, n) is False

    def test_no_evidence_rejected(self, engine, mining_x) -> None:
        """An (almost-)nowhere-defined source cannot be rewritten by its corner values
        alone: asin(cosh(x0)) is finite ONLY at exactly 0 (a handful of mixture rows),
        far below the evidence gate, so even a corner-consistent replacement fails."""
        x_flat, n = mining_x
        # 'pi/2-ish constant' replacement matches the source AT its only defined point,
        # but evidence (source-finite rows) is ~7 of 1024 < n/8.
        assert engine._core.find_rule(
            ['asin', 'cosh', 'x0'], 3, None, [['<constant>']], ['x0'], x_flat, n) is None

    def test_generically_constant_source_rejects(self, engine, mining_x) -> None:
        """Two checker policies combine to reject certifying asin(cosh(C*x0)) ->
        <constant>, even though the C=0 instance evaluates to pi/2 on every row.
        First, the constant-challenge sweep is measure-consistent: the measure-zero
        C=0 null slice is excluded, so a rule certifiable ONLY via such a slice has
        no generic evidence. Second, the domain-preservation gate rejects rewrites
        that invent a LIVE function off a dead source's domain -- for every C != 0
        the source is defined only at x0=0 while pi/2 is defined everywhere.
        Undefined expressions must not be rewritten to invented values (compare
        pow(C, nan), which cannot be simplified), so the source must NOT certify
        to <constant>."""
        x_flat, n = mining_x
        assert engine._core.find_rule(
            ['asin', 'cosh', '*', '<constant>', 'x0'], 5, None, [['<constant>']],
            ['x0'], x_flat, n) is None

    def test_all_undefined_instance_rejects(self, engine, mining_x) -> None:
        """A const-bearing source with an instance that is defined NOWHERE (here
        asin(cosh(x0) + C^2) for C != 0) is rejected conservatively: the fit has zero
        valid rows for that instance and bails, whatever the other instances say."""
        x_flat, n = mining_x
        assert engine._core.find_rule(
            ['asin', '+', 'cosh', 'x0', 'pow2', '<constant>'], 6, None,
            [['<constant>']], ['x0'], x_flat, n) is None

    def test_evidence_counts_unique_rows_not_repetitions(self, engine, mining_x) -> None:
        """DISCRIMINATING test that evidence counts unique rows, not (row, instance)
        multiplicity. Source asin(cosh(x0 * (C^2 + 1))): the multiplier C^2+1 is
        never zero, so EVERY challenge instance is finite only at x0=0 (~9 corner
        rows). Unique source-finite rows across all ~48 instances = ~9 < 128, so this
        must REJECT -- but a reverted multiplicity-sum gate would see ~48*9 = ~432 and
        wrongly ACCEPT. (In contrast, a family with an instance finite on ALL rows
        would have genuine full evidence.)"""
        x_flat, n = mining_x
        assert engine._core.find_rule(
            ['asin', 'cosh', '*', 'x0', '+', 'pow2', '<constant>', '1'], 8, None,
            [['<constant>']], ['x0'], x_flat, n) is None

    def test_affine_with_intercept_family_certifies(self, engine, mining_x) -> None:
        """REGRESSION for the affine-fit conditioning fix. The whole
        C0*f(x)+C1 family silently REJECTED before: a GLOBAL trace-scaled Tikhonov
        ridge biased the intercept, and the normal-equations solve squared the
        condition number on the wide-magnitude X so the intercept came out ~5e-9 off,
        past rtol. Fixed by a Householder-QR least-squares solve (no A^T A, no ridge)
        + capping the tail at 1e3. A representative slate of affine rules whose true
        constants are moderate must now certify via the affine closed-form path; a
        non-affine target must still reject (soundness)."""
        x_flat, n = mining_x
        cand = ['+', '*', '<constant>', 'x0', '<constant>']  # C0*x0 + C1
        x0 = np.array(x_flat)
        for a, b in [(2.5, -1.3), (7.0, 0.0), (-3.7, 42.0), (0.0, 5.0), (1.0, -1.0)]:
            y = a * x0 + b
            assert engine._core.exist_constants_fit_linear(
                cand, ['x0'], x_flat, n, y.tolist(), 1e-9, 1e-12) is True, (a, b)
        # soundness: a genuinely non-affine target must NOT certify as C0*x0 + C1
        for y in (np.sin(x0), x0 ** 2, x0 ** 3):
            assert engine._core.exist_constants_fit_linear(
                cand, ['x0'], x_flat, n, y.tolist(), 1e-9, 1e-12) is not True

    def test_affine_growing_basis_family_certifies(self, engine, mining_x) -> None:
        """REGRESSION for the growing-basis affine recall gap.
        With a fast-growing basis f (exp/cosh/pow3+), rows where |y| ~ 1e21
        dominate an UNWEIGHTED least-squares solve in absolute terms, and y's own f64
        rounding there (eps*|y|) exceeds an O(1) intercept -- so C0*f(x)+C1 and f(x)+C1
        rejected on exactly-true constants (0/4) while interceptless C0*f(x) passed.
        The row-weighted solve (weights ~ 1/(atol + rtol*|y_r|), mirroring the relative
        accept gate) must certify the family; non-members must still reject."""
        x_flat, n = mining_x
        x0 = np.array(x_flat)
        rng = np.random.default_rng(5)
        for fname, f in (("exp", np.exp), ("cosh", np.cosh), ("pow2", np.square)):
            for _ in range(4):
                c0 = round(float(rng.normal(0, 5)), 3) or 1.7
                c1 = round(float(rng.normal(0, 5)), 3) or -2.3
                with np.errstate(all="ignore"):
                    y_full = c0 * f(x0) + c1
                    y_shift = f(x0) + c1
                assert engine._core.exist_constants_fit_linear(
                    ["+", "*", "<constant>", fname, "x0", "<constant>"],
                    ["x0"], x_flat, n, y_full.tolist(), 1e-9, 1e-12) is True, (fname, c0, c1)
                assert engine._core.exist_constants_fit_linear(
                    ["+", fname, "x0", "<constant>"],
                    ["x0"], x_flat, n, y_shift.tolist(), 1e-9, 1e-12) is True, (fname, c1)
        # soundness: a non-member (additive non-constant part) must still reject
        with np.errstate(all="ignore"):
            y_neg = np.exp(x0) + np.sin(x0)
        assert engine._core.exist_constants_fit_linear(
            ["+", "*", "<constant>", "exp", "x0", "<constant>"],
            ["x0"], x_flat, n, y_neg.tolist(), 1e-9, 1e-12) is not True

    def test_log_linear_pow_rewrite_certifies(self) -> None:
        """REGRESSION for the log-linear recall path + its LM-fallthrough
        fix: exp(x0+x0) == (e^2)^x0 is a valid rewrite to pow(<constant>, x0), and the
        const-bearing fit (closed-form log-space, or the LM restart seeded by it when
        the closed-form is imprecise on a heavy tail) must certify it. Uses the dev
        engine (the minimal test operator set has no `pow`). The code fix that only a
        closed-form ACCEPT short-circuits -- a Some(false) seeds the LM instead of
        rejecting -- is documented at rust/fit.rs (exist_constants_fit_prepared)."""
        dev = SimpliPyEngine.load('dev_7-3', install=True)
        x = np.linspace(-3.0, 3.0, 256).reshape(-1, 1)
        xf = x.flatten(order='C').tolist()
        assert dev._core.find_rule(
            ['exp', '+', 'x0', 'x0'], 4, 3, [['pow', '<constant>', 'x0']],
            ['x0'], xf, x.shape[0]) == ['pow', '<constant>', 'x0']

    def test_confirm_primitive_rejects_shipped_defect(self, engine, mining_x) -> None:
        """The exact dev_7-3 defect asin(cosh(_0)) -> nan, via the stage-2 confirm
        primitive (find_rule with the single paired candidate)."""
        x_flat, n = mining_x
        assert engine._core.find_rule(
            ['asin', 'cosh', 'x0'], 3, None, [['nan']], ['x0'], x_flat, n) is None


class TestMiningSampleX:
    """The mine's evaluation matrix: seeded, heavy-tailed, corner-bearing."""

    def test_deterministic(self, engine) -> None:
        a = engine._mining_sample_x(256, 3, np.random.default_rng(5))
        b = engine._mining_sample_x(256, 3, np.random.default_rng(5))
        assert np.array_equal(a, b)

    def test_covers_corners_and_tails(self, engine) -> None:
        X = engine._mining_sample_x(4096, 2, np.random.default_rng(0))
        assert X.shape == (4096, 2)
        assert (X == 0.0).any(), "exact zero corner missing"
        # tail upper magnitude is ~1e3 (capped from 1e6 so the constant-fit stays well
        # conditioned; still well past saturation |x|~40 and f64 overflow |x|~710).
        assert (np.abs(X) > 1e2).any(), "heavy tail missing"
        assert (np.abs(X) < 1e-3)[X != 0.0].any(), "tiny-magnitude tail missing"
        assert np.abs(X).max() < 1e4, "tail too wide -> would wreck the constant-fit conditioning"


class TestPhase1Universe:
    """The saturating DP enumerator + count DP + uniform sampler (regression: an
    older pass-based closure missed 93.6% of length-4 in the dev_7-3 mine)."""

    OPS = {"u": 1, "b": 2}

    def test_enumeration_matches_count_dp(self) -> None:
        counts = count_expressions(2, self.OPS, 6)
        enumerated = enumerate_expressions(["x", "y"], self.OPS, 6)
        for length, expected in counts.items():
            assert len(enumerated[length]) == expected

    def test_triple_unary_chains_present(self) -> None:
        """The exact regression case: unary(unary(unary(leaf))) at length 4 was absent
        because the old loop exited once any length-max expression appeared."""
        enumerated = enumerate_expressions(["x"], self.OPS, 4)
        assert ("u", "u", "u", "x") in enumerated[4]

    def test_dev_universe_sizes(self) -> None:
        """Independently computed dev-config universe sizes (5 leaves,
        38 operators: 33 unary + 5 binary as of dev_7-3)."""
        arities = {f"u{i}": 1 for i in range(33)} | {f"b{i}": 2 for i in range(5)}
        counts = count_expressions(5, arities, 7)
        assert counts == {1: 5, 2: 165, 3: 5570, 4: 192060, 5: 6752605,
                          6: 241629465, 7: 8783426095}

    def test_sampler_uniform_membership_and_determinism(self) -> None:
        counts = count_expressions(1, self.OPS, 5)
        enumerated = enumerate_expressions(["x"], self.OPS, 5)
        rng = np.random.default_rng(0)
        draws = [sample_expression(5, ["x"], self.OPS, counts, rng) for _ in range(300)]
        assert all(d in enumerated[5] for d in draws)
        rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
        assert [sample_expression(4, ["x"], self.OPS, counts, rng_a) for _ in range(30)] == \
               [sample_expression(4, ["x"], self.OPS, counts, rng_b) for _ in range(30)]

    def test_compositions(self) -> None:
        assert list(compositions(3, 2)) == [(1, 2), (2, 1)]
        assert list(compositions(4, 1)) == [(4,)]
        assert sum(1 for _ in compositions(6, 3)) == 10


class TestEndToEndMineGate:
    """A small mine over operators that CAN express the vacuous pair must not ship it."""

    def test_no_vacuous_rule_ships(self, engine) -> None:
        engine.find_rules(
            max_source_pattern_length=3,
            dummy_variables=1,
            extra_internal_terms=["0", "1", "<constant>"],
            X=512,
            verbose=False,
            seed=42,
        )
        assert len(engine.simplification_rules) > 0
        # The real gate: no rule whose LHS is the vacuous asin(cosh(.)) family (which
        # an equal_nan-only checker mined as -> nan). A blanket rhs != ['nan'] check would be
        # inert here (nan is not a candidate token in this leaf set), so we assert on the
        # LHS family that actually reaches the checker.
        for lhs, rhs in engine.simplification_rules:
            assert not ("asin" in lhs and "cosh" in lhs), (lhs, rhs)

    def test_determinism_across_processes(self, tmp_path) -> None:
        """DISCRIMINATING determinism test: two SEPARATE interpreters with DIFFERENT
        PYTHONHASHSEED must mine the identical ruleset. A same-process replica (the
        weaker TestFindRules.test_deterministic_across_runs) shares set-iteration order
        and cannot catch a removed sorted() reproducibility guard; this can."""
        import json
        import subprocess
        import sys

        (tmp_path / "rules.json").write_text(json.dumps([]))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"operators": _OPERATORS, "rules": "rules.json"}))
        prog = (
            "import json, sys;"
            "from simplipy import SimpliPyEngine;"
            f"e = SimpliPyEngine.from_config({str(tmp_path / 'config.yaml')!r});"
            "e.find_rules(max_source_pattern_length=3, dummy_variables=1,"
            " extra_internal_terms=['0','1','<constant>'], X=256, seed=7, verbose=False);"
            "print(json.dumps(sorted([[list(l), list(r)] for l, r in e.simplification_rules])))"
        )
        env_base = {**os.environ, "PYTHONPATH": "src", "OMP_NUM_THREADS": "1",
                    "RAYON_NUM_THREADS": "2"}
        outs = []
        for hashseed in ("0", "12345"):
            res = subprocess.run([sys.executable, "-c", prog],
                                 env={**env_base, "PYTHONHASHSEED": hashseed},
                                 capture_output=True, text=True, cwd=os.getcwd())
            assert res.returncode == 0, res.stderr
            outs.append(res.stdout.strip())
        assert outs[0] == outs[1], "ruleset differs across PYTHONHASHSEED -> non-reproducible"
        assert outs[0] != "[]"


class TestFoldFilter:
    """Variable-free candidate minimization. Var-free candidates
    of length >= 2 are dominated by the length-1 <constant> candidate (a var-free candidate
    is a constant function of X per constant-assignment; the scan is shortest-first), so
    dropping them must not change ANY mined rule -- while removing the bulk of the
    constant-bearing (LM-fit) candidate arm that dominates const-free source cost."""

    def test_counts_and_inert_guard(self, engine, mining_x) -> None:
        x_flat, n = mining_x
        cands = [["x0"], ["<constant>"], ["exp", "<constant>"], ["pow2", "<constant>"],
                 ["exp", "1"], ["neg", "x0"], ["*", "<constant>", "x0"]]
        lib = engine._core.build_candidate_library(cands, ["x0"], x_flat, n)
        assert lib.n_filtered == 3, "exp(<c>), pow2(<c>), exp(1) must be filtered"
        assert lib.n_candidates == 4
        raw = engine._core.build_candidate_library(cands, ["x0"], x_flat, n, fold_filter=False)
        assert raw.n_filtered == 0 and raw.n_candidates == 7
        # without the bare <constant> candidate the filter must be INERT (conservative guard:
        # dominance needs the length-1 <constant> to actually be scanned first)
        lib2 = engine._core.build_candidate_library(cands[2:], ["x0"], x_flat, n)
        assert lib2.n_filtered == 0

    def test_dominance_holds_at_the_band_edge(self, engine, tmp_path) -> None:
        """Adversarial regression at the acceptance-band edge. The dominance lemma
        REQUIRES the length-1 <constant> to accept whenever ANY feasible constant exists.
        The least-squares mean solve violated that on skewed near-constant sources (63 rows
        at e-2.4e-9, one at e+2.4e-9: v = e is feasible on EVERY row's band but the mean
        sits outside the minority row's), making filtered and unfiltered mines DIVERGE
        (raw selected ['exp','1'] at length 2, filtered returned None). With the exact
        interval-intersection decision for the bare <constant>, both must agree."""
        n = 64
        e_const = float(np.e)
        d = 2.4e-9  # inside the band atol + rtol*e ~ 2.72e-9
        # (1) the PRIMITIVE: the exact <constant> decision accepts the feasible skewed case
        y = np.full(n, e_const - d)
        y[-1] = e_const + d
        assert engine._core.exist_constants_fit_linear(
            ["<constant>"], ["x0"], [float(r) for r in range(n)], n, y.tolist(),
            1e-9, 1e-12) is True
        # ... and still rejects an infeasible spread (soundness)
        y_bad = np.full(n, e_const - 1e-8)
        y_bad[-1] = e_const + 1e-8
        assert engine._core.exist_constants_fit_linear(
            ["<constant>"], ["x0"], [float(r) for r in range(n)], n, y_bad.tolist(),
            1e-9, 1e-12) is not True
        # (2) END-TO-END: the exact divergence case. Crafted X (63 rows
        # x0 = -1, one row x0 = +1); source exp(1) + 2.4e-9*x0 evaluates to the skewed y.
        # The filtered and unfiltered scans must AGREE whatever else gates the match.
        x_flat = [-1.0] * (n - 1) + [1.0]
        src = ["+", "exp", "1", "*", "2.4e-9", "x0"]
        cands = [["<constant>"], ["x0"], ["exp", "1"]]
        results = []
        for ff in (True, False):
            lib = engine._core.build_candidate_library(cands, ["x0"], x_flat, n, fold_filter=ff)
            results.append(engine._core.find_rule_lib(
                src, len(src), 2, lib, challenges=16, retries=16, seed=7,
                rtol=1e-9, atol=1e-12))
        assert results[0] == results[1], f"filtered {results[0]} != unfiltered {results[1]}"
        # The band-edge match itself is asserted with the special-point battery OFF (own
        # subprocess: the switch is a process-lifetime OnceLock). The battery's fixed
        # probe points (|x| up to pi) amplify the crafted 2.4e-9-per-x skew past the
        # acceptance band (a 7.2e-9 real change at x = +-3): a value-soundness
        # rejection orthogonal to the dominance property under test.
        import subprocess
        import sys
        prog = (
            "import json, sys;"
            "from simplipy import SimpliPyEngine;"
            "eng = SimpliPyEngine.from_config(sys.argv[1]);"
            f"x_flat = {x_flat!r}; n = {n};"
            f"cands = {cands!r};"
            "outs = [];\n"
            "for ff in (True, False):\n"
            "    lib = eng._core.build_candidate_library(cands, ['x0'], x_flat, n, fold_filter=ff)\n"
            f"    outs.append(eng._core.find_rule_lib({src!r}, {len(src)}, 2, lib,"
            " challenges=16, retries=16, seed=7, rtol=1e-9, atol=1e-12))\n"
            "assert outs[0] == outs[1], outs\n"
            "assert outs[0] is not None, 'the feasible near-constant source must match'\n"
        )
        (tmp_path / "rules.json").write_text(json.dumps([]))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"operators": _OPERATORS, "rules": "rules.json"}))
        res = subprocess.run(
            [sys.executable, "-c", prog, str(cfg)],
            env={**os.environ, "SIMPLIPY_SPECIAL_BATTERY": "0", "OMP_NUM_THREADS": "1"},
            capture_output=True, text=True)
        assert res.returncode == 0, res.stderr

    def test_mine_parity_filtered_vs_unfiltered(self, tmp_path) -> None:
        """THE PARITY GATE: an end-to-end mine with and
        without the fold-filter must produce the IDENTICAL ruleset. Fit seeds are a pure
        function of (source seed, candidate tokens, instance) -- order-independent -- so the
        two runs draw identical randomness for every shared candidate and can differ ONLY
        via the dropped var-free candidates, which dominance says are never selectable."""
        rulesets = []
        for fold_filter in (True, False):
            tag = str(fold_filter)
            (tmp_path / f"rules_{tag}.json").write_text(json.dumps([]))
            cfg = tmp_path / f"config_{tag}.yaml"
            cfg.write_text(yaml.safe_dump({"operators": _OPERATORS, "rules": f"rules_{tag}.json"}))
            eng = SimpliPyEngine.from_config(str(cfg))
            eng.find_rules(max_source_pattern_length=3, dummy_variables=1,
                           extra_internal_terms=["0", "1", "<constant>"], X=256, seed=7,
                           verbose=False, candidate_fold_filter=fold_filter)
            rulesets.append(sorted((tuple(lhs), tuple(rhs)) for lhs, rhs in eng.simplification_rules))
        assert rulesets[0] == rulesets[1], "fold-filter changed the mined ruleset"
        assert len(rulesets[0]) > 0


class TestProvenance:
    """The mined artifact must carry a reproducibility sidecar, and sampled sources are
    validated as universe members per run (exercised via the sampled length below -- a
    violation raises inside find_rules)."""

    def test_sidecar_written_with_reproducibility_fields(self, tmp_path) -> None:
        (tmp_path / "rules.json").write_text(json.dumps([]))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"operators": _OPERATORS, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(str(cfg))
        out = str(tmp_path / "mined.json")
        eng.find_rules(max_source_pattern_length=3, dummy_variables=1,
                       extra_internal_terms=["0", "1", "<constant>"], X=256, seed=7,
                       verbose=False, output_file=out,
                       source_sample_per_length={3: 500})
        side = json.load(open(out + ".provenance.json"))
        assert side["params"]["seed"] == 7
        assert side["params"]["mine_seed"] and side["params"]["confirm_seed"]
        assert side["params"]["candidate_fold_filter"] is True
        assert side["params"]["source_sample_per_length"] == {"3": 500}
        assert side["X"]["source"].startswith("seeded_mixture")
        assert side["universe"]["3"]["sampled"] is True
        assert 0 < side["universe"]["3"]["coverage"] <= 1
        assert side["universe"]["2"]["coverage"] == 1.0
        assert side["progress"]["final"] is True
        assert side["progress"]["rules_total"] == len(json.load(open(out)))
        assert side["simplipy_version"]


class TestStageTwoConfirmation:
    """The stage-2 confirmation (confirm=True) must actually filter, not pass through."""

    def test_confirm_filters_a_data_luck_rule(self, engine, mining_x) -> None:
        """_confirm_mined_rules re-verifies on an INDEPENDENT wider X. A (source, target)
        pair that the checker rejects there must be dropped. Direct probe of the confirm
        primitive on the shipped defect: asin(cosh(x0)) -> nan is rejected by the confirm
        X, so _confirm_mined_rules returns [] for it while keeping a genuine rule."""
        x_confirm = engine._mining_sample_x(2048, 1, np.random.default_rng(99))
        kept = engine._confirm_mined_rules(
            [(('asin', 'cosh', 'x0'), ('nan',)), (('+', 'x0', '0'), ('x0',))],
            ['x0'], x_confirm, 16, 16, 1e-9, 1e-12, 256, 123)
        assert (('asin', 'cosh', 'x0'), ('nan',)) not in kept
        assert (('+', 'x0', '0'), ('x0',)) in kept


class TestCertifyRules:
    """The public certification API for externally proposed rules (LLM/human proposals)."""

    def test_certifies_true_identity_rejects_false_and_verifies_hint(self, engine) -> None:
        proposals = [
            ["log", "*", "exp", "x0", "exp", "x1"],                  # log(e^a e^b) -> a+b (true, L6)
            ["+", "exp", "x0", "cosh", "x1"],                        # no shorter equivalent -> reject
            ["*", "exp", "x0", "*", "exp", "x1", "exp", "x2"],       # minimal form is 6 tokens
        ]
        hints = [None, None, ["exp", "+", "x0", "+", "x1", "x2"]]
        out = engine.certify_rules(proposals, hints, dummy_variables=3, X=256, seed=7)
        by_src = {tuple(s): (t, c) for s, t, c in out}
        assert by_src[tuple(proposals[0])] == (("+", "x0", "x1"), "minimal")
        assert tuple(proposals[1]) not in by_src
        assert by_src[tuple(proposals[2])] == (("exp", "+", "x0", "+", "x1", "x2"), "verified")

    def test_skips_sources_the_engine_already_reduces(self, engine) -> None:
        engine.find_rules(max_source_pattern_length=3, dummy_variables=1,
                          extra_internal_terms=["0", "1", "<constant>"], X=256,
                          seed=7, verbose=False)
        out = engine.certify_rules([["+", "x0", "0"]], X=256, seed=7)
        assert out == []


@pytest.fixture(scope="module")
def dev():
    return SimpliPyEngine.load('dev_7-3', install=True)


@pytest.fixture(scope="module")
def dev_x1(dev):
    X = dev._mining_sample_x(1024, 1, np.random.default_rng(0))
    return X.flatten(order='C').tolist(), X.shape[0]


class TestSpecialPointCertification:
    """The special-point battery + witness snapping (rust/battery.rs): the miner must
    reject at the contract's special points -- fitted-witness snapping before the
    domain gate, the per-variable battery (pi/2, ..., judged without the hiprec rescue),
    the special source-constant sweep, and the transcendental nan-seam probe -- while
    null-set completions and domain-preserving witnesses keep certifying.
    Uses the dev_7-3 engine: the minimal test operator set has no pow/trig."""

    def test_snapped_witness_domain_extension_rejected(self, dev, dev_x1) -> None:
        """exp(log(x^3)) = x^3 only on x > 0. The raw fitted exponent
        (2.9999999999999996) is NaN on x < 0 and blinds the interval domain gate; the
        SNAPPED witness 3.0 is total on R -- a positive-measure domain extension
        (a 37-rule family in (4,3) mining runs before this phase). The domain-PRESERVING
        non-integer witness exp(log(x)/3) = x^(1/3) must keep certifying."""
        x_flat, n = dev_x1
        assert dev._core.find_rule(
            ['exp', 'log', 'pow3', 'x0'], 4, 3, [['pow', 'x0', '<constant>']],
            ['x0'], x_flat, n) is None
        assert dev._core.find_rule(
            ['exp', 'div3', 'log', 'x0'], 4, 3, [['pow', 'x0', '<constant>']],
            ['x0'], x_flat, n) == ['pow', 'x0', '<constant>']

    def test_contract_point_families_rejected(self, dev, dev_x1) -> None:
        """Deployed-value consistency at the battery points:
        pow(sin x, inf) = 1 at x = pi/2 (f64 sin(pi/2) is EXACTLY 1.0);
        pow(cos c, inf) = 1 at the special source constant c = 0;
        atanh(tanh(tan x)) diverges from tan x by ~3e-7 at x = +-1.5 in deployed f64
        (a live (4,3) family). The flagship rescue identity atanh(tanh(x)) -> x
        must keep certifying (the battery box never reaches its saturation zone)."""
        x_flat, n = dev_x1
        assert dev._core.find_rule(
            ['pow', 'sin', 'x0', 'float("inf")'], 4, 1, [['0']], ['x0'], x_flat, n) is None
        assert dev._core.find_rule(
            ['pow', 'cos', '<constant>', 'float("inf")'], 4, 1, [['0']],
            ['x0'], x_flat, n) is None
        assert dev._core.find_rule(
            ['atanh', 'tanh', 'tan', 'x0'], 4, 2, [['tan', 'x0']], ['x0'], x_flat, n) is None
        assert dev._core.find_rule(
            ['atanh', 'tanh', 'x0'], 3, 1, [['x0']], ['x0'], x_flat, n) == ['x0']

    def test_null_set_completion_still_certifies(self, dev, dev_x1) -> None:
        """The limit-completion doctrine survives every new gate: x/x -> 1
        extends only at the exact (dyadic, precision-stable) point 0."""
        x_flat, n = dev_x1
        assert dev._core.find_rule(
            ['/', 'x0', 'x0'], 3, 1, [['1']], ['x0'], x_flat, n) == ['1']

    def test_proposals_path_rejects_seam_pair(self, dev) -> None:
        """The LLM-proposal channel runs through the identical certification: the
        hint pair (x^3 +- x^2 y)/(x +- y) -> x^2 evaluates 0/0 = nan at the exact f64
        battery pair (pi/2, -+pi/2) but residue/0 = inf at the dps-50 precision rung
        (the two spellings of x^3 round APART): a precision-UNSTABLE extension. Both
        variants must be rejected on BOTH proposal arms (library scan and hint
        verification), while the once-spelled cancellation (x^2 - y^2)/(x + y) -> x - y
        (a certified live (2,1) rule) stays certifiable."""
        seam_plus = ['/', '+', 'pow3', 'x0', '*', 'pow2', 'x0', 'x1', '+', 'x0', 'x1']
        seam_minus = ['/', '-', 'pow3', 'x0', '*', 'pow2', 'x0', 'x1', '-', 'x0', 'x1']
        stable = ['/', '-', 'pow2', 'x0', 'pow2', 'x1', '+', 'x0', 'x1']
        # non-vacuity guard: none of these are already reduced by the dev rules
        # (certify_rules skips already-covered sources without judging them)
        for src in (seam_plus, seam_minus, stable):
            assert len(dev.simplify(list(src))) >= len(src)
        out = dev.certify_rules(
            [seam_plus, seam_minus, stable],
            [['pow2', 'x0'], ['pow2', 'x0'], ['-', 'x0', 'x1']],
            dummy_variables=2, X=1024, seed=7)
        by_src = {tuple(s): t for s, t, _ in out}
        assert tuple(seam_plus) not in by_src
        assert tuple(seam_minus) not in by_src
        assert by_src[tuple(stable)] == ('-', 'x0', 'x1')


class TestProposalChannel:
    """find_rules(proposals=...): the LLM/human proposal channel. PLUMBING around the
    certify machinery -- after the length loop, each proposal runs the exact
    certify_rules chain against the just-mined state with the mine's own matrices and
    master-derived seeds, joins through the same deduplicate_rules path, and lands in
    the provenance sidecar with per-outcome counts."""

    @staticmethod
    def _mine(directory, proposals):
        """One small mine (L<=3 sources, L<=3 targets, one dummy) with a proposal batch."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "rules.json"), "w") as fh:
            fh.write("[]")
        cfg = os.path.join(directory, "config.yaml")
        with open(cfg, "w") as fh:
            fh.write(yaml.safe_dump({"operators": _OPERATORS, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(cfg)
        out = os.path.join(directory, "mined.json")
        eng.find_rules(max_source_pattern_length=3, max_target_pattern_length=3,
                       dummy_variables=1, extra_internal_terms=["0", "1", "<constant>"],
                       X=256, seed=7, verbose=False, output_file=out, proposals=proposals)
        with open(out + ".provenance.json") as fh:
            sidecar = json.load(fh)
        return eng, sidecar, out

    def test_certifiable_proposal_joins_ruleset(self, tmp_path) -> None:
        """A certifiable proposal joins the ruleset through deduplicate_rules: the
        minimal path (library target) and the hint path ('verified') both land as
        canonical rules in the engine AND the written artifact; an exact repeat of a
        certified proposal counts as 'duplicate' and adds nothing."""
        proposals = [
            {"source": ["*", "exp", "x0", "exp", "x0"], "why": "extra keys ignored"},
            {"source": ["+", "pow2", "x0", "pow2", "x0"],
             "target": ["*", "<constant>", "pow2", "x0"]},                # hint honored
            {"source": ["*", "exp", "x0", "exp", "x0"]},                  # duplicate of [0]
        ]
        eng, sidecar, out = self._mine(str(tmp_path), proposals)
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in eng.simplification_rules}
        assert (("*", "exp", "?0", "exp", "?0"), ("pow2", "exp", "?0")) in rules
        assert (("+", "pow2", "?0", "pow2", "?0"), ("*", "<constant>", "pow2", "?0")) in rules
        saved = {tuple(tuple(side) for side in rule) for rule in json.load(open(out))}
        assert rules == saved, "the artifact must contain the merged (mined + certified) ruleset"
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 2, "already_covered": 0, "rejected": 0, "duplicate": 1}
        assert sidecar["proposals"]["count"] == 3 and sidecar["proposals"]["sha256"]

    def test_already_covered_proposal_is_skipped(self, tmp_path) -> None:
        """A proposal the mined rules already shorten is skipped exactly like an
        already-reducible source: counted 'already_covered', ruleset identical to a
        proposal-free mine."""
        eng_plain, _, _ = self._mine(str(tmp_path / "plain"), None)
        eng, sidecar, _ = self._mine(str(tmp_path / "covered"), [{"source": ["+", "x0", "0"]}])
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 0, "already_covered": 1, "rejected": 0, "duplicate": 0}
        assert eng.simplification_rules == eng_plain.simplification_rules

    def test_false_proposal_is_rejected(self, tmp_path) -> None:
        """A numerically false proposal is rejected by the same gates as a mined rule:
        the false hint exp(cosh(x)) for e^x + cosh(x) fails verification, and a
        proposal outside the mine's vocabulary is rejected outright."""
        proposals = [
            {"source": ["+", "exp", "x0", "cosh", "x0"], "target": ["exp", "cosh", "x0"]},
            {"source": ["sin", "x0"]},  # 'sin' is not in this operator set
        ]
        eng, sidecar, _ = self._mine(str(tmp_path), proposals)
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 0, "already_covered": 0, "rejected": 2, "duplicate": 0}
        assert all("sin" not in rule[0] for rule in eng.simplification_rules)
        assert not any(tuple(lhs)[:1] == ("+",) and "cosh" in lhs
                       for lhs, _ in eng.simplification_rules)

    def test_proposal_channel_is_deterministic(self, tmp_path) -> None:
        """Two runs from the same master seed and the same proposals FILE (bare-list
        schema) produce byte-identical rulesets and identical provenance counts --
        file order + content-derived per-proposal seeds, nothing position- or
        entropy-derived."""
        proposals_file = tmp_path / "proposals.json"
        proposals_file.write_text(json.dumps([
            {"source": ["*", "exp", "x0", "exp", "x0"]},
            {"source": ["+", "pow2", "x0", "pow2", "x0"],
             "target": ["*", "<constant>", "pow2", "x0"]},
            {"source": ["+", "exp", "x0", "cosh", "x0"], "target": ["exp", "cosh", "x0"]},
            {"source": ["+", "x0", "0"]},
        ]))
        results = []
        for run in ("one", "two"):
            eng, sidecar, out = self._mine(str(tmp_path / run), str(proposals_file))
            results.append((eng.simplification_rules, sidecar["proposals"],
                            open(out).read()))
        assert results[0][0] == results[1][0], "rulesets differ between identical runs"
        assert results[0][1] == results[1][1], "proposal provenance differs between identical runs"
        assert results[0][2] == results[1][2], "written artifacts differ between identical runs"
        assert results[0][1]["file"] == str(proposals_file)
        assert results[0][1]["outcomes"] == {
            "certified": 2, "already_covered": 1, "rejected": 1, "duplicate": 0}
