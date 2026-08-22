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
6. end-to-end: a small mine on operators that CAN express the vacuous pair must not ship it,
7. extension-measure blindness (F0): a source that is NaN almost everywhere (rootn with an
   expression-position index) must not ship a rule rewriting it to a defined value. The
   interval domain gate must treat an UNANALYZABLE region as undecided (fail-closed), not
   as proven-clean -- the 2026-07-29 acj-4-3 incident, where five such rules shipped and
   had to be killed post hoc by verify_ruleset (bc-positive-measure).
"""
import json
import os

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.mining import RuleMiner
from simplipy.utils import (compositions, count_expressions, enumerate_expressions, sample_expression,
                            violates_wildcard_multiplicity)

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
    config.write_text(yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS, "rules": "rules.json"}))
    eng = SimpliPyEngine.from_config(str(config))
    assert eng._core is not None, "compiled core failed to attach"
    return eng


@pytest.fixture()
def mining_x(engine):
    X = RuleMiner(engine)._mining_sample_x(1024, 1, np.random.default_rng(0))
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
        the closed-form is imprecise on a heavy tail) must certify it. Uses the
        acj-4-3 engine (the minimal test operator set has no `pow`). The code fix
        that only a closed-form ACCEPT short-circuits -- a Some(false) seeds the LM
        instead of rejecting -- is documented at rust/fit.rs
        (exist_constants_fit_prepared). The source is const-FREE, so the certified
        slot RESOLVES to its literal (the Const-count invariant): the correctly-
        rounded f64 of e^2, pinned by the hiprec arbiter, never the raw fit
        witness."""
        from conftest import acj_config_path, require_or_skip
        config = acj_config_path()
        require_or_skip(config, 'acj-4-3 config not staged')
        dev = SimpliPyEngine.from_config(config)
        x = np.linspace(-3.0, 3.0, 256).reshape(-1, 1)
        xf = x.flatten(order='C').tolist()
        assert dev._core.find_rule(
            ['exp', '+', 'x0', 'x0'], 4, 3, [['pow', '<constant>', 'x0']],
            ['x0'], xf, x.shape[0]) == ['pow', '7.38905609893065', 'x0']

    def test_confirm_primitive_rejects_shipped_defect(self, engine, mining_x) -> None:
        """The exact dev_7-3 defect asin(cosh(_0)) -> nan, via the stage-2 confirm
        primitive (find_rule with the single paired candidate). TWO layers refuse it
        (D4/H-043): the old artifact's bare `nan` spelling is a reserved numeric
        spelling and dies at the alphabet boundary before any numerics run; the
        canonical `float("nan")` spelling reaches the checker and is killed by the
        finite-evidence gate."""
        x_flat, n = mining_x
        with pytest.raises(ValueError, match="reserved numeric spelling"):
            engine._core.find_rule(
                ['asin', 'cosh', 'x0'], 3, None, [['nan']], ['x0'], x_flat, n)
        assert engine._core.find_rule(
            ['asin', 'cosh', 'x0'], 3, None, [['float("nan")']], ['x0'],
            x_flat, n) is None


class TestMiningSampleX:
    """The mine's evaluation matrix: seeded, heavy-tailed, corner-bearing."""

    def test_deterministic(self, engine) -> None:
        a = RuleMiner(engine)._mining_sample_x(256, 3, np.random.default_rng(5))
        b = RuleMiner(engine)._mining_sample_x(256, 3, np.random.default_rng(5))
        assert np.array_equal(a, b)

    def test_covers_corners_and_tails(self, engine) -> None:
        X = RuleMiner(engine)._mining_sample_x(4096, 2, np.random.default_rng(0))
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
            promote_sorts=False,
            verbose=False,
            seed=42,
        )
        assert len(engine.simplification_rules) > 0
        # The real gate: no rule whose LHS is the vacuous asin(cosh(.)) family (which
        # an equal_nan-only checker mined as -> nan). A blanket rhs != ['nan'] check would be
        # inert here (nan is not a candidate token in this leaf set), so we assert on the
        # LHS family that actually reaches the checker.
        wc = ("?0", "_0", "!0", "$0")
        for lhs, rhs in engine.simplification_rules:
            # STAGE 2 carve-out: GROUND sources (no wildcard) are pointwise states --
            # `asin cosh 1 -> nan` is an exact hiprec-certified nan collapse of a
            # state the mu-governed fold no longer materializes (cosh(1) stays
            # symbolic), not the vacuous equal_nan wildcard family this gate exists
            # for. Wildcard-bearing asin/cosh sources stay forbidden.
            if not any(t in wc for t in lhs):
                continue
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
            yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS, "rules": "rules.json"}))
        prog = (
            "import json, sys;"
            "from simplipy import SimpliPyEngine;"
            f"e = SimpliPyEngine.from_config({str(tmp_path / 'config.yaml')!r});"
            "e.find_rules(max_source_pattern_length=3, dummy_variables=1,"
            " extra_internal_terms=['0','1','<constant>'], X=256, seed=7, verbose=False,"
            " promote_sorts=False);"
            "print(json.dumps(sorted([[list(l), list(r)] for l, r in e.simplification_rules])))"
        )
        # Propagate the parent's import path: simplipy may be an editable install
        # (src/ holds the compiled core) or a pip-installed package (site-packages);
        # hardcoding either breaks the other.
        env_base = {**os.environ,
                    "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
                    "OMP_NUM_THREADS": "1"}
        # missed-007: docs/rules.md promises reproducibility independent of process,
        # hash randomization AND thread count -- this test used to pin
        # RAYON_NUM_THREADS=2 in BOTH arms, leaving the thread-count leg of the
        # published promise untested. The arms now vary it (1 vs 4) alongside the
        # hash seed, so one comparison covers all three legs.
        outs = []
        for hashseed, rayon in (("0", "1"), ("12345", "4")):
            res = subprocess.run([sys.executable, "-c", prog],
                                 env={**env_base, "PYTHONHASHSEED": hashseed,
                                      "RAYON_NUM_THREADS": rayon},
                                 capture_output=True, text=True, cwd=os.getcwd())
            assert res.returncode == 0, res.stderr
            outs.append(res.stdout.strip())
        assert outs[0] == outs[1], \
            "ruleset differs across PYTHONHASHSEED/RAYON_NUM_THREADS -> non-reproducible"
        assert outs[0] != "[]"


# Failure mode 7 (F0): rootn's expression-position index is a non-integer almost everywhere,
# so any `rootn <lit> tanh/cosh ?0` source is NaN a.e. -- the exact vocabulary of the five
# rules the 2026-07-29 acj-4-3 mine shipped. abs/neg are the anti-vacuity anchor: their
# parity family (cosh(neg .) -> cosh(.), ...) mints SOUND rules in the same mine, so the
# "no poison shipped" assertion can never hold by the mine minting nothing at all.
_ROOTN_OPERATORS = {
    "rootn": {"realization": "simplipy.operators.rootn", "alias": [], "inverse": None, "arity": 2, "precedence": 3, "commutative": False},
    "tanh": {"realization": "simplipy.operators.tanh", "alias": [], "inverse": None, "arity": 1, "precedence": 2, "commutative": False},
    "cosh": {"realization": "simplipy.operators.cosh", "alias": [], "inverse": None, "arity": 1, "precedence": 2, "commutative": False},
    "abs": {"realization": "simplipy.operators.abs", "alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
}


def _subtree_end(tokens, i, arity):
    """Index one past the prefix subtree starting at ``tokens[i]``."""
    stack = 1
    while stack:
        stack += arity.get(tokens[i], 0) - 1
        i += 1
    return i


class TestExtensionMeasureGate:
    """Failure mode 7 -- the F0 incident, end to end. A mine over a vocabulary that CAN
    express the NaN-a.e. rootn sources must not ship any rule whose LHS applies rootn
    with an expression-position index: such a source is undefined almost everywhere, so
    the rewrite is a positive-measure domain extension (contract R3 forbids it at any
    positive measure). At the pre-fix HEAD this exact mine minted four of the five
    production poison rules (`rootn (-1) tanh ?0 -> (-1)`, ...) because the interval
    domain gate treated its unanalyzable boxes as proven-clean."""

    def test_nan_ae_rootn_rules_do_not_ship(self, tmp_path) -> None:
        (tmp_path / "rules.json").write_text(json.dumps([]))
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"engine_generation": 2, "operators": _ROOTN_OPERATORS, "rules": "rules.json"}))
        engine = SimpliPyEngine.from_config(str(config))
        engine.find_rules(
            max_source_pattern_length=4,
            dummy_variables=1,
            extra_internal_terms=["0", "1", "(-1)", "np.e"],
            X=512,
            promote_sorts=False,
            seed=42,
            verbose=False,
        )
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in engine.simplification_rules}
        # anti-vacuity tripwire: something sound must mint in this same mine, or every
        # assertion below holds vacuously. It was the parity family (`cosh neg ?0 ->
        # cosh ?0`) until 2026-08-07, when the parity arm took that whole family into
        # the CONSTRUCTORS and the mine stopped minting it -- the tripwire fired exactly
        # as designed. The successor is a transcendental inverse pair, which no
        # constructor evaluates and so no fold can absorb. This vocabulary has no
        # transcendental pair, so the successor is the ODD-function double negation
        # `-tanh(-x) = tanh(x)`: the outer sign is a bag coefficient rather than an even
        # FUNCTION, so the parity arm (which only plants sign-blindness at an even head)
        # does not reach it.
        assert (("neg", "tanh", "neg", "?0"), ("tanh", "?0")) in rules, sorted(rules)
        # the gate: no shipped LHS applies rootn with an operator-headed (expression)
        # index -- literal-index rootn is AC-native and never reaches the rule table
        arity = {name: spec["arity"] for name, spec in _ROOTN_OPERATORS.items()}
        wc = ("?0", "_0", "!0", "$0")
        for lhs, rhs in rules:
            # STAGE 2 carve-out: a GROUND source (no wildcard) has no domain to
            # extend -- `rootn 0 cosh (-1) -> nan` is a pointwise-exact collapse of a
            # state the mu-governed fold no longer materializes. The positive-measure
            # domain-extension hazard this gate guards against needs a wildcard.
            if not any(t in wc for t in lhs):
                continue
            for i, tok in enumerate(lhs):
                if tok != "rootn":
                    continue
                index_head = lhs[_subtree_end(lhs, i + 1, arity)]
                assert index_head not in arity, (
                    f"NaN-a.e. source shipped: {list(lhs)} -> {list(rhs)} rewrites a "
                    f"rootn with expression-position index ({index_head}(...)), a "
                    f"positive-measure domain extension")


class TestFoldFilter:
    """The library filter MIRRORS THE EMIT GUARD (2026-08-22): every var-free candidate
    of length >= 2 is dropped unconditionally, because the shared scan refuses every
    var-free composite TARGET at the mint and such a candidate can therefore never
    emit. Two 400-source byte-identity controls certified the drop changes no mined
    rule while removing ~97% of the scan's cost on var-free sources. The old
    `fold_filter` switch and its `folds_to_leaf` exemption are retired: the invariant
    is not optional, and the exemption admitted exactly the class the emit guard
    refuses."""

    def test_counts_and_unconditional_drop(self, engine, mining_x) -> None:
        x_flat, n = mining_x
        cands = [["x0"], ["<constant>"], ["exp", "<constant>"], ["pow2", "<constant>"],
                 ["exp", "1"], ["neg", "x0"], ["*", "<constant>", "x0"]]
        lib = engine._core.build_candidate_library(cands, ["x0"], x_flat, n)
        # every var-free composite drops -- including `exp 1`, which the retired
        # `folds_to_leaf` exemption used to admit: the leaf `np.e` is the candidate
        # that emits; the composite spelling never can (emit guard).
        assert lib.n_filtered == 3, "exp(<c>), pow2(<c>) and exp(1) are all var-free composites"
        assert lib.n_candidates == 4
        # and the drop needs no bare `<constant>` guard token: it is unconditional
        lib2 = engine._core.build_candidate_library(cands[2:], ["x0"], x_flat, n)
        assert lib2.n_filtered == 3
        assert lib2.n_candidates == 2

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
        # `exp 1` is dropped from the library (var-free composite); the exact
        # interval-intersection `<constant>` decision must carry the match alone.
        lib = engine._core.build_candidate_library(cands, ["x0"], x_flat, n)
        assert lib.n_filtered == 1
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
            "lib = eng._core.build_candidate_library(cands, ['x0'], x_flat, n)\n"
            f"out = eng._core.find_rule_lib({src!r}, {len(src)}, 2, lib,"
            " challenges=16, retries=16, seed=7, rtol=1e-9, atol=1e-12)\n"
            "assert out is not None, 'the feasible near-constant source must match'\n"
        )
        (tmp_path / "rules.json").write_text(json.dumps([]))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS, "rules": "rules.json"}))
        res = subprocess.run(
            [sys.executable, "-c", prog, str(cfg)],
            env={**os.environ, "SIMPLIPY_SPECIAL_BATTERY": "0", "OMP_NUM_THREADS": "1"},
            capture_output=True, text=True)
        assert res.returncode == 0, res.stderr

    def test_the_mine_never_ships_a_var_free_composite_rhs(self, tmp_path) -> None:
        """The end-to-end form of the invariant. The old parity gate A/B-ran the mine
        with the filter on and off; the switch is retired, so the assertable property is
        the emit guard's own: no mined rule may carry a var-free composite RHS (the
        universal-absorber class -- `cosh asin <constant> -> log <constant>` -- and every
        obfuscated respelling of a value). The shipped acj-4-3 artifact has 0 of 5,319."""
        (tmp_path / "rules.json").write_text(json.dumps([]))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(str(cfg))
        eng.find_rules(max_source_pattern_length=3, dummy_variables=1,
                       extra_internal_terms=["0", "1", "<constant>"], X=256, seed=7,
                       verbose=False, promote_sorts=False)
        rules = sorted((tuple(lhs), tuple(rhs)) for lhs, rhs in eng.simplification_rules)
        assert len(rules) > 0

        def var_free(t: str) -> bool:  # the emit guard's own predicate (worker.rs)
            return t != "x0" and not t.startswith(("_", "$", "?", "!"))

        for lhs, rhs in rules:
            assert not (len(rhs) >= 2 and all(var_free(t) for t in rhs)), (
                f"var-free composite RHS shipped: {lhs} -> {rhs}")


class TestProvenance:
    """The mined artifact must carry a reproducibility sidecar, and sampled sources are
    validated as universe members per run (exercised via the sampled length below -- a
    violation raises inside find_rules)."""

    def test_sidecar_written_with_reproducibility_fields(self, tmp_path) -> None:
        (tmp_path / "rules.json").write_text(json.dumps([]))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(str(cfg))
        out = str(tmp_path / "mined.json")
        eng.find_rules(max_source_pattern_length=3, dummy_variables=1,
                       extra_internal_terms=["0", "1", "<constant>"], X=256, seed=7,
                       verbose=False, output_file=out,
                       source_sample_per_length={3: 500}, promote_sorts=False)
        side = json.load(open(out + ".provenance.json"))
        assert side["params"]["seed"] == 7
        assert side["params"]["mine_seed"] and side["params"]["confirm_seed"]
        assert side["params"]["source_sample_per_length"] == {"3": 500}
        assert side["X"]["source"].startswith("seeded_mixture")
        assert side["universe"]["3"]["sampled"] is True
        assert 0 < side["universe"]["3"]["coverage"] <= 1
        assert side["universe"]["2"]["coverage"] == 1.0
        assert side["progress"]["final"] is True
        assert side["progress"]["rules_total"] == len(
            json.load(open(out.replace(".json", "_corpus.json"))))
        assert side["simplipy_version"]

    def test_sidecar_records_soundness_state(self, tmp_path) -> None:
        """The mine's SOUNDNESS PROVENANCE (audit Tier-1 #4): the four default-ON
        kill-switches ship recorded (a mine run with a soundness layer disabled must
        say so in its artifact), and the three interval fail-closed miss counters are
        read after the mine instead of assumed zero -- the exposure is a NUMBER in the
        sidecar, not a hope."""
        (tmp_path / "rules.json").write_text(json.dumps([]))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(str(cfg))
        out = str(tmp_path / "mined.json")
        eng.find_rules(max_source_pattern_length=3, dummy_variables=1,
                       extra_internal_terms=["0", "1", "<constant>"], X=256, seed=7,
                       verbose=False, output_file=out, promote_sorts=False)
        side = json.load(open(out + ".provenance.json"))
        sound = side["soundness"]
        assert sound["kill_switches"] == {
            "SIMPLIPY_IVL_GATE": True, "SIMPLIPY_IVL_CLASS": True,
            "SIMPLIPY_IVL_REACH": True, "SIMPLIPY_SPECIAL_BATTERY": True}
        assert sound["node_budget_env"] is None  # default budget, no override
        undecided = sound["interval_undecided"]
        assert set(undecided) == {"horizon", "node_budget", "unanalyzable"}
        assert all(isinstance(v, int) and v >= 0 for v in undecided.values())

    def test_sidecar_records_the_environment(self, tmp_path) -> None:
        """R5 + C23c-prov (the G2 gate): the artifact's identity includes the numeric
        stack that minted it -- scipy's PRESENCE alone changes mined output (N3:
        a rule reads PROMOTE with scipy and NO-WITNESS without), at least 5 shipped
        rules exist only because glibc 2.43 is 1 ulp wrong (L9), and none of it was
        recorded anywhere. The sidecar now carries the environment and a
        `libm_fingerprint` computed through the folder's OWN FFI path, so two
        machines can tell whether their mines are even comparable."""
        (tmp_path / "rules.json").write_text(json.dumps([]))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(str(cfg))
        out = str(tmp_path / "mined.json")
        eng.find_rules(max_source_pattern_length=2, dummy_variables=1,
                       X=64, seed=7, verbose=False, output_file=out,
                       promote_sorts=False)
        env = json.load(open(out + ".provenance.json"))["environment"]
        import platform
        assert env["python"] == platform.python_version()
        assert env["platform"] == platform.platform()
        for pkg in ("numpy", "scipy", "mpmath"):
            from importlib.metadata import version
            assert env[pkg] == version(pkg), f"{pkg} version missing or wrong"
        # the fingerprint is a sha256 over a fixed probe battery evaluated through
        # the deployed folding path -- 64 hex chars, deterministic on one host
        assert isinstance(env["libm_fingerprint"], str)
        assert len(env["libm_fingerprint"]) == 64
        assert env["libm_fingerprint"] == eng.libm_fingerprint()

    def test_empty_universe_length_is_vacuously_covered(self, tmp_path) -> None:
        """A binary-only alphabet has NO expressions of even length (a full binary tree
        over binary operators always has an odd token count), so the universe carries an
        empty cell. The mine must complete and record that cell as vacuously covered --
        the coverage quotient used to raise ZeroDivisionError on it."""
        binary_only = {name: spec for name, spec in _OPERATORS.items()
                       if spec["arity"] == 2}
        (tmp_path / "rules.json").write_text(json.dumps([]))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"engine_generation": 2, "operators": binary_only, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(str(cfg))
        out = str(tmp_path / "mined.json")
        eng.find_rules(max_source_pattern_length=3, max_target_pattern_length=2,
                       dummy_variables=1, extra_internal_terms=["0", "1"],
                       X=64, seed=3, verbose=False, output_file=out,
                       promote_sorts=False)
        side = json.load(open(out + ".provenance.json"))
        assert side["universe"]["2"]["complete_count"] == 0
        assert side["universe"]["2"]["used"] == 0
        assert side["universe"]["2"]["coverage"] == 1.0


class TestStageTwoConfirmation:
    """The stage-2 confirmation (confirm=True) must actually filter, not pass through."""

    def test_confirm_filters_a_data_luck_rule(self, engine, mining_x) -> None:
        """_confirm_mined_rules re-verifies on an INDEPENDENT wider X. A (source, target)
        pair that the checker rejects there must be dropped. Direct probe of the confirm
        primitive on the shipped defect (canonical `float("nan")` spelling -- the old
        artifact's bare `nan` is refused at the alphabet boundary, pinned above):
        asin(cosh(x0)) -> nan is rejected by the confirm X, so _confirm_mined_rules
        drops it while keeping a genuine rule."""
        x_confirm = RuleMiner(engine)._mining_sample_x(2048, 1, np.random.default_rng(99))
        kept = RuleMiner(engine)._confirm_mined_rules(
            [(('asin', 'cosh', 'x0'), ('float("nan")',)), (('+', 'x0', '0'), ('x0',))],
            ['x0'], x_confirm, 16, 16, 1e-9, 1e-12, 256, 123)
        assert (('asin', 'cosh', 'x0'), ('float("nan")',)) not in kept
        assert (('+', 'x0', '0'), ('x0',)) in kept

    def test_confirm_answers_the_question_it_was_asked(self, engine, mining_x) -> None:
        """The confirm must verify THE PROPOSED TARGET, not merely that `find_rule` returned
        something.

        `find_rule` does not always answer the question it is asked: for a variable-free
        `<constant>`-bearing source its all-constant short-circuit classifies the source and
        returns that CLASS LITERAL without ever reading the candidate list. While the confirm
        tested only `result is not None`, stage-2 confirmation was therefore VACUOUS for every
        such source -- it accepted whatever target it was handed. That is what let a false
        rule survive the gate whose entire purpose is to kill false rules.

        Each absurd pair below was measured to "confirm" before the fix; the control does not
        hit the short-circuit and failed correctly even then, which is what made the defect
        invisible to the existing test above."""
        x_confirm = RuleMiner(engine)._mining_sample_x(2048, 1, np.random.default_rng(99))
        absurd = [
            (('exp', '<constant>'), ('0',)),
            (('exp', '<constant>'), ('float("nan")',)),
            (('+', '<constant>', '1'), ('float("-inf")',)),
        ]
        kept = RuleMiner(engine)._confirm_mined_rules(
            absurd, ['x0'], x_confirm, 16, 16, 1e-9, 1e-12, 256, 99)
        assert kept == [], f"stage-2 confirmation is vacuous: it confirmed {kept}"
        # Power check: a genuine rule must still survive the same call.
        assert RuleMiner(engine)._confirm_mined_rules(
            [(('+', 'x0', '0'), ('x0',))], ['x0'], x_confirm, 16, 16, 1e-9, 1e-12, 256, 99)


class TestCertifyRules:
    """The public certification API for externally proposed rules (LLM/human proposals)."""

    def test_certifies_true_identity_rejects_false_and_verifies_hint(self, engine) -> None:
        proposals = [
            ["pow2", "tanh", "neg", "x0"],                           # (tanh -a)^2 = (tanh a)^2
            ["+", "exp", "x0", "cosh", "x1"],                        # no shorter equivalent -> reject
            ["*", "exp", "x0", "*", "exp", "x1", "exp", "x2"],       # minimal form is 6 tokens
            ["log", "*", "exp", "x0", "exp", "x1"],                  # log(e^a e^b) -> a+b: TRUE, unrealised
        ]
        hints = [None, None, ["exp", "+", "x0", "+", "x1", "x2"], None]
        out = engine.certify_rules(proposals, hints, dummy_variables=3, X=256, seed=7)
        by_src = {tuple(s): (t, c) for s, t, c in out}
        assert by_src[tuple(proposals[0])] == (("pow2", "tanh", "x0"), "minimal")
        assert tuple(proposals[1]) not in by_src
        assert by_src[tuple(proposals[2])] == (("exp", "+", "x0", "+", "x1", "x2"), "verified")
        # `log(e^a e^b) -> a+b` is exactly true on R and NOT realised by the shipped f64
        # engine -- exp overflows at 709.78, so log(exp(800)*exp(1)) is inf, not 801 --
        # which the deployed lane convicts as ENGINE-MISALIGN.
        #
        # Under the triple that verdict is no longer a death sentence: ENGINE-MISALIGN is
        # the `real` tier, so the pair SURVIVES certification and is routed into
        # rules_real.json / rules_corpus.json and out of rules.json. Deleting it here
        # would break the parity `certify_rules` promises with the mine, which keeps it.
        assert by_src[tuple(proposals[3])][0] == ('+', 'x0', 'x1')
        from simplipy.verify._contract import judge_rule
        assert judge_rule(list(proposals[3]), ['+', 'x0', 'x1'])['tier'] == 'real'

    def test_skips_sources_the_engine_already_reduces(self, engine) -> None:
        engine.find_rules(max_source_pattern_length=3, dummy_variables=1,
                          extra_internal_terms=["0", "1", "<constant>"], X=256,
                          seed=7, verbose=False, promote_sorts=False)
        out = engine.certify_rules([["+", "x0", "0"]], X=256, seed=7)
        assert out == []

    def test_the_instrument_convicts_the_saturation_impostor(self) -> None:
        """The premise of the symbolic-gate test below, pinned on its own: the
        precision-stability discriminator KILLs `tanh(exp(e)) -> 1` as a REAL change
        (mpmath dps=50: 1 - tanh(exp(e)) = 1.37e-13 -- stable as precision doubles,
        so a genuine gap between two functions, not rounding)."""
        from simplipy.verify import verify_rule
        v = verify_rule(["tanh", "exp", "np.e"], ["1"])
        assert v["verdict"] == "KILL"
        assert v["clause"] == "a-real-change"

    def test_symbolic_gate_kills_what_the_tolerance_cannot_see(self, engine) -> None:
        """B3: the docstring promises certified output "exactly as sound as mined
        rules", and the mining path earns that at the symbolic gate in `_finalize`
        (KILLs dropped before anything downstream sees the set). The public API ran
        no such gate, so `tanh(exp(e)) -> 1` -- false by 1.37e-13, below every usable
        numeric tolerance -- came back certified 'verified'. The control identity in
        the same call must still certify: the impostor's absence is then the GATE's
        verdict, not an upstream refusal."""
        out = engine.certify_rules(
            [["tanh", "exp", "np.e"], ["pow2", "tanh", "neg", "x0"]],
            [["1"], None],
            dummy_variables=2, extra_internal_terms=["np.e"], X=256, seed=7)
        by_src = {tuple(s): (t, c) for s, t, c in out}
        # The control must be a true identity the deployed engine ALSO performs, or its
        # own refusal would mask the gate's verdict. `log(e^a e^b) -> a+b` used to sit
        # here and no longer qualifies: true on R, but ENGINE-MISALIGN past exp's
        # overflow (F95). `(tanh -a)^2 = (tanh a)^2` is true and exactly realised --
        # tanh is odd in f64 and squaring kills the sign.
        assert by_src.get(("pow2", "tanh", "neg", "x0")) \
            == (("pow2", "tanh", "x0"), "minimal"), "control identity failed upstream"
        assert ("tanh", "exp", "np.e") not in by_src, \
            f"a symbolically-KILLed pair was certified: {out}"


class TestSymbolicGateFatalBuckets:
    """B7: the symbolic gate dropped only `verdict == 'KILL'`; the other six non-sound
    buckets (ENGINE-MISALIGN / NO-WITNESS / UNRESOLVED-COVERAGE / UNSUPPORTED-SHAPE /
    JUDGE-TIMEOUT) passed silently -- so a rule the judge COULD NOT EVALUATE shipped
    with the same standing as a certified one, and the sidecar recorded nothing. The
    instrument's own `is_clean` already declares all six dirty; the mine must not ship
    what its second authority cannot pass.

    Today the mine's kernel vocabulary is fully covered by the contract, so the
    coverage gap is simulated the way it would really arise -- an operator the mine
    evaluates but the judge does not -- by withdrawing `log` from the contract's
    arity table: every log-bearing rule then buckets UNSUPPORTED-SHAPE, exactly as a
    future vocabulary extension without a matching contract arm would."""

    def _withdraw_log(self, monkeypatch) -> None:
        from simplipy.verify import _contract
        monkeypatch.delitem(_contract.ARITY, "log")

    def test_the_simulated_gap_buckets_unsupported_shape(self, monkeypatch) -> None:
        """Premise pin: with `log` withdrawn the instrument refuses log-bearing rules
        as UNSUPPORTED-SHAPE (and still certifies a log-free identity)."""
        from simplipy.verify import verify_rule
        self._withdraw_log(monkeypatch)
        assert verify_rule(["log", "exp", "1"], ["1"])["verdict"] == "UNSUPPORTED-SHAPE"
        assert verify_rule(["+", "x0", "0"], ["x0"])["verdict"] == "CERTIFIED"

    def test_mine_refuses_what_the_judge_cannot_evaluate(self, tmp_path, monkeypatch) -> None:
        """The plan's acceptance: mining a config with an operator the judge cannot
        evaluate refuses the rule instead of shipping it, and the sidecar carries the
        full bucket census alongside killed/kept. The unpatched fixture mine mints
        five log-bearing rules (log 1 -> 0, log exp 1 -> 1, ...), so a zero census
        here would be a vacuous pass -- the census >= 1 assertion is the anti-vacuity."""
        self._withdraw_log(monkeypatch)
        (tmp_path / "rules.json").write_text(json.dumps([]))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(str(cfg))
        out = str(tmp_path / "mined.json")
        eng.find_rules(max_source_pattern_length=3, dummy_variables=1,
                       extra_internal_terms=["0", "1", "<constant>"], X=256, seed=7,
                       verbose=False, output_file=out, promote_sorts=False)
        mined = json.load(open(out))
        shipped_unjudgeable = [r for r in mined if "log" in r[0] or "log" in r[1]]
        assert shipped_unjudgeable == [], (
            f"rules the judge cannot evaluate shipped: {shipped_unjudgeable}")
        gate = json.load(open(out + ".provenance.json"))["symbolic_gate"]
        assert len(gate["unsupported_shape"]) >= 1, "vacuous: no log rule reached the gate"
        assert gate["killed"] == []
        assert gate["no_witness"] == []  # present even when empty: the channel ran
        census = gate["census"]
        assert census["UNSUPPORTED-SHAPE"] == len(gate["unsupported_shape"])
        assert census["CERTIFIED"] == gate["kept"]
        # `kept` is counted BEFORE the per-mode prune, and `rules.json` is written after
        # it, so the two differ by exactly what f64's own constructor already performs.
        # The sidecar states that rather than leaving "smaller than expected"
        # indistinguishable from "something was lost".
        prov = json.load(open(out + ".provenance.json"))
        assert gate["kept"] == len(mined) + prov["triple"]["preempted"]["f64"]

    def test_certify_rules_refuses_the_same_buckets(self, engine, monkeypatch) -> None:
        """The B3 gate site must apply the same fatal set: a certification the judge
        cannot evaluate is refused, not returned 'minimal'. The log-free control in
        the same call proves the refusal is the gate's."""
        self._withdraw_log(monkeypatch)
        out = engine.certify_rules(
            [["log", "*", "exp", "x0", "exp", "x1"],
             ["*", "exp", "x0", "exp", "x1"]],
            dummy_variables=2, X=256, seed=7)
        by_src = {tuple(s): (t, c) for s, t, c in out}
        assert by_src.get(("*", "exp", "x0", "exp", "x1")) \
            == (("exp", "+", "x0", "x1"), "minimal"), "control identity failed upstream"
        assert ("log", "*", "exp", "x0", "exp", "x1") not in by_src, \
            "a certification the judge cannot evaluate was returned"


class TestCoverageGateSearches:
    """The coverage gate must SEARCH, never decide from a spelling shrink (finding F2,
    audit D14). `* exp x0 exp x0` canon-respells to `pow(exp x0, 2)`: the token count
    drops (5 -> 4) while the semantic complexity does not move (4 -> 4). The old gate
    read that shrink as "already covered" and short-circuited, so the strictly better
    `exp(x0+x0)` (complexity 3) was never searched for -- and because the respelled
    form speaks `pow`/`2`, tokens OUTSIDE the mining alphabet, the family is not
    deferred to a later tier; it is lost. Acceptance ("strictly below the engine's own
    result in the serve ordering") rides the candidate scan itself: a refused match asks
    for the next candidate and an exhausted length moves to the next length, exactly the
    assignment-enumeration doctrine of finding F1."""

    DIAG = ["*", "exp", "x0", "exp", "x0"]
    BETTER = ("exp", "+", "x0", "x0")

    def test_respelled_source_is_searched_not_skipped(self, engine) -> None:
        """The diagonal source must certify through the LIBRARY path. This also pins the
        across-length continuation: at target length 3 the scan meets `pow2 exp x0`,
        numerically equivalent but parsing to the very Ex the engine already reaches
        (not strictly below -> refused); the scan must keep going and take the
        length-4 `exp + x0 x0`. The off-diagonal sibling never hit the gate and is the
        anti-vacuity control."""
        out = engine.certify_rules(
            [self.DIAG, ["*", "exp", "x0", "exp", "x1"]], dummy_variables=2, X=256, seed=7)
        by_src = {tuple(s): (t, c) for s, t, c in out}
        assert by_src.get(tuple(self.DIAG)) == (self.BETTER, "minimal")
        assert by_src.get(("*", "exp", "x0", "exp", "x1")) == (("exp", "+", "x0", "x1"), "minimal")

    def test_hint_certifies_when_the_library_cannot_spell_the_target(self, engine) -> None:
        """Same source, target length capped at 3: the library cannot spell the 4-token
        target, so certification must come through the HINT arm -- which the old gate
        also short-circuited past (the proposal channel refused the correct answer even
        when it was handed over explicitly)."""
        out = engine.certify_rules([self.DIAG], [list(self.BETTER)],
                                   max_target_pattern_length=3, dummy_variables=2,
                                   X=256, seed=7)
        assert [(tuple(s), tuple(t), c) for s, t, c in out] \
            == [(tuple(self.DIAG), self.BETTER, "verified")]

    def test_dead_on_arrival_hint_is_refused(self, engine) -> None:
        """A hint that does not sit strictly below the engine's own result mints a rule
        the serving pass would refuse (the mint-then-drop family): `inv inv x0` folds to
        `x0` natively, so the hint `x0` is exactly the engine's result, not below it."""
        assert engine.certify_rules([["inv", "inv", "x0"]], [["x0"]],
                                    dummy_variables=1, X=256, seed=7) == []

    def test_fully_folded_source_stays_uncertified(self, engine) -> None:
        """`+ x0 0` folds to `x0` at parse; nothing expressible beats `x0`. The verdict
        is the same as before the fix -- but now it is search-verified, not guessed
        from the spelling shrink."""
        assert engine.certify_rules([["+", "x0", "0"]], dummy_variables=1,
                                    X=256, seed=7) == []


class TestEmitParseClosure:
    """The engine must be able to read its own output (emit alphabet <= parse alphabet).
    `pow` and `rootn` are AC-language built-ins the serializer emits regardless of the
    config vocabulary; both must parse and validate under a config that declares
    neither. Found via F2: under this file's pow-less operator set,
    `simplify(simplify(x))` raised ValueError and the mine's relaxed acceptance
    (`ac_ordered_below` on the engine's own output) silently errored into a refusal."""

    def test_engine_rereads_its_own_output(self, engine) -> None:
        out = engine.simplify(["*", "exp", "x0", "exp", "x0"])
        assert out == ["pow", "exp", "x0", "2"], "respell expectation drifted"
        assert engine.is_valid(list(out)), "own output must validate"
        assert engine.simplify(list(out)) == out, "own output must re-simplify to itself"

    def test_ordering_judge_reads_own_output(self, engine) -> None:
        _, _, ac_out = engine._core.ac_judge(["*", "exp", "x0", "exp", "x0"], 48)
        assert engine._core.ac_ordered_below(["exp", "+", "x0", "x0"], ac_out) is True
        assert engine._core.ac_ordered_below(["pow2", "exp", "x0"], ac_out) is False


@pytest.fixture(scope="module")
def dev():
    # The full-vocabulary battery engine (this file's minimal test operator set has
    # no pow/trig): acj-4-3, generation-2 spellings throughout (audit Tier-1 #3 --
    # these batteries were the last dev_7-3-ruleset consumers).
    from conftest import acj_config_path, require_or_skip
    config = acj_config_path()
    require_or_skip(config, 'acj-4-3 config not staged')
    return SimpliPyEngine.from_config(config)


@pytest.fixture(scope="module")
def dev_x1(dev):
    X = RuleMiner(dev)._mining_sample_x(1024, 1, np.random.default_rng(0))
    return X.flatten(order='C').tolist(), X.shape[0]


class TestSpecialPointCertification:
    """The special-point battery + witness snapping (rust/battery.rs): the miner must
    reject at the contract's special points -- fitted-witness snapping before the
    domain gate, the per-variable battery (pi/2, ..., judged without the hiprec rescue),
    the special source-constant sweep, and the transcendental nan-seam probe -- while
    null-set completions and domain-preserving witnesses keep certifying.
    Uses the acj-4-3 engine: the minimal test operator set has no pow/trig."""

    def test_snapped_witness_domain_extension_rejected(self, dev, dev_x1) -> None:
        """exp(log(x^3)) = x^3 only on x > 0. The raw fitted exponent
        (2.9999999999999996) is NaN on x < 0 and blinds the interval domain gate; the
        SNAPPED witness 3.0 is total on R -- a positive-measure domain extension
        (a 37-rule family in (4,3) mining runs before this phase). The domain-PRESERVING
        non-integer witness exp(log(x)/3) = x^(1/3) must keep certifying -- and, the
        source being const-free, its slot resolves to the literal f64 of 1/3 (the
        Const-count invariant)."""
        x_flat, n = dev_x1
        assert dev._core.find_rule(
            ['exp', 'log', 'pow', 'x0', '3'], 5, 3, [['pow', 'x0', '<constant>']],
            ['x0'], x_flat, n) is None
        assert dev._core.find_rule(
            ['exp', '/', 'log', 'x0', '3'], 5, 3, [['pow', 'x0', '<constant>']],
            ['x0'], x_flat, n) == ['pow', 'x0', '0.3333333333333333']

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
        seam_plus = ['/', '+', 'pow', 'x0', '3', '*', 'pow', 'x0', '2', 'x1', '+', 'x0', 'x1']
        seam_minus = ['/', '-', 'pow', 'x0', '3', '*', 'pow', 'x0', '2', 'x1', '-', 'x0', 'x1']
        stable = ['/', '-', 'pow', 'x0', '2', 'pow', 'x1', '2', '+', 'x0', 'x1']
        # non-vacuity guard: none of these are already reduced by the loaded rules
        # (certify_rules skips already-covered sources without judging them)
        for src in (seam_plus, seam_minus, stable):
            assert len(dev.simplify(list(src))) >= len(src)
        out = dev.certify_rules(
            [seam_plus, seam_minus, stable],
            [['pow', 'x0', '2'], ['pow', 'x0', '2'], ['-', 'x0', 'x1']],
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
            fh.write(yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(cfg)
        out = os.path.join(directory, "mined.json")
        eng.find_rules(max_source_pattern_length=3, max_target_pattern_length=3,
                       dummy_variables=1, extra_internal_terms=["0", "1", "<constant>"],
                       X=256, seed=7, verbose=False, output_file=out, proposals=proposals,
                       promote_sorts=False)
        with open(out + ".provenance.json") as fh:
            sidecar = json.load(fh)
        return eng, sidecar, out

    def test_certifiable_proposal_joins_ruleset(self, tmp_path) -> None:
        """A certifiable proposal joins the ruleset through deduplicate_rules: the
        hint path ('verified') lands as a canonical rule in the engine AND the written
        artifact; an exact repeat of a certified proposal counts as 'duplicate' and
        adds nothing. The certifiable case is the coverage-ordering family's own:
        exp(t)*exp(t) collects to pow(exp t, 2) -- the SAME state respelled, no serve-
        ordering reduction -- and the hint exp(t+t) is a genuinely lower state the
        L<=3 library cannot express, so only the hint arm can save it. (The old
        certifiable case here, the masking-era hint `* <constant> pow2 x0`, is now
        refused by the Const-count invariant: a hint may not manufacture
        placeholders.)"""
        proposals = [
            {"source": ["+", "pow2", "x0", "pow2", "x0"], "why": "extra keys ignored"},
            {"source": ["*", "exp", "x0", "exp", "x0"],
             "target": ["exp", "+", "x0", "x0"]},                          # hint honored
            {"source": ["*", "exp", "x0", "exp", "x0"],
             "target": ["exp", "+", "x0", "x0"]},                          # duplicate of [1]
        ]
        eng, sidecar, out = self._mine(str(tmp_path), proposals)
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in eng.simplification_rules}
        # x^2 + x^2 IS 2x^2 as a state (canon collects in the bag), and nothing at
        # L<=3 beats that state: verdict 'rejected' (search exhausted), never a
        # coverage claim nothing backs.
        assert (("*", "exp", "?0", "exp", "?0"), ("exp", "+", "?0", "?0")) in rules
        assert not any("pow2" in lhs and lhs[:1] == ("+",) for lhs, _ in rules)
        # The corpus file is PRUNED: a rule corpus's own constructor performs is not
        # carried there. So the honest check is that every engine rule is either in the
        # file or preempted in that mode -- never silently missing. (This replaced a
        # comparison against the retired `corpus == f64 UNION real` identity.)
        from simplipy.mining import _preempted_by
        saved = {tuple(tuple(side) for side in rule)
                 for rule in json.load(open(out.replace('.json', '_corpus.json')))}
        saved |= {(tuple(lhs), tuple(rhs)) for lhs, rhs in eng.simplification_rules
                  if _preempted_by(eng, lhs, rhs, 'corpus')}
        assert rules == saved, "the artifact must contain the merged (mined + certified) ruleset"
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 1, "already_covered": 0, "rejected": 1, "duplicate": 1,
            "certified_then_dropped": 0}
        assert sidecar["proposals"]["count"] == 3 and sidecar["proposals"]["sha256"]

    def test_already_covered_proposal_is_skipped(self, tmp_path) -> None:
        """A proposal the mined rules already shorten is skipped exactly like an
        already-reducible source: counted 'already_covered', ruleset identical to a
        proposal-free mine."""
        eng_plain, _, _ = self._mine(str(tmp_path / "plain"), None)
        eng, sidecar, _ = self._mine(str(tmp_path / "covered"), [{"source": ["+", "x0", "0"]}])
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 0, "already_covered": 1, "rejected": 0, "duplicate": 0,
            "certified_then_dropped": 0}
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
            "certified": 0, "already_covered": 0, "rejected": 2, "duplicate": 0,
            "certified_then_dropped": 0}
        assert all("sin" not in rule[0] for rule in eng.simplification_rules)
        assert not any(tuple(lhs)[:1] == ("+",) and "cosh" in lhs
                       for lhs, _ in eng.simplification_rules)

    def test_provenance_carries_a_per_candidate_verdict_trail(self, tmp_path) -> None:
        """The sidecar records ONE trail entry per proposal, in file order, naming the gate
        that decided it. The aggregate tally cannot be audited on its own -- "N rejected"
        never says WHICH candidate died WHERE, so a reviewer cannot separate a correctly
        killed hallucination from a wrongly killed identity. Each verdict here is reached
        through a different gate, so the trail is pinned end to end."""
        proposals = [
            {"source": ["*", "exp", "x0", "exp", "x0"]},                      # respell: search-rejected
            {"source": ["+", "x0", "0"]},                                     # already covered (atomic)
            {"source": ["sin", "x0"]},                                        # outside vocabulary
            {"source": ["+", "exp", "x0", "cosh", "x0"],
             "target": ["exp", "cosh", "x0"]},                                # false: no target
            {"source": ["*", "exp", "x0", "exp", "x0"]},                      # repeat of [0]
        ]
        _, sidecar, _ = self._mine(str(tmp_path), proposals)
        trail = sidecar["proposals"]["trail"]
        assert len(trail) == len(proposals), "one entry per proposal, no drops"
        assert [e["source"] for e in trail] == [p["source"] for p in proposals], "file order"
        assert [(e["verdict"], e["stage"]) for e in trail] == [
            # exp*exp collects to pow(exp,2) -- the same state respelled, no serve-
            # ordering reduction, and nothing expressible at L3 beats it: search-
            # rejected, never a coverage claim (state-coverage, the F5 ruling)
            ("rejected", "search"),
            ("already_covered", "covered"),   # + x0 0 IS x0: atomic state, genuine coverage
            ("rejected", "vocabulary"),
            ("rejected", "search"),
            ("rejected", "search"),           # the repeat re-runs the same search gate
        ]
        # The trail and the tally are two views of one decision, never independently derived.
        counts = sidecar["proposals"]["outcomes"]
        for verdict, n in counts.items():
            assert sum(1 for e in trail if e["verdict"] == verdict) == n, verdict

    def test_proposal_channel_is_deterministic(self, tmp_path) -> None:
        """Two runs from the same master seed and the same proposals FILE (bare-list
        schema) produce byte-identical rulesets and identical provenance counts --
        file order + content-derived per-proposal seeds, nothing position- or
        entropy-derived."""
        proposals_file = tmp_path / "proposals.json"
        proposals_file.write_text(json.dumps([
            {"source": ["*", "exp", "x0", "exp", "x0"]},
            {"source": ["*", "exp", "x0", "exp", "x0"],
             "target": ["exp", "+", "x0", "x0"]},
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
            "certified": 1, "already_covered": 1, "rejected": 2, "duplicate": 0,
            "certified_then_dropped": 0}


class TestProposalTargetAllowance:
    """The proposals target allowance (owner ruling 2026-08-18: "If we allow that, the
    rules should still strictly decrease *something*, probably mu' then").

    PINS, DOES NOT CHANGE: measured 2026-08-18, the shipped hint arm ALREADY implements
    the ruling. `_certify_proposals` never bounds a hint by ``max_target_pattern_length``
    (the bound caps only the mine's own candidate SEARCH); what every hinted target must
    pass instead is the strict-descent gate ``ac_ordered_below(hint, ac_out)`` against
    the engine's own endpoint -- the serve-time reduction ordering
    (``ac::rules::ordered_below``: measure first, canonical total order on ties, STRICT).
    The same predicate re-vets every loaded rule statically (gate G7,
    rust/ac/rules.rs `translate`) and every fire at runtime (`oriented`), so a
    proposals-sourced rule that certifies here is exactly as descent-bound as a mined
    one. The gate calls the ORDERING, never a measure by name: when the unified measure
    becomes mu' (feat/mu-prime, 0.14), every gate follows automatically.

    Three pins: (1) the distribute family's 5-token target certifies PAST a
    ``max_target_pattern_length: 3`` mine -- the length cap does not bind hints; (2) a
    numerically TRUE long-target proposal whose target does not strictly descend the
    ordering is REJECTED; (3) an equal-state respell hint (a tie in the ordering) is
    REJECTED -- nothing certifies on equal-measure ties.

    (The acj-4-3-llm batch's distribute family died at the VOCABULARY gate -- x2 with
    ``dummy_variables: 2`` -- not at any target-length bound; see the census replay,
    the proposal census replay. Unlocking it there is a dummy-universe question,
    outside this ruling.)"""

    @staticmethod
    def _mine(directory, proposals, operators=_OPERATORS, dummy_variables=1,
              extra_internal_terms=("0", "1", "<constant>")):
        """One small mine (L<=3 sources, L<=3 targets) with a proposal batch."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "rules.json"), "w") as fh:
            fh.write("[]")
        cfg = os.path.join(directory, "config.yaml")
        with open(cfg, "w") as fh:
            fh.write(yaml.safe_dump({"engine_generation": 2, "operators": operators,
                                     "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(cfg)
        out = os.path.join(directory, "mined.json")
        eng.find_rules(max_source_pattern_length=3, max_target_pattern_length=3,
                       dummy_variables=dummy_variables,
                       extra_internal_terms=list(extra_internal_terms),
                       X=256, seed=7, verbose=False, output_file=out, proposals=proposals,
                       promote_sorts=False)
        with open(out + ".provenance.json") as fh:
            sidecar = json.load(fh)
        return eng, sidecar, out

    # Binary pow beside the module set: the non-descent specimens live in the
    # pow-collect family. A LOCAL table -- extending _OPERATORS would re-mine every
    # other suite's engine.
    _OPERATORS_POW = {**_OPERATORS,
                      "pow": {"realization": "simplipy.operators.pow", "alias": ["power"],
                              "arity": 2, "precedence": 3, "commutative": False}}

    def test_distribute_family_five_token_target_certifies_past_the_bound(self, tmp_path) -> None:
        """`x0*x1 + x0*x2 -> x0*(x1+x2)`: a 7-token source with a 5-token hinted target
        certifies through a ``max_target_pattern_length: 3`` mine (the bound caps the
        candidate search, never the hint), because the target strictly descends the
        serve ordering. The certified rule joins the ruleset, the artifact, and the
        SERVING engine."""
        src = ["+", "*", "x0", "x1", "*", "x0", "x2"]
        tgt = ["*", "x0", "+", "x1", "x2"]
        eng, sidecar, out = self._mine(
            str(tmp_path), [{"source": src, "target": tgt}], dummy_variables=3)
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 1, "already_covered": 0, "rejected": 0, "duplicate": 0,
            "certified_then_dropped": 0}
        assert sidecar["proposals"]["trail"][-1] == {
            "source": src, "target": tgt, "verdict": "certified", "stage": "accepted",
            "certificate": "verified"}
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in eng.simplification_rules}
        assert (("+", "*", "?0", "?1", "*", "?0", "?2"), ("*", "?0", "+", "?1", "?2")) in rules
        # The corpus file is PRUNED: a rule corpus's own constructor performs is not
        # carried there. So the honest check is that every engine rule is either in the
        # file or preempted in that mode -- never silently missing. (This replaced a
        # comparison against the retired `corpus == f64 UNION real` identity.)
        from simplipy.mining import _preempted_by
        saved = {tuple(tuple(side) for side in rule)
                 for rule in json.load(open(out.replace('.json', '_corpus.json')))}
        saved |= {(tuple(lhs), tuple(rhs)) for lhs, rhs in eng.simplification_rules
                  if _preempted_by(eng, lhs, rhs, 'corpus')}
        assert rules == saved, "the artifact must carry the merged ruleset"
        # The allowance is only real if the rule FIRES at serve time: the engine's
        # endpoint for the source must now sit strictly below it in the ordering.
        assert eng._core is not None
        _, _, endpoint = eng._core.ac_judge(src, 48)
        assert eng._core.ac_ordered_below(endpoint, src), \
            "the certified long-target rule must actually reduce its source at serve time"

    def test_true_but_non_descending_long_target_is_rejected(self, tmp_path) -> None:
        """`pow(x0,2)*pow(x1,2) -> pow(x0*x1,2)` is numerically TRUE everywhere and
        spelling-shorter (7 -> 5 tokens), but the target does NOT strictly descend the
        serve ordering below the engine's endpoint -- so the proposal is REJECTED at the
        search/hint stage. The refusal is the ordering gate, pinned directly: descent
        decides, never spelling length, and a true identity is not enough."""
        src = ["*", "pow", "x0", "2", "pow", "x1", "2"]
        tgt = ["pow", "*", "x0", "x1", "2"]
        eng, sidecar, _ = self._mine(
            str(tmp_path), [{"source": src, "target": tgt}],
            operators=self._OPERATORS_POW, dummy_variables=2,
            extra_internal_terms=("0", "1", "2", "<constant>"))
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 0, "already_covered": 0, "rejected": 1, "duplicate": 0,
            "certified_then_dropped": 0}
        assert sidecar["proposals"]["trail"][-1]["verdict"] == "rejected"
        assert sidecar["proposals"]["trail"][-1]["stage"] == "search"
        assert eng._core is not None
        _, _, endpoint = eng._core.ac_judge(src, 48)
        assert not eng._core.ac_ordered_below(tgt, endpoint), \
            "the specimen must be refused BY THE ORDERING (or it belongs in the certify pin above)"
        assert not any(tuple(rhs) == ("pow", "*", "?0", "?1", "2")
                       for _, rhs in eng.simplification_rules)

    def test_equal_state_long_hint_is_rejected_on_the_tie(self, tmp_path) -> None:
        """`exp(x0)*exp(x0) -> pow(exp(x0), 2)`: the hint IS the source's canonical
        state respelled -- an exact TIE in the (measure, canonical-order) pair. Nothing
        may certify on an equal-measure tie: the strict gate refuses both directions and
        the proposal dies 'rejected'/'search', never as a coverage claim (F2/F5: a
        respell is not a reduction)."""
        src = ["*", "exp", "x0", "exp", "x0"]
        tgt = ["pow", "exp", "x0", "2"]
        eng, sidecar, _ = self._mine(
            str(tmp_path), [{"source": src, "target": tgt}],
            operators=self._OPERATORS_POW, dummy_variables=1,
            extra_internal_terms=("0", "1", "2", "<constant>"))
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 0, "already_covered": 0, "rejected": 1, "duplicate": 0,
            "certified_then_dropped": 0}
        assert sidecar["proposals"]["trail"][-1]["verdict"] == "rejected"
        assert sidecar["proposals"]["trail"][-1]["stage"] == "search"
        assert eng._core is not None
        _, _, endpoint = eng._core.ac_judge(src, 48)
        assert not eng._core.ac_ordered_below(tgt, endpoint), "tie: not below"
        assert not eng._core.ac_ordered_below(endpoint, tgt), "tie: not above either"


class TestLadderSnapshots:
    """find_rules(snapshot_at=...): the cells of one `j` are a PREFIX CHAIN, so a tall climb
    can emit the shorter cells it passes through instead of each being re-mined from scratch.

    The re-use is only legitimate if a snapshot is INDISTINGUISHABLE from a one-shot mine of
    that cell, and if emitting one does not disturb the climb -- the post-pass rewrites the
    engine's live rule state, so a missing restore would silently corrupt every length above
    the snapshot. Both directions are pinned here.

    These mine with ``prune='covered'`` and WITHOUT ``promote_sorts``: promotion runs its own
    positive controls over the dev vocabulary (sin/tan/atanh/pow1_3/np.pi/...), so it cannot
    run on a toy operator set at all. The prune exercises the same thing the restore has to
    survive -- a post-pass stage that rebinds ``simplification_rules``, recompiles, and pushes
    to the core. Snapshot-vs-one-shot equality WITH promotion on is gated end-to-end against
    the real publication configs (the prefix-identity gate)."""

    @staticmethod
    def _mine(directory, max_source, snapshot_at=None):
        """One small mine (one dummy, tiny X) -- optionally a climb with snapshots."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "rules.json"), "w") as fh:
            fh.write("[]")
        cfg = os.path.join(directory, "config.yaml")
        with open(cfg, "w") as fh:
            fh.write(yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS, "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(cfg)
        out = os.path.join(directory, "mined.json")
        eng.find_rules(max_source_pattern_length=max_source, max_target_pattern_length=2,
                       dummy_variables=1, extra_internal_terms=["0", "1", "<constant>"],
                       X=128, seed=11, verbose=False, output_file=out, prune="covered",
                       snapshot_at=snapshot_at, promote_sorts=False)
        return eng, out

    def test_snapshot_equals_one_shot_mine_of_that_cell(self, tmp_path) -> None:
        """The artifact a climb emits at length i equals a standalone mine of cell (i,j) --
        byte-for-byte in the rules, and field-for-field in the sidecar apart from the
        ladder-origin record and timestamps."""
        _, one_shot = self._mine(str(tmp_path / "oneshot"), 2)
        snap = str(tmp_path / "climb" / "snap_2.json")
        self._mine(str(tmp_path / "climb"), 3, snapshot_at={2: snap})
        assert open(one_shot).read() == open(snap).read(), \
            "snapshot rules differ from a one-shot mine of the same cell"
        a = json.load(open(one_shot + ".provenance.json"))
        b = json.load(open(snap + ".provenance.json"))
        assert "ladder_snapshot" not in a and b["ladder_snapshot"] == {
            "emitted_at_source_length": 2,
            "climb_max_source_pattern_length": 3,
            "equivalence": b["ladder_snapshot"]["equivalence"],
        }
        assert a["params"] == b["params"], "snapshot params must describe the CELL, not the climb"
        assert a["universe"] == b["universe"], "snapshot universe must be trimmed to its own lengths"
        assert a.get("sort_promotion") == b.get("sort_promotion")

    def test_emitting_a_snapshot_does_not_disturb_the_climb(self, tmp_path) -> None:
        """The top cell is unaffected by whether shorter cells were emitted en route. This is
        the restore path: `_finalize` prunes/promotes the live rule state, and the lengths
        above a snapshot must keep mining against the RAW un-pruned mine."""
        _, plain = self._mine(str(tmp_path / "plain"), 3)
        _, snapped = self._mine(str(tmp_path / "snapped"), 3,
                                snapshot_at={2: str(tmp_path / "snapped" / "snap_2.json")})
        assert open(plain).read() == open(snapped).read(), \
            "emitting a snapshot changed the climb's own output"

    def test_snapshot_outside_the_climb_is_refused(self, tmp_path) -> None:
        """Fail closed: a snapshot at or above the climb's own height would silently write
        nothing, and a missing artifact is discovered days later."""
        for bad in (3, 4, 0):
            with pytest.raises(ValueError, match="snapshot_at length"):
                self._mine(str(tmp_path / f"bad{bad}"), 3, snapshot_at={bad: "x.json"})


class TestMineSingleFlight:
    """Hardening H-002/H-008/H-009 (2026-08-03).

    The interval soundness counters the provenance sidecar reads are process-global,
    so mining is SINGLE-FLIGHT per process -- enforced by a real lock with a loud
    error, not by accident. The old accident was H-008: ``find_rules`` installed its
    SIGINT handler unconditionally, which made any non-main-thread mine die with
    ``ValueError: signal only works in main thread`` at entry. And H-009: the handler
    used to be installed ~100 lines before the try/finally that restores it, so an
    early validation raise (a malformed proposals file -- the documented fail-fast)
    leaked the custom handler into the process."""

    _OPS = {
        "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2,
              "precedence": 2, "commutative": True},
        "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg",
                "arity": 1, "precedence": 2.5, "commutative": False},
        "abs": {"realization": "np.abs", "alias": [], "inverse": None, "arity": 1,
                "precedence": 3, "commutative": False},
    }

    def test_mine_runs_off_main_thread(self, tmp_path):
        import threading
        eng = SimpliPyEngine(operators=dict(self._OPS), rules=[])
        out = str(tmp_path / 'rules.json')
        result: dict = {}

        def run() -> None:
            try:
                eng.find_rules(max_source_pattern_length=2, max_target_pattern_length=1,
                               X=64, output_file=out, reset_rules=True, verbose=False)
                result['ok'] = True
            except Exception as ex:  # pragma: no cover - the failure surface under test
                result['err'] = f'{type(ex).__name__}: {ex}'

        t = threading.Thread(target=run)
        t.start()
        t.join()
        assert result.get('ok'), result.get('err')
        side = json.load(open(out + '.provenance.json'))
        assert 'soundness' in side  # the sidecar section still ships off-main-thread

    def test_second_concurrent_mine_raises(self):
        from simplipy import engine as engine_mod
        eng = SimpliPyEngine(operators=dict(self._OPS), rules=[])
        assert engine_mod._MINE_LOCK.acquire(blocking=False)
        try:
            with pytest.raises(RuntimeError, match='single-flight'):
                eng.find_rules(max_source_pattern_length=2, max_target_pattern_length=1,
                               X=64, verbose=False)
        finally:
            engine_mod._MINE_LOCK.release()

    def test_early_raise_leaks_neither_handler_nor_lock(self):
        import signal
        from simplipy import engine as engine_mod
        before = signal.getsignal(signal.SIGINT)
        eng = SimpliPyEngine(operators=dict(self._OPS), rules=[])
        with pytest.raises(Exception):
            eng.find_rules(max_source_pattern_length=2,
                           proposals='/nonexistent/proposals.json', verbose=False)
        assert signal.getsignal(signal.SIGINT) is before
        assert engine_mod._MINE_LOCK.acquire(blocking=False)
        engine_mod._MINE_LOCK.release()


class TestD25MeasureFingerprintAtLoad:
    """D25 (R6's warn-on-mismatch half): the sidecar records the measure fingerprint
    but nothing ever READ it back -- recording-without-reading is the inert-control
    pattern (R2.2 confirmed write-only at HEAD). A ruleset mined under one measure
    served silently under another is exactly what the fingerprint exists to catch;
    the load path now compares digests and warns loudly on mismatch."""

    def _asset(self, tmp_path, digest):
        (tmp_path / "rules.json").write_text(json.dumps([]))
        (tmp_path / "rules.json.provenance.json").write_text(json.dumps(
            {"measure": {"digest": digest}}))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"engine_generation": 2, "operators": _OPERATORS,
                                       "rules": "rules.json"}))
        return str(cfg)

    def test_mismatch_warns(self, tmp_path) -> None:
        with pytest.warns(UserWarning, match="measure fingerprint"):
            SimpliPyEngine.from_config(self._asset(tmp_path, "0" * 16))

    def test_match_stays_silent(self, tmp_path) -> None:
        import warnings as _w
        probe = SimpliPyEngine.from_config(self._asset(tmp_path, "0" * 16))
        good = probe._measure_fingerprint()["digest"]
        # Scoped to THIS test's subject, the fingerprint. The fixture's ruleset is
        # deliberately empty (`rules.json` == []), which since the rules-less-state
        # ruling legitimately warns on its own account; a blanket "no UserWarning"
        # assertion here would be asserting something this test never meant.
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always", UserWarning)
            SimpliPyEngine.from_config(self._asset(tmp_path, good))
        assert not [w for w in caught if "measure fingerprint" in str(w.message)]


class TestMissed003SingleFlightPreservesRules:
    """missed-003: `find_rules` cleared the ruleset BEFORE acquiring the single-flight
    lock, so a call REJECTED by single-flight had already destroyed the caller's rules
    (measured in the audit: 6,671 -> RuntimeError -> 0, with `simplify` silently
    answering from native arithmetic afterwards). The reset now happens under the
    lock: a rejected call must leave the engine exactly as it found it."""

    def test_rejected_mine_leaves_rules_untouched(self, engine) -> None:
        import simplipy.engine as em
        engine.simplification_rules = [(("+", "?0", "0"), ("?0",))]
        engine.compile_rules()
        before = list(engine.simplification_rules)
        assert em._MINE_LOCK.acquire(blocking=False), 'lock unexpectedly held'
        try:
            with pytest.raises(RuntimeError, match='single-flight'):
                engine.find_rules(max_source_pattern_length=2, dummy_variables=1,
                                  X=64, seed=7, reset_rules=True, promote_sorts=False)
        finally:
            em._MINE_LOCK.release()
        assert engine.simplification_rules == before, \
            'a rejected mine destroyed the caller rules'


class TestD28CertifyLock:
    """D28: one-miner-per-process is doctrine, and `certify_rules` was found
    UNGUARDED -- it runs the same certification machinery (and mutates the same
    process-global interval counters the sidecar records) as a mine, so it must
    hold the same single-flight lock."""

    def test_certify_rejected_while_a_mine_is_active(self, engine) -> None:
        import simplipy.engine as em
        assert em._MINE_LOCK.acquire(blocking=False)
        try:
            with pytest.raises(RuntimeError, match='single-flight'):
                engine.certify_rules([["+", "x0", "0"]], X=64, seed=7)
        finally:
            em._MINE_LOCK.release()


class TestProposalHintLengthAllowance:
    """The proposal hint arm's LENGTH gate, removed in favour of the serve ordering
    (owner ruling 2026-08-18: "Only 'remove' the length gate for the LLM proposed
    rules. We can't afford to do 5-5 in general, for example, that would be way to
    expensive to mine").

    Both hint arms -- :meth:`certify_rules` (the standalone proposal API) and the
    ``find_rules(proposals=...)`` channel -- used to require ``len(hint) < len(source)``
    BEFORE consulting the ordering. That is a RAW TOKEN COUNT, not the engine's
    ordering, and the two disagree: `1 - cos(2*x0) = 2*sin(x0)^2` is 6 tokens against
    6, so the count refused it, while the serve ordering has the target STRICTLY below
    the engine's own endpoint (the `-` spelling costs two extra canonical nodes:
    `Add[1, Mul[-1, Cos[...]]]` against `Mul[2, Pow[Sin[x0], 2]]`). The symbolic gate
    rates the pair CERTIFIED; adding it makes the engine reduce its source. The count
    is now gone from both arms and ``ac_ordered_below`` -- already the next conjunct in
    the same chain, and the same predicate G7 and the serve-time `oriented` check apply
    downstream -- is the sole target-shape gate. It is STRICT, so ties still die.

    The owner's cost argument is untouched: a proposal skips the candidate search
    entirely (the hint IS the target), so dropping its pre-filter costs no mining time.

    THE MINED-CANDIDATE SEARCH NO LONGER KEEPS A TOKEN BOUND (owner ruling, re-mine
    2026-08-20). It is bounded by mu instead: the scan walks mu tiers ascending and stops
    at the first tier that cannot be ordered below the source's mark, which is the SAME
    predicate acceptance applies. What used to be two criteria -- a token count for search,
    mu for acceptance -- is one, and the half that was refusing mu-descending targets for
    being long is gone. The cost argument survives intact, because mu is what bounds the
    scan: a 5-5 mine is unaffordable for the size of its candidate library, not because
    targets were length-capped.

    So the pair below that the hint arm accepts is now reached by the MINED search too,
    and that is the ruling working rather than a leak.
    """

    # Trig + binary pow beside the module set: the half-angle family is where the
    # token count and the ordering disagree. A LOCAL table -- extending _OPERATORS
    # would re-mine every other suite's engine.
    _OPERATORS_TRIG = {
        "+": {"realization": "+", "alias": [], "arity": 2, "precedence": 1, "commutative": True},
        "-": {"realization": "-", "alias": [], "arity": 2, "precedence": 1, "commutative": False},
        "*": {"realization": "*", "alias": [], "arity": 2, "precedence": 2, "commutative": True},
        "pow": {"realization": "simplipy.operators.pow", "alias": ["power"], "arity": 2, "precedence": 3, "commutative": False},
        "sin": {"realization": "simplipy.operators.sin", "alias": [], "arity": 1, "precedence": 2, "commutative": False},
        "cos": {"realization": "simplipy.operators.cos", "alias": [], "arity": 1, "precedence": 2, "commutative": False},
    }

    # (a) EQUAL token count, strictly below in the ordering -- the wrongly refused pair.
    SIN_SRC = ["-", "1", "cos", "*", "2", "x0"]
    SIN_TGT = ["*", "2", "pow", "sin", "x0", "2"]
    # (b) its sibling: equal token count, and the engine's own endpoint sits strictly
    # BELOW the proposed target -- a legitimate refusal that must survive the change.
    COS_SRC = ["+", "1", "cos", "*", "2", "x0"]
    COS_TGT = ["*", "2", "pow", "cos", "x0", "2"]
    # (c) strictly LONGER target (8 -> 9) that strictly descends: x0 - x0*cos(2*x0).
    LONG_SRC = ["-", "x0", "*", "x0", "cos", "*", "2", "x0"]
    LONG_TGT = ["*", "*", "2", "x0", "*", "sin", "x0", "sin", "x0"]
    # (d) a SHORTER target (7 -> 6) that the old count gate waved through: the same
    # canonical state respelled, an exact tie in the ordering. Strictness kills it.
    TIE_SRC = ["*", "2", "*", "sin", "x0", "sin", "x0"]
    TIE_TGT = ["*", "2", "pow", "sin", "x0", "2"]

    @classmethod
    def _engine(cls, directory) -> SimpliPyEngine:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "rules.json"), "w") as fh:
            fh.write("[]")
        cfg = os.path.join(directory, "config.yaml")
        with open(cfg, "w") as fh:
            fh.write(yaml.safe_dump({"engine_generation": 2, "operators": cls._OPERATORS_TRIG,
                                     "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(cfg)
        assert eng._core is not None, "compiled core failed to attach"
        return eng

    @classmethod
    def _mine(cls, directory, proposals):
        """One small mine (L<=3 sources, L<=3 targets, one dummy) with a proposal batch."""
        eng = cls._engine(directory)
        out = os.path.join(directory, "mined.json")
        eng.find_rules(max_source_pattern_length=3, max_target_pattern_length=3,
                       dummy_variables=1, extra_internal_terms=["1", "2", "<constant>"],
                       X=256, seed=7, verbose=False, output_file=out, proposals=proposals,
                       promote_sorts=False)
        with open(out + ".provenance.json") as fh:
            sidecar = json.load(fh)
        return eng, sidecar, out

    def test_equal_length_descending_hint_certifies_and_serves(self, tmp_path) -> None:
        """(a) `1 - cos(2*x0) = 2*sin(x0)^2`: 6 tokens against 6, so the token count
        refused it outright; the ordering has it strictly below the engine's endpoint.
        It must now certify through BOTH hint arms, land in the ruleset and the written
        artifact, and actually SERVE -- `simplify` returns the target."""
        eng = self._engine(str(tmp_path / "certify"))
        out = eng.certify_rules([self.SIN_SRC], [self.SIN_TGT], dummy_variables=1,
                                extra_internal_terms=["1", "2", "<constant>"],
                                max_target_pattern_length=3, X=256, seed=7)
        assert [(tuple(s), tuple(t), c) for s, t, c in out] \
            == [(tuple(self.SIN_SRC), tuple(self.SIN_TGT), "verified")], \
            "certify_rules refused a hint that strictly descends the serve ordering"

        eng, sidecar, artifact = self._mine(
            str(tmp_path / "mine"), [{"source": self.SIN_SRC, "target": self.SIN_TGT}])
        assert sidecar["proposals"]["trail"][-1] == {
            "source": self.SIN_SRC, "target": self.SIN_TGT, "verdict": "certified",
            "stage": "accepted", "certificate": "verified"}
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in eng.simplification_rules}
        assert (("-", "1", "cos", "*", "2", "?0"),
                ("*", "2", "pow", "sin", "?0", "2")) in rules
        saved = {tuple(tuple(side) for side in rule)
                 for rule in json.load(open(artifact.replace('.json', '_corpus.json')))}
        assert rules == saved, "the artifact must carry the merged ruleset"
        # SERVE. `simplify` is DIALECT-PRESERVING (owner-ruled): it answers in the
        # dialect it was handed, and SIN_SRC is explicit binary prefix, so the answer is
        # too -- `to_tagged` is the separate step for the bag spelling. `ac_judge`'s
        # endpoint is the same state in the mine's binary alphabet. Before the merge both
        # are the source itself -- the rule is what moves them.
        assert eng.simplify(list(self.SIN_SRC)) \
            == ["*", "2", "pow", "sin", "x0", "2"], \
            "the certified rule does not fire at serve time"
        assert eng.to_tagged(eng.simplify(list(self.SIN_SRC))) \
            == ["<mul>", "2", "pow", "sin", "x0", "2", "</mul>"], \
            "the bag spelling must still be one conversion away"
        assert eng._core is not None
        _, _, endpoint = eng._core.ac_judge(list(self.SIN_SRC), 48)
        assert endpoint == self.SIN_TGT

    def test_ascending_sibling_is_refused_by_the_ordering_gate(self, tmp_path) -> None:
        """(b) `1 + cos(2*x0) = 2*cos(x0)^2` is refused BEFORE and AFTER -- but the
        deciding gate changes. Every OTHER conjunct of the mint chain is asserted to
        pass here (vocabulary/validity, wildcard multiplicity, the Const-count
        invariant, the licence registry, and the numeric confirmation), and
        `ac_ordered_below` is asserted False: with the token count gone, the ordering
        is the only bar left standing between this pair and a certificate. The
        observable verdict stays `rejected`/`search` -- the documented meaning of that
        stage is "no target in the candidate library and no verifiable hint"."""
        eng = self._engine(str(tmp_path / "probe"))
        core = eng._core
        assert core is not None
        _, _, endpoint = core.ac_judge(list(self.COS_SRC), 48)
        # every gate the pair PASSES
        assert len(self.COS_TGT) == len(self.COS_SRC), "the retired gate refused on this"
        assert eng.is_valid(list(self.COS_TGT))
        assert not violates_wildcard_multiplicity(list(self.COS_SRC), list(self.COS_TGT))
        assert self.COS_TGT.count("<constant>") <= self.COS_SRC.count("<constant>")
        assert core.registry_mint_refusal(
            list(self.COS_SRC), endpoint, list(self.COS_TGT), ["x0"]) is None
        X_data = RuleMiner(eng)._mining_sample_x(256, 1, np.random.default_rng(0))
        assert RuleMiner(eng)._confirm_mined_rules(
            [(tuple(self.COS_SRC), tuple(self.COS_TGT))], ["x0"], X_data,
            16, 16, 1e-9, 1e-12, 32, 7), "the pair is numerically TRUE; it dies elsewhere"
        # the ONE gate it fails -- and it fails it in both directions of the ordering:
        # the engine's own endpoint sits strictly BELOW the proposed target, so the
        # hint is an ASCENT, not a reduction.
        assert not core.ac_ordered_below(list(self.COS_TGT), endpoint), \
            "the ordering must be the gate that refuses this pair"
        assert core.ac_ordered_below(endpoint, list(self.COS_TGT))

        _, sidecar, _ = self._mine(
            str(tmp_path / "mine"), [{"source": self.COS_SRC, "target": self.COS_TGT}])
        assert sidecar["proposals"]["trail"][-1] == {
            "source": self.COS_SRC, "target": None, "verdict": "rejected",
            "stage": "search", "certificate": None}
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 0, "already_covered": 0, "rejected": 1, "duplicate": 0,
            "certified_then_dropped": 0}

    def test_longer_target_that_strictly_descends_certifies(self, tmp_path) -> None:
        """(c) `x0 - x0*cos(2*x0) = 2*x0*sin(x0)*sin(x0)`: the target is strictly
        LONGER (8 -> 9 tokens) and strictly BELOW in the ordering. Nothing about a
        hint's token count may bar it any more."""
        eng = self._engine(str(tmp_path / "certify"))
        core = eng._core
        assert core is not None
        assert len(self.LONG_TGT) > len(self.LONG_SRC), "specimen must be LONGER"
        out = eng.certify_rules([self.LONG_SRC], [self.LONG_TGT], dummy_variables=1,
                                extra_internal_terms=["1", "2", "<constant>"],
                                max_target_pattern_length=3, X=256, seed=7)
        assert [(tuple(s), tuple(t), c) for s, t, c in out] \
            == [(tuple(self.LONG_SRC), tuple(self.LONG_TGT), "verified")]
        _, _, endpoint = core.ac_judge(list(self.LONG_SRC), 48)
        assert core.ac_ordered_below(list(self.LONG_TGT), endpoint), \
            "the specimen certifies BECAUSE it descends, not because it is long"

    def test_equal_measure_respell_still_ties_and_dies(self, tmp_path) -> None:
        """(d) `2*sin(x0)*sin(x0) -> 2*sin(x0)^2` is the SAME canonical state respelled
        -- an exact tie. The retired token gate would have waved it through (7 -> 6
        tokens); the ordering is STRICT in both directions, so it dies 'rejected'.
        Nothing certifies on an equal-measure tie (F2/F5: a respell is not a
        reduction)."""
        eng = self._engine(str(tmp_path / "probe"))
        core = eng._core
        assert core is not None
        assert len(self.TIE_TGT) < len(self.TIE_SRC), "the retired gate would have passed this"
        _, _, endpoint = core.ac_judge(list(self.TIE_SRC), 48)
        assert not core.ac_ordered_below(list(self.TIE_TGT), endpoint), "tie: not below"
        assert not core.ac_ordered_below(endpoint, list(self.TIE_TGT)), "tie: not above either"
        assert eng.certify_rules([self.TIE_SRC], [self.TIE_TGT], dummy_variables=1,
                                 extra_internal_terms=["1", "2", "<constant>"],
                                 max_target_pattern_length=3, X=256, seed=7) == []
        _, sidecar, _ = self._mine(
            str(tmp_path / "mine"), [{"source": self.TIE_SRC, "target": self.TIE_TGT}])
        assert sidecar["proposals"]["trail"][-1]["verdict"] == "rejected"

    def test_the_mined_candidate_search_is_bounded_by_mu_not_length(self, tmp_path) -> None:
        """(e) THE BOUND IS mu (owner ruling, re-mine 2026-08-20). This test used to pin
        the opposite: a 6-token target was hidden from a 6-token source purely by the
        token bound, and raising `simplified_length` to 7 revealed it. Both calls now
        return the target, because the token count was never the criterion the mine
        actually cares about -- the half-angle pair is STRICTLY below the engine's own
        endpoint in the serve ordering, which is precisely what makes it mintable.

        What still binds is mu: a target that does NOT descend is refused at any length,
        pinned here with the (d) tie -- an exact respell of the same canonical state,
        which strictness kills."""
        eng = self._engine(str(tmp_path))
        core = eng._core
        assert core is not None
        X_data = RuleMiner(eng)._mining_sample_x(256, 1, np.random.default_rng(0))
        library = core.build_candidate_library(
            [["x0"], ["1"], ["2"], ["sin", "x0"], ["cos", "x0"], list(self.SIN_TGT)],
            ["x0"], X_data.flatten(order="C").tolist(), X_data.shape[0])
        _, _, endpoint = core.ac_judge(list(self.SIN_SRC), 48)
        scan = dict(challenges=16, retries=16, seed=1, rtol=1e-9, atol=1e-12,
                    min_informative=32, mark=endpoint)
        # Equal token count, strictly mu-below the mark: the mined scan now reaches it,
        # and the source length no longer changes the answer.
        assert core.find_rule_lib(list(self.SIN_SRC), len(self.SIN_SRC), None,
                                  library, **scan) == self.SIN_TGT, \
            "a mu-descending target must be reachable at equal token count"
        assert core.find_rule_lib(list(self.SIN_SRC), len(self.SIN_SRC) + 1, None,
                                  library, **scan) == self.SIN_TGT, \
            "and the source-length parameter must not change it"
        # THE BOUND THAT STILL BINDS: the (d) tie is a respell of one canonical state,
        # so it is not ordered strictly below the mark and no length admits it.
        tie_lib = core.build_candidate_library(
            [["x0"], ["1"], ["2"], ["sin", "x0"], list(self.TIE_TGT)],
            ["x0"], X_data.flatten(order="C").tolist(), X_data.shape[0])
        _, _, tie_end = core.ac_judge(list(self.TIE_SRC), 48)
        tie_scan = dict(scan, mark=tie_end)
        assert core.find_rule_lib(list(self.TIE_SRC), len(self.TIE_SRC), None,
                                  tie_lib, **tie_scan) is None, \
            "an exact mu tie is not a descent and must never mint, at any length"

    def test_a_mine_without_proposals_only_mints_descending_targets(self, engine) -> None:
        """(e, end to end) The invariant at mine scale, restated in the ordering that
        actually governs it. Every rule a proposal-free mine mints must have its target
        STRICTLY BELOW its source in the serve ordering.

        This used to assert `len(rhs) < len(lhs)`, and under the mu bound that is simply
        false -- `inv exp ?0 -> exp neg ?0` is 3 tokens against 3 and strictly cheaper,
        so the token form was refusing sound, descending rules for their spelling. The
        ordering is the invariant that was always meant; the token count was a proxy that
        happened to hold while the search was length-bounded."""
        engine.find_rules(max_source_pattern_length=3, max_target_pattern_length=3,
                          dummy_variables=1, extra_internal_terms=["0", "1", "<constant>"],
                          X=256, seed=7, verbose=False, promote_sorts=False)
        assert engine.simplification_rules, "the control mine produced nothing to check"
        offenders = [(lhs, rhs) for lhs, rhs in engine.simplification_rules
                     if not engine._core.ac_ordered_below(list(rhs), list(lhs))]
        assert not offenders, offenders


# `sinh` and the binary `-` are not in `_OPERATORS`; the certified-then-dropped family
# needs both to express the hyperbolic Pythagorean identity that triggers it.
_HYPERBOLIC_OPERATORS = {
    **_OPERATORS,
    "sinh": {"realization": "np.sinh", "alias": [], "inverse": "asinh",
             "arity": 1, "precedence": 3, "commutative": False},
    "-": {"realization": "-", "alias": [], "inverse": "+",
          "arity": 2, "precedence": 1, "commutative": False},
}

# cosh(x)^2 - sinh(x)^2 = 1 is an exact identity over the reals and certifies on any
# finite sample, but in float64 the two squares overflow to the SAME finite value long
# before they overflow to inf (|x| >~ 12.7 already disagrees; |x| > 355 gives inf - inf
# = nan), so the symbolic gate kills it for positive-measure disagreement. It is the
# 2026-08-18 acj-4-3-llm-v2 finding in miniature: certified by the proposal channel,
# then dropped by a LATER stage that the census never heard about.
_KILLED_PAIR = (["-", "pow2", "cosh", "x0", "pow2", "sinh", "x0"], ["1"])


class TestCertifiedThenDropped:
    """F94: the proposal census must never advertise a rule the artifact does not contain.

    `_certify_proposals` writes its verdicts BEFORE the symbolic gate, the covered-prune
    and the sort promotion have run, and each of those stages may still drop a certified
    pair -- soundly. The census froze at certification time and was never reconciled
    against the finished ruleset, so a pair could read `certified stage=accepted` in the
    sidecar while being absent from rules.json and dead at serve time (measured on the
    PUBLISHED acj-4-3-llm artifact: 19 of the 73 'certified' pairs absent).

    The fix is a truthful terminal outcome, not a rescued rule: every one of these drops
    is a soundness verdict that must stand.
    """

    @staticmethod
    def _mine(directory, proposals, operators=None, **kwargs):
        """One small mine with a proposal batch and the post-mine stages selectable."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "rules.json"), "w") as fh:
            fh.write("[]")
        cfg = os.path.join(directory, "config.yaml")
        with open(cfg, "w") as fh:
            fh.write(yaml.safe_dump({"engine_generation": 2,
                                     "operators": operators or _HYPERBOLIC_OPERATORS,
                                     "rules": "rules.json"}))
        eng = SimpliPyEngine.from_config(cfg)
        out = os.path.join(directory, "mined.json")
        eng.find_rules(max_source_pattern_length=3, max_target_pattern_length=3,
                       dummy_variables=1, extra_internal_terms=["0", "1", "<constant>"],
                       X=256, seed=7, verbose=False, output_file=out, proposals=proposals,
                       **{"promote_sorts": False, **kwargs})
        with open(out + ".provenance.json") as fh:
            sidecar = json.load(fh)
        # THE ARTIFACT IS A TRIPLE. `mined.json` is the f64 set alone, so a rule routed
        # to `real` (true on R, not f64-realised) is absent from it while being perfectly
        # present in the mine's output. For census reconciliation -- "did this certified
        # pair survive, or must it name its drop?" -- the honest denominator is the
        # CORPUS set, which is exactly f64 UNION real.
        with open(out.replace('.json', '_corpus.json')) as fh:
            artifact = [(tuple(lhs), tuple(rhs)) for lhs, rhs in json.load(fh)]
        return eng, sidecar, artifact

    @staticmethod
    def _key(lhs, rhs):
        """A rule identity that survives the sort respelling the promotion applies."""
        def norm(tokens):
            return tuple(("@" + t[1:]) if len(t) > 1 and t[0] in "_!?$" and t[1:].isdigit()
                         else ("@" + t[1:]) if len(t) > 1 and t[0] == "x" and t[1:].isdigit()
                         else t for t in tokens)
        return norm(lhs), norm(rhs)

    def test_rule_identity_survives_the_variable_canonicalisation_and_the_sort_respell(self) -> None:
        """The watch key must be invariant under BOTH spellings a certified pair changes
        between the census and the artifact, or the reconciliation reports a promoted
        rule as dropped. `_certify_proposals` records concrete variables (`x0`),
        `deduplicate_rules` canonicalises them to `?0`, and the promotion respells the
        sigil to `_0`/`!0`/`$0` in place. All four must key alike, while a genuinely
        different rule must not collide."""
        from simplipy.mining import _rule_identity
        dummies = ["x0", "x1"]
        base = _rule_identity(["log", "*", "inv", "x0", "inv", "x1"],
                              ["neg", "log", "*", "x0", "x1"], dummies)
        for sigil in "?_!$":
            respelt = _rule_identity(
                ["log", "*", "inv", f"{sigil}0", "inv", f"{sigil}1"],
                ["neg", "log", "*", f"{sigil}0", f"{sigil}1"], dummies)
            assert respelt == base, f"the `{sigil}` sort must key like the mined spelling"
        # variable RENAMING is canonicalised too (x1,x0 is the same rule up to naming)
        assert _rule_identity(["log", "*", "inv", "x1", "inv", "x0"],
                              ["neg", "log", "*", "x1", "x0"], dummies) == base
        # ... but slot POSITION is not: a different rule keeps a different key
        assert _rule_identity(["log", "*", "inv", "x0", "inv", "x1"],
                              ["neg", "log", "*", "x0", "x0"], dummies) != base

    def test_a_convicted_pair_with_a_tier_is_ROUTED_not_dropped(self, tmp_path) -> None:
        """THE 2026-08-18 DEFECT, re-pointed by the triple. `cosh(x0)^2 - sinh(x0)^2 = 1`
        certifies and the symbolic gate then convicts it -- and when `rules.json` was the
        only destination, the right thing to do was delete it and say so in the census.

        It is not the right thing any more. The identity is exactly true on the extended
        line and fails only because f64 cosh/sinh overflow at 710.475860, which is the
        definition of the `real` tier. So the gate keeps it, `_route_triple` puts it in
        rules_real and rules_corpus and NOT in the f64 set, and the census reads
        `certified` because it did survive -- with `routed_to_tier` naming where.

        The census property this test has always existed to guard is unchanged: a pair
        the census calls `certified` must be findable in the artifact. What moved is that
        "the artifact" is the triple, so the honest place to look is the corpus set."""
        source, target = _KILLED_PAIR
        _, sidecar, artifact = self._mine(str(tmp_path), [{"source": source, "target": target}])

        key = self._key(source, target)
        assert key in {self._key(lhs, rhs) for lhs, rhs in artifact}, \
            "a rule that is true on R must survive the mine somewhere"

        gate = sidecar["symbolic_gate"]
        want_lhs = ["-", "pow2", "cosh", "?0", "pow2", "sinh", "?0"]
        routed = gate.get("routed_to_tier", {})
        assert any(lhs == want_lhs for pairs in routed.values() for lhs, _ in pairs), \
            f"the gate must name the pair as routed, not drop it: {routed}"
        # and it must NOT appear in any drop bucket
        buckets = {b: v for b, v in gate.items()
                   if isinstance(v, list) and all(isinstance(x, list) and len(x) == 2 for x in v)}
        assert not [b for b, pairs in buckets.items() if any(lhs == want_lhs for lhs, _ in pairs)], \
            f"a pair with a tier must not be convicted into a drop bucket: {gate}"

        entry, = sidecar["proposals"]["trail"]
        assert entry["verdict"] == "certified"
        assert entry["stage"] == "accepted" and entry["certificate"] == "minimal"
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 1, "already_covered": 0, "rejected": 0, "duplicate": 0,
            "certified_then_dropped": 0}
        # it is REAL-tier, so the default f64 set must not carry it
        f64 = [(tuple(lhs), tuple(rhs))
               for lhs, rhs in json.load(open(str(tmp_path / "mined.json")))]
        assert key not in {self._key(list(lhs), list(rhs)) for lhs, rhs in f64}

    def test_the_outcome_key_is_present_even_when_nothing_is_dropped(self, tmp_path) -> None:
        """The census column exists whether or not it fired: a key that appears only on
        failure cannot be audited (a reader cannot tell 'none dropped' from 'this mine
        predates the column')."""
        _, sidecar, _ = self._mine(str(tmp_path), [{"source": ["+", "x0", "0"]}])
        assert sidecar["proposals"]["outcomes"] == {
            "certified": 0, "already_covered": 1, "rejected": 0, "duplicate": 0,
            "certified_then_dropped": 0}

    def test_every_certified_pair_is_in_the_artifact_or_names_its_drop(self, tmp_path) -> None:
        """THE INVARIANT, with the symbolic gate AND the covered-prune live. For each
        proposal the census calls `certified`, the pair must be IN the written artifact;
        for each it calls `certified_then_dropped`, the pair must be ABSENT and carry a
        stage and a reason. Nothing may fall between the two. (Spelled with `*` rather
        than `pow2`, and prune-only: the promotion's f64 evaluator has no realization
        for this fixture's `pow2`.)"""
        proposals = [
            # the same overflow identity as above, spelled multiplicatively: gate KILL
            {"source": ["-", "*", "cosh", "x0", "cosh", "x0", "*", "sinh", "x0", "sinh", "x0"],
             "target": ["1"]},
            {"source": ["+", "x0", "0"]},                          # already covered
            {"source": ["asin", "x0"]},                            # nothing shorter: rejected
            {"source": ["*", "exp", "x0", "exp", "x0"],
             "target": ["exp", "+", "x0", "x0"]},                  # certifies and survives
        ]
        operators = {k: v for k, v in _HYPERBOLIC_OPERATORS.items() if k != "pow2"}
        _, sidecar, artifact = self._mine(
            str(tmp_path), proposals, operators=operators, prune="covered")
        present = {self._key(lhs, rhs) for lhs, rhs in artifact}
        trail = sidecar["proposals"]["trail"]

        n_cert = n_dropped = 0
        for entry in trail:
            key = self._key(entry["source"], entry["target"] or [])
            if entry["verdict"] == "certified":
                n_cert += 1
                assert key in present, (
                    f"census says certified but the artifact lacks {entry['source']} -> "
                    f"{entry['target']}: this is the provenance defect itself")
            elif entry["verdict"] == "certified_then_dropped":
                n_dropped += 1
                assert key not in present, "a dropped pair must not be in the artifact"
                assert entry["dropped_at"], "a drop with no stage is an unaudited drop"
                assert entry["drop_reason"], "a drop with no reason is an unaudited drop"
        # NO `certified_then_dropped` case remains in this fixture, and that is the
        # triple working rather than the test decaying: the overflow identity that used
        # to produce one is `real`-tier and now routes into rules_real/rules_corpus. The
        # outcome needs a pair the MINER certifies numerically that the JUDGE then puts
        # in `reject` (false on R AND not f64-realised), which routing has made genuinely
        # rare -- `tanh x -> x` and `exp(-cosh x) -> 0` were both tried and are refused
        # at proposal certification, never reaching the gate. The per-entry assertions
        # above still guard the branch for when it does occur; this is a coverage gap
        # worth closing with a purpose-built fixture, not a reason to contrive one here.
        assert n_dropped == 0
        assert n_cert == 2, "both certified proposals must still read as certified"
        assert sidecar["proposals"]["outcomes"]["certified"] == n_cert
        assert sidecar["proposals"]["outcomes"]["certified_then_dropped"] == n_dropped
        assert sum(sidecar["proposals"]["outcomes"].values()) == len(proposals)


class TestTripleRouting:
    """The artifact is a TRIPLE (owner ruling 2026-08-19): one distinct, complete rule
    set per mode, mined and distributed together, with the rejects RECORDED rather than
    silently absent. A mine run is valid only if all three fall out of it.
    """

    A = [['x0'], ['x0']]
    B = [['y0'], ['y0']]
    C = [['z0'], ['z0']]

    def _allowed(self):
        from simplipy.mining import _TIER_MODES
        return {0: _TIER_MODES['core'], 1: _TIER_MODES['f64'], 2: _TIER_MODES['real']}

    def test_a_correct_triple_passes(self) -> None:
        from simplipy.mining import _assert_triple_invariants
        _assert_triple_invariants(
            {'f64': [self.A, self.B], 'real': [self.A, self.C],
             'corpus': [self.A, self.B, self.C]},
            [], [self.A, self.B, self.C], self._allowed())

    @pytest.mark.parametrize('label,triple,rejected', [
        # served AND rejected -- the two must stay disjoint
        ('inv3-both', {'f64': [A, B], 'real': [A, C], 'corpus': [A, B, C]}, [A]),
        # a rule in a file its tier does not license (invariant 4, the direction the
        # per-mode prune does NOT excuse)
        ('inv4', {'f64': [A, B, C], 'real': [A, C], 'corpus': [A, B, C]}, []),
    ])
    def test_each_invariant_can_actually_fail(self, label, triple, rejected) -> None:
        """An invariant that restates its own construction proves nothing. Two of these
        were tautologies on the first pass and passed every corruption."""
        from simplipy.mining import _assert_triple_invariants
        with pytest.raises(AssertionError, match='triple invariant'):
            _assert_triple_invariants(triple, rejected, [self.A, self.B, self.C],
                                      self._allowed())

    def test_the_per_mode_licence_check_can_fail(self) -> None:
        """The invariant that replaced the two retired ones. It was gated on a
        `preempted` argument only the production path supplies, so every direct call
        skipped it and nothing exercised it at all."""
        from simplipy.mining import _TIER_MODES, _assert_triple_invariants
        allowed = {0: _TIER_MODES['core'], 1: _TIER_MODES['real']}
        # B is real-tier; putting it in rules_f64 is the direction the prune never excuses
        with pytest.raises(AssertionError, match='not licensed for that mode'):
            _assert_triple_invariants(
                {'f64': [self.A, self.B], 'real': [self.A, self.B],
                 'corpus': [self.A, self.B]},
                [], [self.A, self.B], allowed)

    def test_invariant_4_catches_what_the_other_three_cannot(self) -> None:
        """`B` is f64-tier but placed only in `real`. The intersection is still exactly
        core, the union is still exactly corpus, nothing is rejected -- invariants 1-3
        all hold. Only the per-rule licence check sees it."""
        from simplipy.mining import _assert_triple_invariants
        # Either message is a pass: the per-mode licence check and invariant 4 overlap on
        # this corruption (a rule in a file its tier does not license), and the licence
        # check now runs first. What must not happen is silence.
        with pytest.raises(AssertionError, match='invariant 4|not licensed for that mode'):
            _assert_triple_invariants(
                {'f64': [self.A], 'real': [self.A, self.B, self.C],
                 'corpus': [self.A, self.B, self.C]},
                [], [self.A, self.B, self.C], self._allowed())

    def test_corpus_is_derived_so_it_cannot_disagree_with_the_union(self) -> None:
        """`_TIER_MODES` deliberately has no `corpus` entry: a rule serves there iff it
        serves in either other mode, so invariant 2 holds by construction. A fourth entry
        would make it possible to break."""
        from simplipy.mining import _TIER_MODES
        assert 'corpus' not in _TIER_MODES
        assert _TIER_MODES['core'] == frozenset({'f64', 'real'})
        assert _TIER_MODES['reject'] == frozenset()

    def test_routing_the_shipped_artifact_satisfies_every_invariant(self) -> None:
        """The end-to-end smoke test: route a real engine's real rule set. Counts are not
        pinned -- they are a property of the mine, and this artifact predates the triple
        -- but the RELATIONSHIPS must hold, and `_route_triple` raises if they do not."""
        from conftest import acj_config_path, require_or_skip
        from simplipy.mining import _route_triple
        require_or_skip(acj_config_path(), 'the triple routes the shipped rule set')
        engine = SimpliPyEngine.from_config(acj_config_path())
        triple, rejected, meta = _route_triple(engine)
        served = len(engine.simplification_rules)
        # EXACT, not `<=`. The counts were left unpinned on the stated ground that "this
        # artifact predates the triple" -- which stopped being true when the asset became
        # one, and a `<=` bound passes any under-population. The identity that holds under
        # the per-mode prune is: what a file omits, that mode's own constructor performs.
        assert len(triple['corpus']) + len(rejected) + meta['preempted']['corpus'] == served
        assert len(triple['f64']) + meta['preempted']['f64'] <= served
        assert len(triple['real']) + meta['preempted']['real'] <= served
        assert 0 < len(triple['f64']) and 0 < len(triple['real'])


class TestTheGateRoutesInsteadOfDeleting:
    """Every fatal bucket used to drop, which was right when `rules.json` was the only
    destination: a rule the contract convicts had nowhere to live. Under the triple two
    of those buckets DO have somewhere -- ENGINE-MISALIGN is `real`, an f64-realised KILL
    is `f64` -- and deleting them in the gate leaves `_route_triple` nothing to route,
    making rules_real.json and rules.json identical and voiding the split silently.

    This is the hazard the roadmap flagged as "arming the deployed check without the tier
    makes the next mine DELETE sound rules, including the atanh(tanh x) beautification".
    It is closed in the GATE, and only a mine can show that -- routing a loaded engine
    exercises none of it, which is how it survived the first pass.
    """

    def test_a_mine_writes_a_triple_whose_real_set_keeps_the_inverse_pairs(
            self, tmp_path) -> None:
        import yaml
        from conftest import acj_config_path, require_or_skip
        require_or_skip(acj_config_path(), 'needs the acj operator vocabulary')
        ops = yaml.safe_load(open(acj_config_path()))['operators']
        (tmp_path / 'rules.json').write_text(json.dumps([]))
        (tmp_path / 'config.yaml').write_text(
            yaml.safe_dump({'operators': ops, 'rules': 'rules.json'}))
        engine = SimpliPyEngine.from_config(str(tmp_path / 'config.yaml'))
        out = str(tmp_path / 'mined.json')
        engine.find_rules(max_source_pattern_length=3, max_target_pattern_length=3,
                          dummy_variables=1, extra_internal_terms=['0', '1', '(-1)'],
                          X=128, seed=7, promote_sorts=False, output_file=out)

        f64 = json.load(open(out))
        real = json.load(open(str(tmp_path / 'mined_real.json')))
        corpus = json.load(open(str(tmp_path / 'mined_corpus.json')))

        def keys(rs):
            return {(tuple(lhs), tuple(rhs)) for lhs, rhs in rs}

        # `corpus == f64 UNION real` is RETIRED and its replacement is BEHAVIOURAL
        # (`assert_corpus_dominates`), so there is no set identity to assert here. An
        # intermediate version asserted `keys(f64) <= mined` where `mined` was DEFINED as
        # the union of the three files -- unconditionally true. What the files must show
        # is the routing itself, checked below.

        # The inverse-pair family is true on R and not f64-realised (tanh attains 1.0
        # from 18.990341103219276, exp overflows at 709.782713), so it is `real`: present
        # in the real and corpus sets, absent from f64. Before the gate fix all four were
        # simply deleted.
        pairs = [(['atanh', 'tanh', '?0'], ['?0']), (['log', 'exp', '?0'], ['?0']),
                 (['asinh', 'sinh', '?0'], ['?0']), (['acosh', 'cosh', '?0'], ['abs', '?0'])]
        # The point of the routing: they are KEPT, in the two modes whose authority
        # licenses them, and absent from the one that does not. (An earlier version of
        # this test asserted only the absence, on the strength of a comment claiming they
        # were gone from all three files -- which was false.)
        kept = [p for p in pairs if (tuple(p[0]), tuple(p[1])) in keys(real)]
        assert kept, f'no inverse pair survived into rules_real: {sorted(keys(real))[:4]}'
        for lhs, rhs in kept:
            assert (tuple(lhs), tuple(rhs)) in keys(corpus), lhs
            assert (tuple(lhs), tuple(rhs)) not in keys(f64), \
                f'{" ".join(lhs)} is NOT f64-realised and must not be in the f64 set'

        # the carve-out is visible in the sidecar, never silent
        prov = json.load(open(out + '.provenance.json'))
        assert prov['triple']['counts'] == {'f64': len(f64), 'real': len(real),
                                            'corpus': len(corpus)}
        assert prov['symbolic_gate']['routed_to_tier'].get('real')

    def test_the_unjudgeable_buckets_still_fail_closed(self) -> None:
        """Relaxing the gate must not relax it for verdicts that carry no tier at all.
        NO-WITNESS, UNSUPPORTED-SHAPE, JUDGE-TIMEOUT and UNRESOLVED-COVERAGE mean the
        second authority never reached an answer, and "confirmed by two independent
        authorities" is the claim a shipped rule makes."""
        from simplipy.verify import FATAL_BUCKETS
        from simplipy.verify._contract import _tier
        for bucket in FATAL_BUCKETS:
            if bucket in ('ENGINE-MISALIGN', 'KILL'):
                continue
            for realised in (True, False, None):
                assert _tier(bucket, realised) == 'reject', bucket


class TestCertifyRulesRoutesLikeTheMine:
    """`certify_rules` promises its survivors are "exactly as sound as mined rules". That
    parity is a claim about BOTH directions, so when `_finalize` stopped deleting what a
    tier owns, this gate had to stop too -- otherwise a user-supplied pair that a mine
    would route into rules_real.json is destroyed here instead, and the parity sentence
    is false."""

    @staticmethod
    def _bare(tmp_path):
        import yaml
        from conftest import acj_config_path
        ops = yaml.safe_load(open(acj_config_path()))['operators']
        (tmp_path / 'rules.json').write_text('[]')
        (tmp_path / 'config.yaml').write_text(
            yaml.safe_dump({'engine_generation': 2, 'operators': ops, 'rules': 'rules.json'}))
        return SimpliPyEngine.from_config(str(tmp_path / 'config.yaml'))

    def test_a_real_tier_proposal_survives_instead_of_being_deleted(self, tmp_path) -> None:
        """`atanh(tanh x)` and `log(exp x)` are true on R and not f64-realised (tanh
        attains 1.0 from 18.990341103219276, exp overflows at 709.782713), so both are
        ENGINE-MISALIGN, tier `real`. Before the fix the gate deleted both."""
        from conftest import acj_config_path, require_or_skip
        require_or_skip(acj_config_path(), 'needs the acj operator vocabulary')
        engine = self._bare(tmp_path)
        out = engine.certify_rules([['atanh', 'tanh', 'x0'], ['log', 'exp', 'x0']],
                                   dummy_variables=1, X=256, verbose=False)
        survived = {(tuple(r[0]), tuple(r[1])) for r in out}
        assert (('atanh', 'tanh', 'x0'), ('x0',)) in survived
        assert (('log', 'exp', 'x0'), ('x0',)) in survived

    def test_a_reject_tier_proposal_is_still_deleted(self, tmp_path) -> None:
        """The relaxation is exactly one step wide. `sin t -> t` is false on R AND not
        reproduced by f64, so its tier is `reject` and there is nowhere to route it."""
        from conftest import acj_config_path, require_or_skip
        require_or_skip(acj_config_path(), 'needs the acj operator vocabulary')
        engine = self._bare(tmp_path)
        out = engine.certify_rules([['sin', 'x0']], targets=[['x0']],
                                   dummy_variables=1, X=256, verbose=False)
        assert not [r for r in out if tuple(r[1]) == ('x0',)]


class TestTheUntestedBranchesOfTheRouter:
    """Three properties the audit found asserted nowhere. Each is reachable with a small
    fixture, and one of them -- the fail-closed twin intersection -- passed the entire
    suite when flipped to a fail-OPEN union."""

    def test_the_reject_arm_records_rather_than_silently_drops(self, tmp_path) -> None:
        """The owner's ruling has two halves and only one was covered: rules are routed,
        AND the rejects are RECORDED. `prov['triple']['rejected']` was written by no test,
        so a change that stopped emitting it would have gone unnoticed."""
        import yaml
        from conftest import acj_config_path, require_or_skip
        require_or_skip(acj_config_path(), 'needs the acj operator vocabulary')
        ops = yaml.safe_load(open(acj_config_path()))['operators']
        (tmp_path / 'rules.json').write_text('[]')
        (tmp_path / 'config.yaml').write_text(
            yaml.safe_dump({'engine_generation': 2, 'operators': ops, 'rules': 'rules.json'}))
        engine = SimpliPyEngine.from_config(str(tmp_path / 'config.yaml'))
        out = str(tmp_path / 'mined.json')
        engine.find_rules(max_source_pattern_length=3, max_target_pattern_length=3,
                          dummy_variables=1, extra_internal_terms=['0', '1', '(-1)'],
                          X=128, seed=7, promote_sorts=False, output_file=out)
        prov = json.load(open(out + '.provenance.json'))
        assert 'triple' in prov
        assert set(prov['triple']['counts']) == {'f64', 'real', 'corpus'}
        # present even when empty: a key that appears only on failure cannot be audited
        assert 'rejected' in prov['triple']
        assert isinstance(prov['triple']['rejected'], list)
        assert set(prov['triple']['files']) == {'f64', 'real', 'corpus'}

    def test_a_twins_licence_is_the_INTERSECTION_not_the_union(self) -> None:
        """A source rule minting several served entries may serve a mode only if EVERY
        way it can fire is licensed there. Flipping this to a union passed the whole
        mining/mode/verify suite.

        Asserted through `_route_triple` itself. An earlier version hand-rolled `&` on
        `_TIER_MODES` inside the test body, which restates set algebra and says nothing
        about the router's combination operator -- the very thing that could be flipped.
        """
        import simplipy.mining as M

        # ONE source rule (index 0) minting TWO served entries whose tiers differ: `core`
        # through one twin and `f64` through the other. Intersection => f64 only. Union
        # would put it in rules_real as well.
        served = [(['sin', '_0'], ['_0'], 0), (['cos', '_0'], ['_0'], 0)]
        detail = [{'idx': 0, 'tier': 'core', 'lhs': 'sin _0', 'rhs': '_0'},
                  {'idx': 1, 'tier': 'f64', 'lhs': 'cos _0', 'rhs': '_0'}]

        class FakeCore:
            def ac_served_rules(self):
                return served

        class FakeEngine:
            simplification_rules = [(['sin', '_0'], ['_0'])]
            _core = FakeCore()
            _operators_config: dict = {}

        real_verify = M.verify_ruleset if hasattr(M, 'verify_ruleset') else None
        import simplipy.verify as V
        saved = V.verify_ruleset
        V.verify_ruleset = lambda rules, **kw: {'detail': detail}
        try:
            triple, rejected, meta = M._route_triple(FakeEngine())
        finally:
            V.verify_ruleset = saved
            assert real_verify is None or True

        key = (('sin', '_0'), ('_0',))
        got = {m for m in ('f64', 'real') if key in {(tuple(a), tuple(b)) for a, b in triple[m]}}
        assert got == {'f64'}, got            # union would give {'f64', 'real'}
        assert rejected == []

    def test_a_rule_with_no_served_entry_is_licensed_nowhere(self) -> None:
        """The load-minted-twins lesson in the other direction: never infer a licence for
        something the matcher never saw. `_route_triple` reads `allowed.get(i, frozenset())`,
        and the default is the empty set, not `core`."""
        from simplipy.mining import _assert_triple_invariants, _TIER_MODES
        A, B = [['x0'], ['x0']], [['y0'], ['y0']]
        # B is licensed nowhere, so it must be REJECTED, not quietly served
        _assert_triple_invariants({'f64': [A], 'real': [A], 'corpus': [A]},
                                  [B], [A, B], {0: _TIER_MODES['core'], 1: frozenset()})
        with pytest.raises(AssertionError, match='triple invariant'):
            _assert_triple_invariants({'f64': [A, B], 'real': [A], 'corpus': [A, B]},
                                      [], [A, B], {0: _TIER_MODES['core'], 1: frozenset()})


class TestThePerModePrune:
    """Each mode's file carries only what that mode can serve. Folding is mode-dependent,
    so a rule can be load-bearing in `real` and dead weight in `f64`, where the
    constructor reaches the same endpoint without it."""

    def test_a_rule_its_own_constructor_performs_is_pruned(self) -> None:
        """The 18-rule `f(+-10^-k)` family exists only because the constructor was once
        forbidden to fold. In f64 it now folds, so those rules can never fire from
        rules.json -- shipping them would be the computed-rather-than-stated shape the
        load-minted twins already punished."""
        from conftest import acj_config_path, require_or_skip
        from simplipy.mining import _preempted_by
        require_or_skip(acj_config_path(), 'needs the acj vocabulary')
        engine = SimpliPyEngine.from_config(acj_config_path())
        assert _preempted_by(engine, ['asin', '1e-08'], ['1e-08'], 'f64') is True
        # ... and NOT in real, which does not fold, so there the rule is load-bearing
        assert _preempted_by(engine, ['asin', '1e-08'], ['1e-08'], 'real') is False

    def test_the_comparison_is_on_values_not_token_spellings(self) -> None:
        """`1e-08` and `0.00000001` are the same number. An earlier version compared
        tokens and reported 17 preempted where the truth was 29."""
        from conftest import acj_config_path, require_or_skip
        from simplipy.mining import _preempted_by
        require_or_skip(acj_config_path(), 'needs the acj vocabulary')
        engine = SimpliPyEngine.from_config(acj_config_path())
        # BOTH directions. Comparing the rule's own spellings (`1e-08` -> `0.00000001`)
        # is the one specimen a spelling-comparing implementation gets RIGHT, because the
        # engine normalises both sides to the same token before the comparison. The
        # discriminating case is a rule whose SIDES are spelled differently from each
        # other: a spelling comparer answers False, a value comparer answers True.
        assert _preempted_by(engine, ['asin', '1e-08'], ['0.00000001'], 'f64') is True
        assert _preempted_by(engine, ['asin', '0.00000001'], ['1e-08'], 'f64') is True
        # ... and the engine really does normalise them to one token, which is why a
        # token comparison on the RAW sides would have differed
        assert ['1e-08'] != ['0.00000001']
        assert engine.simplify(['1e-08']) == engine.simplify(['0.00000001'])


class TestCorpusDominanceReplacesTheSetInvariant:
    """`rules_corpus == rules_f64 UNION rules_real` was retired: once folding is
    mode-dependent the three modes live in different canonical worlds, so set overlap
    measures SPELLING, not capability. This is the property it stood proxy for."""

    def test_it_holds_on_the_shipped_corpus(self) -> None:
        from conftest import acj_config_path, require_or_skip
        from simplipy.mining import assert_corpus_dominates
        require_or_skip(acj_config_path(), 'needs the acj-4-3 asset')
        from conftest import require_triple_or_skip
        require_triple_or_skip('needs the shipped corpus set')
        # NOT wiped: the shipped artifact's own 5,471-rule real set is the thing corpus
        # has to dominate. Emptying it first made the check trivial.
        engine = SimpliPyEngine.from_config(acj_config_path())
        assert engine._core.mode_rules_len('real') is not None
        rows = json.load(open(os.path.join(
            os.path.dirname(acj_config_path()), '..', '..', '..',
            'benchmarks', 'corpus', 'raw_skeletons_nv.json')))[:120]
        assert assert_corpus_dominates(engine, rows) == []

    def test_it_CAN_fail(self) -> None:
        """An invariant that cannot fail proves nothing -- three tautologies shipped in
        this work before being caught. Cripple corpus to real's semantics and the check
        must report it."""
        from simplipy import Mode
        from conftest import acj_config_path, require_or_skip
        from simplipy.mining import assert_corpus_dominates
        require_or_skip(acj_config_path(), 'needs the acj-4-3 asset')
        engine = SimpliPyEngine.from_config(acj_config_path())
        engine._core.set_mode_rules('real', [])

        class CorpusCrippled:
            def __init__(self, e):
                self._e, self._core = e, e._core

            def simplify(self, expr, mode=None):
                return self._e.simplify(expr, mode=Mode.real if mode is Mode.corpus else mode)

        rows = json.load(open(os.path.join(
            os.path.dirname(acj_config_path()), '..', '..', '..',
            'benchmarks', 'corpus', 'raw_skeletons_nv.json')))[:120]
        assert assert_corpus_dominates(CorpusCrippled(engine), rows) != []


class TestThePruneIsInstanceSafe:
    """The prune asks a RULE-LESS twin whether that mode's constructor already reaches a
    rule's endpoint. Asked about a PATTERN, the answer is about the pattern -- not about
    the instances the rule matches -- and the two genuinely differ."""

    def test_a_slotted_rule_is_never_pruned(self) -> None:
        """THE DEFECT. In corpus the constructor takes `/ $0 $0` to `1` (the slot carries
        a nonzero-a.e. certificate) and the instance `inf / inf` to `nan`. Pruning on the
        pattern deleted `/ $0 $0 -> 1` and with it the mask-sentinel doctrine, so
        `inf/inf * x0` stopped collapsing to `x0` in corpus."""
        import yaml
        from conftest import acj_config_path, require_or_skip
        from simplipy import Mode
        from simplipy.mining import _preempted_by
        require_or_skip(acj_config_path(), 'needs the acj vocabulary')
        engine = SimpliPyEngine.from_config(acj_config_path())
        assert _preempted_by(engine, ['/', '$0', '$0'], ['1'], 'corpus') is False

        bare = SimpliPyEngine(operators=yaml.safe_load(open(acj_config_path()))['operators'],
                              rules=[])
        bare._core.set_mode_rules('real', [])
        # the two answers that must not be conflated
        assert bare.simplify(['/', '$0', '$0'], mode=Mode.corpus) == ['1']
        assert bare.simplify(['/', 'float("inf")', 'float("inf")'],
                             mode=Mode.corpus) == ['float("nan")']

    def test_a_ground_rule_still_prunes(self) -> None:
        """The bound costs nothing the prune was for: every literal evaluation folding
        now performs is ground, including the 18-rule `f(+-10^-k)` family."""
        from conftest import acj_config_path, require_or_skip
        from simplipy.mining import _is_ground, _preempted_by
        require_or_skip(acj_config_path(), 'needs the acj vocabulary')
        engine = SimpliPyEngine.from_config(acj_config_path())
        assert _preempted_by(engine, ['asin', '1e-08'], ['1e-08'], 'f64') is True
        assert _is_ground(['asin', '1e-08']) is True
        for slotted in (['/', '$0', '$0'], ['atanh', 'tanh', '_0'], ['sin', '?0'],
                        ['acos', '<constant>'], ['abs', '!0']):
            assert _is_ground(slotted) is False, slotted


class TestCanonicalSourceClassCertification:
    """CERTIFY THE CANONICAL REPRESENTATIVE ONLY (owner ruling, re-mine 2026-08-20).

    Certification runs in f64 on whichever spelling the enumerator produced, while AC
    treats all regroupings of `a*b*c` as ONE expression. f64 does not: `(1e200*1e200)/1e200`
    is inf and `1e200*(1e200/1e200)` is 1e200. So a class could certify or not depending on
    association -- the 13 rules lost to evaluation order. Mining one representative per
    class makes certification a function of the class.

    The soundness claim that makes skipping safe: `ac_canonical_key` IS the loader's key,
    so a skipped spelling is SERVED by the representative's rule rather than going
    unserved. That is pinned here, not assumed.
    """

    def test_duplicate_spellings_collapse_and_the_census_says_so(self, engine, tmp_path) -> None:
        out = str(tmp_path / "mined.json")
        prov_before = engine.find_rules(
            max_source_pattern_length=3, max_target_pattern_length=None,
            dummy_variables=1, extra_internal_terms=["0", "1", "<constant>"],
            X=256, seed=7, verbose=False, output_file=out, promote_sorts=False)
        del prov_before
        with open(out + ".provenance.json") as fh:
            prov = json.load(fh)
        census = prov.get("ac_classes")
        assert census, "the sidecar must record what the mine actually certified"
        # At least one length must actually collapse, or the dedup is untested here.
        assert any(v["classes"] < v["enumerated"] for v in census.values()), census
        for v in census.values():
            assert 0 < v["classes"] <= v["enumerated"]

    def test_a_skipped_spelling_is_still_served(self, engine) -> None:
        """The whole licence for skipping: same key => the loader builds the same pattern,
        so a spelling the mine never saw is served by the representative's rule.

        Checked on the ENUMERATED universe rather than on whatever the mine happened to
        mint -- the claim is about every class the dedup collapses, not about the subset
        that yielded rules."""
        engine.find_rules(max_source_pattern_length=3, max_target_pattern_length=None,
                          dummy_variables=1, extra_internal_terms=["0", "1", "<constant>"],
                          X=256, seed=7, verbose=False, promote_sorts=False)
        core = engine._core
        # Enumerate the same universe the mine walks, then group it by the loader's key.
        leaves = ["x0", "0", "1", "<constant>"]
        exprs: list[list[str]] = [[leaf] for leaf in leaves]
        for _ in range(2):
            grown: list[list[str]] = []
            for op, spec in _OPERATORS.items():
                if spec["arity"] == 1:
                    grown += [[op] + e for e in exprs]
                else:
                    grown += [[op] + a + b for a in exprs for b in exprs if len(a) + len(b) <= 2]
            exprs = [e for e in exprs + grown if len(e) <= 3]
        classes: dict[tuple, list[list[str]]] = {}
        for e, k in zip(exprs, core.ac_canonical_keys([list(e) for e in exprs])):
            if k is not None:
                classes.setdefault(tuple(k), []).append(e)
        multi = {k: v for k, v in classes.items() if len(v) > 1}
        assert multi, "no class has two spellings: the claim would go unexercised"
        for members in multi.values():
            endpoints = {tuple(engine.simplify(list(m))) for m in members}
            assert len(endpoints) == 1, (members, endpoints)


class TestVarFreeCandidatesMustFoldToALeaf:
    """The fold filter's replacement lemma, pinned (2026-08-21).

    The mu-dominance criterion that briefly replaced the length rule was UNSOUND: it
    admitted every var-free composite, because `<constant>` prices at 67000 while
    `asin sin 2` is 19000. A live mine under it produced 57 rules of the form
    `pi - 2 -> asin(sin(2))` in its first 212 -- true by value, an obfuscation as a
    rewrite. The criterion is now what the CONSTRUCTOR makes of the candidate.
    """

    def test_a_candidate_is_admitted_iff_it_folds_to_a_leaf(self, engine, mining_x) -> None:
        x_flat, n = mining_x
        # `exp 1` folds to `np.e`; `pow2 <c>` and `exp <c>` stay composite.
        cands = [["x0"], ["<constant>"], ["exp", "1"], ["exp", "<constant>"],
                 ["pow2", "<constant>"], ["*", "<constant>", "x0"]]
        lib = engine._core.build_candidate_library(cands, ["x0"], x_flat, n)
        assert lib.n_filtered == 2, "exp(<c>) and pow2(<c>) stay composite -> dropped"
        assert lib.n_candidates == 4, "exp(1) folds to np.e and must survive"

    def test_the_guard_reaches_the_class_that_escaped(self, engine, mining_x) -> None:
        """REGRESSION. The first version of this falsifier ran on a vocabulary where the
        defect could not occur, so it passed while the real mine minted
        `rootn 1 <constant> -> pow 1 <constant>`. The candidate was admitted because the
        CONSTRUCTOR folds `pow 1 <constant>` to `1`; the scan then emitted its own token
        form. This pins the emitted spelling, which is what ships."""
        x_flat, n = mining_x
        # `pow 1 <c>` folds to the leaf `1`, so the library admits it -- the guard has to
        # refuse the composite SPELLING at emission, not the candidate at admission.
        cands = [["x0"], ["<constant>"], ["1"], ["pow", "1", "<constant>"]]
        lib = engine._core.build_candidate_library(cands, ["x0"], x_flat, n)
        assert lib.n_candidates == 4, "the foldable composite must still be ADMITTED"
        src = ["rootn", "1", "<constant>"]
        _, _, mark = engine._core.ac_judge(list(src), 48)
        got = engine._core.find_rule_lib(list(src), len(src), None, lib, challenges=16,
                                         retries=16, seed=1, rtol=1e-9, atol=1e-12,
                                         min_informative=32, mark=mark)
        assert got is None or len(got) < 2 or any(
            t.startswith(("x", "_", "$", "?", "!")) for t in got), \
            f"emitted a var-free composite target: {got}"

    def test_the_mine_never_mints_a_composite_var_free_target(self, engine) -> None:
        """THE FALSIFIER. A mined RHS may be a leaf (`0`, `1`, `np.e`) but never a
        var-free COMPOSITE -- that is the class the broken filter let through."""
        engine.find_rules(max_source_pattern_length=3, max_target_pattern_length=None,
                          dummy_variables=1, extra_internal_terms=["0", "1", "<constant>"],
                          X=256, seed=7, verbose=False, promote_sorts=False)
        offenders = [(lhs, rhs) for lhs, rhs in engine.simplification_rules
                     if len(rhs) >= 2
                     and not any(t.startswith(("x", "_", "$", "?", "!")) for t in rhs)]
        assert not offenders, offenders
