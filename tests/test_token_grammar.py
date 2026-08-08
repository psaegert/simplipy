"""The token grammar boundary (H-007): reserved numeric spellings are REFUSED.

The engine has exactly three kinds of well-formed token:

* NUMERIC LITERALS -- the canonical spellings every layer reads identically:
  ``Rat::parse_decimal`` forms (optional single sign, decimal digits, one optional
  ``.``, optional ``e``/``E`` exponent with optional sign), the exact fraction
  ``p/q``, and the special spellings ``np.pi``/``np.e``/``float("inf")``/
  ``float("-inf")``/``float("nan")`` (plus their parenthesized forms);
* FREE SYMBOLS -- any other well-formed token that NO standard numeric reader
  interprets (``x0``, ``foo``, ``_0``); symbol algebra applies;
* RESERVED SPELLINGS -- tokens a standard numeric reader DOES interpret but the
  symbolic core does not: textual non-finites (``inf``/``infinity``/``nan``, any
  case, optional sign), underscore digit groupings (``1_000``), and base-prefixed
  integer literals (``0x10``/``0o17``/``0b101``). Treating these as symbols is
  UNSOUND under the numeric reading (``inf - inf -> 0`` where IEEE says nan), and
  before H-007 the engine did exactly that while its own numeric layer
  (``evaluate_constant_subtree``) read the SAME tokens as values -- one input, two
  contradictory public answers. They are refused at every semantic boundary.
"""


import pytest

from simplipy import SimpliPyEngine
from simplipy.masking import literal_sites
from simplipy.utils import is_numeric_string, reserved_numeric_spelling
from conftest import acj_config_path


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(
        acj_config_path())


# The three reserved families, with sign and case variants.
RESERVED = [
    "inf", "Inf", "INF", "infinity", "Infinity", "+inf", "-inf",
    "nan", "NaN", "NAN", "+nan", "-nan",
    "1_000", "1_000.5", "1e1_0", "1_0",
    "0x10", "0X10", "0o17", "0b101", "+0x10", "-0x10",
]

# Canonical numeric literals: (input token, simplified spelling).
LEGAL_NUMERIC = [
    ("5", ["5"]),
    ("+5", ["5"]),
    # `.5` PARSES (the grammar guard); it then takes its argmin spelling, which for a
    # 2^a denominator is the fraction -- §10.10(1), owner-ratified 2026-08-07.
    (".5", ["<mul>", "1", "<div>", "2", "</mul>"]),
    ("5.", ["5"]),
    ("1e-05", ["0.00001"]),
    ("1e+5", ["100000"]),
    ("1E5", ["100000"]),
    ("1e+16", ["10000000000000000"]),
]

# Free symbols: no numeric reader interprets these; symbol algebra applies.
LEGAL_SYMBOLS = ["x0", "foo", "x_0", "inf2", "nanx", "in"]

# Sigil-prefixed tokens are the PATTERN language (rule placeholders): legal input,
# but the certificate layer refuses to cancel them (no finite-a.e. certificate for a
# sigil leaf) -- identity round-trip only.
LEGAL_SIGILS = ["_0", "_"]


class TestReservedSpellingsRefused:
    @pytest.mark.parametrize("tok", RESERVED)
    def test_simplify_refuses(self, engine: SimpliPyEngine, tok: str) -> None:
        for expr in ([tok], ["-", tok, tok]):
            with pytest.raises(ValueError, match="reserved numeric spelling"):
                engine.simplify(expr)

    def test_error_names_token_and_canonical_spelling(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match=r"inf") as exc:
            engine.simplify(["inf"])
        assert 'float("inf")' in str(exc.value)

    def test_evaluate_constant_subtree_refuses(self, engine: SimpliPyEngine) -> None:
        # Before H-007 this returned float("nan") while simplify returned ["0"] for the
        # same tokens -- the contradiction this boundary exists to kill.
        with pytest.raises(ValueError, match="reserved numeric spelling"):
            engine._core.evaluate_constant_subtree(["-", "inf", "inf"])

    def test_judge_and_complexity_refuse(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match="reserved numeric spelling"):
            engine._core.ac_judge(["inf"], 48)
        with pytest.raises(ValueError, match="reserved numeric spelling"):
            engine._core.ac_complexity(["nan"])

    def test_parse_normalizes_lowercase_nonfinites(self, engine: SimpliPyEngine) -> None:
        # The infix tokenizer maps lowercase `inf`/`nan` to the CANONICAL spellings:
        # the infix surface reads them numerically (unambiguous there), so nothing
        # reserved survives into the token stream.
        assert engine.parse("inf - inf") == ["-", 'float("inf")', 'float("inf")']
        assert engine.parse("nan + x0", mask_numbers=True) == ["+", 'float("nan")', "x0"]

    def test_parse_refuses_unnormalized_nonfinites(self, engine: SimpliPyEngine) -> None:
        # Case variants the tokenizer does NOT normalize would enter the stream as
        # name tokens; the boundary refuses them BEFORE numbers_to_constant (whose
        # float() semantics WOULD read them) can absorb one into `<constant>`.
        for infix in ("Infinity - 1", "NaN * x0", "INF + 1"):
            with pytest.raises(ValueError, match="reserved numeric spelling"):
                engine.parse(infix)
            with pytest.raises(ValueError, match="reserved numeric spelling"):
                engine.parse(infix, mask_numbers=True)

    def test_prefix_to_infix_refuses(self, engine: SimpliPyEngine) -> None:
        # The serializer feeds eval-bearing surfaces; a reserved token must not
        # escape into a rendering where Python would read it numerically.
        with pytest.raises(ValueError, match="reserved numeric spelling"):
            engine.prefix_to_infix(["-", "inf", "inf"])

    def test_masking_refuses(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match="reserved numeric spelling"):
            literal_sites(["-", "inf", "inf"], engine)


class TestCanonicalGrammarUnaffected:
    @pytest.mark.parametrize("tok,expected", LEGAL_NUMERIC)
    def test_numeric_literals(self, engine: SimpliPyEngine, tok: str, expected: list[str]) -> None:
        assert engine.simplify([tok]) == expected
        assert engine.simplify(["-", tok, tok]) == ["0"]

    def test_fraction_literal(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify(["-", "1/3", "1/3"]) == ["0"]

    @pytest.mark.parametrize("tok", LEGAL_SYMBOLS)
    def test_free_symbols(self, engine: SimpliPyEngine, tok: str) -> None:
        assert engine.simplify([tok]) == [tok]
        assert engine.simplify(["-", tok, tok]) == ["0"]

    @pytest.mark.parametrize("tok", LEGAL_SIGILS)
    def test_sigil_tokens_stay_legal(self, engine: SimpliPyEngine, tok: str) -> None:
        assert engine.simplify([tok]) == [tok]

    def test_special_spellings(self, engine: SimpliPyEngine) -> None:
        assert engine.simplify(['float("inf")']) == ['float("inf")']
        assert engine.simplify(["np.pi"]) == ["np.pi"]


class TestGrammarPredicates:
    def test_reserved_predicate(self) -> None:
        for tok in RESERVED:
            assert reserved_numeric_spelling(tok), tok
        for tok in (LEGAL_SYMBOLS + LEGAL_SIGILS + [t for t, _ in LEGAL_NUMERIC]
                    + ["1/3", "np.pi", 'float("inf")', "(-1)"]):
            assert not reserved_numeric_spelling(tok), tok

    def test_is_numeric_string_covers_engine_emitter(self) -> None:
        # py_float_repr (the Python-parity fold emitter) spells big magnitudes with
        # 'e+' exactly as Python repr does; the masking predicate must recognize the
        # engine's own emissions (closure under emission).
        assert is_numeric_string("1e+16")
        assert is_numeric_string("1e-05")
        assert not is_numeric_string("1_000")
        assert not is_numeric_string("inf")


class TestRecursionCapEveryEntry:
    """H-043 (D4 doctrine-propagation, 2026-08-05): the MAX_TOKENS recursion cap lives in
    `ensure_tokens_are_tokens` (the one choke point) and EVERY token-taking FFI entry runs
    it. Before the sweep, twelve entries reached recursive walkers uncapped and a deep
    chain ABORTED the interpreter (SIGSEGV, probed empirically) -- pyo3 maps panics, not
    stack overflow. Each call below must raise a clean ValueError instead."""

    def test_deep_input_raises_everywhere(self, engine) -> None:
        core = engine._core
        deep = ["neg"] * 200_000 + ["x0"]
        deepc = ["neg"] * 200_000 + ["1"]
        lib = core.build_candidate_library([["x0"]], ["x0"], [1.0] * 8, 8)
        calls = [
            lambda: core.evaluate_batch(deep, ["x0"], [1.0], 1, []),
            lambda: core.evaluate_constant_subtree(deepc),
            lambda: core.interval_finite_ae(deep),
            lambda: core.interval_class(deep),
            lambda: core.interval_value_components(deep),
            lambda: core.interval_value_set_box(deep, [-1.0], [1.0]),
            lambda: core.interval_domain_extension(deep, ["x0"]),
            lambda: core.interval_domain_extension_p(deep, [], ["x0"], []),
            lambda: core.convert_expression(deep),
            lambda: core.equivalent_no_const(deep, ["x0"], ["x0"], [1.0] * 8, 8),
            lambda: core.find_rule(deep, len(deep), None, [["x0"]], ["x0"], [1.0] * 8, 8),
            lambda: core.find_rule_lib(deep, len(deep), None, lib),
            lambda: core.exist_constants_fit(deepc, ["x0"], [1.0] * 8, 8, [1.0] * 8),
            lambda: core.exist_constants_fit_linear(deepc, ["x0"], [1.0] * 8, 8, [1.0] * 8),
            lambda: core.set_rules([(deep, ["x0"])]),
            lambda: core.mine_one_length([deep], lib, None),
            lambda: core.ac_ordered_below(deep, ["x0"]),
            lambda: core.ac_judge(deep, 48),
            lambda: core.parse("neg(" * 100_000 + "x0" + ")" * 100_000, True, False),
        ]
        for i, call in enumerate(calls):
            with pytest.raises(ValueError, match="too long"):
                call()


class TestRealizationDialectIsClosed:
    """C1.13: the realization dialect must be CLOSED over the operators the PROJECTIONS can
    emit -- which are not the config's operators.

    `x0 * x0` serializes as `pow x0 2` under a config whose only operator is `*`, and
    `rootn(x,2) * x` serializes as `pow x0 (3/2)` under one with `rootn` and `/` but no
    `pow`. Falling back to the bare canonical name handed those to Python's builtins, whose
    semantics are NOT the engine's: `pow(-2.0, 1.5)` is a COMPLEX number where the engine
    gives NaN. A config's own realization always wins; the core fallback only fills a gap.
    """

    _MUL = {"realization": "*", "alias": [], "inverse": "/", "arity": 2,
            "precedence": 2, "commutative": True}
    _DIV = {"realization": "simplipy.operators.div", "alias": [], "inverse": "*",
            "arity": 2, "precedence": 2, "commutative": False}
    _ROOTN = {"realization": "simplipy.operators.rootn", "alias": [], "inverse": None,
              "arity": 2, "precedence": 3, "commutative": False}

    def test_emitted_pow_is_the_engines_pow_not_pythons(self) -> None:
        import numpy as np

        import simplipy
        eng = SimpliPyEngine(operators={"*": self._MUL, "/": self._DIV, "rootn": self._ROOTN},
                             rules=[])
        out = eng.simplify(["*", "rootn", "x0", "2", "x0"], form="explicit")
        code = eng.prefix_to_infix(list(out), realization=True)
        assert "simplipy.operators.pow" in code, code
        got = eval(code, {"simplipy": simplipy, "np": np}, {"x0": -2.0})
        assert not isinstance(got, complex), f"Python's builtin pow leaked back in: {got!r}"
        assert np.isnan(got), got

    def test_core_ops_resolve_under_a_config_that_declares_none_of_them(self) -> None:
        eng = SimpliPyEngine(operators={"*": self._MUL}, rules=[])
        # the engine emits `pow` here although the config never declared it
        out = eng.simplify(["*", "x0", "x0"], form="explicit")
        assert eng.prefix_to_infix(list(out), realization=True) == \
            "simplipy.operators.pow(x0, 2)"
        for tokens, want in ((["rootn", "x0", "3"], "simplipy.operators.rootn(x0, 3)"),
                             (["inv", "x0"], "simplipy.operators.inv(x0)"),
                             (["neg", "x0"], "simplipy.operators.neg(x0)"),
                             (["/", "x0", "x1"], "simplipy.operators.div(x0, x1)")):
            assert eng.prefix_to_infix(tokens, realization=True) == want, tokens

    def test_a_declared_realization_always_wins(self, engine: SimpliPyEngine) -> None:
        # The fallback fills a GAP; it never overrides a config. A config declaring its own
        # spelling for a core operator keeps it.
        ops = {"*": self._MUL,
               "pow": {"realization": "np.power", "alias": [], "inverse": None,
                       "arity": 2, "precedence": 3, "commutative": False}}
        eng = SimpliPyEngine(operators=ops, rules=[])
        assert eng.prefix_to_infix(["pow", "x0", "2"], realization=True) == "np.power(x0, 2)"
