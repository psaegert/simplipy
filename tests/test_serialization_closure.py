"""Serialization-closure gates (the emit-parse closure family).

The AC core's serializers emit a FIXED core language -- the sugar operators
``+ - * / neg inv pow rootn``, the tagged bags, the literal spellings, and the
pretty-infix symbols (``^``, function calls, and the reserved constant names
``pi``/``e``/``inf``/``nan``). The parse side must read every spelling the emit
side can produce, under ANY operator config: parsing is canonicalization, and an
engine that cannot re-read its own output turns ``simplify(simplify(e))`` into a
crash (loud breach) or a silent wrong-state read (quiet breach; e.g. ``pi``
re-parsed as a free variable).

These tests are the standing falsifier for the whole family:

* exhaustive prefix closure: the COMPLETE L<=4 universe of each adversarial
  config's own vocabulary, both prefix projections, judged against a
  full-vocabulary referee engine's canonical state;
* infix closure: pretty output feeds back to the same canonical state, including
  reserved constants, ``^`` powers under pow{N}-less configs, ``rootn`` calls,
  and core symbols the config does not declare;
* the config guard: core serialization tokens cannot be redeclared with a
  conflicting arity or shadowed by an alias.
"""
import os
import tempfile

import pytest
import yaml

from simplipy import SimpliPyEngine

OPS_FULL = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "-": {"realization": "-", "alias": [], "inverse": "+", "arity": 2, "precedence": 1, "commutative": False},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "/": {"realization": "simplipy.operators.div", "alias": [], "inverse": "*", "arity": 2, "precedence": 2, "commutative": False},
    "inv": {"realization": "simplipy.operators.inv", "alias": ["inverse"], "inverse": "inv", "arity": 1, "precedence": 4, "commutative": False},
    "pow": {"realization": "simplipy.operators.pow", "alias": ["power"], "inverse": None, "arity": 2, "precedence": 3, "commutative": False},
    # C1.18: rootn was the ONE core sugar token no config in the suite declared.
    "rootn": {"realization": "simplipy.operators.rootn", "alias": [], "inverse": None, "arity": 2, "precedence": 3, "commutative": False},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
    "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    "sin": {"realization": "np.sin", "alias": [], "inverse": "asin", "arity": 1, "precedence": 3, "commutative": False},
    "abs": {"realization": "simplipy.operators.abs", "alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False},
    # C1.18: a non-core BINARY operator (every non-core op used to be unary) and an
    # ARITY-3 operator -- both stay opaque nodes with call rendering, and the battery
    # proves the emit-parse closure survives them anywhere in a term.
    "hypot2": {"realization": "np.hypot", "alias": [], "inverse": None, "arity": 2, "precedence": 3, "commutative": False},
    "fma3": {"realization": "math.fma", "alias": [], "inverse": None, "arity": 3, "precedence": 3, "commutative": False},
}

LEAVES = ["x0", "x1", "2", "-1", "<constant>"]
MAX_LEN = 4

# Each core sugar spelling individually missing from an otherwise-full config, plus
# minimal vocabularies: the adversarial configs the emit side must survive.
# (C1.18 additions: no_pow and no_rootn complete the per-core-token family, and
# minimal_call_ops is a vocabulary of pure call-rendered operators.)
DEGENERATE = {
    "no_neg": [k for k in OPS_FULL if k != "neg"],
    "no_inv": [k for k in OPS_FULL if k != "inv"],
    "no_sub": [k for k in OPS_FULL if k != "-"],
    "no_div": [k for k in OPS_FULL if k != "/"],
    "no_add": [k for k in OPS_FULL if k != "+"],
    "no_mul": [k for k in OPS_FULL if k != "*"],
    "no_pow": [k for k in OPS_FULL if k != "pow"],
    "no_rootn": [k for k in OPS_FULL if k != "rootn"],
    "minimal_add_mul": ["+", "*", "exp", "sin"],
    "minimal_mul_inv": ["*", "inv", "log"],
    "minimal_call_ops": ["fma3", "hypot2", "exp"],
}


def make_engine(op_names: list[str], operators: dict | None = None) -> SimpliPyEngine:
    ops = operators if operators is not None else OPS_FULL
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "rules.json"), "w") as fh:
        fh.write("[]")
    cfg = os.path.join(d, "config.yaml")
    with open(cfg, "w") as fh:
        fh.write(yaml.safe_dump({"operators": {k: ops[k] for k in op_names}, "rules": "rules.json"}))
    return SimpliPyEngine.from_config(cfg)


def universe(op_names: list[str], max_len: int) -> list[list[str]]:
    """The COMPLETE set of prefix expressions over ops+LEAVES with <= max_len tokens.

    n-ary (C1.18): the old generator hard-assumed arity <= 2 and would emit
    malformed prefixes for a ternary operator.
    """
    arity = {k: OPS_FULL[k]["arity"] for k in op_names}

    def args(n: int, budget: int):
        """All n-tuples of subtrees with total length <= budget."""
        if n == 0:
            yield ()
            return
        for first in gen(budget - (n - 1)):
            for rest in args(n - 1, budget - len(first)):
                yield (first,) + rest

    def gen(budget: int):
        if budget < 1:
            return
        for leaf in LEAVES:
            yield (leaf,)
        for op, a in arity.items():
            if budget >= 1 + a:
                for tup in args(a, budget - 1):
                    yield (op,) + tuple(t for sub in tup for t in sub)

    return [list(e) for e in dict.fromkeys(gen(max_len))]


@pytest.fixture(scope="module")
def referee() -> SimpliPyEngine:
    return make_engine(list(OPS_FULL))


def ref_state(referee: SimpliPyEngine, tokens: list[str]) -> tuple[str, ...]:
    """Canonical state under the full-vocabulary referee (the semantic oracle)."""
    return tuple(referee.simplify(list(tokens), form="tagged"))


class TestPrefixClosureExhaustive:
    """Every prefix emission parses, is a fixpoint, and means what the source meant."""

    @pytest.mark.parametrize("config", list(DEGENERATE))
    def test_universe_round_trips(self, config: str, referee: SimpliPyEngine) -> None:
        op_names = DEGENERATE[config]
        engine = make_engine(op_names)
        failures: list[str] = []
        for src in universe(op_names, MAX_LEN):
            want = ref_state(referee, src)
            for form in ("tagged", "explicit"):
                out = engine.simplify(list(src), form=form)
                try:
                    back = engine.simplify(list(out), form=form)
                except ValueError as exc:  # noqa: PERF203 -- collecting all failures
                    failures.append(f"[{form}] {' '.join(src)} -> {out}: RAISE {exc}")
                    continue
                if list(back) != list(out):
                    failures.append(f"[{form}] {' '.join(src)} -> {out} -> {back}: not a fixpoint")
                    continue
                got = ref_state(referee, list(out))
                if got != want:
                    failures.append(f"[{form}] {' '.join(src)} -> {out}: state {got} != {want}")
        assert not failures, f"{len(failures)} closure breaches, first 10:\n" + "\n".join(failures[:10])

    @pytest.mark.parametrize("config", list(DEGENERATE))
    def test_mine_acceptance_reads_own_output(self, config: str) -> None:
        """The F2 acceptance path: ordered_below(out, out) must DECIDE (False), never error,
        on the engine's own explicit output -- a None there silently refuses mined rules."""
        op_names = DEGENERATE[config]
        engine = make_engine(op_names)
        for src in universe(op_names, 3):
            judged = engine._core.ac_judge(list(src), 48)
            assert judged is not None, f"ac_judge failed on in-vocabulary source {src}"
            out = judged[2]
            assert engine._core.ac_ordered_below(list(out), list(out)) is False, (
                f"acceptance cannot read the engine's own output {out} (source {src})"
            )


class TestInfixClosure:
    """The pretty rendering feeds back to the same canonical state."""

    CASES = [
        ["+", "x0", "np.pi"],          # renders 'x0 + pi' -- reserved name, not a variable
        ["*", "np.e", "x0"],           # renders 'e*x0'
        ["/", "1", "0"],               # folds to 'inf'
        ["log", "-1"],                 # folds to 'nan'
        ["neg", "float(\"inf\")"],     # renders '-inf'
        ["pow", "x0", "2"],            # renders 'x0^2'
        ["pow", "x0", "0.5"],          # fractional power without the pow1_N vocabulary
        ["pow", "x0", "/", "1", "3"],  # exact-fraction exponent
        ["rootn", "x0", "3"],          # function-call rendering 'rootn(x0, 3)'
        ["+", "x0", "/", "1", "3"],    # exact fraction literal
        ["+", "x0", "inv", "x0"],      # 'x0 + 1/x0' -- precedence of the core symbols
    ]

    @pytest.mark.parametrize("src", CASES, ids=lambda s: " ".join(s))
    def test_full_config_feedback(self, src: list[str], referee: SimpliPyEngine) -> None:
        rendered = referee.simplify(list(src), form="infix")
        back = referee._core.parse(rendered, True, False)
        assert ref_state(referee, back) == ref_state(referee, src), (
            f"{' '.join(src)} -> {rendered!r} -> {back} changed canonical state"
        )

    def test_reserved_constant_names_parse_as_constants(self, referee: SimpliPyEngine) -> None:
        assert referee.infix_to_prefix("x0 + pi") == ["+", "x0", "np.pi"]
        assert referee.infix_to_prefix("e*x0") == ["*", "np.e", "x0"]
        assert referee.infix_to_prefix("inf") == ['float("inf")']
        assert referee.infix_to_prefix("nan") == ['float("nan")']

    @pytest.mark.parametrize("config", ["no_div", "no_add", "no_mul", "no_pow", "no_rootn",
                                        "minimal_add_mul", "minimal_mul_inv", "minimal_call_ops"])
    def test_degenerate_config_feedback(self, config: str, referee: SimpliPyEngine) -> None:
        """Core infix symbols parse with their standard precedences even when the config
        does not declare them (a missing '/' used to parse 'x0 + 1/x0' as '(x0+1)/x0')."""
        op_names = DEGENERATE[config]
        engine = make_engine(op_names)
        failures: list[str] = []
        for src in universe(op_names, MAX_LEN):
            want = ref_state(referee, src)
            rendered = engine.simplify(list(src), form="infix")
            try:
                back = engine._core.parse(rendered, True, False)
                got = ref_state(referee, back)
            except ValueError as exc:
                failures.append(f"{' '.join(src)} -> {rendered!r}: RAISE {exc}")
                continue
            if got != want:
                failures.append(f"{' '.join(src)} -> {rendered!r} -> {back}: state {got} != {want}")
        assert not failures, f"{len(failures)} infix closure breaches, first 10:\n" + "\n".join(failures[:10])


class TestCoreTokenGuard:
    """Core serialization tokens are reserved words of the engine's language."""

    def test_conflicting_arity_is_a_config_error(self) -> None:
        ops = {"-": {**OPS_FULL["-"], "arity": 1}, "*": OPS_FULL["*"]}
        with pytest.raises(Exception, match="core serialization"):
            make_engine(["-", "*"], operators=ops)

    def test_alias_shadowing_a_core_token_is_a_config_error(self) -> None:
        ops = {"*": OPS_FULL["*"], "exp": {**OPS_FULL["exp"], "alias": ["neg"]}}
        with pytest.raises(Exception, match="core serialization"):
            make_engine(["*", "exp"], operators=ops)

    def test_conflicting_precedence_is_a_config_error(self) -> None:
        """C1.8: the guard covered arity and aliases but NOT precedence, which left the
        closure breach open from the other side.

        The realization rendering is a string PYTHON evaluates, and Python's precedence is
        fixed. `/` at precedence 0.5 renders `(x0+x1)/x2` as `x0 + x1 / x2` -- which this
        engine re-parses CORRECTLY, because its parser shares the same table, so no
        round-trip test can see it -- while Python reads `x0 + (x1/x2)`: 2.5 where the
        value is 2.0. A silent wrong answer at the eval boundary.
        """
        ops = {"+": OPS_FULL["+"], "/": {**OPS_FULL["/"], "precedence": 0.5}}
        with pytest.raises(Exception, match="core serialization"):
            make_engine(["+", "/"], operators=ops)

    def test_declaring_core_tokens_normally_is_fine(self) -> None:
        engine = make_engine(list(OPS_FULL))
        assert engine.simplify(["+", "x0", "x0"], form="explicit") == ["*", "2", "x0"]

    def test_core_token_without_precedence_gets_the_core_value(self) -> None:
        """C1.18 guard hole: a core token declared WITHOUT `precedence:` used to slip
        past the C1.8 conflicting-precedence guard entirely (it compares only present
        values). Absence now means the core's own value -- verified through the one
        consumer that would silently corrupt: the infix rendering Python evaluates."""
        ops = {"+": OPS_FULL["+"],
               "/": {k: v for k, v in OPS_FULL["/"].items() if k != "precedence"}}
        engine = make_engine(["+", "/"], operators=ops)
        assert engine.simplify(["/", "+", "x0", "x1", "x2"], form="infix") == "(x0 + x1)/x2"


class TestOperatorSpecValidation:
    """C1.18: the configs the DEGENERATE battery never built, at the load boundary.

    A spec missing a key used to die as a bare ``KeyError: 'alias'`` naming neither
    the operator nor the file; ``commutative: true`` on a non-core operator was
    consumed by NOTHING (only ``+``/``*`` are AC bags), so ``f(a, b)`` and
    ``f(b, a)`` stayed two canonical states while the config claimed one value.
    """

    def test_missing_keys_are_a_loud_config_error(self) -> None:
        ops = {"+": OPS_FULL["+"],
               "sq": {"realization": "np.square", "arity": 1, "precedence": 3}}
        with pytest.raises(Exception, match=r"'sq'.*'alias'"):
            make_engine(["+", "sq"], operators=ops)

    def test_commutative_on_a_non_core_operator_is_refused(self) -> None:
        # The one honest answer for a flag the engine cannot honor: refusal at load.
        # (Measured before ruling: is_commutative has no consumer for non-core
        # tokens, and 0 of 50 repo configs declare a non-core binary at all.)
        ops = {"+": OPS_FULL["+"], "hypot2": {**OPS_FULL["hypot2"], "commutative": True}}
        with pytest.raises(Exception, match="cannot honor"):
            make_engine(["+", "hypot2"], operators=ops)

    def test_precedence_is_optional_for_non_core_operators(self) -> None:
        # Call rendering never consults precedence, so its absence is not an error.
        ops = {"+": OPS_FULL["+"],
               "sq": {"realization": "np.square", "alias": [], "inverse": None,
                      "arity": 1, "commutative": False}}
        engine = make_engine(["+", "sq"], operators=ops)
        assert engine.simplify(["sq", "+", "x0", "x1"], form="infix") == "sq(x0 + x1)"

    def test_declared_rootn_realization_wins_and_evaluates(self) -> None:
        # rootn was the one core sugar token no suite config DECLARED (C1.18): a
        # declared realization must win over the core fallback and evaluate.
        from simplipy import codify
        engine = make_engine(list(OPS_FULL))
        rendered = engine.prefix_to_infix(["rootn", "x0", "3"], realization=True)
        assert rendered == "simplipy.operators.rootn(x0, 3)"
        assert engine.code_to_lambda(codify(rendered, ["x0"]))(8.0) == pytest.approx(2.0)

    def test_arity3_operator_realization_evaluates(self) -> None:
        from simplipy import codify
        engine = make_engine(list(OPS_FULL))
        rendered = engine.prefix_to_infix(["fma3", "x0", "x1", "2"], realization=True)
        assert rendered == "math.fma(x0, x1, 2)"
        assert engine.code_to_lambda(codify(rendered, ["x0", "x1"]))(2.0, 3.0) == pytest.approx(8.0)


class TestSignedNumericLeafSplit:
    """H-048 (2026-08-05, extreme-literal lane): a SIGNED numeric spelling never
    becomes a leaf -- the sign is STRUCTURE. Beyond-i128 spellings used to keep the
    sign inside the opaque leaf token, while the same value built structurally kept a
    positive leaf under a subtracted section: two canonical states, one function,
    conflated exactly at the infix boundary (the renderer fuses a section sign into
    the literal; parse read it back as a signed leaf -- extreme-lane fuzz row 61)."""

    def test_signed_and_structural_routes_agree(self) -> None:
        engine = make_engine(list(OPS_FULL))
        assert engine.simplify(["-1e309"]) == engine.simplify(["neg", "1e309"])
        big = "1.7976931348623157e308"
        assert engine.simplify([f"-{big}"]) == engine.simplify(["neg", big])

    def test_row61_shape_infix_roundtrip_closes(self) -> None:
        engine = make_engine(list(OPS_FULL))
        expr = "- / - 2 1.7976931348623157e308 10000000000000000001 pow x2 3.402823669209385e38".split()
        out = engine.simplify(list(expr))
        assert engine.simplify(list(out)) == out
        infix = engine.simplify(list(expr), form="infix")
        back = engine.parse(infix, mask_numbers=False)
        assert engine.simplify(list(back)) == out
        exp_out = engine.simplify(list(expr), form="explicit")
        assert engine.simplify(list(exp_out), form="explicit") == exp_out


class TestProvenanceRecordsTheMeasure:
    """C1.4: an artifact mined under one measure must be distinguishable from one mined
    under another.

    The sidecar records version, build and parameters, and EVERY one of those can be
    identical across a measure change -- worse, uncommitted work stamps the same `-dirty`
    build string. The audit flagged it as a blocker to fix BEFORE a re-mine; three
    artifacts were re-mined without it. The fingerprint is BEHAVIOURAL: it records what
    the measure charges on probes chosen to separate the changes it has undergone.
    """

    def test_fingerprint_separates_the_measure_axes(self) -> None:
        engine = make_engine(list(OPS_FULL))
        fp = engine._measure_fingerprint()
        assert fp['unit'] == 'milli-bits'
        assert fp['mu_sym'] == engine.complexity(['x0'])
        assert fp['mu_free'] == engine.complexity(['<constant>'])
        # each probe separates a change the measure has actually undergone
        for probe in ('1000', '1/2', '355/113', '0.2', '<constant>', 'x0'):
            assert isinstance(fp['probes'][probe], int), probe
        assert len(fp['digest']) == 16

    def test_the_digest_is_a_function_of_the_probes(self) -> None:
        # The point of the fingerprint: it must MOVE when the ordering does. The env
        # switches are read once per process (OnceLock), so perturbing the live measure
        # inside one test run is not possible -- instead pin that the digest is derived
        # from the probe VALUES, so any measure change that moves a probe moves it.
        import hashlib
        fp = make_engine(list(OPS_FULL))._measure_fingerprint()
        want = hashlib.sha256(repr(sorted(fp['probes'].items())).encode()).hexdigest()[:16]
        assert fp['digest'] == want
        perturbed = dict(fp['probes'], **{'1/2': fp['probes']['1/2'] + 1})
        moved = hashlib.sha256(repr(sorted(perturbed.items())).encode()).hexdigest()[:16]
        assert moved != fp['digest']

    def test_the_tagged_fraction_bound_is_a_recorded_switch(self) -> None:
        # It changes MINED output, so an artifact must record it; it was missing from the
        # list while every other ordering-affecting switch was present.
        from simplipy.engine import ARTIFACT_ENV_SWITCHES
        assert 'SIMPLIPY_TAGGED_FRACTION_MAX' in ARTIFACT_ENV_SWITCHES


class TestCoreTableHasOneSource:
    """C1.10: the core serialization language was declared in FIVE hand-maintained places.

    A drifting copy is a soundness problem, not a tidiness one: the masking walk treats an
    unknown operator as a LEAF, so its operands inherit the enclosing bag's role and a
    `pow` exponent can be masked -- the exact accident the role API exists to prevent.
    The Rust table is now the single source and every consumer reads it.
    """

    def test_python_consumers_read_the_engine_table(self) -> None:
        from simplipy._core import core_serialization_ops
        from simplipy.engine import _CORE_SERIALIZATION_ARITIES
        from simplipy.verify._contract import ARITY

        table = core_serialization_ops()
        assert set(table) == {'+', '-', '*', '/', 'neg', 'inv', 'pow', 'rootn'}
        for tok, spec in table.items():
            assert _CORE_SERIALIZATION_ARITIES[tok] == spec['arity'], tok
            assert ARITY[tok] == spec['arity'], tok

    def test_the_judges_extra_vocabulary_is_still_its_own(self) -> None:
        # Deriving the core half must not have swallowed the judge's DELIBERATE extras:
        # it keeps the deleted 0.11 hyper-operator spellings so legacy rewrites stay
        # judgeable.
        from simplipy.verify._contract import ARITY
        for legacy in ('pow2', 'pow1_2', 'div5', 'sin', 'atanh'):
            assert legacy in ARITY, legacy

    def test_realizations_come_from_the_same_table(self) -> None:
        # The fourth column: a config that declares none of the core operators still gets
        # the engine's own realizations, because `Operators::from_specs` fills them from
        # this table exactly as it fills arity and precedence.
        from simplipy._core import core_serialization_ops
        table = core_serialization_ops()
        eng = SimpliPyEngine(operators={'*': OPS_FULL['*']}, rules=[])
        for tok in ('pow', 'rootn', 'inv', 'neg', '/'):
            assert table[tok]['realization'] is not None, tok
        assert eng.prefix_to_infix(['pow', 'x0', '2'], realization=True) == \
            f"{table['pow']['realization']}(x0, 2)"
        # ...and +,-,* carry none, because they render as Python infix operators whose
        # semantics already match the engine's.
        for tok in ('+', '-', '*'):
            assert table[tok]['realization'] is None, tok
