"""Sign doctrine of the serializers: one placement rule, every dialect.

A sign lives in the numeric coefficient literal whenever one is emitted; additive term
signs are STRUCTURE (the tagged ``<sub>`` section, the explicit binary minus); ``neg``
spells only the PURE sign, where no literal exists to carry it.

PREMISE RESTATED 2026-08-07 (owner-ratified argmin spelling, contract §10.10(1); audit
F31/F33). The doctrine is unchanged, but the clause "whenever one is emitted" now bites
much more often: a coefficient is spelled as ONE literal only when that is its argmin
spelling, and for a ``2^a`` denominator it is not (``1/2`` spells ``/ 1 2``, not ``0.5``).
So ``-1/2 * x0`` has NO coefficient literal to carry the sign and correctly falls to the
pure-sign case, ``/ neg x0 2`` -- which is this doctrine applying, not deviating from it.
The pins below moved with that premise; the placement RULE did not change. Spelling is
free either way: mu is spelling-independent (all these forms re-parse to one ``Ex``).

The tagged form was born this way (``<mul> -2 x0 </mul>``); the explicit form used to
deviate (``neg * 2 x0``),
which put a spelling wrapper between every negative computed fold and anything reading
the projection -- the coverage-ordering Const-fulfillment walk judged ``* <constant> _0``
unfulfilled by ``neg * 0.841... _0`` purely on the wrapper, keeping abstraction rules
alive on a spelling artifact. The parser is liberal (both spellings map to one canonical
state), so this is an EMISSION contract: pinned here at the Python surface for both
dialects and for the masked skeletons that inherit it.
"""
from simplipy import SimpliPyEngine, masking

_OPS = {
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "/": {"realization": "simplipy.operators.div", "alias": [], "inverse": "*", "arity": 2, "precedence": 2, "commutative": False},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "sin": {"realization": "np.sin", "alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False},
}


def _engine() -> SimpliPyEngine:
    return SimpliPyEngine(operators=_OPS, rules=[])


class TestExplicitSignDoctrine:
    def test_negative_coefficient_spells_signed_literal(self) -> None:
        engine = _engine()
        assert engine.simplify(["*", "(-2)", "x0"]) == ["*", "-2", "x0"]
        # -1/2 has no coefficient LITERAL under argmin spelling (`1/2` beats `0.5`), so
        # the sign has nothing to ride and the pure-sign case applies. Doctrine intact.
        assert engine.simplify(["*", "(-0.5)", "x0"]) == ["/", "neg", "x0", "2"]

    def test_computed_negative_fold_carries_its_sign(self) -> None:
        """The fold of a ground subtree lands as ONE signed token, never `neg` + magnitude
        (the spelling that hid fold results from the Const-fulfillment walk). STAGE 2:
        the fixture moved from sin(-1) (a transcendental rounding, which the mu-governed
        fold now refuses -- pinned below) to an EXACT rational fold, which keeps firing;
        the sign doctrine is unchanged where folds still happen."""
        engine = _engine()
        # -3/2 spells as the fraction under argmin, so the fold lands as a signed
        # NUMERATOR literal (`-3`) rather than a signed decimal (`-1.5`): still ONE
        # signed token, never `neg` + magnitude, which is what this test guards.
        assert engine.simplify(["*", "x0", "/", "(-3)", "2"]) == \
            ["/", "*", "-3", "x0", "2"]
        # The old fixture's state stays symbolic under mu (rounding refused).
        out = engine.simplify(["*", "x0", "sin", "(-1)"])
        assert "sin" in out and not any("0.84" in t for t in out), out

    def test_rational_split_signs_the_numerator_literal(self) -> None:
        """A negative non-decimal rational routes through the division display with the
        sign on the numerator's coefficient literal."""
        engine = _engine()
        assert engine.simplify(["*", "(-2)", "/", "x0", "3"]) == \
            ["/", "*", "-2", "x0", "3"]

    def test_pure_sign_keeps_neg(self) -> None:
        """No literal to carry the sign: `neg` stays -- the lone variable and the
        symbolic product. The -1/3 coefficient's divisor-side spelling (2026-08-01,
        design §8) is AMENDED by H-020 (2026-08-04): the sign never rides a den
        literal (it would re-parse into the divisor bag, where the sign-fold clause
        may absorb it into a Const-bearing sum -- a different parking than the outer
        coefficient), so the pure sign returns to the numerator side as `neg` while
        the den token stays positive. Pretty infix still leads with the sign
        (`-x0/3`)."""
        engine = _engine()
        assert engine.simplify(["neg", "x0"]) == ["neg", "x0"]
        assert engine.simplify(["*", "neg", "np.pi", "x0"]) == \
            ["neg", "*", "np.pi", "x0"]
        assert engine.simplify(["*", "(-1)", "/", "x0", "3"]) == \
            ["/", "neg", "x0", "3"]
        assert engine.simplify(engine.to_infix(["*", "(-1)", "/", "x0", "3"])) == "-x0/3"

    def test_absorbed_spelling_is_fixpoint(self) -> None:
        """The parser reads the absorbed spelling back to the same state: emission is
        idempotent through a parse round-trip."""
        engine = _engine()
        for spelled in (["*", "-2", "x0"],
                        ["/", "neg", "x0", "2"],   # -1/2: the argmin spelling is the fixpoint
                        ["/", "*", "-2", "x0", "3"]):
            assert engine.simplify(spelled) == spelled


class TestDialectCongruence:
    """Explicit now mirrors the tagged form's sign placement exactly: factor signs in the
    coefficient literal, additive term signs as structure (`<sub>` section <-> binary
    minus), `neg` only for the pure sign."""

    def test_tagged_spelling_unchanged(self) -> None:
        engine = _engine()
        assert engine.simplify(engine.to_tagged(["*", "(-2)", "x0"])) == \
            ["<mul>", "-2", "x0", "</mul>"]

    def test_additive_signs_stay_structural_in_both_dialects(self) -> None:
        engine = _engine()
        source = ["-", "x2", "*", "0.5", "x1"]
        # The 1/2 coefficient takes its argmin spelling in BOTH dialects (explicit `/ x1 2`,
        # tagged `<div> 2`); what this test guards is that the ADDITIVE sign stays
        # structural -- binary `-` and `<sub>` -- which it does.
        assert engine.simplify(source) == ["-", "x2", "/", "x1", "2"]
        assert engine.simplify(engine.to_tagged(source)) == \
            ["<add>", "x2", "<sub>", "<mul>", "x1", "<div>", "2", "</mul>", "</add>"]


class TestMaskedSkeletonInheritsSigns:
    """The policy is applied positionally, so skeleton sign placement is decided entirely
    by the emission doctrine: a masked coefficient leaves NO sign wrapper behind in
    either dialect, and additive structure stays as spelled. (The COLLECT stage runs after
    the policy, which is why a `-3/2` coefficient spelled across the bag and its
    `<div>` section still yields ONE `<constant>` -- one per degree of freedom.)"""

    def test_masked_coefficient_has_no_sign_wrapper(self) -> None:
        # STAGE 2: same fixture move as above -- an exact rational fold carries the
        # doctrine (the transcendental one stays symbolic under mu).
        engine = _engine()
        folded = engine.simplify(["*", "x0", "/", "(-3)", "2"])
        skeleton = masking.mask(folded, engine, masking.mask_fittable)
        assert skeleton == ["*", "<constant>", "x0"]
        tagged = engine.simplify(engine.to_tagged(["*", "x0", "/", "(-3)", "2"]))
        assert masking.mask(tagged, engine, masking.mask_fittable) == \
            ["<mul>", "<constant>", "x0", "</mul>"]

    def test_masked_subtraction_stays_structural(self) -> None:
        engine = _engine()
        # The additive sign is STRUCTURE while the literal is exact -- but once the
        # addend is a FREE constant the sign is part of what gets fitted, so the collect
        # stage normalizes `x0 - c` to `x0 + c` (owner ruling 2026-08-07: "I'd prefer
        # addition by a constant instead of subtraction"). A negative fitted value is a
        # RENDERING question at substitution time, not a skeleton one.
        explicit = engine.simplify(["-", "x0", "0.5"])
        assert masking.mask(explicit, engine, masking.mask_fittable) == \
            ["+", "x0", "<constant>"]
