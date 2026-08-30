"""The structure-aware masking toolkit.

Masking is DOWNSTREAM policy; `simplipy.masking` is the mechanism. These tests pin the
role classification on both output forms, the policy plumbing, and the
regression: a position-blind mask used to relabel `rootn`'s structural index into an
unfittable skeleton; the role-aware walk makes that mistake impossible to write silently.
"""

import pytest

from simplipy import SimpliPyEngine, masking
from conftest import acj_config_path


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(
        acj_config_path())


class TestRoleClassification:
    def test_roles_across_both_forms(self, engine: SimpliPyEngine) -> None:
        cases = [
            (["<mul>", "2", "x0", "</mul>"], [("2", masking.Role.COEFFICIENT)]),
            (["<add>", "x2", "2.3", "<sub>", "x1", "</add>"], [("2.3", masking.Role.ADDEND)]),
            (["pow", "x0", "2"], [("2", masking.Role.EXPONENT)]),
            (["rootn", "x0", "3"], [("3", masking.Role.ROOT_INDEX)]),
            (["<mul>", "1/3", "x0", "<div>", "pow", "x1", "0.5", "</mul>"],
             [("1/3", masking.Role.COEFFICIENT), ("0.5", masking.Role.EXPONENT)]),
            (["*", "0.2", "sin", "x0"], [("0.2", masking.Role.COEFFICIENT)]),
            (["-", "x0", "2"], [("2", masking.Role.ADDEND)]),
            (["pow", "2", "x0"], [("2", masking.Role.VALUE)]),
            (["neg", "2"], [("2", masking.Role.VALUE)]),
            (["sin", "<constant>"], []),  # <constant> is never a site
        ]
        for tokens, want in cases:
            got = [(v, r) for _, v, r in masking.literal_sites(tokens, engine)]
            assert got == want, tokens

    def test_malformed_input_raises(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError):
            masking.literal_sites(["<mul>", "2", "x0"], engine)  # unclosed bag
        with pytest.raises(ValueError):
            masking.literal_sites(["pow", "x0"], engine)  # arity underflow

    def test_wrong_engine_object_fails_loudly(self, engine: SimpliPyEngine) -> None:
        # A silent operator-table default (the E1 "DEP_OPS silently emptied" shape) let
        # a wrong `engine` walk with NO function arities: on tagged input the function
        # argument's literal took the enclosing BAG's role (sin(3)+x0 reported the 3 as
        # ADDEND, silently mis-masked), on explicit input the walk blamed the INPUT
        # ("malformed prefix expression") for the instrument's fault. The table is now a
        # direct attribute read: a wrong object fails at the boundary, naming the problem.
        tokens = ["<add>", "sin", "3", "x0", "</add>"]  # sin(3) + x0: the 3 is a VALUE
        roles = [r for _, _, r in masking.literal_sites(tokens, engine)]
        assert roles == [masking.Role.VALUE]
        with pytest.raises(AttributeError):
            masking.literal_sites(tokens, object())
        with pytest.raises(AttributeError):
            masking.mask(["*", "2", "sin", "x0"], object(), masking.mask_all)


class TestPolicies:
    def test_keep_structure_never_touches_indices_or_exponents(
            self, engine: SimpliPyEngine) -> None:
        # The D2c regression: coefficient masked, rootn index AND exponent kept.
        out = engine.simplify(engine.to_tagged(["*", "2", "rootn", "pow", "x0", "5", "3"]))
        masked = masking.mask(out, engine, masking.mask_fittable)
        assert masked == ["<mul>", "<constant>", "rootn", "pow", "x0", "5", "3", "</mul>"]
        # The masked skeleton stays parseable (positional replacement preserves
        # canonical order; complexity() exercises the shared parser).
        assert engine.complexity(masked) > 0

    def test_mask_all_is_the_legacy_equivalent(self, engine: SimpliPyEngine) -> None:
        out = engine.simplify(engine.to_tagged(["+", "x3", "+", "x8", "*", "0.2", "x3"]))
        masked = masking.mask(out, engine, masking.mask_all)
        # The 6/5 coefficient is ONE degree of freedom, so it costs ONE <constant>
        # however the serializer spelled it. Note the TERM ORDER moves: `<constant>`
        # carries `mu_free`, so the masked term's sort key changes and the collect pass
        # re-sorts the bag. The skeleton is canonical for ITSELF, not for the expression
        # it came from.
        assert masked == ["<add>", "x8", "<mul>", "<constant>", "x3", "</mul>", "</add>"]

    def test_policy_sees_value_and_role(self, engine: SimpliPyEngine) -> None:
        seen: list[tuple[str, masking.Role]] = []

        def spy(value: str, role: masking.Role) -> None:
            seen.append((value, role))
            return None  # keep everything

        tokens = ["<mul>", "1/3", "pow", "x0", "2", "</mul>"]
        assert masking.mask(tokens, engine, spy) == tokens
        assert seen == [("1/3", masking.Role.COEFFICIENT), ("2", masking.Role.EXPONENT)]


class TestMaskOnce:
    """Masking DESTROYS information, so chaining it has consequences (owner 2026-08-07:
    "we should perhaps document this with a few examples"). `mask` takes an EXPRESSION,
    never a skeleton. flash-ansr masks exactly once, so this is a documented property
    rather than a defect -- but it is pinned so the property cannot drift silently."""

    def test_a_second_pass_invents_a_degree_of_freedom(self, engine: SimpliPyEngine) -> None:
        # The `<sub>` section is STRUCTURE. Collect moves the negation inside `atan` (an
        # odd function), where it becomes an explicit `-1` coefficient -- structure
        # wearing a literal's clothes. A second pass abstracts it into a free constant
        # the original never had. Measured at ~0.15% of skeletons over 20k corpus rows.
        expr = ["<add>", "<sub>", "atan", "<mul>", "pow", "x4", "3", "pow", "abs", "x4", "3",
                "</mul>", "sin", "<mul>", "0", "log", "x4", "</mul>", "</add>"]
        once = masking.mask(expr, engine, masking.mask_fittable)
        twice = masking.mask(once, engine, masking.mask_fittable)
        assert expr.count("<constant>") == 0
        assert once.count("<constant>") == 1
        assert twice.count("<constant>") == 2   # <- the invented degree of freedom
        assert "-1" in once and "-1" not in twice

    def test_masking_is_stable_where_no_structure_becomes_literal(
            self, engine: SimpliPyEngine) -> None:
        # The common case IS stable -- the caveat is narrow, not the rule.
        for tokens in (["*", "2/3", "x0"], ["+", "*", "3", "x0", "5"], ["-", "x0", "0.5"]):
            out = engine.simplify(list(tokens))
            once = masking.mask(out, engine, masking.mask_fittable)
            assert masking.mask(once, engine, masking.mask_fittable) == once


class TestSpecialConstantsAreSites:
    """Owner ruling 2026-07-31: `mask all should mask all, even pi or e or other
    specials.` Special constants (np.pi, np.e) are literal SITES like any numeric
    token -- the POLICY decides their fate, so a future pi-vocabulary pipeline keeps
    them with a one-line policy instead of a mechanism change. Poles and nan are NOT
    sites: a fitted `<constant>` is finite by doctrine, so masking an infinity would
    manufacture an unfittable skeleton."""

    def test_mask_all_masks_specials(self, engine: SimpliPyEngine) -> None:
        out = engine.simplify(engine.to_tagged(["*", "np.pi", "x0"]))
        assert masking.mask(out, engine, masking.mask_all) == \
            ["<mul>", "<constant>", "x0", "</mul>"]
        # 2*pi*x: both the integer and the special are SITES, so both are masked -- but
        # `2 * pi` is ONE value and therefore ONE degree of freedom, so the collect stage
        # leaves ONE <constant>. Two sites, one constant.
        out = engine.simplify(engine.to_tagged(["*", "2", "*", "np.pi", "x0"]))
        assert masking.mask(out, engine, masking.mask_all) == \
            ["<mul>", "<constant>", "x0", "</mul>"]
        out = engine.simplify(engine.to_tagged(["+", "x0", "np.e"]))
        assert masking.mask(out, engine, masking.mask_all) == \
            ["<add>", "x0", "<constant>", "</add>"]

    def test_keep_structure_keeps_special_exponents(self, engine: SimpliPyEngine) -> None:
        # In a STRUCTURAL position the keep-structure policy keeps the special exactly
        # like it keeps a structural integer; in a value position it masks it.
        out = engine.simplify(engine.to_tagged(["pow", "x0", "np.e"]))
        assert masking.mask(out, engine, masking.mask_fittable) == \
            ["pow", "x0", "np.e"]
        out = engine.simplify(engine.to_tagged(["*", "np.pi", "x0"]))
        assert masking.mask(out, engine, masking.mask_fittable) == \
            ["<mul>", "<constant>", "x0", "</mul>"]

    def test_sites_report_specials_with_roles(self, engine: SimpliPyEngine) -> None:
        got = [(v, r) for _, v, r in masking.literal_sites(
            ["<mul>", "np.pi", "pow", "x0", "np.e", "</mul>"], engine)]
        assert got == [("np.pi", masking.Role.COEFFICIENT),
                       ("np.e", masking.Role.EXPONENT)]

    def test_symbolic_constant_folds_into_the_free_constant(self, engine: SimpliPyEngine) -> None:
        """Owner ruling 2026-08-07: `2*pi` masks to ONE `<constant>`, never
        `<constant> * pi`.

        A free constant ABSORBS a symbolic one: `c * pi` is an arbitrary float, exactly
        like `c`, so the factorization the second form appears to preserve does not exist.
        It would only be worth keeping if the consumer predicted the coefficient exactly
        (an integer or a fraction) -- and those are masked here anyway, so it buys
        nothing. Both shipped kinds agree."""
        for policy in (masking.mask_fittable, masking.mask_all):
            out = engine.simplify(engine.to_tagged(["*", "2", "*", "np.pi", "x0"]))
            assert masking.mask(out, engine, policy) == \
                ["<mul>", "<constant>", "x0", "</mul>"]
            # ...and a SEPARATE additive term keeps its own degree of freedom.
            out = engine.simplify(engine.to_tagged(["+", "2", "*", "np.pi", "x0"]))
            assert masking.mask(out, engine, policy) == \
                ["<add>", "<mul>", "<constant>", "x0", "</mul>", "<constant>", "</add>"]

    def test_poles_and_nan_are_never_sites(self, engine: SimpliPyEngine) -> None:
        for tokens in (["*", 'float("inf")', "x0"],
                       ["+", "x0", 'float("-inf")'],
                       ["*", 'float("nan")', "x0"]):
            assert masking.literal_sites(list(tokens), engine) == []
            assert masking.mask(list(tokens), engine, masking.mask_all) == tokens

    def test_keep_specials_is_one_policy_line(self, engine: SimpliPyEngine) -> None:
        # MECHANISM test, NOT a recommended policy. The policy sees the value string, so
        # keeping specials needs no toolkit change -- but owner ruling 2026-08-07: keeping
        # a symbolic constant beside a MASKED coefficient is pointless, because
        # `<constant> * np.pi` just fits an arbitrary float and an arbitrary float times
        # pi is an arbitrary float. Contract 10.11 (2026-08-08) promoted that ruling into
        # the CANON: the engine itself absorbs a special beside a fitted mask (Mul-scale
        # bijection), so a policy-kept special adjacent to a `<constant>` no longer
        # survives canonicalization -- keeping a special is meaningful only where no
        # mask lands beside it, pinned below on `sin(pi*x0)`.
        # See `test_symbolic_constant_folds_into_the_free_constant`.
        def mask_all_keep_specials(value: str, role: masking.Role):
            return None if value in ("np.pi", "np.e") else "<constant>"

        out = engine.simplify(engine.to_tagged(["*", "2", "*", "np.pi", "x0"]))
        # The kept `np.pi` sits beside the mask the policy minted for `2`, and the
        # canon's absorption folds them: pi * <constant> IS <constant>.
        assert masking.mask(out, engine, mask_all_keep_specials) == \
            ["<mul>", "<constant>", "x0", "</mul>"]
        # Where the policy mints no adjacent mask, the kept special DOES survive
        # (no other literal in the bag), while the shipped policies mask it:
        out = engine.simplify(engine.to_tagged(["sin", "*", "np.pi", "x0"]))
        assert masking.mask(list(out), engine, mask_all_keep_specials) == \
            ["sin", "<mul>", "np.pi", "x0", "</mul>"]
        assert masking.mask(list(out), engine, masking.mask_all) == \
            ["sin", "<mul>", "<constant>", "x0", "</mul>"]


class TestEngineMaskFrontDoor:
    """``engine.mask(expr, policy='all')`` -- the owner-approved front door over
    this module (design D8): pure delegation, str->str / list->list symmetry,
    the three shipped policies by name or any callable policy."""

    def test_list_in_list_out_delegates_byte_identically(self, engine: SimpliPyEngine) -> None:
        toks = engine.read_infix('2*sin(x0) + 1.5')
        out = engine.mask(toks)
        assert isinstance(out, list)
        assert out == masking.mask(toks, engine, masking.mask_all)
        assert out == ['+', '*', '<constant>', 'sin', 'x0', '<constant>']

    def test_str_in_str_out(self, engine: SimpliPyEngine) -> None:
        out = engine.mask('2*sin(x0) + 1.5')
        assert isinstance(out, str)
        # the infix rendering is the structure-preserving renderer's (pure conversion)
        assert out == '<constant> * sin(x0) + <constant>'

    def test_default_policy_is_all(self, engine: SimpliPyEngine) -> None:
        toks = engine.read_infix('x0**2 * 0.5')
        assert engine.mask(toks) == engine.mask(toks, policy='all')

    def test_named_policy_all_masks_exponents(self, engine: SimpliPyEngine) -> None:
        tagged = engine.simplify(engine.to_tagged(['*', '2', 'pow', 'x0', '3']))
        assert engine.mask(tagged, policy='all') \
            == ['<mul>', '<constant>', 'pow', 'x0', '<constant>', '</mul>']

    def test_named_policy_fittable_keeps_exponents(self, engine: SimpliPyEngine) -> None:
        tagged = engine.simplify(engine.to_tagged(['*', '2', 'pow', 'x0', '3']))
        out = engine.mask(tagged, policy='fittable')
        assert out == ['<mul>', '<constant>', 'pow', 'x0', '3', '</mul>']
        # the emitted dialect is preserved (delegation, not conversion)
        assert out[0] == '<mul>'

    def test_named_policy_values_is_the_fittable_alias(self, engine: SimpliPyEngine) -> None:
        toks = engine.read_infix('x0**2 * 0.5')
        assert engine.mask(toks, policy='values') == ['*', '<constant>', 'pow', 'x0', '2']
        assert engine.mask(toks, policy='values') == engine.mask(toks, policy='fittable')

    def test_custom_callable_policy(self, engine: SimpliPyEngine) -> None:
        def keep_ints(value: str, role: masking.Role) -> str | None:
            return None if value.lstrip('-').isdigit() else '<constant>'

        toks = engine.read_infix('2.5*sin(x0) + 3')
        out = engine.mask(toks, policy=keep_ints)
        assert out == ['+', '*', '<constant>', 'sin', 'x0', '3']
        assert out == masking.mask(toks, engine, keep_ints)

    def test_unknown_policy_name_raises(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match="'all', 'fittable', 'values'"):
            engine.mask(['*', '2', 'x0'], policy='everything')

    def test_wrong_policy_type_raises(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(TypeError):
            engine.mask(['*', '2', 'x0'], policy=42)

    def test_wrong_expression_type_raises(self, engine: SimpliPyEngine) -> None:
        # The front door reads an infix str or a token LIST -- nothing else (D8).
        with pytest.raises(TypeError):
            engine.mask(('*', '2', 'x0'))
