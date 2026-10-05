"""`parse` is RENAMED to `read_infix` (ruling 2026-08-18, evening batch 2 item 3).

The name had to state the contract, because the contract is a CAPABILITY the rest
of the 0.14 surface deliberately refuses: `read_infix` is a vocabulary-TOLERANT,
spelling-PRESERVING infix reader. It passes an unknown function through as a bare
leaf (`sqrt(x0)` -> `['sqrt', 'x0']`), which every conversion-trio entry rejects,
and it does NOT canonicalise (`x0+x0` stays `+ x0 x0`; since the
conversion/simplification split `to_prefix` preserves the spelling too, so the
canonicalising contrast is with `simplify`). That capability is load-bearing downstream -- 46.6% of the curated
symbolic-data corpus and 26/120 FastSRB expressions only read through it -- which
is why the earlier `parse` REMOVAL was overturned and a rename ordered instead.

`parse` therefore survives as a deprecated alias that must keep working byte for
byte, `mask_numbers=` included (its removal is separately blocked).
"""

import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


class TestReadInfixExists:
    def test_read_infix_is_the_name(self, engine: SimpliPyEngine) -> None:
        assert hasattr(engine, 'read_infix'), 'engine declares no read_infix'

    def test_read_infix_reads_infix(self, engine: SimpliPyEngine) -> None:
        assert engine.read_infix('2*atan(x0)/3') == \
            ['/', '*', '2', 'atan', 'x0', '3']


class TestTheContractTheNameStates:
    """The three claims the docstring makes, each pinned by a test."""

    def test_tolerates_unknown_vocabulary(self, engine: SimpliPyEngine) -> None:
        """`sqrt` is not in the engine's operator table; it survives as a leaf."""
        assert 'sqrt' not in engine.operator_arity
        assert engine.read_infix('sqrt(x0)') == ['sqrt', 'x0']

    def test_the_conversion_trio_refuses_what_read_infix_accepts(
            self, engine: SimpliPyEngine) -> None:
        """The capability is exclusive to this reader -- that is why it stays."""
        for convert in (engine.to_prefix, engine.to_tagged, engine.to_infix):
            with pytest.raises(ValueError):
                convert('sqrt(x0)')

    def test_preserves_spelling_does_not_canonicalise(
            self, engine: SimpliPyEngine) -> None:
        """The contrast the docstring is required to state, verbatim.

        Since the conversion/simplification split the CONVERSIONS are
        spelling-preserving too (they are notation, not content), so the contrast
        `read_infix` draws is now with `simplify` -- the one entry that canonicalises."""
        assert engine.read_infix('x0+x0') == ['+', 'x0', 'x0']
        assert engine.to_prefix('x0+x0') == ['+', 'x0', 'x0']
        assert engine.simplify(engine.read_infix('x0+x0')) == ['*', '2', 'x0']

    def test_docstring_states_the_contract(self, engine: SimpliPyEngine) -> None:
        doc = type(engine).read_infix.__doc__ or ''
        low = doc.lower()
        assert 'tolerant' in low, 'docstring does not state vocabulary tolerance'
        assert 'spelling' in low, 'docstring does not state spelling preservation'
        assert 'canonical' in low, 'docstring does not state the non-canonicalisation'
        assert 'to_prefix' in doc, 'docstring does not contrast with to_prefix'
        assert 'simplify' in doc, 'docstring does not contrast with simplify'


class TestImplicitMultiplication:
    """Infix reads implicit products (owner ruling 2026-10-05); token lists, which are expected
    well formed, do not. The inserted `*` is an ordinary `*`, with its precedence."""

    @pytest.mark.parametrize("implicit, explicit", [
        ("2x1", "2*x1"),
        ("0x1", "0*x1"),
        ("0x10", "0*x10"),
        ("2(x1 + 1)", "2*(x1 + 1)"),
        ("(x1 + 1)(x2 - 1)", "(x1 + 1)*(x2 - 1)"),
        ("2pi", "2*pi"),
        ("3e2x1", "3e2*x1"),
        ("2sin(x1)", "2*sin(x1)"),
        ("1/2x1", "(1/2)*x1"),
        ("2^3x1", "(2^3)*x1"),
        ("2\tx1", "2*x1"),
    ])
    def test_implicit_equals_explicit(self, engine: SimpliPyEngine, implicit: str, explicit: str) -> None:
        # the structure, not only the simplified value (`0*x1` and `0` simplify alike)
        assert engine.read_infix(implicit) == engine.read_infix(explicit)
        assert engine.simplify(implicit) == engine.simplify(explicit)

    def test_a_name_before_a_paren_is_a_call(self, engine: SimpliPyEngine) -> None:
        # `read_infix` passes an unknown function through as a bare leaf; a name never
        # starts an implicit product, known or not.
        assert engine.read_infix('sqrt(x0)') == ['sqrt', 'x0']
        assert engine.read_infix('sin(x0)') == ['sin', 'x0']

    @pytest.mark.parametrize("text", ["1_000", "2_0", "3.14_15"])
    def test_digit_grouping_is_no_product(self, engine: SimpliPyEngine, text: str) -> None:
        # Python reads `1_000` as 1000; a product with the placeholder name `_000` would be
        # a second, silent reading, so the input stays malformed as before.
        assert '*' not in engine.read_infix(text)
        with pytest.raises(ValueError):
            engine.simplify(text)

    def test_token_lists_stay_strict(self, engine: SimpliPyEngine) -> None:
        with pytest.raises(ValueError, match="reserved numeric spelling"):
            engine.simplify(['*', '2x1', 'x2'])
