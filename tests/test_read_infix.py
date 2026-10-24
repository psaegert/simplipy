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

    def test_the_strict_entries_name_the_unknown_function(
            self, engine: SimpliPyEngine) -> None:
        """The refusal names the call it cannot read, in every strict entry that reads
        infix -- not a generic "malformed prefix expression" about text that is infix."""
        for entry in (engine.simplify, engine.complexity, engine.to_prefix, engine.to_tagged,
                      engine.to_infix, engine.evaluate_constants):
            with pytest.raises(ValueError, match=r"unknown function 'sqrt' in 'sqrt\(x1\^2\)'"):
                entry('sqrt(x1^2)')
        with pytest.raises(ValueError, match=r"unknown functions 'foo', 'x1' in"):
            engine.simplify('foo(x2) + x1(x3) + foo(1)')
        with pytest.raises(ValueError, match=r"unknown function 'sqrt' in"):
            engine.mask('sqrt(x1) + 2')
        # declared names, aliases and plain products are not calls of unknown functions:
        # the strict reading of such text is the tolerant one
        text = 'sin(x1) + pow(x1, 2) + 2(x1 + 1) + (x1)(x2)'
        assert engine._core.parse(text, strict=True) == engine.read_infix(text)
        # the tolerant reader keeps the call as written
        assert engine._core.parse('sqrt(x1)', strict=False) == ['sqrt', 'x1']

    def test_legacy_spellings_are_read_where_they_are_called(
            self, engine: SimpliPyEngine) -> None:
        """`pow_1(a)` and `pow1(a)` are the legacy reciprocal and identity; a bare `pow_1` is a
        symbol, as `pow_2` is (review F7: it read as `* x0 inv`, an operator with no operand)."""
        assert engine.read_infix('pow_1(x0)') == ['inv', 'x0']
        assert engine.read_infix('x1*pow1(x0)') == ['*', 'x1', 'x0']
        assert engine.read_infix('x0*pow_1') == ['*', 'x0', 'pow_1']
        assert engine.read_infix('x0*pow_2') == ['*', 'x0', 'pow_2']
        assert engine.simplify('x0*pow_1') == engine.simplify('x0*pow_2').replace('pow_2', 'pow_1')

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


class TestWhitespace:
    """Whitespace separates tokens (owner ruling 2026-10-05). A declared one-argument function
    without parentheses applies to the operand after it, taking powers and signs but not
    products, quotients or sums; `* *` and a spaced exponent part still join; any other two
    operands with only whitespace between them are a user error."""

    @pytest.mark.parametrize("spaced, explicit", [
        ("sin x0^2", "sin(x0^2)"),
        ("log x0 / 2", "log(x0)/2"),
        ("x0 * * 2", "x0**2"),
        ("1 e-5", "1e-5"),
        ("1e -5", "1*e - 5"),
        ("1 e-5x0", "1e-5*x0"),
        ("1 e5x0", "1e5*x0"),
        ("2 e+1x0", "2e+1*x0"),
        ("sin x0", "sin(x0)"),
        ("sin x0 + 1", "sin(x0) + 1"),
        ("sin x0 * x1", "sin(x0)*x1"),
        ("exp -x0^2 / 2", "exp(-x0^2)/2"),
        ("-sin x0", "-sin(x0)"),
        ("sin -x0", "sin(-x0)"),
        ("sin - x0", "sin(-x0)"),
        ("sin-x0", "sin(-x0)"),
        ("sin cos x0", "sin(cos(x0))"),
        ("x0^sin x1", "x0^sin(x1)"),
        ("2 sin x0", "2*sin(x0)"),
        ("sin\tx0^2", "sin(x0^2)"),
        # unchanged: a name before a parenthesis is a call; products as before
        ("sin (x0)^2", "sin(x0)^2"),
        ("sqrt (x0)", "sqrt(x0)"),
        ("2 x0", "2*x0"),
        ("2 (x0 + 1)", "2*(x0 + 1)"),
        ("(x0) (x1)", "(x0)*(x1)"),
        ("(x0) 2", "(x0)*2"),
        ("x0 + x1", "x0+x1"),
    ])
    def test_spaced_equals_explicit(self, engine: SimpliPyEngine, spaced: str, explicit: str) -> None:
        assert engine.read_infix(spaced) == engine.read_infix(explicit)

    @pytest.mark.parametrize("text", [
        "sin 2x0", "sin 2 x0", "sin 2(x0 + 1)", "sin x0^2 cos x0", "cos 2 pi",
        # a function without parentheses inside another one's argument keeps the outer one open
        "sin -(cos x0) 2", "exp abs((exp x2)) pi", "sin x0^(cos x1) 2",
    ])
    def test_a_product_inside_an_argument_without_parentheses_is_refused(
            self, engine: SimpliPyEngine, text: str) -> None:
        # sin 2x0 is sin(2x0) in a textbook and sin(2)*x0 by the precedence of `*`: ambiguous
        with pytest.raises(ValueError, match="ambiguous"):
            engine.read_infix(text)

    def test_a_product_outside_the_argument_stands(self, engine: SimpliPyEngine) -> None:
        assert engine.read_infix('2 sin x0') == engine.read_infix('2*sin(x0)')
        assert engine.read_infix('sin x0 + 2x1') == engine.read_infix('sin(x0) + 2*x1')
        assert engine.read_infix('sin(2x0)') == engine.read_infix('sin(2*x0)')

    def test_every_entry_point_that_reads_infix_refuses(self, engine: SimpliPyEngine) -> None:
        for read in (engine.to_prefix, engine.to_infix, engine.to_tagged, engine.complexity, engine.mask):
            with pytest.raises(ValueError, match="separated only by whitespace"):
                read('x0 x1')
        assert engine.is_valid('x0 x1') is False

    def test_euler_after_a_trailing_e(self, engine: SimpliPyEngine) -> None:
        # `1e` is no numeral: it is 1*e, so `1e -5` is e - 5
        assert engine.simplify('1e -5') == engine.simplify('e - 5')

    @pytest.mark.parametrize("text", ["1 $ e5", "1 $ e-5", "3 \u00d7 e-2", "2 \u00b7 e+1"])
    def test_a_dropped_character_never_fuses_a_number(self, engine: SimpliPyEngine, text: str) -> None:
        # the exponent join is for whitespace alone; a character the tokenizer drops keeps the
        # input malformed, as on main
        with pytest.raises(ValueError):
            engine.simplify(text)

    @pytest.mark.parametrize("text", ["x0 x1", "x 1", "2 3", "1 000", "sin x0 x1", "sqrt x0", "pi x0"])
    def test_two_operands_with_only_whitespace_between_are_refused(
            self, engine: SimpliPyEngine, text: str) -> None:
        for read in (engine.read_infix, engine.infix_to_prefix, engine.simplify):
            with pytest.raises(ValueError, match="separated only by whitespace"):
                read(text)


class TestCallArgumentCounts:
    """A call gives its operator exactly its arity in arguments. `max` and `min` take any
    number of at least two, nested to the right, because sympy prints them n-ary
    (`Max(x0, x1, x2)`); every other count is refused by name, never read into a malformed
    prefix list."""

    @pytest.fixture(scope="class")
    def minmax_engine(self) -> SimpliPyEngine:
        import warnings

        from simplipy.io import load_config
        ops = dict(load_config(acj_config_path())['operators'])
        for name in ('max', 'min'):
            ops[name] = {'realization': f'simplipy.operators.{name}', 'alias': [name.capitalize()],
                         'arity': 2, 'commutative': False}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # the engine has no rules, and says so
            return SimpliPyEngine(operators=ops, rules=[])

    def test_max_and_min_nest_to_the_right(self, minmax_engine: SimpliPyEngine) -> None:
        e = minmax_engine
        assert e.read_infix('Max(x0, x1, x2)') == ['max', 'x0', 'max', 'x1', 'x2']
        assert e.read_infix('Max(Min(x0, x1), Min(Max(x0, x1), x2))') == \
            ['max', 'min', 'x0', 'x1', 'min', 'max', 'x0', 'x1', 'x2']
        assert e.read_infix('min(x0, 2, -x1, x2)') == ['min', 'x0', 'min', '2', 'min', 'neg', 'x1', 'x2']
        assert e.to_prefix('Max(x0, x1, x2)') == e.to_prefix('max(x0, max(x1, x2))')

    def test_a_right_nested_chain_prints_as_one_call(self, minmax_engine: SimpliPyEngine) -> None:
        """The engine reads every nesting of `max` as one bag, and the printer writes the
        right-nested chain it answers with as one call, which reads back to the same tokens."""
        e = minmax_engine
        assert e.simplify('Max(x2, Max(x0, x1))') == 'max(x0, x1, x2)'
        assert e.prefix_to_infix(['max', 'x0', 'max', 'x1', 'x2']) == 'max(x0, x1, x2)'
        assert e.read_infix('max(x0, x1, x2)') == ['max', 'x0', 'max', 'x1', 'x2']
        # a left nesting is no chain to the reader, so it keeps its call
        assert e.prefix_to_infix(['max', 'max', 'x0', 'x1', 'x2']) == 'max(max(x0, x1), x2)'
        # a realization is a binary function
        assert e.prefix_to_infix(['max', 'x0', 'max', 'x1', 'x2'], realization=True) == \
            'simplipy.operators.max(x0, simplipy.operators.max(x1, x2))'

    @pytest.mark.parametrize('text, message', [
        ('sin(x0, x1)', 'sin takes 1 argument, got 2'),
        ('pow(x0)', 'pow takes 2 arguments, got 1'),
        ('pow(x0, x1, x2)', 'pow takes 2 arguments, got 3'),
        ('max(x0)', 'max takes 2 arguments, got 1'),
        ('(x0, x1)', "a comma outside a call's argument list"),
        ('x0 + (x1, x2)', "a comma outside a call's argument list"),
    ])
    def test_every_other_count_is_refused(self, minmax_engine: SimpliPyEngine, text: str, message: str) -> None:
        import re
        for read in (minmax_engine.read_infix, minmax_engine.infix_to_prefix, minmax_engine.simplify):
            with pytest.raises(ValueError, match=re.escape(message)):
                read(text)

    def test_an_undeclared_call_still_reads_tolerantly(self, engine: SimpliPyEngine) -> None:
        assert engine.read_infix('foo(x0, x1)') == ['foo', 'x0', 'x1']
