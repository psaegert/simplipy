"""`is_valid` accepts and verifies ALL THREE forms (ruling 2026-08-18, evening
batch 2 item 5): "is_valid should support and verify all forms, prefix, infix and
tagged".

This closes a v24 seam. The v24.0 target format is the TAGGED canonical form, so
flash-ansr's decoded beams arrive tagged -- and an `is_valid` that only reads the
explicit binary dialect answered False for every one of them: not "I cannot read
this", but a wrong VERDICT, silently, on the exact form the training target now
uses. An infix `str` was worse: `list('x0+x0')` walks the CHARACTERS, so a
perfectly valid expression came back False.

Form detection follows the conversion trio (D2): the TYPE decides str-vs-tokens,
and for tokens the liberal reader decides dialect. Every pre-existing call --
explicit binary prefix, the only form callers could use -- must answer exactly as
before, so the explicit verdicts below are the recorded pre-change answers.
"""
import numpy as np
import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


# Verdicts recorded on the pre-change tree; not one of them may move.
EXPLICIT_UNCHANGED = [
    (['+', 'x0', 'x0'], True),
    (['*', '2', 'atan', 'x0'], True),
    (['x0'], True),
    (['2'], True),
    (['pow', 'x0', '2'], True),
    (['+', 'x0'], False),                 # too few operands
    (['x0', 'x1'], False),                # two roots
    ([], False),                          # the one valid case is_valid rejects
    (['+', 'x0', 'x0', 'x1'], False),     # trailing token
    (['sqrt', 'x0'], False),              # undeclared vocabulary
]


class TestExplicitPrefixIsUntouched:
    @pytest.mark.parametrize('tokens,expected', EXPLICIT_UNCHANGED,
                             ids=[str(t) for t, _ in EXPLICIT_UNCHANGED])
    def test_recorded_verdicts_do_not_move(
            self, engine: SimpliPyEngine, tokens: list, expected: bool) -> None:
        assert engine.is_valid(tokens) is expected

    def test_tuple_input_keeps_working(self, engine: SimpliPyEngine) -> None:
        assert engine.is_valid(('+', 'x0', 'x0')) is True

    def test_verbose_still_explains(self, engine: SimpliPyEngine, capsys) -> None:
        assert engine.is_valid(['+', 'x0'], verbose=True) is False
        assert capsys.readouterr().out.strip() != ''


class TestTaggedForm:
    """The v24 seam: tagged beams must be VERIFIED, not refused."""

    @pytest.mark.parametrize('tokens', [
        ['<mul>', '2', 'atan', 'x0', '</mul>'],
        ['<add>', 'x0', '<sub>', 'x1', '</add>'],
        ['<mul>', '2', 'x0', '<div>', '3', '</mul>'],
        ['<add>', 'x0', 'x1', '</add>'],
    ], ids=['mul-bag', 'add-with-sub', 'mul-with-div', 'plain-add'])
    def test_well_formed_tagged_is_valid(
            self, engine: SimpliPyEngine, tokens: list) -> None:
        assert engine.is_valid(tokens) is True

    @pytest.mark.parametrize('tokens', [
        ['<add>', 'x0'],                       # unclosed
        ['<mul>', 'x0', '</add>'],             # mismatched closer
        ['<add>', 'x0', '</add>', 'x1'],       # trailing token
        ['</add>', 'x0', '<add>'],             # closer first
        ['<mul>', 'atan', '</mul>'],           # operator starved of its operand
    ], ids=['unclosed', 'mismatched', 'trailing', 'inverted', 'starved'])
    def test_malformed_tagged_is_invalid(
            self, engine: SimpliPyEngine, tokens: list) -> None:
        assert engine.is_valid(tokens) is False

    def test_the_engines_own_tagged_output_validates(
            self, engine: SimpliPyEngine) -> None:
        """The strongest form of the seam: what simplify EMITS, is_valid accepts."""
        for expr in ('2*atan(x0)/3', 'x0+x0', 'x1 - sin(x0)', '(x0 + 1)**2'):
            tagged = engine.to_tagged(expr)
            assert engine.is_valid(tagged) is True, tagged


class TestInfixForm:
    @pytest.mark.parametrize('text', [
        'x0+x0', '2*atan(x0)/3', 'x0', '2', '(x0 + 1)**2', '-x0/3',
    ])
    def test_well_formed_infix_is_valid(
            self, engine: SimpliPyEngine, text: str) -> None:
        assert engine.is_valid(text) is True

    @pytest.mark.parametrize('text', [
        'x0+', '', '*', 'x0 x1)', 'atan(', 'x0)',
    ], ids=['dangling-op', 'empty', 'bare-op', 'stray-paren', 'unclosed-call',
            'trailing-paren'])
    def test_malformed_infix_is_invalid(
            self, engine: SimpliPyEngine, text: str) -> None:
        assert engine.is_valid(text) is False

    @pytest.mark.parametrize('text,reads_as', [
        ('((x0', ['x0']),
        ('((x0+x1)', ['+', 'x0', 'x1']),
    ], ids=['dropped-openers', 'one-unmatched-opener'])
    def test_unmatched_OPENING_parens_are_a_reader_tolerance_not_a_verdict(
            self, engine: SimpliPyEngine, text: str, reads_as: list) -> None:
        """PINNED QUIRK, pre-existing and NOT introduced by the all-forms change:
        the shunting-yard reader silently DROPS unmatched opening parentheses, so
        '((x0' reads as 'x0' and is then a perfectly valid expression. is_valid
        reports on what the reader reads -- deliberately, because the alternative
        is a second, partial infix grammar in Python diverging from the one the
        core implements. Unmatched CLOSING parens survive as a stray token and DO
        fail (see 'trailing-paren' above), so the tolerance is one-sided.

        The honest fix belongs in the core's reader; it is out of this lane's
        scope because it would change read_infix/parse output that downstream
        corpora are pinned to."""
        assert engine.read_infix(text) == reads_as
        assert engine.is_valid(text) is True

    def test_infix_is_verified_against_the_vocabulary(
            self, engine: SimpliPyEngine) -> None:
        """is_valid VERIFIES; read_infix TOLERATES. The two must not be confused:
        an undeclared function reads fine and is still not a valid expression for
        THIS engine."""
        assert engine.read_infix('sqrt(x0)') == ['sqrt', 'x0']
        assert engine.is_valid('sqrt(x0)') is False


class TestFormDetectionMatchesTheTrio:
    def test_ndarray_of_tokens_is_read_as_tokens(
            self, engine: SimpliPyEngine) -> None:
        assert engine.is_valid(np.array(['+', 'x0', 'x0'], dtype=object)) is True
        assert engine.is_valid(np.array(['+', 'x0'], dtype=object)) is False

    def test_a_non_expression_type_refuses_loudly(
            self, engine: SimpliPyEngine) -> None:
        with pytest.raises(TypeError):
            engine.is_valid(5)

    def test_all_three_forms_of_one_value_agree(
            self, engine: SimpliPyEngine) -> None:
        """The point of the ruling: one value, three spellings, one verdict."""
        for expr in ('2*atan(x0)/3', 'x0+x0', 'x1 - sin(x0)'):
            assert engine.is_valid(expr) is True
            assert engine.is_valid(engine.to_prefix(expr)) is True
            assert engine.is_valid(engine.to_tagged(expr)) is True
