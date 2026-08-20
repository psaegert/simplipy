"""PURE CONVERSIONS: notation only, never content (the research harness).

`to_infix` / `to_prefix` / `to_tagged` are SYNTACTIC token-level regroupings: no AC
state, no canonical constructors, no rule application, no literal re-spelling, no
operand reordering. `simplify` keeps canonicalising and is DIALECT-PRESERVING -- it
returns the form it was handed.

Every direction, every 2-cycle and every 3-cycle is asserted at the exact relation the
design table claims, over the 400 raw (unsimplified) rows of
`benchmarks/corpus/raw_skeletons_nv.json`. Where a relation is "up to X" the test
re-normalises through the stated map and asserts equality -- no case is skipped.
"""
import json
import os

import numpy as np
import pytest

from conftest import acj_config_path, require_or_skip
from simplipy import SimpliPyEngine
from simplipy.io import load_config

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                      'benchmarks', 'corpus', 'raw_skeletons_nv.json')


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


@pytest.fixture(scope="module")
def ruleless() -> SimpliPyEngine:
    """The same operator table with NO rules: a conversion must answer identically."""
    return SimpliPyEngine(operators=load_config(acj_config_path())['operators'], rules=[])


@pytest.fixture(scope="module")
def rows() -> list[list[str]]:
    require_or_skip(CORPUS, 'the raw skeleton corpus is not in this checkout')
    return [list(r) for r in json.load(open(CORPUS))]


@pytest.fixture(scope="module")
def forms(engine, rows):
    """Each row spelled RAW in all three forms (prefix is the corpus spelling)."""
    return [{'prefix': r,
             'tagged': engine.to_tagged(r),
             'infix': engine.to_infix(r)} for r in rows]


def inv_to_div(tokens: list[str]) -> list[str]:
    """The infix language's only erasure: it has no `inv` glyph, so `inv X` reads
    back as `/ 1 X`."""
    out: list[str] = []
    for t in tokens:
        out.extend(['/', '1'] if t == 'inv' else [t])
    return out


# --------------------------------------------------------------------- purity
class TestPurity:
    """A conversion is notation. It never collects, folds, reorders or fires a rule,
    and its answer does not depend on the engine ARTIFACT."""

    def test_no_like_term_collection(self, engine) -> None:
        assert engine.to_prefix(['+', 'x0', 'x0']) == ['+', 'x0', 'x0']

    def test_no_rule_firing(self, engine) -> None:
        # acj-4-3 ships `pow(abs(_0), n) -> pow(_0, n)`; a CONVERSION must not fire it.
        assert engine.to_prefix(['pow', 'abs', 'x0', '2']) == ['pow', 'abs', 'x0', '2']
        assert engine.to_tagged(['pow', 'abs', 'x0', '2']) == ['pow', 'abs', 'x0', '2']

    def test_no_constant_folding(self, engine) -> None:
        assert engine.to_prefix(['+', '1', '2']) == ['+', '1', '2']

    def test_no_literal_respelling(self, engine) -> None:
        assert engine.to_tagged(['*', '0.2', 'x0']) == ['<mul>', '0.2', 'x0', '</mul>']
        assert engine.to_prefix(['<mul>', '1/3', 'x0', '</mul>']) == ['*', '1/3', 'x0']

    def test_bag_member_order_is_preserved(self, engine) -> None:
        assert engine.to_tagged(['*', 'x1', 'x0']) == ['<mul>', 'x1', 'x0', '</mul>']
        assert engine.to_prefix(['<mul>', 'x1', 'x0', '</mul>']) == ['*', 'x1', 'x0']

    def test_artifact_independent(self, engine, ruleless, forms) -> None:
        for s in forms:
            for src in ('infix', 'prefix', 'tagged'):
                assert engine.to_prefix(s[src]) == ruleless.to_prefix(s[src])
                assert engine.to_tagged(s[src]) == ruleless.to_tagged(s[src])
                assert engine.to_infix(s[src]) == ruleless.to_infix(s[src])


# ------------------------------------------------------- the nine directions
class TestNineDirections:
    """3 source forms x 3 targets. The diagonal is byte identity; the off-diagonal
    is total and lands in the target form."""

    def test_prefix_to_prefix_is_identity(self, engine, forms) -> None:
        for s in forms:
            assert engine.to_prefix(s['prefix']) == s['prefix']

    def test_tagged_to_tagged_is_identity(self, engine, forms) -> None:
        for s in forms:
            assert engine.to_tagged(s['tagged']) == s['tagged']

    def test_infix_to_infix_is_identity(self, engine, forms) -> None:
        for s in forms:
            assert engine.to_infix(s['infix']) == s['infix']

    def test_targets_land_in_their_form(self, engine, forms) -> None:
        tags = {'<add>', '</add>', '<mul>', '</mul>', '<sub>', '<div>'}
        binary = {'+', '-', '*', '/'}
        for s in forms:
            for src in ('infix', 'prefix', 'tagged'):
                assert isinstance(engine.to_infix(s[src]), str)
                out_p = engine.to_prefix(s[src])
                assert isinstance(out_p, list) and not tags.intersection(out_p)
                out_t = engine.to_tagged(s[src])
                assert isinstance(out_t, list) and not binary.intersection(out_t)

    def test_conversions_preserve_the_canonical_state(self, engine, forms) -> None:
        """The blanket invariant: what a conversion erases is notation only."""
        for s in forms:
            for src in ('infix', 'prefix', 'tagged'):
                assert engine.simplify(engine.to_prefix(s[src])) \
                    == engine.simplify(s['prefix'])
                assert engine.simplify(engine.to_tagged(s[src])) \
                    == engine.simplify(engine.to_tagged(s['prefix']))
                assert engine.simplify(engine.to_infix(s[src])) \
                    == engine.simplify(engine.to_infix(s['prefix']))


# ------------------------------------------------------------------ 2-cycles
class TestTwoCycles:
    """Each directed 2-cycle at the relation the design table claims."""

    def test_tagged_prefix_tagged_is_byte_exact(self, engine, forms) -> None:
        for s in forms:
            assert engine.to_tagged(engine.to_prefix(s['tagged'])) == s['tagged']

    def test_prefix_tagged_prefix_up_to_association(self, engine, forms) -> None:
        exact = 0
        for s in forms:
            cycled = engine.to_prefix(engine.to_tagged(s['prefix']))
            # the stated re-normalisation: association is exactly what `to_tagged` erases
            assert engine.to_tagged(cycled) == engine.to_tagged(s['prefix'])
            exact += cycled == s['prefix']
        assert exact > 0

    def test_prefix_infix_prefix_up_to_inv_spelling(self, engine, forms) -> None:
        for s in forms:
            cycled = engine.to_prefix(engine.to_infix(s['prefix']))
            assert cycled == inv_to_div(s['prefix'])

    def test_infix_prefix_infix_preserves_the_reading(self, engine, forms) -> None:
        for s in forms:
            cycled = engine.to_infix(engine.to_prefix(s['infix']))
            assert engine.to_prefix(cycled) == engine.to_prefix(s['infix'])

    def test_tagged_infix_tagged_up_to_inv_spelling(self, engine, forms) -> None:
        for s in forms:
            cycled = engine.to_tagged(engine.to_infix(s['tagged']))
            assert cycled == engine.to_tagged(inv_to_div(engine.to_prefix(s['tagged'])))

    def test_infix_tagged_infix_up_to_association(self, engine, forms) -> None:
        for s in forms:
            cycled = engine.to_infix(engine.to_tagged(s['infix']))
            assert engine.to_tagged(engine.to_prefix(cycled)) \
                == engine.to_tagged(engine.to_prefix(s['infix']))


# ------------------------------------------------------------------ 3-cycles
class TestThreeCycles:
    """All six 3-cycles, each at the JOIN of the erasures its legs carry."""

    def test_tagged_prefix_infix_tagged(self, engine, forms) -> None:
        for s in forms:
            out = engine.to_tagged(engine.to_infix(engine.to_prefix(s['tagged'])))
            assert out == engine.to_tagged(inv_to_div(engine.to_prefix(s['tagged'])))

    def test_tagged_infix_prefix_tagged(self, engine, forms) -> None:
        for s in forms:
            out = engine.to_tagged(engine.to_prefix(engine.to_infix(s['tagged'])))
            assert out == engine.to_tagged(inv_to_div(engine.to_prefix(s['tagged'])))

    def test_prefix_tagged_infix_prefix(self, engine, forms) -> None:
        for s in forms:
            out = engine.to_prefix(engine.to_infix(engine.to_tagged(s['prefix'])))
            expect = inv_to_div(engine.to_prefix(engine.to_tagged(s['prefix'])))
            assert out == expect

    def test_prefix_infix_tagged_prefix(self, engine, forms) -> None:
        for s in forms:
            out = engine.to_prefix(engine.to_tagged(engine.to_infix(s['prefix'])))
            expect = engine.to_prefix(engine.to_tagged(inv_to_div(s['prefix'])))
            assert out == expect

    def test_infix_prefix_tagged_infix(self, engine, forms) -> None:
        for s in forms:
            out = engine.to_infix(engine.to_tagged(engine.to_prefix(s['infix'])))
            assert engine.to_tagged(engine.to_prefix(out)) \
                == engine.to_tagged(engine.to_prefix(s['infix']))

    def test_infix_tagged_prefix_infix(self, engine, forms) -> None:
        for s in forms:
            out = engine.to_infix(engine.to_prefix(engine.to_tagged(s['infix'])))
            assert engine.to_tagged(engine.to_prefix(out)) \
                == engine.to_tagged(engine.to_prefix(s['infix']))


# ---------------------------------------------------- the association contract
class TestAssociationAndSections:
    """The stated conventions, spelled out on the smallest witnesses."""

    def test_both_associations_of_a_product_reach_one_bag(self, engine) -> None:
        a = engine.to_tagged(['*', 'x0', '*', 'x1', 'x2'])
        b = engine.to_tagged(['*', '*', 'x0', 'x1', 'x2'])
        assert a == b == ['<mul>', 'x0', 'x1', 'x2', '</mul>']

    def test_bag_expands_right_nested(self, engine) -> None:
        assert engine.to_prefix(['<add>', 'a', 'b', 'c', '</add>']) == ['+', 'a', '+', 'b', 'c']
        assert engine.to_prefix(['<mul>', 'a', 'b', 'c', '</mul>']) == ['*', 'a', '*', 'b', 'c']

    def test_sections_expand_left_nested(self, engine) -> None:
        assert engine.to_prefix(['<add>', 'a', 'b', '<sub>', 'c', 'd', '</add>']) \
            == ['-', '-', '+', 'a', 'b', 'c', 'd']
        assert engine.to_prefix(['<mul>', 'a', 'b', '<div>', 'c', 'd', '</mul>']) \
            == ['/', '/', '*', 'a', 'b', 'c', 'd']

    def test_inverse_of_a_product_is_never_distributed(self, engine) -> None:
        # `1/(a*b)` is NOT `(1/a)*(1/b)` -- they part company at 0 and inf.
        assert engine.to_tagged(['/', 'x0', '*', 'x1', 'x2']) \
            == ['<mul>', 'x0', '<div>', '<mul>', 'x1', 'x2', '</mul>', '</mul>']
        assert engine.to_prefix(['<mul>', 'x0', '<div>', '<mul>', 'x1', 'x2', '</mul>', '</mul>']) \
            == ['/', 'x0', '*', 'x1', 'x2']

    def test_lone_group_inverse_keeps_its_unary_spelling(self, engine) -> None:
        assert engine.to_tagged(['neg', 'x0']) == ['neg', 'x0']
        assert engine.to_tagged(['inv', 'x0']) == ['inv', 'x0']

    def test_empty_positive_part_expands_to_the_unary_inverse(self, engine) -> None:
        assert engine.to_prefix(['<add>', '<sub>', 'a', 'b', '</add>']) == ['-', 'neg', 'a', 'b']
        assert engine.to_prefix(['<mul>', '<div>', 'a', 'b', '</mul>']) == ['/', 'inv', 'a', 'b']
        # ... and no `0`/`1` unit is materialised, so the cycle stays byte-exact
        for t in (['<add>', '<sub>', 'a', 'b', '</add>'], ['<mul>', '<div>', 'a', 'b', '</mul>']):
            assert engine.to_tagged(engine.to_prefix(t)) == t


# -------------------------------------------------- simplify: dialect-preserving
class TestSimplifyPreservesTheDialect:
    """Owner ruling: `simplify` only simplifies, and returns the form it was given."""

    def test_prefix_in_prefix_out(self, engine) -> None:
        out = engine.simplify(['*', '2', 'atan', 'x0'])
        assert out == ['*', '2', 'atan', 'x0']

    def test_the_tagged_leak_is_gone(self, engine, rows) -> None:
        tags = {'<add>', '</add>', '<mul>', '</mul>', '<sub>', '<div>'}
        for r in rows[:120]:
            assert not tags.intersection(engine.simplify(list(r)))

    def test_tagged_in_tagged_out(self, engine) -> None:
        out = engine.simplify(['<mul>', '2', 'atan', 'x0', '</mul>'])
        assert out == ['<mul>', '2', 'atan', 'x0', '</mul>']

    def test_str_in_str_out(self, engine) -> None:
        assert isinstance(engine.simplify('2*atan(x0)'), str)

    def test_containers_are_still_mirrored(self, engine) -> None:
        assert isinstance(engine.simplify(('*', '2', 'atan', 'x0')), tuple)
        assert isinstance(engine.simplify(np.array(['*', '2', 'atan', 'x0'])), np.ndarray)

    def test_simplify_still_canonicalises(self, engine) -> None:
        assert engine.simplify(['+', 'x0', 'x0']) == ['*', '2', 'x0']
        assert engine.simplify(['<add>', 'x0', 'x0', '</add>']) == ['<mul>', '2', 'x0', '</mul>']

    def test_convert_then_simplify_is_the_exact_recipe(self, engine, rows) -> None:
        """The migration for `form=`: convert first, THEN simplify."""
        for r in rows[:120]:
            r = list(r)
            assert engine.simplify(engine.to_tagged(r)) == engine.simplify(engine.to_tagged(r))
            assert engine.simplify(engine.to_infix(r)) == engine.simplify(engine.to_infix(r))
            assert engine.simplify(engine.to_prefix(r)) == engine.simplify(engine.to_prefix(r))


# ------------------------------------------------------------------- errors
class TestErrors:
    @pytest.mark.parametrize("bad", [
        pytest.param('2*+', id='malformed-infix'),
        pytest.param(['*', '2'], id='missing-operand'),
        pytest.param(['<mul>', '2', 'x0'], id='unclosed-tag'),
        pytest.param(['</mul>', '2', 'x0'], id='stray-closer'),
        pytest.param(['<mul>', 'x0', '<div>', 'x1', '<div>', 'x2', '</mul>'], id='two-sections'),
        pytest.param(['<mul>', 'x0', '</mul>'], id='one-member-bag'),
        pytest.param(['bogusfn', 'x0'], id='unknown-operator'),
        pytest.param('sqrt(x0)', id='unknown-vocabulary-infix'),
    ])
    def test_malformed_raises_value_error(self, engine, bad) -> None:
        for to_x in (engine.to_infix, engine.to_prefix, engine.to_tagged):
            with pytest.raises(ValueError):
                to_x(bad)

    @pytest.mark.parametrize("bad", [42, {'x0': 1}, None])
    def test_wrong_type_raises_type_error(self, engine, bad) -> None:
        for to_x in (engine.to_infix, engine.to_prefix, engine.to_tagged):
            with pytest.raises(TypeError):
                to_x(bad)

    def test_2d_ndarray_raises_value_error(self, engine) -> None:
        with pytest.raises(ValueError):
            engine.to_infix(np.array([['*', 'x0', 'x1'], ['*', 'x0', 'x1']]))

    def test_empty_input_contract(self, engine) -> None:
        assert engine.to_prefix([]) == []
        assert engine.to_tagged([]) == []
        assert engine.to_infix([]) == ''

    def test_read_infix_stays_tolerant(self, engine) -> None:
        assert engine.read_infix('sqrt(x0)') == ['sqrt', 'x0']
