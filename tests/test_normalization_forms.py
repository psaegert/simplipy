"""Acceptance tests for the normalization rename + all-forms intake.

Owner ruling 2026-08-18: `normalize_skeleton` -> `to_skeleton`,
`normalize_expression` -> `to_expression`. The name must state the OUTPUT form,
matching the engine's `to_infix`/`to_prefix`/`to_tagged`; the old names hid a
load-bearing behaviour (a skeleton masks EVERY numeric literal, `pow` exponents
and `rootn` indices included). `normalize_variable_token` keeps its name: it is a
single-TOKEN helper returning `(token, is_variable)`, not a form producer.

Follow-up ruling, same day: both take all three external forms (infix `str`,
explicit binary prefix, tagged) and RETURN THE CALLER'S DIALECT -- masking names
an abstraction level, not an output dialect, so it must not silently re-spell.

And the correction that decides the mechanism (audit the research harness):
skeleton-ness is NOT a free axis over dialect. Positional masking of the caller's
own spelling gives a DIFFERENT NUMBER of `<constant>` slots depending on which
dialect the same expression arrived in, in both directions --

    -x0*x1   explicit `neg * x0 x1`          -> 0 <constant>  (the sign is structure)
             tagged   `<mul> -1 x0 x1 </mul>` -> 1 <constant>  (the bag folds the sign
                                                                into a -1 member)
    -1/x0    explicit `/ -1 x0`               -> 1 <constant>
             tagged   `neg inv x0`            -> 0 <constant>

-- measured at 35/400 rows (8.8%) of the repo corpus for the pre-rename
positional pass. Since `to_skeleton` output is what symbolic-data's holdout and
decontamination key on, a dialect-dependent answer means a tagged-era pipeline
silently fails to match explicit-era keys. So the CONTENT must be
dialect-invariant: every input is canonicalised into the ONE internal AC state,
masked there, and only then rendered back into the caller's dialect.
"""
import json
import os
import warnings

import numpy as np
import pytest

from simplipy import SimpliPyEngine, normalization
from simplipy.normalization import to_expression, to_skeleton
from conftest import acj_config_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(REPO, 'benchmarks', 'corpus', 'raw_skeletons_nv.json')


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


# One value, spelled in each of the three external forms (the conversion API's
# own fixture, so the two surfaces are pinned against the same bytes).
INFIX = '2*atan(x0)/3'
PREFIX = ['/', '*', '2', 'atan', 'x0', '3']
TAGGED = ['<mul>', '2', 'atan', 'x0', '<div>', '3', '</mul>']

# The shapes the audit measured: the sign/reciprocal families where the two token
# dialects disagree about how many literals the expression even HAS.
DIVERGENT_SHAPES = ['-x0*x1', '-1/x0']

BATTERY = DIVERGENT_SHAPES + [
    'x0 + x0',
    '(x0 + 1)**2',
    'x1 - sin(x0)',
    '2*atan(x0)/3',
    'rootn(x0, 3)',
    '<constant>*x0',
    '-x0/3',
]




class TestAllThreeInputForms:
    """Three forms in; the SAME dialect out (str->str, prefix->prefix,
    tagged->tagged). Detection is the conversion API's (design D2): type first,
    then a liberal read of the token sequence."""

    @pytest.mark.parametrize("fn", [to_skeleton, to_expression])
    def test_infix_in_infix_out(self, engine: SimpliPyEngine, fn) -> None:
        out = fn(INFIX, engine)
        assert isinstance(out, str)

    @pytest.mark.parametrize("fn", [to_skeleton, to_expression])
    def test_prefix_in_prefix_out(self, engine: SimpliPyEngine, fn) -> None:
        out = fn(PREFIX, engine)
        assert isinstance(out, list)
        assert not any(t in ('<mul>', '<add>', '<div>', '<sub>') for t in out)

    @pytest.mark.parametrize("fn", [to_skeleton, to_expression])
    def test_tagged_in_tagged_out(self, engine: SimpliPyEngine, fn) -> None:
        out = fn(TAGGED, engine)
        assert isinstance(out, list)
        assert any(t in ('<mul>', '<add>') for t in out)

    def test_the_nine_combinations_are_the_diagonal(self, engine: SimpliPyEngine) -> None:
        """The dialect is never re-spelled: each input form yields its own form,
        and the off-diagonal is one conversion call away."""
        assert to_skeleton(INFIX, engine) == engine.to_infix(to_skeleton(PREFIX, engine))
        assert to_skeleton(TAGGED, engine) == engine.to_tagged(to_skeleton(PREFIX, engine))
        assert to_skeleton(PREFIX, engine) == engine.to_prefix(to_skeleton(TAGGED, engine))

    def test_tuple_input_returns_a_list(self, engine: SimpliPyEngine) -> None:
        assert to_skeleton(tuple(PREFIX), engine) == to_skeleton(PREFIX, engine)
        assert isinstance(to_skeleton(tuple(PREFIX), engine), list)

    def test_ndarray_input_returns_a_list(self, engine: SimpliPyEngine) -> None:
        out = to_skeleton(np.array(PREFIX), engine)
        assert isinstance(out, list)
        assert out == to_skeleton(PREFIX, engine)

    def test_variables_are_canonicalised_in_every_form(self, engine: SimpliPyEngine) -> None:
        assert to_expression(['+', 'V1', '2.5'], engine) == to_expression(['+', 'x1', '2.5'], engine)
        assert to_expression('sin(v2)', engine) == to_expression('sin(x2)', engine)

    def test_a_rename_that_moves_the_canonical_order_is_re_sorted(self, engine: SimpliPyEngine) -> None:
        """`* v9 x1` renames to `* x9 x1`, which is no longer canonically
        ordered; the state is rebuilt from the renamed spelling."""
        assert to_expression(['*', 'v9', 'x1'], engine) == ['*', 'x1', 'x9']


class TestDialectInvariance:
    """THE ruling: the same expression yields the same skeleton no matter which
    dialect it arrived in. Compared in one common dialect, because the three
    renderings of one state legitimately SPELL signs and reciprocals differently
    (`neg * x0 x1` and `<mul> -1 x0 x1 </mul>` are one state)."""

    @pytest.mark.parametrize("expr", BATTERY)
    def test_skeleton_content_does_not_depend_on_the_input_dialect(
            self, engine: SimpliPyEngine, expr: str) -> None:
        prefix, tagged = engine.to_prefix(expr), engine.to_tagged(expr)
        from_infix = engine.to_prefix(to_skeleton(expr, engine))
        from_prefix = engine.to_prefix(to_skeleton(prefix, engine))
        from_tagged = engine.to_prefix(to_skeleton(tagged, engine))
        assert from_infix == from_prefix == from_tagged

    @pytest.mark.parametrize("expr", BATTERY)
    def test_expression_content_does_not_depend_on_the_input_dialect(
            self, engine: SimpliPyEngine, expr: str) -> None:
        prefix, tagged = engine.to_prefix(expr), engine.to_tagged(expr)
        assert (engine.to_prefix(to_expression(expr, engine))
                == engine.to_prefix(to_expression(prefix, engine))
                == engine.to_prefix(to_expression(tagged, engine)))

    #: The audit's ACTUAL divergent spellings, given literally.
    #:
    #: They used to be produced by re-spelling a source string through
    #: `engine.to_tagged`, and since the pure-conversion split that no longer emits them
    #: -- conversion became notation-only, so both dialects agreed and the fixture stopped
    #: producing the case. Measured: with the canonicalisation removed entirely, all 20
    #: parametrised instances stayed GREEN. Feeding the pairs literally restores the test.
    DIVERGENT_PAIRS = [
        (['neg', '*', 'x0', 'x1'], ['<mul>', '-1', 'x0', 'x1', '</mul>']),
        (['/', '-1', 'x0'], ['neg', 'inv', 'x0']),
    ]

    @pytest.mark.parametrize("left,right", DIVERGENT_PAIRS)
    def test_the_constant_count_is_the_same_from_every_dialect(
            self, engine: SimpliPyEngine, left: list, right: list) -> None:
        """The audit's failing shapes, stated as the quantity that actually
        breaks downstream: how many fitted parameters the skeleton claims."""
        counts = {tuple(engine.to_prefix(to_skeleton(form, engine))).count('<constant>')
                  for form in (left, right)}
        assert len(counts) == 1, f'{left} vs {right}: dialect-dependent count {counts}'

    @pytest.mark.parametrize("left,right", DIVERGENT_PAIRS)
    def test_the_two_spellings_reach_one_state(
            self, engine: SimpliPyEngine, left: list, right: list) -> None:
        """...and the skeletons agree once both are read into the same dialect. This is
        the property the canonicalisation exists for; removing it must turn this red."""
        a = engine.simplify(engine.to_prefix(to_skeleton(left, engine)))
        b = engine.simplify(engine.to_prefix(to_skeleton(right, engine)))
        assert list(a) == list(b), f'{left} -> {a} vs {right} -> {b}'

    def test_the_corpus_agrees_across_dialects(self, engine: SimpliPyEngine) -> None:
        if not os.path.exists(CORPUS):
            pytest.skip('corpus not present')
        with open(CORPUS) as fh:
            corpus = json.load(fh)
        disagreeing = []
        for tokens in corpus[:120]:
            prefix, tagged = engine.to_prefix(tokens), engine.to_tagged(tokens)
            # The two skeletons are canonical in DIFFERENT dialects, whose canonical
            # emitters carry different sign/inverse/literal doctrine; a pure conversion
            # between them is notation, not a re-canonicalisation. The invariance claim
            # is about the STATE, so re-simplify the converted one.
            from_tagged = engine.simplify(engine.to_prefix(to_skeleton(tagged, engine)))
            if list(from_tagged) != to_skeleton(prefix, engine):
                disagreeing.append(tokens)
        assert not disagreeing, f'{len(disagreeing)} rows still dialect-dependent'


class TestSkeletonMasksEveryNumeric:
    """The behaviour the old name hid: EVERY numeric literal, exponents and root
    indices included."""

    def test_pow_exponents_are_masked(self, engine: SimpliPyEngine) -> None:
        assert to_skeleton(['pow', 'x0', '3'], engine) == ['pow', 'x0', '<constant>']

    def test_rootn_indices_are_masked(self, engine: SimpliPyEngine) -> None:
        assert to_skeleton(['rootn', 'x0', '3'], engine) == ['rootn', 'x0', '<constant>']

    def test_the_expression_form_keeps_them(self, engine: SimpliPyEngine) -> None:
        assert to_expression(['pow', 'x0', '3'], engine) == ['pow', 'x0', '3']

    def test_the_placeholder_spellings_fold(self, engine: SimpliPyEngine) -> None:
        """`<c>` and `<constant>` are two spellings of ONE placeholder."""
        assert to_expression(['+', 'x1', '<c>'], engine) == to_expression(['+', 'x1', '<constant>'], engine)


class TestComposition:
    """`to_skeleton` is NOT a third masking path: it is variable/placeholder
    canonicalisation (`to_expression`) followed by the ratified front door
    `engine.mask(..., policy='all')`."""

    @pytest.mark.parametrize("expr", BATTERY)
    def test_skeleton_is_expression_then_mask_all(self, engine: SimpliPyEngine, expr: str) -> None:
        canonical = to_expression(engine.to_prefix(expr), engine)
        assert to_skeleton(engine.to_prefix(expr), engine) == engine.mask(canonical, 'all')


class TestRoundTrips:
    @pytest.mark.parametrize("expr", BATTERY)
    def test_expression_is_idempotent(self, engine: SimpliPyEngine, expr: str) -> None:
        once = to_expression(expr, engine)
        assert to_expression(once, engine) == once

    @pytest.mark.parametrize("expr", BATTERY)
    def test_skeleton_of_the_expression_form_is_the_skeleton(
            self, engine: SimpliPyEngine, expr: str) -> None:
        assert to_skeleton(to_expression(expr, engine), engine) == to_skeleton(expr, engine)


class TestErrorTaxonomy:
    def test_none_maps_to_none(self, engine: SimpliPyEngine) -> None:
        assert to_skeleton(None, engine) is None
        assert to_expression(None, engine) is None

    def test_empty_maps_to_empty(self, engine: SimpliPyEngine) -> None:
        assert to_skeleton([], engine) == []
        assert to_expression([], engine) == []

    @pytest.mark.parametrize("bad", [3, 3.5, {'x': 1}, None.__class__])
    def test_a_non_expression_type_raises_type_error(self, engine: SimpliPyEngine, bad) -> None:
        with pytest.raises(TypeError):
            to_skeleton(bad, engine)

    def test_the_engine_is_required(self) -> None:
        """The canonicalisation is what makes the answer dialect-invariant, so
        there is no engine-free path: a missing engine is a loud TypeError, never
        a silent fall-back to the positional pass."""
        with pytest.raises(TypeError):
            to_skeleton(PREFIX)  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad", [['+', 'x0'], ['add', 'v1', '3.14'], ['+', 'x0', 'inf'],
                                     ['*', '1_000', 'x0']])
    def test_malformed_or_refused_vocabulary_raises_value_error(
            self, engine: SimpliPyEngine, bad: list) -> None:
        with pytest.raises(ValueError):
            to_skeleton(bad, engine)
