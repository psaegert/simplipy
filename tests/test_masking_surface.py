"""The masking redundancy, cleared (ruling 2026-08-18, evening batch 2 item 4):
"The masking redundancies should be cleared of course and everything streamlined."

VERDICT: one policy, not two. `mask_values_keep_structure` and `mask_fittable`
are the SAME function object today, and the evidence says that is not an
oversight to differentiate but a rename to finish:

* `mask_values_keep_structure` was the ORIGINAL function (commit aefaa527); the
  rename to `mask_fittable` landed in 6be4229d, whose message records the intent
  verbatim -- "Two shipped kinds: mask_all and mask_fittable
  (mask_values_keep_structure kept as an alias)". The body never differed: both
  are `if role in (EXPONENT, ROOT_INDEX): return None; return "<constant>"`.
* There is no third semantics to implement. `Role` has exactly five members, and
  "mask the values, keep the structure" partitions them as
  {COEFFICIENT, ADDEND, VALUE} vs {EXPONENT, ROOT_INDEX} -- which IS
  `mask_fittable`'s partition. The two names describe one criterion from two
  sides: WHAT it keeps, and WHY (an unfittable literal controls the DOMAIN, not
  the magnitude).
* The module docstring's own "THE KINDS SIMPLIPY SHIPS" already lists exactly
  two kinds.

So `mask_fittable` is THE name and the old one deprecates. Also pinned here: the
declared surface, and the `collect` escape the engine front door was missing.
"""
import warnings

import pytest

from simplipy import SimpliPyEngine, masking
from conftest import acj_config_path


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


class TestDeclaredSurface:
    def test_all_declares_the_live_surface_only(self) -> None:
        """The deprecated spelling is not advertised; the typing helpers the
        module imports were never part of its surface."""
        assert set(masking.__all__) == {
            'Role', 'literal_sites', 'mask', 'mask_all', 'mask_fittable'}

    def test_star_import_binds_exactly_the_declared(self) -> None:
        ns: dict = {}
        exec('from simplipy.masking import *', ns)
        bound = {n for n in ns if not n.startswith('__')}
        assert bound == set(masking.__all__)

    def test_star_import_leaks_no_typing_helpers(self) -> None:
        ns: dict = {}
        exec('from simplipy.masking import *', ns)
        for helper in ('Callable', 'Optional', 'cast', 'TYPE_CHECKING'):
            assert helper not in ns, f'{helper} leaked into the public namespace'

    def test_every_declared_name_resolves(self) -> None:
        for name in masking.__all__:
            assert hasattr(masking, name), f'{name} is declared but missing'


class TestEngineMaskReachesTheToolkit:
    """`engine.mask` had no `collect=False` escape while the toolkit did."""

    EXPR = '2*x0/3'

    def test_collect_false_is_reachable_from_the_front_door(
            self, engine: SimpliPyEngine) -> None:
        tokens = engine.to_tagged(self.EXPR)
        assert engine.mask(tokens, 'all', collect=False) == \
            masking.mask(tokens, engine, masking.mask_all, collect=False)

    def test_collect_default_stays_byte_identical(
            self, engine: SimpliPyEngine) -> None:
        tokens = engine.to_tagged(self.EXPR)
        assert engine.mask(tokens, 'all') == \
            masking.mask(tokens, engine, masking.mask_all)

    def test_collect_is_what_enforces_one_constant_per_dof(
            self, engine: SimpliPyEngine) -> None:
        """The reason the escape matters: 2*x0/3 is ONE degree of freedom that a
        positional pass splits into two."""
        tokens = engine.to_tagged(self.EXPR)
        assert engine.mask(tokens, 'all').count('<constant>') == 1
        assert engine.mask(tokens, 'all', collect=False).count('<constant>') == 2

    def test_the_two_policy_spellings_agree(self, engine: SimpliPyEngine) -> None:
        tokens = engine.to_tagged(self.EXPR)
        assert engine.mask(tokens, 'values') == engine.mask(tokens, 'fittable')

    def test_the_front_door_does_not_route_through_the_deprecated_name(
            self, engine: SimpliPyEngine) -> None:
        tokens = engine.to_tagged(self.EXPR)
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            engine.mask(tokens, 'values')
            engine.mask(tokens, 'fittable')

    def test_docstring_documents_the_collect_stage(
            self, engine: SimpliPyEngine) -> None:
        doc = type(engine).mask.__doc__ or ''
        assert 'collect' in doc, 'the collect stage is undocumented on the front door'
        assert 'order' in doc.lower(), 'the re-ordering of the collect stage is undocumented'


class TestCollectReallyReorders:
    """The documented gap is real, so it is pinned rather than described."""

    def test_collect_moves_terms(self, engine: SimpliPyEngine) -> None:
        # the CANONICAL tagged state: `to_tagged` is a pure conversion since the split,
        # so the canonical ORDER this test is about comes from `simplify`.
        tokens = engine.simplify(engine.to_tagged('sin(x0) + 2*x1'))
        collected = engine.mask(tokens, 'all')
        positional = engine.mask(tokens, 'all', collect=False)
        assert collected != positional
        assert collected.index('sin') < positional.index('sin')
