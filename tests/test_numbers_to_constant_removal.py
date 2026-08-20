"""`simplipy.utils.numbers_to_constant` is REMOVED in 0.14.0.

Owner ruling 2026-08-18, verbatim: "I'd like to have the removal in 0.14.0
already and the downstream packages to adapt." The 0.13.0 CHANGELOG and
`docs/compatibility.md` already stated the removal version; this suite lands it
and pins the boundary of WHAT GOES against WHAT STAYS, because two nearby names
are easy to remove by association and neither may move:

* `explicit_constant_placeholders(..., convert_numbers_to_constant=)` -- a
  DIFFERENT surface (a keyword on a mechanical code-generation helper), still
  required-keyword in this release and passed explicitly by every shipped
  downstream call site. It stays.
* `rust/utils.rs::numbers_to_constant` -- the INTERNAL Rust port behind
  `read_infix(..., mask_numbers=True)`. Not public, not reachable by name from
  Python, and the reader's documented behaviour. It stays.

The replacement for the removed helper is `simplipy.masking.mask(tokens, engine,
policy)` (or the `engine.mask` front door). It is NOT a drop-in, and the
differences are the reason the helper is going -- they are pinned below so a
migration reads them here rather than discovering them in production:

1. RESERVED SPELLINGS. The helper classified by a bare `float()` probe, so
   `inf`/`nan` (any case or sign) and underscore groupings (`1_000`) were minted
   into `<constant>` -- a finite-by-doctrine placeholder standing for a
   non-finite literal. The masking module refuses them at the boundary (H-007).
   Laundering becomes a loud `ValueError`.
2. `np.pi`/`np.e`. `float('np.pi')` raises, so the helper left the special
   constants ALONE. They are masking SITES (owner ruling 2026-07-31), so
   `mask_all` masks them.
3. The exact-fraction literal `1/3`. `float('1/3')` raises, so the helper left
   it alone; `is_numeric_string` accepts it and `mask_all` masks it.
4. ROLE BLINDNESS. The helper masked `pow` exponents and `rootn` indices, whose
   integrality controls the DOMAIN. `mask_all` still does (that is its contract,
   and the honest legacy-equivalent policy); `mask_fittable` is the policy that
   does not.
5. VALIDATION and COLLECT. `mask` walks the expression, so a malformed token
   sequence raises where the helper silently rewrote it, and the default
   `collect=True` re-runs the engine (one `<constant>` per degree of freedom).
   `collect=False` is the positional 1:1 substitution the helper approximated.
"""
import os
import re

import pytest

from simplipy import SimpliPyEngine, masking
import simplipy.utils as utils
from conftest import acj_config_path


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


class TestTheStandaloneHelperIsGone:
    def test_the_name_is_gone(self) -> None:
        assert not hasattr(utils, 'numbers_to_constant')

    def test_the_declared_surface_no_longer_lists_it(self) -> None:
        assert 'numbers_to_constant' not in utils.__all__

    def test_star_import_does_not_bind_it(self) -> None:
        ns: dict = {}
        exec('from simplipy.utils import *', ns)
        assert 'numbers_to_constant' not in ns

    def test_the_api_reference_no_longer_renders_it(self) -> None:
        """The mkdocstrings member list is a BUILD dependency: a member that no
        longer exists fails the docs build, so the page moves with the code."""
        page = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'docs', 'api.md')
        with open(page, encoding='utf-8') as f:
            listed = re.findall(r'^\s*-\s+(\S+)\s*$', f.read(), flags=re.M)
        assert 'numbers_to_constant' not in listed


class TestWhatDeliberatelyStays:
    def test_the_placeholder_keyword_is_untouched(self) -> None:
        """A different surface with a shipped downstream on it: the keyword is
        required, both values keep working, and `False` stays the meaning every
        migrated call site passes."""
        expr, constants = utils.explicit_constant_placeholders(
            ['+', '<constant>', '2.5'], convert_numbers_to_constant=False)
        assert (expr, constants) == (['+', 'C_0', '2.5'], ['C_0'])

        with pytest.warns(DeprecationWarning):
            expr, constants = utils.explicit_constant_placeholders(
                ['+', '<constant>', '25'], convert_numbers_to_constant=True)
        assert (expr, constants) == (['+', 'C_0', 'C_1'], ['C_0', 'C_1'])

        with pytest.raises(TypeError):
            utils.explicit_constant_placeholders(['+', '<constant>', '2.5'])  # type: ignore[call-arg]

    def test_the_reader_still_masks_numbers(self, engine: SimpliPyEngine) -> None:
        """`read_infix(..., mask_numbers=True)` runs the INTERNAL Rust port. The
        Python helper's removal must not reach it."""
        assert engine.read_infix('2.5 * x0 + 3', mask_numbers=True) == \
            ['+', '*', '<constant>', 'x0', '<constant>']



class TestTheReplacementIsNotADropIn:
    """The four measured behaviour differences, stated as tests. A migration that
    wants the OLD answer on these inputs does not exist -- the old answer was
    wrong on each of them."""

    def test_canonical_literals_agree_with_the_removed_helper(self, engine: SimpliPyEngine) -> None:
        """On the inputs both accept, `mask_all` + `collect=False` IS the legacy
        result, token for token."""
        tokens = ['+', 'x0', '*', '3.14', 'x1']
        assert masking.mask(tokens, engine, masking.mask_all, collect=False) == \
            ['+', 'x0', '*', '<constant>', 'x1']

    @pytest.mark.parametrize('reserved', ['inf', '-inf', 'NaN', 'nan', '1_000', '0x10'])
    def test_reserved_spellings_raise_instead_of_being_laundered(
            self, engine: SimpliPyEngine, reserved: str) -> None:
        """The helper minted a finite `<constant>` for these. The masking module
        refuses the token (H-007) -- the laundering becomes loud."""
        with pytest.raises(ValueError, match='reserved numeric spelling'):
            masking.mask(['+', 'x0', reserved], engine, masking.mask_all, collect=False)

    def test_special_constants_are_masked_now(self, engine: SimpliPyEngine) -> None:
        """`float('np.pi')` raised, so the helper kept it. It is a site."""
        assert masking.mask(['*', 'np.pi', 'x0'], engine, masking.mask_all, collect=False) == \
            ['*', '<constant>', 'x0']

    def test_exact_fractions_are_masked_now(self, engine: SimpliPyEngine) -> None:
        """`float('1/3')` raised, so the helper kept it. `is_numeric_string`
        accepts the AC core's one-token rational."""
        assert masking.mask(['*', '1/3', 'x0'], engine, masking.mask_all, collect=False) == \
            ['*', '<constant>', 'x0']

    def test_malformed_input_raises_instead_of_being_rewritten(self, engine: SimpliPyEngine) -> None:
        """The helper was a positional `map` and never looked at the structure."""
        with pytest.raises(ValueError):
            masking.mask(['+', 'x0'], engine, masking.mask_all, collect=False)
