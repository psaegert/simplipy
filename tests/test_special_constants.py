"""Stage-1 special-constant policy wall (owner-ratified 2026-07-31 / 2026-08-01).

A special constant (``np.pi``, ``np.e``) denotes the SYMBOL -- pi itself, e itself --
not its f64 image: one vocabulary token with zero degrees of freedom. Folding
``* 2 np.pi`` into ``6.283185307179586`` rounds an exact object into ~52 bits, which is
LOSSY -- and simplify is lossless compression (design/UNIFIED_SIMPLICITY_MEASURE.md).
Rounding belongs to the fitter and the planned post-fit quantization/rationalization
stage, never to simplify.

This wall pins the stage-1 licences (§5 of the design doc; each is a future theorem of
the unified measure and gets DELETED when the measure ships):

1. GROUND-FOLD LICENCE: the numeric-fold fallback refuses special-bearing ground
   compounds -- ``2*pi``, ``cos(e)``, ``sin(pi)``, ``pi^2`` all stay symbolic. Exact
   collapses (``sin pi -> 0``) are the MINER's job as certified rules, never the f64
   fold's (whose ``sin(np.pi) = 1.2246e-16`` wart this licence retires). Ground
   compounds with +-inf/nan literals never folded to begin with (the evaluator is
   finite-only), so absorption behavior is untouched.
2. CONST-ABSORPTION LICENCE (as AMENDED by contract 10.11, owner 2026-08-08): a
   special absorbs into an EXISTING ``<constant>`` through the bijective bag sites
   exactly like every other finite ground -- ``C*pi -> C``, ``C+e -> C`` (the same
   ``C + g -> C'`` reparametrization P3' ships for ``cosh(1)``). The 2026-08-01
   blanket refusal ("for the post-fit rationalizer") is superseded. What STAYS
   refused: a special may never vanish into an INTRODUCED ``<constant>`` (bare
   ``pi -> C`` and ``pi*x0 -> C*x0`` -- determined-source licence at mint,
   Const-count invariant at load), and non-bag positions never absorb a special
   (``pi^C`` keeps its structure; its value set is (0, inf), not "anything").
   ``e^C`` collapses as before via the EXACT identity ``pow(e, t) == exp(t)``,
   after which ``exp(C) -> C`` is ordinary special-free absorption.
3. RESOLUTION LICENCE (miner): literal resolution never materializes a special-bearing
   source -- see ``TestResolutionLicence``.

Structural EXACT arithmetic on specials is deliberately outside all three licences:
bag cancellation (``pi/pi -> 1``, ``e - e -> 0``) is exact symbol algebra, not rounding.
"""
import os

import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = acj_config_path()


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj-4-3 config not staged')
    return SimpliPyEngine.from_config(CONFIG)


def check(eng, tokens, expected):
    """simplify, assert the exact canonical output and its idempotence.

    The expectations are the canonical TAGGED form, which `simplify` alone no longer
    returns for explicit input (it answers in the dialect it was handed, owner ruling
    2026-08-18): convert first, then simplify -- the exact migration for the retired
    `form='tagged'`."""
    out = eng.simplify(eng.to_tagged(list(tokens)))
    assert list(out) == list(expected), f'{tokens} -> {out}, expected {expected}'
    again = eng.simplify(list(out))
    assert list(again) == list(out), f'not idempotent: {out} -> {again}'
    return out


class TestGroundFoldLicence:
    """Special-bearing ground compounds stay symbolic; special-free folding is untouched."""

    def test_rational_times_special_stays(self, eng):
        # The ratified table rows "integer x special" and "rational x special": the
        # coefficient PRESERVES the symbol (T1 of the design doc).
        check(eng, ['*', '2', 'np.pi'], ['<mul>', '2', 'np.pi', '</mul>'])
        # The 1/2 coefficient takes its argmin spelling (the `<div>` side); what this
        # row guards is that the SYMBOL survives -- np.e is never folded to a float.
        check(eng, ['*', '0.5', 'np.e'], ['<mul>', 'np.e', '<div>', '2', '</mul>'])

    def test_function_of_special_stays(self, eng):
        # `f(special)` collapses ONLY to exact values, and those ship as MINED certified
        # rules -- the f64 fold may not approximate. This retires the documented
        # `sin(np.pi) -> 1.2246e-16` wart as a POLICY consequence (P-B3).
        check(eng, ['cos', 'np.e'], ['cos', 'np.e'])
        check(eng, ['exp', 'np.pi'], ['exp', 'np.pi'])
        # `sin(np.pi) -> 0` is exactly true and NOT f64-realised: f64 computes
        # 1.2246467991473532e-16 there, so the rewrite changes what the deployed
        # evaluator returns. From 0.14.0 it is a `real`-tier rule -- served by `real`
        # and `corpus`, declined by the default mode, which is the point of the split.
        from simplipy import Mode
        assert eng.simplify(['sin', 'np.pi'], mode=Mode.f64) == ['sin', 'np.pi']
        assert eng.simplify(['sin', 'np.pi'], mode=Mode.real) == ['0']
        assert eng.simplify(['sin', 'np.pi'], mode=Mode.corpus) == ['0']

    def test_pow_with_special_stays(self, eng):
        check(eng, ['pow', 'np.pi', '2'], ['pow', 'np.pi', '2'])
        check(eng, ['pow', '2', 'np.e'], ['pow', '2', 'np.e'])

    def test_special_op_special_stays(self, eng):
        # special o special stays symbolic: pi*e, pi+e, e^pi (the ratified headline;
        # exp(pi*x) > pow(23.14..., x) is the same refusal one wildcard away). e^pi
        # canonically respells through the EXACT identity pow(e, t) == exp(t) -- the
        # owner-preferred exp spelling -- and then stays: no materialization.
        check(eng, ['*', 'np.pi', 'np.e'], ['<mul>', 'np.pi', 'np.e', '</mul>'])
        check(eng, ['+', 'np.pi', 'np.e'], ['<add>', 'np.pi', 'np.e', '</add>'])
        check(eng, ['pow', 'np.e', 'np.pi'], ['exp', 'np.pi'])
        # The T2 headline itself: exp(pi*x) survives as the canonical symbolic form.
        check(eng, ['pow', 'np.e', '*', 'np.pi', 'x0'],
              ['exp', '<mul>', 'np.pi', 'x0', '</mul>'])

    def test_nested_ground_special_stays(self, eng):
        # The refusal is bottom-up: the inner product no longer materializes, so the
        # outer function never sees a foldable literal (cos(2*pi) = 1 belongs to the
        # miner as a certified exact rule, not to the f64 fold).
        # REFRESH 2026-08-01: cos(2*pi) = 1 now serves AS the certified mined rule
        # (never the f64 fold) -- the refusal below it is unchanged: the inner product
        # still refuses to materialize, which is exactly why the RULE is needed.
        check(eng, ['cos', '*', '2', 'np.pi'], ['1'])

    def test_exact_symbol_algebra_still_fires(self, eng):
        # Bag cancellation is exact symbol arithmetic, NOT rounding: outside the licence.
        check(eng, ['/', 'np.pi', 'np.pi'], ['1'])
        check(eng, ['-', 'np.e', 'np.e'], ['0'])

    def test_special_free_folding_unchanged(self, eng):
        check(eng, ['+', '2', '3'], ['5'])
        check(eng, ['cos', '0'], ['1'])
        # STAGE 2 (the flip this pin predicted): under the mu-governed fold a
        # transcendental ROUNDING never descends, so sin(-1) stays symbolic exactly as
        # the owner ruled ("sin(-1) should stay as is. It makes sense under the MDL").
        check(eng, ['sin', 'neg', '1'], ['sin', '-1'])

    def test_inf_nan_rows(self, eng):
        # H-007 (2026-08-04) supersedes the old pin: bare 'inf'/'nan' are RESERVED
        # spellings, refused at the boundary (they were opaque symbols here while the
        # numeric layer read them as values -- the contradiction the boundary kills).
        import pytest
        for row in (['*', 'np.pi', 'inf'], ['*', 'np.e', 'nan']):
            with pytest.raises(ValueError, match='reserved numeric spelling'):
                eng.simplify(row)
        # The canonical spellings fold CORRECTLY under the determined-ground arms
        # (np.pi is certified nonzero-positive, so pi * inf is determined; nan absorbs).
        check(eng, ['*', 'np.pi', 'float("inf")'], ['float("inf")'])
        check(eng, ['*', 'np.e', 'float("nan")'], ['float("nan")'])
        # The licence-scope point the old pin actually carried: an OPAQUE leaf next to
        # a special constant stays unfolded.
        check(eng, ['*', 'np.pi', 'q'], ['<mul>', 'np.pi', 'q', '</mul>'])


class TestExactCollapsesMint:
    """Owner ruling 2026-08-01: `cos(pi)` SHOULD simplify to -1 -- the bare ground exact
    collapses mint as certified rules (the policy's `f(special) -> exact value` row).

    Root cause of the earlier refusal (this test was red): the mint itself worked, but
    stage-2 confirmation re-runs `find_rule` with the target as the only candidate, and
    the var-free short-circuit's `Class::Finite -> <constant>` arm HIJACKED the answer
    for special-bearing sources -- returning `<constant>` instead of testing `(-1)`,
    which the result-must-equal-target guard rightly scored as unconfirmed. The arm is
    now gated on special-free sources (the same absorption the Const licence forbids,
    refused at the channel), so confirmation falls through to the scan and honestly
    verifies the asked target. Inexact values still cannot mint: their correctly-rounded
    literals are not in the candidate alphabet, resolution refuses special-bearing
    sources, and the hiprec point layer refutes near-misses."""

    @pytest.fixture(scope='class')
    def mined(self, tmp_path_factory):
        import json
        import yaml
        ops = {
            "cos": {"realization": "np.cos", "alias": [], "inverse": None,
                    "arity": 1, "precedence": 3, "commutative": False},
            "sin": {"realization": "np.sin", "alias": [], "inverse": None,
                    "arity": 1, "precedence": 3, "commutative": False},
            "log": {"realization": "np.log", "alias": [], "inverse": None,
                    "arity": 1, "precedence": 3, "commutative": False},
            "exp": {"realization": "np.exp", "alias": [], "inverse": None,
                    "arity": 1, "precedence": 3, "commutative": False},
            "*": {"realization": "*", "alias": [], "inverse": "/",
                  "arity": 2, "precedence": 2, "commutative": True},
        }
        d = tmp_path_factory.mktemp('exact_collapse_mine')
        (d / 'rules.json').write_text(json.dumps([]))
        cfg = d / 'config.yaml'
        cfg.write_text(yaml.safe_dump({'operators': ops, 'rules': 'rules.json'}))
        engine = SimpliPyEngine.from_config(str(cfg))
        engine.find_rules(max_source_pattern_length=3, dummy_variables=1,
                          extra_internal_terms=['0', '1', '(-1)', 'np.pi', 'np.e',
                                                '<constant>'],
                          X=256, seed=11, verbose=False)
        return engine

    def test_bare_exact_collapses_mint(self, mined):
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in mined.simplification_rules}
        assert (('cos', 'np.pi'), ('(-1)',)) in rules, sorted(rules)
        assert (('sin', 'np.pi'), ('0',)) in rules, sorted(rules)
        assert (('log', 'np.e'), ('1',)) in rules, sorted(rules)

    def test_mined_engine_serves_the_collapse(self, mined):
        # The owner's ask, end to end: cos(pi) simplifies to -1 (via the mined rule).
        assert mined.simplify(['cos', 'np.pi']) == ['-1']
        assert mined.simplify(['log', 'np.e']) == ['1']
        # `sin(np.pi)` is the one of the three f64 does not realise, so it does not serve
        # the default mode. Its presence in the MINE's output is what this test can speak
        # to, and it is asserted from the mined rule set itself -- an earlier version
        # installed the rule into `real` and then asserted that same rule fires, which
        # says nothing about the mine at all.
        # This fixture is a freshly-mined engine with ONE rule set (no triple), so the
        # rule is in its default set and fires here. That is what "the mined engine serves
        # the collapse" can mean for it. Where the rule LANDS -- `real`, not f64, because
        # f64 computes 1.2246467991473532e-16 -- is a property of the ROUTED artifact and
        # is pinned in `TestGroundFoldLicence` against the shipped triple.
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in mined.simplification_rules}
        assert (('sin', 'np.pi'), ('0',)) in rules
        assert mined.simplify(['sin', 'np.pi']) == ['0']
        from simplipy.verify._contract import judge_rule
        assert judge_rule(['sin', 'np.pi'], ['0'])['tier'] == 'real'

    def test_ground_rules_bypass_sort_promotion(self, eng):
        # The promotion sorts classify WILDCARD instantiation domains; a ground rule has
        # no wildcard to carry a sort (one instantiation: itself, certified at mint +
        # stage-2 confirmation). The reference ladder's universe is wildcard rules, and
        # feeding it ground rules silently discarded them (measured 2026-08-01: the
        # owner-ruled exact collapses minted, confirmed, survived the covered-prune,
        # then vanished at promote_sorts). Ground rules must pass through unchanged.
        from simplipy.promotion import promote
        bare = [(('cos', 'np.pi'), ('(-1)',)), (('sin', 'np.pi'), ('0',)),
                (('tan', 'np.pi'), ('0',)), (('log', 'np.e'), ('1',))]
        controls = [(('cos', '+', '_0', 'np.pi'), ('neg', 'cos', '_0')),
                    (('pow', 'np.e', '_0'), ('exp', '_0'))]
        # The bypass is ALL pure-literal ground rules (widened with fold
        # unification, 2026-08-02: special-free exact collapses like `cos 0 -> 1`
        # are now legitimate mined rows, and the ladder's wildcard universe would
        # still eat them). `<constant>`-bearing ground rules keep flowing to the
        # ladder's ground tier (judge_ground) -- sound ones ship, unsound ones die
        # there, exactly as in the reference. The f64-respell family
        # (`+ 1 exp (-1) -> 1.3678794411714423` respells the engine's exact
        # `1.36787944117144233` to its f64 rounding) is therefore NO LONGER
        # contained by promotion: promote() passes it through shape-blind, and
        # containment lives where it is principled -- the mint-side strict-mu +
        # skeleton + atomic-mark guards, and the SYMBOLIC GATE in _finalize
        # (asserted below: the gate KILLs the row promotion now admits).
        cground_pass = (('/', '<constant>', 'float("inf")'), ('0',))
        cground_kill = (('acos', 'pow', '<constant>', 'float("inf")'), ('<constant>',))
        f64_respell = (('+', '1', 'exp', '(-1)'), ('1.3678794411714423',))
        kept, report = promote(bare + controls + [cground_pass, cground_kill, f64_respell],
                               eng)
        kept_set = {(tuple(lhs), tuple(rhs)) for lhs, rhs in kept}
        for rule in bare + controls:
            assert rule in kept_set, (rule, sorted(kept_set))
        assert cground_pass in kept_set
        assert cground_kill not in kept_set
        assert f64_respell in kept_set  # shape-blind bypass: promotion no longer contains
        assert report['stage_counts']['ground_passthrough'] == 5
        # The pipeline's answer to the respell CHANGED with the mode triple, and the
        # old claim -- "the symbolic gate refuses it" -- is retired. The bucket is still
        # KILL: `1 + e^-1 -> 1.3678794411714423` respells the exact value as its f64
        # rounding, so it is mathematically false. But it is EXACTLY what f64 computes,
        # so its tier is `f64` and the gate ROUTES it into rules.json rather than
        # dropping it. That is correct under the split: in the mode whose authority is
        # the deployed evaluator, the rewrite is exact.
        from simplipy.verify import verify_ruleset
        from simplipy.verify._contract import judge_rule
        gate = verify_ruleset([[list(f64_respell[0]), list(f64_respell[1])]])
        assert gate['buckets']['KILL'] == [0], gate['buckets']
        verdict = judge_rule(list(f64_respell[0]), list(f64_respell[1]))
        assert verdict['realised'] is True and verdict['tier'] == 'f64', verdict
        # ... and it is therefore clean for the f64 file and NOT for the real one
        assert verify_ruleset([[list(f64_respell[0]), list(f64_respell[1])]],
                              mode='f64')['is_clean'] is True
        assert verify_ruleset([[list(f64_respell[0]), list(f64_respell[1])]],
                              mode='real')['is_clean'] is False

    def test_inexact_values_still_refused(self, mined):
        # cos(e), exp(pi), e*pi have no exact alphabet spelling: nothing may mint, and
        # no special-bearing rule may carry a rounded decimal literal.
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in mined.simplification_rules}
        assert not any(lhs == ('cos', 'np.e') for lhs, _ in rules), sorted(rules)
        assert not any(lhs == ('exp', 'np.pi') for lhs, _ in rules), sorted(rules)
        for lhs, rhs in rules:
            if any(t in ('np.pi', 'np.e') for t in lhs):
                assert not any('.' in t and t not in ('np.pi', 'np.e') for t in rhs), \
                    (lhs, rhs)


class TestConstAbsorptionLicence:
    """Contract 10.11 (owner 2026-08-08): specials absorb into an EXISTING mask at
    the bijective bag sites; introduced masks and non-bag positions still refuse."""

    def test_const_times_special_absorbs(self, eng):
        # C*pi -> C: the Mul-scale bijection (c' = c*pi), same family as C*cosh(1).
        check(eng, ['*', '<constant>', 'np.pi'], ['<constant>'])
        check(eng, ['*', '*', '<constant>', 'np.pi', 'x0'],
              ['<mul>', '<constant>', 'x0', '</mul>'])

    def test_const_plus_special_absorbs(self, eng):
        # C+e -> C, C-pi -> C: the Add-shift bijection (c' = c+g), both signs.
        check(eng, ['+', '<constant>', 'np.e'], ['<constant>'])
        check(eng, ['-', '<constant>', 'np.pi'], ['<constant>'])
        # A special-BEARING ground member absorbs as a whole: C + 2*pi -> C.
        check(eng, ['+', '<constant>', '*', '2', 'np.pi'], ['<constant>'])

    def test_bare_special_never_masked(self, eng):
        # The introduced-mask direction stays refused: pi alone, and pi beside
        # variables with no mask present, keep their exact structure.
        check(eng, ['np.pi'], ['np.pi'])
        assert eng.simplify(eng.to_tagged(['*', 'np.pi', 'x0'])) \
            == ['<mul>', 'np.pi', 'x0', '</mul>']

    def test_special_base_pow_const(self, eng):
        # pi^C keeps its structure: pow is not an absorption site -- the family's
        # value set is (0, inf), not "anything", so the mask may not claim it.
        out = eng.simplify(['pow', 'np.pi', '<constant>'])
        assert out != ['<constant>'], 'pi^C absorbed the special'
        assert 'np.pi' in out and '<constant>' in out
        assert eng.simplify(list(out)) == out
        # e^C is DIFFERENT by design: pow(e, t) == exp(t) is an EXACT identity that
        # eliminates the special losslessly BEFORE any mask meets it, and the resulting
        # `exp <constant>` collapses under the pre-existing special-free value-set
        # absorption (same family as `sin <constant> -> <constant>`). The licence
        # protects specials from ROUNDING and ABSORPTION, not from exact identities.
        check(eng, ['pow', 'np.e', '<constant>'], ['<constant>'])
        check(eng, ['exp', '<constant>'], ['<constant>'])

    def test_rational_still_absorbs(self, eng):
        # The contract's forall-exists direction is untouched for pure literals.
        check(eng, ['*', '<constant>', '2.5'], ['<constant>'])
        check(eng, ['abs', '<constant>'], ['<constant>'])

    def test_const_pow_special_still_refused(self, eng):
        # C^pi was already refused (mixed value class); the licence must not regress it.
        check(eng, ['pow', '<constant>', 'np.pi'], ['pow', '<constant>', 'np.pi'])

    def test_const_inf_still_unabsorbed(self, eng):
        # H-007 (2026-08-04): bare 'inf' is a reserved spelling; the canonical token
        # carries the licence point -- Const never absorbs a non-Finite class factor.
        check(eng, ['*', '<constant>', 'float("inf")'],
              ['<mul>', 'float("inf")', '<constant>', '</mul>'])
