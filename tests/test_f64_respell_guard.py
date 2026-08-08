"""Mint-side guard for the f64-respell leak (finding (a) of the 2026-08-01 audit).

The miner's evaluate short-circuit answers a special-free all-literal ground source
with its f64 evaluation. The engine's own fold answers the SAME source with exact
decimal-rational arithmetic on top of any materialized transcendental, and the two
values differ whenever the exact sum picks up digits the f64 sum rounds away:
``+ 1 exp (-1)`` folds exactly to ``1.36787944117144233`` (18 significant digits)
while f64 arithmetic gives ``1.3678794411714423`` -- a DIFFERENT value that happens
to print shorter. The full reduction ordering's literal tie tiers (lit_size) scored
that shorter print "below" the engine's mark, so the family minted: ~1.4k
mint-then-drop rules per canonical mine, measured 2026-08-01, surviving only because
sort promotion's UNSUPPORTED bucket happened to eat them -- and ``promote_sorts``
defaults to FALSE, so any default ``find_rules`` run shipped them as literal->literal
rewrites that ROUND exact values at serve time.

The fix is the same strict gate the resolution channel already carries: an
f64-EVALUATED literal is a computed value, not an alphabet spelling, so the evaluate
short-circuit now honors ``accept_resolved`` (mining passes
strict-semantic-complexity-below-mark; fail closed on literal-vs-literal ties).
Confirmation and public ``find_rule`` callers pass no criterion and keep the
short-circuit as a query tool. Under the unified measure the family dies by
arithmetic (transcendental folds refuse, marks stay symbolic and cheap), and this
licence-style guard gets deleted with the rest of stage 1.
"""
import json

import pytest
import yaml

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path


@pytest.fixture(scope='module')
def mined(tmp_path_factory):
    # {+, exp, cos} over {0, 1, (-1), np.pi}: length 4 reaches `+ 1 exp (-1)`
    # (the respell family's minimal member) and length 2 reaches `cos np.pi`
    # (the exact-collapse positive control). promote_sorts EXPLICITLY off (it
    # defaults ON since 2026-08-01): the artifact must pin the MINT level with
    # no downstream containment to hide behind.
    ops = {
        "+": {"realization": "+", "alias": [], "inverse": "-",
              "arity": 2, "precedence": 1, "commutative": True},
        "exp": {"realization": "np.exp", "alias": [], "inverse": None,
                "arity": 1, "precedence": 3, "commutative": False},
        "cos": {"realization": "np.cos", "alias": [], "inverse": None,
                "arity": 1, "precedence": 3, "commutative": False},
    }
    d = tmp_path_factory.mktemp('f64_respell_mine')
    (d / 'rules.json').write_text(json.dumps([]))
    cfg = d / 'config.yaml'
    cfg.write_text(yaml.safe_dump({'operators': ops, 'rules': 'rules.json'}))
    engine = SimpliPyEngine.from_config(str(cfg))
    engine.find_rules(max_source_pattern_length=4, dummy_variables=1,
                      extra_internal_terms=['0', '1', '(-1)', 'np.pi'],
                      X=256, seed=11, promote_sorts=False, verbose=False)
    return engine


SPECIALS = ('np.pi', 'np.e')


def _is_literal(tok, engine):
    return tok not in SPECIALS and tok not in engine.operators \
        and tok not in ('<constant>',) and not tok.startswith('_') \
        and not (tok.startswith('x') and tok[1:].isdigit())


class TestF64RespellMintGuard:

    def test_minimal_member_does_not_mint(self, mined):
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in mined.simplification_rules}
        assert (('+', '1', 'exp', '(-1)'), ('1.3678794411714423',)) not in rules, \
            sorted(rules)

    def test_no_special_free_ground_respell_mints(self, mined):
        # Sweep, re-earned under fold unification (2026-08-02): special-free
        # pure-literal ground sources no longer fold completely -- transcendental
        # evaluation left the engine, so exact collapses (`cos 0 -> 1`) now
        # legitimately ARRIVE AS RULES. What stays unmintable is the respell
        # class this guard was built for: a target literal carrying a COMPUTED
        # f64 decimal (`+ 1 exp (-1) -> 1.3678794411714423`) is a rounding of an
        # exact value. On this alphabet every exactly-representable special-free
        # ground value is dot-free (rational collapses fold natively and are
        # unmintable below an atomic mark), so a decimal-pointed RHS literal is
        # definitionally a respell.
        ops = set(mined.operators)
        for lhs, rhs in mined.simplification_rules:
            ground = all(t in ops or _is_literal(t, mined) for t in lhs)
            special_free = not any(t in SPECIALS for t in lhs)
            if ground and special_free and any(t not in ops for t in lhs):
                for tok in rhs:
                    if _is_literal(tok, mined):
                        assert '.' not in tok.strip('()'), \
                            f'f64-respell RHS minted: {lhs} -> {rhs}'

    def test_exact_collapse_still_mints(self, mined):
        # Positive control: the guard must not touch the owner-ruled exact
        # collapses (special-bearing, minted through the candidate scan).
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in mined.simplification_rules}
        assert (('cos', 'np.pi'), ('(-1)',)) in rules, sorted(rules)

    def test_serve_never_rounds(self, mined):
        # The footgun, end to end: with the leak, this artifact (promotion OFF)
        # carried a serving rewrite that ROUNDED an exact value. Stage 1 pinned the
        # engine's exact fold (`1.36787944117144233`) standing unrounded; stage 2's
        # mu-governed fold goes further -- exp(-1) never materializes at all, so the
        # state stays symbolic and NO reading of it carries the f64 respell.
        out = mined.simplify(['+', '1', 'exp', '(-1)'])
        assert 'exp' in out, out
        assert not any(t.startswith('1.3678') or t.startswith('0.3678') for t in out), out


@pytest.fixture(scope='module')
def mined_const(tmp_path_factory):
    # The same universe with `<constant>` in the alphabet, mined WITHOUT stage-2
    # confirmation: this pins the MINT stage itself. When the strict gate closed
    # the evaluate short-circuit, the respell family's sources fell through to the
    # interval-class `Finite -> <constant>` arm (below the literal mark in the
    # ordering: equal complexity, canonical-order tier), and the mine emitted
    # `+ 1 exp (-1) -> <constant>` instead -- a Const-INTRODUCING rule for a
    # DETERMINED source (var-free, Const-free, all leaves finite literals) that
    # only died because confirmation's unguarded short-circuit answers the f64
    # literal, which is not the asked target (measured: canonical mine 2026-08-01,
    # mint counts unchanged at 15,579 and confirmation rejections up by exactly
    # the 210-rule family). A determined source is its own reduction: the engine
    # folds it to its exact value and never absorbs a determined value into a
    # mask, so the mint must agree and propose NOTHING.
    ops = {
        "+": {"realization": "+", "alias": [], "inverse": "-",
              "arity": 2, "precedence": 1, "commutative": True},
        "exp": {"realization": "np.exp", "alias": [], "inverse": None,
                "arity": 1, "precedence": 3, "commutative": False},
        "cos": {"realization": "np.cos", "alias": [], "inverse": None,
                "arity": 1, "precedence": 3, "commutative": False},
    }
    d = tmp_path_factory.mktemp('f64_respell_const_mine')
    (d / 'rules.json').write_text(json.dumps([]))
    cfg = d / 'config.yaml'
    cfg.write_text(yaml.safe_dump({'operators': ops, 'rules': 'rules.json'}))
    engine = SimpliPyEngine.from_config(str(cfg))
    engine.find_rules(max_source_pattern_length=4, dummy_variables=1,
                      extra_internal_terms=['0', '1', '(-1)', 'np.pi', '<constant>'],
                      X=256, seed=11, confirm=False, promote_sorts=False,
                      verbose=False)
    return engine


@pytest.fixture(scope='module')
def mined_acj():
    return SimpliPyEngine.from_config(
        acj_config_path())


class TestMuEraResolutionRespell:
    """Stage-2 reflection find (2026-08-01): mu REOPENED the respell disease at the
    resolution channel for LONG-exact-literal marks. The morning's strict-descent
    guard worked because equal-value-class literals had EQUAL coarse complexity;
    under mu a rounded literal has FEWER BITS than a long exact rational, so for
    `(0.3333333333333333)^2 * x0` -- whose exact canon coefficient is the 32-digit
    `0.11111111111111108888888888888889` (~210 bits) -- the f64 respell
    `0.1111111111111111` (~105 bits) STRICTLY DESCENDS and resolution minted a
    value change (~1e-17). Unreachable from the canonical alphabet (no long
    decimals) but live on the proposals/hint channel and the public library scan.
    The guard is the doctrine already written on the channel: resolution recovers
    STRUCTURE, never respells literals -- a resolved target whose literal-skeleton
    (all tokens except numeric literals) equals the mark's is a respell, refused."""

    def test_long_literal_mark_is_not_respelled(self, mined_acj):
        import numpy as np
        core = mined_acj._core
        src = ['*', '*', '0.3333333333333333', '0.3333333333333333', 'x0']
        mark = mined_acj.simplify(list(src), form='explicit')
        assert mark[1].startswith('0.111111111111111088'), mark  # exact square stands
        x = np.random.default_rng(0).uniform(-8, 8, 256).tolist()
        lib = core.build_candidate_library(
            [['*', '<constant>', 'x0'], ['x0'], ['<constant>']], ['x0'], x, 256)
        res = core.find_rule_lib(src, len(src), 4, lib, mark=mark)
        assert res is None, res

    def test_structural_resolution_still_lives(self, mined_acj):
        # POSITIVE CONTROL: the crown-jewel exact-quotient family must keep
        # resolving -- `/ acos 0 np.pi` has a SYMBOLIC mark when no rule covers it,
        # its skeleton differs from a bare literal, and the resolved value 0.5 is
        # exact. The guard keys on skeleton identity, not on resolution per se.
        # (Since the 2026-08-01 artifact refresh the SHIPPED engine serves 0.5
        # directly -- pinned first -- so the resolution probe runs on a RULE-FREE
        # twin of the same vocabulary, the mining-time situation.)
        import numpy as np
        # The RESOLUTION is the guard; the resolved exact 1/2 then takes its argmin
        # spelling (`/ 1 2`, not `0.5`) -- §10.10(1), owner-ratified 2026-08-07.
        assert mined_acj.simplify(['/', 'acos', '0', 'np.pi'], form='explicit') \
            == ['/', '1', '2']
        bare = SimpliPyEngine(operators=mined_acj._operators_config, rules=[])
        core = bare._core
        src = ['/', 'acos', '0', 'np.pi']
        mark = bare.simplify(list(src), form='explicit')
        assert 'acos' in mark, mark
        x = np.random.default_rng(0).uniform(-8, 8, 256).tolist()
        lib = core.build_candidate_library([['<constant>'], ['x0']], ['x0'], x, 256)
        res = core.find_rule_lib(src, len(src), 3, lib, mark=mark)
        assert res == ['0.5'], res


class TestDeterminedSourceConstIntroduction:

    def test_determined_source_mints_no_const_abstraction(self, mined_const):
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in mined_const.simplification_rules}
        assert (('+', '1', 'exp', '(-1)'), ('<constant>',)) not in rules, sorted(rules)

    def test_no_determined_source_mints_anything(self, mined_const):
        # Sweep at the mint: no rule may have a determined LHS and a
        # Const-bearing RHS anywhere in the artifact.
        ops = set(mined_const.operators)
        for lhs, rhs in mined_const.simplification_rules:
            determined = ('<constant>' not in lhs
                          and not any(t in SPECIALS for t in lhs)
                          and all(t in ops or _is_literal(t, mined_const) for t in lhs))
            if determined and '<constant>' in rhs:
                assert False, f'determined source minted Const abstraction: {lhs} -> {rhs}'

    def test_exact_collapse_still_mints_here_too(self, mined_const):
        # Positive control: the special-bearing exact collapse is untouched by the
        # determined-source clause (its source is not Const-free-var-free-literal:
        # it carries a special, and its target is an alphabet literal, not
        # `<constant>`). The Finite arm's legitimate value-set job (`atanh cos
        # <constant> -> <constant>`) cannot be controlled in THIS vocabulary --
        # every Const compound here absorbs natively in the engine, so the mark is
        # already `<constant>` -- the canonical re-mine (1753-identical) is that
        # control.
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in mined_const.simplification_rules}
        assert (('cos', 'np.pi'), ('(-1)',)) in rules, sorted(rules)
