"""The Const-count invariant: a fire never increases the number of ``<constant>``
placeholders -- one invariant, every layer.

``<constant>`` is masking's token, and abstraction is DOWNSTREAM policy, so the miner
must not manufacture it. The old boolean translate gate (RHS has Const, LHS none)
dropped the masking-era const-introducing family at LOAD, after the mine had already
spent its budget minting it: 195 of 817 rules at the coverage-ordering canonical
re-mine were minted and then unconditionally dropped (mint-then-drop). The invariant
now holds at every layer:

* MINT -- for a const-BEARING source, candidates with more slots than the source are
  skipped outright; for a const-FREE source, a matched Const-bearing candidate's slots
  are fixed reals and are RESOLVED to literal tokens: the correctly-rounded value of
  the true constant, pinned by sign-bisection at the hiprec arbiter (1024 bits, where
  ``np.pi``/``np.e`` denote their TRANSCENDENTAL values), never the raw least-squares
  witness (measured ~20 ulp off ``e^pi``). The ordering acceptance then judges the
  RESOLVED spelling, so a resolution the engine already reaches natively (the fold
  family, `* sin (-1) _0`) lands on the mark's own state and dies there, while a
  genuine SPECIAL-FREE restructuring mints as a zero-mask SERVING rule -- recovered
  content the old channel minted abstractly and translation then discarded.
  Under the special-constant policy the resolution channel additionally refuses
  every SPECIAL-BEARING source: `exp(pi*x) -> pow(23.14069263277927, x)` -- once this
  file's flagship recovery -- is now the flagship REFUSAL (the exact
  symbolic form; the literal is its ~52-bit rounding).
* TRANSLATE -- the load gate is the count form on the canonical sides, subsuming the
  boolean gate and additionally refusing Const-bearing count-increasing shapes
  (one slot in the LHS, two in the RHS) the boolean let through.
* PROPOSALS -- a count-increasing hint is refused; the library scan may still resolve
  a literal target for the same source.
* SERVE -- end-to-end, ``simplify`` never increases the ``<constant>`` count.
"""
import json
import os

import pytest

from simplipy import SimpliPyEngine
from simplipy.verify import default_corpus
from conftest import acj_config_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = acj_config_path()

MUL = {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True}
ADD = {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True}
SIN = {"realization": "np.sin", "alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False}
EXP = {"realization": "np.exp", "alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False}
POW = {"realization": "simplipy.operators.pow", "alias": [], "inverse": None,
       "arity": 2, "precedence": 3, "commutative": False}


def _n_const(tokens) -> int:
    return sum(1 for t in tokens if t == '<constant>')


def _mine(tmp_path, ops, extra, max_src=4, max_tgt=3, proposals=None):
    (tmp_path / 'rules.json').write_text('[]')
    import yaml
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump({'operators': ops, 'rules': 'rules.json'}))
    engine = SimpliPyEngine.from_config(str(cfg))
    out = str(tmp_path / 'mined.json')
    # promote_sorts EXPLICITLY off (defaults ON): these tests pin
    # MINT-level invariants, and promotion both reshapes sorts and adds the ladder's
    # seeded identities, which would shadow the mint-level assertions.
    engine.find_rules(max_source_pattern_length=max_src, max_target_pattern_length=max_tgt,
                      dummy_variables=1, extra_internal_terms=extra, X=256,
                      constants_fit_challenges=2, constants_fit_retries=2,
                      seed=7, verbose=False, output_file=out, proposals=proposals,
                      promote_sorts=False)
    return engine, out


class TestMintCountInvariant:
    def test_leaf_rename_family_never_mints(self, tmp_path) -> None:
        """`* sin (-1) _0 -> * <constant> _0`-class never mints -- under mu (stage 2)
        for the STRONGEST reason yet: `<constant>` is the priciest atom (128), so a
        Const-introducing target ASCENDS from any cheap symbolic mark. And since the
        generalized Const-shift absorption arm the value-set
        absorption family is ENGINE-NATIVE too: `C * sin(-1)` folds to `<constant>`
        at serve (special-free ground member, interval-certified finite nonzero), so
        the source reaches an atomic mark and the previously-mined absorption pair
        self-retires (atomic-mark clause). The mine is EMPTY; serve covers the family
        structurally, in every sign and spelling the enumerator never reached."""
        engine, _ = _mine(tmp_path, {'*': MUL, 'sin': SIN}, ['(-1)', '<constant>'])
        rules = {(tuple(lhs), tuple(rhs)) for lhs, rhs in engine.simplification_rules}
        assert rules == set(), rules
        assert engine.simplify(['*', '<constant>', 'sin', '(-1)']) == ['<constant>']
        assert engine.simplify(['*', 'sin', '(-1)', '<constant>']) == ['<constant>']

    def test_restructure_family_refused_for_special_sources(self, tmp_path) -> None:
        """POLICY REVERSAL (the resolution licence). This test USED to pin
        `exp(pi*x) -> pow(23.14069263277927, x)` as recovered mint content -- the literal
        resolution channel's flagship. The special-constant policy inverts the
        preference: `exp(pi*x)` IS the preferred form (pi is one exact symbol; the
        literal is its ~52-bit rounding), so literal resolution refuses every
        special-bearing source outright and the family stays SYMBOLIC."""
        engine, _ = _mine(tmp_path, {'exp': EXP, 'pow': POW, '*': MUL},
                          ['np.pi', '<constant>'])
        rules = [(list(lhs), list(rhs)) for lhs, rhs in engine.simplification_rules]
        assert not any(lhs == ['exp', '*', 'np.pi', '?0'] for lhs, _ in rules), rules
        # no mined rule materializes a special anywhere in this vocabulary
        for lhs, rhs in rules:
            if 'np.pi' in lhs:
                assert not any('.' in t for t in rhs), (lhs, rhs)
        # the invariant holds across the whole mined set
        assert all(_n_const(rhs) <= _n_const(lhs) for lhs, rhs in rules)
        # and the state SERVES as itself: the preferred symbolic form
        assert engine.simplify(['exp', '*', 'np.pi', 'x0']) == \
            ['exp', '*', 'np.pi', 'x0']

    def test_const_preserving_family_is_the_engines(self, tmp_path) -> None:
        """The servable Const algebra collapses in the engine, so the mine has nothing to
        write: `(e^C1)^C2 = e^(C1*C2)` is one fitted constant, and so is its special-bearing
        twin, which the canon respells to `pow(C, pi)` (exp(C) -> C): finite for C >= 0, it
        collapses by the constant-collapse criterion the mint reads, as `C^2.5` does -- pi is
        a fixed positive real there like a literal, as at `pi^C`."""
        engine, _ = _mine(tmp_path, {'exp': EXP, 'pow': POW, '*': MUL},
                          ['np.pi', '<constant>'])
        rules = [(list(lhs), list(rhs)) for lhs, rhs in engine.simplification_rules]
        for source in (['pow', 'exp', '<constant>', '<constant>'],
                       ['pow', 'exp', '<constant>', 'np.pi']):
            assert (source, ['<constant>']) not in rules
            assert engine.simplify(source) == ['<constant>']


class TestTranslationCountGate:
    _OPS = {'+': ADD, '*': MUL, 'pow': POW, 'exp': EXP}

    def test_count_increasing_rule_dropped_at_load(self) -> None:
        """One Const slot in the LHS, two in the RHS: KB-oriented (complexity 5 -> 4)
        and Const-bearing, so the old boolean gate loaded it. The count form drops it
        at translation."""
        rule = (('pow', '+', '?0', '<constant>', '2'),
                ('+', '<constant>', '*', '<constant>', '?0'))
        engine = SimpliPyEngine(operators=self._OPS, rules=[rule])
        kept, subsumed, dropped, twins = engine._core.ac_rules_info_in_mode(rule_mode='default')
        assert (kept, subsumed, dropped, twins) == (0, 0, 1, 0)

    def test_count_preserving_rule_loads(self) -> None:
        """Control: the servable Const algebra (1 -> 1) passes the count gate."""
        rule = (('pow', 'exp', '<constant>', 'np.pi'), ('<constant>',))
        engine = SimpliPyEngine(operators=self._OPS, rules=[rule])
        kept, subsumed, dropped, twins = engine._core.ac_rules_info_in_mode(rule_mode='default')
        assert (kept, subsumed, dropped, twins) == (1, 0, 0, 0)


class TestProposalChannel:
    def test_const_introducing_hint_refused_end_to_end(self, tmp_path) -> None:
        """A proposal arriving with the masking-era hint spelling is now refused on
        EVERY channel: the hint itself dies at the count
        gate, and the library scan's resolved literal `pow(23.14..., x0)` is a ~117-
        unit materialization of the 32-unit `exp(pi*x0)` -- an ascent the strict mu
        gate rejects. Pre-mu, the proposal channel still resolved it (the stage-1
        resolution licence was scoped to the batch miner; the hint channel was the
        one spelling that could resurrect a materialization). The owner's original
        preference is now channel-independent arithmetic."""
        proposals = [{'source': ['exp', '*', 'np.pi', 'x0'],
                      'target': ['pow', '<constant>', 'x0']}]
        engine, out = _mine(tmp_path, {'exp': EXP, 'pow': POW, '*': MUL},
                            ['np.pi', '<constant>'], max_src=3, proposals=proposals)
        with open(out + '.provenance.json') as fh:
            outcomes = json.load(fh)['proposals']['outcomes']
        assert outcomes == {'certified': 0, 'already_covered': 0,
                            'rejected': 1, 'duplicate': 0,
                            'certified_then_dropped': 0}
        rules = [(list(lhs), list(rhs)) for lhs, rhs in engine.simplification_rules]
        assert all(_n_const(rhs) <= _n_const(lhs) for lhs, rhs in rules)
        out_form = engine.simplify(['exp', '*', 'np.pi', 'x0'])
        assert 'np.pi' in out_form and not any('23.14' in t for t in out_form), out_form


@pytest.mark.skipif(not os.path.exists(CONFIG), reason='acj-4-3 assets not present')
class TestEndToEndInvariant:
    def test_simplify_never_increases_const_count(self) -> None:
        """The serve-side statement of the invariant, on the shipped pairing and the
        tracked 400-row corpus plus a Const-bearing battery, in both dialects."""
        engine = SimpliPyEngine.from_config(CONFIG)
        corpus = default_corpus()
        extra = [['acos', 'cos', '<constant>'],
                 ['*', '<constant>', '<constant>'],
                 ['+', 'x0', '*', '<constant>', 'x1'],
                 ['pow', '<constant>', 'x0']]
        for row in corpus + extra:
            n_in = _n_const(row)
            for convert in (engine.to_prefix, engine.to_tagged):
                out = engine.simplify(convert(list(row)))
                assert _n_const(out) <= n_in, (row, out)
