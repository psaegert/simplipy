"""Engines must survive pickling: the compiled core (``simplipy._core.Engine``) has no
serialization surface, so the wrapper pickles its construction recipe (operator config +
rules) and rebuilds the core on unpickle. The reported failure mode was data-generation
workers in a ``multiprocessing`` spawn context: every argument crosses the process
boundary through pickle, and the engine died with ``TypeError: cannot pickle
'simplipy._core.Engine' object``.
"""
import multiprocessing as mp
import pickle

from simplipy import SimpliPyEngine

_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "inv": {"realization": "simplipy.operators.inv", "alias": ["inverse"], "inverse": "inv", "arity": 1, "precedence": 4, "commutative": False},
    "/": {"realization": "simplipy.operators.div", "alias": [], "inverse": "*", "arity": 2, "precedence": 2, "commutative": False},
    "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
    # Added 2026-08-07 for the probe below: the vocabulary a CROSS-FUNCTION identity needs
    # (`inv` was already here).
    "cosh": {"realization": "np.cosh", "alias": [], "inverse": "acosh", "arity": 1, "precedence": 3, "commutative": False},
    "asinh": {"realization": "np.arcsinh", "alias": [], "inverse": "sinh", "arity": 1, "precedence": 3, "commutative": False},
    "cos": {"realization": "np.cos", "alias": [], "inverse": "acos", "arity": 1, "precedence": 3, "commutative": False},
    "atan": {"realization": "np.arctan", "alias": [], "inverse": "tan", "arity": 1, "precedence": 3, "commutative": False},
}

# A rewrite the BARE engine provably does not perform (verified by the negative
# controls below), so a firing after the round trip proves the rebuilt core LOADED
# the rules rather than merely carrying the list.
# A CROSS-FUNCTION identity, deliberately: `1/cosh(asinh t) = cos(atan t)`.
# This slot held `log exp ?0 -> ?0` until 2026-08-07, when the inverse-pair table
# made that CONSTRUCTOR-NATIVE and the negative control below stopped being a
# fixpoint -- the test would have started passing vacuously. An identity BETWEEN
# different functions is not a collapse any constructor table can absorb.
PROBE = ['inv', 'cosh', 'asinh', 'x1']
INVERSE_RULE = (('inv', 'cosh', 'asinh', '?0'), ('cos', 'atan', '?0'))

PROBES = [
    ['+', 'x0', 'x1'],
    ['log', 'exp', 'x2'],
    ['*', 'x0', 'inv', 'x0'],
    ['+', 'x0', 'neg', 'x0'],
]


def _spawn_worker(engine: SimpliPyEngine, out: 'mp.Queue') -> None:
    out.put([engine.simplify(p) for p in PROBES])


class TestEnginePickle:
    def test_round_trip_parity(self) -> None:
        engine = SimpliPyEngine(_OPERATORS, rules=[INVERSE_RULE])
        clone = pickle.loads(pickle.dumps(engine))
        assert clone.simplification_rules == engine.simplification_rules
        assert clone._operators_config == engine._operators_config
        for probe in PROBES:
            assert clone.simplify(probe) == engine.simplify(probe)

    def test_rebuilt_core_fires_rules(self) -> None:
        # Negative control first: without the rule this probe is a fixpoint, so the
        # positive assertion cannot pass vacuously through built-in simplification.
        bare = SimpliPyEngine(_OPERATORS)
        assert bare.simplify(PROBE) == PROBE
        engine = SimpliPyEngine(_OPERATORS, rules=[INVERSE_RULE])
        clone = pickle.loads(pickle.dumps(engine))
        assert clone.simplify(PROBE) == ['cos', 'atan', 'x1']

    def test_runtime_rule_changes_survive(self) -> None:
        # Rules added AFTER construction (the documented mutate-then-compile_rules
        # path) are part of the pickled state.
        engine = SimpliPyEngine(_OPERATORS)
        assert engine.simplify(PROBE) == PROBE
        engine.simplification_rules.append(INVERSE_RULE)
        engine.compile_rules()
        clone = pickle.loads(pickle.dumps(engine))
        assert clone.simplify(PROBE) == ['cos', 'atan', 'x1']

    def test_spawn_context_worker(self) -> None:
        # The reported scenario: a data-generation worker in a SPAWN context (fresh
        # interpreter, every argument crosses via pickle).
        engine = SimpliPyEngine(_OPERATORS, rules=[INVERSE_RULE])
        ctx = mp.get_context('spawn')
        out: 'mp.Queue' = ctx.Queue()
        proc = ctx.Process(target=_spawn_worker, args=(engine, out))
        proc.start()
        try:
            results = out.get(timeout=120)
        finally:
            proc.join(timeout=30)
        assert proc.exitcode == 0
        assert results == [engine.simplify(p) for p in PROBES]
