"""C1.9: the rule recipe is refused at pickle time when it is ambiguous.

The engine keeps rules twice: ``simplification_rules`` (the Python recipe) and the
compiled core (what ``simplify`` executes). The documented contract is
mutate-then-``compile_rules``, but nothing verified agreement -- and pickling
rebuilds a worker from the LIST, whatever the core was doing. Measured before the
fix: wiping the core's rules while the list kept all 6,661 left 50/400 corpus rows
simplifying differently between the parent and its own unpickled worker, silently.

The fix: the core echoes a digest of the rules it was actually pushed
(``rules_in_sync``), and ``__getstate__`` raises on mismatch. In-sync engines --
every engine that follows the documented contract -- pickle exactly as before.
"""
import copy
import pickle

import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path


@pytest.fixture()
def engine() -> SimpliPyEngine:
    # Function-scoped on purpose: every test mutates rule state.
    return SimpliPyEngine.from_config(acj_config_path())


def test_in_sync_engine_pickles_and_worker_matches(engine: SimpliPyEngine) -> None:
    worker = pickle.loads(pickle.dumps(engine))
    probe = ['-', '0', 'sin', 'x0']
    assert worker.simplify(list(probe)) == engine.simplify(list(probe))


def test_list_mutated_without_compile_refuses_to_pickle(engine: SimpliPyEngine) -> None:
    engine.simplification_rules.append((('sin', '_0'), ('sin', '_0')))
    with pytest.raises(ValueError, match='compile_rules'):
        pickle.dumps(engine)


def test_core_rules_set_directly_refuses_to_pickle(engine: SimpliPyEngine) -> None:
    engine._core.set_rules([])
    with pytest.raises(ValueError, match='compile_rules'):
        pickle.dumps(engine)


def test_deepcopy_takes_the_same_gate(engine: SimpliPyEngine) -> None:
    # deepcopy rides __getstate__: a diverged engine is exactly as ambiguous there.
    engine._core.set_rules([])
    with pytest.raises(ValueError, match='compile_rules'):
        copy.deepcopy(engine)


def test_compile_rules_resyncs_and_pickling_resumes(engine: SimpliPyEngine) -> None:
    engine.simplification_rules.append((('sin', '_0'), ('sin', '_0')))
    engine.compile_rules()
    worker = pickle.loads(pickle.dumps(engine))
    assert len(worker.simplification_rules) == len(engine.simplification_rules)
    probe = ['-', '0', 'sin', 'x0']
    assert worker.simplify(list(probe)) == engine.simplify(list(probe))
