"""PER-MODE LAZY RULE LOADING -- lean is opted into, never inherited.

The knob: ``modes=`` on ``load``/``from_config``/``__init__`` names the mode rule
sets built EAGERLY. The default ``'all'`` is byte-identical to the historical
behavior; a set outside a lean selection builds LAZILY on its mode's first use --
once, lock-guarded, announced with ONE loud line -- and :meth:`unload_mode` is the
explicit drop that hands the RAM back (the next use rebuilds). The machinery is
ADDITIVE-ONLY: nothing is ever evicted implicitly.

THE IDENTITY, pinned below on the staged acj-4 cell:

    eager-loaded == lazy-loaded-on-first-use == reloaded-after-unload

byte-identical ``simplify`` outputs per mode, because all three paths install the
same file into the same core through the same ``set_mode_rules`` entry.
"""
import io
import json
import logging
import os
import pickle
import threading
import time
import warnings
from contextlib import contextmanager

import pytest
import yaml

import simplipy
from conftest import legacy_from_config  # noqa: E402
from conftest import acj_config_path, acj_corpus_rules_path, acj_rules_path, require_or_skip
from simplipy import Mode, SimpliPyEngine
from test_mode_rulesets import (
    P_ALL, P_CORPUS, P_DEFAULT, P_OTHER, instantiate, write_artifact)


def _lazy_pair(tmp_path) -> str:
    """The small synthetic two-file artifact the machinery tests run on, byte-for-byte the
    `pair` fixture of the mode-ruleset suite -- rebuilt here because these tests need
    several independent copies."""
    return write_artifact(
        tmp_path,
        rules=[P_ALL, P_DEFAULT],
        corpus=[P_ALL, P_OTHER, P_CORPUS])


@contextmanager
def _announcements(buf: io.StringIO):
    """Capture the `simplipy` logger's records (the first-use announcement) into `buf`."""
    handler = logging.StreamHandler(buf)
    logger = logging.getLogger('simplipy')
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


def _quiet_from_config(*args, **kwargs) -> SimpliPyEngine:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return SimpliPyEngine.from_config(*args, **kwargs)


class TestModesValidation:
    """The knob refuses nonsense LOUDLY, before any file is read."""

    def test_a_bare_mode_name_string_is_refused_with_the_tuple_spelling(self, tmp_path):
        with pytest.raises(TypeError, match=r"did you mean modes=\('f64',\)"):
            _quiet_from_config(_lazy_pair(tmp_path), modes='f64')

    def test_an_unknown_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown mode 'nope'"):
            _quiet_from_config(_lazy_pair(tmp_path), modes=('f64', 'nope'))

    def test_a_non_sequence_raises(self, tmp_path):
        with pytest.raises(TypeError, match='modes must be'):
            _quiet_from_config(_lazy_pair(tmp_path), modes=3)

    def test_a_non_mode_element_raises(self, tmp_path):
        with pytest.raises(TypeError, match='modes entries must be'):
            _quiet_from_config(_lazy_pair(tmp_path), modes=(3,))

    def test_mode_members_and_spellings_are_accepted(self, tmp_path):
        cfg = _lazy_pair(tmp_path)
        for modes in (' ALL ', Mode.f64, (Mode.f64, 'permissive'), ['PERMISSIVE'], ()):
            _quiet_from_config(cfg, modes=modes)

    def test_the_retired_mode_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match='the real mode is retired'):
            _quiet_from_config(_lazy_pair(tmp_path), modes=('f64', 'real'))

    def test_validation_fires_before_the_files_are_read(self, tmp_path):
        """A mistyped `modes` fails on an artifact whose rule files do not even
        exist: the refusal precedes every read."""
        cfg = write_artifact(tmp_path, rules=[], declare=('permissive',))
        with pytest.raises(TypeError, match='did you mean'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                SimpliPyEngine.from_config(cfg, modes='permissive')


class TestLazyMachinery:
    """The deferred set's lifecycle on the small synthetic triple."""

    def test_deferred_sets_are_not_read_and_the_attributes_say_so(self, tmp_path):
        e = _quiet_from_config(_lazy_pair(tmp_path), modes=('f64',))
        assert e.permissive_simplification_rules is None
        # ...but the engine knows the file: the recipe is recorded, not dropped.
        assert set(e._mode_rule_sources) == {'permissive'}

    def test_first_use_builds_once_with_one_loud_line(self, tmp_path):
        e = _quiet_from_config(_lazy_pair(tmp_path), modes=('f64',))
        lhs, rhs = instantiate(P_CORPUS)
        buf = io.StringIO()
        with _announcements(buf):
            first = e.simplify(lhs, mode='permissive')
        assert first == rhs
        assert buf.getvalue().count('simplipy:') == 1
        assert "building the 'permissive' rule set" in buf.getvalue()
        # The set is now materialized -- and the second use is silent.
        assert len(e.permissive_simplification_rules) == 3
        buf = io.StringIO()
        with _announcements(buf):
            assert e.simplify(lhs, mode='permissive') == rhs
        assert buf.getvalue() == ''

    def test_complexity_triggers_the_build_too(self, tmp_path):
        e = _quiet_from_config(_lazy_pair(tmp_path), modes=('f64',))
        buf = io.StringIO()
        with _announcements(buf):
            e.complexity(['sin', 'x0'], mode='permissive')
        assert buf.getvalue().count('simplipy:') == 1
        assert e.permissive_simplification_rules is not None

    def test_modes_all_never_logs_at_call_time(self, tmp_path):
        e = _quiet_from_config(_lazy_pair(tmp_path))
        buf = io.StringIO()
        with _announcements(buf):
            for mode in (Mode.f64, Mode.permissive):
                e.simplify(['sin', 'x0'], mode=mode)
        assert buf.getvalue() == ''

    def test_a_lazy_build_evicts_nothing(self, tmp_path):
        """ADDITIVE-ONLY: building `permissive` on first use leaves the already-loaded
        set exactly where it was (same core object, set still installed)."""
        e = _quiet_from_config(_lazy_pair(tmp_path), modes=('f64',))
        core_before = e._core
        e.simplify(['sin', 'x0'], mode='permissive')
        assert e._core is core_before
        assert e._core.mode_rules_len(rule_mode='default') == 2
        assert e._core.mode_rules_len(rule_mode='permissive') == 3

    def test_unload_then_reload_round_trips(self, tmp_path):
        e = _quiet_from_config(_lazy_pair(tmp_path), modes=('f64',))
        lhs, rhs = instantiate(P_OTHER)
        first = e.simplify(lhs, mode='permissive')
        e.unload_mode('permissive')
        # File-backed: the parsed list is dropped with the core structures...
        assert e.permissive_simplification_rules is None
        assert e._core.mode_rules_len(rule_mode='permissive') is None
        # ...and the next use rebuilds from the recorded file, announced again.
        buf = io.StringIO()
        with _announcements(buf):
            again = e.simplify(lhs, mode='permissive')
        assert again == first == rhs
        assert buf.getvalue().count('simplipy:') == 1

    def test_unload_works_on_an_eager_engine_as_well(self, tmp_path):
        """`modes='all'` engines get the same operational knob: the file recipe is
        recorded at load, so a drop can release the parsed list too."""
        e = _quiet_from_config(_lazy_pair(tmp_path))
        lhs, rhs = instantiate(P_CORPUS)
        assert e.simplify(lhs, mode='permissive') == rhs
        e.unload_mode(Mode.permissive)
        assert e.permissive_simplification_rules is None
        assert e.simplify(lhs, mode='permissive') == rhs

    def test_unload_of_an_in_memory_set_keeps_the_recipe_list(self, tmp_path):
        """A set handed to `__init__` has no file behind it: the wrapper's list IS
        the recipe, so only the compiled core structures are dropped."""
        config = yaml.safe_load(open(acj_config_path()))
        e = SimpliPyEngine(operators=config['operators'], rules=[list(P_ALL)],
                           rules_permissive=[list(P_OTHER)])
        lhs, rhs = instantiate(P_OTHER)
        assert e.simplify(lhs, mode='permissive') == rhs
        e.unload_mode('permissive')
        assert e.permissive_simplification_rules is not None  # the sole recipe stays
        assert e._core.mode_rules_len(rule_mode='permissive') is None  # the built structures went
        buf = io.StringIO()
        with _announcements(buf):
            assert e.simplify(lhs, mode='permissive') == rhs  # rebuilt from the list
        assert 'in-memory recipe' in buf.getvalue()

    def test_lazy_construction_from_in_memory_rules(self, tmp_path):
        """`__init__` honours the knob too: a lean selection defers the PUSH of an
        in-memory set (the list stays, the core builds it on first use)."""
        config = yaml.safe_load(open(acj_config_path()))
        e = SimpliPyEngine(operators=config['operators'], rules=[list(P_ALL)],
                           rules_permissive=[list(P_OTHER)], modes=('f64',))
        assert e.permissive_simplification_rules is not None
        assert e._core.mode_rules_len(rule_mode='permissive') is None
        lhs, rhs = instantiate(P_OTHER)
        buf = io.StringIO()
        with _announcements(buf):
            assert e.simplify(lhs, mode='permissive') == rhs
        assert buf.getvalue().count('simplipy:') == 1

    def test_the_default_set_refuses_to_unload(self, tmp_path):
        e = _quiet_from_config(_lazy_pair(tmp_path))
        with pytest.raises(ValueError, match='cannot be unloaded'):
            e.unload_mode('f64')
        with pytest.raises(ValueError, match='cannot be unloaded'):
            e.unload_mode(Mode.f64)
        with pytest.raises(ValueError, match='the real mode is retired'):
            e.unload_mode('real')

    def test_unload_refuses_nonsense_like_every_mode_surface(self, tmp_path):
        e = _quiet_from_config(_lazy_pair(tmp_path))
        with pytest.raises(ValueError, match="unknown mode 'nope'"):
            e.unload_mode('nope')
        with pytest.raises(TypeError, match='mode must be'):
            e.unload_mode(3)

    def test_unload_of_a_mode_naming_no_set_is_a_noop(self, tmp_path):
        cfg = write_artifact(tmp_path, rules=[P_ALL, P_DEFAULT])
        e = _quiet_from_config(cfg)
        e.unload_mode('permissive')
        e.unload_mode('permissive')  # idempotent
        lhs, rhs = instantiate(P_DEFAULT)
        # ...and the mode still serves the DEFAULT set (the absence fallback),
        # exactly as before the no-op unloads.
        assert e.simplify(lhs, mode='permissive') == rhs

    def test_a_deferred_missing_file_still_warns_at_load(self, tmp_path):
        """The existence check stays at load time even for a set that would not be
        read until first use: a broken artifact layout must not go quiet because
        the load went lean."""
        cfg = write_artifact(tmp_path, rules=[P_ALL], declare=('permissive',))
        with pytest.warns(UserWarning, match='could not be resolved'):
            SimpliPyEngine.from_config(cfg, modes=('f64',))

    def test_compile_rules_keeps_deferred_sets_deferred_and_loaded_sets_loaded(
            self, tmp_path):
        lean = _quiet_from_config(_lazy_pair(tmp_path), modes=('f64',))
        lean.compile_rules()
        assert lean.permissive_simplification_rules is None  # still deferred
        lhs, rhs = instantiate(P_OTHER)
        assert lean.simplify(lhs, mode='permissive') == rhs  # and still lazily loadable
        eager = _quiet_from_config(_lazy_pair(tmp_path))
        eager.compile_rules()
        assert eager._core.mode_rules_len(rule_mode='permissive') == 3  # still loaded

    def test_a_lean_engine_pickles_and_the_worker_stays_lean(self, tmp_path):
        e = _quiet_from_config(_lazy_pair(tmp_path), modes=('f64',))
        back = pickle.loads(pickle.dumps(e))
        assert back.permissive_simplification_rules is None  # deferred travelled deferred
        lhs, rhs = instantiate(P_OTHER)
        buf = io.StringIO()
        with _announcements(buf):
            assert back.simplify(lhs, mode='permissive') == rhs  # loads from the recorded file
        assert buf.getvalue().count('simplipy:') == 1
        # A pickle taken AFTER the load carries the materialized set eagerly.
        back2 = pickle.loads(pickle.dumps(back))
        assert back2.permissive_simplification_rules is not None

    def test_two_threads_first_calls_build_once(self, tmp_path):
        e = _quiet_from_config(_lazy_pair(tmp_path), modes=('f64',))
        lhs, rhs = instantiate(P_CORPUS)
        results: list = []

        def call() -> None:
            results.append(tuple(e.simplify(lhs, mode='permissive')))

        buf = io.StringIO()
        with _announcements(buf):
            threads = [threading.Thread(target=call) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert buf.getvalue().count('simplipy:') == 1  # ONE build, not four
        assert set(results) == {tuple(rhs)}

    def test_a_build_lands_under_concurrent_default_mode_traffic(self, tmp_path):
        """The install fairness gate: a first-use build must land while other
        threads keep hammering the loaded default mode (measured pre-gate: the
        core's exclusive entry stayed closed past a 60 s deadline)."""
        e = _quiet_from_config(_lazy_pair(tmp_path), modes=('f64',))
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                e.simplify(['*', '(-1)', 'asin', 'x0'])

        readers = [threading.Thread(target=reader) for _ in range(2)]
        for t in readers:
            t.start()
        try:
            lhs, rhs = instantiate(P_OTHER)
            buf = io.StringIO()
            with _announcements(buf):
                assert e.simplify(lhs, mode='permissive') == rhs
            assert buf.getvalue().count('simplipy:') == 1
        finally:
            stop.set()
            for t in readers:
                t.join()


class TestConcurrentCoreAccess:
    """Calls racing a push or a retraction on other threads."""

    def test_a_waiting_push_does_not_stall_the_interpreter(self):
        """Every read guard is taken with the GIL released: a call queued behind a waiting
        push (std's RwLock prefers writers) waits without holding the GIL, so pure-Python
        threads keep running. Measured before the fix: the interpreter froze for the long
        reader's whole remaining time (14.9 s on the review's probe)."""
        cfg = acj_config_path()
        require_or_skip(acj_corpus_rules_path(), 'the acj-4 permissive set is not staged')
        config = yaml.safe_load(open(cfg))
        permissive = [tuple(map(tuple, r))
                      for r in json.load(open(acj_corpus_rules_path()))[:500]]
        e = SimpliPyEngine(operators=config['operators'],
                           rules=[tuple(map(tuple, r)) for r in json.load(open(acj_rules_path()))[:500]],
                           rules_permissive=permissive)
        rows = [r for r in json.load(open(os.path.join(
            os.path.dirname(simplipy.__file__), 'verify', 'data', 'raw_skeletons_nv.json')))]
        start = time.perf_counter()
        e._core.ac_canonical_keys(rows, rule_mode='permissive')
        per_batch = time.perf_counter() - start
        # A long in-flight reader: about 1.5 s of work, whatever this machine's speed.
        batch = rows * max(1, int(1.5 / max(per_batch, 1e-6)))
        gaps: list[float] = []
        stop = threading.Event()

        def ticker() -> None:
            last = time.perf_counter()
            while not stop.is_set():
                time.sleep(0.005)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        tick = threading.Thread(target=ticker)
        tick.start()
        try:
            for call in (lambda: e._core.mode_rules_len(rule_mode='permissive'),
                         lambda: e.simplify(['sin', 'x0']),
                         lambda: e._core.interval_finite_ae(['x0'])):
                reader = threading.Thread(target=lambda: e._core.ac_canonical_keys(
                    batch, rule_mode='permissive'))
                pusher = threading.Thread(target=lambda: e._core.set_mode_rules(
                    'permissive', [(list(lhs), list(rhs)) for lhs, rhs in permissive]))
                reader.start()
                time.sleep(0.2)
                pusher.start()
                time.sleep(0.2)
                gaps.clear()
                waited = time.perf_counter()
                call()  # queued behind the waiting push until the reader returns
                waited = time.perf_counter() - waited
                stall = max(gaps, default=0.0)
                reader.join()
                pusher.join()
                assert stall < 0.5, f'the interpreter stalled {stall:.2f} s (call waited {waited:.2f} s)'
        finally:
            stop.set()
            tick.join()


# ---------------------------------------------------------------------------
# THE IDENTITY on a shipped artifact.
# ---------------------------------------------------------------------------

def _identity_config() -> tuple[str, str]:
    """(config_path, label) of the staged acj-4 cell, which ships both mode files."""
    cfg = acj_config_path()
    require_or_skip(cfg, 'the staged acj-4 cell does not resolve in this environment')
    config = yaml.safe_load(open(cfg))
    declared = next((config[k] for k in ('rules_corpus', 'rules_permissive') if config.get(k)),
                    None)
    path = declared and os.path.normpath(os.path.join(os.path.dirname(cfg), declared))
    if not (path and os.path.exists(path)):
        pytest.skip('the staged acj-4 cell ships no permissive file to test the identity on')
    return cfg, 'acj-4'


#: The identity battery: canonical-rewrite probes, arithmetic the AC core folds,
#: certificate-gated collections, and an infix round-trip -- enough surface that a
#: wrong or half-installed set cannot answer all of them identically by luck.
_BATTERY = [
    ['pow', 'x0', '1'],
    ['*', '2', 'x0'],
    ['+', 'x0', 'x0'],
    ['sin', '*', '0', 'x1'],
    ['/', 'x0', 'x0'],
    ['*', '(-1)', 'asin', 'x0'],
    ['+', '*', '2', 'sin', 'x0', '*', '3', 'sin', 'x0'],
    ['log', 'exp', 'abs', 'x2'],
    ['-', 'cosh', 'x0', 'sinh', 'x0'],
    ['/', '+', 'x0', 'x1', '+', 'x1', 'x0'],
    ['pow', 'pow', 'x0', '2', '0.5'],
    ['*', 'inv', 'x3', 'x3'],
]


@pytest.fixture(scope='module')
def identity_engines():
    """One eager and one f64-only-lean engine over the SAME artifact, module-scoped:
    the identity tests read them, nothing mutates them."""
    cfg, label = _identity_config()
    # the cell was mined before 0.15: read through the legacy opt-in
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        eager = legacy_from_config(cfg)
        lean = legacy_from_config(cfg, modes=('f64',))
    return eager, lean, label


class TestTheIdentity:
    """eager == lazy-first-use == reloaded-after-unload, byte-identical per mode."""

    @pytest.mark.parametrize('mode', [Mode.f64, Mode.permissive])
    def test_lazy_equals_eager_per_mode(self, identity_engines, mode):
        eager, lean, _ = identity_engines
        buf = io.StringIO()
        for expression in _BATTERY:
            expected = eager.simplify(expression, mode=mode)
            with _announcements(buf):
                assert lean.simplify(expression, mode=mode) == expected
            # ...in the infix rendering too (the other output path).
            infix = eager.to_infix(expression)
            with _announcements(buf):
                assert lean.simplify(infix, mode=mode) == eager.simplify(
                    infix, mode=mode)
        # However many probes ran, the mode's set was built at most ONCE.
        assert buf.getvalue().count('simplipy:') == (0 if mode is Mode.f64 else 1)

    def test_reload_after_unload_equals_eager(self, identity_engines):
        eager, _, _ = identity_engines
        cfg, _ = _identity_config()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cycled = legacy_from_config(cfg, modes=('f64',))
        for mode in (Mode.permissive,):
            expected = [eager.simplify(x, mode=mode) for x in _BATTERY]
            with _announcements(io.StringIO()):
                first = [cycled.simplify(x, mode=mode) for x in _BATTERY]
                cycled.unload_mode(mode)
                again = [cycled.simplify(x, mode=mode) for x in _BATTERY]
            assert first == expected
            assert again == expected

    def test_the_served_sets_agree_in_size(self, identity_engines):
        """Beyond outputs: once built, the lean engine's per-mode serve counts equal
        the eager engine's -- the same files landed in both cores."""
        eager, lean, _ = identity_engines
        with _announcements(io.StringIO()):
            lean.simplify(['sin', 'x0'], mode=Mode.permissive)
        for rule_mode in ('default', 'permissive'):
            assert (lean._core.mode_rules_len(rule_mode=rule_mode)
                    == eager._core.mode_rules_len(rule_mode=rule_mode))


class TestTheDefaultAndItsMeasuredFootprint:
    """``modes='all'`` stays the default of every loading entry, and every memory figure
    the loading docstrings publish names the artifact it was measured on: the figures
    belong to that artifact and are re-measured when the default artifact changes."""

    @pytest.mark.parametrize('entry', ['__init__', 'from_config', 'load'])
    def test_all_is_the_default(self, entry):
        import inspect
        assert inspect.signature(getattr(SimpliPyEngine, entry)).parameters['modes'].default == 'all'

    @pytest.mark.parametrize('owner', ['class', 'from_config', 'load', 'unload_mode'])
    def test_every_memory_figure_names_its_artifact(self, owner):
        import re
        doc = SimpliPyEngine.__doc__ if owner == 'class' else getattr(SimpliPyEngine, owner).__doc__
        figure = re.compile(r'\d\s*[GM]B\b|\bRSS\b')
        paragraphs = [p for p in re.split(r'\n\s*\n', doc) if figure.search(p)]
        assert paragraphs, f'{owner} states no memory figure'
        for p in paragraphs:
            assert 'acj-5-4-llm' in p, p
