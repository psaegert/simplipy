"""The public surface refuses malformed input loudly, and keeps its promises under odd input.

Pre-audit review of the public API (angle "api", 2026-11-27, at 3a6ed3a). Each test pins a
behaviour that was measured to hold, so a regression fails here, by name:

* malformed expressions raise ``ValueError`` naming the defect, in every entry that reads one
  (``simplify``, ``complexity``, ``written_complexity``, the three conversions, ``prettify``,
  ``mask``), and ``is_valid`` answers ``False`` for them instead of raising;
* the 4,096-token cap refuses before any recursion can overflow a stack;
* token sequences: ``list``, ``tuple`` and 1-D ``ndarray`` of dtype ``U``, ``S`` (F21) and ``O``
  read alike and answer in the container and dtype kind they came in; a non-token element or
  array is refused with an error naming it;
* defined functions and aliases at every entry;
* the strict mode's pre-0.15 names select it with a ``DeprecationWarning`` attributed to the
  caller, and the retired ``real`` is refused everywhere;
* engines and ``Mode`` members pickle at every protocol, and concurrent ``simplify`` calls on one
  engine answer as serial ones do;
* answers do not depend on the interpreter's hash seed.

The review's findings API-1 to API-5, API-7 and API-8 were strict xfails in
``test_api_findings.py`` until the 0.15 fix round fixed them; their tests are here now, each under
a comment naming its finding.
"""
import copy
import os
import pickle
import re
import subprocess
import sys
import threading
import warnings

import numpy as np
import pytest

from simplipy import Mode, SimpliPyEngine
from conftest import acj_config_path, legacy_from_config, require_or_skip

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BARE_CONFIG = os.path.join(_ROOT, 'assets', 'engines', 'acj2-4', 'config.yaml')

#: Every entry that reads an expression and refuses a malformed one.
_READERS = ('simplify', 'complexity', 'written_complexity', 'to_prefix', 'to_infix', 'to_tagged',
            'prettify', 'mask')


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    """The shipped 0.15 vocabulary with its defined functions, no rules: reading, conversion and
    the engine's own steps, independent of any mined file."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # "NO simplification rules"
        return SimpliPyEngine.from_config(_BARE_CONFIG)


@pytest.fixture(scope='module')
def ruled() -> SimpliPyEngine:
    require_or_skip(acj_config_path(), 'the acj-4 fixture is not staged')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return legacy_from_config(acj_config_path())


# ---- malformed expressions --------------------------------------------------------------------

_MALFORMED_INFIX = [
    ('sin(x0, x1)', 'sin takes 1 argument, got 2'),
    ('pow(x0)', 'pow takes 2 arguments, got 1'),
    ('sin()', 'sin takes 1 argument, got 0'),
    ('zeta(x0)', "unknown function 'zeta'"),
    ('x0 +', 'missing operand'),
    ('x0, x1', "a comma outside a call's argument list"),
    ('NaN', 'reserved numeric spelling'),
    ('Infinity', 'reserved numeric spelling'),
    (' + '.join(['x0'] * 3000), 'prefix expression too long'),
]


@pytest.mark.parametrize('text, message', _MALFORMED_INFIX)
@pytest.mark.parametrize('entry', [r for r in _READERS if r != 'to_infix'])
def test_malformed_infix_is_refused_by_every_reader(engine, entry, text, message) -> None:
    with pytest.raises(ValueError, match=message.replace('(', r'\(').replace(')', r'\)')):
        getattr(engine, entry)(text)


@pytest.mark.parametrize('text', [t for t, _ in _MALFORMED_INFIX] + ['', 'sin(x0))', 'x0 x1'])
def test_is_valid_answers_false_on_malformed_infix_never_raises(engine, text) -> None:
    assert engine.is_valid(text) is False


# API-1 (fixed in the 0.15 fix round): a character the infix reader has no token for used to be
# dropped silently, so text outside the grammar read as another expression (`|x0|` as `x0`, `x0!`
# as `x0`, `?0 + x1` as `0 + x1`, a fullwidth `x0` as the empty expression). It is refused now,
# naming the character and its position, by every reader, `read_infix` and `infix_to_prefix`
# included, and `is_valid` answers False.
_OUTSIDE_THE_GRAMMAR = [('|x0|', "'|' at position 0"), ('x0!', "'!' at position 2"),
                        ('?0 + x1', "'?' at position 0"), ('~x0', "'~' at position 0"),
                        ('<constant_a-b>*x0', "'<' at position 0"), ('ｘ０', "'ｘ' at position 0"),
                        ('[x0]', "'[' at position 0"), ('x0\x00 + 1', r"'\0' at position 2"),
                        ('x0 $ x1', "'$' at position 3"), ('2é', "'é' at position 1")]


@pytest.mark.parametrize('text, named', _OUTSIDE_THE_GRAMMAR)
def test_a_character_outside_the_grammar_is_refused(engine, text, named) -> None:
    for entry in _READERS + ('read_infix', 'infix_to_prefix'):
        with pytest.raises(ValueError, match=re.escape(f'no token for the character {named}')):
            getattr(engine, entry)(text)
    assert engine.is_valid(text) is False


def test_a_hexadecimal_spelling_is_an_implicit_product_not_a_character_outside_the_grammar(engine) -> None:
    """`0x1F` has a token for every character: the numeral `0` and the variable `x1F`, an
    implicit product (as `0x10` is), so it reads and is not refused."""
    assert engine.read_infix('0x1F') == ['*', '0', 'x1F']


_MALFORMED_TOKENS = [
    (['+', 'x0'], None),
    (['sin', 'x0', 'x1'], None),
    (['+'], None),
    (['frob', 'x0'], None),
    (['<add>', 'x0', 'x1'], None),
    (['x0', '</add>'], None),
    (['<add>', 'x0', '<mul>', 'x1', '</add>', '</mul>'], None),
    (['+', 'nan', 'x0'], 'reserved numeric spelling'),
    (['+', '1/0', 'x0'], 'reserved numeric spelling'),
    (['+', 'x 0', 'x0'], 'tokens must be non-empty and whitespace-free'),
    (['+', '', 'x0'], 'tokens must be non-empty and whitespace-free'),
    (['+'] * 2500 + ['x0'] * 2501, 'prefix expression too long'),
]


@pytest.mark.parametrize('tokens, message', _MALFORMED_TOKENS)
@pytest.mark.parametrize('entry', _READERS)
def test_malformed_tokens_are_refused_by_every_reader(engine, entry, tokens, message) -> None:
    with pytest.raises(ValueError, match=message):
        getattr(engine, entry)(list(tokens))


@pytest.mark.parametrize('tokens', [t for t, m in _MALFORMED_TOKENS if m is None])
def test_is_valid_answers_false_on_malformed_tokens(engine, tokens) -> None:
    assert engine.is_valid(list(tokens)) is False


# API-2 (fixed in the 0.15 fix round): `is_valid` asked only the core's arity oracle, so it
# answered True for token sequences every reader refuses: a reserved numeric spelling, an empty
# or spaced token, and a sequence past the 4,096-token cap. It runs the readers' token grammar
# (`check_form`) too now.
@pytest.mark.parametrize('tokens', [['+', 'nan', 'x0'], ['+', 'inf', 'x0'], ['+', '1/0', 'x0'],
                                    ['+', '1/-3', 'x0'], ['+', '', 'x0'], ['+', 'x 0', 'x0'],
                                    ['sin'] * 4096 + ['x0']])
def test_is_valid_agrees_with_the_readers(engine, tokens) -> None:
    with pytest.raises(ValueError):
        engine.to_prefix(tokens)
    assert engine.is_valid(tokens) is False


def test_the_token_cap_refuses_before_any_recursion(engine) -> None:
    deep = ['sin'] * 4096 + ['x0']
    for entry in _READERS:
        with pytest.raises(ValueError, match=r'too long \(4097 tokens > 4096\)'):
            getattr(engine, entry)(deep)
    assert engine.is_valid(deep) is False
    # one under the cap reads, and the core's recursion holds at that depth
    assert engine.simplify(['sin'] * 4095 + ['x0']) == ['sin'] * 4095 + ['x0']


# API-4 (fixed in the 0.15 fix round): below the 4,096-token cap, `mask` (and through it
# `normalization.to_skeleton`) recursed in Python and raised RecursionError from depth 1,000 on.
# The role walk is iterative now; `as_callable` keeps Python's parser limit of 200 nested
# parentheses, which its docstring states.
def test_mask_reads_every_depth_the_core_reads(engine) -> None:
    from simplipy.normalization import to_skeleton
    deep = ['sin'] * 4095 + ['x0']
    assert engine.simplify(deep) == deep
    assert engine.mask(deep) == deep
    assert engine.mask(deep, 'fittable') == deep
    tagged = ['sin'] * 3000 + ['<mul>', '2', 'x0', '</mul>']
    assert engine.mask(tagged) == ['sin'] * 3000 + ['<mul>', '<constant>', 'x0', '</mul>']
    assert to_skeleton(['sin'] * 3000 + ['*', '2', 'x0'], engine) == ['sin'] * 3000 + ['*', '<constant>', 'x0']
    engine.as_callable(['sin'] * 200 + ['x0'])
    with pytest.raises(SyntaxError, match='too many nested parentheses'):
        engine.as_callable(['sin'] * 201 + ['x0'])


@pytest.mark.parametrize('literal', ['1e999999999999', '1e-999999999999', '1.' + '3' * 5000, '9' * 4000])
def test_extreme_literals_are_read_without_error(engine, literal) -> None:
    """A literal far past float64's range, or thousands of digits long, is a number like any
    other: read, priced and written back without an overflow or a hang."""
    assert engine.is_valid(literal)
    assert engine.complexity(literal) > 0
    assert engine.written_complexity(literal) > 0
    assert engine.to_prefix(literal) == [literal]


# ---- token sequences and their containers -------------------------------------------------------

_TOKENS = ['+', 'x0', '*', '2', 'x1']


@pytest.mark.parametrize('array', [
    np.array(_TOKENS), np.array(_TOKENS, dtype='S'), np.array(_TOKENS, dtype=object),
    np.array([t for t in _TOKENS for _ in (0, 1)])[::2],
], ids=['U', 'S', 'O', 'U-strided'])
def test_string_arrays_read_as_their_tokens_and_answer_in_kind(engine, array) -> None:
    assert engine.to_prefix(array) == _TOKENS
    assert engine.is_valid(array) is True
    answer = engine.simplify(array)
    assert isinstance(answer, np.ndarray) and answer.dtype.kind == array.dtype.kind
    assert [t.decode() if isinstance(t, bytes) else str(t) for t in answer.tolist()] == \
        engine.simplify(_TOKENS)


def test_a_fold_wider_than_every_input_token_is_not_truncated(engine) -> None:
    """The answer's string width is the answer's, not the input's (`1/0` folds to a 12-character
    token from 1-character ones)."""
    for dtype in ('U', 'S'):
        answer = engine.simplify(np.array(['/', '1', '0'], dtype=dtype))
        assert answer.tolist() in (['float("inf")'], [b'float("inf")'], ['float("nan")'], [b'float("nan")'])


@pytest.mark.parametrize('array, message', [
    (np.array([_TOKENS]), 'one-dimensional'),
    (np.array('x0'), 'one-dimensional'),
    (np.array([1.0, 2.0]), 'string-like'),
    (np.array([1, 2]), 'string-like'),
    (np.zeros(2, dtype='V4'), 'string-like'),
])
def test_non_token_arrays_are_refused(engine, array, message) -> None:
    for entry in ('simplify', 'complexity', 'to_prefix', 'to_tagged', 'prettify'):
        with pytest.raises(ValueError, match=message):
            getattr(engine, entry)(array)
    with pytest.raises(TypeError):
        engine.is_valid(array)


@pytest.mark.parametrize('element', [True, None, b'x0', ['x0'], 1j])
def test_non_token_elements_are_refused_naming_their_type(engine, element) -> None:
    for entry in ('simplify', 'complexity', 'written_complexity', 'to_prefix', 'to_infix',
                  'to_tagged', 'prettify'):
        with pytest.raises(ValueError, match=f'a token must be a str or a number, not {type(element).__name__}'):
            getattr(engine, entry)(['+', element, 'x0'])


def test_a_tuple_answers_as_a_tuple(engine) -> None:
    assert engine.simplify(('+', 'x0', 'x0')) == ('*', '2', 'x0')


@pytest.mark.parametrize('expression', [12, 1.5, {'+': 1}, object()])
def test_a_non_expression_type_is_a_type_error(engine, expression) -> None:
    for entry in ('simplify', 'complexity', 'to_prefix', 'to_tagged', 'to_infix', 'is_valid'):
        with pytest.raises(TypeError):
            getattr(engine, entry)(expression)


# ---- defined functions and aliases at every entry -----------------------------------------------

def test_a_defined_function_reads_at_every_entry(engine) -> None:
    assert engine.read_infix('sqrt(x0)') == ['sqrt', 'x0']
    assert engine.to_prefix('sqrt(x0)') == ['sqrt', 'x0']
    assert engine.to_tagged('sqrt(x0)') == ['sqrt', 'x0']
    assert engine.to_infix(['sqrt', 'x0']) == 'sqrt(x0)'
    assert engine.is_valid('sqrt(x0)') and engine.is_valid(['sqrt', 'x0'])
    # every canonical entry reads the definition
    assert engine.simplify('sqrt(x0)') == engine.simplify('rootn(x0, 2)')
    assert engine.simplify(['sqrt', 'x0']) == engine.simplify(['rootn', 'x0', '2'])
    assert engine.complexity('sqrt(x0)') == engine.complexity('rootn(x0, 2)')
    assert engine.prettify('sqrt(x0)') == engine.prettify('rootn(x0, 2)')
    assert engine.mask('sqrt(x0)') == engine.mask('rootn(x0, 2)')
    # a defined function at the wrong arity is malformed, like an operator
    with pytest.raises(ValueError):
        engine.simplify(['sqrt', 'x0', 'x1'])
    assert engine.is_valid(['sqrt', 'x0', 'x1']) is False


@pytest.mark.parametrize('alias, operator', [('ln', 'log'), ('arcsin', 'asin'), ('Abs', 'abs')])
def test_an_alias_reads_as_its_operator_in_infix(engine, alias, operator) -> None:
    for entry in ('simplify', 'to_prefix', 'complexity', 'written_complexity'):
        assert getattr(engine, entry)(f'{alias}(x0)') == getattr(engine, entry)(f'{operator}(x0)')
    assert engine.convert_expression([alias, 'x0']) == [operator, 'x0']


# API-5 (fixed in the 0.15 fix round): an alias naming another declared operator loaded silently,
# and then the infix reader and the prefix readers disagreed (with `sin: {alias: [cos]}`,
# `simplify('cos(x0)')` was `sin(x0)` and `simplify(['cos', 'x0'])` was `cos(x0)`); odd operator
# names and aliases loaded too. A name is a function name now: an identifier starting with a
# letter, not a constant's or a variable's spelling, the core's `+ - * /` aside; an alias names
# one operator and is none.
def _with_operator(**changes):
    from simplipy.io import load_config
    config = load_config(_BARE_CONFIG)
    operators = copy.deepcopy(config['operators'])
    for name, spec in changes.items():
        operators[name] = dict(operators.get(name, operators['sin']), **spec)
    defined = dict(config.get('defined_functions') or {})
    return SimpliPyEngine(operators, [], defined_functions=defined)


@pytest.mark.parametrize('alias, message', [
    ('cos', 'which is an operator of this config'), ('sin', 'which is an operator of this config'),
    ('ln', 'declared by both'), ('3', 'not a function name'), ('x1', 'spelling of a variable'),
    ('pi', 'infix spelling of a constant'), ('np.sin', 'not a function name')])
def test_an_alias_naming_another_operator_or_no_function_is_refused(alias, message) -> None:
    with pytest.raises(ValueError, match=message):
        _with_operator(sin={'alias': [alias]})


@pytest.mark.parametrize('name, message', [
    ('', 'not a function name'), ('(', 'not a function name'), ('<add>', 'not a function name'),
    ('np.pi', 'not a function name'), ('_0', 'not a function name'), ('x0', 'spelling of a variable'),
    ('v12', 'spelling of a variable'), ('e', 'infix spelling of a constant')])
def test_an_odd_operator_name_is_refused(name, message) -> None:
    with pytest.raises(ValueError, match=message):
        _with_operator(**{name: {'alias': []}})


def test_a_defined_function_name_follows_the_operator_name_rule() -> None:
    from simplipy.io import load_config
    config = load_config(_BARE_CONFIG)
    defined = dict(config['defined_functions'], x0={'arity': 1, 'definition': 'sin(_0)'})
    with pytest.raises(ValueError, match='spelling of a variable'):
        SimpliPyEngine(config['operators'], [], defined_functions=defined)


def test_the_shipped_names_and_ordinary_new_ones_load() -> None:
    _with_operator(sin={'alias': ['Sin', 'sine']}, xy={'alias': []}, f_2={'alias': ['g']})


# API-7 (fixed in the 0.15 fix round): `as_code` interpolated every leaf token and every
# `variables=` entry into Python source, and the token grammar admits any whitespace-free string
# as a variable, so a token that is Python source ran when the compiled function was called. A
# leaf is a number, a special constant or a Python identifier now, a variable an identifier.
def test_as_code_refuses_a_leaf_that_is_not_an_identifier(engine, tmp_path) -> None:
    mark = tmp_path / 'ran'
    payload = f"__import__('pathlib').Path({str(mark)!r}).touch()"
    with pytest.raises(ValueError, match='neither a number, a constant nor a Python identifier'):
        engine.as_callable(['+', 'x0', payload], variables=['x0'])(np.array([1.0]))
    with pytest.raises(ValueError, match='not a Python identifier'):
        engine.as_callable(['+', 'x0', 'x1'], variables=['x0', f'x1={payload}'])(np.array([1.0]))
    assert not mark.exists()
    for leaf, message in (('?0', 'neither a number'), ('<constant>', "anonymous '<constant>'"),
                          ('lambda', 'neither a number')):
        with pytest.raises(ValueError, match=message):
            engine.as_code(['*', '2', leaf])
    # numbers, constants, identifiers and named constants compile as before
    f = engine.as_callable(['+', '*', '(-3)', 'x_1', '+', 'np.pi', '*', '1/4', '<constant_a>'])
    assert f(np.array([1.0]), 2.0)[0] == -3.0 + np.pi + 0.5


# API-3 (fixed in the 0.15 fix round): `mask` returned some malformed tagged sequences unchanged
# instead of raising, while every other reader refused them. It runs the readers' check now.
@pytest.mark.parametrize('tokens', [['<add>', '</add>'], ['<add>', '<sub>', 'x0', '</add>'],
                                    ['<mul>', 'x0', '<sub>', 'x1', '</mul>']])
def test_mask_refuses_a_malformed_tagged_sequence(engine, tokens) -> None:
    with pytest.raises(ValueError):
        engine.to_prefix(tokens)
    with pytest.raises(ValueError):
        engine.mask(tokens)


# API-8 (fixed in the 0.15 fix round): a pre-0.15 engine pickle unpickled in 0.15 without the gate
# `from_config` applies to the same rule files, and served its 0.14-mined rules in the strict
# canon (measured with a real simplipy 0.14.7 pickle of the acj-4 engine: 5,338 + 5,499 rules,
# no warning). `__setstate__` refuses such a state now: the 0.14.7 state's shape (no
# `_rule_canons`, a `real_simplification_rules` attribute) is simulated here.
def test_a_pre_015_engine_pickle_is_refused(engine) -> None:
    from simplipy.compat import IncompatibleArtifactError
    state = engine.__getstate__()
    state.pop('_rule_canons', None)
    state.pop('defined_functions', None)
    state['real_simplification_rules'] = None
    state['simplification_rules'] = [(('sin', '_0'), ('_0',))]  # rules the 0.15 gate never saw
    no_real = {k: v for k, v in state.items() if k != 'real_simplification_rules'}
    for old in (state, no_real, dict(state, _rule_canons={})):
        clone = SimpliPyEngine.__new__(SimpliPyEngine)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(IncompatibleArtifactError, match='simplipy 0.14 or earlier'):
                clone.__setstate__(dict(old))


# ---- the mode's names -------------------------------------------------------------------------

def test_the_deprecated_names_select_strict_and_warn_at_the_caller(engine) -> None:
    spellings = [lambda: Mode.f64, lambda: Mode['f64'], lambda: Mode('f64'),
                 lambda: engine.simplify('x0 + x0', mode='f64'),
                 lambda: engine.simplify('x0 + x0', mode='default'),
                 lambda: engine.complexity('x0', mode='F64'),
                 lambda: engine.prettify('x0', mode='f64')]
    for spelling in spellings:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            spelling()
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1, [str(w.message) for w in caught]
        assert 'removed in simplipy 0.16' in str(deprecations[0].message)
        assert deprecations[0].filename == __file__  # attributed to the caller's line
    assert Mode.f64 is Mode.strict
    assert 'f64' not in Mode.__members__  # not a member: iteration and listings show two modes
    assert list(Mode) == [Mode.strict, Mode.permissive]


@pytest.mark.parametrize('call', [
    lambda e: Mode('real'), lambda e: e.simplify('x0', mode='real'),
    lambda e: e.complexity('x0', mode='real'), lambda e: e.prettify('x0', mode='real'),
    lambda e: SimpliPyEngine.from_config(_BARE_CONFIG, modes=('real',)),
])
def test_the_retired_real_mode_is_refused_everywhere(engine, call) -> None:
    with pytest.raises(ValueError, match='the real mode is retired'):
        call(engine)


@pytest.mark.parametrize('mode', [0, None, 1.5, b'strict'])
def test_a_mode_of_another_type_is_a_type_error(engine, mode) -> None:
    with pytest.raises(TypeError, match='mode must be a simplipy.Mode'):
        engine.simplify('x0', mode=mode)


# ---- pickling, threads, determinism ------------------------------------------------------------

@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
def test_engines_and_modes_pickle_at_every_protocol(engine, protocol) -> None:
    clone = pickle.loads(pickle.dumps(engine, protocol=protocol))
    for text in ('sqrt(x0)*sqrt(x0)', 'cot(x0) + x0/x0', '2.5*x1 - 0.5*x1'):
        assert clone.simplify(text) == engine.simplify(text)
        assert clone.complexity(text) == engine.complexity(text)
    assert clone.defined_functions == engine.defined_functions
    assert [pickle.loads(pickle.dumps(m, protocol=protocol)) for m in Mode] == list(Mode)
    assert copy.deepcopy(engine).simplify('cot(x0)') == engine.simplify('cot(x0)')


def test_concurrent_simplify_on_one_engine_answers_as_serial(ruled) -> None:
    ops = ['sin', 'cos', 'exp', 'log', 'tanh', 'abs']
    texts = [f'{ops[i % 6]}(x{i % 3}) * (x{(i + 1) % 3} + {i % 4 + 1}) - {ops[(i + 2) % 6]}(x0*x1)/(x{i % 3} + 1)'
             for i in range(60)]
    serial = {m: [ruled.simplify(t, mode=m) for t in texts] for m in ('strict', 'permissive')}
    failures: list = []

    def work() -> None:
        try:
            for m in ('strict', 'permissive'):
                for t, expected in zip(texts, serial[m]):
                    if ruled.simplify(t, mode=m) != expected:
                        failures.append((m, t))
        except Exception as error:  # noqa: BLE001 -- any error is a failure of the contract
            failures.append(repr(error))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


def test_answers_do_not_depend_on_the_hash_seed() -> None:
    code = (
        'import warnings; warnings.simplefilter("ignore")\n'
        'from simplipy import SimpliPyEngine\n'
        f'e = SimpliPyEngine.from_config({_BARE_CONFIG!r})\n'
        'texts = ["x2*x1 + x0*x2 + x1*x0 + 3*x2*x1", "sin(x1)*(x0 + x1) - x1*sin(x1)",'
        ' "exp(x0)*exp(x1)/exp(x0 + x1)", "1/(x0 + x1) + 1/(x1 + x0)"]\n'
        'print([e.simplify(t, mode=m) for m in ("strict", "permissive") for t in texts])\n')
    answers = set()
    for seed in ('0', '1', '4242'):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        answers.add(subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                                   text=True, check=True).stdout)
    assert len(answers) == 1, answers
