"""THE JUDGE'S MODES: their names, the core's names of their rule sets, the strict mode's names
before 0.15, and the tables keyed by mode.

The mode that preserves what the deployed float64 evaluator computes is ``strict``; it was named
``f64`` before 0.15, and its rule-file entries (``rule_mode=``) used ``'default'``. Both old names
select it with a ``DeprecationWarning`` until 0.16, as the engine's ``Mode`` does. Names that
describe float64's arithmetic rather than the mode (``f64_bar``, ``f64_measure``, ``realised_f6``,
the f64 lane) keep their names."""
import warnings

#: The modes, in the order every table lists them.
MODES = ('strict', 'permissive')

#: The strict mode's names before 0.15, read as ``'strict'`` with a ``DeprecationWarning`` until
#: 0.16.
PRE_015_NAMES = frozenset({'f64', 'default'})

#: Each mode's rule set as the core names it (``rule_mode=``).
RULE_MODES = {'strict': 'default', 'permissive': 'permissive'}


def warn_pre_015(spelling, stacklevel=4):
    """The one warning of the strict mode's pre-0.15 names (``f64``, ``'default'``), removed in
    0.16."""
    warnings.warn(
        f"{spelling} is deprecated and will be removed in simplipy 0.16: the mode is named "
        f"'strict' (Mode.strict)", DeprecationWarning, stacklevel=stacklevel)


def judge_mode(mode):
    """THE JUDGE'S NAME OF A MODE, ``'strict'`` or ``'permissive'``: one of those names in any
    case, a :class:`~simplipy.Mode` member (read by its name; a member still named ``f64`` reads
    as strict), or the strict mode's pre-0.15 names ``'f64'`` and ``'default'``, read as
    ``'strict'`` with a ``DeprecationWarning`` until 0.16. Anything else raises, as the engine's
    mode coercion raises (the retired ``'real'`` with its retirement message)."""
    if isinstance(mode, str):
        key = mode.strip().lower()
        if key in MODES:
            return key
        if key in PRE_015_NAMES:
            warn_pre_015(f'the mode name {mode!r}')
            return 'strict'
    from ..engine import _coerce_mode
    name = _coerce_mode(mode).name
    return 'strict' if name == 'f64' else name


def engine_mode(mode):
    """The engine's :class:`~simplipy.Mode` member of a judge mode (`judge_mode`), whichever name
    the engine gives the strict mode, for a call into the engine or the miner."""
    from ..engine import Mode
    if judge_mode(mode) == 'permissive':
        return Mode.permissive
    members = Mode.__members__
    return members['strict'] if 'strict' in members else members['f64']


class ModeKeyed(dict):
    """A table keyed by the judge's modes (``MODES``, and any other keys it holds) that answers
    the strict mode's pre-0.15 names ``'f64'`` and ``'default'`` with the strict entry and a
    ``DeprecationWarning`` until 0.16: ``MODE_TIERS``, a judgment's ``basis``, ``exact`` and
    ``unread_owed``, :func:`~simplipy.verify.verify_artifact`'s ``modes``."""

    def _key(self, key):
        if isinstance(key, str) and not dict.__contains__(self, key) \
                and key.strip().lower() in PRE_015_NAMES:
            warn_pre_015(f'the mode name {key!r}', stacklevel=5)
            return 'strict'
        return key

    def __getitem__(self, key):
        return dict.__getitem__(self, self._key(key))

    def get(self, key, default=None):
        return dict.get(self, self._key(key), default)

    def __contains__(self, key):
        return dict.__contains__(self, self._key(key))
