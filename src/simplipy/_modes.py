"""The modes of the engine: :class:`Mode`, the names each mode answers to, and the one coercion
every entry that takes a mode uses."""

import json
import os
import warnings
from enum import Enum, EnumType
from typing import Any


def _warn_deprecated_mode(spelling: str, stacklevel: int = 3) -> None:
    """The one warning of the mode's pre-0.15 names (``f64``, ``'default'``), removed in 0.16."""
    warnings.warn(
        f"{spelling} is deprecated and will be removed in simplipy 0.16: the mode is named "
        f"'strict' (Mode.strict)", DeprecationWarning, stacklevel=stacklevel)


class _ModeType(EnumType):
    """:class:`Mode`'s metaclass: the pre-0.15 member name ``f64`` answers with ``Mode.strict``
    and a ``DeprecationWarning``, as an attribute and as an item, until 0.16."""

    def __getattr__(cls, name: str) -> Any:
        if name == 'f64':
            _warn_deprecated_mode('Mode.f64')
            return cls._member_map_['strict']
        raise AttributeError(f"type object {cls.__name__!r} has no attribute {name!r}")

    def __getitem__(cls, name: str) -> Any:
        if name == 'f64':
            _warn_deprecated_mode("Mode['f64']")
            name = 'strict'
        return super().__getitem__(name)


class Mode(Enum, metaclass=_ModeType):
    """What a simplification preserves.

    - ``strict`` (the default): every step is reproduced by the deployed float64 evaluator
      (within 8 units in the last place), or is true in the real numbers with float64 giving
      both sides the same kind of value at every point (both finite, the same infinity, both
      undefined, a zero's sign where it changes a value on a region). Where an operation
      overflows or underflows, a rewrite may give the true value instead (``exp(x)/exp(x)`` is
      ``1`` also past ``exp``'s overflow). This is the inference and scoring mode. The promise
      is per step: an answer is not promised to agree with float64's evaluation of the input
      within rounding where that evaluation is ill-conditioned (``atanh(tanh(19*sin(x)))``
      becomes ``19*sin(x)``, 19 where float64 evaluates the input to 18.71 at ``x = pi/2``).
    - ``permissive``: reads real numbers, includes every rule ``strict`` holds, and accepts
      what holds up to sign or domain as well (``(x^2)^(1/2)`` is ``x``,
      ``(-1)^<constant>`` is ``<constant>``), trading equivalence for shorter forms. Each
      rule agrees with the real values up to float resolution, or holds in ``strict``; its own
      named simplifications may drop a domain or a sign. A whole answer is not promised to
      agree, because floor, ceil and cancellations magnify a
      rounding-sized difference. It is meant for training-corpus canonicalisation, where the
      training data is generated from the simplified form, so there is no external function
      to violate. Do not use it on an inference or scoring path.

    Each mode names one complete rule set of its own -- ``rules_strict.json`` and
    ``rules_permissive.json``; configs name their files -- so selecting a mode selects a file,
    and what is loaded is what is served.

    ``strict`` was named ``f64`` before 0.15: ``Mode.f64``, ``Mode('f64')`` and the names
    ``'f64'`` and ``'default'`` select it with a ``DeprecationWarning`` and are removed in 0.16.
    The ``real`` mode is retired: asking for it raises ``ValueError``. The names ``SOUND`` and
    ``LOSSY`` were removed in 0.14.0.
    """

    strict = 'strict'
    permissive = 'permissive'

    @classmethod
    def _missing_(cls, value: object) -> 'Mode':
        if isinstance(value, str) and value.strip().lower() == 'real':
            raise ValueError(_REAL_MODE_RETIRED)
        if isinstance(value, str) and value.strip().lower() == 'f64':
            _warn_deprecated_mode("Mode('f64')", stacklevel=5)
            return cls.strict
        return None  # type: ignore[return-value]


#: THE MAP from the public ``Mode`` onto the core's RULE MODE -- the one place the two
#: vocabularies meet on the Python side. ``'default'`` is the core's name for the strict set
#: (the ``rules_strict.json`` file).
_RULE_MODE: dict[Mode, str] = {Mode.strict: 'default', Mode.permissive: 'permissive'}


#: Every name a mode answers to, matched case-insensitively.
_MODE_NAMES: dict[str, Mode] = {'strict': Mode.strict, 'permissive': Mode.permissive}
#: The strict mode's pre-0.15 names, read with a ``DeprecationWarning`` until 0.16:
#: ``'f64'``, and ``'default'``, the name its rule-file entries (``rule_mode=``) used.
_DEPRECATED_MODE_NAMES: dict[str, Mode] = {'f64': Mode.strict, 'default': Mode.strict}
_MODE_CHOICES = "'strict' or 'permissive'"

#: The one refusal of the retired mode, raised wherever a mode is named (every mode argument,
#: a config's ``rules_real`` key, a sidecar's or a canon's mode). Its values remain the
#: reference permissive is defined against.
_REAL_MODE_RETIRED = (
    "the real mode is retired (simplipy 0.15): use mode='strict' for what the deployed float64 "
    "evaluator computes, or mode='permissive' for training-corpus canonicalisation")


def _canon_of_record(rules_path: str) -> str:
    """The canon the rule file at ``rules_path`` was certified in, as its provenance sidecar
    records it: the mode of a per-mode mine (``'strict'``, ``'permissive'``), else
    ``'legacy'``. Mines before per-mode mines (simplipy 0.14 and earlier) mined all three files
    in one canon and recorded no mode: they wrote one sidecar, beside the strict file. A sidecar
    recording the mode's pre-0.15 name ``'f64'`` reads as ``'strict'``, with a
    ``DeprecationWarning`` until 0.16."""
    sidecar = rules_path + '.provenance.json'
    if not os.path.exists(sidecar):
        return 'legacy'
    with open(sidecar, 'r') as f:
        record = json.load(f)
    mode = record.get('mode') if isinstance(record, dict) else None
    if mode is None:
        return 'legacy'
    if mode == 'real':
        raise ValueError(f"the sidecar '{sidecar}' records the mode 'real': {_REAL_MODE_RETIRED}")
    if mode == 'f64':
        _warn_deprecated_mode(f"the sidecar '{sidecar}' records the mode 'f64', which")
        return Mode.strict.name
    if mode not in {m.name for m in Mode}:
        raise ValueError(f"the sidecar '{sidecar}' records the mode {mode!r}, which this "
                         f"simplipy does not know: expected one of {[m.name for m in Mode]}")
    return str(mode)


def _coerce_mode(mode: 'Mode | str') -> Mode:
    """A :class:`Mode` member, or one of its names in any case; anything else raises.

    The one coercion every entry that takes a mode uses (``mode=``, ``modes=``,
    ``rule_mode=``): the strict mode's pre-0.15 names ``'f64'`` and ``'default'`` select it
    with a ``DeprecationWarning`` until 0.16, and the retired ``'real'`` raises the retirement
    message. A bare int does not bind: only members and their names select a mode.
    """
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, str):
        if mode.strip().lower() == 'real':
            raise ValueError(_REAL_MODE_RETIRED)
        old = _DEPRECATED_MODE_NAMES.get(mode.strip().lower())
        if old is not None:
            _warn_deprecated_mode(f"the mode name {mode!r}", stacklevel=4)
            return old
        match = _MODE_NAMES.get(mode.strip().lower())
        if match is None:
            raise ValueError(
                f"unknown mode {mode!r}: expected {_MODE_CHOICES} (or a simplipy.Mode)")
        return match
    raise TypeError(
        f"mode must be a simplipy.Mode or one of {_MODE_CHOICES}, "
        f"not {type(mode).__name__} ({mode!r})")


def _coerce_rule_mode(mode: 'Mode | str') -> str:
    """The core's name for a mode's rule set (``'default'``, ``'permissive'``), from any
    spelling :func:`_coerce_mode` accepts."""
    return _RULE_MODE[_coerce_mode(mode)]


#: The wrapper attribute carrying each NON-DEFAULT rule mode's own set -- the set the lazy
#: machinery (``modes=...`` at construction, :meth:`SimpliPyEngine.unload_mode`) can defer
#: and drop. The default (strict) set is deliberately absent from this map: it is the core's
#: construction substrate (the canon context, the dedup key, and the set every absent mode
#: falls back to), so it is always built eagerly and never unloadable.
_MODE_RULES_ATTR: dict[str, str] = {
    'permissive': 'permissive_simplification_rules',
}
