"""The realization trust model: which modules a config may cause to be imported.

An operator config declares a **realization** for each operator -- the Python
callable that computes it (``simplipy.operators.sin``, ``np.sin``). The engine
imports the modules those realizations name, because the Python evaluation path
(``prefix_to_infix(realization=True)`` -> :func:`simplipy.codify` ->
:meth:`SimpliPyEngine.code_to_lambda`) resolves the names against them. The Rust
core never needs them: it dispatches on canonical operator names through a native
table, so ``simplify`` and the miner are unaffected by everything in this module.

**Importing is executing.** Python runs a module's top-level code the first time it
is imported, so naming a module in a config is enough to run code -- before any
expression is evaluated and before any operator is called. Since a config can arrive
from anywhere (a colleague, a Hugging Face asset via :meth:`SimpliPyEngine.load`, a
cloned repository), an unrestricted scan makes every config file executable. This
module is the allowlist that closes that (register C1.12, owner-ruled 2026-08-09).

**One spelling per module.** The trusted roots are the canonical spellings:

* ``np``        -- numpy. This, not ``numpy``, is the project's spelling: ``np.pi``
                  and ``np.e`` are normative token grammar (contract sections 1 and
                  10.4) and appear in 316 shipped rules, so the alias is frozen.
                  A realization written ``numpy.sin`` is REFUSED with a hint.
* ``math``
* ``scipy``
* ``simplipy``  -- covers every realization simplipy itself ships.

**Trust is granted from OUTSIDE the config, never by the config.** A
``trusted_modules:`` key inside the YAML would be worthless: the author of the
dangerous line is the author of the permission line, so a hostile file would
authorize itself. The two legitimate surfaces are therefore

1. the caller: ``SimpliPyEngine.from_config(path, trusted_modules=['mylab'])``,
2. the process environment: ``SIMPLIPY_TRUSTED_MODULES=mylab``.

Both are written by whoever decides to trust, in a place the config author cannot
reach. There is deliberately no process-wide setter (mutable global state lets a
library widen trust for an application that never asked).

**What this does and does not buy.** It stops a config from silently executing code
at load, and (with the per-engine evaluation namespace, see
:meth:`SimpliPyEngine.code_to_lambda`) it stops unrelated modules from being
reachable inside evaluated expressions. It does NOT make :func:`simplipy.codify`
safe against a hostile EXPRESSION: compiling and evaluating attacker-supplied source
is unsafe by construction in Python, and no allowlist changes that.
"""
import os
import re
from typing import Iterable

__all__ = [
    'DEFAULT_TRUSTED_MODULES',
    'TRUSTED_MODULES_ENV_VAR',
    'UntrustedModuleError',
    'realization_root',
    'check_realization',
    'resolve_trusted',
    'check_root',
    'package_for',
]

#: The canonical spellings trusted without any opt-in.
DEFAULT_TRUSTED_MODULES = ('math', 'np', 'scipy', 'simplipy')

#: Environment variable carrying extra trusted roots (comma-separated).
TRUSTED_MODULES_ENV_VAR = 'SIMPLIPY_TRUSTED_MODULES'

#: Canonical spelling -> the importable package behind it.
_PACKAGE_OF = {'np': 'numpy'}

#: Refused spelling -> the canonical one to use instead (one spelling per module).
_CANONICAL_SPELLING_OF = {'numpy': 'np'}

# A realization is a module reference when it is a dotted name (`np.sin`,
# `simplipy.operators.sin`). Bare operators (`+`, `*`) and bare builtins (`abs`)
# reference nothing and need no import.
_DOTTED = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+$')


# The two shapes that reference no module and so need no import. Everything a real
# config carries is either _DOTTED or one of these: a survey of the 13 configs on this
# machine (shipped acj-*, base, fixtures) finds 36 distinct dotted realizations and the
# three bare operator symbols, nothing else.
_OPERATOR_SYMBOL = frozenset({'+', '-', '*', '/', '**', '//', '%'})
#: A bare builtin (``abs``). Dunders are refused: no realization is legitimately spelled
#: ``__import__``, and that name is the whole attack.
_BARE_NAME = re.compile(r'^(?!_)[a-zA-Z][a-zA-Z0-9_]*$')
_EXAMPLE_DOTTED = 'np.sin, simplipy.operators.sin'


class UntrustedModuleError(ValueError):
    """A config's realization would import a module that was not trusted."""


def realization_root(realization: str) -> str | None:
    """The module root a realization references, or ``None`` if it references none.

    Examples
    --------
    >>> realization_root('simplipy.operators.sin')
    'simplipy'
    >>> realization_root('np.sin')
    'np'
    >>> realization_root('+') is None
    True
    """
    match = _DOTTED.match(realization.strip())
    return match.group(1) if match else None


def check_realization(realization: str, operator: str) -> str | None:
    """The module root a realization references -- or a refusal. TOTAL, by construction.

    :func:`realization_root` answers "which module does this name?" and returns ``None``
    for everything that names none. That answer was read as "nothing to check", so a
    realization that is not a dotted name was neither checked NOR refused -- it simply
    travelled on to :meth:`SimpliPyEngine.code_to_lambda`, which interpolates it into
    generated source. A realization of ``__import__("...").write_text("...") or abs``
    is not a dotted name; it loaded without complaint and executed at first evaluation
    (register C1.12 / audit B14).

    So the shape is decided here for EVERY realization: a dotted name (root returned,
    trust checked later at import), a bare operator symbol or bare builtin (references
    nothing, no import, allowed), or a refusal. There is no fourth case.
    """
    text = str(realization).strip()
    if _DOTTED.match(text):
        return _DOTTED.match(text).group(1)  # type: ignore[union-attr]
    if text in _OPERATOR_SYMBOL or _BARE_NAME.match(text):
        return None
    raise UntrustedModuleError(
        f"operator {operator!r} declares the realization {text!r}, which is neither a dotted module "
        f"reference ({_EXAMPLE_DOTTED}), a bare operator symbol ({', '.join(sorted(_OPERATOR_SYMBOL))}), "
        f"nor a bare builtin name (abs). A realization is COMPILED INTO GENERATED SOURCE, so anything "
        f"else is arbitrary Python arriving from a config file -- refused at load, before it can run.")


def resolve_trusted(trusted_modules: Iterable[str] | None = None) -> frozenset[str]:
    """The effective trusted set: the defaults, plus the environment, plus the caller.

    Both opt-in surfaces are additive and neither can be revoked by a config, which
    is the whole point (see the module docstring).
    """
    trusted = set(DEFAULT_TRUSTED_MODULES)
    trusted.update(
        part.strip() for part in os.environ.get(TRUSTED_MODULES_ENV_VAR, '').split(',') if part.strip())
    if trusted_modules is not None:
        trusted.update(str(module).strip() for module in trusted_modules if str(module).strip())
    return frozenset(trusted)


def package_for(root: str) -> str:
    """The importable package name behind a canonical root (``np`` -> ``numpy``)."""
    return _PACKAGE_OF.get(root, root)


def check_root(root: str, operators: Iterable[str], trusted: frozenset[str]) -> None:
    """Raise :class:`UntrustedModuleError` unless ``root`` is trusted.

    ``operators`` names the operators whose realizations asked for it, so the error
    points at the line to look at rather than at the engine.
    """
    if root in trusted:
        return

    asked_by = ', '.join(repr(operator) for operator in sorted(operators))
    canonical = _CANONICAL_SPELLING_OF.get(root)
    if canonical is not None:
        raise UntrustedModuleError(
            f"operator(s) {asked_by} spell a realization as {root!r}, but simplipy spells that module "
            f"{canonical!r} -- one spelling per module, and {canonical}.pi / {canonical}.e are normative "
            f"token grammar. Write the realization as {canonical}.<name>.")

    raise UntrustedModuleError(
        f"operator(s) {asked_by} declare a realization that would import the module {root!r}, which is "
        f"not trusted. Importing a module EXECUTES its top-level code, so a config that names one is "
        f"executable input; only {', '.join(DEFAULT_TRUSTED_MODULES)} are trusted by default. "
        f"If you trust it, say so outside the config (a config cannot authorize itself): pass "
        f"trusted_modules=[{root!r}] to SimpliPyEngine.from_config / load / the constructor, or set "
        f"{TRUSTED_MODULES_ENV_VAR}={root} in the environment.")
