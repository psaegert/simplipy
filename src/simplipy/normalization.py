"""Expression-token normalization helpers.

Canonicalize a prefix token sequence so two expressions that are "the same" compare
equal: variable tokens (``v1``/``x1`` style, case-insensitive) are renamed to a stable
``x{n}``, and numeric literals are folded to a ``<constant>`` placeholder (for the
*skeleton* form) or kept (for the *expression* form).

These are pure-string helpers (no engine state) so every consumer -- holdout matching,
symbolic-recovery scoring, dataset preparation -- can reuse identical behavior. They were
relocated here (0.3.1) from flash-ansr, behavior-identical, so the canonicalizer lives at
the shared expression-engine leaf that symbolic-data, flash-ansr, and srbf all depend on.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

__all__ = ["normalize_variable_token", "normalize_skeleton", "normalize_expression"]

_VAR_TOKEN_PATTERN = re.compile(r"^[vx](\d+)$", re.IGNORECASE)


def normalize_variable_token(token: str) -> tuple[str, bool]:
    """Return ``(normalized_token, is_variable)``.

    Recognizes tokens like ``v1`` or ``x2`` (case-insensitive) and returns them as
    ``x{n}``. Non-variable tokens are returned unchanged with ``is_variable=False``.
    """
    match = _VAR_TOKEN_PATTERN.match(token)
    if match:
        return f"x{int(match.group(1))}", True
    return token, False


def normalize_skeleton(tokens: Sequence[str | Any] | None) -> list[str] | None:
    """Normalize a skeleton/prefix into a list of canonical tokens.

    - Variable tokens (``v1``/``x1``) are normalized to ``x{n}``.
    - Numeric literals are converted to the ``"<constant>"`` placeholder.
    - Existing ``"<constant>"`` / ``"<c>"`` tokens are preserved as ``"<constant>"``.
    - Operators and other tokens are left unchanged.
    - ``None`` maps to ``None``.
    """
    if tokens is None:
        return None
    normalized: list[str] = []
    for token in tokens:
        token_str = str(token)
        normalized_token, is_var = normalize_variable_token(token_str)
        if is_var:
            normalized.append(normalized_token)
            continue
        if token_str in {"<constant>", "<c>"}:
            normalized.append("<constant>")
            continue
        # numeric literal -> constant placeholder
        try:
            float(token_str)
        except ValueError:
            normalized.append(token_str)
        else:
            normalized.append("<constant>")
    return normalized


def normalize_expression(tokens: Sequence[str | Any] | None) -> list[str] | None:
    """Normalize an expression/prefix while keeping numeric literals intact.

    Converts variable tokens to canonical ``x{n}`` names but leaves numeric literals
    untouched (so the expression form can still carry concrete float strings).
    ``None`` maps to ``None``.
    """
    if tokens is None:
        return None
    normalized: list[str] = []
    for token in tokens:
        token_str = str(token)
        normalized_token, is_var = normalize_variable_token(token_str)
        normalized.append(normalized_token if is_var else token_str)
    return normalized
