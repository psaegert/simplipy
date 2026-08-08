"""Numerically safe realizations of the mathematical operators.

Provides the element-wise Python/NumPy functions that back the engine's operator
tokens: the unary maps (``neg``, ``inv``, ``abs``, the trig/hyperbolic families,
``exp``, ``log``) and the binary operators ``div`` (``x / y``), ``pow`` (``x ** y``,
IEEE/numpy conventions incl. the negative-base non-integer-exponent NaN mask) and
``rootn`` (the deployed n-th-root convention: finite nonzero integer index, odd
signed / even principal). The retired hyper-operator families (``mult2``..``div5``,
``pow2``..``pow5``, ``pow1_2``..``pow1_5``) were deleted with the 0.12 clean
vocabulary -- their spellings are handled at the conversion boundary, never realized.
The functions handle singular inputs (such as division by zero) by returning signed
infinities or NaN rather than raising, and operate on both scalars and array-likes.
"""
import math
from typing import Iterable
from types import ModuleType
import numpy as np


_torch_module: ModuleType | None = None
_torch_checked = False


def neg(x: float) -> float:
    """Return the element-wise negation of x."""
    return -x


def inv(x: float) -> float:
    """Return the element-wise multiplicative inverse of x."""
    # numpy will handle the x = 0 case
    if isinstance(x, Iterable):
        return 1 / x

    # Manually handle scalar case (python floats raise on 1/0). Sign-aware like the numpy
    # array branch above: 1/+0.0 -> +inf, 1/-0.0 -> -inf (IEEE).
    if x == 0:
        return math.copysign(float('inf'), x)

    # All safe
    return 1 / x


def div(x: float, y: float) -> float:
    """Return the element-wise division of x by y."""
    # numpy will handle the x = 0 case
    if isinstance(y, Iterable):
        return x / y

    # Manually handle scalar case (python floats raise on x/0). IEEE like the numpy array
    # branch: the SIGN OF THE ZERO DIVISOR participates (1/-0.0 -> -inf); 0/0 -> nan.
    if y == 0:
        # x iterable / scalar zero: let numpy do IEEE division (keeps divisor sign semantics)
        if isinstance(x, Iterable):
            return x / np.float64(y)

        if not isinstance(x, complex):
            if x == 0 or math.isnan(x):
                return float('nan')
            return math.copysign(float('inf'), x) * math.copysign(1.0, y)

        # complex numerator: undefined here
        return float('nan')

    # All safe
    return x / y


def abs(x: float) -> float:
    """Return the element-wise absolute value of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.abs(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.abs(x)
    if isinstance(x, complex):
        # Handle complex numbers
        return (x.real ** 2 + x.imag ** 2) ** 0.5
    # Handle scalar case
    return x if x >= 0 else -x  # Ensure non-negative result


def sin(x: float) -> float:
    """Return the element-wise sine of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.sin(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.sin(x)
    # Handle scalar case
    return np.sin(x)  # Use numpy for scalar sine calculation


def cos(x: float) -> float:
    """Return the element-wise cosine of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.cos(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.cos(x)
    # Handle scalar case
    return np.cos(x)  # Use numpy for scalar cosine calculation


def tan(x: float) -> float:
    """Return the element-wise tangent of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.tan(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.tan(x)
    # Handle scalar case
    return np.tan(x)  # Use numpy for scalar tangent calculation


def asin(x: float) -> float:
    """Return the element-wise inverse sine of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.arcsin(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.asin(x)
    # Handle scalar case
    return np.arcsin(x)  # Use numpy for scalar arcsine calculation


def acos(x: float) -> float:
    """Return the element-wise inverse cosine of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.arccos(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.acos(x)
    # Handle scalar case
    return np.arccos(x)  # Use numpy for scalar arccosine calculation


def atan(x: float) -> float:
    """Return the element-wise inverse tangent of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.arctan(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.atan(x)
    # Handle scalar case
    return np.arctan(x)  # Use numpy for scalar arctangent calculation


def sinh(x: float) -> float:
    """Return the element-wise hyperbolic sine of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.sinh(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.sinh(x)
    # Handle scalar case
    return np.sinh(x)  # Use numpy for scalar hyperbolic sine calculation


def cosh(x: float) -> float:
    """Return the element-wise hyperbolic cosine of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.cosh(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.cosh(x)
    # Handle scalar case
    return np.cosh(x)  # Use numpy for scalar hyperbolic cosine calculation


def tanh(x: float) -> float:
    """Return the element-wise hyperbolic tangent of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.tanh(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.tanh(x)
    # Handle scalar case
    return np.tanh(x)  # Use numpy for scalar hyperbolic tangent calculation


def asinh(x: float) -> float:
    """Return the element-wise inverse hyperbolic sine of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.arcsinh(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.asinh(x)
    # Handle scalar case
    return np.arcsinh(x)  # Use numpy for scalar inverse hyperbolic sine calculation


def acosh(x: float) -> float:
    """Return the element-wise inverse hyperbolic cosine of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.arccosh(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.acosh(x)
    # Handle scalar case
    return np.arccosh(x)  # Use numpy for scalar inverse hyperbolic cosine calculation


def atanh(x: float) -> float:
    """Return the element-wise inverse hyperbolic tangent of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.arctanh(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.atanh(x)
    # Handle scalar case
    return np.arctanh(x)  # Use numpy for scalar inverse hyperbolic tangent calculation


def exp(x: float) -> float:
    """Return the element-wise exponential of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.exp(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.exp(x)
    # Handle scalar case
    return np.exp(x)  # Use numpy for scalar exponential calculation


def log(x: float) -> float:
    """Return the element-wise natural logarithm of x."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        return np.log(x)
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.log(x)
    # Handle scalar case
    return np.log(x)  # Use numpy for scalar logarithm calculation


def _neginf_nonint_pow_mask(x: 'np.ndarray | float', y: 'np.ndarray | float') -> 'np.ndarray':
    """Cells where pow is contract-undefined: -inf base with a finite non-integer exponent.
    (Finite negative bases already yield NaN from np.power; infinite exponents keep the
    magnitude-step semantics.)"""
    yf = np.asarray(y, dtype=float)
    return np.isneginf(np.asarray(x, dtype=float)) & np.isfinite(yf) & (np.mod(yf, 1.0) != 0.0)


def rootn(x: float, n: float) -> float:
    """Return the real n-th root of x: IEEE-754 rootn, honest for EVERY integer index.

    * n odd:  the signed root, total on R (``rootn(-8, 3) = -2``);
    * n even: the principal root, NaN on negatives (identical to ``pow(x, 1/n)``);
    * n == 1: the identity; n negative: ``1 / rootn(x, -n)``; n == 0 or a non-integer
      index: NaN (invalid operation).

    One semantics, five surfaces: this realization, ``numeric.rs``, ``interval.rs`` and
    both ``hiprec.rs`` evaluators implement the same table, pinned against each other by
    the cross-evaluator parity tests.
    """
    n_arr = np.asarray(n, dtype=float)
    is_int = (n_arr == np.floor(n_arr)) & np.isfinite(n_arr)
    with np.errstate(invalid='ignore', divide='ignore'):
        xa = np.asarray(x, dtype=float)
        k = np.where(is_int & (n_arr != 0), np.abs(n_arr), np.nan)  # |index|, NaN if invalid
        r = 1.0 / k
        odd = np.mod(k, 2) == 1
        mag = np.abs(xa) ** r
        signed = np.where(xa < 0, -mag, mag)                    # odd: sign-preserving
        principal = np.where(xa < 0, np.nan, mag)               # even: NaN on negatives
        root = np.where(odd, signed, principal)
        root = np.where(k == 1, xa, root)                       # unit index: identity, exactly
        out = np.where(n_arr < 0, 1.0 / root, root)             # negative index: reciprocal
        out = np.where(np.isnan(k), np.nan, out)
    if isinstance(x, np.ndarray) or isinstance(n, np.ndarray):
        return out
    return float(out)


def pow(x: float, y: float) -> float:
    """Return x raised to the power of y, element-wise (NaN at -inf base with finite
    non-integer exponent: contract-aligned real semantics)."""
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        # Handle numpy arrays
        with np.errstate(invalid='ignore'):
            return np.where(_neginf_nonint_pow_mask(x, y), np.nan, np.power(x, y))
    if type(x).__module__ == 'torch' and type(x).__name__ == 'Tensor':
        if not _torch_checked:
            try:
                import torch  # type:ignore
                _torch_module = torch
            except ImportError:
                _torch_module = None
            _torch_checked = True

        if _torch_module is None:
            raise ImportError("PyTorch is required to process torch tensors")

        # Handle torch tensors
        return _torch_module.pow(x, y)
    # Handle scalar case
    with np.errstate(invalid='ignore'):
        if isinstance(x, int):
            x = float(x)
        if isinstance(y, int):
            y = float(y)
        if bool(_neginf_nonint_pow_mask(x, y)):
            return float('nan')
        return np.power(x, y)  # Use numpy for scalar power calculation
