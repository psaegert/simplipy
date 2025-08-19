from typing import Iterable
from types import ModuleType
import numpy as np


_torch_module: ModuleType | None = None
_torch_checked = False


def neg(x: float) -> float:
    return -x


def inv(x: float) -> float:
    # numpy will handle the x = 0 case
    if isinstance(x, Iterable):
        return 1 / x

    # Manually handle scalar case
    if x == 0:
        return float('inf')

    # All safe
    return 1 / x


def div(x: float, y: float) -> float:
    # numpy will handle the x = 0 case
    if isinstance(y, Iterable):
        return x / y

    # Manually handle scalar case
    if y == 0:
        # When x is an iterable, multiply with infinity to let the sign determine the result
        if isinstance(x, Iterable):
            return x * float('inf')

        # When x is a scalar, return inf or -inf depending on the sign of x
        if not isinstance(x, complex):
            if x > 0:
                return float('inf')
            elif x < 0:
                return float('-inf')

        # Both x and y are zero.
        # Return NaN to indicate an undefined result
        return float('nan')

    # All safe
    return x / y


def mult2(x: float) -> float:
    return 2 * x


def mult3(x: float) -> float:
    return 3 * x


def mult4(x: float) -> float:
    return 4 * x


def mult5(x: float) -> float:
    return 5 * x


def div2(x: float) -> float:
    return x / 2


def div3(x: float) -> float:
    return x / 3


def div4(x: float) -> float:
    return x / 4


def div5(x: float) -> float:
    return x / 5


def pow2(x: float) -> float:
    return x ** 2


def pow3(x: float) -> float:
    return x ** 3


def pow4(x: float) -> float:
    return x ** 4


def pow5(x: float) -> float:
    return x ** 5


def pow1_2(x: float) -> float:
    return x ** 0.5


def pow1_3(x: float) -> float:
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        if np.iscomplexobj(x):
            # Handle complex numbers
            return x ** (1 / 3)
        x = np.asarray(x)
        x = np.where(x < 0, -(-x) ** (1 / 3), x ** (1 / 3))
        return x

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
        if x.dtype == torch.complex64 or x.dtype == torch.complex128:  # type:ignore
            # Handle complex numbers
            return x ** (1 / 3)
        x = torch.where(x < 0, -(-x) ** (1 / 3), x ** (1 / 3))
        return x

    if not isinstance(x, complex) and x < 0:
        # Discard imaginary component
        return - (-x) ** (1 / 3)
    else:
        return x ** (1 / 3)


def pow1_4(x: float) -> float:
    return x ** 0.25


def pow1_5(x: float) -> float:
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        if np.iscomplexobj(x):
            # Handle complex numbers
            return x ** (1 / 5)
        x = np.asarray(x)
        x = np.where(x < 0, -(-x) ** (1 / 5), x ** (1 / 5))
        return x

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
        if x.dtype == torch.complex64 or x.dtype == torch.complex128:  # type:ignore
            # Handle complex numbers
            return x ** (1 / 5)
        x = torch.where(x < 0, -(-x) ** (1 / 5), x ** (1 / 5))
        return x

    if not isinstance(x, complex) and x < 0:
        # Discard imaginary component
        return - (-x) ** (1 / 5)
    else:
        return x ** (1 / 5)


def abs(x: float) -> float:
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


def pow(x: float, y: float) -> float:
    global _torch_module, _torch_checked
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        # Handle numpy arrays
        with np.errstate(invalid='ignore'):
            return np.power(x, y)
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
        return np.power(x, y)  # Use numpy for scalar power calculation
