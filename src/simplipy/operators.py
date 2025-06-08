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
            return np.cbrt(x)
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
