"""Numerically safe realizations of the mathematical operators.

Provides the element-wise functions that back the engine's operator tokens: the unary maps
(``neg``, ``inv``, ``abs``, the trig/hyperbolic families, ``exp``, ``log``, ``floor``,
``ceil``, ``erf``, ``harmonic``), the arithmetic (``add``, ``sub``, ``mul``, ``div``) and the
binary functions ``pow`` (``x ** y``, NaN for a negative base under a non-integer exponent),
``rootn`` (the real n-th root: finite nonzero integer index, odd signed / even principal),
``max``, ``min``, ``mod`` and ``atan2``. The retired hyper-operator families (``mult2``..
``div5``, ``pow2``..``pow5``, ``pow1_2``..``pow1_5``) were deleted with the 0.12 clean
vocabulary -- their spellings are handled at the conversion boundary, never realized.

EVERY REALIZATION IS THE ENGINE'S OWN KERNEL. On real operands each function evaluates the
float64 kernel the engine's evaluator uses for its token (``simplipy._core``, through
``kernel_unary``/``kernel_binary``), so the deployed value, the constant fold, the miner's
evaluator and the rule judge compute the same bits on every host. The transcendentals come
from the platform's libm (as Python's ``math`` module does), the exact operations are
IEEE-754's, and ``harmonic`` is the core's own kernel. Singular inputs give signed
infinities or NaN rather than raising.

Scalars give numpy float64 scalars and array-likes arrays, broadcast like numpy. A torch
tensor takes torch's own functions (the torch branch is torch's arithmetic, not the core's);
a complex operand of an operator that has a complex continuation takes numpy's.
"""
import builtins
import math
from typing import Any
from types import ModuleType
import numpy as np

# The declared surface: the operator realizations that engine configs name
# (`simplipy.operators.sin`, ...). These spellings are normative -- the trust
# model resolves them and shipped config.yaml files reference them.
__all__ = [
    'neg', 'inv', 'div', 'abs',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'exp', 'log', 'rootn', 'pow',
    'add', 'sub', 'mul',
    'floor', 'ceil', 'max', 'min', 'mod', 'harmonic', 'erf', 'atan2', 'besselj',
]

_torch_module: ModuleType | None = None
_torch_checked = False


def _is_tensor(x: Any) -> bool:
    t = type(x)
    return t.__name__ == 'Tensor' and t.__module__ == 'torch'


def _torch_of(*xs: Any) -> ModuleType | None:
    """The torch module if any operand is a torch tensor (torch imported on first use), else
    None."""
    global _torch_module, _torch_checked
    if not any(_is_tensor(x) for x in xs):
        return None
    if not _torch_checked:
        try:
            import torch  # type:ignore
            _torch_module = torch
        except ImportError:
            _torch_module = None
        _torch_checked = True
    if _torch_module is None:
        raise ImportError("PyTorch is required to process torch tensors")
    return _torch_module


def _torch_pair(torch: ModuleType, a: Any, b: Any) -> tuple[Any, Any]:
    """Both operands as tensors, a non-tensor one taking the other's dtype and device."""
    ref = a if type(a).__module__ == 'torch' else b
    return (torch.as_tensor(a, dtype=ref.dtype, device=ref.device),
            torch.as_tensor(b, dtype=ref.dtype, device=ref.device))


_F64 = np.dtype(np.float64)
#: Scalars that are already float64 values (`np.float64` is a `float`).
_FLOATS = (float, np.float64)
_core_module: ModuleType | None = None


def _load_core() -> ModuleType:
    """The compiled core, imported on first use (`import simplipy` never needs it)."""
    global _core_module
    if _core_module is None:
        from simplipy import _core as core
        _core_module = core
    return _core_module


def _f64(x: Any) -> np.ndarray:
    """`x` as a float64 array. A complex operand is refused: the kernels are real."""
    if type(x) is np.ndarray and x.dtype == _F64:
        return x
    if np.iscomplexobj(x):
        raise TypeError('this operator is defined on the real numbers; got a complex operand')
    return np.asarray(x, dtype=np.float64)


def _contiguous(a: np.ndarray, shape: tuple) -> np.ndarray:
    """`a` laid out for the kernel: a one-element operand flat (the kernel broadcasts it
    without a copy; a 0-d array has no shape to hand a buffer), any other at `shape`,
    C-contiguous."""
    if a.size == 1:
        return a.reshape(1)
    if a.shape != shape:
        a = np.broadcast_to(a, shape)
    return a if a.flags.c_contiguous else np.ascontiguousarray(a)


def _kernel1(name: str, x: Any) -> Any:
    """The engine's float64 kernel `name` (``simplipy._core``) on `x`, element-wise, with the
    GIL released for large arrays. A scalar takes the kernel's scalar entry and gives a numpy
    float64."""
    core = _core_module or _load_core()
    if isinstance(x, _FLOATS):
        return np.float64(core.kernel_unary_scalar(name, x))
    a = _f64(x)
    if a.ndim == 0:
        return np.float64(core.kernel_unary_scalar(name, float(a)))
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    out = np.empty(a.shape, dtype=np.float64)
    core.kernel_unary(name, a, out)
    return out


def _kernel2(name: str, x: Any, y: Any) -> Any:
    """The binary kernel `name` over the numpy broadcast of `x` and `y`, as `_kernel1`."""
    core = _core_module or _load_core()
    if isinstance(x, _FLOATS) and isinstance(y, _FLOATS):
        return np.float64(core.kernel_binary_scalar(name, x, y))
    a, b = _f64(x), _f64(y)
    if a.ndim == 0 and b.ndim == 0:
        return np.float64(core.kernel_binary_scalar(name, float(a), float(b)))
    shape = a.shape if a.shape == b.shape else np.broadcast_shapes(a.shape, b.shape)
    out = np.empty(shape, dtype=np.float64)
    core.kernel_binary(name, _contiguous(a, shape), _contiguous(b, shape), out)
    return out[()] if out.ndim == 0 else out


def _plain(x: Any) -> bool:
    """A float scalar or a float64 array: neither a tensor nor complex, no test needed."""
    return isinstance(x, _FLOATS) or (type(x) is np.ndarray and x.dtype == _F64)


def _unary(token: str, x: Any, torch_name: str, complex_fn: Any = None) -> Any:
    """A unary realization: torch's `torch_name` on a tensor, `complex_fn` (numpy's) on a
    complex operand where the function has a complex continuation, else the core's kernel."""
    if not _plain(x):
        torch = _torch_of(x)
        if torch is not None:
            return getattr(torch, torch_name)(x)
        if complex_fn is not None and np.iscomplexobj(x):
            return complex_fn(x)
    return _kernel1(token, x)


def _binary(token: str, a: Any, b: Any, torch_fn: Any, complex_fn: Any = None) -> Any:
    """A binary realization: `torch_fn(torch, a, b)` on tensors, `complex_fn` on a complex
    operand where the function has a complex continuation, else the core's kernel."""
    if not (_plain(a) and _plain(b)):
        torch = _torch_of(a, b)
        if torch is not None:
            return torch_fn(torch, a, b)
        if complex_fn is not None and (np.iscomplexobj(a) or np.iscomplexobj(b)):
            return complex_fn(a, b)
    return _kernel2(token, a, b)


# ------------------------------------------------------------------ arithmetic

def neg(x: float) -> float:
    """Return the element-wise negation of x."""
    if not _plain(x) and (_torch_of(x) is not None or np.iscomplexobj(x)):
        return -x
    return _kernel1('neg', x)


def inv(x: float) -> float:
    """Return the element-wise multiplicative inverse of x (IEEE: ``1/+0 = +inf``,
    ``1/-0 = -inf``)."""
    if not _plain(x) and (_torch_of(x) is not None or np.iscomplexobj(x)):
        return 1 / x
    return _kernel1('inv', x)


def add(x: float, y: float) -> float:
    """Return the element-wise sum x + y."""
    return _binary('+', x, y, lambda t, a, b: a + b, lambda a, b: np.add(a, b))


def sub(x: float, y: float) -> float:
    """Return the element-wise difference x - y."""
    return _binary('-', x, y, lambda t, a, b: a - b, lambda a, b: np.subtract(a, b))


def mul(x: float, y: float) -> float:
    """Return the element-wise product x * y."""
    return _binary('*', x, y, lambda t, a, b: a * b, lambda a, b: np.multiply(a, b))


def div(x: float, y: float) -> float:
    """Return the element-wise division of x by y (IEEE: the sign of a zero divisor
    participates, ``1/-0 = -inf``; ``0/0 = nan``)."""
    if not (_plain(x) and _plain(y)):
        if _torch_of(x, y) is not None:
            return x / y
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.divide(x, y)
    return _kernel2('/', x, y)


def abs(x: float) -> float:
    """Return the element-wise absolute value of x (``+0`` at either zero). A Python scalar
    keeps its own type, an int staying an int; a complex operand gives its magnitude."""
    torch = _torch_of(x)
    if torch is not None:
        return torch.abs(x)
    if isinstance(x, np.ndarray):
        return np.abs(x) if np.iscomplexobj(x) else _kernel1('abs', x)
    if isinstance(x, complex):
        return (x.real ** 2 + x.imag ** 2) ** 0.5
    # an exact operation: the scalar's own abs gives the kernel's bits in its own type
    return builtins.abs(x)


# ------------------------------------------------------------------ transcendentals

def sin(x: float) -> float:
    """Return the element-wise sine of x."""
    return _unary('sin', x, 'sin', np.sin)


def cos(x: float) -> float:
    """Return the element-wise cosine of x."""
    return _unary('cos', x, 'cos', np.cos)


def tan(x: float) -> float:
    """Return the element-wise tangent of x."""
    return _unary('tan', x, 'tan', np.tan)


def asin(x: float) -> float:
    """Return the element-wise inverse sine of x."""
    return _unary('asin', x, 'asin', np.arcsin)


def acos(x: float) -> float:
    """Return the element-wise inverse cosine of x."""
    return _unary('acos', x, 'acos', np.arccos)


def atan(x: float) -> float:
    """Return the element-wise inverse tangent of x."""
    return _unary('atan', x, 'atan', np.arctan)


def sinh(x: float) -> float:
    """Return the element-wise hyperbolic sine of x."""
    return _unary('sinh', x, 'sinh', np.sinh)


def cosh(x: float) -> float:
    """Return the element-wise hyperbolic cosine of x."""
    return _unary('cosh', x, 'cosh', np.cosh)


def tanh(x: float) -> float:
    """Return the element-wise hyperbolic tangent of x."""
    return _unary('tanh', x, 'tanh', np.tanh)


def asinh(x: float) -> float:
    """Return the element-wise inverse hyperbolic sine of x."""
    return _unary('asinh', x, 'asinh', np.arcsinh)


def acosh(x: float) -> float:
    """Return the element-wise inverse hyperbolic cosine of x."""
    return _unary('acosh', x, 'acosh', np.arccosh)


def atanh(x: float) -> float:
    """Return the element-wise inverse hyperbolic tangent of x."""
    return _unary('atanh', x, 'atanh', np.arctanh)


def exp(x: float) -> float:
    """Return the element-wise exponential of x."""
    return _unary('exp', x, 'exp', np.exp)


def log(x: float) -> float:
    """Return the element-wise natural logarithm of x."""
    return _unary('log', x, 'log', np.log)


# ------------------------------------------------------------------ powers and roots

def _rootn_table(xp: ModuleType, x, n):  # type: ignore[no-untyped-def]
    """`rootn`'s table over one array namespace (here torch's), for arrays `x`, `n`."""
    k = xp.where((n == xp.floor(n)) & xp.isfinite(n) & (n != 0), xp.abs(n), math.nan)  # |index|, NaN if invalid
    mag = xp.abs(x) ** (1.0 / k)
    odd = xp.remainder(k, 2) == 1
    root = xp.where(odd, xp.copysign(mag, x), xp.where(x < 0, math.nan, mag))  # odd: signed, even: principal
    root = xp.where(k == 1, x, root)                                           # unit index: identity, exactly
    out = xp.where(n < 0, 1.0 / root, root)                                    # negative index: reciprocal
    return xp.where(xp.isnan(k), math.nan, out)


def rootn(x: float, n: float) -> float:
    """Return the real n-th root of x: IEEE-754 rootn, honest for EVERY integer index.

    * n odd:  the signed root, total on R (``rootn(-8, 3) = -2``) and odd at zero as
      everywhere else (``rootn(-0, 3) = -0``, so ``rootn(-0, -3) = -inf``);
    * n even: the principal root, NaN on negatives (identical to ``pow(x, 1/n)``),
      ``+0`` at either zero;
    * n == 1: the identity; n negative: ``1 / rootn(x, -n)``; n == 0 or a non-integer
      index: NaN (invalid operation).

    One semantics, five surfaces: this realization (the core's kernel), ``numeric.rs``,
    ``interval.rs`` and both ``hiprec.rs`` evaluators implement the same table, pinned
    against each other by the cross-evaluator parity tests. Scalars give a Python float.
    """
    torch = None if _plain(x) else _torch_of(x)
    if torch is not None:
        xt = x if x.is_floating_point() else x.to(torch.get_default_dtype())  # type: ignore[attr-defined]
        nt = torch.as_tensor(n, dtype=xt.dtype, device=xt.device)  # type: ignore[union-attr]
        return _rootn_table(torch, xt, nt)
    out = _kernel2('rootn', x, n)
    if isinstance(x, np.ndarray) or isinstance(n, np.ndarray):
        return out
    return float(out)


def pow(x: float, y: float) -> float:
    """Return x raised to the power of y, element-wise: C's ``pow``, except that a negative
    base, ``-inf`` included, under a finite non-integer exponent is NaN (the real-analysis
    reading; C99 gives ``pow(-inf, 0.5) = inf``). A zero base under a finite non-integer
    exponent gives +0 (or +inf for a negative exponent) whatever the zero's sign, and integer
    exponents keep IEEE's signed results (``pow(-0, 3) = -0``)."""
    if not (_plain(x) and _plain(y)):
        torch = _torch_of(x)
        if torch is not None:
            yt = torch.as_tensor(y, dtype=x.dtype, device=x.device)  # type: ignore[attr-defined]
            nonint = torch.isfinite(yt) & (yt != torch.floor(yt))
            return torch.pow(torch.where((x == 0) & nonint, torch.zeros_like(x), x), y)
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            return np.power(x, y)
    return _kernel2('pow', x, y)


# ------------------------------------------------------------------ the 0.15 operators


def floor(x: Any) -> Any:
    """The largest integer not above x, element-wise (exact; ``floor(-0) = -0``, infinities
    and NaN unchanged)."""
    return _unary('floor', x, 'floor')


def ceil(x: Any) -> Any:
    """The smallest integer not below x, element-wise (exact; ``ceil(-0.5) = -0``)."""
    return _unary('ceil', x, 'ceil')


def max(a: Any, b: Any) -> Any:
    """The larger of a and b, element-wise: IEEE 754-2019 ``maximum``. NaN if either operand
    is NaN, and -0 < +0, so ``max(-0, +0) = max(+0, -0) = +0``: commutative bit for bit
    (``np.maximum`` returns its second operand on a zero tie)."""
    def torch_max(torch: ModuleType, a: Any, b: Any) -> Any:
        at, bt = _torch_pair(torch, a, b)
        return torch.where((at == bt) & (at == 0), at + bt, torch.maximum(at, bt))
    return _binary('max', a, b, torch_max)


def min(a: Any, b: Any) -> Any:
    """The smaller of a and b, element-wise: IEEE 754-2019 ``minimum``. NaN if either operand
    is NaN, and -0 < +0, so ``min(-0, +0) = min(+0, -0) = -0``: commutative bit for bit."""
    def torch_min(torch: ModuleType, a: Any, b: Any) -> Any:
        at, bt = _torch_pair(torch, a, b)
        return torch.where((at == bt) & (at == 0), -((-at) + (-bt)), torch.minimum(at, bt))
    return _binary('min', a, b, torch_min)


def mod(a: Any, b: Any) -> Any:
    """The remainder of a by b with the sign of the divisor, element-wise: ``np.mod`` on
    float64 (Python's ``%``). ``mod(-7, 3) = 2``, ``mod(7, -3) = -2``; NaN for a zero divisor
    (integer operands too: they are taken as float64) and for an infinite dividend; a zero
    result takes the divisor's sign; ``mod(3, inf) = 3``, ``mod(-3, inf) = inf``,
    ``mod(3, -inf) = -inf``, ``mod(-3, -inf) = -3``; a tiny negative dividend can give b
    itself (``mod(-1e-20, 3) = 3``, the correctly rounded 3 - 1e-20)."""
    def torch_mod(torch: ModuleType, a: Any, b: Any) -> Any:
        at, bt = _torch_pair(torch, a, b)
        m = torch.fmod(at, bt)  # numpy's npy_remainder, step for step
        m = torch.where((m != 0) & ((bt < 0) != (m < 0)), m + bt, m)
        return torch.where(m == 0, torch.copysign(torch.zeros_like(m), bt), m)
    return _binary('mod', a, b, torch_mod)


def harmonic(x: Any) -> Any:
    """The harmonic number H(x) = digamma(x + 1) + euler_gamma, element-wise:
    ``H(n) = 1 + 1/2 + ... + 1/n`` and ``H(x) - H(x - 1) = 1/x``.

    The core's own kernel (the literal ``digamma(x + 1) + euler_gamma`` loses every digit near
    0, where it is exactly 0 on ``|x| < 1e-17``): within 0.53 ulp of the true value on
    x > -1, and below -1 within one ulp plus 2^-58 of the larger reflection term (H's zero
    between two poles is where those cancel). ``H(+-0) = +-0``, ``H(+inf) = +inf``,
    ``H(-inf) = nan``, and NaN at every negative integer, where H has a pole whose one-sided
    limits differ. The torch branch is torch's digamma, whose accuracy near 0 is torch's."""
    torch = _torch_of(x)
    if torch is not None:
        xt = x if x.is_floating_point() else x.to(torch.get_default_dtype())
        pole = (xt < 0) & (xt == torch.floor(xt))
        h = torch.where(pole, torch.full_like(xt, math.nan), torch.special.digamma(xt + 1) + np.euler_gamma)
        return torch.where(xt == 0, xt, h)
    return _kernel1('harmonic', x)


def erf(x: Any) -> Any:
    """The error function, element-wise: ``erf(+-0) = +-0``, ``erf(+-inf) = +-1``, and exactly
    +-1 in float64 beyond about |x| = 5.92 (from the platform's libm)."""
    return _unary('erf', x, 'erf')


def atan2(y: Any, x: Any) -> Any:
    """The angle of the point (x, y) in (-pi, pi], element-wise, argument order as numpy's
    ``arctan2(y, x)`` (from the platform's libm, with C99's cells: ``atan2(+-0, +0) = +-0``,
    ``atan2(+-0, -0) = +-pi``, ``atan2(+-inf, +-inf) = +-pi/4`` or ``+-3pi/4``)."""
    def torch_atan2(torch: ModuleType, a: Any, b: Any) -> Any:
        at, bt = _torch_pair(torch, a, b)
        return torch.atan2(at, bt)
    return _binary('atan2', y, x, torch_atan2)


def besselj(nu: Any, x: Any) -> Any:
    """The Bessel function of the first kind J_nu(x) of real order nu, element-wise.

    Bit for bit ``scipy.special.jv`` (scipy 1.18.1, whose code the core carries: AMOS, with
    the reflection through Y for a negative order and Cephes where AMOS fails), and so
    scipy's accuracy: measured against 50-digit mpmath on nu in [0, 50], x in [0, 100],
    within 2,634 ulp at least 0.1 away from a zero of J_nu and within 1.4e-14 absolute near
    one. NaN for a non-integer order of a negative argument (the value is complex there) and
    for an infinite argument; ``J_0(0) = 1``, ``J_nu(0) = 0`` for nu > 0 and for a negative
    integer nu, ``J_nu(0) = inf`` for a negative non-integer nu. Torch has no Bessel function
    of real order, so a tensor is evaluated by the same kernel on the host and returned as a
    tensor of its dtype and device (not differentiable)."""
    def torch_besselj(torch: ModuleType, a: Any, b: Any) -> Any:
        at, bt = _torch_pair(torch, a, b)
        host = lambda t: t.detach().to('cpu', torch.float64).numpy()  # noqa: E731
        return torch.as_tensor(besselj(host(at), host(bt)), dtype=at.dtype, device=at.device)
    return _binary('besselj', nu, x, torch_besselj)
