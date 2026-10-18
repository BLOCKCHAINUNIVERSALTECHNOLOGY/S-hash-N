"""N-dimensional tensors implemented in pure Python (no dependencies).

A tensor stores its elements as a flat row-major list of floats plus a
shape tuple. Arithmetic broadcasts like NumPy. This is meant for learning
and for small/medium data; it is not a replacement for a GPU library.
"""
import math
import operator
import os
import random

try:  # optional accelerator; S#N works identically without it
    if os.environ.get("SN_BACKEND", "").lower() == "python":
        raise ImportError
    import numpy as _np
except ImportError:  # pragma: no cover - depends on the environment
    _np = None

from .errors import SNRuntimeError
from .values import NativeObject, fmt_number, type_name, call_value
from . import stats

RNG = random.Random()
NUMPY_MIN_WORK = 2000   # use NumPy for a matmul when m*k*n reaches this


def backend():
    """Name of the active numeric backend: "numpy" or "python"."""
    return "numpy" if _np is not None else "python"


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


# ------------------------------------------------------------ scalar helpers
def _div(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        if x == 0 or x != x:
            return math.nan
        return math.inf if (x > 0) == (math.copysign(1.0, y) > 0) else -math.inf


def _floordiv(x, y):
    try:
        return float(x // y)
    except ZeroDivisionError:
        return _div(x, y)


def _mod(x, y):
    try:
        return x % y
    except ZeroDivisionError:
        return math.nan


def _pow(x, y):
    try:
        r = x ** y
    except OverflowError:
        return math.inf
    except ZeroDivisionError:
        return math.inf
    return math.nan if isinstance(r, complex) else r


def exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def log(x):
    if x > 0:
        return math.log(x)
    return -math.inf if x == 0 else math.nan


def sqrt(x):
    return math.sqrt(x) if x >= 0 else math.nan


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def relu(x):
    return x if x > 0 else 0.0


_OPS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "/": _div, "//": _floordiv, "%": _mod, "**": _pow,
    "==": lambda x, y: 1.0 if x == y else 0.0,
    "!=": lambda x, y: 1.0 if x != y else 0.0,
    "<": lambda x, y: 1.0 if x < y else 0.0,
    ">": lambda x, y: 1.0 if x > y else 0.0,
    "<=": lambda x, y: 1.0 if x <= y else 0.0,
    ">=": lambda x, y: 1.0 if x >= y else 0.0,
}
_BOOL_OPS = {"==", "!=", "<", ">", "<=", ">="}


def broadcast_shape(s1, s2):
    n = max(len(s1), len(s2))
    a = (1,) * (n - len(s1)) + tuple(s1)
    b = (1,) * (n - len(s2)) + tuple(s2)
    out = []
    for x, y in zip(a, b):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise SNRuntimeError(
                f"shapes {list(s1)} and {list(s2)} cannot be broadcast together"
            )
    return tuple(out)


def _bidx(shape, out_shape):
    """Flat indices into an array of `shape` for every cell of `out_shape`."""
    nd = len(out_shape)
    shape = (1,) * (nd - len(shape)) + tuple(shape)
    strides = []
    s = 1
    for d in reversed(shape):
        strides.append(s if d != 1 else 0)
        s *= d
    strides.reverse()
    idx = [0]
    for dim, st in zip(out_shape, strides):
        idx = [i + k * st for i in idx for k in range(dim)]
    return idx


def _int_index(k, dim):
    if isinstance(k, bool) or not isinstance(k, (int, float)):
        raise SNRuntimeError(f"tensor index must be an integer, got {type_name(k)}")
    if isinstance(k, float):
        if not k.is_integer():
            raise SNRuntimeError(f"tensor index must be an integer, got {k}")
        k = int(k)
    if k < -dim or k >= dim:
        raise SNRuntimeError(f"index {k} is out of range for an axis of size {dim}")
    return k + dim if k < 0 else k


def _plain(v):
    if isinstance(v, Tensor):
        return v.to_nested()
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


class Tensor(NativeObject):
    __slots__ = ("data", "shape", "is_bool")
    sn_type = "tensor"

    def __init__(self, data, shape=None, is_bool=False):
        self.data = data
        self.shape = tuple(shape) if shape is not None else (len(data),)
        self.is_bool = is_bool

    # ----------------------------------------------------------- construction
    @staticmethod
    def from_value(v):
        if isinstance(v, Tensor):
            return v
        if isinstance(v, (int, float)):
            return Tensor([float(v)], ())
        if isinstance(v, list):
            v = _plain(v)
            shape = []
            x = v
            while isinstance(x, list):
                shape.append(len(x))
                if not x:
                    break
                x = x[0]
            flat = []

            def walk(node, depth):
                if depth == len(shape):
                    if isinstance(node, (int, float)):
                        flat.append(float(node))
                    else:
                        raise SNRuntimeError(
                            f"tensor elements must be numbers, got {type_name(node)}"
                        )
                else:
                    if not isinstance(node, list) or len(node) != shape[depth]:
                        raise SNRuntimeError("all rows of a tensor must have the same length")
                    for c in node:
                        walk(c, depth + 1)

            walk(v, 0)
            return Tensor(flat, shape)
        raise SNRuntimeError(f"cannot convert {type_name(v)} to a tensor")

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return len(self.data)

    def copy(self):
        return Tensor(list(self.data), self.shape, self.is_bool)

    def _strides(self):
        strides = []
        s = 1
        for d in reversed(self.shape):
            strides.append(s)
            s *= d
        strides.reverse()
        return strides

    def to_nested(self):
        if not self.shape:
            return self.data[0]
        shape = self.shape
        data = self.data

        def build(off, dim):
            n = shape[dim]
            if dim == len(shape) - 1:
                return data[off:off + n]
            stride = prod(shape[dim + 1:])
            return [build(off + i * stride, dim + 1) for i in range(n)]

        return build(0, 0)

    def _axis(self, axis):
        if isinstance(axis, float) and axis.is_integer():
            axis = int(axis)
        if not isinstance(axis, int) or isinstance(axis, bool):
            raise SNRuntimeError("axis must be an integer")
        nd = self.ndim
        if axis < -nd or axis >= nd:
            raise SNRuntimeError(f"axis {axis} is out of range for a {nd}-dimensional tensor")
        return axis + nd if axis < 0 else axis

    # ------------------------------------------------------------ arithmetic
    def binary(self, op, other):
        o = other if isinstance(other, Tensor) else Tensor.from_value(other)
        f = _OPS[op]
        if self.shape == o.shape:
            data = [f(x, y) for x, y in zip(self.data, o.data)]
            shape = self.shape
        elif not o.shape:
            y = o.data[0]
            data = [f(x, y) for x in self.data]
            shape = self.shape
        elif not self.shape:
            x = self.data[0]
            data = [f(x, y) for y in o.data]
            shape = o.shape
        else:
            shape = broadcast_shape(self.shape, o.shape)
            ia = _bidx(self.shape, shape)
            ib = _bidx(o.shape, shape)
            a, b = self.data, o.data
            data = [f(a[i], b[j]) for i, j in zip(ia, ib)]
        return Tensor(data, shape, op in _BOOL_OPS)

    def __add__(self, o):
        return self.binary("+", o)

    __radd__ = __add__

    def __sub__(self, o):
        return self.binary("-", o)

    def __rsub__(self, o):
        return Tensor.from_value(o).binary("-", self)

    def __mul__(self, o):
        return self.binary("*", o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self.binary("/", o)

    def __rtruediv__(self, o):
        return Tensor.from_value(o).binary("/", self)

    def __pow__(self, o):
        return self.binary("**", o)

    def __neg__(self):
        return Tensor([-x for x in self.data], self.shape)

    def __matmul__(self, o):
        return matmul(self, o)

    def map(self, f):
        return Tensor([f(x) for x in self.data], self.shape)

    # ------------------------------------------------------------ reductions
    def _reduce(self, fn, axis=None, keepdims=False, cast=float):
        if axis is None:
            v = cast(fn(self.data))
            if not keepdims:
                return v
            return Tensor([float(v)], (1,) * self.ndim)
        ax = self._axis(axis)
        n = self.shape[ax]
        outer = prod(self.shape[:ax])
        inner = prod(self.shape[ax + 1:])
        out = []
        data = self.data
        for o in range(outer):
            base = o * n * inner
            for i in range(inner):
                out.append(float(cast(fn(data[base + i: base + n * inner: inner]))))
        shape = list(self.shape)
        if keepdims:
            shape[ax] = 1
        else:
            del shape[ax]
        return Tensor(out, shape)

    def m_sum(self, axis=None, keepdims=False):
        return self._reduce(math.fsum, axis, keepdims)

    def m_mean(self, axis=None, keepdims=False):
        return self._reduce(stats.mean, axis, keepdims)

    def m_max(self, axis=None, keepdims=False):
        return self._reduce(max, axis, keepdims)

    def m_min(self, axis=None, keepdims=False):
        return self._reduce(min, axis, keepdims)

    def m_argmax(self, axis=None):
        return self._reduce(lambda xs: xs.index(max(xs)), axis, False, int)

    def m_argmin(self, axis=None):
        return self._reduce(lambda xs: xs.index(min(xs)), axis, False, int)

    def m_prod(self, axis=None):
        return self._reduce(prod, axis)

    def m_std(self, axis=None, ddof=0):
        return self._reduce(lambda xs: stats.std(xs, ddof), axis)

    def m_var(self, axis=None, ddof=0):
        return self._reduce(lambda xs: stats.variance(xs, ddof), axis)

    def m_median(self, axis=None):
        return self._reduce(stats.median, axis)

    def m_all(self):
        return all(x != 0 for x in self.data)

    def m_any(self):
        return any(x != 0 for x in self.data)

    def m_norm(self):
        return math.sqrt(math.fsum(x * x for x in self.data))

    # --------------------------------------------------------------- shaping
    def m_reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, Tensor)):
            shape = shape[0].data if isinstance(shape[0], Tensor) else shape[0]
        shape = [int(s) for s in shape]
        if shape.count(-1) > 1:
            raise SNRuntimeError("only one dimension may be -1")
        if -1 in shape:
            known = prod(s for s in shape if s != -1)
            if known == 0 or self.size % known:
                raise SNRuntimeError(f"cannot reshape {self.size} elements into {shape}")
            shape[shape.index(-1)] = self.size // known
        if prod(shape) != self.size:
            raise SNRuntimeError(f"cannot reshape {self.size} elements into {shape}")
        return Tensor(self.data, shape, self.is_bool)

    def m_flatten(self):
        return Tensor(self.data, (self.size,), self.is_bool)

    def m_transpose(self):
        return self.transpose()

    def transpose(self):
        nd = self.ndim
        if nd < 2:
            return self
        if nd == 2:
            m, n = self.shape
            data = []
            for j in range(n):
                data.extend(self.data[j::n])
            return Tensor(data, (n, m))
        shape = list(reversed(self.shape))
        strides = list(reversed(self._strides()))
        offs = [0]
        for dim, st in zip(shape, strides):
            offs = [o + k * st for o in offs for k in range(dim)]
        return Tensor([self.data[o] for o in offs], shape)

    def m_tolist(self):
        return self.to_nested()

    def m_copy(self):
        return self.copy()

    def m_item(self):
        if self.size != 1:
            raise SNRuntimeError("item() needs a tensor with exactly one element")
        return self.data[0]

    def m_dot(self, other):
        return matmul(self, other)

    def m_abs(self):
        return self.map(abs)

    def m_sqrt(self):
        return self.map(sqrt)

    def m_exp(self):
        return self.map(exp)

    def m_log(self):
        return self.map(log)

    def m_round(self, decimals=0):
        return self.map(lambda x: float(round(x, int(decimals))))

    def m_clip(self, lo=None, hi=None):
        def f(x):
            if lo is not None and x < lo:
                return float(lo)
            if hi is not None and x > hi:
                return float(hi)
            return x
        return self.map(f)

    def m_apply(self, fn):
        return Tensor([float(call_value(fn, x)) for x in self.data], self.shape)

    def m_sort(self):
        if self.ndim != 1:
            raise SNRuntimeError("sort() supports 1-D tensors only")
        return Tensor(sorted(self.data), self.shape)

    def m_argsort(self):
        if self.ndim != 1:
            raise SNRuntimeError("argsort() supports 1-D tensors only")
        return sorted(range(self.size), key=self.data.__getitem__)

    def m_take(self, indices):
        return self.take(indices)

    def take(self, indices):
        """Select rows along axis 0."""
        if isinstance(indices, Tensor):
            indices = indices.data
        if not self.shape:
            raise SNRuntimeError("cannot take from a 0-dimensional tensor")
        inner = prod(self.shape[1:])
        data = []
        for i in indices:
            i = _int_index(i, self.shape[0])
            data.extend(self.data[i * inner:(i + 1) * inner])
        return Tensor(data, (len(indices),) + self.shape[1:])

    # ----------------------------------------------------------- properties
    @property
    def p_shape(self):
        return list(self.shape)

    @property
    def p_ndim(self):
        return self.ndim

    @property
    def p_size(self):
        return self.size

    @property
    def p_T(self):
        return self.transpose()

    # ------------------------------------------------------------- indexing
    def _select(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self.ndim:
            raise SNRuntimeError(
                f"too many indices ({len(key)}) for a tensor with {self.ndim} dimension(s)"
            )
        key = key + (slice(None),) * (self.ndim - len(key))
        offsets = [0]
        out_shape = []
        for k, dim, st in zip(key, self.shape, self._strides()):
            if isinstance(k, slice):
                idx = range(*k.indices(dim))
                out_shape.append(len(idx))
            elif isinstance(k, (Tensor, list)):
                items = k.data if isinstance(k, Tensor) else k
                idx = [_int_index(x, dim) for x in items]
                out_shape.append(len(idx))
            else:
                idx = [_int_index(k, dim)]
            offsets = [o + i * st for o in offsets for i in idx]
        return offsets, out_shape

    def _mask_get(self, mask):
        if mask.shape == self.shape:
            return Tensor([x for x, m in zip(self.data, mask.data) if m])
        if mask.ndim == 1 and self.ndim >= 1 and mask.shape[0] == self.shape[0]:
            return self.take([i for i, m in enumerate(mask.data) if m])
        raise SNRuntimeError(f"mask shape {list(mask.shape)} does not match tensor shape {list(self.shape)}")

    def sn_getitem(self, key):
        if isinstance(key, Tensor) and key.is_bool:
            return self._mask_get(key)
        offsets, out_shape = self._select(key)
        if not out_shape:
            return self.data[offsets[0]]
        return Tensor([self.data[o] for o in offsets], out_shape)

    def sn_setitem(self, key, value):
        if isinstance(key, Tensor) and key.is_bool and key.shape == self.shape:
            idx = [i for i, m in enumerate(key.data) if m]
            vals = Tensor.from_value(value)
            if vals.size == 1:
                for i in idx:
                    self.data[i] = vals.data[0]
            elif vals.size == len(idx):
                for i, v in zip(idx, vals.data):
                    self.data[i] = v
            else:
                raise SNRuntimeError("value size does not match the number of selected elements")
            return
        offsets, out_shape = self._select(key)
        vt = Tensor.from_value(value)
        if tuple(out_shape) != vt.shape:
            if broadcast_shape(vt.shape, tuple(out_shape)) != tuple(out_shape):
                raise SNRuntimeError(
                    f"cannot assign shape {list(vt.shape)} into selection of shape {out_shape}"
                )
            vals = [vt.data[i] for i in _bidx(vt.shape, out_shape)]
        else:
            vals = vt.data
        for o, v in zip(offsets, vals):
            self.data[o] = v

    def sn_iter(self):
        if not self.shape:
            raise SNRuntimeError("cannot iterate over a 0-dimensional tensor")
        if self.ndim == 1:
            return list(self.data)
        return [self.sn_getitem(i) for i in range(self.shape[0])]

    def sn_len(self):
        if not self.shape:
            raise SNRuntimeError("len() of a 0-dimensional tensor")
        return self.shape[0]

    def sn_truthy(self):
        if self.size == 1:
            return bool(self.data[0])
        raise SNRuntimeError(
            f"the truth value of a tensor with {self.size} elements is ambiguous; use .any() or .all()"
        )

    def sn_contains(self, item):
        return item in self.data

    # --------------------------------------------------------------- display
    def sn_repr(self):
        nd = self.ndim
        if self.size > 400:
            head = ", ".join(fmt_number(x, 6) for x in self.data[:6])
            return f"tensor(shape={list(self.shape)}, [{head}, ...])"
        if nd == 0:
            return f"tensor({fmt_number(self.data[0], 6)})"
        nested = self.to_nested()

        def fmt(node, depth, pad):
            if depth == nd - 1:
                return "[" + ", ".join(fmt_number(x, 6) for x in node) + "]"
            inner = [fmt(c, depth + 1, pad + 1) for c in node]
            sep = "," + "\n" * (nd - depth - 1) + " " * (pad + 1)
            return "[" + sep.join(inner) + "]"

        return "tensor(" + fmt(nested, 0, 7) + ")"


# ------------------------------------------------------------------ functions
def matmul(a, b):
    a = Tensor.from_value(a)
    b = Tensor.from_value(b)
    if a.ndim == 0 or b.ndim == 0:
        raise SNRuntimeError("matmul does not accept scalars; use * instead")
    if a.ndim > 2 or b.ndim > 2:
        raise SNRuntimeError("matmul supports 1-D and 2-D tensors only")
    if a.ndim == 1 and b.ndim == 1:
        if a.shape != b.shape:
            raise SNRuntimeError(f"dot product shape mismatch: {list(a.shape)} and {list(b.shape)}")
        return Tensor([float(sum(map(operator.mul, a.data, b.data)))], ())
    m, k = (1, a.shape[0]) if a.ndim == 1 else a.shape
    k2, n = (b.shape[0], 1) if b.ndim == 1 else b.shape
    if k != k2:
        raise SNRuntimeError(f"matmul shape mismatch: {list(a.shape)} @ {list(b.shape)}")
    if _np is not None and m * k * n >= NUMPY_MIN_WORK:
        A = _np.array(a.data, dtype=float).reshape(m, k)
        B = _np.array(b.data, dtype=float).reshape(k, n)
        out = (A @ B).ravel().tolist()
    else:
        rows = [a.data[i * k:(i + 1) * k] for i in range(m)]
        cols = [b.data[j::n] for j in range(n)] if b.ndim == 2 else [b.data]
        mul = operator.mul
        out = [float(sum(map(mul, r, c))) for r in rows for c in cols]
    if a.ndim == 1:
        return Tensor(out, (n,))
    if b.ndim == 1:
        return Tensor(out, (m,))
    return Tensor(out, (m, n))


def softmax(t, axis=-1):
    t = Tensor.from_value(t)
    shifted = t - t._reduce(max, axis, True)
    e = shifted.map(exp)
    return e / e._reduce(math.fsum, axis, True)


def _shape_args(args):
    if len(args) == 1 and isinstance(args[0], (list, Tensor)):
        args = args[0].data if isinstance(args[0], Tensor) else args[0]
    shape = []
    for a in args:
        if isinstance(a, bool) or not isinstance(a, (int, float)) or int(a) != a or a < 0:
            raise SNRuntimeError(f"invalid dimension {a!r}")
        shape.append(int(a))
    return tuple(shape)


def zeros(*shape):
    s = _shape_args(shape)
    return Tensor([0.0] * prod(s), s)


def ones(*shape):
    s = _shape_args(shape)
    return Tensor([1.0] * prod(s), s)


def full(shape, value):
    s = _shape_args([shape])
    return Tensor([float(value)] * prod(s), s)


def eye(n):
    n = int(n)
    data = [0.0] * (n * n)
    for i in range(n):
        data[i * n + i] = 1.0
    return Tensor(data, (n, n))


def arange(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    if step == 0:
        raise SNRuntimeError("arange step cannot be zero")
    n = max(0, math.ceil((stop - start) / step))
    return Tensor([float(start + i * step) for i in range(n)])


def linspace(start, stop, num=50):
    num = int(num)
    if num < 1:
        raise SNRuntimeError("linspace needs num >= 1")
    if num == 1:
        return Tensor([float(start)])
    step = (stop - start) / (num - 1)
    return Tensor([float(start + i * step) for i in range(num)])


def rand(*shape):
    s = _shape_args(shape)
    return Tensor([RNG.random() for _ in range(prod(s))], s)


def randn(*shape):
    s = _shape_args(shape)
    return Tensor([RNG.gauss(0.0, 1.0) for _ in range(prod(s))], s)


def concat(items, axis=0):
    ts = [Tensor.from_value(t) for t in items]
    if not ts:
        raise SNRuntimeError("concat needs at least one tensor")
    first = ts[0]
    ax = first._axis(axis)
    for t in ts:
        if t.ndim != first.ndim or any(
            d1 != d2 for i, (d1, d2) in enumerate(zip(t.shape, first.shape)) if i != ax
        ):
            raise SNRuntimeError("concat: all tensors must have the same shape except along the axis")
    outer = prod(first.shape[:ax])
    out = []
    for o in range(outer):
        for t in ts:
            chunk = prod(t.shape[ax:])
            out.extend(t.data[o * chunk:(o + 1) * chunk])
    shape = list(first.shape)
    shape[ax] = sum(t.shape[ax] for t in ts)
    return Tensor(out, shape)


def stack(items):
    ts = [Tensor.from_value(t) for t in items]
    if not ts:
        raise SNRuntimeError("stack needs at least one tensor")
    if any(t.shape != ts[0].shape for t in ts):
        raise SNRuntimeError("stack: all tensors must have the same shape")
    data = []
    for t in ts:
        data.extend(t.data)
    return Tensor(data, (len(ts),) + ts[0].shape)


def where(cond, a, b):
    c, x, y = Tensor.from_value(cond), Tensor.from_value(a), Tensor.from_value(b)
    shape = broadcast_shape(broadcast_shape(c.shape, x.shape), y.shape)
    ic, ix, iy = _bidx(c.shape, shape), _bidx(x.shape, shape), _bidx(y.shape, shape)
    data = [x.data[i] if c.data[k] else y.data[j] for k, i, j in zip(ic, ix, iy)]
    return Tensor(data, shape)


def gauss_solve(A, B):
    """Solve A X = B for X (lists of lists) with partial pivoting."""
    n = len(A)
    M = [list(map(float, A[i])) + list(map(float, B[i])) for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            raise SNRuntimeError("matrix is singular (or nearly singular)")
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        rc = M[c]
        for r in range(n):
            if r != c:
                f = M[r][c]
                if f != 0.0:
                    M[r] = [x - f * y for x, y in zip(M[r], rc)]
    return [row[n:] for row in M]


def _square(t, name):
    t = Tensor.from_value(t)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise SNRuntimeError(f"{name} needs a square 2-D tensor")
    return t


def solve(A, b):
    A = _square(A, "solve")
    b = Tensor.from_value(b)
    n = A.shape[0]
    if b.ndim == 1:
        if b.shape[0] != n:
            raise SNRuntimeError("solve: shape mismatch")
        X = gauss_solve(A.to_nested(), [[v] for v in b.data])
        return Tensor([r[0] for r in X])
    if b.ndim != 2 or b.shape[0] != n:
        raise SNRuntimeError("solve: shape mismatch")
    X = gauss_solve(A.to_nested(), b.to_nested())
    return Tensor.from_value(X)


def inv(A):
    A = _square(A, "inv")
    n = A.shape[0]
    ident = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return Tensor.from_value(gauss_solve(A.to_nested(), ident))


def det(A):
    A = _square(A, "det")
    n = A.shape[0]
    M = A.to_nested()
    d = 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            return 0.0
        if p != c:
            M[c], M[p] = M[p], M[c]
            d = -d
        d *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return d
