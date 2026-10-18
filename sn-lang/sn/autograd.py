"""Reverse-mode automatic differentiation (autograd) for S#N.

A ``Var`` wraps a tensor and remembers how it was computed.  Calling
``backward()`` on a single-number result walks the recorded graph in reverse
and fills in ``.grad`` on every leaf ``Var`` that asked for gradients.

    let w = Var([1.0, 2.0])
    let loss = ((w * w).sum())
    loss.backward()
    print(w.grad)            # tensor([2., 4.])

Every operation below stores a small closure that maps the gradient flowing
into its output onto gradients for its inputs.  Gradients are plain ``Tensor``
objects, so the whole thing runs on the same code (and the same NumPy
accelerated ``matmul``) as the rest of the language.
"""
import math

from .errors import SNRuntimeError
from .tensor import Tensor, matmul
from .values import NativeObject, call_value, suggest, type_name


# ------------------------------------------------------------------ helpers
def _unbroadcast(g, shape):
    """Sum ``g`` down to ``shape`` (undoing NumPy-style broadcasting)."""
    shape = tuple(shape)
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g._reduce(math.fsum, 0)
    for ax, (gd, sd) in enumerate(zip(g.shape, shape)):
        if sd == 1 and gd != 1:
            g = g._reduce(math.fsum, ax, True)
    return g


def _as_tensor(v):
    if isinstance(v, Var):
        return v.value
    if isinstance(v, bool):
        return Tensor([float(v)], ())
    if isinstance(v, (int, float, list, Tensor)):
        return Tensor.from_value(v)
    raise SNRuntimeError(f"cannot use {type_name(v)} in a differentiable expression")


def lift(v):
    """Turn numbers, lists and tensors into constant (no-gradient) Vars."""
    if isinstance(v, Var):
        return v
    return Var(_as_tensor(v), requires_grad=False)


def _node(value, parents, back):
    if any(p.requires_grad for p in parents):
        return Var(value, parents, back, True)
    return Var(value, requires_grad=False)


# --------------------------------------------------------------------- Var
class Var(NativeObject):
    """A tensor that records the operations applied to it."""

    __slots__ = ("value", "grad", "parents", "back", "requires_grad")
    sn_type = "var"

    def __init__(self, value, parents=(), back=None, requires_grad=True):
        self.value = value
        self.grad = None
        self.parents = parents
        self.back = back
        self.requires_grad = requires_grad

    # ----------------------------------------------------------- properties
    @property
    def p_value(self):
        return self.value

    @property
    def p_grad(self):
        return self.grad

    @property
    def p_shape(self):
        return list(self.value.shape)

    @property
    def p_ndim(self):
        return self.value.ndim

    @property
    def p_size(self):
        return self.value.size

    @property
    def p_T(self):
        return transpose(self)

    @property
    def p_requires_grad(self):
        return self.requires_grad

    @property
    def p_is_leaf(self):
        return self.back is None

    # -------------------------------------------------------------- methods
    def m_backward(self, grad=None):
        backward(self, grad)

    def m_zero_grad(self):
        self.grad = None

    def m_detach(self):
        return Var(self.value, requires_grad=False)

    def m_item(self):
        if self.value.size != 1:
            raise SNRuntimeError("item() needs a single-number var")
        return self.value.data[0]

    def m_tolist(self):
        return self.value.to_nested()

    def m_sum(self, axis=None, keepdims=False):
        return sum_(self, axis, keepdims)

    def m_mean(self, axis=None, keepdims=False):
        return mean(self, axis, keepdims)

    def m_exp(self):
        return exp(self)

    def m_log(self):
        return log(self)

    def m_sqrt(self):
        return sqrt(self)

    def m_abs(self):
        return abs_(self)

    def m_tanh(self):
        return tanh(self)

    def m_sigmoid(self):
        return sigmoid(self)

    def m_relu(self):
        return relu(self)

    def m_sin(self):
        return sin(self)

    def m_cos(self):
        return cos(self)

    def m_square(self):
        return mul(self, self)

    def m_pow(self, other):
        return pow_(self, other)

    def m_matmul(self, other):
        return matmul_(self, other)

    m_dot = m_matmul

    def m_softmax(self, axis=-1):
        return softmax(self, axis)

    def m_reshape(self, *shape):
        return reshape(self, *shape)

    def m_flatten(self):
        return reshape(self, -1)

    def m_transpose(self):
        return transpose(self)

    # -------------------------------------------------------- object protocol
    def sn_set(self, name, value):
        if name == "value":
            if self.back is not None:
                raise SNRuntimeError("only a leaf var can be assigned a new value")
            t = _as_tensor(value)
            if t.shape != self.value.shape:
                raise SNRuntimeError(
                    f"shape mismatch: var has shape {list(self.value.shape)}, got {list(t.shape)}"
                )
            self.value = t.copy()
        elif name == "grad":
            if value is None:
                self.grad = None
            else:
                t = _as_tensor(value)
                if t.shape != self.value.shape:
                    raise SNRuntimeError("grad must have the same shape as the var")
                self.grad = t.copy()
        else:
            super().sn_set(name, value)

    def sn_getitem(self, key):
        return getitem(self, key)

    def sn_len(self):
        return self.value.sn_len()

    def sn_iter(self):
        if not self.value.shape:
            raise SNRuntimeError("cannot iterate over a 0-dimensional var")
        return [getitem(self, i) for i in range(self.value.shape[0])]

    def sn_truthy(self):
        return self.value.sn_truthy()

    def sn_repr(self):
        inner = self.value.sn_repr()
        if inner.startswith("tensor(") and inner.endswith(")"):
            inner = inner[7:-1]
        return f"var({inner})"

    def __format__(self, spec):
        if self.value.size != 1:
            raise ValueError("only single-number vars support format specs")
        return format(self.value.data[0], spec)

    def __float__(self):
        return float(self.m_item())


# --------------------------------------------------------------- operations
def add(a, b):
    a, b = lift(a), lift(b)
    out = a.value + b.value
    sa, sb = a.value.shape, b.value.shape
    return _node(out, (a, b), lambda g: (_unbroadcast(g, sa), _unbroadcast(g, sb)))


def sub(a, b):
    a, b = lift(a), lift(b)
    out = a.value - b.value
    sa, sb = a.value.shape, b.value.shape
    return _node(out, (a, b), lambda g: (_unbroadcast(g, sa), _unbroadcast(-g, sb)))


def mul(a, b):
    a, b = lift(a), lift(b)
    av, bv = a.value, b.value
    out = av * bv
    return _node(out, (a, b), lambda g: (_unbroadcast(g * bv, av.shape), _unbroadcast(g * av, bv.shape)))


def div(a, b):
    a, b = lift(a), lift(b)
    av, bv = a.value, b.value
    out = av / bv

    def back(g):
        return (_unbroadcast(g / bv, av.shape),
                _unbroadcast(-(g * av) / (bv * bv), bv.shape))

    return _node(out, (a, b), back)


def neg(a):
    a = lift(a)
    return _node(-a.value, (a,), lambda g: (-g,))


def pow_(a, b):
    a, b = lift(a), lift(b)
    av, bv = a.value, b.value
    out = av ** bv

    def back(g):
        da = _unbroadcast(g * bv * (av ** (bv - 1.0)), av.shape) if a.requires_grad else None
        db = None
        if b.requires_grad:
            db = _unbroadcast(g * out * av.map(lambda x: math.log(x) if x > 0 else math.nan), bv.shape)
        return da, db

    return _node(out, (a, b), back)


def matmul_(a, b):
    a, b = lift(a), lift(b)
    av, bv = a.value, b.value
    out = matmul(av, bv)
    ash, bsh = av.shape, bv.shape

    def back(g):
        A2 = Tensor(av.data, (1, ash[0])) if len(ash) == 1 else av
        B2 = Tensor(bv.data, (bsh[0], 1)) if len(bsh) == 1 else bv
        G = Tensor(g.data, (A2.shape[0], B2.shape[1]))
        dA = matmul(G, B2.transpose())
        dB = matmul(A2.transpose(), G)
        return Tensor(dA.data, ash), Tensor(dB.data, bsh)

    return _node(out, (a, b), back)


def sum_(a, axis=None, keepdims=False):
    a = lift(a)
    av = a.value
    if axis is None:
        out = Tensor([math.fsum(av.data)], (1,) * av.ndim if keepdims else ())

        def back(g):
            return (Tensor([g.data[0]] * av.size, av.shape),)
    else:
        ax = av._axis(axis)
        out = av._reduce(math.fsum, ax, keepdims)
        kshape = list(av.shape)
        kshape[ax] = 1

        def back(g):
            return (Tensor([0.0] * av.size, av.shape) + Tensor(g.data, kshape),)

    return _node(out, (a,), back)


def mean(a, axis=None, keepdims=False):
    a = lift(a)
    if axis is None:
        n = a.value.size
    else:
        n = a.value.shape[a.value._axis(axis)]
    if n == 0:
        raise SNRuntimeError("mean of an empty var")
    return div(sum_(a, axis, keepdims), float(n))


def _unary_op(a, f, df):
    a = lift(a)
    av = a.value
    out = av.map(f)
    return _node(out, (a,), lambda g: (g * df(av, out),))


def exp(a):
    from .tensor import exp as f
    return _unary_op(a, f, lambda x, o: o)


def log(a, base=None):
    from .tensor import log as f
    r = _unary_op(a, f, lambda x, o: 1.0 / x)
    return r if base is None else div(r, math.log(base))


def sqrt(a):
    from .tensor import sqrt as f
    return _unary_op(a, f, lambda x, o: 0.5 / o)


def tanh(a):
    return _unary_op(a, math.tanh, lambda x, o: 1.0 - o * o)


def sigmoid(a):
    from .tensor import sigmoid as f
    return _unary_op(a, f, lambda x, o: o * (1.0 - o))


def relu(a):
    return _unary_op(a, lambda v: v if v > 0 else 0.0,
                     lambda x, o: x.map(lambda v: 1.0 if v > 0 else 0.0))


def abs_(a):
    return _unary_op(a, abs, lambda x, o: x.map(lambda v: (v > 0) - (v < 0) + 0.0))


def sin(a):
    return _unary_op(a, math.sin, lambda x, o: x.map(math.cos))


def cos(a):
    return _unary_op(a, math.cos, lambda x, o: x.map(lambda v: -math.sin(v)))


def reshape(a, *shape):
    a = lift(a)
    av = a.value
    out = av.m_reshape(*shape)
    return _node(out, (a,), lambda g: (Tensor(g.data, av.shape),))


def transpose(a):
    a = lift(a)
    return _node(a.value.transpose(), (a,), lambda g: (g.transpose(),))


def getitem(a, key):
    a = lift(a)
    av = a.value
    if isinstance(key, Tensor) and key.is_bool:
        if key.shape == av.shape:
            offsets = [i for i, m in enumerate(key.data) if m]
            out_shape = [len(offsets)]
        elif key.ndim == 1 and av.ndim >= 1 and key.shape[0] == av.shape[0]:
            offsets, out_shape = av._select([i for i, m in enumerate(key.data) if m])
        else:
            raise SNRuntimeError(
                f"mask shape {list(key.shape)} does not match var shape {list(av.shape)}"
            )
    else:
        if isinstance(key, Var):
            key = key.value
        offsets, out_shape = av._select(key)
    out = Tensor([av.data[o] for o in offsets], out_shape)

    def back(g):
        z = [0.0] * av.size
        for o, gv in zip(offsets, g.data):
            z[o] += gv
        return (Tensor(z, av.shape),)

    return _node(out, (a,), back)


def softmax(a, axis=-1):
    a = lift(a)
    shift = a.value._reduce(max, axis, True)          # constant: keeps exp() from overflowing
    e = exp(sub(a, shift))
    return div(e, sum_(e, axis, True))


def norm(a):
    return sqrt(sum_(mul(a, a)))


_COMPARE = {"==", "!=", "<", ">", "<=", ">="}
_ARITH = {"+": add, "-": sub, "*": mul, "/": div, "**": pow_, "@": matmul_}


def var_binop(op, a, b):
    """Called by the interpreter when either operand of ``op`` is a Var."""
    for v in (a, b):
        if not isinstance(v, (Var, Tensor, int, float, list)) or isinstance(v, bool):
            raise SNRuntimeError(
                f"unsupported operand types for '{op}': {type_name(a)} and {type_name(b)}"
            )
    if op in _COMPARE:
        return _as_tensor(a).binary(op, _as_tensor(b))
    fn = _ARITH.get(op)
    if fn is None:
        raise SNRuntimeError(f"operator '{op}' is not differentiable; use .detach() first")
    return fn(a, b)


# ---------------------------------------------------------------- backward
def backward(root, grad=None):
    if not root.requires_grad:
        raise SNRuntimeError("backward(): this value does not depend on any Var that requires gradients")
    if grad is None:
        if root.value.size != 1:
            raise SNRuntimeError(
                "backward() needs a single-number result; reduce with .sum() or .mean(), "
                "or pass the gradient to start from"
            )
        g0 = Tensor([1.0], root.value.shape)
    else:
        g0 = _as_tensor(grad)
        if g0.shape != root.value.shape:
            raise SNRuntimeError(
                f"backward(): gradient shape {list(g0.shape)} does not match {list(root.value.shape)}"
            )

    order, seen = [], set()
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for p in node.parents:
            if p.requires_grad and id(p) not in seen:
                stack.append((p, False))

    grads = {id(root): g0}
    for node in reversed(order):
        g = grads.pop(id(node), None)
        if g is None:
            continue
        if node.back is None:
            node.grad = g if node.grad is None else node.grad + g
            continue
        for p, pg in zip(node.parents, node.back(g)):
            if pg is None or not p.requires_grad:
                continue
            prev = grads.get(id(p))
            grads[id(p)] = pg if prev is None else prev + pg


# ----------------------------------------------------------- user functions
def make_var(value, requires_grad=True):
    """``Var(x)``: a trainable value (a copy of ``x``)."""
    if isinstance(value, Var):
        value = value.value
    return Var(_as_tensor(value).copy(), requires_grad=bool(requires_grad))


def _scalar_result(r, fname):
    if not isinstance(r, Var):
        if isinstance(r, (int, float)) and not isinstance(r, bool):
            return None
        raise SNRuntimeError(f"{fname}(): the function must return a number computed from its argument")
    if r.value.size != 1:
        raise SNRuntimeError(f"{fname}(): the function must return a single number (got shape {list(r.value.shape)})")
    return r


def value_and_grad(f, x):
    """Return ``[f(x), df/dx]`` computed exactly by reverse-mode autodiff."""
    leaf = make_var(x)
    r = call_value(f, leaf)
    s = _scalar_result(r, "value_and_grad")
    if s is None:
        value, g = float(r), None
    else:
        value = s.value.data[0]
        g = None
        if s.requires_grad:
            backward(s)
            g = leaf.grad
    if g is None:
        g = Tensor([0.0] * leaf.value.size, leaf.value.shape)
    scalar_in = isinstance(x, (int, float)) and not isinstance(x, bool)
    return [value, g.data[0] if scalar_in else g]


def autograd(f, x):
    """Exact gradient of ``f`` at ``x`` (number, list or tensor)."""
    return value_and_grad(f, x)[1]


# --------------------------------------------------------------- optimizers
def _params(params, who):
    if isinstance(params, Var):
        params = [params]
    if not isinstance(params, list) or not params:
        raise SNRuntimeError(f"{who}() needs a non-empty list of Vars")
    for p in params:
        if not isinstance(p, Var) or p.back is not None or not p.requires_grad:
            raise SNRuntimeError(f"{who}() parameters must be Vars created with Var(...)")
    return list(params)


class _Optimizer(NativeObject):
    __slots__ = ("params", "lr", "weight_decay", "t")

    def m_zero_grad(self):
        for p in self.params:
            p.grad = None

    def sn_repr(self):
        return f"<{self.sn_type} lr={self.lr:g} params={len(self.params)}>"

    def _grad(self, p):
        g = p.grad
        if g is not None and self.weight_decay:
            g = g + p.value * self.weight_decay
        return g

    @property
    def p_lr(self):
        return self.lr

    def sn_set(self, name, value):
        if name == "lr":
            self.lr = float(value)
        else:
            super().sn_set(name, value)


class SGD(_Optimizer):
    """Stochastic gradient descent with optional momentum."""

    __slots__ = ("momentum", "vel")
    sn_type = "SGD"

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = _params(params, "SGD")
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.vel = [None] * len(self.params)
        self.t = 0

    def m_step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = self._grad(p)
            if g is None:
                continue
            if self.momentum:
                v = g if self.vel[i] is None else self.vel[i] * self.momentum + g
                self.vel[i] = v
                g = v
            p.value = p.value - g * self.lr


class Adam(_Optimizer):
    """Adam optimizer (Kingma & Ba, 2015) with bias correction."""

    __slots__ = ("b1", "b2", "eps", "m", "v")
    sn_type = "Adam"

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.params = _params(params, "Adam")
        self.lr = float(lr)
        self.b1, self.b2, self.eps = float(beta1), float(beta2), float(eps)
        self.weight_decay = float(weight_decay)
        self.m = [None] * len(self.params)
        self.v = [None] * len(self.params)
        self.t = 0

    def m_step(self):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            g = self._grad(p)
            if g is None:
                continue
            m = g * (1.0 - self.b1) if self.m[i] is None else self.m[i] * self.b1 + g * (1.0 - self.b1)
            v = (g * g) * (1.0 - self.b2) if self.v[i] is None else self.v[i] * self.b2 + (g * g) * (1.0 - self.b2)
            self.m[i], self.v[i] = m, v
            denom = (v / c2).map(math.sqrt) + self.eps
            p.value = p.value - (m / c1) / denom * self.lr
