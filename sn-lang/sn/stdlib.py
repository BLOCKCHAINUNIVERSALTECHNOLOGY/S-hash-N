"""Built-in functions and constants available to every S#N program."""
import json
import math
import os
import random as _random
import sys
import time as _time

import functools

from . import autograd as ag, ml, plot as _plot, stats, trees
from . import tensor as T
from .dataframe import DataFrame, make_dataframe, read_csv
from .errors import SNRuntimeError
from .ops import binop, equals, iterate, length
from .tensor import RNG, Tensor
from .objects import Class, Instance
from .values import NativeFn, NativeObject, call_value, to_repr, to_str, truthy, type_name

SN_VERSION = "1.2.1"
BUILTINS = {}
_MISSING = object()


def builtin(*names):
    def deco(fn):
        for nm in names:
            BUILTINS[nm] = NativeFn(names[0], fn)
        return fn
    return deco


def _register(name, fn, *aliases):
    for nm in (name,) + aliases:
        BUILTINS[nm] = NativeFn(name, fn)


BUILTINS.update({
    "PI": math.pi, "E": math.e, "INF": math.inf, "NAN": math.nan,
    "SN_VERSION": SN_VERSION,
})


def _num(v, fname):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SNRuntimeError(f"{fname}() expects a number, got {type_name(v)}")
    return v


def _int(v, fname="value"):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, bool) or not isinstance(v, int):
        raise SNRuntimeError(f"{fname} must be an integer, got {to_repr(v)}")
    return v


def _scalar(v):
    if isinstance(v, Tensor):
        if v.size != 1:
            raise SNRuntimeError("function must return a single number")
        return v.data[0]
    return float(_num(v, "function result"))


# ------------------------------------------------------------------ I/O
@builtin("print", "छापो")
def sn_print(*args, sep=" ", end="\n"):
    sys.stdout.write(sep.join(to_str(a) for a in args) + end)


@builtin("input", "इनपुट")
def sn_input(prompt=""):
    try:
        return input(prompt)
    except EOFError:
        return None


@builtin("read_file")
def sn_read_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise SNRuntimeError(f"file not found: {path}") from None


@builtin("write_file")
def sn_write_file(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_str(text))
    return path


@builtin("append_file")
def sn_append_file(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(to_str(text))
    return path


@builtin("exists")
def sn_exists(path):
    return os.path.exists(path)


@builtin("exit")
def sn_exit(code=0):
    raise SystemExit(code)


@builtin("assert")
def sn_assert(cond, message="assertion failed"):
    if not truthy(cond):
        raise SNRuntimeError(f"Assertion failed: {to_str(message)}")
    return True


def _jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, Tensor):
        return v.to_nested()
    if isinstance(v, DataFrame):
        return {c: _jsonable(x) for c, x in v.cols.items()}
    raise SNRuntimeError(f"cannot convert {type_name(v)} to JSON")


@builtin("to_json")
def sn_to_json(v, indent=None):
    return json.dumps(_jsonable(v), indent=indent, ensure_ascii=False)


@builtin("parse_json")
def sn_parse_json(text):
    try:
        return json.loads(text)
    except ValueError as e:
        raise SNRuntimeError(f"invalid JSON: {e}") from None


@builtin("time")
def sn_time():
    return _time.time()


@builtin("clock")
def sn_clock():
    return _time.perf_counter()


@builtin("sleep")
def sn_sleep(seconds):
    _time.sleep(_num(seconds, "sleep"))


# ---------------------------------------------------------------- types
@builtin("type", "प्रकार")
def sn_type(v):
    return type_name(v)


@builtin("str", "पाठ")
def sn_str(v=""):
    return to_str(v)


@builtin("is_a")
def sn_is_a(value, kind):
    """is_a(obj, ClassName) or is_a(value, "int")."""
    if isinstance(kind, Class):
        return isinstance(value, Instance) and value.cls.is_subclass(kind)
    if isinstance(kind, str):
        return type_name(value) == kind
    raise SNRuntimeError("is_a() expects a class or a type name string")


@builtin("repr")
def sn_repr(v):
    return to_repr(v)


@builtin("int", "पूर्णांक")
def sn_int(v, base=10):
    if isinstance(v, str):
        s = v.strip()
        try:
            return int(s, base)
        except ValueError:
            try:
                return int(float(s))
            except (ValueError, OverflowError):
                raise SNRuntimeError(f"cannot convert {to_repr(v)} to int") from None
    if isinstance(v, (int, float)):
        if v != v or v in (math.inf, -math.inf):
            raise SNRuntimeError("cannot convert inf/nan to int")
        return int(v)
    raise SNRuntimeError(f"cannot convert {type_name(v)} to int")


@builtin("float", "दशमलव")
def sn_float(v):
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            raise SNRuntimeError(f"cannot convert {to_repr(v)} to float") from None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Tensor) and v.size == 1:
        return v.data[0]
    raise SNRuntimeError(f"cannot convert {type_name(v)} to float")


@builtin("bool")
def sn_bool(v=False):
    return truthy(v)


@builtin("list")
def sn_list(v=_MISSING):
    return [] if v is _MISSING else list(iterate(v))


@builtin("len", "लंबाई")
def sn_len(v):
    return length(v)


@builtin("range", "श्रेणी")
def sn_range(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    start, stop, step = _int(start, "range start"), _int(stop, "range stop"), _int(step, "range step")
    if step == 0:
        raise SNRuntimeError("range() step cannot be zero")
    if abs((stop - start) // step) > 20_000_000:
        raise SNRuntimeError("range() is too large")
    return list(range(start, stop, step))


@builtin("format")
def sn_format(v, spec=""):
    from .ops import format_value
    return format_value(v, spec or None)


# ----------------------------------------------------------------- math
def _unary(name, f, *aliases):
    def fn(x):
        if isinstance(x, Tensor):
            return x.map(f)
        if isinstance(x, list):
            return Tensor.from_value(x).map(f)
        if isinstance(x, (int, float)):
            return f(x)
        raise SNRuntimeError(f"{name}() expects a number or tensor, got {type_name(x)}")
    _register(name, fn, *aliases)


_unary("sqrt", T.sqrt)
_unary("exp", T.exp)
_unary("sin", math.sin)
_unary("cos", math.cos)
_unary("tan", math.tan)
_unary("asin", math.asin)
_unary("acos", math.acos)
_unary("atan", math.atan)
_unary("tanh", math.tanh)
_unary("log2", lambda x: T.log(x) / math.log(2))
_unary("log10", lambda x: T.log(x) / math.log(10))
_unary("sigmoid", T.sigmoid)
_unary("relu", T.relu)


@builtin("log")
def sn_log(x, base=None):
    f = (lambda v: T.log(v)) if base is None else (lambda v: T.log(v) / math.log(base))
    if isinstance(x, (Tensor, list)):
        return Tensor.from_value(x).map(f)
    return f(_num(x, "log"))


@builtin("abs", "निरपेक्ष")
def sn_abs(x):
    if isinstance(x, (Tensor, list)):
        return Tensor.from_value(x).map(abs)
    return abs(_num(x, "abs"))


@builtin("floor")
def sn_floor(x):
    if isinstance(x, (Tensor, list)):
        return Tensor.from_value(x).map(lambda v: float(math.floor(v)))
    return math.floor(_num(x, "floor"))


@builtin("ceil")
def sn_ceil(x):
    if isinstance(x, (Tensor, list)):
        return Tensor.from_value(x).map(lambda v: float(math.ceil(v)))
    return math.ceil(_num(x, "ceil"))


@builtin("round", "गोल")
def sn_round(x, digits=0):
    if isinstance(x, (Tensor, list)):
        return Tensor.from_value(x).m_round(digits)
    x = _num(x, "round")
    digits = _int(digits, "digits")
    return int(round(x)) if digits == 0 else round(x, digits)


@builtin("pow")
def sn_pow(a, b):
    return binop("**", a, b)


@builtin("atan2")
def sn_atan2(y, x):
    return math.atan2(y, x)


@builtin("clip")
def sn_clip(x, lo=None, hi=None):
    if isinstance(x, (Tensor, list)):
        return Tensor.from_value(x).m_clip(lo, hi)
    if lo is not None and x < lo:
        return lo
    if hi is not None and x > hi:
        return hi
    return x


def _numbers(v, fname):
    if isinstance(v, Tensor):
        return v.data
    if isinstance(v, list):
        if any(isinstance(x, list) for x in v):
            return Tensor.from_value(v).data
        for x in v:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise SNRuntimeError(f"{fname}() needs a list of numbers, found {type_name(x)}")
        return v
    raise SNRuntimeError(f"{fname}() expects a list or tensor, got {type_name(v)}")


def _minmax(name, fn):
    def impl(*args, key=None):
        if len(args) == 1:
            seq = args[0]
            seq = seq.data if isinstance(seq, Tensor) else iterate(seq)
        else:
            seq = args
        if len(seq) == 0:
            raise SNRuntimeError(f"{name}() of an empty sequence")
        if key is not None:
            return fn(seq, key=lambda v: call_value(key, v))
        return fn(seq)
    _register(name, impl)


_minmax("min", min)
_minmax("max", max)
BUILTINS["न्यूनतम"] = BUILTINS["min"]
BUILTINS["अधिकतम"] = BUILTINS["max"]


@builtin("sum", "योग")
def sn_sum(v, start=0):
    if isinstance(v, Tensor):
        return math.fsum(v.data) + start
    return sum(_numbers(v, "sum"), start)


def _stat(name, fn_list, tensor_method, *aliases, extra=()):
    def impl(v, axis=None, **kw):
        if axis is not None:
            return getattr(Tensor.from_value(v), tensor_method)(axis=axis, **kw)
        return fn_list(_numbers(v, name), **kw)
    _register(name, impl, *aliases)


_stat("mean", stats.mean, "m_mean", "औसत")
_stat("median", stats.median, "m_median", "माध्यिका")
_stat("var", stats.variance, "m_var")
_stat("std", stats.std, "m_std")


@builtin("percentile")
def sn_percentile(v, q):
    return stats.percentile(_numbers(v, "percentile"), q)


@builtin("mode")
def sn_mode(v):
    return stats.mode(iterate(v) if not isinstance(v, Tensor) else v.data)


@builtin("corr")
def sn_corr(a, b):
    return stats.pearson(_numbers(a, "corr"), _numbers(b, "corr"))


# ---------------------------------------------------------- collections
@builtin("sorted", "क्रमबद्ध")
def sn_sorted(v, key=None, reverse=False):
    seq = v.data if isinstance(v, Tensor) else iterate(v)
    return sorted(seq, key=(lambda x: call_value(key, x)) if key is not None else None,
                  reverse=truthy(reverse))


@builtin("reversed")
def sn_reversed(v):
    if isinstance(v, str):
        return v[::-1]
    return list(reversed(iterate(v)))


@builtin("map")
def sn_map(xs, f):
    return [call_value(f, x) for x in iterate(xs)]


@builtin("filter")
def sn_filter(xs, f):
    return [x for x in iterate(xs) if truthy(call_value(f, x))]


@builtin("reduce")
def sn_reduce(xs, f, init=_MISSING):
    it = iter(iterate(xs))
    if init is _MISSING:
        try:
            acc = next(it)
        except StopIteration:
            raise SNRuntimeError("reduce() of an empty sequence with no initial value") from None
    else:
        acc = init
    for x in it:
        acc = call_value(f, acc, x)
    return acc


@builtin("zip")
def sn_zip(*seqs):
    return [list(t) for t in zip(*[iterate(s) for s in seqs])]


@builtin("enumerate")
def sn_enumerate(xs, start=0):
    return [[i, x] for i, x in enumerate(iterate(xs), _int(start))]


@builtin("all")
def sn_all(xs):
    return all(truthy(x) for x in iterate(xs))


@builtin("any")
def sn_any(xs):
    return any(truthy(x) for x in iterate(xs))


@builtin("unique")
def sn_unique(xs):
    seen, out = [], []
    for x in iterate(xs):
        if not any(equals(x, s) for s in seen):
            seen.append(x)
            out.append(x)
    return out


@builtin("flatten")
def sn_flatten(xs):
    out = []

    def walk(v):
        if isinstance(v, Tensor):
            out.extend(v.data)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        else:
            out.append(v)

    walk(xs)
    return out


@builtin("keys")
def sn_keys(d):
    if not isinstance(d, dict):
        raise SNRuntimeError(f"keys() expects a map, got {type_name(d)}")
    return list(d.keys())


@builtin("values")
def sn_values(d):
    if not isinstance(d, dict):
        raise SNRuntimeError(f"values() expects a map, got {type_name(d)}")
    return list(d.values())


@builtin("items")
def sn_items(d):
    if not isinstance(d, dict):
        raise SNRuntimeError(f"items() expects a map, got {type_name(d)}")
    return [[k, v] for k, v in d.items()]


# --------------------------------------------------------------- random
@builtin("seed")
def sn_seed(n):
    RNG.seed(n)


@builtin("random")
def sn_random():
    return RNG.random()


@builtin("randint")
def sn_randint(lo, hi):
    return RNG.randint(_int(lo), _int(hi))


@builtin("uniform")
def sn_uniform(lo, hi):
    return RNG.uniform(lo, hi)


@builtin("normal")
def sn_normal(mu=0.0, sigma=1.0):
    return RNG.gauss(mu, sigma)


@builtin("choice")
def sn_choice(xs):
    seq = iterate(xs)
    if not seq:
        raise SNRuntimeError("choice() of an empty sequence")
    return RNG.choice(seq)


@builtin("shuffle")
def sn_shuffle(xs):
    out = list(iterate(xs))
    RNG.shuffle(out)
    return out


@builtin("sample")
def sn_sample(xs, k):
    return RNG.sample(list(iterate(xs)), _int(k))


# -------------------------------------------------------------- tensors
@builtin("tensor")
def sn_tensor(v):
    return v.copy() if isinstance(v, Tensor) else Tensor.from_value(v)


_register("zeros", T.zeros)
_register("ones", T.ones)
_register("full", T.full)
_register("eye", T.eye)
_register("arange", T.arange)
_register("linspace", T.linspace)
_register("rand", T.rand)
_register("randn", T.randn)
_register("concat", T.concat)
_register("stack", T.stack)
_register("where", T.where)
_register("solve", T.solve)
_register("inv", T.inv)
_register("det", T.det)


@builtin("matmul", "dot")
def sn_matmul(a, b):
    return T.matmul(a, b)


@builtin("shape")
def sn_shape(v):
    if isinstance(v, DataFrame):
        return v.p_shape
    return Tensor.from_value(v).p_shape


@builtin("reshape")
def sn_reshape(v, *shape):
    return Tensor.from_value(v).m_reshape(*shape)


@builtin("transpose")
def sn_transpose(v):
    return Tensor.from_value(v).transpose()


@builtin("softmax")
def sn_softmax(v, axis=-1):
    return T.softmax(v, axis)


@builtin("argmax")
def sn_argmax(v, axis=None):
    return Tensor.from_value(v).m_argmax(axis)


@builtin("argmin")
def sn_argmin(v, axis=None):
    return Tensor.from_value(v).m_argmin(axis)


@builtin("norm")
def sn_norm(v):
    return Tensor.from_value(v).m_norm()


@builtin("equal")
def sn_equal(a, b):
    if isinstance(a, (Tensor, list)) and isinstance(b, (Tensor, list)):
        ta, tb = Tensor.from_value(a), Tensor.from_value(b)
        return ta.shape == tb.shape and ta.data == tb.data
    return equals(a, b)


@builtin("allclose")
def sn_allclose(a, b, tol=1e-6):
    ta, tb = Tensor.from_value(a), Tensor.from_value(b)
    d = ta.binary("-", tb)
    return all(abs(x) <= tol for x in d.data)


@builtin("grad")
def sn_grad(f, x, eps=1e-5):
    """Numerical gradient of f at x (central differences)."""
    if isinstance(x, (int, float)):
        return (_scalar(call_value(f, x + eps)) - _scalar(call_value(f, x - eps))) / (2 * eps)
    t = Tensor.from_value(x)
    g = []
    for i in range(t.size):
        hi, lo = t.copy(), t.copy()
        hi.data[i] += eps
        lo.data[i] -= eps
        g.append((_scalar(call_value(f, hi)) - _scalar(call_value(f, lo))) / (2 * eps))
    return Tensor(g, t.shape)


@builtin("gradient_descent")
def sn_gradient_descent(f, x0, lr=0.1, steps=100):
    x = x0 if isinstance(x0, (int, float)) else Tensor.from_value(x0)
    for _ in range(_int(steps)):
        x = x - sn_grad(f, x) * lr
    return x


# ---------------------------------------------------------------- autograd
_register("Var", ag.make_var)
_register("autograd", ag.autograd)
_register("value_and_grad", ag.value_and_grad)
_register("SGD", ag.SGD)
_register("Adam", ag.Adam)


def _var_aware(names, handler):
    """Make existing builtins accept Vars (and record gradients) too."""
    for name in names:
        orig = BUILTINS[name].fn

        def make(orig):
            @functools.wraps(orig)
            def wrapper(*args, **kwargs):
                if any(isinstance(a, ag.Var) for a in args):
                    return handler(*args, **kwargs)
                return orig(*args, **kwargs)
            return wrapper

        wrapper = make(orig)
        for k, v in list(BUILTINS.items()):
            if isinstance(v, NativeFn) and v.fn is orig:
                BUILTINS[k] = NativeFn(v.name, wrapper)


_var_aware(["exp"], ag.exp)
_var_aware(["sqrt"], ag.sqrt)
_var_aware(["sin"], ag.sin)
_var_aware(["cos"], ag.cos)
_var_aware(["tanh"], ag.tanh)
_var_aware(["sigmoid"], ag.sigmoid)
_var_aware(["relu"], ag.relu)
_var_aware(["log"], ag.log)
_var_aware(["abs"], ag.abs_)
_var_aware(["sum"], lambda v, start=0: ag.add(ag.sum_(v), start) if start else ag.sum_(v))
_var_aware(["mean"], lambda v, axis=None, **kw: ag.mean(v, axis, **kw))
_var_aware(["matmul"], ag.matmul_)
_var_aware(["softmax"], ag.softmax)
_var_aware(["reshape"], ag.reshape)
_var_aware(["transpose"], ag.transpose)
_var_aware(["pow"], ag.pow_)
_var_aware(["norm"], ag.norm)
_var_aware(["shape"], lambda v: v.p_shape)


# ----------------------------------------------------------------- charts
_register("Plot", _plot.Plot)
_register("plot", _plot.plot)
_register("scatter", _plot.scatter)
_register("bar_chart", _plot.bar_chart)
_register("histogram", _plot.histogram)
_register("heatmap", _plot.heatmap)


@builtin("backend")
def sn_backend():
    """"numpy" when NumPy accelerates big matrix products, else "python"."""
    return T.backend()


@builtin("make_dir")
def sn_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------- data and ML
_register("DataFrame", make_dataframe)
_register("read_csv", read_csv)
_register("LinearRegression", ml.LinearRegression)
_register("LogisticRegression", ml.LogisticRegression)
_register("KNN", ml.KNN)
_register("KMeans", ml.KMeans)
_register("NeuralNet", ml.NeuralNet)
_register("DecisionTree", trees.DecisionTree)
_register("RandomForest", trees.RandomForest)
_register("StandardScaler", ml.StandardScaler)
_register("MinMaxScaler", ml.MinMaxScaler)
_register("load_model", ml.load_model)
_register("train_test_split", ml.train_test_split)
_register("label_encode", ml.label_encode)
_register("one_hot", ml.one_hot)
for _name in ("accuracy", "mse", "rmse", "mae", "r2", "confusion_matrix",
              "precision_score", "recall_score", "f1_score"):
    _register(_name, getattr(ml, _name))
