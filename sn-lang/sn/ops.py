"""Operators and the object protocol (indexing, iteration, member access)."""
import functools
import math

from .errors import SNRuntimeError
from .methods import DICT_METHODS, LIST_METHODS, STR_METHODS
from . import autograd as _ag
from .objects import Instance, instance_binop
from .tensor import Tensor, matmul
from .values import NativeFn, NativeObject, suggest, to_repr, truthy, type_name


# ---------------------------------------------------------------- arithmetic
def _pow(a, b):
    try:
        r = a ** b
    except ZeroDivisionError:
        raise SNRuntimeError("0 cannot be raised to a negative power") from None
    except OverflowError:
        raise SNRuntimeError("result of ** is too large") from None
    if isinstance(r, complex):
        raise SNRuntimeError("math domain error: negative number raised to a fractional power")
    return r


def _div(a, b):
    if b == 0:
        raise SNRuntimeError("division by zero")
    return a / b


def _floordiv(a, b):
    if b == 0:
        raise SNRuntimeError("division by zero")
    return a // b


def _mod(a, b):
    if b == 0:
        raise SNRuntimeError("modulo by zero")
    return a % b


_NUM_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": _div,
    "//": _floordiv,
    "%": _mod,
    "**": _pow,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}
_NUM_TYPES = (int, float, bool)


def _unsupported(op, a, b):
    hint = ""
    if op == "+" and (type(a) is str or type(b) is str):
        hint = " (convert with str(), or use an f-string: f\"... {value}\")"
    return SNRuntimeError(
        f"unsupported operand types for '{op}': {type_name(a)} and {type_name(b)}{hint}"
    )


def equals(a, b):
    if a is b:
        return True
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        return False
    return a == b


def binop(op, a, b):
    ta, tb = type(a), type(b)
    if ta in _NUM_TYPES and tb in _NUM_TYPES:
        fn = _NUM_OPS.get(op)
        if fn is None:
            raise _unsupported(op, a, b)
        return fn(a, b)
    if ta is _ag.Var or tb is _ag.Var:
        if op in ("in", "not in"):
            raise _unsupported(op, a, b)
        return _ag.var_binop(op, a, b)
    if ta is Tensor or tb is Tensor:
        if op == "in":
            return contains(b, a)
        if op == "not in":
            return not contains(b, a)
        for v in (a, b):
            if not isinstance(v, (Tensor, int, float, list)):
                raise _unsupported(op, a, b)
        ta_, tb_ = Tensor.from_value(a), Tensor.from_value(b)
        if op == "@":
            return matmul(ta_, tb_)
        return ta_.binary(op, tb_)
    if ta is Instance or tb is Instance:
        return instance_binop(op, a, b)
    if op == "==":
        return equals(a, b)
    if op == "!=":
        return not equals(a, b)
    if op == "in":
        return contains(b, a)
    if op == "not in":
        return not contains(b, a)
    if ta is str and tb is str:
        if op == "+":
            return a + b
        if op in ("<", ">", "<=", ">="):
            return _NUM_OPS[op](a, b)
    elif ta is str and tb in _NUM_TYPES and op == "*" or tb is str and ta in _NUM_TYPES and op == "*":
        s, n = (a, b) if ta is str else (b, a)
        if isinstance(n, float) and not n.is_integer():
            raise SNRuntimeError("can only repeat a string an integer number of times")
        return s * int(n)
    elif ta is list and tb is list and op == "+":
        return a + b
    elif op == "*" and (ta is list and tb in _NUM_TYPES or tb is list and ta in _NUM_TYPES):
        xs, n = (a, b) if ta is list else (b, a)
        if isinstance(n, float) and not n.is_integer():
            raise SNRuntimeError("can only repeat a list an integer number of times")
        return xs * int(n)
    elif ta is list and tb is list and op in ("<", ">", "<=", ">="):
        return _NUM_OPS[op](a, b)
    raise _unsupported(op, a, b)


def unary(op, v):
    t = type(v)
    if op == "not":
        return not truthy(v)
    if t in _NUM_TYPES:
        return -v if op == "-" else +v
    if t is Tensor:
        return -v if op == "-" else v
    if t is _ag.Var:
        return _ag.neg(v) if op == "-" else v
    raise SNRuntimeError(f"bad operand type for unary '{op}': {type_name(v)}")


def contains(container, item):
    t = type(container)
    if t is str:
        if type(item) is not str:
            raise SNRuntimeError("'in <string>' requires a string on the left side")
        return item in container
    if t is list:
        return any(equals(x, item) for x in container)
    if t is dict:
        try:
            return item in container
        except TypeError:
            return False
    if isinstance(container, NativeObject) and hasattr(container, "sn_contains"):
        return container.sn_contains(item)
    raise SNRuntimeError(f"argument of type '{type_name(container)}' is not iterable")


# -------------------------------------------------------------- object protocol
def to_index(k, what="index"):
    if type(k) is int:
        return k
    if type(k) is float and k.is_integer():
        return int(k)
    raise SNRuntimeError(f"{what} must be an integer, got {to_repr(k)}")


def make_slice(start, stop, step):
    vals = [None if v is None else to_index(v, "slice bound") for v in (start, stop, step)]
    if vals[2] == 0:
        raise SNRuntimeError("slice step cannot be zero")
    return slice(*vals)


def iterate(v):
    t = type(v)
    if t is list:
        return v
    if t is str:
        return list(v)
    if t is dict:
        return list(v.keys())
    if isinstance(v, NativeObject):
        return v.sn_iter()
    raise SNRuntimeError(f"'{type_name(v)}' value is not iterable")


def length(v):
    t = type(v)
    if t is list or t is str or t is dict:
        return len(v)
    if isinstance(v, NativeObject):
        return v.sn_len()
    raise SNRuntimeError(f"object of type '{type_name(v)}' has no length")


def getitem(obj, key):
    t = type(obj)
    if t is list or t is str:
        if type(key) is slice:
            return obj[key]
        i = to_index(key)
        n = len(obj)
        if not -n <= i < n:
            raise SNRuntimeError(f"index {i} out of range (length {n})")
        return obj[i]
    if t is dict:
        try:
            return obj[key]
        except KeyError:
            raise SNRuntimeError(f"key {to_repr(key)} not found in map") from None
        except TypeError:
            raise SNRuntimeError(f"{type_name(key)} cannot be used as a map key") from None
    if isinstance(obj, NativeObject):
        return obj.sn_getitem(key)
    raise SNRuntimeError(f"'{type_name(obj)}' value does not support indexing")


def setitem(obj, key, value):
    t = type(obj)
    if t is list:
        i = to_index(key)
        n = len(obj)
        if not -n <= i < n:
            raise SNRuntimeError(f"index {i} out of range (length {n})")
        obj[i] = value
    elif t is dict:
        try:
            obj[key] = value
        except TypeError:
            raise SNRuntimeError(f"{type_name(key)} cannot be used as a map key") from None
    elif isinstance(obj, NativeObject):
        obj.sn_setitem(key, value)
    else:
        raise SNRuntimeError(f"'{type_name(obj)}' value does not support item assignment")


_TABLES = {list: ("list", LIST_METHODS), str: ("string", STR_METHODS)}


def get_member(obj, name):
    t = type(obj)
    if t is dict:
        if name in obj:
            return obj[name]
        fn = DICT_METHODS.get(name)
        if fn is None:
            raise SNRuntimeError(
                f"map has no key or method '{name}'." + suggest(name, list(map(str, obj)) + list(DICT_METHODS))
            )
        return NativeFn(f"map.{name}", functools.partial(fn, obj))
    if isinstance(obj, NativeObject):
        return obj.sn_get(name)
    entry = _TABLES.get(t)
    if entry is None:
        raise SNRuntimeError(f"'{type_name(obj)}' value has no member '{name}'")
    label, table = entry
    fn = table.get(name)
    if fn is None:
        raise SNRuntimeError(f"{label} has no method '{name}'." + suggest(name, table))
    return NativeFn(f"{label}.{name}", functools.partial(fn, obj))


def set_member(obj, name, value):
    if type(obj) is dict:
        obj[name] = value
    elif isinstance(obj, NativeObject):
        obj.sn_set(name, value)
    else:
        raise SNRuntimeError(f"cannot set member '{name}' on {type_name(obj)}")


def format_value(v, spec):
    """Implements {value:spec} inside f-strings."""
    from .values import to_str
    if spec is None:
        return to_str(v)
    try:
        if isinstance(v, bool):
            v = int(v)
        return format(v, spec)
    except (ValueError, TypeError):
        raise SNRuntimeError(f"cannot format {type_name(v)} with '{spec}'") from None
