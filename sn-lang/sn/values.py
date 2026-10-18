"""Runtime value types and helpers shared by every S#N module.

S#N values are represented by plain Python objects wherever possible:

    nil -> None        bool -> bool        int / float -> int / float
    string -> str      list -> list        map -> dict

Everything else (functions, tensors, data frames, models) is one of the
classes defined in this package.
"""
import difflib
import inspect
import math
import os

from .errors import SNError, SNRuntimeError

DEBUG = bool(os.environ.get("SN_DEBUG"))
HOOK = {"interp": None}


def call_value(fn, *args, **kwargs):
    """Call an S#N function value from Python code."""
    return HOOK["interp"].call(fn, list(args), kwargs or None)


def suggest(name, candidates):
    match = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return f" Did you mean '{match[0]}'?" if match else ""


# --------------------------------------------------------------------------
# Control-flow signals (BaseException so natives can never swallow them)
# --------------------------------------------------------------------------
class ReturnSignal(BaseException):
    def __init__(self, value=None):
        self.value = value


class BreakSignal(BaseException):
    pass


class ContinueSignal(BaseException):
    pass


class Env:
    """A scope: variables plus a link to the enclosing scope."""

    __slots__ = ("vars", "parent")

    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def names(self):
        out, e = [], self
        while e is not None:
            out.extend(e.vars)
            e = e.parent
        return out

    def assign(self, name, value):
        e = self
        while e is not None:
            if name in e.vars:
                e.vars[name] = value
                return
            e = e.parent
        raise SNRuntimeError(
            f"cannot assign to undefined variable '{name}'; declare it first with 'let'."
            + suggest(name, self.names())
        )


# --------------------------------------------------------------------------
# Callables
# --------------------------------------------------------------------------
class Function:
    """A function written in S#N."""

    sn_type = "function"
    __slots__ = ("name", "params", "body", "is_expr", "env", "file")

    def __init__(self, name, params, body, is_expr, env, file):
        self.name = name
        self.params = params
        self.body = body
        self.is_expr = is_expr
        self.env = env
        self.file = file


class NativeFn:
    """A function implemented in Python and exposed to S#N."""

    sn_type = "function"
    __slots__ = ("name", "fn", "_sig")

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn
        self._sig = None

    def call(self, args, kwargs=None):
        kwargs = kwargs or {}
        sig = self._sig
        if sig is None:
            try:
                sig = inspect.signature(self.fn)
            except (TypeError, ValueError):
                sig = False
            self._sig = sig
        if sig:
            try:
                sig.bind(*args, **kwargs)
            except TypeError as e:
                raise SNRuntimeError(f"{self.name}(): {e}") from None
        try:
            result = self.fn(*args, **kwargs)
        except SNError:
            raise
        except Exception as e:  # noqa: BLE001 - turn Python errors into S#N errors
            if DEBUG:
                raise
            raise SNRuntimeError(f"{self.name}(): {e}") from None
        return list(result) if type(result) is tuple else result


class NativeObject:
    """Base class for runtime objects implemented in Python.

    Subclasses expose members to S#N code by naming Python attributes:
    ``m_<name>`` for methods and ``p_<name>`` (a ``property``) for read-only
    properties.
    """

    __slots__ = ()
    sn_type = "object"

    def sn_get(self, name):
        attr = getattr(type(self), "p_" + name, None)
        if isinstance(attr, property):
            return getattr(self, "p_" + name)
        method = getattr(self, "m_" + name, None)
        if method is not None:
            return NativeFn(f"{self.sn_type}.{name}", method)
        raise SNRuntimeError(
            f"{self.sn_type} has no member '{name}'." + suggest(name, self.sn_members())
        )

    def sn_members(self):
        return [n[2:] for n in dir(type(self)) if n[:2] in ("p_", "m_")]

    def sn_set(self, name, value):
        raise SNRuntimeError(f"cannot assign to '{name}' on {self.sn_type}")

    def sn_getitem(self, key):
        raise SNRuntimeError(f"{self.sn_type} does not support indexing")

    def sn_setitem(self, key, value):
        raise SNRuntimeError(f"{self.sn_type} does not support item assignment")

    def sn_iter(self):
        raise SNRuntimeError(f"'{self.sn_type}' is not iterable")

    def sn_len(self):
        raise SNRuntimeError(f"{self.sn_type} has no length")

    def sn_truthy(self):
        return True

    def sn_repr(self):
        return f"<{self.sn_type}>"


# --------------------------------------------------------------------------
# Type names, truthiness and text conversion
# --------------------------------------------------------------------------
def type_name(v):
    if v is None:
        return "nil"
    t = type(v)
    if t is bool:
        return "bool"
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is str:
        return "string"
    if t is list:
        return "list"
    if t is dict:
        return "map"
    return getattr(v, "sn_type", t.__name__)


def truthy(v):
    if v is None or v is False:
        return False
    if v is True:
        return True
    t = type(v)
    if t is int or t is float:
        return v != 0
    if t is str or t is list or t is dict:
        return len(v) > 0
    if isinstance(v, NativeObject):
        return v.sn_truthy()
    return True


def fmt_number(v, digits=10):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if v != v:
        return "nan"
    if v == math.inf:
        return "inf"
    if v == -math.inf:
        return "-inf"
    s = "%.*g" % (digits, v)
    if "e" not in s and "." not in s:
        s += ".0"
    return s


def _quote(s):
    return (
        '"'
        + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        + '"'
    )


def to_repr(v, depth=0):
    """Debug-style text: strings are quoted."""
    if v is None:
        return "nil"
    t = type(v)
    if t is bool or t is int or t is float:
        return fmt_number(v)
    if t is str:
        return _quote(v)
    if depth > 8:
        return "..."
    if t is list:
        if len(v) > 200:
            body = ", ".join(to_repr(x, depth + 1) for x in v[:100])
            return f"[{body}, ... ({len(v) - 100} more)]"
        return "[" + ", ".join(to_repr(x, depth + 1) for x in v) + "]"
    if t is dict:
        return (
            "{"
            + ", ".join(f"{to_repr(k, depth + 1)}: {to_repr(x, depth + 1)}" for k, x in v.items())
            + "}"
        )
    if t is Function:
        return f"<fn {v.name}>"
    if t is NativeFn:
        return f"<builtin {v.name}>"
    if isinstance(v, NativeObject):
        return v.sn_repr()
    return str(v)


def to_str(v):
    """User-facing text: strings are shown as-is."""
    return v if type(v) is str else to_repr(v)
