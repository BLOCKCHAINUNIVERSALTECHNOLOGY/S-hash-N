"""Built-in methods of lists, strings and maps (``xs.push(1)``, ``s.upper()``...)."""
from .errors import SNRuntimeError
from .values import call_value, to_str, truthy


def _keyfn(key):
    if key is None:
        return None
    return lambda v: call_value(key, v)


def _int(i, what="index"):
    if isinstance(i, float) and i.is_integer():
        return int(i)
    if isinstance(i, bool) or not isinstance(i, int):
        raise SNRuntimeError(f"{what} must be an integer")
    return i


# ---------------------------------------------------------------- list methods
def l_push(xs, x):
    xs.append(x)
    return xs


def l_pop(xs, index=-1):
    if not xs:
        raise SNRuntimeError("pop from an empty list")
    i = _int(index)
    if not -len(xs) <= i < len(xs):
        raise SNRuntimeError(f"pop index {i} out of range")
    return xs.pop(i)


def l_insert(xs, index, x):
    xs.insert(_int(index), x)
    return xs


def l_extend(xs, other):
    xs.extend(other)
    return xs


def l_remove(xs, x):
    for i, v in enumerate(xs):
        if v == x:
            del xs[i]
            return xs
    raise SNRuntimeError("remove(): value not in list")


def l_index(xs, x):
    for i, v in enumerate(xs):
        if v == x:
            return i
    return -1


def l_contains(xs, x):
    return any(v == x for v in xs)


def l_count(xs, x):
    return sum(1 for v in xs if v == x)


def l_reverse(xs):
    xs.reverse()
    return xs


def l_sort(xs, key=None, reverse=False):
    xs.sort(key=_keyfn(key), reverse=truthy(reverse))
    return xs


def l_copy(xs):
    return list(xs)


def l_clear(xs):
    xs.clear()
    return xs


def l_join(xs, sep=""):
    return sep.join(to_str(x) for x in xs)


def l_map(xs, f):
    return [call_value(f, x) for x in xs]


def l_filter(xs, f):
    return [x for x in xs if truthy(call_value(f, x))]


def l_reduce(xs, f, *init):
    if len(init) > 1:
        raise SNRuntimeError("reduce takes at most one initial value")
    it = iter(xs)
    if init:
        acc = init[0]
    else:
        try:
            acc = next(it)
        except StopIteration:
            raise SNRuntimeError("reduce of an empty list with no initial value") from None
    for x in it:
        acc = call_value(f, acc, x)
    return acc


LIST_METHODS = {
    "push": l_push, "append": l_push, "pop": l_pop, "insert": l_insert,
    "extend": l_extend, "remove": l_remove, "index": l_index,
    "contains": l_contains, "count": l_count, "reverse": l_reverse,
    "sort": l_sort, "copy": l_copy, "clear": l_clear, "join": l_join,
    "map": l_map, "filter": l_filter, "reduce": l_reduce,
}


# --------------------------------------------------------------- string methods
STR_METHODS = {
    "upper": lambda s: s.upper(),
    "lower": lambda s: s.lower(),
    "title": lambda s: s.title(),
    "capitalize": lambda s: s.capitalize(),
    "strip": lambda s, chars=None: s.strip(chars),
    "lstrip": lambda s, chars=None: s.lstrip(chars),
    "rstrip": lambda s, chars=None: s.rstrip(chars),
    "split": lambda s, sep=None: s.split(sep),
    "join": lambda s, xs: s.join(to_str(x) for x in xs),
    "replace": lambda s, old, new: s.replace(old, new),
    "starts_with": lambda s, p: s.startswith(p),
    "ends_with": lambda s, p: s.endswith(p),
    "contains": lambda s, sub: sub in s,
    "find": lambda s, sub: s.find(sub),
    "count": lambda s, sub: s.count(sub),
    "chars": lambda s: list(s),
    "lines": lambda s: s.splitlines(),
    "is_numeric": lambda s: _is_numeric(s),
    "pad_left": lambda s, width, fill=" ": s.rjust(_int(width), fill),
    "pad_right": lambda s, width, fill=" ": s.ljust(_int(width), fill),
}


def _is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------- map methods
def d_remove(d, key):
    if key not in d:
        raise SNRuntimeError(f"key {key!r} not found")
    del d[key]
    return d


def d_update(d, other):
    if not isinstance(other, dict):
        raise SNRuntimeError("update() expects a map")
    d.update(other)
    return d


def d_clear(d):
    d.clear()
    return d


DICT_METHODS = {
    "keys": lambda d: list(d.keys()),
    "values": lambda d: list(d.values()),
    "items": lambda d: [[k, v] for k, v in d.items()],
    "has": lambda d, key: key in d,
    "get": lambda d, key, default=None: d.get(key, default),
    "remove": d_remove,
    "update": d_update,
    "copy": lambda d: dict(d),
    "clear": d_clear,
}
