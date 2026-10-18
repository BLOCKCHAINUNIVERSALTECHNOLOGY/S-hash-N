"""Classes and instances."""
from .errors import SNRuntimeError
from .values import Env, Function, NativeObject, call_value, suggest, to_repr, to_str


def bind(fn, owner, inst):
    """Return a copy of method `fn` (defined in class `owner`) bound to `inst`."""
    env = Env(fn.env)
    env.vars["self"] = inst
    env.vars["super"] = SuperProxy(owner.parent, inst) if owner.parent is not None else None
    return Function(f"{owner.name}.{fn.name}", fn.params, fn.body, fn.is_expr, env, fn.file)


class Class(NativeObject):
    sn_type = "class"

    def __init__(self, name, parent, methods):
        self.name = name
        self.parent = parent
        self.methods = methods

    def find(self, name):
        c = self
        while c is not None:
            f = c.methods.get(name)
            if f is not None:
                return f, c
            c = c.parent
        return None, None

    def is_subclass(self, other):
        c = self
        while c is not None:
            if c is other:
                return True
            c = c.parent
        return False

    def instantiate(self, interp, args, kwargs):
        inst = Instance(self)
        init, owner = self.find("init")
        if init is not None:
            interp.call_user(bind(init, owner, inst), args, kwargs)
        elif args or kwargs:
            raise SNRuntimeError(f"{self.name}() takes no arguments (the class has no init method)")
        return inst

    def sn_repr(self):
        return f"<class {self.name}>"


class Instance(NativeObject):
    def __init__(self, cls):
        self.cls = cls
        self.fields = {}
        self._busy = False
        self._bound = {}

    @property
    def sn_type(self):
        return self.cls.name

    def method(self, name):
        m = self._bound.get(name)
        if m is None:
            f, owner = self.cls.find(name)
            if f is None:
                return None
            m = self._bound[name] = bind(f, owner, self)
        return m

    def call_method(self, name, *args):
        m = self.method(name)
        if m is None:
            raise SNRuntimeError(f"{self.cls.name} has no '{name}' method")
        return call_value(m, *args)

    def sn_get(self, name):
        if name in self.fields:
            return self.fields[name]
        m = self.method(name)
        if m is not None:
            return m
        known = list(self.fields)
        c = self.cls
        while c is not None:
            known.extend(c.methods)
            c = c.parent
        raise SNRuntimeError(
            f"{self.cls.name} object has no field or method '{name}'." + suggest(name, known)
        )

    def sn_set(self, name, value):
        self.fields[name] = value

    def sn_repr(self):
        m = self.method("str")
        if m is not None:
            return to_str(call_value(m))
        if self._busy:
            return f"{self.cls.name}(...)"
        self._busy = True
        try:
            body = ", ".join(f"{k}={to_repr(v)}" for k, v in self.fields.items())
        finally:
            self._busy = False
        return f"{self.cls.name}({body})"

    def sn_len(self):
        if self.method("len") is None:
            return super().sn_len()
        return self.call_method("len")

    def sn_getitem(self, key):
        if self.method("getitem") is None:
            return super().sn_getitem(key)
        return self.call_method("getitem", key)

    def sn_setitem(self, key, value):
        if self.method("setitem") is None:
            return super().sn_setitem(key, value)
        self.call_method("setitem", key, value)

    def sn_iter(self):
        if self.method("iter") is None:
            return super().sn_iter()
        r = self.call_method("iter")
        if not isinstance(r, list):
            raise SNRuntimeError("iter() must return a list")
        return r

    def sn_contains(self, item):
        if self.method("contains") is None:
            raise SNRuntimeError(f"{self.cls.name} does not support 'in' (define a contains method)")
        return bool(self.call_method("contains", item))

    def sn_truthy(self):
        if self.method("bool") is None:
            return True
        from .values import truthy
        return truthy(self.call_method("bool"))


class SuperProxy(NativeObject):
    sn_type = "super"

    def __init__(self, parent, inst):
        self.parent = parent
        self.inst = inst

    def sn_get(self, name):
        f, owner = self.parent.find(name)
        if f is None:
            raise SNRuntimeError(f"parent class {self.parent.name} has no method '{name}'")
        return bind(f, owner, self.inst)

    def sn_repr(self):
        return f"<super of {self.parent.name}>"


_BINARY_METHODS = {
    "+": "add", "-": "sub", "*": "mul", "/": "div", "//": "floordiv", "%": "mod",
    "**": "pow", "@": "matmul", "==": "eq", "<": "lt", ">": "gt", "<=": "le", ">=": "ge",
}


def instance_binop(op, a, b):
    """Operator overloading: a + b calls a.add(b), falling back to b.radd(a)."""
    if op == "!=":
        return not instance_binop("==", a, b)
    if op == "in":
        return b.sn_contains(a) if isinstance(b, Instance) else _no_op(op, a, b)
    if op == "not in":
        return not instance_binop("in", a, b)
    name = _BINARY_METHODS.get(op)
    if name is None:
        _no_op(op, a, b)
    if isinstance(a, Instance) and a.method(name) is not None:
        return a.call_method(name, b)
    if isinstance(b, Instance) and b.method("r" + name) is not None:
        return b.call_method("r" + name, a)
    if op == "==":
        return a is b
    return _no_op(op, a, b)


def _no_op(op, a, b):
    from .values import type_name
    raise SNRuntimeError(
        f"unsupported operand types for '{op}': {type_name(a)} and {type_name(b)} "
        f"(define a '{_BINARY_METHODS.get(op, op)}' method to support it)"
    )
