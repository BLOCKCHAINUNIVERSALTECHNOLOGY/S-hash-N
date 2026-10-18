"""Tree-walking interpreter for S#N."""
import operator
import os

from . import ast as A
from . import ops
from .errors import SNError, SNRuntimeError, SNThrow
from .lexer import Lexer
from .parser import Parser, parse
from .objects import Class
from .stdlib import BUILTINS
from .values import (
    HOOK, Env, BreakSignal, ContinueSignal, Function, NativeFn, NativeObject, ReturnSignal,
    suggest, to_repr, to_str, truthy, type_name,
)


_BRK, _CNT, _RET = 1, 2, 3          # statement results: None means "carry on"
_DECLARES = (A.Let, A.FnDef, A.ClassDef, A.Import)
_CAPTURES = (A.Lambda, A.FnDef, A.ClassDef)
_FAST = {"+": operator.add, "-": operator.sub, "*": operator.mul, "<": operator.lt,
         ">": operator.gt, "<=": operator.le, ">=": operator.ge,
         "==": operator.eq, "!=": operator.ne}


def _walk(x):
    if isinstance(x, A.Node):
        yield x
        for f in type(x).__slots__:
            yield from _walk(getattr(x, f, None))
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _walk(y)


def _captures(*nodes):
    """True if evaluating these nodes may create a closure over the current scope."""
    return any(isinstance(n, _CAPTURES) for x in nodes for n in _walk(x))


def _declares(stmts):
    return any(isinstance(s, _DECLARES) for s in stmts)


def _literal_equal(p, v):
    if p is None or isinstance(p, bool) or isinstance(v, bool):
        return p is v or (isinstance(p, bool) and isinstance(v, bool) and p == v)
    if isinstance(p, str):
        return type(v) is str and p == v
    return type(v) in (int, float) and p == v


class Interpreter:
    MAX_DEPTH = 900

    def __init__(self, filename="<main>", argv=None):
        self.globals = Env()
        self.globals.vars.update(BUILTINS)
        self.globals.vars["args"] = list(argv or [])
        self.cur_file = filename
        self.sources = {}
        self.modules = {}
        self.depth = 0
        self.stack = []
        self.cur_line = 0
        self.last = None
        HOOK["interp"] = self
        self.retval = None
        self._cache = {}

    # ------------------------------------------------------------- entry points
    def run(self, source, filename="<main>", repl=False):
        """Run source text in the global scope. In REPL mode returns
        ``(has_value, value)`` for a trailing expression statement."""
        self.sources[filename] = source
        prev = self.cur_file
        self.cur_file = filename
        try:
            prog = parse(source, filename)
            self.last = None
            try:
                r = self._cb(prog)(self.globals)
                if r == _RET:
                    raise ReturnSignal()
                if r:
                    raise BreakSignal()
            except ReturnSignal:
                raise SNRuntimeError("'return' outside of a function", filename=filename) from None
            except (BreakSignal, ContinueSignal):
                raise SNRuntimeError("'break'/'continue' outside of a loop", filename=filename) from None
            except RecursionError:
                raise SNRuntimeError("maximum recursion depth exceeded", filename=filename) from None
            if repl:
                return (bool(prog) and isinstance(prog[-1], A.ExprStmt), self.last)
            return None
        finally:
            self.cur_file = prev

    def run_file(self, path):
        with open(path, encoding="utf-8") as f:
            source = f.read()
        return self.run(source, path)

    # ------------------------------------------------------------------ helpers
    def _annotate(self, e, line):
        if e.line is None:
            e.line = line
            e.filename = e.filename or self.cur_file
            e.trace = list(reversed(self.stack))[:50]

    def eval(self, n, env):
        return self._cx(n)(env)

    def _cached(self, key_obj, build):
        ent = self._cache.get(id(key_obj))
        if ent is None:
            ent = (key_obj, build(key_obj))     # keep key_obj alive so its id stays unique
            self._cache[id(key_obj)] = ent
        return ent[1]

    def _cx(self, n):
        ent = self._cache.get(id(n))
        if ent is not None:
            return ent[1]
        return self._cached(n, self.c_expr)

    def _cb(self, stmts):
        ent = self._cache.get(id(stmts))
        if ent is not None:
            return ent[1]
        return self._cached(stmts, self.c_block)

    def call(self, f, args, kwargs=None):
        t = type(f)
        if t is Function:
            return self.call_user(f, args, kwargs)
        if t is NativeFn:
            return f.call(args, kwargs)
        if t is Class:
            return f.instantiate(self, args, kwargs)
        raise SNRuntimeError(f"'{type_name(f)}' value is not callable")

    def _call_at(self, f, args, kwargs, line):
        self.cur_line = line
        try:
            return self.call(f, args, kwargs)
        except SNError as e:
            if e.line is None:
                self._annotate(e, line)
            raise

    def call_user(self, fn, args, kwargs):
        if self.depth >= self.MAX_DEPTH:
            raise SNRuntimeError(f"maximum recursion depth exceeded in {fn.name}()")
        env = Env(fn.env)
        vars_ = env.vars
        params = fn.params
        na = len(args)
        if not kwargs and na == len(params):
            for p, a in zip(params, args):
                vars_[p[0]] = a
        else:
            if na > len(params):
                raise SNRuntimeError(
                    f"{fn.name}() takes at most {len(params)} argument(s) but {na} were given"
                )
            if kwargs:
                known = {p[0] for p in params}
                for k in kwargs:
                    if k not in known:
                        raise SNRuntimeError(f"{fn.name}() got an unexpected argument '{k}'")
            for i, (pname, default) in enumerate(params):
                if i < na:
                    if kwargs and pname in kwargs:
                        raise SNRuntimeError(f"{fn.name}() got multiple values for argument '{pname}'")
                    vars_[pname] = args[i]
                elif kwargs and pname in kwargs:
                    vars_[pname] = kwargs[pname]
                elif default is not None:
                    vars_[pname] = self.eval(default, env)
                else:
                    raise SNRuntimeError(f"{fn.name}() missing required argument '{pname}'")
        self.depth += 1
        self.stack.append((fn.name, self.cur_line))
        saved = self.cur_file
        self.cur_file = fn.file
        try:
            if fn.is_expr:
                return self._cx(fn.body)(env)
            try:
                r = self._cb(fn.body)(env)
            except ReturnSignal as sig:
                return sig.value
            except (BreakSignal, ContinueSignal):
                raise SNRuntimeError("'break'/'continue' outside of a loop") from None
            if r == _RET:
                v = self.retval
                self.retval = None
                return v
            if r:
                raise SNRuntimeError("'break'/'continue' outside of a loop")
            return None
        finally:
            self.depth -= 1
            self.stack.pop()
            self.cur_file = saved

    def _unpack(self, names, item, vars_):
        values = ops.iterate(item)
        if len(values) != len(names):
            raise SNRuntimeError(f"cannot unpack {len(values)} value(s) into {len(names)} variable(s)")
        for name, v in zip(names, values):
            vars_[name] = v

    def _resolve_import(self, path):
        cands = [path] if path.endswith(".sn") else [path + ".sn", path]
        dirs = []
        cf = self.cur_file
        dirs.append(os.path.dirname(os.path.abspath(cf)) if cf and not cf.startswith("<") else os.getcwd())
        dirs.append(os.getcwd())
        dirs.extend(p for p in os.environ.get("SN_PATH", "").split(os.pathsep) if p)
        for d in dirs:
            for c in cands:
                full = c if os.path.isabs(c) else os.path.join(d, c)
                if os.path.isfile(full):
                    return os.path.abspath(full)
        raise SNRuntimeError(f"cannot find module '{path}'")

    # ====================================================================
    # Compilation: every AST node becomes a Python closure taking an Env.
    # Statement closures return None, or _BRK / _CNT / _RET to unwind.
    # ====================================================================
    def c_block(self, stmts):
        pairs = [(self.c_stmt(s), s.line) for s in stmts]
        annotate = self._annotate
        if not pairs:
            return lambda env: None
        if len(pairs) == 1:
            f0, l0 = pairs[0]

            def run1(env):
                try:
                    return f0(env)
                except SNError as e:
                    if e.line is None:
                        annotate(e, l0)
                    raise
            return run1

        def run(env):
            for f, line in pairs:
                try:
                    r = f(env)
                except SNError as e:
                    if e.line is None:
                        annotate(e, line)
                    raise
                if r:
                    return r
        return run

    def c_stmt(self, n):
        return getattr(self, "s_" + type(n).__name__)(n)

    def c_expr(self, n):
        return getattr(self, "e_" + type(n).__name__)(n)

    # ------------------------------------------------------------- statements
    def s_ExprStmt(self, n):
        e = self._cx(n.expr)
        interp = self

        def f(env):
            interp.last = e(env)
        return f

    def s_Let(self, n):
        names = n.names
        val = self._cx(n.value) if n.value is not None else None
        if len(names) == 1:
            name = names[0]

            def f(env):
                v = val(env) if val is not None else None
                if type(v) is Function and v.name == "<lambda>":
                    v.name = name
                env.vars[name] = v
        else:
            def f(env):
                v = val(env) if val is not None else None
                self._unpack(names, v, env.vars)
        return f

    def s_Assign(self, n):
        t, op = n.target, n.op
        val = self._cx(n.value)
        bop = op[:-1] if op != "=" else None
        binop = ops.binop
        tt = type(t)
        if tt is A.Ident:
            name = t.name
            if bop is None:
                def f(env):
                    v = val(env)
                    e = env
                    while e is not None:
                        vs = e.vars
                        if name in vs:
                            vs[name] = v
                            return
                        e = e.parent
                    env.assign(name, v)
            else:
                cur = self._cx(t)

                def f(env):
                    v = val(env)
                    a = cur(env)
                    if (type(a) is int or type(a) is float) and (type(v) is int or type(v) is float) \
                            and bop in _FAST:
                        v = _FAST[bop](a, v)
                    else:
                        v = binop(bop, a, v)
                    env.assign(name, v)
        elif tt is A.Index:
            obj = self._cx(t.obj)
            keys = [self._ckey(k) for k in t.keys]

            def f(env):
                v = val(env)
                o = obj(env)
                ks = [k(env) for k in keys]
                key = ks[0] if len(ks) == 1 else tuple(ks)
                if bop is not None:
                    v = binop(bop, ops.getitem(o, key), v)
                ops.setitem(o, key, v)
        else:
            obj = self._cx(t.obj)
            name = t.name

            def f(env):
                v = val(env)
                o = obj(env)
                if bop is not None:
                    v = binop(bop, ops.get_member(o, name), v)
                ops.set_member(o, name, v)
        return f

    def _scoped(self, body):
        """Closure running `body` in a fresh scope only when it declares names."""
        run = self._cb(body)
        if _declares(body):
            return lambda env: run(Env(env))
        return run

    def s_If(self, n):
        conds = [self._cx(c) for c, _ in n.branches]
        bodies = [self._scoped(b) for _, b in n.branches]
        orelse = self._scoped(n.orelse) if n.orelse is not None else None
        if len(conds) == 1:
            c0, b0 = conds[0], bodies[0]

            def f(env):
                v = c0(env)
                if v is True or (v is not False and truthy(v)):
                    return b0(env)
                if orelse is not None:
                    return orelse(env)
            return f
        pairs = list(zip(conds, bodies))

        def f(env):
            for c, b in pairs:
                v = c(env)
                if v is True or (v is not False and truthy(v)):
                    return b(env)
            if orelse is not None:
                return orelse(env)
        return f

    def s_While(self, n):
        cond = self._cx(n.cond)
        body = self._cb(n.body)
        fresh = _declares(n.body)

        def f(env):
            while True:
                v = cond(env)
                if not (v is True or (v is not False and truthy(v))):
                    return
                try:
                    r = body(Env(env) if fresh else env)
                except BreakSignal:
                    return
                except ContinueSignal:
                    continue
                if r:
                    if r == _BRK:
                        return
                    if r == _CNT:
                        continue
                    return r
        return f

    def s_For(self, n):
        it = self._cx(n.iterable)
        names = n.names
        single = len(names) == 1
        name0 = names[0]
        body = self._cb(n.body)
        reuse = not _declares(n.body) and not _captures(n.body)

        def f(env):
            seq = ops.iterate(it(env))
            scope = Env(env) if reuse else None
            for item in seq:
                if not reuse:
                    scope = Env(env)
                if single:
                    scope.vars[name0] = item
                else:
                    self._unpack(names, item, scope.vars)
                try:
                    r = body(scope)
                except BreakSignal:
                    return
                except ContinueSignal:
                    continue
                if r:
                    if r == _BRK:
                        return
                    if r == _CNT:
                        continue
                    return r
        return f

    def s_Return(self, n):
        val = self._cx(n.value) if n.value is not None else None
        interp = self

        def f(env):
            interp.retval = val(env) if val is not None else None
            return _RET
        return f

    def s_Break(self, n):
        return lambda env: _BRK

    def s_Continue(self, n):
        return lambda env: _CNT

    def s_FnDef(self, n):
        interp = self

        def f(env):
            env.vars[n.name] = Function(n.name, n.params, n.body, False, env, interp.cur_file)
        return f

    def s_ClassDef(self, n):
        parent_fn = self._cx(n.parent) if n.parent is not None else None
        interp = self

        def f(env):
            parent = None
            if parent_fn is not None:
                parent = parent_fn(env)
                if not isinstance(parent, Class):
                    raise SNRuntimeError(
                        f"class {n.name} can only extend another class, not {type_name(parent)}")
            methods = {m.name: Function(m.name, m.params, m.body, False, env, interp.cur_file)
                       for m in n.methods}
            env.vars[n.name] = Class(n.name, parent, methods)
        return f

    def s_Try(self, n):
        body = self._cb(n.body)
        handler = self._cb(n.handler)
        name = n.name

        def f(env):
            try:
                return body(Env(env))
            except (SNError, RecursionError) as e:
                if isinstance(e, SNThrow):
                    value = e.value
                elif isinstance(e, SNError):
                    value = e.message
                else:
                    value = "maximum recursion depth exceeded"
                scope = Env(env)
                if name:
                    scope.vars[name] = value
                return handler(scope)
        return f

    def s_Throw(self, n):
        val = self._cx(n.value)

        def f(env):
            value = val(env)
            raise SNThrow(value, to_str(value))
        return f

    def s_Import(self, n):
        return lambda env: self._import(n, env)

    def _import(self, n, env):
        path = self._resolve_import(n.path)
        mod = self.modules.get(path)
        if mod is None:
            with open(path, encoding="utf-8") as f:
                src = f.read()
            self.sources[path] = src
            mod = Env(self.globals)
            self.modules[path] = mod
            saved = self.cur_file
            self.cur_file = path
            try:
                r = self._cb(parse(src, path))(mod)
                if r:
                    raise ReturnSignal()
            except (ReturnSignal, BreakSignal, ContinueSignal):
                self.modules.pop(path, None)
                raise SNRuntimeError(f"stray return/break/continue at top level of '{path}'") from None
            except BaseException:
                self.modules.pop(path, None)
                raise
            finally:
                self.cur_file = saved
        exports = {k: v for k, v in mod.vars.items() if not k.startswith("_")}
        if n.alias:
            env.vars[n.alias] = exports
        else:
            env.vars.update(exports)

    # ------------------------------------------------------------- expressions
    def e_Num(self, n):
        v = n.value
        return lambda env: v

    e_Str = e_Num
    e_Bool = e_Num

    def e_Nil(self, n):
        return lambda env: None

    def e_FStr(self, n):
        parts = [p if type(p) is str else (self._cx(p[0]), p[1]) for p in n.parts]
        fmt = ops.format_value

        def f(env):
            return "".join(p if type(p) is str else fmt(p[0](env), p[1]) for p in parts)
        return f

    def e_Ident(self, n):
        name = n.name

        def f(env):
            e = env
            while e is not None:
                vs = e.vars
                if name in vs:
                    return vs[name]
                e = e.parent
            raise SNRuntimeError(f"undefined variable '{name}'." + suggest(name, env.names()))
        return f

    def e_ListLit(self, n):
        items = [self._cx(x) for x in n.items]
        return lambda env: [x(env) for x in items]

    def e_MapLit(self, n):
        entries = [(self._cx(k), self._cx(v)) for k, v in n.entries]

        def f(env):
            out = {}
            for k, v in entries:
                key = k(env)
                if type(key) in (list, dict) or isinstance(key, NativeObject):
                    raise SNRuntimeError(f"{type_name(key)} cannot be used as a map key")
                out[key] = v(env)
            return out
        return f

    def e_Unary(self, n):
        operand = self._cx(n.operand)
        op = n.op
        unary = ops.unary
        if op == "-":
            def f(env):
                v = operand(env)
                if type(v) is int or type(v) is float:
                    return -v
                return unary(op, v)
        else:
            def f(env):
                return unary(op, operand(env))
        return f

    def e_Binary(self, n):
        op = n.op
        left = self._cx(n.left)
        binop = ops.binop
        fast = _FAST.get(op)
        if fast is None:
            right = self._cx(n.right)
            return lambda env: binop(op, left(env), right(env))
        if isinstance(n.right, A.Num) and type(n.right.value) in (int, float):
            k = n.right.value

            def f(env):
                a = left(env)
                ta = type(a)
                if ta is int or ta is float:
                    return fast(a, k)
                return binop(op, a, k)
            return f
        right = self._cx(n.right)

        def f(env):
            a = left(env)
            b = right(env)
            ta = type(a)
            tb = type(b)
            if (ta is int or ta is float) and (tb is int or tb is float):
                return fast(a, b)
            return binop(op, a, b)
        return f

    def e_Ternary(self, n):
        cond, then, orelse = self._cx(n.cond), self._cx(n.then), self._cx(n.orelse)

        def f(env):
            v = cond(env)
            return then(env) if (v is True or (v is not False and truthy(v))) else orelse(env)
        return f

    def e_Logical(self, n):
        left, right = self._cx(n.left), self._cx(n.right)
        if n.op == "or":
            def f(env):
                v = left(env)
                return v if truthy(v) else right(env)
        else:
            def f(env):
                v = left(env)
                return right(env) if truthy(v) else v
        return f

    def _call_parts(self, c):
        callee = self._cx(c.callee)
        args = [self._cx(a) for a in c.args]
        kwargs = [(k, self._cx(v)) for k, v in c.kwargs] if c.kwargs else None
        return callee, args, kwargs

    def e_Call(self, n):
        callee, args, kwargs = self._call_parts(n)
        line = n.line
        interp = self
        annotate = self._annotate

        def f(env):
            fn = callee(env)
            a = [x(env) for x in args]
            kw = {k: v(env) for k, v in kwargs} if kwargs else None
            interp.cur_line = line
            try:
                if type(fn) is Function:
                    return interp.call_user(fn, a, kw)
                return interp.call(fn, a, kw)
            except SNError as e:
                if e.line is None:
                    annotate(e, line)
                raise
        return f

    def e_Pipe(self, n):
        left = self._cx(n.left)
        r = n.right
        interp = self
        if type(r) is A.Call:
            callee, args, kwargs = self._call_parts(r)
            line = r.line

            def f(env):
                lv = left(env)
                fn = callee(env)
                a = [lv] + [x(env) for x in args]
                kw = {k: v(env) for k, v in kwargs} if kwargs else None
                return interp._call_at(fn, a, kw, line)
        else:
            rf = self._cx(r)
            line = n.line

            def f(env):
                lv = left(env)
                return interp._call_at(rf(env), [lv], None, line)
        return f

    def _ckey(self, k):
        if type(k) is A.Slice:
            parts = [None if x is None else self._cx(x) for x in (k.start, k.stop, k.step)]
            return lambda env: ops.make_slice(*(None if p is None else p(env) for p in parts))
        return self._cx(k)

    def e_Index(self, n):
        obj = self._cx(n.obj)
        keys = [self._ckey(k) for k in n.keys]
        getitem = ops.getitem
        if len(keys) == 1:
            k0 = keys[0]
            return lambda env: getitem(obj(env), k0(env))
        return lambda env: getitem(obj(env), tuple(k(env) for k in keys))

    def e_Member(self, n):
        obj = self._cx(n.obj)
        name = n.name
        get_member = ops.get_member
        return lambda env: get_member(obj(env), name)

    def e_Lambda(self, n):
        interp = self
        return lambda env: Function("<lambda>", n.params, n.body, n.is_expr, env, interp.cur_file)

    def e_ListComp(self, n):
        it = self._cx(n.iterable)
        expr = self._cx(n.expr)
        cond = self._cx(n.cond) if n.cond is not None else None
        names = n.names
        single = len(names) == 1
        name0 = names[0]
        reuse = not _captures(n.expr, n.cond)

        def f(env):
            out = []
            scope = Env(env) if reuse else None
            for item in ops.iterate(it(env)):
                if not reuse:
                    scope = Env(env)
                if single:
                    scope.vars[name0] = item
                else:
                    self._unpack(names, item, scope.vars)
                if cond is not None and not truthy(cond(scope)):
                    continue
                out.append(expr(scope))
            return out
        return f

    # ------------------------------------------------------------------ match
    def e_Match(self, n):
        subject = self._cx(n.subject)
        arms = []
        for pat, guard, body, is_block in n.arms:
            m = self._cpat(pat)
            g = self._cx(guard) if guard is not None else None
            if is_block:
                last_expr = isinstance(body[-1], A.ExprStmt) if body else False
                init = self._cb(body[:-1] if last_expr else body)
                tail = self._cx(body[-1].expr) if last_expr else None
                b = (init, tail)
            else:
                b = self._cx(body)
            arms.append((m, g, b, is_block))
        interp = self

        def f(env):
            value = subject(env)
            for m, g, b, is_block in arms:
                scope = Env(env)
                if not m(value, scope):
                    continue
                if g is not None and not truthy(g(scope)):
                    continue
                if not is_block:
                    return b(scope)
                init, tail = b
                r = init(scope)
                if r:
                    if r == _RET:
                        raise ReturnSignal(interp.retval)
                    raise (BreakSignal() if r == _BRK else ContinueSignal())
                return tail(scope) if tail is not None else None
            raise SNRuntimeError(f"no match arm matched {to_repr(value)}; add a '_ => ...' arm")
        return f

    def _cpat(self, p):
        t = type(p)
        if t is A.PWild:
            return lambda v, s: True
        if t is A.PBind:
            name = p.name

            def bind(v, s):
                s.vars[name] = v
                return True
            return bind
        if t is A.PLit:
            lit = p.value
            return lambda v, s: _literal_equal(lit, v)
        if t is A.POr:
            alts = [self._cpat(a) for a in p.alts]
            return lambda v, s: any(a(v, s) for a in alts)
        if t is A.PList:
            subs = [self._cpat(x) for x in p.items]
            rest = p.rest
            n_items = len(subs)

            def mlist(v, s):
                if type(v) is not list:
                    return False
                if len(v) < n_items or (rest is None and len(v) != n_items):
                    return False
                for sub, item in zip(subs, v):
                    if not sub(item, s):
                        return False
                if rest is not None and rest != "_":
                    s.vars[rest] = v[n_items:]
                return True
            return mlist
        if t is A.PMap:
            entries = [(k, self._cpat(sub)) for k, sub in p.entries]

            def mmap(v, s):
                if type(v) is not dict:
                    return False
                for key, sub in entries:
                    if key not in v or not sub(v[key], s):
                        return False
                return True
            return mmap
        raise SNRuntimeError("invalid pattern")
