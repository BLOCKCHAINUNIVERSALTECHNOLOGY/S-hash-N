"""AST node definitions for S#N."""


class Node:
    __slots__ = ("line",)


def _node(name, *fields):
    def __init__(self, *args, line=0):
        if len(args) != len(fields):
            raise TypeError(f"{name} expects {len(fields)} fields, got {len(args)}")
        for f, a in zip(fields, args):
            setattr(self, f, a)
        self.line = line

    def __repr__(self):
        return f"{name}({', '.join(repr(getattr(self, f)) for f in fields)})"

    return type(name, (Node,), {"__slots__": fields, "__init__": __init__, "__repr__": __repr__})


# ---- expressions --------------------------------------------------------
Num = _node("Num", "value")
Str = _node("Str", "value")
FStr = _node("FStr", "parts")            # parts: str | (expr, format_spec)
Bool = _node("Bool", "value")
Nil = _node("Nil")
Ident = _node("Ident", "name")
ListLit = _node("ListLit", "items")
MapLit = _node("MapLit", "entries")      # [(key_expr, value_expr)]
Unary = _node("Unary", "op", "operand")
Binary = _node("Binary", "op", "left", "right")
Logical = _node("Logical", "op", "left", "right")
Call = _node("Call", "callee", "args", "kwargs")   # kwargs: [(name, expr)]
Index = _node("Index", "obj", "keys")
Slice = _node("Slice", "start", "stop", "step")
Member = _node("Member", "obj", "name")
Lambda = _node("Lambda", "params", "body", "is_expr")
Pipe = _node("Pipe", "left", "right")
Ternary = _node("Ternary", "then", "cond", "orelse")
ListComp = _node("ListComp", "expr", "names", "iterable", "cond")
Match = _node("Match", "subject", "arms")   # arms: [(pattern, guard, body, is_block)]

# ---- patterns (used by match) -------------------------------------------
PLit = _node("PLit", "value")
PWild = _node("PWild")
PBind = _node("PBind", "name")
PList = _node("PList", "items", "rest")     # rest: None | name ("_" = anonymous)
PMap = _node("PMap", "entries")             # [(key, pattern)]
POr = _node("POr", "alts")

# ---- statements ---------------------------------------------------------
Let = _node("Let", "names", "value")
Assign = _node("Assign", "target", "op", "value")
ExprStmt = _node("ExprStmt", "expr")
If = _node("If", "branches", "orelse")   # branches: [(cond, [stmts])]
While = _node("While", "cond", "body")
For = _node("For", "names", "iterable", "body")
Return = _node("Return", "value")
Break = _node("Break")
Continue = _node("Continue")
FnDef = _node("FnDef", "name", "params", "body")
Try = _node("Try", "body", "name", "handler")
Throw = _node("Throw", "value")
Import = _node("Import", "path", "alias")
ClassDef = _node("ClassDef", "name", "parent", "methods")
