"""Recursive-descent parser for S#N."""
from . import ast as A
from .errors import SNSyntaxError
from .lexer import Lexer

COMPARE = {"==", "!=", "<", ">", "<=", ">="}
ASSIGN_OPS = {"=", "+=", "-=", "*=", "/="}


class Parser:
    def __init__(self, tokens, filename="<string>"):
        self.toks = tokens
        self.pos = 0
        self.arm_arrow = -1
        self.filename = filename

    # ------------------------------------------------------------------ utils
    @property
    def tok(self):
        return self.toks[self.pos]

    def peek(self, k=1):
        return self.toks[min(self.pos + k, len(self.toks) - 1)]

    def advance(self):
        t = self.toks[self.pos]
        if t.type != "EOF":
            self.pos += 1
        return t

    def describe(self, tok=None):
        t = tok or self.tok
        if t.type == "NEWLINE":
            return "end of line"
        if t.type == "EOF":
            return "end of file"
        if t.type == "STR":
            return "a string"
        if t.type == "FSTR":
            return "an f-string"
        return repr(t.value)

    def error(self, msg, tok=None):
        t = tok or self.tok
        raise SNSyntaxError(msg, t.line, t.col, self.filename)

    def is_op(self, v, tok=None):
        t = tok or self.tok
        return t.type == "OP" and t.value == v

    def is_kw(self, v, tok=None):
        t = tok or self.tok
        return t.type == "KW" and t.value == v

    def accept_op(self, v):
        if self.is_op(v):
            self.advance()
            return True
        return False

    def expect_op(self, v, ctx=""):
        if not self.is_op(v):
            self.error(f"expected '{v}'{ctx}, found {self.describe()}")
        return self.advance()

    def expect_kw(self, v):
        if not self.is_kw(v):
            self.error(f"expected '{v}', found {self.describe()}")
        return self.advance()

    def expect_ident(self, what="a name"):
        if self.tok.type != "IDENT":
            self.error(f"expected {what}, found {self.describe()}")
        return self.advance().value

    def skip_nl(self):
        while self.tok.type == "NEWLINE":
            self.advance()

    def end_stmt(self):
        t = self.tok
        if t.type == "NEWLINE":
            self.skip_nl()
        elif t.type == "EOF" or self.is_op("}"):
            return
        else:
            self.error(f"unexpected {self.describe()} (expected end of statement)")

    # -------------------------------------------------------------- statements
    def parse_program(self):
        stmts = []
        self.skip_nl()
        while self.tok.type != "EOF":
            stmts.append(self.statement())
            self.end_stmt()
        return stmts

    def block(self):
        self.expect_op("{", " to start a block")
        stmts = []
        self.skip_nl()
        while not self.is_op("}"):
            if self.tok.type == "EOF":
                self.error("missing '}' to close the block")
            stmts.append(self.statement())
            self.end_stmt()
        self.advance()
        return stmts

    def statement(self):
        t = self.tok
        if t.type == "KW":
            k = t.value
            if k == "let":
                return self.let_stmt()
            if k == "fn" and self.peek().type == "IDENT":
                return self.fn_def()
            if k == "if":
                return self.if_stmt()
            if k == "while":
                self.advance()
                cond = self.expr()
                return A.While(cond, self.block(), line=t.line)
            if k == "for":
                return self.for_stmt()
            if k == "return":
                self.advance()
                value = None
                if self.tok.type not in ("NEWLINE", "EOF") and not self.is_op("}"):
                    value = self.expr()
                return A.Return(value, line=t.line)
            if k == "break":
                self.advance()
                return A.Break(line=t.line)
            if k == "continue":
                self.advance()
                return A.Continue(line=t.line)
            if k == "try":
                return self.try_stmt()
            if k == "throw":
                self.advance()
                return A.Throw(self.expr(), line=t.line)
            if k == "import":
                return self.import_stmt()
            if k == "class":
                return self.class_stmt()
        return self.expr_or_assign()

    def let_stmt(self):
        line = self.advance().line
        if self.accept_op("["):
            names = self.name_list()
            self.expect_op("]", " to close the unpacking list")
        else:
            names = [self.expect_ident("a variable name after 'let'")]
            while self.accept_op(","):
                names.append(self.expect_ident())
        value = None
        if self.accept_op("="):
            self.skip_nl()
            value = self.expr()
        return A.Let(names, value, line=line)

    def params(self):
        self.expect_op("(", " to start the parameter list")
        params = []
        seen_default = False
        while not self.is_op(")"):
            name = self.expect_ident("a parameter name")
            default = None
            if self.accept_op("="):
                default = self.expr()
                seen_default = True
            elif seen_default:
                self.error(f"parameter '{name}' without default follows a default parameter")
            params.append((name, default))
            if not self.accept_op(","):
                break
        self.expect_op(")", " to close the parameter list")
        return params

    def fn_def(self):
        line = self.advance().line
        name = self.expect_ident()
        params = self.params()
        return A.FnDef(name, params, self.block(), line=line)

    def _peek_keyword_after_newlines(self, *kws):
        save = self.pos
        self.skip_nl()
        if self.tok.type == "KW" and self.tok.value in kws:
            return True
        self.pos = save
        return False

    def if_stmt(self):
        line = self.advance().line
        cond = self.expr()
        branches = [(cond, self.block())]
        orelse = None
        while True:
            if not self._peek_keyword_after_newlines("elif", "else"):
                break
            if self.is_kw("elif"):
                self.advance()
                cond = self.expr()
                branches.append((cond, self.block()))
                continue
            self.advance()  # else
            if self.is_kw("if"):
                self.advance()
                cond = self.expr()
                branches.append((cond, self.block()))
                continue
            orelse = self.block()
            break
        return A.If(branches, orelse, line=line)

    def name_list(self):
        if self.accept_op("["):
            names = self.name_list()
            self.expect_op("]", " to close the unpacking list")
            return names
        names = [self.expect_ident()]
        while self.accept_op(","):
            names.append(self.expect_ident())
        return names

    def for_stmt(self):
        line = self.advance().line
        names = self.name_list()
        self.expect_kw("in")
        it = self.expr()
        return A.For(names, it, self.block(), line=line)

    def try_stmt(self):
        line = self.advance().line
        body = self.block()
        if not self._peek_keyword_after_newlines("catch"):
            self.error("'try' must be followed by 'catch'")
        self.advance()
        name = None
        if self.tok.type == "IDENT":
            name = self.advance().value
        return A.Try(body, name, self.block(), line=line)

    def class_stmt(self):
        line = self.advance().line
        name = self.expect_ident("a class name")
        parent = None
        if self.is_kw("extends"):
            self.advance()
            parent = self.postfix()
        self.expect_op("{", " to start the class body")
        methods = []
        self.skip_nl()
        while not self.is_op("}"):
            if self.tok.type == "EOF":
                self.error("missing '}' to close the class")
            if not (self.is_kw("fn") and self.peek().type == "IDENT"):
                self.error("only methods (fn name(...) { ... }) may appear inside a class")
            methods.append(self.fn_def())
            self.skip_nl()
        self.advance()
        return A.ClassDef(name, parent, methods, line=line)

    def import_stmt(self):
        line = self.advance().line
        if self.tok.type != "STR":
            self.error("import expects a file path string, e.g. import \"utils.sn\"")
        path = self.advance().value
        alias = None
        if self.is_kw("as"):
            self.advance()
            alias = self.expect_ident()
        return A.Import(path, alias, line=line)

    def expr_or_assign(self):
        e = self.expr()
        t = self.tok
        if t.type == "OP" and t.value in ASSIGN_OPS:
            if not isinstance(e, (A.Ident, A.Index, A.Member)):
                self.error("invalid assignment target", t)
            self.advance()
            self.skip_nl()
            return A.Assign(e, t.value, self.expr(), line=t.line)
        return A.ExprStmt(e, line=e.line)

    # ------------------------------------------------------------- expressions
    def expr(self):
        """expression, including the conditional form  `a if cond else b`"""
        value = self.pipe_expr()
        if self.is_kw("if"):
            line = self.advance().line
            cond = self.pipe_expr()
            if not self.is_kw("else"):
                self.error("a conditional expression needs an 'else' part:  a if cond else b")
            self.advance()
            self.skip_nl()
            return A.Ternary(value, cond, self.expr(), line=line)
        return value

    def pipe_expr(self):
        left = self.logic_or()
        while self.is_op("|>"):
            line = self.advance().line
            self.skip_nl()
            left = A.Pipe(left, self.logic_or(), line=line)
        return left

    def logic_or(self):
        left = self.logic_and()
        while self.is_kw("or"):
            line = self.advance().line
            self.skip_nl()
            left = A.Logical("or", left, self.logic_and(), line=line)
        return left

    def logic_and(self):
        left = self.logic_not()
        while self.is_kw("and"):
            line = self.advance().line
            self.skip_nl()
            left = A.Logical("and", left, self.logic_not(), line=line)
        return left

    def logic_not(self):
        if self.is_kw("not"):
            line = self.advance().line
            return A.Unary("not", self.logic_not(), line=line)
        return self.comparison()

    def comparison(self):
        left = self.additive()
        while True:
            t = self.tok
            if t.type == "OP" and t.value in COMPARE:
                op = t.value
                self.advance()
            elif self.is_kw("in"):
                op = "in"
                self.advance()
            elif self.is_kw("not") and self.is_kw("in", self.peek()):
                op = "not in"
                self.advance()
                self.advance()
            else:
                break
            self.skip_nl()
            left = A.Binary(op, left, self.additive(), line=t.line)
        return left

    def additive(self):
        left = self.multiplicative()
        while self.tok.type == "OP" and self.tok.value in ("+", "-"):
            t = self.advance()
            self.skip_nl()
            left = A.Binary(t.value, left, self.multiplicative(), line=t.line)
        return left

    def multiplicative(self):
        left = self.unary()
        while self.tok.type == "OP" and self.tok.value in ("*", "/", "//", "%", "@"):
            t = self.advance()
            self.skip_nl()
            left = A.Binary(t.value, left, self.unary(), line=t.line)
        return left

    def unary(self):
        t = self.tok
        if t.type == "OP" and t.value in ("-", "+"):
            self.advance()
            return A.Unary(t.value, self.unary(), line=t.line)
        return self.power()

    def power(self):
        base = self.postfix()
        if self.is_op("**"):
            t = self.advance()
            self.skip_nl()
            return A.Binary("**", base, self.unary(), line=t.line)
        return base

    def postfix(self):
        e = self.primary()
        while True:
            if self.is_op("("):
                e = self.call(e)
            elif self.is_op("["):
                e = self.index(e)
            elif self.is_op("."):
                line = self.advance().line
                if self.tok.type != "IDENT":
                    self.error(f"expected a member name after '.', found {self.describe()}")
                e = A.Member(e, self.advance().value, line=line)
            else:
                break
        return e

    def call(self, callee):
        line = self.advance().line
        args, kwargs = [], []
        while not self.is_op(")"):
            if self.tok.type == "IDENT" and self.is_op("=", self.peek()):
                name = self.advance().value
                self.advance()
                kwargs.append((name, self.expr()))
            else:
                if kwargs:
                    self.error("positional argument follows keyword argument")
                args.append(self.expr())
            if not self.accept_op(","):
                break
        self.expect_op(")", " to close the call")
        return A.Call(callee, args, kwargs, line=line)

    def index(self, obj):
        line = self.advance().line
        keys = []
        while True:
            keys.append(self.index_item())
            if not self.accept_op(","):
                break
        self.expect_op("]", " to close the index")
        return A.Index(obj, keys, line=line)

    def index_item(self):
        start = None
        line = self.tok.line
        if not self.is_op(":"):
            start = self.expr()
        if self.is_op(":"):
            self.advance()
            stop = step = None
            if not (self.is_op(":") or self.is_op("]") or self.is_op(",")):
                stop = self.expr()
            if self.accept_op(":"):
                if not (self.is_op("]") or self.is_op(",")):
                    step = self.expr()
            return A.Slice(start, stop, step, line=line)
        return start

    # ---------------------------------------------------------------- primary
    def _paren_lambda_ahead(self):
        toks = self.toks
        i = self.pos + 1

        def at(k, v):
            return k < len(toks) and toks[k].type == "OP" and toks[k].value == v

        if at(i, ")"):
            return at(i + 1, "=>")
        while True:
            if i >= len(toks) or toks[i].type != "IDENT":
                return False
            i += 1
            if at(i, ","):
                i += 1
                continue
            return at(i, ")") and at(i + 1, "=>")

    def primary(self):
        t = self.tok
        tt = t.type
        if tt == "NUM":
            self.advance()
            return A.Num(t.value, line=t.line)
        if tt == "STR":
            self.advance()
            return A.Str(t.value, line=t.line)
        if tt == "FSTR":
            self.advance()
            return self.fstring(t)
        if tt == "IDENT":
            if self.is_op("=>", self.peek()) and self.pos + 1 != self.arm_arrow:
                self.advance()
                self.advance()
                self.skip_nl()
                return A.Lambda([(t.value, None)], self.expr(), True, line=t.line)
            self.advance()
            return A.Ident(t.value, line=t.line)
        if tt == "KW":
            k = t.value
            if k == "true" or k == "false":
                self.advance()
                return A.Bool(k == "true", line=t.line)
            if k == "nil":
                self.advance()
                return A.Nil(line=t.line)
            if k == "fn":
                return self.lambda_expr()
            if k == "match":
                return self.match_expr()
            self.error(f"unexpected keyword '{t.value}'")
        if tt == "OP":
            if t.value == "(":
                if self._paren_lambda_ahead():
                    params = self.params()
                    self.expect_op("=>")
                    self.skip_nl()
                    return A.Lambda(params, self.expr(), True, line=t.line)
                self.advance()
                e = self.expr()
                self.expect_op(")", " to close the parenthesis")
                return e
            if t.value == "[":
                return self.list_literal()
            if t.value == "{":
                return self.map_literal()
        self.error(f"unexpected {self.describe()}")

    # ------------------------------------------------------------------ match
    def match_expr(self):
        line = self.advance().line
        subject = self.expr()
        self.expect_op("{", " to start the match arms")
        arms = []
        self.skip_nl()
        while not self.is_op("}"):
            if self.tok.type == "EOF":
                self.error("missing '}' to close the match")
            pat = self.pattern()
            guard = None
            if self.is_kw("if"):
                self.advance()
                saved, self.arm_arrow = self.arm_arrow, self._find_arm_arrow()
                guard = self.expr()
                self.arm_arrow = saved
            self.expect_op("=>", " after the pattern")
            self.skip_nl()
            if self.is_op("{"):
                body, is_block = self.block(), True
            else:
                body, is_block = self.expr(), False
            arms.append((pat, guard, body, is_block))
            self.accept_op(",")
            self.skip_nl()
        self.advance()
        if not arms:
            self.error("a match needs at least one arm")
        return A.Match(subject, arms, line=line)

    def _find_arm_arrow(self):
        """Index of the '=>' that ends a guard (first one outside brackets)."""
        depth = 0
        i = self.pos
        toks = self.toks
        while i < len(toks) and toks[i].type not in ("NEWLINE", "EOF"):
            t = toks[i]
            if t.type == "OP":
                if t.value in "([{":
                    depth += 1
                elif t.value in ")]}":
                    depth -= 1
                elif t.value == "=>" and depth == 0:
                    return i
            i += 1
        return -1

    def pattern(self):
        alts = [self.pattern_atom()]
        while self.is_kw("or"):
            self.advance()
            alts.append(self.pattern_atom())
        return alts[0] if len(alts) == 1 else A.POr(alts, line=alts[0].line)

    def pattern_atom(self):
        t = self.tok
        tt = t.type
        if tt == "NUM":
            self.advance()
            return A.PLit(t.value, line=t.line)
        if tt == "OP" and t.value == "-" and self.peek().type == "NUM":
            self.advance()
            return A.PLit(-self.advance().value, line=t.line)
        if tt == "STR":
            self.advance()
            return A.PLit(t.value, line=t.line)
        if tt == "KW" and t.value in ("true", "false", "nil"):
            self.advance()
            return A.PLit({"true": True, "false": False, "nil": None}[t.value], line=t.line)
        if tt == "IDENT":
            self.advance()
            return A.PWild(line=t.line) if t.value == "_" else A.PBind(t.value, line=t.line)
        if tt == "OP" and t.value == "(":
            self.advance()
            p = self.pattern()
            self.expect_op(")", " to close the pattern")
            return p
        if tt == "OP" and t.value == "[":
            return self.list_pattern()
        if tt == "OP" and t.value == "{":
            return self.map_pattern()
        self.error(f"unexpected {self.describe()} in a pattern")

    def list_pattern(self):
        line = self.advance().line
        items, rest = [], None
        while not self.is_op("]"):
            if self.is_op("."):
                for _ in range(3):
                    self.expect_op(".", " (use ... for the rest of a list)")
                rest = self.expect_ident("a name after '...' (or _)") if self.tok.type == "IDENT" else "_"
                if not self.is_op("]"):
                    self.error("the '...' rest pattern must come last")
                break
            items.append(self.pattern())
            if not self.accept_op(","):
                break
        self.expect_op("]", " to close the list pattern")
        return A.PList(items, rest, line=line)

    def map_pattern(self):
        line = self.advance().line
        entries = []
        self.skip_nl()
        while not self.is_op("}"):
            t = self.tok
            if t.type == "IDENT":
                key = t.value
            elif t.type in ("STR", "NUM"):
                key = t.value
            else:
                self.error(f"unexpected {self.describe()} as a map pattern key")
            self.advance()
            if self.accept_op(":"):
                self.skip_nl()
                pat = self.pattern()
            elif t.type == "IDENT":
                pat = A.PBind(key, line=t.line)          # {name} binds name = map["name"]
            else:
                self.error("expected ':' after the map pattern key")
            entries.append((key, pat))
            self.skip_nl()
            if not self.accept_op(","):
                break
            self.skip_nl()
        self.skip_nl()
        self.expect_op("}", " to close the map pattern")
        return A.PMap(entries, line=line)

    def lambda_expr(self):
        line = self.advance().line
        if self.tok.type == "IDENT":
            self.error("function declarations are statements; give an anonymous fn no name")
        params = self.params()
        if self.accept_op("=>"):
            self.skip_nl()
            return A.Lambda(params, self.expr(), True, line=line)
        return A.Lambda(params, self.block(), False, line=line)

    def list_literal(self):
        line = self.advance().line
        if self.accept_op("]"):
            return A.ListLit([], line=line)
        first = self.expr()
        if self.is_kw("for"):
            self.advance()
            names = self.name_list()
            self.expect_kw("in")
            it = self.logic_or()
            cond = None
            if self.is_kw("if"):
                self.advance()
                cond = self.expr()
            self.expect_op("]", " to close the list comprehension")
            return A.ListComp(first, names, it, cond, line=line)
        items = [first]
        while self.accept_op(","):
            if self.is_op("]"):
                break
            items.append(self.expr())
        self.expect_op("]", " to close the list")
        return A.ListLit(items, line=line)

    def map_literal(self):
        line = self.advance().line
        entries = []
        self.skip_nl()
        while not self.is_op("}"):
            t = self.tok
            if t.type == "IDENT" and self.is_op(":", self.peek()):
                key = A.Str(t.value, line=t.line)
                self.advance()
            elif self.is_op("["):
                self.advance()
                key = self.expr()
                self.expect_op("]")
            else:
                key = self.expr()
            self.expect_op(":", " after the map key")
            self.skip_nl()
            entries.append((key, self.expr()))
            self.skip_nl()
            if not self.accept_op(","):
                break
            self.skip_nl()
        self.skip_nl()
        self.expect_op("}", " to close the map")
        return A.MapLit(entries, line=line)

    def fstring(self, tok):
        parts = []
        for p in tok.value:
            if isinstance(p, str):
                parts.append(p)
                continue
            text, spec, line = p
            sub = Parser(Lexer(text, self.filename, line).tokenize(), self.filename)
            sub.skip_nl()
            e = sub.expr()
            sub.skip_nl()
            if sub.tok.type != "EOF":
                sub.error(f"unexpected {sub.describe()} in f-string expression")
            parts.append((e, spec))
        return A.FStr(parts, line=tok.line)


def parse(source, filename="<string>"):
    return Parser(Lexer(source, filename).tokenize(), filename).parse_program()
