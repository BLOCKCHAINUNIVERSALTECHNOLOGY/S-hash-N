"""Lexer: turns S#N source text into a list of tokens.

Keywords exist in English and in Hindi (Devanagari); both spellings map to
the same canonical keyword, so programs can freely mix them.
"""
import re
import unicodedata

from .errors import SNSyntaxError


def _nfc(s):
    return unicodedata.normalize("NFC", s)


_KEYWORDS = {
    "let": "let", "मान": "let",
    "fn": "fn", "कार्य": "fn",
    "if": "if", "अगर": "if",
    "elif": "elif", "वरना_अगर": "elif",
    "else": "else", "वरना": "else",
    "while": "while", "जबतक": "while",
    "for": "for", "हर": "for",
    "in": "in", "में": "in",
    "return": "return", "लौटाओ": "return",
    "break": "break", "रोको": "break",
    "continue": "continue", "आगे": "continue",
    "true": "true", "सच": "true",
    "false": "false", "झूठ": "false",
    "nil": "nil", "शून्य": "nil",
    "and": "and", "और": "and",
    "or": "or", "या": "or",
    "not": "not", "नहीं": "not",
    "try": "try", "कोशिश": "try",
    "catch": "catch", "पकड़ो": "catch",
    "throw": "throw", "फेंको": "throw",
    "import": "import", "आयात": "import",
    "as": "as", "जैसे": "as",
    "class": "class", "वर्ग": "class",
    "extends": "extends", "विस्तार": "extends",
    "match": "match", "मिलान": "match",
}
KEYWORDS = {_nfc(k): v for k, v in _KEYWORDS.items()}

NUM_RE = re.compile(r"\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?")
FMT_SPEC_RE = re.compile(r"^[<>^]?[+\- ]?,?\d*(?:\.\d+)?[dfeEgGs%]?$")
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\",
           '"': '"', "'": "'", "{": "{", "}": "}"}
TWO_CHAR = {"**", "==", "!=", "<=", ">=", "|>", "=>", "+=", "-=", "*=", "/=", "//"}
ONE_CHAR = set("+-*/%@<>=()[]{},:;.")
OPENERS = {"(": ")", "[": "]", "{": "}"}


class Token:
    __slots__ = ("type", "value", "line", "col")

    def __init__(self, type_, value, line, col):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.col})"


def is_ident_start(ch):
    return ch == "_" or ch.isalpha()


def is_ident_part(ch):
    return (
        ch == "_"
        or ch.isalnum()
        or ch in "‌‍"
        or unicodedata.category(ch) in ("Mn", "Mc")
    )


class Lexer:
    def __init__(self, src, filename="<string>", line=1):
        self.src = src
        self.filename = filename
        self.i = 0
        self.line = line
        self.line_start = 0
        self.tokens = []
        self.brackets = []

    # -- helpers ----------------------------------------------------------
    def _col(self, i=None):
        return (self.i if i is None else i) - self.line_start + 1

    def _error(self, msg, line=None, col=None):
        raise SNSyntaxError(msg, line or self.line, col or self._col(), self.filename)

    def _emit(self, type_, value, line=None, col=None):
        self.tokens.append(Token(type_, value, line or self.line, col or self._col()))

    # -- main loop --------------------------------------------------------
    def tokenize(self):
        src = self.src
        n = len(src)
        while self.i < n:
            ch = src[self.i]
            if ch in " \t\r":
                self.i += 1
            elif ch == "\n":
                self._newline()
            elif ch == "#":
                while self.i < n and src[self.i] != "\n":
                    self.i += 1
            elif ch.isdecimal():
                self._number()
            elif ch in "\"'":
                self._string(ch, False)
            elif ch == "f" and src[self.i + 1:self.i + 2] in ('"', "'") and (
                self.i == 0 or not is_ident_part(src[self.i - 1])
            ):
                self.i += 1
                self._string(src[self.i], True, start_col=self._col(self.i - 1))
            elif is_ident_start(ch):
                self._ident()
            else:
                self._operator()
        if self.brackets:
            op, line, col = self.brackets[-1]
            self._error(f"unclosed '{op}'", line, col)
        if self.tokens and self.tokens[-1].type != "NEWLINE":
            self._emit("NEWLINE", "\n")
        self._emit("EOF", None)
        return self.tokens

    def _newline(self):
        src = self.src
        self.i += 1
        self.line += 1
        self.line_start = self.i
        if self.brackets and self.brackets[-1][0] in "([":
            return
        # A line that starts with '|>' or '.method' continues the previous one.
        j = self.i
        n = len(src)
        while j < n and src[j] in " \t\r\n":
            j += 1
        if src.startswith("|>", j) or (
            src.startswith(".", j) and j + 1 < n and is_ident_start(src[j + 1])
        ):
            return
        if self.tokens and self.tokens[-1].type != "NEWLINE":
            self._emit("NEWLINE", "\n", self.line - 1, 1)

    def _number(self):
        m = NUM_RE.match(self.src, self.i)
        text = m.group(0)
        end = m.end()
        if end < len(self.src) and is_ident_start(self.src[end]):
            self._error(f"invalid number literal '{text}{self.src[end]}...'")
        clean = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text).replace("_", "")
        is_float = "." in clean or "e" in clean or "E" in clean
        value = float(clean) if is_float else int(clean)
        self._emit("NUM", value)
        self.i = end

    def _ident(self):
        src = self.src
        start = self.i
        i = start + 1
        n = len(src)
        while i < n and is_ident_part(src[i]):
            i += 1
        text = _nfc(src[start:i])
        if text == _nfc("खुद"):
            text = "self"
        col = self._col(start)
        kw = KEYWORDS.get(text)
        if kw:
            self._emit("KW", kw, col=col)
        else:
            self._emit("IDENT", text, col=col)
        self.i = i

    def _operator(self):
        src = self.src
        two = src[self.i:self.i + 2]
        col = self._col()
        if two in TWO_CHAR:
            self._emit("OP", two, col=col)
            self.i += 2
            return
        ch = src[self.i]
        if ch not in ONE_CHAR:
            self._error(f"unexpected character {ch!r}")
        if ch in OPENERS:
            self.brackets.append((ch, self.line, col))
        elif ch in ")]}":
            if not self.brackets or OPENERS[self.brackets[-1][0]] != ch:
                self._error(f"unmatched '{ch}'")
            self.brackets.pop()
        if ch == ";":
            if self.tokens and self.tokens[-1].type != "NEWLINE":
                self._emit("NEWLINE", ";", col=col)
        else:
            self._emit("OP", ch, col=col)
        self.i += 1

    def _string(self, quote, fmt, start_col=None):
        src = self.src
        n = len(src)
        line = self.line
        col = start_col or self._col()
        i = self.i + 1
        buf = []
        parts = []
        while True:
            if i >= n or src[i] == "\n":
                self._error("unterminated string", line, col)
            c = src[i]
            if c == quote:
                i += 1
                break
            if c == "\\":
                if i + 1 >= n:
                    self._error("unterminated string", line, col)
                e = src[i + 1]
                buf.append(ESCAPES.get(e, "\\" + e))
                i += 2
                continue
            if fmt and c == "{":
                if src[i + 1:i + 2] == "{":
                    buf.append("{")
                    i += 2
                    continue
                depth = 0
                j = i + 1
                q = None
                spec_at = -1
                while j < n:
                    d = src[j]
                    if q:
                        if d == "\\":
                            j += 1
                        elif d == q:
                            q = None
                    elif d in "\"'":
                        q = d
                    elif d in "([{":
                        depth += 1
                    elif d in ")]":
                        depth -= 1
                    elif d == "}":
                        if depth == 0:
                            break
                        depth -= 1
                    elif d == ":" and depth == 0:
                        spec_at = j
                    elif d == "\n":
                        break
                    j += 1
                if j >= n or src[j] != "}":
                    self._error("unterminated '{' in f-string", line, col)
                expr = src[i + 1:j]
                spec = None
                if spec_at >= 0:
                    cand = src[spec_at + 1:j]
                    if cand and FMT_SPEC_RE.match(cand):
                        expr = src[i + 1:spec_at]
                        spec = cand
                if not expr.strip():
                    self._error("empty expression in f-string", line, col)
                if buf:
                    parts.append("".join(buf))
                    buf = []
                parts.append((expr, spec, line))
                i = j + 1
                continue
            if fmt and c == "}":
                if src[i + 1:i + 2] == "}":
                    buf.append("}")
                    i += 2
                    continue
                self._error("single '}' in f-string (use '}}' for a literal brace)", line, col)
            buf.append(c)
            i += 1
        if fmt:
            if buf:
                parts.append("".join(buf))
            self.tokens.append(Token("FSTR", parts, line, col))
        else:
            self.tokens.append(Token("STR", "".join(buf), line, col))
        self.i = i
