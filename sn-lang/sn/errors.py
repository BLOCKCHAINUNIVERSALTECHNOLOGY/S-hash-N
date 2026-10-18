"""Error types for S#N.

Every error that should be shown to the user derives from :class:`SNError`.
"""


class SNError(Exception):
    """Base class for errors reported to the user of the language."""

    kind = "Error"

    def __init__(self, message, line=None, col=None, filename=None):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col
        self.filename = filename
        self.trace = []  # [(function name, call line), ...] innermost first

    def format(self, source=None):
        """Return a human friendly, multi-line description of the error."""
        out = [f"{self.kind}: {self.message}"]
        if self.line is not None:
            loc = f"{self.filename or '<input>'}:{self.line}"
            if self.col is not None:
                loc += f":{self.col}"
            out.append(f"  --> {loc}")
            if source is not None:
                lines = source.splitlines()
                if 1 <= self.line <= len(lines):
                    gutter = str(self.line)
                    pad = " " * len(gutter)
                    out.append(f"{pad} |")
                    out.append(f"{gutter} | {lines[self.line - 1]}")
                    if self.col is not None:
                        out.append(f"{pad} | {' ' * (self.col - 1)}^")
        for name, line in self.trace[:12]:
            out.append(f"  called from {name}() at line {line}")
        if len(self.trace) > 12:
            out.append(f"  ... {len(self.trace) - 12} more frames")
        return "\n".join(out)


class SNSyntaxError(SNError):
    kind = "SyntaxError"


class SNRuntimeError(SNError):
    kind = "RuntimeError"


class SNThrow(SNRuntimeError):
    """Raised by the ``throw`` statement; carries an arbitrary S#N value."""

    def __init__(self, value, message, line=None):
        super().__init__(message, line)
        self.value = value
