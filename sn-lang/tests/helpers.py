import contextlib
import io
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sn.interpreter import Interpreter  # noqa: E402


def run(source, cwd=None):
    """Run S#N source and return everything it printed."""
    buf = io.StringIO()
    old = os.getcwd()
    if cwd:
        os.chdir(cwd)
    try:
        with contextlib.redirect_stdout(buf):
            Interpreter("<test>").run(source, "<test>")
    finally:
        os.chdir(old)
    return buf.getvalue()


def lines(source):
    return run(source).strip().splitlines()
