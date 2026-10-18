"""Command line interface: ``sn run file.sn``, ``sn repl``, ``sn check file.sn``."""
import os
import sys
import threading

from . import __version__
from .errors import SNError
from .interpreter import Interpreter
from .parser import parse
from .values import to_repr

HELP = """S#N {version} - a language for AI, ML and data

Usage:
  sn <file.sn> [args...]      run a program (same as 'sn run')
  sn run <file.sn> [args...]  run a program
  sn check <file.sn>          check syntax without running
  sn eval "<code>"            run a snippet of code
  sn repl                     interactive prompt (also: run 'sn' with no arguments)
  sn --version | --help

Keywords can be written in English or Hindi, e.g.  if / अगर,  fn / कार्य.
"""


def _report(err, interp=None, source=None):
    if source is None and interp is not None and err.filename:
        source = interp.sources.get(err.filename)
    sys.stderr.write(err.format(source) + "\n")


def run_file(path, argv):
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        sys.stderr.write(f"sn: cannot read '{path}': {e.strerror}\n")
        return 2
    interp = Interpreter(path, argv)
    try:
        interp.run(source, path)
    except SNError as e:
        sys.stdout.flush()
        _report(e, interp, source if e.filename in (None, path) else None)
        return 1
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130
    return 0


def check_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        sys.stderr.write(f"sn: cannot read '{path}': {e.strerror}\n")
        return 2
    try:
        parse(source, path)
    except SNError as e:
        _report(e, source=source)
        return 1
    print(f"{path}: OK")
    return 0


def eval_code(code):
    interp = Interpreter("<eval>")
    try:
        interp.run(code, "<eval>")
    except SNError as e:
        _report(e, interp, code)
        return 1
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    return 0


def _balance(text):
    depth = 0
    in_str = None
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
        elif c == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        i += 1
    return depth > 0 or in_str is not None


def repl():
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    interp = Interpreter("<repl>")
    print(f"S#N {__version__}  -  type exit() or press Ctrl-D to quit")
    buf = ""
    while True:
        try:
            line = input("sn> " if not buf else "... ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            buf = ""
            continue
        buf = buf + line + "\n" if buf else line + "\n"
        if _balance(buf):
            continue
        source, buf = buf, ""
        if not source.strip():
            continue
        try:
            has_value, value = interp.run(source, "<repl>", repl=True)
            if has_value and value is not None:
                print(to_repr(value) if not isinstance(value, str) else to_repr(value))
        except SNError as e:
            _report(e, interp, source)
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return repl()
    cmd = argv[0]
    if cmd in ("-h", "--help", "help"):
        print(HELP.format(version=__version__))
        return 0
    if cmd in ("-v", "--version", "version"):
        from .tensor import backend
        print(f"S#N {__version__} (numeric backend: {backend()})")
        return 0
    if cmd == "repl":
        return repl()
    if cmd == "check":
        if len(argv) != 2:
            sys.stderr.write("usage: sn check <file.sn>\n")
            return 2
        return check_file(argv[1])
    if cmd in ("eval", "-e"):
        if len(argv) != 2:
            sys.stderr.write("usage: sn eval \"<code>\"\n")
            return 2
        return eval_code(argv[1])
    if cmd == "run":
        if len(argv) < 2:
            sys.stderr.write("usage: sn run <file.sn> [args...]\n")
            return 2
        return run_file(argv[1], argv[2:])
    return run_file(cmd, argv[1:])


def entry():
    """Console-script entry point: run on a thread with a big stack."""
    sys.setrecursionlimit(60000)
    threading.stack_size(512 * 1024 * 1024)
    result = []
    t = threading.Thread(target=lambda: result.append(main()))
    t.start()
    t.join()
    sys.stdout.flush()
    sys.exit(result[0] if result else 1)
