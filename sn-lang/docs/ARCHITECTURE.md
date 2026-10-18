# How S#N is built

S#N is a closure-compiling interpreter: the parser builds an AST, and `sn/interpreter.py` turns each node into a Python closure once (cached), so running a program never re-dispatches on node types. A program flows through four stages:

```
source text --Lexer--> tokens --Parser--> AST --Interpreter--> result
```

| File | Role |
|---|---|
| `sn/lexer.py` | Turns text into tokens. Handles Unicode identifiers (Devanagari letters *and* vowel signs), English and Hindi keywords (normalised to one canonical keyword each), f-string parts, comments, and decides when a newline ends a statement (it does not inside `()`/`[]`, or before a line starting with `|>` or `.name`). |
| `sn/ast.py` | Node classes, generated from a one-line description each. |
| `sn/parser.py` | Recursive-descent parser. One method per precedence level (`expr` -> `logic_or` -> ... -> `postfix` -> `primary`). Produces precise `SyntaxError`s with line and column. |
| `sn/interpreter.py` | Compiles the AST to closures. `e_<Node>` methods build expression closures, `s_<Node>` build statement closures (cached per node). Statement closures return `None` or a status code (`_BRK`, `_CNT`, `_RET`) to unwind loops and functions without exceptions; a `return`/`break` inside a `match` expression arm is raised as a `BaseException` signal because expressions cannot return codes. Scopes are chained `Env` objects, created only where a block declares names or a closure could capture them. |
| `sn/ops.py` | Operator semantics and the object protocol: arithmetic, comparison, indexing, iteration and member access for every value type. |
| `sn/values.py` | Runtime types (`Function`, `NativeFn`, `NativeObject`) and text conversion. S#N values reuse Python types: nil=`None`, numbers, `str`, `list`, `dict`. |
| `sn/methods.py` | Methods on lists, strings and maps. |
| `sn/objects.py` | Classes: `Class`, `Instance`, `SuperProxy`, method binding (a method is copied with `self`/`super` placed in a fresh scope) and operator overloading. |
| `sn/tensor.py` | `Tensor`: flat row-major list + shape; broadcasting via precomputed index lists; reductions by strided slices; matmul with row/column slices and `sum(map(mul, ...))`. |
| `sn/dataframe.py` | Column-oriented `DataFrame` and CSV loading with type inference. |
| `sn/ml.py` | Models, scalers, metrics, splitting. Built only from `Tensor` operations. |
| `sn/autograd.py` | Reverse-mode autodiff: `Var` nodes hold a value, their parents and a closure that maps the output gradient to input gradients; `backward()` walks the graph in reverse topological order. Optimizers `SGD` and `Adam`. |
| `sn/trees.py` | CART decision trees (sorted sweep for the best split) and random forests (bootstrap + feature subsampling, out-of-bag score). |
| `sn/models_more.py` | `PCA`, `NaiveBayes` and `GradientBoosting` (boosting reuses the tree builder from `trees.py`). |
| `sn/plot.py` | SVG chart renderer (axes, nice ticks, legend, tooltips) behind the `Plot` object. |
| `sn/stdlib.py` | The built-in function table (`BUILTINS`). |
| `sn/cli.py` | `sn` command, REPL, error printing. Runs on a thread with a large stack so deep recursion is safe. |
| `sn/errors.py` | Error classes. Errors carry a line, a column (syntax errors) and a call trace. |

## Extending the language from Python

Add a built-in function in `sn/stdlib.py`:

```python
@builtin("double", "दुगुना")        # English name, then optional aliases
def sn_double(x):
    return x * 2
```

Python arguments map onto S#N arguments, including keyword arguments; any Python exception is
turned into a normal S#N runtime error (set `SN_DEBUG=1` to see the traceback).

Add a new object type by subclassing `NativeObject` (`sn/values.py`): methods named `m_<name>` and read-only
properties named `p_<name>` are visible to S#N code, and `sn_getitem`, `sn_iter`, `sn_len`, `sn_repr` hook into
indexing, loops, `len` and printing. `Tensor`, `DataFrame` and every model are built this way.

Call S#N functions from Python with `values.call_value(fn, *args)`.

## Design decisions

* **Pipe-friendly APIs.** The data argument comes first (`map(xs, f)`), so `xs |> map(f)` reads naturally.
* **Errors that help.** Unknown names and members suggest the closest match; shape mismatches print both shapes; training
  that blows up says "lower lr or scale features".
* **Mutable tensors, immutable operations.** Operators always create new tensors; only indexed assignment mutates.
* **No required dependencies.** Everything runs with the Python standard library, which keeps installation trivial and the code easy to read.
  NumPy is used only as an optional accelerator for a few heavy kernels (`tensor.matmul`, `KNN.predict`, `KMeans._assign`); each has an
  equivalent pure-Python path that the tests compare it against.
* **Python recursion.** Each S#N call uses several Python frames, so the CLI raises the recursion limit and runs the program in a
  thread with a 512 MB stack. S#N itself caps recursion at 900 nested calls and reports a normal error beyond that.
