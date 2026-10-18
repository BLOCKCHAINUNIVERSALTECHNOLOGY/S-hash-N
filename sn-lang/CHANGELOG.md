# Changelog

## 1.3.0
* **New models:** `PCA` (integer or explained-variance fraction, `inverse_transform`; eigenvectors by Jacobi rotations, checked against NumPy),
  `NaiveBayes` (Gaussian), and `GradientBoosting` (regression and multi-class classification, optional `subsample`).
  All support `save` / `load_model`.
* Example `17_more_models.sn`, 14 new tests (178 in total).

## 1.2.1
* **About 2.5x faster interpreter.** The evaluator now compiles each AST node once into a Python closure instead of
  dispatching on node types at run time. Statements return a status code instead of raising exceptions for
  `return`/`break`/`continue`; loops and `if` blocks only create a new scope when they declare names; numeric
  operators have fast paths; bound methods are cached per instance. Behaviour, error messages and call traces are unchanged
  (20 new tests check scoping, closures, control flow and error lines, and pass on both the old and the new evaluator).
  Benchmark (`fib(24)`, 300k-iteration `for`, list comprehension, `while`, 100k method calls): 3.0 s -> 1.2 s.

## 1.2.0
* **Automatic differentiation:** `Var` wraps a tensor and records operations; `loss.backward()` fills `.grad`
  (reverse mode, broadcasting-aware, works with `@`, indexing, `sum`/`mean`, `exp`/`log`/`tanh`/`sigmoid`/`relu`/`softmax`, ...).
  `autograd(f, x)` and `value_and_grad(f, x)` give exact gradients; `SGD` (with momentum) and `Adam` optimizers.
  Gradients are checked against numerical differentiation in the tests.
* **Decision trees and random forests:** `DecisionTree` and `RandomForest` for classification and regression
  (Gini/entropy/squared error), `predict_proba`, feature importances, out-of-bag score, readable `rules()`, save/load.
* **`match` / `मिलान`:** pattern matching with literals, `or` alternatives, bindings, guards (`n if n > 0`),
  list patterns with `...rest`, map patterns, nested patterns and block arms.
* 140+ tests, three new examples (`14`-`16`).

## 1.1.0
* **Classes:** `class` / `वर्ग`, `extends` / `विस्तार`, `self` (`खुद`), `super`, instances with fields and methods,
  operator methods (`add sub mul div eq lt gt le ge len getitem setitem iter contains str bool`), `is_a()`.
* **Plotting:** `Plot` object plus `plot`, `scatter`, `bar_chart`, `histogram`, `heatmap`; saves standalone `.svg` or `.html`.
* **NumPy acceleration (optional):** big matrix products, k-NN and k-means use NumPy when it is installed
  (`backend()` tells which is active; `SN_BACKEND=python` turns it off). Results match the pure-Python path.
* Conditional expression `a if cond else b`, Devanagari digits (`१२३`), `make_dir()`.
* 100+ tests, two new examples.

## 1.0.0
First release: lexer, parser and tree-walking interpreter with English and
Hindi keywords; pure-Python tensors, data frames and a small ML toolkit
(linear/logistic regression, k-NN, k-means, neural networks, scalers,
metrics); CLI with REPL; 80+ automated tests; 11 example programs.
