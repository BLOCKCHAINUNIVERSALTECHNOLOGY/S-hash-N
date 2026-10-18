# S#N

**A small, readable programming language for AI, ML and data work.**
Write in English, Hindi (हिंदी) or a mix of both.

```sn
let houses = read_csv("examples/data/houses.csv")
let X = houses.select("area_sqft", "bedrooms", "age_years").to_tensor()
let [X_train, X_test, y_train, y_test] = train_test_split(X, houses["price_lakh"], seed=42)

let model = LinearRegression().fit(X_train, y_train)
print(f"R2 on unseen data: {model.score(X_test, y_test):.3f}")
```

```sn
# वही भाषा, हिंदी कीवर्ड के साथ
कार्य अभाज्य_है(n) {
    अगर n < 2 { लौटाओ झूठ }
    हर d में श्रेणी(2, n) {
        अगर d * d > n { रोको }
        अगर n % d == 0 { लौटाओ झूठ }
    }
    लौटाओ सच
}
छापो([n for n in श्रेणी(30) if अभाज्य_है(n)])
```

## What is in the box

| Area | What you get |
|---|---|
| **Language** | variables, functions, **classes with inheritance and operator methods**, **`match` pattern matching**, closures, lambdas, default and keyword arguments, lists, maps, comprehensions, conditional expressions, f-strings with format specs, `try`/`catch`/`throw`, modules (`import`), pipe operator `|>` |
| **Tensors** | n-dimensional arrays with NumPy-style broadcasting, slicing, masks, `@` matrix multiply, reductions, `solve`/`inv`/`det` |
| **Data** | `DataFrame` with CSV I/O, filter, sort, group-by, describe, missing-value handling, boolean masks |
| **ML** | `LinearRegression`, `LogisticRegression`, `KNN`, `KMeans`, `NeuralNet` (mini-batch Adam/SGD), **`DecisionTree`, `RandomForest`, `GradientBoosting`, `NaiveBayes`, `PCA`**, `StandardScaler`, `MinMaxScaler`, metrics, `train_test_split`, model save/load |
| **Autodiff** | **reverse-mode automatic differentiation**: `Var`, `backward()`, `autograd(f, x)`, `SGD` and `Adam` optimizers for training your own models |
| **Charts** | line, scatter, bar, histogram and heatmap charts saved as standalone SVG or HTML files |
| **Tooling** | `sn` command with REPL, `sn check`, readable errors with source line and call trace, VS Code syntax highlighting, 100+ tests |
| **Dependencies** | **None required.** Pure Python 3.9+ (tested on 3.13). If NumPy happens to be installed, big matrix products, k-NN and k-means use it automatically |

## Install and run

You only need Python 3.9 or newer.

```bash
# Option 1: run straight from this folder (nothing to install)
python3 -m sn examples/01_hello.sn        # Windows: python -m sn examples\01_hello.sn
./bin/sn examples/01_hello.sn             # Linux / macOS shortcut

# Option 2: install the `sn` command
pip install .
sn examples/01_hello.sn
```

```bash
sn                      # interactive REPL
sn file.sn [args...]    # run a program (script arguments appear in `args`)
sn check file.sn        # syntax check only
sn eval "print(2 ** 10)"
sn --version
```

Run the test suite: `cd tests && python3 -m unittest discover`

## Learn the language

1. Start with `examples/` - run them in order, they build on each other:
   `01_hello` -> `02_language_tour` -> `04_tensors` -> `05_data_pipeline` -> `06_linear_regression` -> `07_classification`
   -> `12_classes` -> `13_plotting` -> `14_autograd` -> `15_trees_forest` -> `16_match` -> `17_more_models`.
2. [`docs/LANGUAGE.md`](docs/LANGUAGE.md) - the language reference.
3. [`docs/STDLIB.md`](docs/STDLIB.md) - every built-in function, tensor/data-frame method and model.
4. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - how the interpreter is built (good for learning or extending it).

### Hindi keywords

| English | हिंदी | | English | हिंदी |
|---|---|---|---|---|
| `let` | `मान` | | `try` | `कोशिश` |
| `fn` | `कार्य` | | `catch` | `पकड़ो` |
| `if` / `elif` / `else` | `अगर` / `वरना_अगर` / `वरना` | | `throw` | `फेंको` |
| `while` | `जबतक` | | `import` / `as` | `आयात` / `जैसे` |
| `for` / `in` | `हर` / `में` | | `true` / `false` / `nil` | `सच` / `झूठ` / `शून्य` |
| `return` | `लौटाओ` | | `and` / `or` / `not` | `और` / `या` / `नहीं` |
| `break` / `continue` | `रोको` / `आगे` | | `class` / `extends` | `वर्ग` / `विस्तार` |
| `match` | `मिलान` | | | |
| `self` | `खुद` | | digits | `०१२३४५६७८९` also work |

Common built-ins also have Hindi names: `छापो` (print), `लंबाई` (len), `श्रेणी` (range), `योग` (sum),
`औसत` (mean), `अधिकतम` / `न्यूनतम` (max / min), `क्रमबद्ध` (sorted), `पाठ` (str), `पूर्णांक` (int), `दशमलव` (float).
Variable and function names may be written in any script, including Devanagari.

## Honest limitations

S#N is a complete, working language, but it is an **interpreter written in Python** (the program is compiled to Python closures, then run).

* **Speed.** Ordinary code and element-wise tensor maths run at pure-Python speed (`fib(23)` takes about 0.17 s;
  element-wise operations are far slower than NumPy). Big matrix products, k-NN and k-means are accelerated when
  NumPy is installed (a 200x200 matrix product: 0.19 s in pure Python, 0.005 s with NumPy). It is great for learning,
  prototyping and small or medium data (thousands of rows), not for large datasets or deep learning.
* **No GPU.** Automatic differentiation (`Var`, `backward()`) works on the same pure-Python tensors, so it suits small models and
  learning, not big networks. `grad()` is the older numerical version; `NeuralNet` has its own hand-written backprop.
* **Small standard library.** No plotting, no networking, no package manager; modules are plain `.sn` files.
* **No static types, generics or multiple inheritance.** Classes support single inheritance only.
* Tested on Python 3.13 only; it uses just the standard library. The `pip install .` route follows the standard setuptools layout
  but could not be tried in the build sandbox (no internet), so if it fails use `python3 -m sn` or `./bin/sn`, which are tested.

### Roadmap ideas

Compile to bytecode for speed -> storing tensors as NumPy arrays (not only accelerating a few kernels) -> optional type annotations ->
bytecode compiler for a further speed-up -> more models (SVM, t-SNE) -> autograd for `max`/`min`/`concat` -> class patterns in `match` -> language server for editors.

## Project layout

```
sn/            interpreter and libraries (lexer, parser, interpreter, objects, tensor, dataframe, ml, plot, stdlib, cli)
examples/      17 runnable programs + sample data (iris.csv, houses.csv)
tests/         unit and end-to-end tests
docs/          language reference, standard library, architecture
editors/vscode syntax highlighting extension (copy the folder into ~/.vscode/extensions/)
tools/         make_data.py regenerates the sample CSV files
bin/           launcher scripts
```

The Iris data is Fisher's classic public-domain data set; `houses.csv` is synthetic.

Licensed under the MIT License.
