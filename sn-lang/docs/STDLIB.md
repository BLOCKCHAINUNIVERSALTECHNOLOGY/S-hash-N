# S#N Standard Library

Everything here is available without importing. Functions marked *(pipe)* take their main data as the
first argument so they work with `|>`. Many functions accept keyword arguments, e.g. `sum(xs, start=10)`.

## Constants
`PI`, `E`, `INF`, `NAN`, `SN_VERSION`, and `args` (list of command-line arguments).

## Input / output and system
| Function | Description |
|---|---|
| `print(*values, sep=" ", end="\n")` (`छापो`) | write text |
| `input(prompt="")` (`इनपुट`) | read a line (nil at end of input) |
| `read_file(path)`, `write_file(path, text)`, `append_file(path, text)`, `exists(path)` | files |
| `to_json(value, indent=nil)`, `parse_json(text)` | JSON (tensors and data frames can be written) |
| `time()`, `clock()`, `sleep(seconds)` | wall clock, high-resolution timer, pause |
| `exit(code=0)` | stop the program |
| `assert(condition, message)` | raise an error when false |

## Types and conversion
`type(v)` (`प्रकार`), `str(v)` (`पाठ`), `repr(v)`, `int(v)` (`पूर्णांक`), `float(v)` (`दशमलव`), `bool(v)`,
`list(iterable)`, `len(v)` (`लंबाई`), `range(stop)` / `range(start, stop, step)` (`श्रेणी`), `format(value, spec)`.

## Math (work on numbers, lists and tensors)
`abs` (`निरपेक्ष`), `round(x, digits=0)` (`गोल`), `floor`, `ceil`, `sqrt`, `exp`, `log(x, base=nil)`, `log2`, `log10`,
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y, x)`, `tanh`, `sigmoid`, `relu`, `pow(a, b)`, `clip(x, lo, hi)`.

## Statistics
`sum` (`योग`), `mean` (`औसत`), `median` (`माध्यिका`), `std`, `var`, `min` (`न्यूनतम`), `max` (`अधिकतम`),
`percentile(xs, q)`, `mode(xs)`, `corr(a, b)`. `mean`, `std`, `var` and `median` take `axis=` for tensors;
`std`/`var` take `ddof=` (default 0 = population). `min`/`max` accept several values or one list, plus `key=`.

## Collections
| Function | Description |
|---|---|
| `map(xs, f)`, `filter(xs, f)`, `reduce(xs, f, initial)` | functional tools *(pipe)* |
| `sorted(xs, key=nil, reverse=false)` (`क्रमबद्ध`), `reversed(xs)`, `unique(xs)`, `flatten(xs)` | *(pipe)* |
| `zip(a, b, ...)`, `enumerate(xs, start=0)` | produce lists of `[.., ..]` pairs |
| `all(xs)`, `any(xs)` | truthiness tests |
| `keys(m)`, `values(m)`, `items(m)` | map helpers (also available as methods) |

## Random numbers
`seed(n)`, `random()`, `randint(lo, hi)` (inclusive), `uniform(lo, hi)`, `normal(mu, sigma)`,
`choice(xs)`, `shuffle(xs)` (returns a copy), `sample(xs, k)`. `seed` also makes `rand`, `randn` and
`train_test_split` reproducible when they are not given their own `seed=`.

## Tensors
Create: `tensor(nested_list)`, `zeros(2, 3)`, `ones(...)`, `full(shape, value)`, `eye(n)`, `arange(stop)` /
`arange(start, stop, step)`, `linspace(a, b, n)`, `rand(shape...)`, `randn(shape...)`.

Functions: `shape(x)`, `reshape(x, ...)`, `transpose(x)`, `matmul(a, b)` / `dot`, `concat(list, axis=0)`,
`stack(list)`, `where(cond, a, b)`, `softmax(x, axis=-1)`, `argmax(x, axis)`, `argmin`, `norm(x)`,
`solve(A, b)`, `inv(A)`, `det(A)`, `equal(a, b)`, `allclose(a, b, tol)`.

Properties: `t.shape`, `t.ndim`, `t.size`, `t.T`.
Methods: `sum mean max min std var median prod all any` (take `axis=`, `keepdims=` where it makes sense), `argmax argmin`,
`reshape flatten transpose tolist copy item dot abs sqrt exp log round clip apply(fn) sort argsort take(indices) norm`.

Indexing: `t[i]`, `t[i, j]`, `t[1:3]`, `t[:, 0]`, `t[[0, 2]]`, boolean mask `t[t > 0]`; assignment works with the same forms.

## DataFrame
Create with `read_csv(path, sep=",", header=true)` or `DataFrame({col: [values]})` / `DataFrame([row_maps])`.
Empty cells become `nil`; column types (int, float, string) are detected automatically.

Properties: `df.columns`, `df.shape`; `len(df)` is the row count; `"col" in df`.

Indexing: `df["col"]` (a tensor if numeric, else a list), `df[["a", "b"]]`, `df[3]` (row as map), `df[1:5]`, `df[mask]`,
`df["new"] = values`.

Methods *(all return a new DataFrame unless noted)*:
`head(n)`, `tail(n)`, `select(cols...)`, `drop(cols...)`, `filter(row => bool)`, `sort_by(col, desc=false)`,
`add_column(name, values_or_fn)`, `map_column(name, fn)`, `rename(old, new)`, `drop_na()`, `fill_na(value)`,
`group_by(col, agg="mean")` (agg: mean, sum, min, max, median, std, count), `describe()`, `corr()`, `sample(n, seed)`,
and non-DataFrame results: `col(name)`, `rows()`, `unique(col)`, `value_counts(col)`, `to_tensor(cols=nil)`, `to_csv(path)`.

## Machine learning

Inputs: `X` is a tensor, nested list or DataFrame with one row per sample; `y` is a list or tensor.
Every model has `.fit(X, y)` (returns the model, so calls can be chained), `.predict(X)`, `.score(X, y)` and `.save(path)`;
reload with `load_model(path)`. Classifiers return labels as a list; regressors return a tensor.
`score` is accuracy for classifiers and R² for regressors.

| Model | Constructor | Notes |
|---|---|---|
| `LinearRegression` | `(lr=0.01, epochs=0, l2=0.0, fit_intercept=true)` | `epochs=0` solves the normal equations exactly; `epochs>0` uses gradient descent. Properties `weights`, `bias`. |
| `LogisticRegression` | `(lr=0.1, epochs=500, l2=0.0)` | multi-class softmax regression. `predict_proba(X)`, property `classes`. |
| `KNN` | `(k=3, task="classification")` | `task="regression"` averages neighbours. |
| `KMeans` | `(k=3, iters=100, seed=nil, n_init=3)` | `fit_predict(X)`, properties `centroids`, `labels`, `inertia`. k-means++ start. |
| `NeuralNet` | `(layers, activation="relu", task="classification", lr=0.01, epochs=200, batch_size=32, l2=0, optimizer="adam", seed=nil, verbose=0)` | `layers` includes input and output sizes, e.g. `[4, 16, 3]`. Activations: relu, tanh, sigmoid, linear. `task="regression"` trains with mean squared error. Calling `fit` again continues training; `reset()` re-initialises. Properties `history` (loss per epoch), `layers`, `classes`. `verbose=10` prints every 10 epochs. |
| `DecisionTree` | `(task="classification", max_depth=nil, min_samples_split=2, min_samples_leaf=1, max_features=nil, criterion="gini", seed=nil)` | CART tree. `criterion` is "gini" or "entropy" (classification); regression minimises squared error. `predict_proba(X)`, `rules(feature_names=nil)` (if/else text), properties `feature_importances`, `depth`, `n_leaves`, `classes`, `feature_names`. Unlimited depth memorises the training data, so set `max_depth`. |
| `RandomForest` | `(n_trees=50, task="classification", max_depth=nil, min_samples_split=2, min_samples_leaf=1, max_features="default", criterion="gini", bootstrap=true, seed=nil)` | Bagged trees. `max_features` is "sqrt" for classification and all features for regression (also nil, "log2", an int or a fraction). `predict_proba(X)`, `importance_table()` (largest first), properties `oob_score` (accuracy or R² on out-of-bag rows), `feature_importances`, `n_estimators`. |
| `GradientBoosting` | `(n_trees=100, lr=0.1, max_depth=3, task="classification", min_samples_split=2, min_samples_leaf=1, subsample=1.0, seed=nil)` | Boosted regression trees; classification uses softmax cross-entropy with one tree per class per round. `subsample<1` gives stochastic boosting. `predict_proba(X)`, properties `feature_importances`, `n_estimators`, `classes`. |
| `NaiveBayes` | `(var_smoothing=1e-9)` | Gaussian naive Bayes classifier. `predict_proba(X)`, properties `classes`, `means`. |
| `PCA` | `(n_components=2)` | Principal component analysis. An integer keeps that many components, a fraction below 1 keeps enough to explain that share of the variance. `fit`, `transform`, `fit_transform`, `inverse_transform`; properties `components` (one row each), `explained_variance`, `explained_variance_ratio`, `mean`. |
| `StandardScaler`, `MinMaxScaler` | `()` | `fit`, `transform`, `fit_transform`, `inverse_transform`. |

Tips: scale features (`StandardScaler`) before logistic regression or neural networks; if training reports
that it diverged, lower `lr`.

Data helpers: `train_test_split(X, y, test_size=0.2, seed=nil, shuffle=true)` returns `[X_train, X_test, y_train, y_test]`;
`label_encode(values)` returns `{codes, classes}`; `one_hot(labels, k)`.

Metrics: `accuracy`, `mse`, `rmse`, `mae`, `r2`, `precision_score`, `recall_score`, `f1_score` (macro averaged),
`confusion_matrix(y_true, y_pred)` returns `{labels, matrix}`.

## Charts
Charts are written as standalone **SVG** or **HTML** files (`fig.save("chart.svg")`); the folder is created if needed.
Hover over points and bars in a browser to see their values.

```sn
let fig = Plot(title="Loss", xlabel="epoch", ylabel="value", width=720, height=440)
fig.line(xs, ys, label="train")        # fig.line(ys) uses 0, 1, 2, ... as x
fig.scatter(xs, ys, label="points", color="#D55E00")
fig.save("loss.svg")
```

| Call | Description |
|---|---|
| `Plot(title, xlabel, ylabel, width, height)` | a chart; every method below returns the chart so calls chain |
| `.line(xs, ys=nil, label, color)`, `.scatter(xs, ys, label, color)` | several series can share one chart; labelled series get a legend |
| `.bar(labels, values, label, color)` | call repeatedly with the same labels for grouped bars |
| `.hist(values, bins=10, label, color)` | histogram |
| `.heatmap(matrix, x_labels, y_labels)` | annotated colour grid (use it for `confusion_matrix(...).matrix`) |
| `.save(path)` | `.svg` or `.html`; returns the path |
| `.svg()` | the SVG text |
| `.show()` | write a temporary HTML file and open it in the browser |
| `plot(xs, ys)`, `scatter(xs, ys)`, `bar_chart(labels, values)`, `histogram(values, bins)`, `heatmap(matrix, ...)` | one-line shortcuts that return a `Plot` (keywords `title`, `xlabel`, `ylabel`) |

Bar charts cannot be mixed with line/scatter series, and a heatmap stands alone. `nan`/`inf` values are skipped.

## Other
`backend()` returns `"numpy"` or `"python"` (see below), `make_dir(path)` creates a folder, `is_a(obj, Class)` tests inheritance
(or `is_a(value, "int")` for a type name).

**NumPy acceleration.** If NumPy is installed, matrix products of a useful size, `KNN.predict` and `KMeans` use it
automatically; everything else is plain Python. The results are the same up to floating-point rounding.
Set the environment variable `SN_BACKEND=python` to force the pure-Python path.

## Automatic differentiation
`Var(x, requires_grad=true)` - a trainable copy of a number, list or tensor. Operators (`+ - * / ** @`, unary `-`), indexing, and
the functions `exp log sqrt sin cos tanh sigmoid relu abs sum mean softmax norm matmul reshape transpose pow` all accept a `Var` and
record how the result was computed. Comparisons give plain boolean tensors. `// % in` are not differentiable.

Properties: `value`, `grad` (nil until `backward()`), `shape`, `ndim`, `size`, `T`, `is_leaf`, `requires_grad`.
Methods: `backward(grad=nil)` (a single-number result, or pass the starting gradient), `zero_grad()`, `detach()`, `item()`, `tolist()`,
plus `sum(axis, keepdims)`, `mean(axis, keepdims)`, `exp`, `log`, `sqrt`, `abs`, `tanh`, `sigmoid`, `relu`, `sin`, `cos`, `square`, `pow`, `matmul`,
`softmax(axis)`, `reshape`, `flatten`, `transpose`. `w.value = ...` replaces a leaf's value.
Gradients accumulate in `.grad` across `backward()` calls until you reset them; only leaf `Var`s keep a gradient.

`autograd(f, x)` - exact gradient of `f` at `x` (number, list or tensor); `value_and_grad(f, x)` returns `[value, gradient]`.
`SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)` and `Adam(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)` take a
list of leaf `Var`s; call `opt.zero_grad()`, `loss.backward()`, `opt.step()` in a loop (see `examples/14_autograd.sn`).

## Optimisation
`grad(f, x, eps=1e-5)` - numerical gradient of `f` at a number or tensor.
`gradient_descent(f, x0, lr=0.1, steps=100)` - minimise `f` starting at `x0`.
