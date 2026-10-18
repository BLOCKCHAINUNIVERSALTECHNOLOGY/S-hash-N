"""Decision trees and random forests (CART), pure Python.

``DecisionTree`` greedily splits the data on the feature/threshold that
reduces impurity the most (Gini or entropy for classification, squared error
for regression).  ``RandomForest`` averages many such trees, each trained on a
bootstrap sample and a random subset of the features.
"""
import math

from . import stats
from .dataframe import DataFrame
from .errors import SNRuntimeError
from .ml import (MODELS, Model, _X, _accuracy, _check_len, _label, _labels, _r2, _rng,
                 _sorted_classes, _vals)
from .tensor import Tensor
from .values import fmt_number, to_str


# ------------------------------------------------------------------ builder
def _resolve_max_features(spec, d):
    if spec is None:
        return d
    if spec == "sqrt":
        return max(1, int(math.sqrt(d)))
    if spec == "log2":
        return max(1, int(math.log2(d))) if d > 1 else 1
    if isinstance(spec, bool) or not isinstance(spec, (int, float)):
        raise SNRuntimeError("max_features must be nil, 'sqrt', 'log2', an int or a fraction")
    if isinstance(spec, float) and 0 < spec <= 1:
        return max(1, int(round(spec * d)))
    if spec >= 1 and float(spec).is_integer():
        return min(int(spec), d)
    raise SNRuntimeError("max_features must be between 1 and the number of features")


class _Builder:
    """Grows one tree over rows ``X`` with targets ``y`` (class indices or floats)."""

    def __init__(self, X, y, task, n_classes, max_depth, min_split, min_leaf,
                 max_features, criterion, rng):
        self.X, self.y, self.task, self.k = X, y, task, n_classes
        self.max_depth, self.min_split, self.min_leaf = max_depth, min_split, min_leaf
        self.d = len(X[0]) if X else 0
        self.mf = _resolve_max_features(max_features, self.d)
        self.criterion, self.rng = criterion, rng
        self.importance = [0.0] * self.d
        self.total = 0

    # -- impurity -----------------------------------------------------------
    def _imp_counts(self, counts, n):
        if n == 0:
            return 0.0
        if self.criterion == "entropy":
            return -math.fsum((c / n) * math.log2(c / n) for c in counts if c)
        return 1.0 - math.fsum((c / n) ** 2 for c in counts)

    def _leaf(self, idx):
        n = len(idx)
        if self.task == "classification":
            counts = [0] * self.k
            for i in idx:
                counts[self.y[i]] += 1
            return {"n": n, "v": counts.index(max(counts)), "d": [c / n for c in counts]}
        return {"n": n, "v": math.fsum(self.y[i] for i in idx) / n}

    def _node_impurity(self, idx):
        n = len(idx)
        if self.task == "classification":
            counts = [0] * self.k
            for i in idx:
                counts[self.y[i]] += 1
            return self._imp_counts(counts, n), max(counts) == n
        ys = [self.y[i] for i in idx]
        m = math.fsum(ys) / n
        var = math.fsum((v - m) ** 2 for v in ys) / n
        return var, var <= 1e-18

    # -- split search -------------------------------------------------------
    def _best_split(self, idx, parent_imp):
        n = len(idx)
        feats = list(range(self.d))
        if self.mf < self.d:
            feats = self.rng.sample(feats, self.mf)
        best = None                       # (weighted impurity, feature, threshold, pos, order)
        msl = self.min_leaf
        X, y = self.X, self.y
        for f in feats:
            order = sorted(idx, key=lambda i: X[i][f])
            vals = [X[i][f] for i in order]
            if vals[0] == vals[-1]:
                continue
            if self.task == "classification":
                left = [0] * self.k
                right = [0] * self.k
                for i in order:
                    right[y[i]] += 1
                for pos in range(n - 1):
                    c = y[order[pos]]
                    left[c] += 1
                    right[c] -= 1
                    if vals[pos] == vals[pos + 1]:
                        continue
                    nl, nr = pos + 1, n - pos - 1
                    if nl < msl or nr < msl:
                        continue
                    w = (nl * self._imp_counts(left, nl) + nr * self._imp_counts(right, nr)) / n
                    if best is None or w < best[0] - 1e-15:
                        best = (w, f, (vals[pos] + vals[pos + 1]) / 2.0, order, nl)
            else:
                tot = math.fsum(y[i] for i in order)
                tot2 = math.fsum(y[i] * y[i] for i in order)
                sl = sl2 = 0.0
                for pos in range(n - 1):
                    v = y[order[pos]]
                    sl += v
                    sl2 += v * v
                    if vals[pos] == vals[pos + 1]:
                        continue
                    nl, nr = pos + 1, n - pos - 1
                    if nl < msl or nr < msl:
                        continue
                    sr, sr2 = tot - sl, tot2 - sl2
                    sse = (sl2 - sl * sl / nl) + (sr2 - sr * sr / nr)
                    w = max(sse, 0.0) / n
                    if best is None or w < best[0] - 1e-15:
                        best = (w, f, (vals[pos] + vals[pos + 1]) / 2.0, order, nl)
        return best

    def build(self, idx, depth=0):
        n = len(idx)
        if depth == 0:
            self.total = n
        imp, pure = self._node_impurity(idx)
        if pure or n < self.min_split or n < 2 * self.min_leaf or \
                (self.max_depth is not None and depth >= self.max_depth):
            return self._leaf(idx)
        best = self._best_split(idx, imp)
        if best is None:
            return self._leaf(idx)
        w, f, thr, order, nl = best
        self.importance[f] += (n / self.total) * max(imp - w, 0.0)
        return {
            "n": n, "f": f, "t": thr,
            "l": self.build(order[:nl], depth + 1),
            "r": self.build(order[nl:], depth + 1),
        }


def _walk(node, row):
    while "f" in node:
        node = node["l"] if row[node["f"]] <= node["t"] else node["r"]
    return node


def _depth(node):
    return 0 if "f" not in node else 1 + max(_depth(node["l"]), _depth(node["r"]))


def _leaves(node):
    return 1 if "f" not in node else _leaves(node["l"]) + _leaves(node["r"])


def _normalise(v):
    s = math.fsum(v)
    return [x / s for x in v] if s > 0 else [0.0] * len(v)


def _opt_int(v, name):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or v < 1:
        raise SNRuntimeError(f"{name} must be a positive integer or nil")
    return int(v)


def _feature_names(X, d):
    if isinstance(X, DataFrame):
        names = [c for c in X.cols if X._is_numeric(c)]
        if len(names) == d:
            return names
    return [f"x{i}" for i in range(d)]


# --------------------------------------------------------------- base model
class _TreeModel(Model):
    def _common_init(self, task, criterion, max_depth, min_samples_split, min_samples_leaf):
        if task not in ("classification", "regression"):
            raise SNRuntimeError("task must be 'classification' or 'regression'")
        if criterion not in ("gini", "entropy"):
            raise SNRuntimeError("criterion must be 'gini' or 'entropy'")
        self.task, self.criterion = task, criterion
        self.max_depth = _opt_int(max_depth, "max_depth")
        self.min_samples_split = _opt_int(min_samples_split, "min_samples_split") or 2
        self.min_samples_leaf = _opt_int(min_samples_leaf, "min_samples_leaf") or 1
        self.fitted = False
        self.classes, self.names, self.n_features = [], [], 0

    def _prepare(self, X, y):
        Xt = _X(X)
        if self.task == "classification":
            labels = _labels(y)
            _check_len(Xt, labels)
            self.classes = _sorted_classes(labels)
            lookup = {c: i for i, c in enumerate(self.classes)}
            ys = [lookup[c] for c in labels]
        else:
            ys = [float(v) for v in _vals(y)]
            _check_len(Xt, ys)
        if Xt.shape[0] == 0:
            raise SNRuntimeError("cannot fit on an empty dataset")
        rows = Xt.to_nested()
        for r in rows:
            for v in r:
                if v != v:
                    raise SNRuntimeError("X contains nan; clean or fill missing values first")
        self.n_features = Xt.shape[1]
        self.names = _feature_names(X, self.n_features)
        return rows, ys

    def _rows(self, X):
        self._need_fit()
        Xt = _X(X)
        if Xt.shape[1] != self.n_features:
            raise SNRuntimeError(f"expected {self.n_features} features but got {Xt.shape[1]}")
        return Xt.to_nested()

    def _decode(self, idx):
        return self.classes[idx]

    def m_score(self, X, y):
        pred = self.m_predict(X)
        if self.task == "regression":
            return _r2([float(v) for v in _vals(y)], pred.data)
        return _accuracy(_labels(y), pred)

    @property
    def p_classes(self):
        self._need_fit()
        return list(self.classes) if self.task == "classification" else None

    @property
    def p_feature_names(self):
        self._need_fit()
        return list(self.names)

    def _need_proba(self):
        if self.task != "classification":
            raise SNRuntimeError("predict_proba() is only available for classification")


# ------------------------------------------------------------ decision tree
class DecisionTree(_TreeModel):
    """A single CART decision tree.

    DecisionTree(task="classification", max_depth=nil, min_samples_split=2,
                 min_samples_leaf=1, max_features=nil, criterion="gini", seed=nil)
    """

    sn_type = "DecisionTree"
    _repr_params = ("task", "max_depth", "min_samples_split", "min_samples_leaf")

    def __init__(self, task="classification", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, criterion="gini", seed=None):
        self._common_init(task, criterion, max_depth, min_samples_split, min_samples_leaf)
        _resolve_max_features(max_features, 1)          # validate early
        self.max_features, self.seed = max_features, seed
        self.root, self.importances = None, []

    def m_fit(self, X, y):
        rows, ys = self._prepare(X, y)
        b = _Builder(rows, ys, self.task, len(self.classes), self.max_depth,
                     self.min_samples_split, self.min_samples_leaf,
                     self.max_features, self.criterion, _rng(self.seed))
        self.root = b.build(list(range(len(rows))))
        self.importances = _normalise(b.importance)
        self.fitted = True
        return self

    def m_predict(self, X):
        rows = self._rows(X)
        leaves = [_walk(self.root, r) for r in rows]
        if self.task == "regression":
            return Tensor([lf["v"] for lf in leaves])
        return [self._decode(lf["v"]) for lf in leaves]

    def m_predict_proba(self, X):
        self._need_proba()
        rows = self._rows(X)
        data = []
        for r in rows:
            data.extend(_walk(self.root, r)["d"])
        return Tensor(data, (len(rows), len(self.classes)))

    def m_rules(self, feature_names=None):
        """The tree as readable if/else text."""
        self._need_fit()
        names = list(feature_names) if feature_names is not None else self.names
        if len(names) != self.n_features:
            raise SNRuntimeError(f"feature_names needs {self.n_features} names")
        out = []

        def leaf_text(node):
            if self.task == "classification":
                return to_str(self._decode(node["v"]))
            return fmt_number(node["v"], 6)

        def rec(node, ind):
            pad = "    " * ind
            if "f" not in node:
                out.append(f"{pad}-> {leaf_text(node)}  (n={node['n']})")
                return
            nm = to_str(names[node["f"]])
            out.append(f"{pad}if {nm} <= {fmt_number(node['t'], 6)}:")
            rec(node["l"], ind + 1)
            out.append(f"{pad}else:")
            rec(node["r"], ind + 1)

        rec(self.root, 0)
        return "\n".join(out)

    @property
    def p_feature_importances(self):
        self._need_fit()
        return Tensor(list(self.importances))

    @property
    def p_depth(self):
        self._need_fit()
        return _depth(self.root)

    @property
    def p_n_leaves(self):
        self._need_fit()
        return _leaves(self.root)

    def to_params(self):
        self._need_fit()
        return {"task": self.task, "criterion": self.criterion, "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "classes": self.classes, "names": self.names, "n_features": self.n_features,
                "importances": self.importances, "root": self.root}

    @classmethod
    def from_params(cls, p):
        m = cls(p["task"], p["max_depth"], p["min_samples_split"], p["min_samples_leaf"],
                criterion=p["criterion"])
        m.classes, m.names, m.n_features = p["classes"], p["names"], p["n_features"]
        m.importances, m.root, m.fitted = p["importances"], p["root"], True
        return m


# ------------------------------------------------------------ random forest
class RandomForest(_TreeModel):
    """An ensemble of decision trees trained on bootstrap samples.

    RandomForest(n_trees=50, task="classification", max_depth=nil,
                 min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                 criterion="gini", bootstrap=true, seed=nil)

    ``max_features`` defaults to "sqrt" for classification and to all features
    for regression.  After fitting, ``oob_score`` is the accuracy (or R^2) on
    the rows each tree did not see.
    """

    sn_type = "RandomForest"
    _repr_params = ("n_trees", "task", "max_depth", "max_features")

    def __init__(self, n_trees=50, task="classification", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="default", criterion="gini", bootstrap=True,
                 seed=None):
        self._common_init(task, criterion, max_depth, min_samples_split, min_samples_leaf)
        self.n_trees = _opt_int(n_trees, "n_trees")
        if self.n_trees is None:
            raise SNRuntimeError("n_trees must be a positive integer")
        if max_features == "default":
            max_features = "sqrt" if task == "classification" else None
        _resolve_max_features(max_features, 1)
        self.max_features, self.bootstrap, self.seed = max_features, bool(bootstrap), seed
        self.trees, self.importances, self.oob = [], [], None

    def m_fit(self, X, y):
        rows, ys = self._prepare(X, y)
        n, k = len(rows), len(self.classes)
        rng = _rng(self.seed)
        self.trees, imp = [], [0.0] * self.n_features
        oob_sum = [None] * n
        oob_cnt = [0] * n
        for _ in range(self.n_trees):
            if self.bootstrap:
                idx = [rng.randrange(n) for _ in range(n)]
            else:
                idx = list(range(n))
            b = _Builder(rows, ys, self.task, k, self.max_depth, self.min_samples_split,
                         self.min_samples_leaf, self.max_features, self.criterion,
                         _rng(rng.getrandbits(32)))
            root = b.build(idx)
            self.trees.append(root)
            for j, v in enumerate(_normalise(b.importance)):
                imp[j] += v
            if self.bootstrap:
                inbag = set(idx)
                for i in range(n):
                    if i in inbag:
                        continue
                    leaf = _walk(root, rows[i])
                    cur = leaf["d"] if self.task == "classification" else [leaf["v"]]
                    if oob_sum[i] is None:
                        oob_sum[i] = list(cur)
                    else:
                        oob_sum[i] = [a + c for a, c in zip(oob_sum[i], cur)]
                    oob_cnt[i] += 1
        self.importances = _normalise(imp)
        self.oob = None
        have = [i for i in range(n) if oob_cnt[i]]
        if have:
            if self.task == "classification":
                pred = [oob_sum[i].index(max(oob_sum[i])) for i in have]
                self.oob = sum(1 for i, p in zip(have, pred) if ys[i] == p) / len(have)
            else:
                self.oob = _r2([ys[i] for i in have], [oob_sum[i][0] / oob_cnt[i] for i in have])
        self.fitted = True
        return self

    def _proba_rows(self, rows):
        k = len(self.classes)
        out = []
        for r in rows:
            acc = [0.0] * k
            for t in self.trees:
                for c, p in enumerate(_walk(t, r)["d"]):
                    acc[c] += p
            out.append([a / len(self.trees) for a in acc])
        return out

    def m_predict(self, X):
        rows = self._rows(X)
        if self.task == "regression":
            return Tensor([math.fsum(_walk(t, r)["v"] for t in self.trees) / len(self.trees)
                           for r in rows])
        return [self._decode(p.index(max(p))) for p in self._proba_rows(rows)]

    def m_predict_proba(self, X):
        self._need_proba()
        rows = self._rows(X)
        data = [v for p in self._proba_rows(rows) for v in p]
        return Tensor(data, (len(rows), len(self.classes)))

    @property
    def p_feature_importances(self):
        self._need_fit()
        return Tensor(list(self.importances))

    @property
    def p_oob_score(self):
        self._need_fit()
        return self.oob

    @property
    def p_n_estimators(self):
        self._need_fit()
        return len(self.trees)

    def m_importance_table(self):
        """Feature importances as a list of [name, importance], largest first."""
        self._need_fit()
        pairs = sorted(zip(self.names, self.importances), key=lambda p: -p[1])
        return [[n, v] for n, v in pairs]

    def to_params(self):
        self._need_fit()
        return {"n_trees": self.n_trees, "task": self.task, "criterion": self.criterion,
                "max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features,
                "bootstrap": self.bootstrap, "classes": self.classes, "names": self.names,
                "n_features": self.n_features, "importances": self.importances,
                "oob": self.oob, "trees": self.trees}

    @classmethod
    def from_params(cls, p):
        m = cls(p["n_trees"], p["task"], p["max_depth"], p["min_samples_split"],
                p["min_samples_leaf"], p["max_features"], p["criterion"], p["bootstrap"])
        m.classes, m.names, m.n_features = p["classes"], p["names"], p["n_features"]
        m.importances, m.oob, m.trees, m.fitted = p["importances"], p["oob"], p["trees"], True
        return m


MODELS.update({c.sn_type: c for c in (DecisionTree, RandomForest)})
