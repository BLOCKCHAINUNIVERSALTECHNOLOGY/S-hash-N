"""More models: PCA, Gaussian naive Bayes and gradient boosting."""
import math

from . import stats
from .errors import SNRuntimeError
from .ml import (MODELS, Model, _X, _accuracy, _check_len, _finite, _labels, _r2, _rng,
                 _sorted_classes, _vals)
from .tensor import Tensor
from .trees import _Builder, _TreeModel, _normalise, _opt_int, _walk


# ------------------------------------------------------------------------ PCA
def _jacobi_eigh(A, sweeps=60):
    """Eigenvalues/vectors of a symmetric matrix (list of lists) by cyclic Jacobi rotations."""
    n = len(A)
    A = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = math.fsum(A[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
        if off < 1e-22:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-300:
                    continue
                theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [A[i][i] for i in range(n)], V


class PCA(Model):
    """Principal component analysis.

    PCA(n_components=2)    keep that many components
    PCA(n_components=0.95) keep enough components to explain 95% of the variance
    """

    sn_type = "PCA"
    _repr_params = ("n_components",)

    def __init__(self, n_components=2):
        if isinstance(n_components, bool) or not isinstance(n_components, (int, float)) or n_components <= 0:
            raise SNRuntimeError("n_components must be a positive integer or a fraction between 0 and 1")
        if isinstance(n_components, float) and not n_components.is_integer() and n_components >= 1:
            raise SNRuntimeError("n_components must be an integer, or a fraction below 1")
        self.n_components = n_components if (isinstance(n_components, float) and n_components < 1) \
            else int(n_components)
        self.fitted = False
        self.mean, self.components, self.variance, self.ratio = [], [], [], []

    def m_fit(self, X, y=None):
        Xt = _X(X)
        n, d = Xt.shape
        if n < 2:
            raise SNRuntimeError("PCA needs at least two samples")
        rows = Xt.to_nested()
        mean = [math.fsum(r[j] for r in rows) / n for j in range(d)]
        cen = [[r[j] - mean[j] for j in range(d)] for r in rows]
        cov = [[math.fsum(c[i] * c[j] for c in cen) / (n - 1) for j in range(d)] for i in range(d)]
        for i in range(d):                       # exact symmetry
            for j in range(i):
                cov[i][j] = cov[j][i] = (cov[i][j] + cov[j][i]) / 2.0
        vals, V = _jacobi_eigh(cov)
        order = sorted(range(d), key=lambda i: -vals[i])
        total = math.fsum(max(v, 0.0) for v in vals)
        comps, var = [], []
        for i in order:
            vec = [V[k][i] for k in range(d)]
            big = max(vec, key=abs)
            if big < 0:                           # deterministic sign
                vec = [-v for v in vec]
            comps.append(vec)
            var.append(max(vals[i], 0.0))
        ratio = [v / total for v in var] if total > 0 else [0.0] * d
        if isinstance(self.n_components, float):
            k, acc = 0, 0.0
            while k < d and acc < self.n_components - 1e-12:
                acc += ratio[k]
                k += 1
            k = max(k, 1)
        else:
            if self.n_components > d:
                raise SNRuntimeError(f"n_components ({self.n_components}) cannot exceed the number of features ({d})")
            k = self.n_components
        self.mean, self.components = mean, comps[:k]
        self.variance, self.ratio = var[:k], ratio[:k]
        self.n_features, self.fitted = d, True
        return self

    def m_transform(self, X):
        self._need_fit()
        Xt = _X(X)
        if Xt.shape[1] != self.n_features:
            raise SNRuntimeError(f"expected {self.n_features} features but got {Xt.shape[1]}")
        cen = Xt - Tensor(self.mean)
        return cen @ Tensor.from_value(self.components).transpose()

    def m_fit_transform(self, X, y=None):
        return self.m_fit(X).m_transform(X)

    def m_inverse_transform(self, Z):
        self._need_fit()
        Zt = _X(Z)
        if Zt.shape[1] != len(self.components):
            raise SNRuntimeError(f"expected {len(self.components)} components but got {Zt.shape[1]}")
        return (Zt @ Tensor.from_value(self.components)) + Tensor(self.mean)

    @property
    def p_components(self):
        self._need_fit()
        return Tensor.from_value(self.components)

    @property
    def p_explained_variance(self):
        self._need_fit()
        return Tensor(self.variance)

    @property
    def p_explained_variance_ratio(self):
        self._need_fit()
        return Tensor(self.ratio)

    @property
    def p_mean(self):
        self._need_fit()
        return Tensor(self.mean)

    def to_params(self):
        self._need_fit()
        return {"n_components": self.n_components, "mean": self.mean, "components": self.components,
                "variance": self.variance, "ratio": self.ratio, "n_features": self.n_features}

    @classmethod
    def from_params(cls, p):
        m = cls(p["n_components"])
        m.mean, m.components, m.variance, m.ratio = p["mean"], p["components"], p["variance"], p["ratio"]
        m.n_features, m.fitted = p["n_features"], True
        return m


# ------------------------------------------------------------ naive Bayes
class NaiveBayes(Model):
    """Gaussian naive Bayes classifier."""

    sn_type = "NaiveBayes"
    _repr_params = ("var_smoothing",)

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = float(var_smoothing)
        self.fitted = False
        self.classes, self.prior, self.mu, self.var = [], [], [], []

    def m_fit(self, X, y):
        Xt = _X(X)
        labels = _labels(y)
        _check_len(Xt, labels)
        self.classes = _sorted_classes(labels)
        if len(self.classes) < 2:
            raise SNRuntimeError("NaiveBayes needs at least two classes in y")
        rows = Xt.to_nested()
        n, d = len(rows), Xt.shape[1]
        floor = self.var_smoothing * max(
            max(stats.variance([r[j] for r in rows]) for j in range(d)), 1e-12)
        self.prior, self.mu, self.var = [], [], []
        for c in self.classes:
            sub = [r for r, l in zip(rows, labels) if l == c]
            self.prior.append(len(sub) / n)
            self.mu.append([math.fsum(r[j] for r in sub) / len(sub) for j in range(d)])
            self.var.append([
                math.fsum((r[j] - self.mu[-1][j]) ** 2 for r in sub) / len(sub) + floor
                for j in range(d)])
        self.n_features, self.fitted = d, True
        return self

    def _log_joint(self, rows):
        out = []
        for r in rows:
            scores = []
            for k in range(len(self.classes)):
                s = math.log(self.prior[k])
                for x, m, v in zip(r, self.mu[k], self.var[k]):
                    s -= 0.5 * (math.log(2.0 * math.pi * v) + (x - m) ** 2 / v)
                scores.append(s)
            out.append(scores)
        return out

    def _rows(self, X):
        self._need_fit()
        Xt = _X(X)
        if Xt.shape[1] != self.n_features:
            raise SNRuntimeError(f"expected {self.n_features} features but got {Xt.shape[1]}")
        return Xt.to_nested()

    def m_predict_proba(self, X):
        rows = self._rows(X)
        data = []
        for s in self._log_joint(rows):
            top = max(s)
            e = [math.exp(v - top) for v in s]
            z = math.fsum(e)
            data.extend(v / z for v in e)
        return Tensor(data, (len(rows), len(self.classes)))

    def m_predict(self, X):
        rows = self._rows(X)
        return [self.classes[s.index(max(s))] for s in self._log_joint(rows)]

    def m_score(self, X, y):
        return _accuracy(_labels(y), self.m_predict(X))

    @property
    def p_classes(self):
        self._need_fit()
        return list(self.classes)

    @property
    def p_means(self):
        self._need_fit()
        return Tensor.from_value(self.mu)

    def to_params(self):
        self._need_fit()
        return {"var_smoothing": self.var_smoothing, "classes": self.classes, "prior": self.prior,
                "mu": self.mu, "var": self.var, "n_features": self.n_features}

    @classmethod
    def from_params(cls, p):
        m = cls(p["var_smoothing"])
        m.classes, m.prior, m.mu, m.var = p["classes"], p["prior"], p["mu"], p["var"]
        m.n_features, m.fitted = p["n_features"], True
        return m


# -------------------------------------------------------- gradient boosting
def _softmax_rows(F):
    out = []
    for row in F:
        top = max(row)
        e = [math.exp(v - top) for v in row]
        z = math.fsum(e)
        out.append([v / z for v in e])
    return out


class GradientBoosting(_TreeModel):
    """Gradient boosted regression trees.

    GradientBoosting(n_trees=100, lr=0.1, max_depth=3, task="classification",
                     min_samples_split=2, min_samples_leaf=1, subsample=1.0, seed=nil)

    Regression minimises squared error; classification minimises the softmax
    cross-entropy (one tree per class per round).
    """

    sn_type = "GradientBoosting"
    _repr_params = ("n_trees", "lr", "max_depth", "task")

    def __init__(self, n_trees=100, lr=0.1, max_depth=3, task="classification",
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, seed=None):
        self._common_init(task, "gini", max_depth, min_samples_split, min_samples_leaf)
        self.n_trees = _opt_int(n_trees, "n_trees")
        if self.n_trees is None:
            raise SNRuntimeError("n_trees must be a positive integer")
        if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr <= 0:
            raise SNRuntimeError("lr must be a positive number")
        if not isinstance(subsample, (int, float)) or isinstance(subsample, bool) or not 0 < subsample <= 1:
            raise SNRuntimeError("subsample must be in (0, 1]")
        self.lr, self.subsample, self.seed = float(lr), float(subsample), seed
        self.init, self.rounds, self.importances = [], [], []

    def m_fit(self, X, y):
        rows, ys = self._prepare(X, y)
        n, d = len(rows), self.n_features
        rng = _rng(self.seed)
        k = len(self.classes) if self.task == "classification" else 1
        if self.task == "classification":
            if k < 2:
                raise SNRuntimeError("GradientBoosting needs at least two classes in y")
            counts = [0] * k
            for c in ys:
                counts[c] += 1
            self.init = [math.log(c / n) for c in counts]
            F = [list(self.init) for _ in range(n)]
        else:
            self.init = [math.fsum(ys) / n]
            F = [[self.init[0]] for _ in range(n)]
        self.rounds = []
        raw = [0.0] * d
        for _ in range(self.n_trees):
            if self.task == "classification":
                P = _softmax_rows(F)
                targets = [[(1.0 if ys[i] == c else 0.0) - P[i][c] for i in range(n)] for c in range(k)]
            else:
                targets = [[ys[i] - F[i][0] for i in range(n)]]
            if self.subsample < 1.0:
                idx = rng.sample(range(n), max(2, int(round(self.subsample * n))))
            else:
                idx = list(range(n))
            trees = []
            for c in range(k):
                b = _Builder(rows, targets[c], "regression", 0, self.max_depth,
                             self.min_samples_split, self.min_samples_leaf, None, "gini", rng)
                root = b.build(list(idx))
                trees.append(root)
                for j, v in enumerate(b.importance):
                    raw[j] += v
                for i in range(n):
                    F[i][c] += self.lr * _walk(root, rows[i])["v"]
            self.rounds.append(trees)
            if len(self.rounds) % 20 == 0:
                _finite(*(f for row in F[:50] for f in row))
        self.importances = _normalise(raw)
        self.fitted = True
        return self

    def _scores(self, rows):
        out = []
        for r in rows:
            s = list(self.init)
            for trees in self.rounds:
                for c, t in enumerate(trees):
                    s[c] += self.lr * _walk(t, r)["v"]
            out.append(s)
        return out

    def m_predict(self, X):
        rows = self._rows(X)
        scores = self._scores(rows)
        if self.task == "regression":
            return Tensor([s[0] for s in scores])
        return [self.classes[s.index(max(s))] for s in scores]

    def m_predict_proba(self, X):
        self._need_proba()
        rows = self._rows(X)
        P = _softmax_rows(self._scores(rows))
        return Tensor([v for p in P for v in p], (len(rows), len(self.classes)))

    @property
    def p_feature_importances(self):
        self._need_fit()
        return Tensor(list(self.importances))

    @property
    def p_n_estimators(self):
        self._need_fit()
        return len(self.rounds)

    def to_params(self):
        self._need_fit()
        return {"n_trees": self.n_trees, "lr": self.lr, "max_depth": self.max_depth, "task": self.task,
                "min_samples_split": self.min_samples_split, "min_samples_leaf": self.min_samples_leaf,
                "subsample": self.subsample, "classes": self.classes, "names": self.names,
                "n_features": self.n_features, "init": self.init, "rounds": self.rounds,
                "importances": self.importances}

    @classmethod
    def from_params(cls, p):
        m = cls(p["n_trees"], p["lr"], p["max_depth"], p["task"], p["min_samples_split"],
                p["min_samples_leaf"], p["subsample"])
        m.classes, m.names, m.n_features = p["classes"], p["names"], p["n_features"]
        m.init, m.rounds, m.importances, m.fitted = p["init"], p["rounds"], p["importances"], True
        return m


MODELS.update({c.sn_type: c for c in (PCA, NaiveBayes, GradientBoosting)})
