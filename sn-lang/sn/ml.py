"""Machine-learning toolkit: models, scalers, metrics and data helpers.

Everything is built on top of the pure-Python Tensor type, so it needs no
third-party packages. It is aimed at learning and small/medium datasets.
"""
import heapq
import json
import math
import operator
import random
import sys

from . import stats
from .dataframe import DataFrame
from .errors import SNRuntimeError
from . import tensor as _tensor
from .tensor import RNG, Tensor, exp, gauss_solve, log, matmul, relu, sigmoid, softmax, sqrt
from .values import NativeObject, fmt_number, type_name


# ------------------------------------------------------------------- helpers
def _rng(seed):
    return random.Random(seed if seed is not None else RNG.getrandbits(32))


def _X(v, name="X"):
    if isinstance(v, DataFrame):
        v = v.m_to_tensor()
    t = Tensor.from_value(v)
    if t.ndim == 1:
        t = Tensor(t.data, (t.size, 1))
    if t.ndim != 2:
        raise SNRuntimeError(f"{name} must be a 2-D tensor (one row per sample)")
    return t


def _vals(v):
    if isinstance(v, Tensor):
        return list(v.data)
    if isinstance(v, list):
        return list(v)
    raise SNRuntimeError(f"expected a list or tensor, got {type_name(v)}")


def _label(v):
    return int(v) if isinstance(v, float) and v.is_integer() else v


def _labels(v):
    return [_label(x) for x in _vals(v)]


def _sorted_classes(labels):
    uniq = list(dict.fromkeys(labels))
    try:
        return sorted(uniq)
    except TypeError:
        return sorted(uniq, key=str)


def _check_len(X, y):
    if X.shape[0] != len(y):
        raise SNRuntimeError(f"X has {X.shape[0]} rows but y has {len(y)} values")


def _finite(*tensors_or_numbers):
    for t in tensors_or_numbers:
        data = t.data if isinstance(t, Tensor) else [t]
        for x in data:
            if x != x or x in (math.inf, -math.inf):
                raise SNRuntimeError(
                    "training diverged (values became inf/nan). Try a smaller lr or scale your features "
                    "with StandardScaler()."
                )


def _r2(y, pred):
    mean = stats.mean(y)
    ss_tot = math.fsum((a - mean) ** 2 for a in y)
    ss_res = math.fsum((a - b) ** 2 for a, b in zip(y, pred))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else math.nan
    return 1.0 - ss_res / ss_tot


def _accuracy(y, pred):
    if len(y) != len(pred):
        raise SNRuntimeError("accuracy needs two sequences of equal length")
    if not y:
        raise SNRuntimeError("accuracy of an empty sequence")
    return sum(1 for a, b in zip(y, pred) if _label(a) == _label(b)) / len(y)


# --------------------------------------------------------------------- base
class Model(NativeObject):
    sn_type = "model"
    _repr_params = ()

    def _need_fit(self):
        if not getattr(self, "fitted", False):
            raise SNRuntimeError(f"this {self.sn_type} is not fitted yet; call fit(X, y) first")

    def sn_repr(self):
        parts = ", ".join(f"{k}={self._fmt(getattr(self, k))}" for k in self._repr_params)
        return f"{self.sn_type}({parts})"

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return fmt_number(v, 6)
        return str(v)

    def to_params(self):  # pragma: no cover - overridden
        raise SNRuntimeError(f"{self.sn_type} cannot be saved")

    def m_save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sn_model": self.sn_type, "params": self.to_params()}, f)
        return path


# ---------------------------------------------------------- linear regression
class LinearRegression(Model):
    sn_type = "LinearRegression"
    _repr_params = ("lr", "epochs", "l2", "fit_intercept")

    def __init__(self, lr=0.01, epochs=0, l2=0.0, fit_intercept=True):
        self.lr, self.epochs, self.l2, self.fit_intercept = lr, int(epochs), l2, bool(fit_intercept)
        self.w, self.b, self.fitted = None, 0.0, False

    def m_fit(self, X, y):
        X = _X(X)
        yv = [float(v) for v in _vals(y)]
        _check_len(X, yv)
        n, d = X.shape
        if self.epochs == 0:
            cols = [X.data[j::d] for j in range(d)]
            if self.fit_intercept:
                cols.append([1.0] * n)
            k = len(cols)
            A = [[0.0] * k for _ in range(k)]
            for i in range(k):
                for j in range(i, k):
                    A[i][j] = A[j][i] = math.fsum(map(operator.mul, cols[i], cols[j]))
            for i in range(d):
                A[i][i] += n * self.l2
            rhs = [[math.fsum(map(operator.mul, c, yv))] for c in cols]
            try:
                sol = [s[0] for s in gauss_solve(A, rhs)]
            except SNRuntimeError:
                raise SNRuntimeError(
                    "the features are linearly dependent, so the normal equations are singular; "
                    "try l2=0.01 or remove redundant columns"
                ) from None
            if self.fit_intercept:
                self.b, sol = sol[-1], sol[:-1]
            else:
                self.b = 0.0
            self.w = Tensor(sol, (d,))
        else:
            w = Tensor([0.0] * d, (d,))
            b = 0.0
            yt = Tensor(yv, (n,))
            Xt = X.transpose()
            for _ in range(self.epochs):
                err = (matmul(X, w) + b) - yt
                gw = matmul(Xt, err) / n
                if self.l2:
                    gw = gw + w * self.l2
                w = w - gw * self.lr
                if self.fit_intercept:
                    b -= self.lr * err.m_sum() / n
                if _ % 50 == 0:
                    _finite(w, b)
            _finite(w, b)
            self.w, self.b = w, b
        self.fitted = True
        return self

    def m_predict(self, X):
        self._need_fit()
        X = _X(X)
        if X.shape[1] != self.w.size:
            raise SNRuntimeError(f"expected {self.w.size} features but got {X.shape[1]}")
        return matmul(X, self.w) + self.b

    def m_score(self, X, y):
        return _r2([float(v) for v in _vals(y)], self.m_predict(X).data)

    @property
    def p_weights(self):
        self._need_fit()
        return self.w

    @property
    def p_bias(self):
        self._need_fit()
        return self.b

    def to_params(self):
        self._need_fit()
        return {"lr": self.lr, "epochs": self.epochs, "l2": self.l2,
                "fit_intercept": self.fit_intercept, "w": self.w.data, "b": self.b}

    @classmethod
    def from_params(cls, p):
        m = cls(p["lr"], p["epochs"], p["l2"], p["fit_intercept"])
        m.w, m.b, m.fitted = Tensor(list(p["w"])), p["b"], True
        return m


# -------------------------------------------------------- logistic regression
class LogisticRegression(Model):
    """Multinomial logistic (softmax) regression trained by gradient descent."""

    sn_type = "LogisticRegression"
    _repr_params = ("lr", "epochs", "l2")

    def __init__(self, lr=0.1, epochs=500, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, int(epochs), l2
        self.fitted = False
        self.classes, self.W, self.b = [], None, None

    def m_fit(self, X, y):
        X = _X(X)
        labels = _labels(y)
        _check_len(X, labels)
        n, d = X.shape
        self.classes = _sorted_classes(labels)
        k = len(self.classes)
        if k < 2:
            raise SNRuntimeError("LogisticRegression needs at least two classes in y")
        pos = {c: i for i, c in enumerate(self.classes)}
        Y = [0.0] * (n * k)
        for i, lab in enumerate(labels):
            Y[i * k + pos[lab]] = 1.0
        Y = Tensor(Y, (n, k))
        W = Tensor([0.0] * (d * k), (d, k))
        b = Tensor([0.0] * k, (k,))
        Xt = X.transpose()
        for epoch in range(self.epochs):
            P = softmax(matmul(X, W) + b, axis=1)
            G = P - Y
            gW = matmul(Xt, G) / n
            if self.l2:
                gW = gW + W * self.l2
            W = W - gW * self.lr
            b = b - G.m_mean(axis=0) * self.lr
            if epoch % 50 == 0:
                _finite(W)
        _finite(W, b)
        self.W, self.b, self.fitted = W, b, True
        return self

    def _proba(self, X):
        self._need_fit()
        X = _X(X)
        if X.shape[1] != self.W.shape[0]:
            raise SNRuntimeError(f"expected {self.W.shape[0]} features but got {X.shape[1]}")
        return softmax(matmul(X, self.W) + self.b, axis=1)

    def m_predict_proba(self, X):
        return self._proba(X)

    def m_predict(self, X):
        P = self._proba(X)
        return [self.classes[int(i)] for i in P.m_argmax(axis=1).data]

    def m_score(self, X, y):
        return _accuracy(_labels(y), self.m_predict(X))

    @property
    def p_classes(self):
        self._need_fit()
        return list(self.classes)

    @property
    def p_weights(self):
        self._need_fit()
        return self.W

    def to_params(self):
        self._need_fit()
        return {"lr": self.lr, "epochs": self.epochs, "l2": self.l2, "classes": self.classes,
                "W": self.W.to_nested(), "b": self.b.data}

    @classmethod
    def from_params(cls, p):
        m = cls(p["lr"], p["epochs"], p["l2"])
        m.classes, m.W, m.b = p["classes"], Tensor.from_value(p["W"]), Tensor(list(p["b"]))
        m.fitted = True
        return m


# ------------------------------------------------------------------------ KNN
class KNN(Model):
    sn_type = "KNN"
    _repr_params = ("k", "task")

    def __init__(self, k=3, task="classification"):
        if task not in ("classification", "regression"):
            raise SNRuntimeError("task must be 'classification' or 'regression'")
        self.k, self.task, self.fitted = int(k), task, False
        self.rows, self.targets = [], []

    def m_fit(self, X, y):
        X = _X(X)
        targets = _labels(y) if self.task == "classification" else [float(v) for v in _vals(y)]
        _check_len(X, targets)
        if self.k < 1 or self.k > X.shape[0]:
            raise SNRuntimeError(f"k must be between 1 and the number of samples ({X.shape[0]})")
        self.rows = X.to_nested()
        self.targets = targets
        self.fitted = True
        return self

    def _predict_row(self, r):
        near = heapq.nsmallest(
            self.k,
            ((math.fsum((a - b) ** 2 for a, b in zip(r, row)), i) for i, row in enumerate(self.rows)),
        )
        if self.task == "regression":
            return stats.mean([self.targets[i] for _, i in near])
        votes = {}
        for rank, (_, i) in enumerate(near):
            c, first = votes.get(self.targets[i], (0, rank))
            votes[self.targets[i]] = (c + 1, first)
        return max(votes.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]

    def _predict_numpy(self, queries):
        np = _tensor._np
        T = np.array(self.rows, dtype=float)
        Q = np.array(queries, dtype=float)
        D = (Q * Q).sum(1)[:, None] + (T * T).sum(1)[None, :] - 2.0 * (Q @ T.T)
        order = np.argsort(D, axis=1, kind="stable")[:, :self.k]
        out = []
        for neighbours in order.tolist():
            if self.task == "regression":
                out.append(stats.mean([self.targets[i] for i in neighbours]))
                continue
            votes = {}
            for rank, i in enumerate(neighbours):
                c, first = votes.get(self.targets[i], (0, rank))
                votes[self.targets[i]] = (c + 1, first)
            out.append(max(votes.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0])
        return out

    def m_predict(self, X):
        self._need_fit()
        X = _X(X)
        if self.rows and X.shape[1] != len(self.rows[0]):
            raise SNRuntimeError(f"expected {len(self.rows[0])} features but got {X.shape[1]}")
        if _tensor._np is not None and X.shape[0] * len(self.rows) >= 2000:
            out = self._predict_numpy(X.to_nested())
        else:
            out = [self._predict_row(r) for r in X.to_nested()]
        return Tensor(out) if self.task == "regression" else out

    def m_score(self, X, y):
        pred = self.m_predict(X)
        if self.task == "regression":
            return _r2([float(v) for v in _vals(y)], pred.data)
        return _accuracy(_labels(y), pred)

    def to_params(self):
        self._need_fit()
        return {"k": self.k, "task": self.task, "rows": self.rows, "targets": self.targets}

    @classmethod
    def from_params(cls, p):
        m = cls(p["k"], p["task"])
        m.rows, m.targets, m.fitted = p["rows"], p["targets"], True
        return m


# --------------------------------------------------------------------- KMeans
class KMeans(Model):
    sn_type = "KMeans"
    _repr_params = ("k", "iters")

    def __init__(self, k=3, iters=100, seed=None, n_init=3):
        self.k, self.iters, self.seed, self.n_init = int(k), int(iters), seed, int(n_init)
        self.fitted = False
        self.centers, self._labels_, self.inertia = [], [], math.nan

    @staticmethod
    def _d2(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def _init_centers(self, rows, rng):
        centers = [rows[rng.randrange(len(rows))]]
        while len(centers) < self.k:
            d2 = [min(self._d2(r, c) for c in centers) for r in rows]
            total = sum(d2)
            if total == 0:
                centers.append(rows[rng.randrange(len(rows))])
                continue
            target = rng.random() * total
            acc = 0.0
            for r, d in zip(rows, d2):
                acc += d
                if acc >= target:
                    centers.append(r)
                    break
            else:
                centers.append(rows[-1])
        return [list(c) for c in centers]

    def _assign(self, rows, centers):
        if _tensor._np is not None and len(rows) * len(centers) >= 500:
            np = _tensor._np
            R = np.array(rows, dtype=float)
            C = np.array(centers, dtype=float)
            D = (R * R).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * (R @ C.T)
            D = np.maximum(D, 0.0)
            labels = D.argmin(axis=1)
            return labels.tolist(), float(D[np.arange(len(rows)), labels].sum())
        labels, inertia = [], 0.0
        for r in rows:
            best, bd = 0, math.inf
            for j, c in enumerate(centers):
                d = self._d2(r, c)
                if d < bd:
                    best, bd = j, d
            labels.append(best)
            inertia += bd
        return labels, inertia

    def m_fit(self, X, y=None):
        X = _X(X)
        rows = X.to_nested()
        if self.k < 1 or self.k > len(rows):
            raise SNRuntimeError(f"k must be between 1 and the number of samples ({len(rows)})")
        rng = _rng(self.seed)
        d = X.shape[1]
        best = None
        for _ in range(max(1, self.n_init)):
            centers = self._init_centers(rows, rng)
            labels = None
            for _it in range(self.iters):
                new_labels, inertia = self._assign(rows, centers)
                if new_labels == labels:
                    break
                labels = new_labels
                for j in range(self.k):
                    members = [r for r, l in zip(rows, labels) if l == j]
                    if members:
                        centers[j] = [math.fsum(col) / len(members) for col in zip(*members)]
                    else:
                        centers[j] = list(rows[rng.randrange(len(rows))])
            labels, inertia = self._assign(rows, centers)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia)
        self.centers, self._labels_, self.inertia = best
        self.fitted = True
        return self

    def m_predict(self, X):
        self._need_fit()
        X = _X(X)
        return self._assign(X.to_nested(), self.centers)[0]

    def m_fit_predict(self, X):
        self.m_fit(X)
        return list(self._labels_)

    @property
    def p_centroids(self):
        self._need_fit()
        return Tensor.from_value(self.centers)

    @property
    def p_labels(self):
        self._need_fit()
        return list(self._labels_)

    @property
    def p_inertia(self):
        self._need_fit()
        return self.inertia

    def to_params(self):
        self._need_fit()
        return {"k": self.k, "iters": self.iters, "n_init": self.n_init,
                "centers": self.centers, "inertia": self.inertia}

    @classmethod
    def from_params(cls, p):
        m = cls(p["k"], p["iters"], None, p["n_init"])
        m.centers, m.inertia, m.fitted = p["centers"], p["inertia"], True
        return m


# ----------------------------------------------------------------- NeuralNet
_ACTIVATIONS = {
    "relu": (lambda Z: Z.map(relu), lambda A: A.map(lambda a: 1.0 if a > 0 else 0.0)),
    "tanh": (lambda Z: Z.map(math.tanh), lambda A: A.map(lambda a: 1.0 - a * a)),
    "sigmoid": (lambda Z: Z.map(sigmoid), lambda A: A.map(lambda a: a * (1.0 - a))),
    "linear": (lambda Z: Z, lambda A: A.map(lambda a: 1.0)),
}


class NeuralNet(Model):
    """A fully connected feed-forward network trained with mini-batch Adam/SGD.

    ``layers`` lists the sizes of every layer *including* input and output,
    e.g. ``[4, 16, 3]`` = 4 features, one hidden layer of 16, 3 classes.
    Calling ``fit`` repeatedly keeps training from the current weights.
    """

    sn_type = "NeuralNet"
    _repr_params = ("layers", "activation", "task", "optimizer")

    def __init__(self, layers, activation="relu", task="classification", lr=0.01, epochs=200,
                 batch_size=32, l2=0.0, optimizer="adam", seed=None, verbose=0):
        if not isinstance(layers, list) or len(layers) < 2 or not all(
            isinstance(n, (int, float)) and int(n) == n and n >= 1 for n in layers
        ):
            raise SNRuntimeError("layers must be a list of positive integers like [4, 8, 3]")
        if activation not in _ACTIVATIONS:
            raise SNRuntimeError(f"unknown activation '{activation}'. Use one of {list(_ACTIVATIONS)}")
        if task not in ("classification", "regression"):
            raise SNRuntimeError("task must be 'classification' or 'regression'")
        if optimizer not in ("adam", "sgd"):
            raise SNRuntimeError("optimizer must be 'adam' or 'sgd'")
        self.layers = [int(n) for n in layers]
        self.activation, self.task, self.lr = activation, task, lr
        self.epochs, self.batch_size, self.l2 = int(epochs), int(batch_size), l2
        self.optimizer, self.seed, self.verbose = optimizer, seed, int(verbose)
        self.history = []
        self.classes = []
        self.fitted = False
        self._init_weights()

    def _init_weights(self):
        rng = _rng(self.seed)
        self.rng = rng
        self.W, self.b = [], []
        for n_in, n_out in zip(self.layers, self.layers[1:]):
            scale = math.sqrt((2.0 if self.activation == "relu" else 1.0) / n_in)
            self.W.append(Tensor([rng.gauss(0.0, scale) for _ in range(n_in * n_out)], (n_in, n_out)))
            self.b.append(Tensor([0.0] * n_out, (n_out,)))
        self.opt_t = 0
        self.opt_m = [Tensor([0.0] * p.size, p.shape) for p in self.W + self.b]
        self.opt_v = [Tensor([0.0] * p.size, p.shape) for p in self.W + self.b]
        self.history = []

    def m_reset(self):
        self._init_weights()
        self.fitted = False
        return self

    # ---- core maths
    def _forward(self, X):
        act = _ACTIVATIONS[self.activation][0]
        acts = [X]
        A = X
        last = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = matmul(A, W) + b
            A = act(Z) if l < last else Z
            acts.append(A)
        return acts

    def _loss_grad(self, out, target):
        m, k = out.shape
        if self.task == "classification":
            P = softmax(out, axis=1)
            loss = 0.0
            dz = list(P.data)
            for i, c in enumerate(target):
                loss -= log(max(P.data[i * k + c], 1e-12))
                dz[i * k + c] -= 1.0
            return loss / m, Tensor(dz, (m, k)) / m
        diff = out - target
        loss = math.fsum(d * d for d in diff.data) / diff.size
        return loss, diff * (2.0 / diff.size)

    def _backward(self, acts, dZ):
        deriv = _ACTIVATIONS[self.activation][1]
        nl = len(self.W)
        gW, gb = [None] * nl, [None] * nl
        for l in range(nl - 1, -1, -1):
            g = matmul(acts[l].transpose(), dZ)
            if self.l2:
                g = g + self.W[l] * self.l2
            gW[l] = g
            gb[l] = dZ.m_sum(axis=0)
            if l > 0:
                dZ = matmul(dZ, self.W[l].transpose()) * deriv(acts[l])
        return gW + gb

    def _step(self, grads):
        params = self.W + self.b
        if self.optimizer == "sgd":
            new = [p - g * self.lr for p, g in zip(params, grads)]
        else:
            self.opt_t += 1
            t, b1, b2 = self.opt_t, 0.9, 0.999
            new = []
            for i, (p, g) in enumerate(zip(params, grads)):
                m = self.opt_m[i] * b1 + g * (1 - b1)
                v = self.opt_v[i] * b2 + (g * g) * (1 - b2)
                self.opt_m[i], self.opt_v[i] = m, v
                mh = m / (1 - b1 ** t)
                vh = v / (1 - b2 ** t)
                new.append(p - (mh * self.lr) / (vh.map(sqrt) + 1e-8))
        n = len(self.W)
        self.W, self.b = new[:n], new[n:]

    # ---- public API
    def m_fit(self, X, y):
        X = _X(X)
        n, d = X.shape
        if d != self.layers[0]:
            raise SNRuntimeError(f"layers[0] is {self.layers[0]} but X has {d} features")
        if self.task == "classification":
            labels = _labels(y)
            _check_len(X, labels)
            classes = _sorted_classes(labels)
            if self.fitted and any(c not in self.classes for c in classes):
                raise SNRuntimeError("y contains classes that were not seen in the first fit")
            if not self.fitted:
                self.classes = classes
            if len(self.classes) != self.layers[-1]:
                raise SNRuntimeError(
                    f"layers[-1] is {self.layers[-1]} but y has {len(self.classes)} classes"
                )
            pos = {c: i for i, c in enumerate(self.classes)}
            targets = [pos[c] for c in labels]
        else:
            Y = Tensor.from_value(y)
            if Y.ndim == 1:
                Y = Tensor(Y.data, (Y.size, 1))
            if Y.ndim != 2 or Y.shape[0] != n or Y.shape[1] != self.layers[-1]:
                raise SNRuntimeError(
                    f"y must have shape [{n}] or [{n}, {self.layers[-1]}] to match layers[-1]"
                )
            targets = Y
        bs = n if self.batch_size <= 0 else min(self.batch_size, n)
        for epoch in range(self.epochs):
            order = list(range(n))
            self.rng.shuffle(order)
            total = 0.0
            for s in range(0, n, bs):
                idx = order[s:s + bs]
                Xb = X.take(idx)
                tb = [targets[i] for i in idx] if self.task == "classification" else targets.take(idx)
                acts = self._forward(Xb)
                loss, dZ = self._loss_grad(acts[-1], tb)
                self._step(self._backward(acts, dZ))
                total += loss * len(idx)
            epoch_loss = total / n
            if epoch_loss != epoch_loss or epoch_loss == math.inf:
                raise SNRuntimeError("training diverged (loss is inf/nan). Try a smaller lr or scale your features.")
            self.history.append(epoch_loss)
            if self.verbose and (epoch + 1) % self.verbose == 0:
                sys.stdout.write(f"epoch {epoch + 1}/{self.epochs}  loss={epoch_loss:.6f}\n")
        self.fitted = True
        return self

    def _out(self, X):
        self._need_fit()
        X = _X(X)
        if X.shape[1] != self.layers[0]:
            raise SNRuntimeError(f"expected {self.layers[0]} features but got {X.shape[1]}")
        return self._forward(X)[-1]

    def m_predict_proba(self, X):
        if self.task != "classification":
            raise SNRuntimeError("predict_proba is only available for classification")
        return softmax(self._out(X), axis=1)

    def m_predict(self, X):
        out = self._out(X)
        if self.task == "classification":
            return [self.classes[int(i)] for i in out.m_argmax(axis=1).data]
        return Tensor(out.data) if out.shape[1] == 1 else out

    def m_score(self, X, y):
        pred = self.m_predict(X)
        if self.task == "classification":
            return _accuracy(_labels(y), pred)
        yt = Tensor.from_value(y)
        return _r2(list(yt.data), list(pred.data))

    @property
    def p_history(self):
        return list(self.history)

    @property
    def p_layers(self):
        return list(self.layers)

    @property
    def p_classes(self):
        return list(self.classes)

    def to_params(self):
        self._need_fit()
        return {"layers": self.layers, "activation": self.activation, "task": self.task,
                "lr": self.lr, "classes": self.classes,
                "W": [w.to_nested() for w in self.W], "b": [b.data for b in self.b]}

    @classmethod
    def from_params(cls, p):
        m = cls(p["layers"], p["activation"], p["task"], p["lr"])
        m.classes = p["classes"]
        m.W = [Tensor.from_value(w) for w in p["W"]]
        m.b = [Tensor(list(b)) for b in p["b"]]
        m.fitted = True
        return m


# ------------------------------------------------------------------- scalers
class StandardScaler(Model):
    """Rescale every column to zero mean and unit variance."""

    sn_type = "StandardScaler"

    def __init__(self):
        self.fitted = False
        self.mean, self.scale = [], []

    def m_fit(self, X):
        X = _X(X)
        cols = [X.data[j::X.shape[1]] for j in range(X.shape[1])]
        self.mean = [stats.mean(c) for c in cols]
        self.scale = [stats.std(c) or 1.0 for c in cols]
        self.fitted = True
        return self

    def m_transform(self, X):
        self._need_fit()
        X = _X(X)
        return (X - Tensor(self.mean)) / Tensor(self.scale)

    def m_fit_transform(self, X):
        return self.m_fit(X).m_transform(X)

    def m_inverse_transform(self, X):
        self._need_fit()
        return _X(X) * Tensor(self.scale) + Tensor(self.mean)

    @property
    def p_mean(self):
        self._need_fit()
        return Tensor(self.mean)

    @property
    def p_std(self):
        self._need_fit()
        return Tensor(self.scale)

    def to_params(self):
        self._need_fit()
        return {"mean": self.mean, "scale": self.scale}

    @classmethod
    def from_params(cls, p):
        m = cls()
        m.mean, m.scale, m.fitted = p["mean"], p["scale"], True
        return m


class MinMaxScaler(Model):
    """Rescale every column to the range [0, 1]."""

    sn_type = "MinMaxScaler"

    def __init__(self):
        self.fitted = False
        self.lo, self.span = [], []

    def m_fit(self, X):
        X = _X(X)
        cols = [X.data[j::X.shape[1]] for j in range(X.shape[1])]
        self.lo = [min(c) for c in cols]
        self.span = [(max(c) - min(c)) or 1.0 for c in cols]
        self.fitted = True
        return self

    def m_transform(self, X):
        self._need_fit()
        return (_X(X) - Tensor(self.lo)) / Tensor(self.span)

    def m_fit_transform(self, X):
        return self.m_fit(X).m_transform(X)

    def m_inverse_transform(self, X):
        self._need_fit()
        return _X(X) * Tensor(self.span) + Tensor(self.lo)

    def to_params(self):
        self._need_fit()
        return {"lo": self.lo, "span": self.span}

    @classmethod
    def from_params(cls, p):
        m = cls()
        m.lo, m.span, m.fitted = p["lo"], p["span"], True
        return m


MODELS = {c.sn_type: c for c in (
    LinearRegression, LogisticRegression, KNN, KMeans, NeuralNet, StandardScaler, MinMaxScaler)}


def load_model(path):
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except FileNotFoundError:
        raise SNRuntimeError(f"file not found: {path}") from None
    kind = blob.get("sn_model") if isinstance(blob, dict) else None
    if kind not in MODELS:
        raise SNRuntimeError(f"'{path}' is not a saved S#N model")
    return MODELS[kind].from_params(blob["params"])


# ------------------------------------------------------------------- metrics
def accuracy(y_true, y_pred):
    return _accuracy(_vals(y_true), _vals(y_pred))


def mse(y_true, y_pred):
    a, b = _vals(y_true), _vals(y_pred)
    if len(a) != len(b) or not a:
        raise SNRuntimeError("mse needs two non-empty sequences of equal length")
    return math.fsum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    a, b = _vals(y_true), _vals(y_pred)
    if len(a) != len(b) or not a:
        raise SNRuntimeError("mae needs two non-empty sequences of equal length")
    return math.fsum(abs(x - y) for x, y in zip(a, b)) / len(a)


def r2(y_true, y_pred):
    a, b = _vals(y_true), _vals(y_pred)
    if len(a) != len(b) or not a:
        raise SNRuntimeError("r2 needs two non-empty sequences of equal length")
    return _r2(a, b)


def confusion_matrix(y_true, y_pred):
    a, b = _labels(y_true), _labels(y_pred)
    if len(a) != len(b):
        raise SNRuntimeError("confusion_matrix needs sequences of equal length")
    classes = _sorted_classes(a + b)
    pos = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    for t, p in zip(a, b):
        m[pos[t]][pos[p]] += 1
    return {"labels": classes, "matrix": m}


def _prf(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    m, out = cm["matrix"], []
    for i in range(len(m)):
        tp = m[i][i]
        fp = sum(m[r][i] for r in range(len(m))) - tp
        fn = sum(m[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append((p, r, 2 * p * r / (p + r) if p + r else 0.0))
    return out


def precision_score(y_true, y_pred):
    return stats.mean([x[0] for x in _prf(y_true, y_pred)])


def recall_score(y_true, y_pred):
    return stats.mean([x[1] for x in _prf(y_true, y_pred)])


def f1_score(y_true, y_pred):
    return stats.mean([x[2] for x in _prf(y_true, y_pred)])


# ------------------------------------------------------------- data helpers
def _take(v, idx):
    if isinstance(v, Tensor):
        return v.take(idx)
    if isinstance(v, DataFrame):
        return v.take_rows(idx)
    if isinstance(v, list):
        return [v[i] for i in idx]
    raise SNRuntimeError(f"cannot split a {type_name(v)}")


def _nrows(v):
    if isinstance(v, Tensor):
        return v.shape[0]
    if isinstance(v, DataFrame):
        return v.nrows
    if isinstance(v, list):
        return len(v)
    raise SNRuntimeError(f"cannot split a {type_name(v)}")


def train_test_split(X, y=None, test_size=0.2, seed=None, shuffle=True):
    """Returns [X_train, X_test, y_train, y_test] (or [train, test] when y is nil)."""
    n = _nrows(X)
    if y is not None and _nrows(y) != n:
        raise SNRuntimeError("X and y must have the same number of rows")
    n_test = int(round(n * test_size)) if 0 < test_size < 1 else int(test_size)
    if not 1 <= n_test < n:
        raise SNRuntimeError("test_size leaves no data for training or testing")
    idx = list(range(n))
    if shuffle:
        _rng(seed).shuffle(idx)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    out = [_take(X, train_idx), _take(X, test_idx)]
    if y is not None:
        out += [_take(y, train_idx), _take(y, test_idx)]
    return out


def label_encode(values):
    vals = [_label(v) for v in _vals(values)]
    classes = _sorted_classes(vals)
    pos = {c: i for i, c in enumerate(classes)}
    return {"codes": [pos[v] for v in vals], "classes": classes}


def one_hot(labels, k=None):
    codes = [int(v) for v in _vals(labels)]
    k = int(k) if k is not None else (max(codes) + 1 if codes else 0)
    data = [0.0] * (len(codes) * k)
    for i, c in enumerate(codes):
        if not 0 <= c < k:
            raise SNRuntimeError(f"label {c} is outside 0..{k - 1}")
        data[i * k + c] = 1.0
    return Tensor(data, (len(codes), k))
