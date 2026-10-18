"""A small column-oriented data frame (think: a tiny, dependency-free pandas)."""
import csv
import math
import random

from . import stats
from .errors import SNRuntimeError
from .tensor import RNG, Tensor
from .values import NativeObject, call_value, fmt_number, suggest, truthy, type_name

_MISSING = {"", "na", "nan", "null", "none", "n/a"}


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _infer_column(raw):
    vals = [None if (s is None or s.strip().lower() in _MISSING) else s.strip() for s in raw]
    present = [v for v in vals if v is not None]
    for conv in (int, float):
        try:
            conv_iter = iter([conv(v) for v in present])
        except ValueError:
            continue
        return [None if v is None else next(conv_iter) for v in vals]
    return vals


def _as_list(v, what="column"):
    if isinstance(v, Tensor):
        return list(v.data) if v.ndim == 1 else v.to_nested()
    if isinstance(v, list):
        return list(v)
    raise SNRuntimeError(f"a {what} must be a list or tensor, got {type_name(v)}")


class DataFrame(NativeObject):
    sn_type = "dataframe"

    def __init__(self, cols=None):
        self.cols = {}
        for k, v in (cols or {}).items():
            self.cols[str(k)] = _as_list(v)
        lens = {len(v) for v in self.cols.values()}
        if len(lens) > 1:
            raise SNRuntimeError("all columns of a DataFrame must have the same length")

    # ------------------------------------------------------------- helpers
    @property
    def nrows(self):
        return len(next(iter(self.cols.values()))) if self.cols else 0

    def _col(self, name):
        if name not in self.cols:
            raise SNRuntimeError(
                f"no column '{name}'. Columns: {list(self.cols)}." + suggest(str(name), self.cols)
            )
        return self.cols[name]

    def _is_numeric(self, name):
        vals = [v for v in self.cols[name] if v is not None]
        return bool(vals) and all(_is_num(v) for v in vals)

    def numeric_columns(self):
        return [c for c in self.cols if self._is_numeric(c)]

    def row(self, i):
        return {c: v[i] for c, v in self.cols.items()}

    def take_rows(self, idx):
        return DataFrame({c: [v[i] for i in idx] for c, v in self.cols.items()})

    def _names(self, cols):
        if len(cols) == 1 and isinstance(cols[0], list):
            cols = cols[0]
        for c in cols:
            self._col(c)
        return list(cols)

    # ------------------------------------------------------------ properties
    @property
    def p_columns(self):
        return list(self.cols)

    @property
    def p_shape(self):
        return [self.nrows, len(self.cols)]

    # --------------------------------------------------------------- methods
    def m_head(self, n=5):
        return self.take_rows(range(min(int(n), self.nrows)))

    def m_tail(self, n=5):
        n = min(int(n), self.nrows)
        return self.take_rows(range(self.nrows - n, self.nrows))

    def m_select(self, *cols):
        names = self._names(cols)
        return DataFrame({c: self.cols[c] for c in names})

    def m_drop(self, *cols):
        names = set(self._names(cols))
        return DataFrame({c: v for c, v in self.cols.items() if c not in names})

    def m_filter(self, fn):
        keep = [i for i in range(self.nrows) if truthy(call_value(fn, self.row(i)))]
        return self.take_rows(keep)

    def m_sort_by(self, col, desc=False):
        vals = self._col(col)
        present = [i for i, v in enumerate(vals) if v is not None]
        missing = [i for i, v in enumerate(vals) if v is None]
        present.sort(key=lambda i: vals[i], reverse=truthy(desc))
        return self.take_rows(present + missing)

    def m_add_column(self, name, values):
        out = dict(self.cols)
        n = self.nrows if self.cols else None
        if isinstance(values, (int, float, str)) or values is None:
            if n is None:
                raise SNRuntimeError("cannot broadcast a scalar into an empty DataFrame")
            vals = [values] * n
        elif isinstance(values, (list, Tensor)):
            vals = _as_list(values)
        else:
            vals = [call_value(values, self.row(i)) for i in range(self.nrows)]
        out[str(name)] = vals
        return DataFrame(out)

    def m_map_column(self, name, fn):
        out = dict(self.cols)
        out[name] = [call_value(fn, v) for v in self._col(name)]
        return DataFrame(out)

    def m_rename(self, old, new):
        self._col(old)
        return DataFrame({(new if c == old else c): v for c, v in self.cols.items()})

    def m_drop_na(self):
        keep = [i for i in range(self.nrows) if all(v[i] is not None for v in self.cols.values())]
        return self.take_rows(keep)

    def m_fill_na(self, value):
        return DataFrame({c: [value if x is None else x for x in v] for c, v in self.cols.items()})

    def m_col(self, name):
        return list(self._col(name))

    def m_rows(self):
        return [self.row(i) for i in range(self.nrows)]

    def m_copy(self):
        return DataFrame(self.cols)

    def m_unique(self, col):
        seen, out = set(), []
        for v in self._col(col):
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def m_value_counts(self, col):
        counts = {}
        for v in self._col(col):
            counts[v] = counts.get(v, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def m_sample(self, n, seed=None):
        rng = random.Random(seed) if seed is not None else RNG
        n = min(int(n), self.nrows)
        return self.take_rows(rng.sample(range(self.nrows), n))

    def m_to_tensor(self, cols=None):
        names = self.numeric_columns() if cols is None else self._names([cols])
        if not names:
            raise SNRuntimeError("the DataFrame has no numeric columns")
        for c in names:
            if not self._is_numeric(c):
                raise SNRuntimeError(f"column '{c}' is not numeric")
            if any(v is None for v in self.cols[c]):
                raise SNRuntimeError(f"column '{c}' has missing values; use drop_na() or fill_na()")
        data = [float(self.cols[c][i]) for i in range(self.nrows) for c in names]
        return Tensor(data, (self.nrows, len(names)))

    def m_describe(self):
        names = self.numeric_columns()
        labels = ["count", "mean", "std", "min", "median", "max"]
        out = {"stat": labels}
        for c in names:
            v = [float(x) for x in self.cols[c] if x is not None]
            out[c] = [
                float(len(v)),
                stats.mean(v) if v else math.nan,
                stats.std(v, 1) if len(v) > 1 else math.nan,
                min(v) if v else math.nan,
                stats.median(v) if v else math.nan,
                max(v) if v else math.nan,
            ]
        return DataFrame(out)

    def m_group_by(self, col, agg="mean"):
        funcs = {
            "mean": stats.mean, "sum": math.fsum, "min": min, "max": max,
            "median": stats.median, "std": lambda v: stats.std(v, 1) if len(v) > 1 else math.nan,
            "count": len,
        }
        if agg not in funcs:
            raise SNRuntimeError(f"unknown aggregation '{agg}'. Use one of {list(funcs)}")
        keys = self._col(col)
        groups = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        out = {col: list(groups)}
        if agg == "count":
            out["count"] = [len(ix) for ix in groups.values()]
            return DataFrame(out)
        for c in self.numeric_columns():
            if c == col:
                continue
            col_vals = []
            for ix in groups.values():
                v = [float(self.cols[c][i]) for i in ix if self.cols[c][i] is not None]
                col_vals.append(funcs[agg](v) if v else None)
            out[c] = col_vals
        return DataFrame(out)

    def m_corr(self):
        names = self.numeric_columns()
        data = {"column": names}
        for a in names:
            va = [float(x) for x in self.cols[a]]
            data[a] = []
            for b in names:
                vb = [float(x) for x in self.cols[b]]
                data[a].append(stats.pearson(va, vb))
        return DataFrame(data)

    def m_to_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.cols)
            for i in range(self.nrows):
                w.writerow(["" if v[i] is None else v[i] for v in self.cols.values()])
        return path

    # --------------------------------------------------------------- protocol
    def sn_getitem(self, key):
        if isinstance(key, str):
            vals = self._col(key)
            if self._is_numeric(key) and all(v is not None for v in vals):
                return Tensor([float(v) for v in vals])
            return list(vals)
        if isinstance(key, slice):
            return self.take_rows(range(*key.indices(self.nrows)))
        if isinstance(key, Tensor) and key.is_bool:
            if key.size != self.nrows:
                raise SNRuntimeError("mask length does not match the number of rows")
            return self.take_rows([i for i, m in enumerate(key.data) if m])
        if isinstance(key, list):
            if key and all(isinstance(k, str) for k in key):
                return self.m_select(key)
            if len(key) == self.nrows and all(isinstance(k, bool) for k in key):
                return self.take_rows([i for i, m in enumerate(key) if m])
            raise SNRuntimeError("index a DataFrame with a column name, list of names, row number or mask")
        if isinstance(key, (int, float)) and not isinstance(key, bool) and float(key).is_integer():
            i = int(key)
            if not -self.nrows <= i < self.nrows:
                raise SNRuntimeError(f"row {i} out of range ({self.nrows} rows)")
            return self.row(i % self.nrows)
        raise SNRuntimeError("index a DataFrame with a column name, list of names, row number or mask")

    def sn_setitem(self, key, value):
        if not isinstance(key, str):
            raise SNRuntimeError("assign to a DataFrame with a column name: df[\"name\"] = values")
        if isinstance(value, (list, Tensor)):
            vals = _as_list(value)
        else:
            vals = [value] * self.nrows
        if self.cols and len(vals) != self.nrows:
            raise SNRuntimeError(f"column has {len(vals)} values but the DataFrame has {self.nrows} rows")
        self.cols[key] = vals

    def sn_iter(self):
        return [self.row(i) for i in range(self.nrows)]

    def sn_len(self):
        return self.nrows

    def sn_contains(self, item):
        return item in self.cols

    def sn_truthy(self):
        return self.nrows > 0

    def sn_repr(self):
        if not self.cols:
            return "DataFrame(empty)"
        n = self.nrows
        names = list(self.cols)
        show = list(range(n)) if n <= 20 else list(range(10)) + [None] + list(range(n - 5, n))

        def cell(v):
            if v is None:
                return "nil"
            if isinstance(v, float):
                return fmt_number(v, 6)
            s = str(v)
            return s if len(s) <= 28 else s[:27] + "…"

        numeric = {c: self._is_numeric(c) for c in names}
        body = []
        for i in show:
            if i is None:
                body.append(["..."] * (len(names) + 1))
            else:
                body.append([str(i)] + [cell(self.cols[c][i]) for c in names])
        header = [""] + names
        widths = [max([len(header[j])] + [len(r[j]) for r in body]) for j in range(len(header))]

        def fmt_row(cells, is_header=False):
            out = []
            for j, text in enumerate(cells):
                right = j == 0 or numeric.get(header[j], False)
                out.append(text.rjust(widths[j]) if right else text.ljust(widths[j]))
            return "  ".join(out).rstrip()

        lines = [fmt_row(header, True)] + [fmt_row(r) for r in body]
        lines.append(f"[{n} rows x {len(names)} columns]")
        return "\n".join(lines)


def read_csv(path, sep=",", header=True):
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.reader(f, delimiter=sep) if any(c.strip() for c in r)]
    except FileNotFoundError:
        raise SNRuntimeError(f"file not found: {path}") from None
    if not rows:
        return DataFrame({})
    if header:
        names, rows = [h.strip() for h in rows[0]], rows[1:]
    else:
        names = [f"col{i}" for i in range(len(rows[0]))]
    width = len(names)
    cols = {}
    for j, name in enumerate(names):
        raw = [r[j] if j < len(r) else None for r in rows]
        cols[name or f"col{j}"] = _infer_column(raw)
    return DataFrame(cols)


def make_dataframe(data=None):
    """DataFrame({"a": [1, 2]})  or  DataFrame([{"a": 1}, {"a": 2}])."""
    if data is None:
        return DataFrame({})
    if isinstance(data, dict):
        return DataFrame(data)
    if isinstance(data, list):
        if not all(isinstance(r, dict) for r in data):
            raise SNRuntimeError("DataFrame(list) expects a list of maps (one per row)")
        names = []
        for r in data:
            for k in r:
                if k not in names:
                    names.append(k)
        return DataFrame({k: [r.get(k) for r in data] for k in names})
    raise SNRuntimeError("DataFrame() expects a map of columns or a list of row maps")
