"""Charts rendered as standalone SVG (no dependencies).

    let fig = Plot(title="Loss", xlabel="epoch", ylabel="loss")
    fig.line(range(len(h)), h, label="train")
    fig.save("loss.svg")        # or "loss.html"
"""
import math
import os
import tempfile
from xml.sax.saxutils import escape

from .errors import SNRuntimeError
from .tensor import Tensor
from .values import NativeObject, type_name

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#8C8C00", "#555555"]
INK, MUTED, GRID, BG = "#222222", "#6b6b6b", "#e4e4e4", "#ffffff"


def _nums(v, what="values"):
    if isinstance(v, Tensor):
        return list(v.data)
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise SNRuntimeError(f"chart {what} must be numbers, found {type_name(x)}")
            out.append(float(x))
        return out
    raise SNRuntimeError(f"chart {what} must be a list or tensor, got {type_name(v)}")


def _finite(x):
    return x == x and x not in (math.inf, -math.inf)


def nice_ticks(lo, hi, n=6):
    if not _finite(lo) or not _finite(hi):
        return [0.0, 1.0]
    if lo == hi:
        lo, hi = (lo - 1, hi + 1) if lo == 0 else (lo - abs(lo) * 0.1, hi + abs(hi) * 0.1)
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag + 1e-12)
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step
    count = int(round((end - start) / step))
    return [round(start + i * step, 12) for i in range(count + 1)]


def _fmt(v):
    if abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.3g}"
    return f"{v:.6g}"


class Plot(NativeObject):
    sn_type = "Plot"

    def __init__(self, title="", xlabel="", ylabel="", width=720, height=440):
        self.title, self.xlabel, self.ylabel = str(title), str(xlabel), str(ylabel)
        self.width, self.height = int(width), int(height)
        self.series = []
        self.heat = None

    # ------------------------------------------------------------ building
    def _color(self, color):
        return str(color) if color else PALETTE[len(self.series) % len(PALETTE)]

    def _check_kind(self, kind):
        if self.heat is not None:
            raise SNRuntimeError("a heatmap cannot be combined with other series")
        cats = any(s["kind"] == "bar" for s in self.series)
        if (kind == "bar") != cats and self.series:
            raise SNRuntimeError("bar charts cannot be mixed with line/scatter/histogram series")
        if kind == "hist" and any(s["kind"] != "hist" for s in self.series):
            raise SNRuntimeError("histograms cannot be mixed with other series")

    def m_line(self, xs, ys=None, label=None, color=None):
        self._check_kind("line")
        if ys is None:
            ys = xs
            xs = list(range(len(_nums(ys))))
        x, y = _nums(xs, "x values"), _nums(ys, "y values")
        if len(x) != len(y):
            raise SNRuntimeError(f"line(): x has {len(x)} values but y has {len(y)}")
        self.series.append({"kind": "line", "x": x, "y": y, "label": label, "color": self._color(color)})
        return self

    def m_scatter(self, xs, ys, label=None, color=None):
        self._check_kind("scatter")
        x, y = _nums(xs, "x values"), _nums(ys, "y values")
        if len(x) != len(y):
            raise SNRuntimeError(f"scatter(): x has {len(x)} values but y has {len(y)}")
        self.series.append({"kind": "scatter", "x": x, "y": y, "label": label, "color": self._color(color)})
        return self

    def m_bar(self, labels, values, label=None, color=None):
        self._check_kind("bar")
        if not isinstance(labels, list):
            raise SNRuntimeError("bar(): labels must be a list")
        v = _nums(values)
        if len(labels) != len(v):
            raise SNRuntimeError(f"bar(): {len(labels)} labels but {len(v)} values")
        first = next((s for s in self.series), None)
        if first and first["cats"] != [str(c) for c in labels]:
            raise SNRuntimeError("bar(): every bar series must use the same labels")
        self.series.append({"kind": "bar", "cats": [str(c) for c in labels], "y": v,
                            "label": label, "color": self._color(color)})
        return self

    def m_hist(self, values, bins=10, label=None, color=None):
        self._check_kind("hist")
        v = [x for x in _nums(values) if _finite(x)]
        if not v:
            raise SNRuntimeError("hist(): no finite values")
        bins = max(1, int(bins))
        lo, hi = min(v), max(v)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        width = (hi - lo) / bins
        counts = [0] * bins
        for x in v:
            counts[min(int((x - lo) / width), bins - 1)] += 1
        self.series.append({"kind": "hist", "lo": lo, "width": width, "y": [float(c) for c in counts],
                            "label": label, "color": self._color(color)})
        return self

    def m_heatmap(self, matrix, x_labels=None, y_labels=None):
        if self.series:
            raise SNRuntimeError("a heatmap cannot be combined with other series")
        rows = matrix.to_nested() if isinstance(matrix, Tensor) else matrix
        if not isinstance(rows, list) or not rows or not all(isinstance(r, list) for r in rows):
            raise SNRuntimeError("heatmap(): matrix must be a non-empty list of rows")
        rows = [_nums(r) for r in rows]
        if len({len(r) for r in rows}) != 1:
            raise SNRuntimeError("heatmap(): all rows must have the same length")
        xl = [str(v) for v in (x_labels or range(len(rows[0])))]
        yl = [str(v) for v in (y_labels or range(len(rows)))]
        if len(xl) != len(rows[0]) or len(yl) != len(rows):
            raise SNRuntimeError("heatmap(): label counts do not match the matrix shape")
        self.heat = {"rows": rows, "x": xl, "y": yl}
        return self

    # ------------------------------------------------------------ rendering
    def _svg_text(self, x, y, text, size=12, anchor="start", fill=INK, weight="normal", rotate=None):
        tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
        return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
                f'fill="{fill}" font-weight="{weight}"{tr}>{escape(text)}</text>')

    def m_svg(self):
        return self.render()

    def render(self):
        W, H = self.width, self.height
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
               f'font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">',
               f'<rect width="{W}" height="{H}" fill="{BG}"/>']
        if self.title:
            out.append(self._svg_text(W / 2, 28, self.title, 16, "middle", weight="600"))
        if self.heat is not None:
            out.extend(self._render_heat())
        elif not self.series:
            out.append(self._svg_text(W / 2, H / 2, "(empty chart)", 14, "middle", MUTED))
        else:
            out.extend(self._render_axes_chart())
        out.append("</svg>")
        return "\n".join(out)

    def _render_axes_chart(self):
        W, H = self.width, self.height
        left, right, top, bottom = 68, 24, 48, 58
        pw, ph = W - left - right, H - top - bottom
        series = self.series
        cats = series[0]["cats"] if series[0]["kind"] == "bar" else None
        is_hist = series[0]["kind"] == "hist"
        out = []

        # data ranges
        if cats is not None:
            ys = [v for s in series for v in s["y"] if _finite(v)] + [0.0]
            xlo, xhi = 0.0, float(len(cats))
        elif is_hist:
            ys = [v for s in series for v in s["y"]] + [0.0]
            xlo = min(s["lo"] for s in series)
            xhi = max(s["lo"] + s["width"] * len(s["y"]) for s in series)
        else:
            xs = [v for s in series for v in s["x"] if _finite(v)]
            ys = [v for s in series for v in s["y"] if _finite(v)]
            if not xs or not ys:
                return [self._svg_text(W / 2, H / 2, "(no finite data)", 14, "middle", MUTED)]
            xlo, xhi = min(xs), max(xs)
        yticks = nice_ticks(min(ys), max(ys))
        ylo, yhi = yticks[0], yticks[-1]
        if cats is None:
            xticks = nice_ticks(xlo, xhi, 7)
            if not is_hist:
                xlo, xhi = xticks[0], xticks[-1]
        else:
            xticks = []

        def sx(v):
            return left + (v - xlo) / ((xhi - xlo) or 1) * pw

        def sy(v):
            return top + ph - (v - ylo) / ((yhi - ylo) or 1) * ph

        # grid and tick labels
        for t in yticks:
            y = sy(t)
            out.append(f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
            out.append(self._svg_text(left - 8, y + 4, _fmt(t), 11, "end", MUTED))
        if cats is None:
            for t in xticks:
                if xlo - 1e-9 <= t <= xhi + 1e-9:
                    x = sx(t)
                    out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top + ph}" y2="{top + ph + 5}" stroke="{MUTED}"/>')
                    out.append(self._svg_text(x, top + ph + 19, _fmt(t), 11, "middle", MUTED))
        else:
            slot = pw / len(cats)
            for i, c in enumerate(cats):
                x = left + slot * (i + 0.5)
                label = c if len(c) <= 12 else c[:11] + "…"
                rot = -30 if len(cats) > 8 else None
                out.append(self._svg_text(x, top + ph + 19, label, 11, "end" if rot else "middle", MUTED, rotate=rot))
        out.append(f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" stroke="{MUTED}"/>')
        out.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + ph}" stroke="{MUTED}"/>')

        # marks
        if cats is not None:
            n = len(series)
            slot = pw / len(cats)
            bw = slot * 0.7 / n
            zero = sy(0.0)
            for si, s in enumerate(series):
                for i, v in enumerate(s["y"]):
                    if not _finite(v):
                        continue
                    x = left + slot * i + slot * 0.15 + bw * si
                    y = sy(v)
                    out.append(f'<rect x="{x:.1f}" y="{min(y, zero):.1f}" width="{bw:.1f}" height="{abs(zero - y):.1f}" '
                               f'fill="{s["color"]}"><title>{escape(cats[i])}: {_fmt(v)}</title></rect>')
        elif is_hist:
            zero = sy(0.0)
            for s in series:
                for i, v in enumerate(s["y"]):
                    x0 = sx(s["lo"] + s["width"] * i)
                    x1 = sx(s["lo"] + s["width"] * (i + 1))
                    lo_e, hi_e = s["lo"] + s["width"] * i, s["lo"] + s["width"] * (i + 1)
                    out.append(f'<rect x="{x0:.1f}" y="{sy(v):.1f}" width="{max(x1 - x0 - 1, 0.5):.1f}" '
                               f'height="{zero - sy(v):.1f}" fill="{s["color"]}" fill-opacity="0.85">'
                               f'<title>{_fmt(lo_e)} to {_fmt(hi_e)}: {int(v)}</title></rect>')
        else:
            for s in series:
                pts = [(sx(a), sy(b), a, b) for a, b in zip(s["x"], s["y"]) if _finite(a) and _finite(b)]
                if s["kind"] == "line":
                    segs, cur = [], []
                    for a, b in zip(s["x"], s["y"]):
                        if _finite(a) and _finite(b):
                            cur.append(f"{sx(a):.1f},{sy(b):.1f}")
                        elif cur:
                            segs.append(cur)
                            cur = []
                    if cur:
                        segs.append(cur)
                    for seg in segs:
                        out.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="{s["color"]}" '
                                   f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
                    show_pts = len(pts) <= 60
                else:
                    show_pts = True
                if show_pts:
                    r = 3 if s["kind"] == "line" else 3.5
                    for px, py, a, b in pts:
                        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{s["color"]}" '
                                   f'fill-opacity="{0.9 if s["kind"] == "line" else 0.7}">'
                                   f'<title>({_fmt(a)}, {_fmt(b)})</title></circle>')

        # axis labels
        if self.xlabel:
            out.append(self._svg_text(left + pw / 2, H - 12, self.xlabel, 12, "middle"))
        if self.ylabel:
            out.append(self._svg_text(16, top + ph / 2, self.ylabel, 12, "middle", rotate=-90))

        # legend
        labelled = [s for s in series if s.get("label")]
        if labelled:
            # one horizontal row above the plot area, right-aligned, so it never covers data
            widths = [len(str(s["label"])) * 7 + 34 for s in labelled]
            x = left + pw - sum(widths)
            yy = top - 14
            for s, w in zip(labelled, widths):
                out.append(f'<rect x="{x:.1f}" y="{yy - 10}" width="12" height="12" rx="2" fill="{s["color"]}"/>')
                out.append(self._svg_text(x + 18, yy, str(s["label"]), 12))
                x += w
        return out

    def _render_heat(self):
        W, H = self.width, self.height
        h = self.heat
        rows, xl, yl = h["rows"], h["x"], h["y"]
        left, right, top, bottom = 90, 24, 48, 64
        pw, ph = W - left - right, H - top - bottom
        cw, ch = pw / len(xl), ph / len(yl)
        flat = [v for r in rows for v in r if _finite(v)]
        lo, hi = (min(flat), max(flat)) if flat else (0.0, 1.0)
        out = []
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                t = 0.0 if hi == lo or not _finite(v) else (v - lo) / (hi - lo)
                red = int(255 - t * (255 - 0))
                green = int(255 - t * (255 - 114))
                blue = int(255 - t * (255 - 178))
                fg = "#ffffff" if t > 0.55 else INK
                x, y = left + j * cw, top + i * ch
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
                           f'fill="rgb({red},{green},{blue})" stroke="{GRID}"><title>{escape(yl[i])} / {escape(xl[j])}: {_fmt(v)}</title></rect>')
                out.append(self._svg_text(x + cw / 2, y + ch / 2 + 4, _fmt(v), 12, "middle", fg))
        for j, label in enumerate(xl):
            out.append(self._svg_text(left + cw * (j + 0.5), top + ph + 18, label, 11, "middle", MUTED))
        for i, label in enumerate(yl):
            out.append(self._svg_text(left - 8, top + ch * (i + 0.5) + 4, label, 11, "end", MUTED))
        if self.xlabel:
            out.append(self._svg_text(left + pw / 2, H - 14, self.xlabel, 12, "middle"))
        if self.ylabel:
            out.append(self._svg_text(16, top + ph / 2, self.ylabel, 12, "middle", rotate=-90))
        return out

    # ------------------------------------------------------------- output
    def _html(self):
        title = escape(self.title or "S#N chart")
        return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
                f'content="width=device-width,initial-scale=1"><title>{title}</title>'
                f'<style>body{{margin:0;display:flex;justify-content:center;padding:24px;background:#fff}}'
                f'svg{{max-width:100%;height:auto}}</style></head><body>{self.render()}</body></html>')

    def m_save(self, path):
        path = str(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".svg":
            text = self.render()
        elif ext in (".html", ".htm"):
            text = self._html()
        else:
            raise SNRuntimeError("save() supports .svg and .html files")
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def m_show(self):
        """Write the chart to a temporary HTML file and open it in the browser."""
        import webbrowser
        fd, path = tempfile.mkstemp(suffix=".html", prefix="sn_chart_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(self._html())
        try:
            webbrowser.open("file://" + path)
        except Exception:  # noqa: BLE001 - headless environments
            pass
        return path

    def sn_repr(self):
        n = 1 if self.heat is not None else len(self.series)
        return f"Plot({self.title!r}, {n} series)" if self.title else f"Plot({n} series)"


# Convenience constructors ------------------------------------------------------
def plot(xs, ys=None, title="", xlabel="", ylabel="", label=None):
    return Plot(title, xlabel, ylabel).m_line(xs, ys, label)


def scatter(xs, ys, title="", xlabel="", ylabel="", label=None):
    return Plot(title, xlabel, ylabel).m_scatter(xs, ys, label)


def bar_chart(labels, values, title="", xlabel="", ylabel=""):
    return Plot(title, xlabel, ylabel).m_bar(labels, values)


def histogram(values, bins=10, title="", xlabel="", ylabel="count"):
    return Plot(title, xlabel, ylabel).m_hist(values, bins)


def heatmap(matrix, x_labels=None, y_labels=None, title="", xlabel="", ylabel=""):
    return Plot(title, xlabel, ylabel).m_heatmap(matrix, x_labels, y_labels)
