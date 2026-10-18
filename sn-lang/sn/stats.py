"""Small statistics helpers working on plain Python lists of numbers."""
import math


def _check(xs, name):
    if len(xs) == 0:
        raise ValueError(f"{name} of an empty sequence")


def mean(xs):
    _check(xs, "mean")
    return math.fsum(xs) / len(xs)


def variance(xs, ddof=0):
    _check(xs, "variance")
    n = len(xs)
    if n - ddof <= 0:
        raise ValueError("not enough values for the requested ddof")
    m = mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (n - ddof)


def std(xs, ddof=0):
    return math.sqrt(variance(xs, ddof))


def median(xs):
    _check(xs, "median")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2


def percentile(xs, q):
    """Linear-interpolated percentile, q in [0, 100]."""
    _check(xs, "percentile")
    if not 0 <= q <= 100:
        raise ValueError("percentile q must be between 0 and 100")
    s = sorted(xs)
    pos = (len(s) - 1) * q / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def pearson(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("corr needs sequences of equal length")
    _check(xs, "corr")
    mx, my = mean(xs), mean(ys)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return math.nan
    return sxy / math.sqrt(sxx * syy)


def mode(xs):
    _check(xs, "mode")
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    for x in xs:
        if counts[x] == best:
            return x
