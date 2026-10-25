"""Operator functions used by the grammars, copied from the reference implementation operators/custom_operators.json."""
import numpy as np


def add(a, b):
    return np.add(a, b)


def sub(a, b):
    return np.subtract(a, b)


def mul(a, b):
    return np.multiply(a, b)


def pdiv(a, b):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(b == 0, np.ones_like(a), a / b)
    except ZeroDivisionError:
        return 1.0


def and_(a, b):
    return np.logical_and(a, b)


def or_(a, b):
    return np.logical_or(a, b)


def not_(a):
    return np.logical_not(a)


def less_than_or_equal(a, b):
    return a <= b


def greater_than_or_equal(a, b):
    return a >= b


# ---- multiclass: per-class scores, prediction = argmax; a tie between the highest scores -> -1

def where(cond, a, b):
    """if cond then a else b, per row."""
    return np.where(cond, a, b)


def _argmax(*scores):
    """Per row, the index of the highest score; -1 where two or more scores share the maximum."""
    s = np.broadcast_arrays(*[np.asarray(v, dtype=np.float64) for v in scores])
    s = np.stack(s)
    top = s.max(axis=0)
    winner = s.argmax(axis=0)
    return np.where((s == top).sum(axis=0) > 1, -1, winner)


def argmax3(a, b, c):
    return _argmax(a, b, c)


def argmax4(a, b, c, d):
    return _argmax(a, b, c, d)


def argmax5(a, b, c, d, e):
    return _argmax(a, b, c, d, e)


def argmax6(a, b, c, d, e, f):                           # 6..10 classes
    return _argmax(a, b, c, d, e, f)


def argmax7(a, b, c, d, e, f, g):
    return _argmax(a, b, c, d, e, f, g)


def argmax8(a, b, c, d, e, f, g, h):
    return _argmax(a, b, c, d, e, f, g, h)


def argmax9(a, b, c, d, e, f, g, h, i):
    return _argmax(a, b, c, d, e, f, g, h, i)


def argmax10(a, b, c, d, e, f, g, h, i, j):
    return _argmax(a, b, c, d, e, f, g, h, i, j)


def _argmax_k(k: int):
    """argmax11..argmax20: the same rule over k scores."""
    def f(*scores):
        if len(scores) != k:
            raise TypeError(f"argmax{k} takes {k} scores, got {len(scores)}")
        return _argmax(*scores)
    f.__name__ = f"argmax{k}"
    return f


for _k in range(11, 21):
    globals()[f"argmax{_k}"] = _argmax_k(_k)

OPERATORS = {f.__name__: f for f in (add, sub, mul, pdiv, and_, or_, not_, less_than_or_equal, greater_than_or_equal,
                                     where, argmax3, argmax4, argmax5, argmax6, argmax7,
                                     argmax8, argmax9, argmax10, *(globals()[f"argmax{k}"] for k in range(11, 21)))}
