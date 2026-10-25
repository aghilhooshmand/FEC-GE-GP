"""Tests for the multiclass operators."""
import numpy as np

import operators as ops


def test_argmax_picks_highest_and_ties_are_minus_one():
    a = np.array([5.0, 1.0, 3.0, 2.0])
    b = np.array([5.0, 4.0, 3.0, 9.0])
    c = np.array([3.0, 2.0, 3.0, 1.0])
    assert ops.argmax3(a, b, c).tolist() == [-1, 1, -1, 1]       # row 0: tie a=b; row 2: three-way tie


def test_argmax_broadcasts_constants():
    x = np.array([0.1, 0.9])
    assert ops.argmax3(ops.where(x > 0.5, 7.0, 1.0), 4.0, 2.0).tolist() == [1, 0]
    assert np.ndim(ops.argmax3(1.0, 2.0, 0.5)) == 0              # no feature: scalar -> invalid by the shared rule


def test_where_and_argmax4_5():
    assert ops.where(np.array([True, False]), 1.0, 2.0).tolist() == [1.0, 2.0]
    assert ops.argmax4(1, 2, 3, 4).item() == 3 and ops.argmax5(5, 1, 1, 1, 1).item() == 0


def test_argmax6_to_10_ties_are_minus_one():
    assert ops.argmax6(1, 2, 3, 4, 5, 6).item() == 5 and ops.argmax6(6, 1, 1, 1, 1, 6).item() == -1
    a = np.array([0.0, 9.0]); b = np.array([0.0, 1.0])
    assert ops.argmax10(*([b] * 9 + [a])).tolist() == [-1, 9]
    assert all(f"argmax{k}" in ops.OPERATORS for k in range(3, 11))
