from math import sqrt

from pytest import approx, raises

from camfi import util


def test_sec_trivial_colinear():
    points = [(0.0, 0.0), (1.0, 1.0,), (4.0, 4.0)]

    assert util._sec_trivial(points) == approx((2.0, 2.0, 2 * sqrt(2)))


def test_sec_trivial_too_many_points():
    points = [(-1.0, -1.0), (0.0, 0.0,), (4.0, 4.0), (1.0, 2.0)]

    with raises(ValueError):
        util._sec_trivial(points)
