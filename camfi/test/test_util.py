from math import sqrt
from pathlib import Path

from pytest import approx, raises

from camfi import util


def test_sec_trivial_colinear():
    points = [
        (0.0, 0.0),
        (1.0, 1.0,),
        (4.0, 4.0),
    ]

    assert util._sec_trivial(points) == approx((2.0, 2.0, 2 * sqrt(2)))


def test_sec_trivial_too_many_points():
    points = [
        (-1.0, -1.0),
        (0.0, 0.0),
        (4.0, 4.0),
        (1.0, 2.0),
    ]

    with raises(ValueError):
        util._sec_trivial(points)


class TestSubDirDict:
    def test_getitem_fails(self):
        subdir_dict = util.SubDirDict()
        with raises(
            TypeError,
            match="SubDirDict can only be indexed by Path instances. Got <class 'int'>",
        ):
            subdir_dict[1]

    def test_setitem_fails(self):
        subdir_dict = util.SubDirDict()
        with raises(
            TypeError,
            match="SubDirDict can only be indexed by Path instances. Got <class 'int'>",
        ):
            subdir_dict[1] = "foo"

    def test_iter(self):
        subdir_dict = util.SubDirDict({Path("foo"): "bar"})
        for e in subdir_dict:
            assert e == Path("foo")

    def test_len(self):
        assert len(util.SubDirDict()) == 0
        assert len(util.SubDirDict({Path("foo"): "bar"})) == 1

    def test_keys(self):
        subdir_dict = util.SubDirDict({Path("foo"): "bar"})
        for key in subdir_dict.keys():
            assert key == Path("foo")

    def test_values(self):
        subdir_dict = util.SubDirDict({Path("foo"): "bar"})
        for value in subdir_dict.values():
            assert value == "bar"

    def test_items(self):
        subdir_dict = util.SubDirDict({Path("foo"): "bar"})
        for key, value in subdir_dict.items():
            assert key == Path("foo")
            assert value == "bar"
