from collections.abc import Mapping
import functools
import itertools
from math import sqrt
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np


# Hack to get cache decorator to play nice with mypy
T = TypeVar("T")


def cache(func: Callable[..., T]) -> T:
    return functools.lru_cache(maxsize=None)(func)  # type: ignore


def _sec_trivial(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    if len(points) == 3:
        (x1, y1), (x2, y2), (x3, y3) = points
        A = np.array([[x3 - x1, y3 - y1], [x3 - x2, y3 - y2]])
        Y = np.array(
            [
                (x3 ** 2 + y3 ** 2 - x1 ** 2 - y1 ** 2),
                (x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2),
            ]
        )
        if np.linalg.det(A) == 0:
            min_point = min(points)
            max_point = max(points)
            return (
                0.5 * (min_point[0] + max_point[0]),
                0.5 * (min_point[1] + max_point[1]),
                0.5
                * sqrt(
                    (min_point[0] - max_point[0]) ** 2
                    + (min_point[1] - max_point[1]) ** 2
                ),
            )
        Ainv = np.linalg.inv(A)
        X = 0.5 * np.dot(Ainv, Y)
        return X[0], X[1], sqrt((X[0] - x1) ** 2 + (X[1] - y1) ** 2)
    elif len(points) == 2:
        return (
            0.5 * (points[0][0] + points[1][0]),
            0.5 * (points[0][1] + points[1][1]),
            0.5
            * sqrt(
                (points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2
            ),
        )
    elif len(points) == 1:
        return points[0][0], points[0][1], 0.0
    elif len(points) == 0:
        return 0.0, 0.0, 0.0
    else:
        raise ValueError(f"{len(points)} points given. Maximum for trivial case is 3.")


def smallest_enclosing_circle(
    points: Union[Iterable[Tuple[float, float]], np.ndarray]
) -> Tuple[float, float, float]:
    """Performs Welzl's algorithm to find the smallest enclosing circle of a set of
    points in a cartesian plane.

    Parameters
    ----------
    points : iterable of 2-tuples or (N, 2)-array

    Returns
    -------
    x, y, r : floats

    Examples
    --------
    If no points are given, values are still returned:

    >>> smallest_enclosing_circle([])
    (0.0, 0.0, 0.0)

    If one point is given, r will be 0.0:

    >>> smallest_enclosing_circle([(1.0, 2.0)])
    (1.0, 2.0, 0.0)

    Two points trivial case:

    >>> smallest_enclosing_circle([(0.0, 0.0), (2.0, 0.0)])
    (1.0, 0.0, 1.0)

    Three points trivial case:

    >>> np.allclose(
    ...     smallest_enclosing_circle([(0.0, 0.0), (2.0, 0.0), (1.0, sqrt(3))]),
    ...     (1.0, sqrt(3) / 3, 2 * sqrt(3) / 3)
    ... )
    True

    Extra points within the circle don't affect the circle:

    >>> np.allclose(
    ...     smallest_enclosing_circle([
    ...                                (0.0, 0.0),
    ...                                (2.0, 0.0),
    ...                                (1.0, sqrt(3)),
    ...                                (0.5, 0.5)]),
    ...     (1.0, sqrt(3) / 3, 2 * sqrt(3) / 3)
    ... )
    True

    If points are inscribed on a circle, the correct circle is also given:

    >>> np.allclose(
    ...     smallest_enclosing_circle([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]),
    ...     (1.0, 1.0, sqrt(2))
    ... )
    True
    """

    def welzl(P, R):
        if len(P) == 0 or len(R) == 3:
            return _sec_trivial(R)
        p = P[0]
        D = welzl(P[1:], R)
        if (D[0] - p[0]) ** 2 + (D[1] - p[1]) ** 2 < D[2] ** 2:
            return D

        return welzl(P[1:], R + (p,))

    P = [tuple(p) for p in points]

    return welzl(tuple(P), ())


def dilate_idx(
    rr: Union[np.ndarray, int],
    cc: Union[np.ndarray, int],
    d: int,
    img_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Takes index arrays rr and cc and performs a morphological dilation of size d on
    them.

    Parameters
    ----------
    rr : array or int
        row indices
    cc : array or int
        column indices (must have same shape as rr)
    d : int
        dilation factor, must be at least 1 (or a ValueError is raised)
    img_shape : (rows, cols)
        shape of image (indices which lie outside this will be ommitted)

    Returns
    -------
    rr_dilated : array
    cc_dilated : array

    Examples
    --------
    >>> a = np.array([[0, 0, 0, 0, 0],
    ...               [0, 0, 1, 0, 0],
    ...               [0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0]])
    >>> rr, cc = np.nonzero(a)
    >>> rr_dilated, cc_dilated = dilate_idx(rr, cc, 1)
    >>> a[rr_dilated, cc_dilated] = 1
    >>> a
    array([[0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])

    If shape is given, omits indices larger than the dimensions given

    >>> a = np.array([[0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 1],
    ...               [0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0]])
    >>> rr, cc = np.nonzero(a)
    >>> rr_dilated, cc_dilated = dilate_idx(rr, cc, 1, a.shape)
    >>> a[rr_dilated, cc_dilated] = 1
    >>> a
    array([[0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1],
           [0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])

    If we didn't give the shape argument in the above example, we get an IndexError

    >>> a = np.array([[0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 1],
    ...               [0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0]])
    >>> rr, cc = np.nonzero(a)
    >>> rr_dilated, cc_dilated = dilate_idx(rr, cc, 1)
    >>> a[rr_dilated, cc_dilated] = 1
    Traceback (most recent call last):
    ...
    IndexError: index 5 is out of bounds for axis 1 with size 5

    But we don't need the shape parameter to filter out negative indices

    >>> a = np.array([[1, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0]])
    >>> rr, cc = np.nonzero(a)
    >>> rr_dilated, cc_dilated = dilate_idx(rr, cc, 1)
    >>> a[rr_dilated, cc_dilated] = 1
    >>> a
    array([[1, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])

    Dilation is based on euclidean distance

    >>> a = np.array([[0, 0, 0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0, 0, 0],
    ...               [0, 0, 0, 1, 0, 0, 0],
    ...               [0, 0, 0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0, 0, 0],
    ...               [0, 0, 0, 0, 0, 0, 0]])
    >>> rr, cc = np.nonzero(a)
    >>> rr_dilated, cc_dilated = dilate_idx(rr, cc, 3, a.shape)
    >>> a[rr_dilated, cc_dilated] = 1
    >>> a
    array([[0, 0, 0, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 1, 0, 0, 0]])

    If input is sorted, then the ouptut will be too (with precedence rr, cc)

    >>> rr, cc = np.array([50]), np.array([50])
    >>> dilate_idx(rr, cc, 1, (100, 100))
    (array([49, 50, 50, 50, 51]), array([50, 49, 50, 51, 50]))

    If a non-positive dilation factor is given, a ValueError is raised

    >>> dilate_idx(1, 2, 0)
    Traceback (most recent call last):
    ...
    ValueError: d=0. Should be positive.
    """
    if d < 1:
        raise ValueError(f"d={d}. Should be positive.")

    d2 = d * d
    offset_r, offset_c = zip(
        *itertools.filterfalse(
            lambda x: x[0] ** 2 + x[1] ** 2 > d2,
            itertools.product(range(-d, d + 1), repeat=2),
        )
    )
    rr_dilated = np.stack([rr + i for i in offset_r]).ravel()
    cc_dilated = np.stack([cc + i for i in offset_c]).ravel()
    mask = rr_dilated >= 0
    mask[cc_dilated < 0] = False
    if img_shape is not None:
        mask[rr_dilated >= img_shape[0]] = False
        mask[cc_dilated >= img_shape[1]] = False

    return rr_dilated[mask], cc_dilated[mask]


V = TypeVar("V")


class SubDirDict(Mapping[Path, V]):
    """A mapping which returns self['foo/bar'] if self['foo/bar/baz'] is missing
    from the dict.

    Examples
    --------
    >>> d = SubDirDict()
    >>> d["foo"] = "foo"
    >>> d["foo"]
    'foo'
    >>> d["foo/bar"]
    'foo'
    >>> d["foo/bar/baz"]
    'foo'
    >>> d["bar"]
    Traceback (most recent call last):
    ...
    KeyError: "'bar' not in SubDirDict({'foo': 'foo'})"
    """

    def __init__(self):
        self._lastkey = None
        self._prevkey = None
        self._dict = {}

    def __getitem__(self, key):
        if isinstance(key, str):
            key = Path(key)
        elif not isinstance(key, Path):
            raise TypeError(
                f"SubDirDict can only be indexed by Path instances. Got {type(key)}"
            )
        self._prevkey = self._lastkey
        self._lastkey = key
        return self._dict[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = Path(key)
        elif not isinstance(key, Path):
            raise TypeError(
                f"SubDirDict can only be indexed by Path instances. Got {type(key)}"
            )
        self._dict[key] = value

    def __missing__(self, key):
        if key == Path():
            raise KeyError(f"'{self._prevkey}' not in {str(self)}")

        return self[key.parent]

    def __repr__(self):
        return f"{str(type(self)).split('.')[-1][:-2]}({super().__repr__()})"

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()
