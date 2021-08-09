from datetime import datetime, timedelta, timezone, tzinfo
from functools import wraps
import itertools
from math import sqrt
from pathlib import Path
from textwrap import wrap as _wrap
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
from pydantic import Field as _pydantic_field, NonNegativeInt
import torch


DatetimeCorrector = Callable[[datetime], datetime]


def fill(text, width=50, **kwargs):
    return "\n\n".join(_wrap(text, width=width, **kwargs))


@wraps(_pydantic_field)
def Field(*args, **kwargs):
    if "description" in kwargs:
        kwargs["description"] = fill(kwargs["description"])
    return _pydantic_field(*args, **kwargs)


# Hack to get cache decorator to play nice with mypy
T = TypeVar("T")


def _sec_trivial(points: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
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
    points: Union[Iterable[tuple[float, float]], np.ndarray]
) -> tuple[float, float, float]:
    """Performs Welzl's algorithm to find the smallest enclosing circle of a set of
    points in a cartesian plane.

    Parameters
    ----------
    points : Union[Iterable[tuple[float, float]], np.ndarray]
        Iterable of 2-tuples or (N, 2)-array, with each tuple defining the coordinates
        of a point.

    Returns
    -------
    x : float
        x-coordinate of centre of circle.
    y : float
        y-coordinate of centre of circle.
    r : float
        Radius of circle.

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
    rr: np.ndarray,
    cc: np.ndarray,
    d: int,
    img_shape: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Takes index arrays rr and cc and performs a morphological dilation of size d on
    them.

    Parameters
    ----------
    rr : np.ndarray
        Row indices.
    cc : np.ndarray
        Column indices (must have same shape as rr).
    d : int
        Dilation factor, must be at least 1 (or a ValueError is raised).
    img_shape : Optional[tuple[int, int]]
        Shape of image (rows, columns). Indices which lie outside this will be ommitted.

    Returns
    -------
    rr_dilated : np.ndarray
        Row indices after morphological dilation.
    cc_dilated : np.ndarray
        Column indices after morphological dilation.

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
    """A mapping from subdirectory Paths to V which returns self['foo/bar'] if
    'foo/bar/baz' is missing from the available keys.

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
    KeyError: "'bar' not in SubDirDict({Path('foo'): 'foo'})"

    SubDirDict can be initialised from a dictionary

    >>> SubDirDict({"foo": "bar", "foobar": "baz"})
    SubDirDict({Path('foo'): 'bar', Path('foobar'): 'baz'})
    """

    def __init__(self, mapping: Optional[Mapping[Path, V]] = None):
        """Initialises SubDirDict.

        Parameters
        ----------
        mapping: Optional[Mapping[Path, V]]
            E.g. an instance of type dictt[Path, V]
        """
        self._lastkey = None
        self._prevkey = None
        self._dict: dict[Path, V] = {}
        if mapping is not None:
            self._dict.update(mapping)

    def __getitem__(self, key):
        if isinstance(key, str):
            key = Path(key)
        elif not isinstance(key, Path):
            raise TypeError(
                f"SubDirDict can only be indexed by Path instances. Got {type(key)}"
            )
        self._prevkey = self._lastkey
        self._lastkey = key
        try:
            return self._dict[key]
        except KeyError:
            return self.__missing__(key)

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
        """String representation of SubDirDict. Uses "Path" instead of platform
        dependant "PosixPath" or "WindowsPath".
        """
        s = ", ".join(
            (f"Path('{str(path)}'): {value!r}" for path, value in self._dict.items())
        )
        return f"SubDirDict({{{s}}})"

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


def endpoint_truncate(
    fit_mask_vals: np.ndarray, n: NonNegativeInt
) -> tuple[NonNegativeInt, NonNegativeInt]:
    """An implementation of an endpoint_method.

    Parameters
    ----------
    fit_mask_vals : np.ndarray
        Array to find the endpoints of.
    n : NonNegativeInt
        Number of values to truncate off start and end of fit_mask_vals.

    Returns
    -------
    start_index : NonNegativeInt
        Index of endpoint of polyline annotation.
    end_index : NonNegativeInt
        Index of endpoint of polyline annotation.
    """
    return (n, len(fit_mask_vals) - n)


endpoint_methods = {"truncate": endpoint_truncate}


def weighted_intersection_over_minimum(
    mask0: torch.Tensor, mask1: torch.Tensor
) -> float:
    """Calculates the weighted intersection over minimum (IoM) between two segmentation
    masks.

    Parameters
    ----------
    mask0 : torch.Tensor
        Instance segmentation mask to compare to mask1.
    mask1 : torch.Tensor
        Instance segmentation mask to compare to mask0. Should have the same shape as
        mask0.

    Returns
    -------
    iom : float
        Weighted intersection over union of mask0 and mask1.

    Examples
    --------
    >>> mask0 = torch.tensor([0.0, 0.5, 0.5, 0.0])
    >>> mask1 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    >>> weighted_intersection_over_minimum(mask0, mask1)
    0.5
    """
    return float(torch.minimum(mask0, mask1).sum()) / min(
        float(mask0.sum()), float(mask1.sum())
    )


def parse_timezone(value: str) -> timezone:
    if value == "Z":
        return timezone.utc

    offset_mins = int(value[-2:]) if len(value) > 3 else 0
    offset = 60 * int(value[1:3]) + offset_mins
    if value[0] == "-":
        offset = -offset
    return timezone(timedelta(minutes=offset))


def encode_timezone(tz: Optional[tzinfo]) -> Optional[str]:
    if tz is None:
        return None
    if tz == timezone.utc:
        return "Z"
    tot_seconds = int(tz.utcoffset(None).total_seconds())  # type: ignore[union-attr]
    hours, minutes = divmod(abs(tot_seconds) // 60, 60)
    sign = "-" if tot_seconds < 0 else "+"
    return f"{sign}{hours:02}:{minutes:02}"


class Timezone(tzinfo):
    """Provides pydantic validation for timezones."""

    _timezone: timezone

    def __init__(self, v):
        if isinstance(v, str):
            self._timezone = parse_timezone(v)
        elif isinstance(v, timezone):
            self._timezone = v
        else:
            self._timezone = timezone(v)

    def utcoffset(self, dt: Optional[datetime]) -> Optional[timedelta]:
        return self._timezone.utcoffset(dt)

    def dst(self, dt: Optional[datetime]) -> Optional[timedelta]:
        return self._timezone.dst(dt)

    def tzname(self, dt: Optional[datetime]) -> Optional[str]:
        return self._timezone.tzname(dt)

    def fromutc(self, dt: datetime) -> datetime:
        if dt.tzinfo is not self:
            raise ValueError
        return self._timezone.fromutc(dt.replace(tzinfo=self._timezone)).replace(
            tzinfo=self
        )

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            title="Timezone",
            pattern=r"^Z|[+-]\d{2}(?::?\d{2})?$",
            examples=["Z", "+10:00", "-05"],
            type="string",
        )

    @classmethod
    def validate(cls, v):
        return cls(v)

    def __str__(self):
        return encode_timezone(self._timezone)

    def __repr__(self):
        return f"Timezone({self._timezone!r})"
