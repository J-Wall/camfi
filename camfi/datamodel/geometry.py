"""Defines geometry related classes and methods used throughout camfi. Depends on
camfi.util."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import atan2, degrees, sqrt
from typing import Optional

from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    validator,
)
from shapely.geometry import LineString
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pad, rotate

from camfi.util import smallest_enclosing_circle, Field


class BoundingBox(BaseModel):
    """Defines a bounding box, and provides various convenience methods for working with
    boxes on images.

    Parameters
    ----------
    x0: NonNegativeInt
        Minimum inclusive (horizontal) x-coordinate of box.
    y0: NonNegativeInt
        Minimum inclusive (vertical) y-coordinate of box.
    x1: NonNegativeInt
        Maximum exclusive (horizontal) x-coordinate of box.
    y1: NonNegativeInt
        Maximum exclusive (vertical) y-coordinate of box.
    """

    x0: NonNegativeInt = Field(
        ..., description="Minimum inclusive (horizontal) x-coordinate of box."
    )
    y0: NonNegativeInt = Field(
        ..., description="Minimum inclusive (vertical) y-coordinate of box."
    )
    x1: NonNegativeInt = Field(
        ..., description="Maximum exclusive (horizontal) x-coordinate of box."
    )
    y1: NonNegativeInt = Field(
        ..., description="Maximum exclusive (vertical) y-coordinate of box."
    )

    class Config:
        schema_extra = {
            "description": "Defines a bounding box with integer (pixel) coordinates."
        }

    @validator("x1")
    def x1_gt_x0(cls, v, values):
        """Pydantic validation method, called when instantiating BoundingBox.
        Ensures that x1 > x0."""
        if "x0" in values and v <= values["x0"]:
            raise ValueError("x1 and y1 must be larger than x0 and y0")
        return v

    @validator("y1")
    def y1_gt_y0(cls, v, values):
        """Pydantic validation method, called when instantiating BoundingBox.
        Ensures that y1 > y0."""
        if "y0" in values and v <= values["y0"]:
            raise ValueError("x1 and y1 must be larger than x0 and y0")
        return v

    @classmethod
    def from_shape(
        self, shape: tuple[PositiveInt, PositiveInt], border: NonNegativeInt = 0
    ) -> BoundingBox:
        """Creates an instance of BoundingBox from an image shape, useful for defining
        a region of interest within an image, not too close to the edge.

        Parameters
        ----------
        shape: tuple[PositiveInt, PositiveInt]
            Shape of image (height, width).
        border: NonNegativeInt
            Width of border. If 0 (default), then the bounding box will contain the
            entire image.

        Returns
        -------
        box : BoundingBox
            Optionally contracted bounding box of image.

        Examples
        --------
        >>> BoundingBox.from_shape((10, 15))
        BoundingBox(x0=0, y0=0, x1=16, y1=11)

        >>> BoundingBox.from_shape((10, 15), border=3)
        BoundingBox(x0=3, y0=3, x1=13, y1=8)
        """
        return BoundingBox(
            x0=border, y0=border, x1=shape[1] + 1 - border, y1=shape[0] + 1 - border
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Gets the (height, width) of the bounding box.

        Returns
        -------
        shape : tuple[int, int]
            Height and width of the bounding box

        Examples
        --------
        >>> BoundingBox(x0=5, y0=1, x1=15, y1=3).shape
        (2, 10)
        """
        return (self.y1 - self.y0, self.x1 - self.x0)

    def add_margin(
        self,
        margin: NonNegativeInt,
        shape: Optional[tuple[PositiveInt, PositiveInt]] = None,
    ) -> None:
        """Expands self by a fixed margin. Operates in-place.

        Parameters
        ----------
        margin: PositiveInt
            Margin to add to self.
        shape: Optional[tuple[PositiveInt, PositiveInt]] = (height, width)
            Shape of image. If set, will constrain self to image shape.
        """
        self.x0 = max(0, self.x0 - margin)
        self.y0 = max(0, self.y0 - margin)

        self.x1 += margin
        self.y1 += margin

        if shape is not None:
            self.x0 = min(shape[1], self.x0)
            self.y0 = min(shape[0], self.y0)
            self.x1 = min(shape[1], self.x1)
            self.y1 = min(shape[0], self.y1)

    def get_area(self) -> PositiveInt:
        """Get the area enclosed by self.

        Returns
        -------
        area : PositiveInt
            Area enclosed by box, in pixels ** 2.

        Examples
        --------
        >>> box = BoundingBox(x0=0, y0=1, x1=2, y1=3)
        >>> box.get_area()
        4
        """
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if self is contained in box.

        Parameters
        ----------
        box : BoundingBox
            Other box to test against.

        Returns
        -------
        is_in_box : bool
            True if self is completely contained within box.

        Examples
        --------
        >>> box0 = BoundingBox(x0=1, y0=2, x1=3, y1=4)
        >>> box1 = BoundingBox(x0=0, y0=1, x1=4, y1=5)
        >>> box0.in_box(box1)
        True
        >>> box1.in_box(box0)
        False

        A box is always in itself

        >>> box0.in_box(box0)
        True
        """
        return (
            self.x0 >= box.x0
            and self.y0 >= box.y0
            and self.x1 <= box.x1
            and self.y1 <= box.y1
        )

    def overlaps(self, box: BoundingBox) -> bool:
        """Returns True if two bounding boxes overlap, and False otherwise.

        Parameters
        ----------
        box: BoundingBox
            Another bounding box to compare to.

        Returns
        -------
        overlap : bool
            True if self and box intersect.

        Examples
        --------
        >>> box0 = BoundingBox(x0=0, y0=0, x1=1, y1=1)
        >>> box1 = BoundingBox(x0=2, y0=2, x1=3, y1=3)
        >>> box2 = BoundingBox(x0=0, y0=0, x1=2, y1=2)
        >>> box3 = BoundingBox(x0=1, y0=1, x1=3, y1=3)
        >>> box0.overlaps(box1)
        False
        >>> box2.overlaps(box3)
        True

        Overlaps can happen in either dimension:

        >>> box0 = BoundingBox(x0=0, y0=0, x1=2, y1=2)
        >>> box1 = BoundingBox(x0=0, y0=1, x1=1, y1=3)
        >>> box2 = BoundingBox(x0=1, y0=0, x1=3, y1=1)
        >>> box3 = BoundingBox(x0=0, y0=2, x1=2, y1=4)
        >>> box4 = BoundingBox(x0=2, y0=0, x1=4, y1=2)
        >>> box0.overlaps(box1)
        True
        >>> box0.overlaps(box2)
        True

        Overlaps are not inclusive of edges:

        >>> box0.overlaps(box3)
        False
        >>> box0.overlaps(box4)
        False
        """
        return (
            self.x0 < box.x1
            and self.y0 < box.y1
            and self.x1 > box.x0
            and self.y1 > box.y0
        )

    def intersection(self, box: BoundingBox) -> NonNegativeInt:
        """Get the intersectional area of two boxes.

        Parameters
        ----------
        box: BoundingBox
            Another bounding box to compare to.

        Returns
        -------
        intersectional_area : NonNegativeInt
            Area of intersection of two boxes.

        Examples
        --------
        >>> box0 = BoundingBox(x0=0, y0=0, x1=1, y1=1)
        >>> box1 = BoundingBox(x0=2, y0=2, x1=3, y1=3)
        >>> box2 = BoundingBox(x0=0, y0=0, x1=2, y1=2)
        >>> box3 = BoundingBox(x0=1, y0=1, x1=3, y1=3)
        >>> box0.intersection(box1)
        0
        >>> box2.intersection(box3)
        1

        Intersection is commutative

        >>> from itertools import product
        >>> pairs = product([box0, box1, box2, box3], repeat=2)
        >>> all(b0.intersection(b1) == b1.intersection(b0) for b0, b1 in pairs)
        True
        """
        if self.overlaps(box):
            return BoundingBox(
                x0=max(self.x0, box.x0),
                y0=max(self.y0, box.y0),
                x1=min(self.x1, box.x1),
                y1=min(self.y1, box.y1),
            ).get_area()
        return 0

    def intersection_over_union(self, box: BoundingBox) -> NonNegativeFloat:
        """Get the intersection over union of two boxes.

        Parameters
        ----------
        box: BoundingBox
            Another bounding box to compare to.

        Returns
        -------
        iou : NonNegativeFloat
            Intersection over Union of two boxes, between 0.0 and 1.0.

        Examples
        --------
        >>> box0 = BoundingBox(x0=0, y0=0, x1=1, y1=1)
        >>> box1 = BoundingBox(x0=2, y0=2, x1=3, y1=3)
        >>> box2 = BoundingBox(x0=0, y0=0, x1=2, y1=2)
        >>> box3 = BoundingBox(x0=1, y0=0, x1=4, y1=2)
        >>> box0.intersection_over_union(box1)
        0.0
        >>> box1.intersection_over_union(box2)
        0.0
        >>> box2.intersection_over_union(box3)
        0.25
        >>> box0.intersection_over_union(box2)
        0.25

        Intersection over union is commutative

        >>> from itertools import product
        >>> pairs = product([box0, box1, box2, box3], repeat=2)
        >>> all(
        ...     b0.intersection_over_union(b1) == b1.intersection_over_union(b0)
        ...     for b0, b1 in pairs
        ... )
        True
        """
        intersection = self.intersection(box)
        union = self.get_area() + box.get_area() - intersection

        return intersection / union

    def is_portrait(self) -> bool:
        """Returns True if bounding box is at least as tall as it is wide.

        Returns
        -------
        portrait : bool
            True if bounding box is at least as tall as it is wide.

        Examples
        --------
        >>> BoundingBox(x0=1, y0=0, x1=11, y1=10).is_portrait()
        True
        >>> BoundingBox(x0=0, y0=0, x1=11, y1=10).is_portrait()
        False
        """
        return self.y1 - self.y0 >= self.x1 - self.x0

    def crop_image(self, image: torch.Tensor) -> torch.Tensor:
        """Returns a view of an image cropped to the bounding box.

        Parameters
        ----------
        image : torch.Tensor
            With shape [..., height, width].

        Returns
        -------
        torch.Tensor
            with shape [..., self.y1 - self.y0, self.x1 - self.x0], assuming
            height <= self.y1 - self.y0 and width <= self.x1 - self.x0.

        Examples
        --------
        >>> box = BoundingBox(x0=7, x1=15, y0=3, y1=7)
        >>> grey_image = torch.zeros(10, 20)
        >>> box.crop_image(grey_image).shape
        torch.Size([4, 8])
        >>> colour_image = torch.zeros(3, 10, 20)
        >>> box.crop_image(colour_image).shape
        torch.Size([3, 4, 8])

        If BoundingBox goes outside image.shape, then output size will be truncated in
        the expected way

        >>> box = BoundingBox(x0=15, x1=25, y0=3, y1=7)
        >>> box.crop_image(grey_image).shape
        torch.Size([4, 5])
        >>> box = BoundingBox(x0=7, x1=15, y0=7, y1=13)
        >>> box.crop_image(grey_image).shape
        torch.Size([3, 8])
        """
        return image[..., self.y0 : self.y1, self.x0 : self.x1]


class ViaShapeAttributes(BaseModel, ABC):
    """Abstract base class for via region shapes. These define the geometry data of
    annotations of flying insects in images.

    Parameters
    ----------
    name : str
        Name of shape type (e.g. "point", "circle", or "polyline").
    """

    name: str

    @abstractmethod
    def get_bounding_box(self) -> BoundingBox:
        """Returns a BoundingBox object which contains the coordinates in self.
        Note that CircleShapeAttributes are treated like PointShapeAttributes (i.e. r is
        ignored).

        Returns
        -------
        box : BoundingBox
            Bounding box of ViaShapeAttributes instance.
        """

    @abstractmethod
    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if all points in self are within bounding box.

        Parameters
        ----------
        box : BoundingBox
            Bounding box to test against.

        Returns
        -------
        is_in_box : bool
            True if within box.
        """

    def y_diff(self) -> PositiveInt:
        """Returns the total height (y-dimension) of the annotation (in pixels).

        Returns
        -------
        y_diff : PositiveInt
            Height of self in pixels.
        """
        bounding_box = self.get_bounding_box()
        return bounding_box.y1 - bounding_box.y0 - 1

    def intersection_over_union(self, other: ViaShapeAttributes) -> NonNegativeFloat:
        """Get the intersection over union of bounding boxes.

        Parameters
        ----------
        other: ViaShapeAttributes
            Other shape to compare to.

        Returns
        -------
        iou : NonNegativeFloat
            Intersection over Union of two bounding boxes, between 0.0 and 1.0.

        Examples
        --------
        >>> class MockShapeAttributes(ViaShapeAttributes):
        ...     bounding_box: BoundingBox
        ...     def get_bounding_box(self) -> BoundingBox:
        ...         return self.bounding_box
        ...     def in_box(self, box: BoundingBox) -> bool:
        ...         return self.bounding_box.in_box(box)
        >>> shape_attributes0 = MockShapeAttributes(
        ...     bounding_box=BoundingBox(x0=0, y0=0, x1=2, y1=1),
        ...     name="mock_shape",
        ... )
        >>> shape_attributes1 = MockShapeAttributes(
        ...     bounding_box=BoundingBox(x0=1, y0=0, x1=4, y1=1),
        ...     name="mock_shape",
        ... )
        >>> shape_attributes0.intersection_over_union(shape_attributes1)
        0.25
        """
        return self.get_bounding_box().intersection_over_union(other.get_bounding_box())


class PointShapeAttributes(ViaShapeAttributes):
    """Defines a point geometry.

    Parameters
    ----------
    cx : NonNegativeFloat
        x-coordinate of point.
    cy : NonNegativeFloat
        y-coordinate of point.
    name : str
        Name of shape (must be "point" for PointShapeAttributes instances).
    """

    cx: NonNegativeFloat
    cy: NonNegativeFloat
    name: str = Field("point", regex=r"^point$")

    def get_bounding_box(self) -> BoundingBox:
        """Finds the bounding box of the point.

        Returns
        -------
        box : BoundingBox
            Bounding box of point

        Examples
        --------
        >>> point = PointShapeAttributes(cx=10, cy=15)
        >>> point.get_bounding_box()
        BoundingBox(x0=10, y0=15, x1=11, y1=16)
        """
        return BoundingBox(
            x0=int(self.cx), y0=int(self.cy), x1=int(self.cx) + 1, y1=int(self.cy) + 1
        )

    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if all points in self are within bounding box.

        Parameters
        ----------
        box: BoundingBox
           Box to test against..

        Returns
        -------
        is_in_box : bool
            True if point is within box.

        Examples
        --------
        >>> point = PointShapeAttributes(cx=2, cy=13)
        >>> point.in_box(BoundingBox(x0=2, y0=13, x1=4, y1=15))
        True
        >>> point.in_box(BoundingBox(x0=3, y0=13, x1=4, y1=15))
        False
        >>> point.in_box(BoundingBox(x0=2, y0=14, x1=4, y1=15))
        False
        >>> point.in_box(BoundingBox(x0=1, y0=13, x1=2, y1=15))
        False
        >>> point.in_box(BoundingBox(x0=2, y0=12, x1=4, y1=13))
        False
        """
        return self.get_bounding_box().in_box(box)


class CircleShapeAttributes(ViaShapeAttributes):
    """Defines a circle geometry.

    Parameters
    ----------
    cx : NonNegativeInt
        x-coordinate of centre of circle.
    cy : NonNegativeInt
        y-coordinate of centre of circle.
    name : str
        Name of shape (must be "circle" for CircleShapeAttributes instances).
    r : NonNegativeFloat
        Radius of circle.
    """

    cx: NonNegativeFloat
    cy: NonNegativeFloat
    name: str = Field("circle", regex=r"^circle$")
    r: NonNegativeFloat

    @classmethod
    def from_bounding_box(cls, box: BoundingBox) -> CircleShapeAttributes:
        """Takes a BoundingBox and returns a CircleShapeAttributes, which encloses the
        BoundingBox.

        Parameters
        ----------
        box : BoundingBox
            Box to enclose with circle

        Returns
        -------
        circle : CircleShapeAttributes
            Smallest circle enclosing box.

        Examples
        --------
        >>> from pytest import approx
        >>> box = BoundingBox(x0=0, y0=0, x1=2, y1=2)
        >>> circle = CircleShapeAttributes.from_bounding_box(box)
        >>> circle.cx == approx(1.0)
        True
        >>> circle.cy == approx(1.0)
        True
        >>> circle.r == approx(sqrt(2))
        True
        """
        all_points_x = [box.x0, box.x0, box.x1, box.x1]
        all_points_y = [box.y0, box.y1, box.y0, box.y1]
        cx, cy, r = smallest_enclosing_circle(zip(all_points_x, all_points_y))
        return CircleShapeAttributes(cx=cx, cy=cy, r=r)

    def as_point(self):
        """Converts a CircleShapeAttributes instance to a PointShapeAttributes instance.

        Returns
        -------
        point : PointShapeAttributes
            Point at centre of circle.
        """
        return PointShapeAttributes(cx=self.cx, cy=self.cy)

    def get_bounding_box(self) -> BoundingBox:
        """Finds the bounding box of the point at the centre of the circle.

        **Note: this does not return the bounding box of the entire circle, just it's
        centre.**

        Returns
        -------
        box : BoundingBox
            Bounding box of point at the centre of the circle.

        Examples
        --------
        >>> circle = CircleShapeAttributes(cx=10, cy=15, r=10)
        >>> circle.get_bounding_box()
        BoundingBox(x0=10, y0=15, x1=11, y1=16)
        """
        return self.as_point().get_bounding_box()

    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if all points in self are within bounding box.

        Parameters
        ----------
        box : BoundingBox
            Box to test against.

        Returns
        -------
        is_in_box : bool
            True if centre of circle is contained in box.

        Examples
        --------
        >>> circle = CircleShapeAttributes(cx=2, cy=13, r=10)
        >>> circle.in_box(BoundingBox(x0=2, y0=13, x1=4, y1=15))
        True
        >>> circle.in_box(BoundingBox(x0=3, y0=13, x1=4, y1=15))
        False
        >>> circle.in_box(BoundingBox(x0=2, y0=14, x1=4, y1=15))
        False
        >>> circle.in_box(BoundingBox(x0=1, y0=13, x1=2, y1=15))
        False
        >>> circle.in_box(BoundingBox(x0=2, y0=12, x1=4, y1=13))
        False
        """
        return self.get_bounding_box().in_box(box)


class PolylineShapeAttributes(ViaShapeAttributes):
    """Defines a polyline geometry.

    Parameters
    ----------
    all_points_x : list[NonNegativeFloat]
        list of the x-coordinates of the points defining the polyline.
    all_points_y : list[NonNegativeFloat]
        list of the y-coordinates of the points defining the polyline.
    name : str
        Name of shape (must be "polyline" for PolylineShapeAttributes instances).
    """

    all_points_x: list[NonNegativeFloat]
    all_points_y: list[NonNegativeFloat]
    name: str = Field("polyline", regex=r"^polyline$")

    @validator("all_points_y")
    def same_number_of_points_in_both_dimensions(cls, v, values):
        if "all_points_x" in values and len(v) != len(values["all_points_x"]):
            raise ValueError("must have same number of y points as x points")
        return v

    def as_circle(self) -> CircleShapeAttributes:
        """Calculates the smallest enclosing circle of the polyline.

        Returns
        -------
        smallest_enclosing_circle : CircleShapeAttributes
            Smallest enclosing circle of polyline.

        Examples
        --------
        >>> polyline = PolylineShapeAttributes(all_points_x=[0, 1], all_points_y=[0, 0])
        >>> polyline.as_circle()
        CircleShapeAttributes(name='circle', cx=0.5, cy=0.0, r=0.5)
        """
        cx, cy, r = smallest_enclosing_circle(zip(self.all_points_x, self.all_points_y))
        return CircleShapeAttributes(cx=cx, cy=cy, r=r)

    def get_bounding_box(self) -> BoundingBox:
        """Finds the bounding box of all the points in the polyline.

        Returns
        -------
        box : BoundingBox
            Bounding box of all the points in the polyline.

        Examples
        --------
        >>> polyline = PolylineShapeAttributes(all_points_x=[0, 1], all_points_y=[0, 0])
        >>> polyline.get_bounding_box()
        BoundingBox(x0=0, y0=0, x1=2, y1=1)
        """
        x_min = min(self.all_points_x)
        x_max = max(self.all_points_x)
        y_min = min(self.all_points_y)
        y_max = max(self.all_points_y)

        return BoundingBox(
            x0=int(x_min), y0=int(y_min), x1=int(x_max) + 1, y1=int(y_max) + 1
        )

    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if all points in self are within bounding box.

        Parameters
        ----------
        box: BoundingBox
            Box to test against.

        Returns
        -------
        is_in_box : bool
            True if entire polyline is contained in box.

        Examples
        --------
        >>> polyline = PolylineShapeAttributes(
        ...     all_points_x=[1, 3],
        ...     all_points_y=[15, 13],
        ... )
        >>> polyline.in_box(BoundingBox(x0=1, y0=13, x1=4, y1=16))
        True
        >>> polyline.in_box(BoundingBox(x0=2, y0=13, x1=4, y1=16))
        False
        >>> polyline.in_box(BoundingBox(x0=1, y0=14, x1=4, y1=16))
        False
        >>> polyline.in_box(BoundingBox(x0=1, y0=13, x1=3, y1=16))
        False
        >>> polyline.in_box(BoundingBox(x0=1, y0=13, x1=4, y1=15))
        False
        """
        return self.get_bounding_box().in_box(box)

    def extract_region_of_interest(
        self, image: torch.Tensor, scan_distance: PositiveInt
    ) -> torch.Tensor:
        """Extracts region of interest (ROI) from an image tensor.

        Parameters
        ----------
        image : torch.Tensor
            Image to extract region of interest from. Should be greyscale (ie. just have
            two axes)
        scan_distance : PositiveInt
            Half-width of rois for motion blurs.

        Returns
        -------
        roi : torch.Tensor
           Rotated, cropped, and straightened region of interest.

        Examples
        --------
        >>> image = torch.tensor([
        ...     [0.0, 0.1, 0.2, 0.3, 0.4],
        ...     [1.0, 1.1, 1.2, 1.3, 1.4],
        ...     [2.0, 2.1, 2.2, 2.3, 2.4],
        ...     [3.0, 3.1, 3.2, 3.3, 3.4],
        ...     [4.0, 4.1, 4.2, 4.3, 4.4],
        ... ])
        >>> polyline = PolylineShapeAttributes(
        ...     all_points_x=[1, 4],
        ...     all_points_y=[2, 2],
        ... )
        >>> polyline.extract_region_of_interest(image, 1)
        tensor([[2.1000, 2.2000, 2.3000]])
        >>> polyline.extract_region_of_interest(image, 2)
        tensor([[1.1000, 1.2000, 1.3000],
                [2.1000, 2.2000, 2.3000],
                [3.1000, 3.2000, 3.3000]])

        Also works for multi-segment polylines

        >>> polyline = PolylineShapeAttributes(
        ...     all_points_x=[1, 2, 4],
        ...     all_points_y=[2, 2, 2],
        ... )
        >>> polyline.extract_region_of_interest(image, 1)
        tensor([[2.1000, 2.2000, 2.3000]])
        >>> polyline.extract_region_of_interest(image, 2)
        tensor([[1.1000, 1.2000, 1.3000],
                [2.1000, 2.2000, 2.3000],
                [3.1000, 3.2000, 3.3000]])

        Segments can have different angles to each other

        >>> polyline = PolylineShapeAttributes(
        ...     all_points_x=[1, 3, 3],
        ...     all_points_y=[2, 2, 0],
        ... )
        >>> polyline.extract_region_of_interest(image, 1)
        tensor([[2.1000, 2.2000, 2.3000, 1.3000]])
        >>> polyline.extract_region_of_interest(image, 2)
        tensor([[1.1000, 1.2000, 2.2000, 1.2000],
                [2.1000, 2.2000, 2.3000, 1.3000],
                [3.1000, 3.2000, 2.4000, 1.4000]])

        And segments can be at arbitrary angles. This example starts towards the top-
        right corner and travels towards the bottom-left.

        >>> polyline = PolylineShapeAttributes(
        ...     all_points_x=[3, 0],
        ...     all_points_y=[1, 4],
        ... )
        >>> polyline.extract_region_of_interest(image, 2)
        tensor([[2.0778, 2.7142, 3.3506, 3.9870],
                [1.3000, 1.9364, 2.5728, 3.2092],
                [0.5222, 1.1586, 1.7950, 2.4314]])

        And this one from the top-left, heading down and left

        >>> polyline = PolylineShapeAttributes(
        ...     all_points_x=[1, 4],
        ...     all_points_y=[1, 4],
        ... )
        >>> polyline.extract_region_of_interest(image, 2)
        tensor([[0.4636, 1.2414, 2.0192, 2.7971],
                [1.1000, 1.8778, 2.6556, 3.4335],
                [1.7364, 2.5142, 3.2920, 4.0698]])
        """

        def pair(items):
            return zip(items[:-1], items[1:])

        # Torch transforms need colour channel, and image needs to be padded
        img = pad(image.reshape((1,) + image.shape), scan_distance)

        sections = []
        for (x0, x1), (y0, y1) in zip(pair(self.all_points_x), pair(self.all_points_y)):
            # Correct for padding
            cx = x0 + scan_distance + 0.5
            cy = y0 + scan_distance + 0.5

            # Calculate angle of section
            rotation = atan2(y1 - y0, x1 - x0)

            # Calculate section length
            section_length = int(round(sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)))

            # Determine if image needs extra padding along x-axis
            if img.shape[-1] - cx < section_length:
                img = pad(img, [0, 0, int(section_length + cx - img.shape[-1] + 1), 0])

            # Rotate and translate image
            rotated_img = rotate(
                img,
                degrees(rotation),
                interpolation=InterpolationMode.BILINEAR,
                center=[cx, cy],
            )

            # Crop rotated image to ROI
            cropped_img = rotated_img.reshape(rotated_img.shape[1:])[
                int(cy) - scan_distance + 1 : int(cy) + scan_distance,
                int(cx) : int(cx) + section_length,
            ]

            sections.append(cropped_img)

        # Join sections to form complete ROI
        joined_img = torch.hstack(sections)

        return joined_img

    def length(self):
        """Get the sum of lengths of all the polyline segments.

        Returns
        -------
        length : float
            Length of polyline in pixels.

        Examples
        --------
        >>> from pytest import approx
        >>> polyline = PolylineShapeAttributes(
        ...     all_points_x=[0, 1, 2],
        ...     all_points_y=[0, 1, 1],
        ... )
        >>> polyline.length() == approx(sqrt(2) + 1)
        True
        """
        xs = torch.tensor(self.all_points_x)
        ys = torch.tensor(self.all_points_y)
        return float(((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2).sqrt().sum())

    def to_shapely(self) -> LineString:
        """Casts self to a shapely.geometry.LineString instance.

        Returns
        -------
        line_string : LineString
            Shapely representation of a polyline.

        Examples
        --------
        >>> polyline = PolylineShapeAttributes(
        ...     all_points_x=[0, 1, 1],
        ...     all_points_y=[1, 1, 2],
        ... )
        >>> line_string = polyline.to_shapely()
        >>> isinstance(line_string, LineString)
        True
        >>> print(line_string)
        LINESTRING (0 1, 1 1, 1 2)
        """
        return LineString(zip(self.all_points_x, self.all_points_y))

    def hausdorff_distance(self, polyline: PolylineShapeAttributes) -> NonNegativeFloat:
        """Returns the Hausdorff distance between two PolylineShapeAttributes instances.

        Parameters
        ----------
        polyline : PolylineShapeAttributes
            Other polyline to compare to

        Returns
        -------
        h_dist : NonNegativeFloat
            Hausdorff distance between self and polyline.

        Examples
        --------
        >>> polyline0 = PolylineShapeAttributes(
        ...     all_points_x=[0, 1],
        ...     all_points_y=[0, 0],
        ... )
        >>> polyline1 = PolylineShapeAttributes(
        ...     all_points_x=[0, 1],
        ...     all_points_y=[1, 1],
        ... )
        >>> polyline0.hausdorff_distance(polyline1)
        1.0
        """
        return self.to_shapely().hausdorff_distance(polyline.to_shapely())
