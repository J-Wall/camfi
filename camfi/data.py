from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from math import atan2, cos, inf, sin, sqrt
from pathlib import Path
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import exif
from multimethod import multimethod
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    root_validator,
    validator,
)
from skimage import draw
from skimage.transform import EuclideanTransform, warp
from torch import from_numpy, hstack, stack, tensor, Tensor, zeros
from torch.utils.data import Dataset
import torchvision.io

from camfi.util import smallest_enclosing_circle, dilate_idx


class ViaFileAttributes(BaseModel):
    datetime_corrected: Optional[datetime]
    datetime_original: Optional[datetime]
    exposure_time: Optional[PositiveFloat]
    location: Optional[str]
    pixel_x_dimension: Optional[PositiveInt]
    pixel_y_dimension: Optional[PositiveInt]

    @validator("datetime_corrected", "datetime_original", pre=True)
    def valid_datetime_str(cls, v):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return datetime.strptime(v, "%Y:%m:%d %H:%M:%S")


class ViaRegionAttributes(BaseModel):
    score: Optional[float] = Field(None, ge=0, le=1)


class ViaShapeAttributes(BaseModel, ABC):
    """Abstract base class for via region shapes"""

    name: str

    @abstractmethod
    def get_bounding_box(self) -> BoundingBox:
        """Returns a BoundingBox object which contains the coordinates in self.
        Note that CircleShapeAttributes are treated like PointShapeAttributes (i.e. r is
        ignored)."""

    @abstractmethod
    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if all points in self are within bounding box.

        Parameters
        ----------
        box: BoundingBox
        """


class PointShapeAttributes(ViaShapeAttributes):
    cx: NonNegativeFloat
    cy: NonNegativeFloat
    name: str = Field("point", regex=r"^point$")

    def get_bounding_box(self) -> BoundingBox:
        return BoundingBox(
            x0=int(self.cx), y0=int(self.cy), x1=int(self.cx) + 1, y1=int(self.cy) + 1
        )

    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if all points in self are within bounding box.

        Parameters
        ----------
        box: BoundingBox

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
    cx: NonNegativeFloat
    cy: NonNegativeFloat
    name: str = Field("circle", regex=r"^circle$")
    r: NonNegativeFloat

    def as_point(self):
        return PointShapeAttributes(cx=self.cx, cy=self.cy)

    def get_bounding_box(self) -> BoundingBox:
        return self.as_point().get_bounding_box()

    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if all points in self are within bounding box.

        Parameters
        ----------
        box: BoundingBox

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
    all_points_x: List[NonNegativeFloat]
    all_points_y: List[NonNegativeFloat]
    name: str = Field("polyline", regex=r"^polyline$")

    best_peak: Optional[float] = Field(
        None, gt=0.0, description="period of wingbeat in pixels"
    )
    blur_length: Optional[float] = Field(
        None, gt=0.0, description="length of motion blur in pixels"
    )
    snr: Optional[float] = Field(None, description="signal to noise ratio of best peak")
    wb_freq_up: Optional[float] = Field(
        None,
        ge=0.0,
        description="wingbeat frequency estimate, assuming upward motion (and zero body-length)",
    )
    wb_freq_down: Optional[float] = Field(
        None,
        ge=0.0,
        description="wingbeat frequency estimate, assuming downward motion (and zero body-length)",
    )
    et_up: Optional[float] = Field(
        None,
        ge=0.0,
        description="corrected moth exposure time, assuming upward motion",
    )
    et_dn: Optional[float] = Field(
        None,
        ge=0.0,
        description="corrected moth exposure time, assuming downward motion",
    )

    @validator("all_points_y")
    def same_number_of_points_in_both_dimensions(cls, v, values):
        if "all_points_x" in values and len(v) != len(values["all_points_x"]):
            raise ValueError("must have same number of y points as x points")
        return v

    def as_circle(self) -> CircleShapeAttributes:
        """Returns a CircleShapeAttributes object from the smallest enclosing circle
        of the points in self."""
        cx, cy, r = smallest_enclosing_circle(zip(self.all_points_x, self.all_points_y))
        return CircleShapeAttributes(cx=cx, cy=cy, r=r)

    def get_bounding_box(self) -> BoundingBox:
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
        self, image: Tensor, scan_distance: PositiveInt
    ) -> Tensor:
        """Extracts region of interest (ROI) from an image tensor

        Parameters
        ----------
        image: Tensor
            image to extract region of interest from. Should be greyscale (ie. just have
            two axes)
        scan_distance : PositiveInt
            half-width of rois for motion blurs

        Returns
        -------
        Tensor
           Region of interest

        Examples
        --------
        >>> image = tensor([
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

        img = image.numpy()  # Using skimage, which operates on numpy arrays

        sections = []
        for (x0, x1), (y0, y1) in zip(pair(self.all_points_x), pair(self.all_points_y)):
            # Calculate angle of section
            rotation = atan2(y1 - y0, x1 - x0)

            # Calculate upper corner of ROI
            x_translation = x0 + (scan_distance - 1) * sin(rotation)
            y_translation = y0 - (scan_distance - 1) * cos(rotation)

            # Rotate and translate image
            transform_matrix = EuclideanTransform(
                rotation=rotation, translation=(x_translation, y_translation)
            )
            warped_img = warp(img, transform_matrix)

            # Crop rotated image to ROI
            section_length = int(round(sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)))
            cropped_img = warped_img[: 2 * scan_distance - 1, :section_length]

            sections.append(from_numpy(cropped_img))  # Convert back to Tensor

        # Join sections to form complete ROI
        joined_img = hstack(sections)

        return joined_img


class ViaRegion(BaseModel):
    region_attributes: ViaRegionAttributes
    shape_attributes: Union[
        PolylineShapeAttributes, CircleShapeAttributes, PointShapeAttributes
    ]

    def get_bounding_box(self) -> BoundingBox:
        return self.shape_attributes.get_bounding_box()

    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if all points in region are within bounding box.

        Parameters
        ----------
        box: BoundingBox

        Examples
        --------
        >>> polyline = PolylineShapeAttributes(
        ...     all_points_x=[1, 3],
        ...     all_points_y=[15, 13],
        ... )
        >>> region = ViaRegion(
        ...     region_attributes=ViaRegionAttributes(),
        ...     shape_attributes=polyline,
        ... )
        >>> region.in_box(BoundingBox(x0=1, y0=13, x1=4, y1=16))
        True
        >>> region.in_box(BoundingBox(x0=2, y0=13, x1=4, y1=16))
        False
        >>> region.in_box(BoundingBox(x0=1, y0=14, x1=4, y1=16))
        False
        >>> region.in_box(BoundingBox(x0=1, y0=13, x1=3, y1=16))
        False
        >>> region.in_box(BoundingBox(x0=1, y0=13, x1=4, y1=15))
        False
        """
        return self.shape_attributes.in_box(box)


class ViaMetadata(BaseModel):
    file_attributes: ViaFileAttributes
    filename: Path
    regions: List[ViaRegion]
    size: int = -1

    def get_bounding_boxes(self) -> List[BoundingBox]:
        """Calls .get_bounding_box on each region in self.regions.

        Returns
        -------
        List[BoundingBox]
        """
        return [region.get_bounding_box() for region in self.regions]

    def get_labels(self) -> List[PositiveInt]:
        """Gets a list full of 1's with same length as self.regions

        Returns
        -------
        List[int]
            [1, 1, 1, ...]
        """
        return [1 for _ in range(len(self.regions))]

    def get_iscrowd(self) -> List[int]:
        """Gets a list full of 0's with same length as self.regions

        Returns
        -------
        List[int]
            [0, 0, 0, ...]
        """
        return [0 for _ in range(len(self.regions))]

    def load_exif_metadata(
        self,
        location: Optional[str] = None,
        datetime_corrector: Optional[Callable[[datetime], datetime]] = None,
    ) -> None:
        """Extract EXIF metadata from an image file and put it in self.file_attributes.
        Note: this will overwrite all contents in self.file_attributes.

        Parameters
        ----------
        location: Optional[str]
            Option to also apply a location
        datetime_corrector: Optional[Callable[[datetime], datetime]]
            If set, then will be used to calculate datetime_corrected

        EXIF tags loaded
        ----------------
        datetime_original: datetime
        exposure_time: PositiveFloat
        pixel_x_dimension: PositiveInt
        pixel_Y_dimension: PositiveInt

        Extra tags
        ----------
        datetime_corrected: datetime
            if datetime_corrector is set, this is calculated by
            calling datetime_corrector(datetime_original).
        location: str
            set if location is set

        Returns
        -------
        None (operates in place)

        Examples
        --------
        >>> metadata = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="camfi/test/data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> metadata.load_exif_metadata()
        >>> metadata.file_attributes.datetime_original
        datetime.datetime(2019, 11, 14, 20, 30, 29)
        >>> print(round(metadata.file_attributes.exposure_time, 6))
        0.111111
        >>> metadata.file_attributes.pixel_y_dimension
        3456
        >>> metadata.file_attributes.pixel_x_dimension
        4608

        If location is set, this will be reflected
        >>> metadata = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="camfi/test/data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> metadata.load_exif_metadata(location="cabramurra")
        >>> metadata.file_attributes.location
        'cabramurra'

        If a time correction needs to be made (for example if the camera's clock is
        known to have been incorrectly set), then we can correct the datetime by
        supplying a function to the `datetime_corrector` parameter.
        >>> metadata = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="camfi/test/data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> metadata.load_exif_metadata(
        ...     datetime_corrector=lambda dt: dt - timedelta(days=30)
        ... )
        >>> metadata.file_attributes.datetime_original
        datetime.datetime(2019, 11, 14, 20, 30, 29)
        >>> metadata.file_attributes.datetime_corrected
        datetime.datetime(2019, 10, 15, 20, 30, 29)
        """
        with open(self.filename, "rb") as image_file:
            image = exif.Image(image_file)

        self.file_attributes = ViaFileAttributes(
            datetime_original=image.datetime_original,
            exposure_time=image.exposure_time,
            pixel_x_dimension=image.pixel_x_dimension,
            pixel_y_dimension=image.pixel_y_dimension,
            location=location,
        )

        if (
            datetime_corrector is not None
            and self.file_attributes.datetime_original is not None
        ):
            self.file_attributes.datetime_corrected = datetime_corrector(
                self.file_attributes.datetime_original
            )


class ViaProject(BaseModel):
    via_attributes: Dict
    via_img_metadata: Dict[str, ViaMetadata]
    via_settings: Dict

    class Config:
        fields = {
            "via_attributes": "_via_attributes",
            "via_img_metadata": "_via_img_metadata",
            "via_settings": "_via_settings",
        }


class MaskMaker:
    def __init__(
        self,
        shape: Tuple[PositiveInt, PositiveInt],
        mask_dilate: Optional[PositiveInt] = None,
    ):
        self.shape = shape
        self.mask_dilate = mask_dilate

    @multimethod
    def get_mask(self, point: PointShapeAttributes) -> Tensor:
        """Produces a mask Tensor for a given point.
        Raises a ValueError if point lies outise self.shape"""
        cx = int(point.cx)
        cy = int(point.cy)

        if cx >= self.shape[1] or cy >= self.shape[0]:
            raise ValueError(
                f"point ({cy}, {cx}) lies outside image with shape {self.shape}"
            )

        mask = zeros(self.shape)
        if self.mask_dilate is None:
            rr, cc = cy, cx
        else:
            rr, cc = dilate_idx(cy, cx, self.mask_dilate, img_shape=self.shape)

        mask[rr, cc] = 1

        return mask

    @multimethod  # type: ignore[no-redef]
    def get_mask(self, circle: CircleShapeAttributes) -> Tensor:
        """Produces a mask Tensor for a given circle.
        Raises a ValueError if centre lies outise self.shape"""
        return self.get_mask(circle.as_point())

    @multimethod  # type: ignore[no-redef]
    def get_mask(self, polyline: PolylineShapeAttributes) -> Tensor:
        """Produces a mask Tensor for a given polyline.
        Raises a ValueError if any point lies outise self.shape"""
        x = polyline.all_points_x
        y = polyline.all_points_y
        mask = zeros(self.shape)

        for i in range(len(x) - 1):
            if x[i] >= self.shape[1] or y[i] >= self.shape[0]:
                raise ValueError(
                    f"point ({x[i]}, {y[i]}) lies outside image with shape {self.shape}"
                )

            rr, cc = draw.line(int(y[i]), int(x[i]), int(y[i + 1]), int(x[i + 1]))
            if self.mask_dilate is not None:
                rr, cc = dilate_idx(rr, cc, self.mask_dilate, img_shape=self.shape)

            mask[rr, cc] = 1

        return mask

    def get_masks(self, metadata: ViaMetadata) -> List[Tensor]:
        """Calls self.get_mask on all regions in metadata

        Parameters
        ----------
        metadata : ViaMetadata
            has a field called "regions"

        Returns
        -------
        List[Tensor]
            list of masks
        """
        return [self.get_mask(region.shape_attributes) for region in metadata.regions]


class BoundingBox(BaseModel):
    x0: NonNegativeInt
    y0: NonNegativeInt
    x1: NonNegativeInt
    y1: NonNegativeInt

    @validator("x1")
    def x1_gt_x0(cls, v, values):
        if "x0" in values and v <= values["x0"]:
            raise ValueError("x1 and y1 must be larger than x0 and y0")
        return v

    @validator("y1")
    def y1_gt_y0(cls, v, values):
        if "y0" in values and v <= values["y0"]:
            raise ValueError("x1 and y1 must be larger than x0 and y0")
        return v

    @classmethod
    def from_shape(
        self, shape: Tuple[PositiveInt, PositiveInt], border: NonNegativeInt = 0
    ) -> BoundingBox:
        """Creates an instance of BoundingBox from an image shape, useful for defining
        a region of interest within an image, not too close to the edge.

        Parameters
        ----------
        shape: Tuple[PositiveInt, PositiveInt]
            shape of image (height, width)
        border: NonNegativeInt
            width of border. If 0 (default), then the bounding box will contain the
            entire image.

        Returns
        -------
        BoundingBox

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

    def add_margin(
        self,
        margin: NonNegativeInt,
        shape: Optional[Tuple[PositiveInt, PositiveInt]] = None,
    ):
        """Expands self by a fixed margin. Operates in-place.

        Parameters
        ----------
        margin: PositiveInt
            Margin to add to self
        shape: Optional[Tuple[PositiveInt, PositiveInt]] = (height, width)
            Shape of image. If set, will constrain self to image shape
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
        PositiveInt
            area

        Examples
        --------
        >>> box = BoundingBox(x0=0, y0=1, x1=2, y1=3)
        >>> box.get_area()
        4
        """
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if self is contained in box

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
        """Returns True if two bounding boxes overlap, and False otherwise

        Parameters
        ----------
        box: BoundingBox
            another bounding box to compare to

        Returns
        -------
        bool

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
        """Get the intersectional area of two boxes

        Parameters
        ----------
        box: BoundingBox
            another bounding box to compare to

        Returns
        -------
        NonNegativeInt
            Area of intersection of two boxes

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
        """Get the intersection over union of two boxes

        Parameters
        ----------
        box: BoundingBox
            another bounding box to compare to

        Returns
        -------
        NonNegativeFloat
            Intersection over Union of two boxes, between 0.0 and 1.0

        Examples
        --------
        >>> box0 = BoundingBox(x0=0, y0=0, x1=1, y1=1)
        >>> box1 = BoundingBox(x0=2, y0=2, x1=3, y1=3)
        >>> box2 = BoundingBox(x0=0, y0=0, x1=2, y1=2)
        >>> box3 = BoundingBox(x0=1, y0=1, x1=3, y1=3)
        >>> box0.intersection_over_union(box1)
        0.0
        >>> box2.intersection_over_union(box3)
        0.125

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
        union = self.get_area() + box.get_area()

        return intersection / union


class Target(BaseModel):
    boxes: List[BoundingBox]
    labels: List[PositiveInt]
    image_id: NonNegativeInt
    area: List[PositiveInt]
    iscrowd: List[int]
    masks: List[Tensor]

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def all_fields_have_same_length(cls, values):
        try:
            length = len(values["boxes"])
            if not all(
                len(values[k]) == length for k in ["labels", "area", "iscrowd", "masks"]
            ):
                raise ValueError("Fields must have same length")
        except KeyError:
            raise ValueError("Invalid parameters given to Target")

        return values

    @validator("masks")
    def all_masks_same_shape(cls, v):
        if len(v) <= 1:
            return v

        shape = v[0].shape
        for mask in v[1:]:
            if mask.shape != shape:
                raise ValueError("All masks must have same shape")

        return v

    def to_tensor_dict(self) -> Dict[str, Tensor]:
        return dict(
            boxes=tensor([[b.x0, b.y0, b.x1, b.y1] for b in self.boxes]),
            labels=tensor(self.labels),
            image_id=tensor([self.image_id]),
            area=tensor(self.area),
            iscrowd=tensor(self.iscrowd),
            masks=stack(self.masks),
        )

    @classmethod
    def from_tensor_dict(cls, tensor_dict: Dict[str, Tensor]) -> Target:
        return Target(
            boxes=[
                BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
                for x0, y0, x1, y1 in tensor_dict["boxes"]
            ],
            labels=[int(v) for v in tensor_dict["labels"]],
            image_id=int(tensor_dict["image_id"]),
            area=[int(v) for v in tensor_dict["area"]],
            iscrowd=[int(v) for v in tensor_dict["iscrowd"]],
            masks=list(tensor_dict["masks"]),
        )

    @classmethod
    def empty(cls, image_id: NonNegativeInt = 0) -> Target:
        """Initialises a Target with and image_id but no other data.

        Parameters
        ----------
        image_id : NonNegativeInt

        Returns
        -------
        Target

        Examples
        --------
        >>> Target.empty()
        Target(boxes=[], labels=[], image_id=0, area=[], iscrowd=[], masks=[])
        """
        return Target(
            boxes=[], labels=[], image_id=image_id, area=[], iscrowd=[], masks=[]
        )


class ImageTransform(BaseModel, ABC):
    """Abstract base class for transforms on images with segmentation data."""

    def __call__(self, image: Tensor, target: Target) -> Tuple[Tensor, Target]:
        image, target_dict = self.apply_to_tensor_dict(image, target.to_tensor_dict())
        return image, Target.from_tensor_dict(target_dict)

    @abstractmethod
    def apply_to_tensor_dict(
        self, image: Tensor, target: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Subclasses of ImageTransform should implement this to return a transformed
        image and target dict."""


def read_image(path: Path) -> Tensor:
    """Read an image from a file

    Parameters
    ----------
    path: Path
        path to file

    Returns
    -------
    Tensor[colour, height (y), width (x)]
        image as RGB float32 tensor

    Examples
    --------
    >>> image = read_image(Path("examples/data/calibration_img0.jpg"))
    >>> image.shape == (3, 3456, 4608)
    True
    >>> image.dtype
    torch.float32
    """
    image = torchvision.io.read_image(
        str(path), mode=torchvision.io.image.ImageReadMode.RGB
    )
    return image / 255  # Converts from uint8 to float32


class CamfiDataset(BaseModel, Dataset):
    root: Path
    via_project: ViaProject
    crop: Optional[Tuple[int, int, int, int]] = None  # [x0, y0, x1, y1]

    inference_mode: bool = False

    # Only set if inference_mode = False
    mask_maker: Optional[MaskMaker] = None
    transform: Optional[ImageTransform] = None
    min_annotations: int = 0
    max_annotations: float = inf

    # Optionally exclude some files
    exclude: set = None  # type: ignore[assignment]

    # Automatically generated. No need to set.
    keys: List = None  # type: ignore[assignment]

    class Config:
        arbitrary_types_allowed = True

    @validator("exclude", pre=True, always=True)
    def default_exclude(cls, v):
        if v is None:
            return set()
        return v

    @validator("transform", "min_annotations", "max_annotations")
    def only_set_if_not_inference_mode(cls, v, values):
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v in {0, inf, None}, "Only set if inference_mode=False"
        return v

    @validator("mask_maker", always=True)
    def set_iff_not_inference_mode(cls, v, values):
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v is None, "Only set if inference_mode=False"
        else:
            assert isinstance(
                v, MaskMaker
            ), "mask_maker must be set if inference_mode=False"
        return v

    @validator("keys", pre=True, always=True)
    def generate_filtered_keys(cls, v, values):
        min_annotations = values.get("min_annotations", 0)
        max_annotations = values.get("max_annotations", inf)
        return list(
            dict(
                filter(
                    lambda e: e[1].filename not in values["exclude"]
                    and min_annotations <= len(e[1].regions) <= max_annotations,
                    values["via_project"].via_img_metadata.items(),
                )
            )
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Target]:
        metadata = self.via_project.via_img_metadata[self.keys[idx]]
        image = read_image(self.root / metadata.filename)

        if self.crop is not None:
            image = image[:, self.crop[1] : self.crop[3], self.crop[0] : self.crop[2]]

        if self.inference_mode:
            target = Target.empty(image_id=idx)
        else:  # Training mode
            boxes = metadata.get_bounding_boxes()
            target = Target(
                boxes=boxes,
                labels=metadata.get_labels(),
                image_id=idx,
                area=[box.get_area() for box in boxes],
                iscrowd=metadata.get_iscrowd(),
                masks=self.mask_maker.get_masks(metadata),  # type: ignore[union-attr]
            )

            if self.transform is not None:
                image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.keys)
