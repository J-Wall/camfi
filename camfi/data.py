from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from math import atan2, cos, fsum, inf, sin, sqrt
from pathlib import Path
import random
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import exif
from numpy import array
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
from shapely.geometry import LineString
from skimage import draw
from skimage.transform import EuclideanTransform, warp
from torch import from_numpy, hstack, stack, tensor, Tensor, zeros
from torch.utils.data import Dataset
import torchvision.io

from camfi.util import SubDirDict, smallest_enclosing_circle, dilate_idx


DatetimeCorrector = Callable[[datetime], datetime]


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
    best_peak: Optional[int] = Field(
        None, gt=0, description="period of wingbeat in pixels"
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

    def y_diff(self):
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
        NonNegativeFloat
            Intersection over Union of two bounding boxes, between 0.0 and 1.0

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
        CircleShapeAttributes

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

    def length(self):
        """Get the sum of lengths of all the polyline segments

        Returns
        -------
        float

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
        xs = tensor(self.all_points_x)
        ys = tensor(self.all_points_y)
        return float(((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2).sqrt().sum())

    def to_shapely(self) -> LineString:
        """Casts self to a shapely.geometry.LineString instance

        Returns
        -------
        LineString

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
        NonNegativeFloat

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


class ViaRegion(BaseModel):
    region_attributes: ViaRegionAttributes
    shape_attributes: Union[
        PolylineShapeAttributes, CircleShapeAttributes, PointShapeAttributes
    ]

    @validator("shape_attributes")
    def only_polylines_get_wingbeat_data(cls, v, values):
        if "region_attributes" in values and v.name != "polyline":
            for field in [
                values["region_attributes"].best_peak,
                values["region_attributes"].blur_length,
                values["region_attributes"].snr,
                values["region_attributes"].wb_freq_up,
                values["region_attributes"].wb_freq_down,
                values["region_attributes"].et_up,
                values["region_attributes"].et_dn,
            ]:
                assert field is None, "Wingbeat data is invalid for non-polylines"
        return v

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

    def load_exif_metadata(
        self,
        root: Path = Path(),
        location: Optional[str] = None,
        datetime_corrector: Optional[DatetimeCorrector] = None,
    ) -> None:
        """Extract EXIF metadata from an image file and put it in self.file_attributes.
        Note: this will overwrite all contents in self.file_attributes.

        Parameters
        ----------
        root: Path
            Root directory from which the relative path in self.filename is resolved.
            Defaults to current working directory. If a str is passed it will be coerced
            to a Path.
        location: Optional[str]
            Option to also apply a location
        datetime_corrector: Optional[DatetimeCorrector]
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

        Optionally specify root directory. Here we are loading the same file, but using
        a root parameter. Note that root may also be a relative path (as in this case).
        Absolute paths are also acceptable.
        >>> metadata_with_root = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> metadata_with_root.load_exif_metadata(root="camfi/test")
        >>> metadata_with_root.file_attributes == metadata.file_attributes
        True

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
        with open(Path(root) / self.filename, "rb") as image_file:
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

    def read_image(self, root: Path = Path()) -> Tensor:
        """Read an image from a file

        Parameters
        ----------
        root: Path
            Root directory from which the relative path in self.filename is resolved.
            Defaults to current working directory. If a str is passed it will be coerced
            to a Path.

        Returns
        -------
        Tensor[colour, height (y), width (x)]
            image as RGB float32 tensor

        Examples
        --------
        >>> metadata = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="camfi/test/data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> image = metadata.read_image()
        >>> image.shape == (3, 3456, 4608)
        True
        >>> image.dtype
        torch.float32

        Optionally specify root directory. Here we are loading the same file, but using
        a root parameter. Note that root may also be a relative path (as in this case).
        Absolute paths are also acceptable.
        >>> metadata_with_root = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> image_with_root = metadata_with_root.read_image(root="camfi/test")
        >>> image_with_root.allclose(image)
        True
        """
        image = torchvision.io.read_image(
            str(Path(root) / self.filename), mode=torchvision.io.image.ImageReadMode.RGB
        )
        return image / 255  # Converts from uint8 to float32


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

    def load_all_exif_metadata(
        self,
        root: Path = Path(),
        location_dict: Optional[Mapping[Path, Optional[str]]] = None,
        datetime_correctors: Optional[
            Mapping[Path, Optional[DatetimeCorrector]]
        ] = None,
    ) -> None:
        """Calls the `.load_exif_metadata` method of all ViaMetadata instances in
        self.via_img_metadata, extracting the EXIF metadata from each image file

        Parameters
        ----------
        root: Path
            Root directory from which the relative path in self.filename is resolved.
            Defaults to current working directory. If a str is passed it will be coerced
            to a Path.
        location_dict: Optional[Mapping[Path, Optional[str]]]
            A mapping from filenames (i.e. relative paths to images under root) to
            location strings, which are passed to `ViaMetadata.load_exif_metadata`.
            Typically, an instance of `camfi.util.SubDirDict` should be used.
        datetime_correctors: Optional[Mapping[Path, Optional[DatetimeCorrector]]]
            A mapping from filenames (i.e. relative paths to images under root) to
            DatetimeCorrector instances, which are passed to
            `ViaMetadata.load_exif_metadata`
            Typically, an instance of `camfi.util.SubDirDict` should be used.

        Returns
        -------
        None (operates in place)

        Examples
        --------
        >>> with open("camfi/test/data/sample_project_images_included.json") as f:
        ...     project = ViaProject.parse_raw(f.read())

        The file which has been loaded contains no metadata
        >>> for meta in project.via_img_metadata.values():
        ...     print(meta.filename, str(meta.file_attributes.datetime_original))
        DSCF0010.JPG None
        DSCF0011.JPG None

        After load_all_exif_metadata is called, `project` does contain image metadata
        >>> project.load_all_exif_metadata(root=Path("camfi/test/data"))
        >>> for meta in project.via_img_metadata.values():
        ...     print(meta.filename, str(meta.file_attributes.datetime_original))
        DSCF0010.JPG 2019-11-14 20:30:29
        DSCF0011.JPG 2019-11-14 20:40:32

        If `location_dict` and/or `datetime_correctors` are set, the metadata will
        include `location` and/or `datetime_corrected`, respectively. Normally, these
        would be set as instances of `camfi.util.SubDirDict`, but for brevity we use
        a regular `dict` for each of them here.
        >>> project.load_all_exif_metadata(
        ...     root=Path("camfi/test/data"),
        ...     location_dict={
        ...         Path("DSCF0010.JPG"): "loc0", Path("DSCF0011.JPG"): "loc1"
        ...     },
        ...     datetime_correctors={
        ...         Path("DSCF0010.JPG"): lambda dt: dt + timedelta(hours=1),
        ...         Path("DSCF0011.JPG"): lambda dt: dt - timedelta(hours=1),
        ...     },
        ... )
        >>> for meta in project.via_img_metadata.values():
        ...     print(
        ...         meta.filename,
        ...         meta.file_attributes.location,
        ...         str(meta.file_attributes.datetime_corrected),
        ...     )
        DSCF0010.JPG loc0 2019-11-14 21:30:29
        DSCF0011.JPG loc1 2019-11-14 19:40:32
        """
        if location_dict is None:
            location_dict = defaultdict(lambda: None)
        if datetime_correctors is None:
            datetime_correctors = defaultdict(lambda: None)

        for metadata in self.via_img_metadata.values():
            metadata.load_exif_metadata(
                root=root,
                location=location_dict[metadata.filename],
                datetime_corrector=datetime_correctors[metadata.filename],
            )


class LocationTime(BaseModel):
    camera_start_time: datetime
    actual_start_time: Optional[datetime] = None  # Defaults to camera_start_time
    camera_end_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    location: Optional[str] = None

    @validator("actual_start_time", always=True)
    def default_actual_start_time(cls, v, values):
        if v is None and "camera_start_time" in values:
            return values["camera_start_time"]
        return v

    @root_validator
    def all_offset_aware_or_naive(cls, values):
        is_offset_aware = values["camera_start_time"].tzinfo is not None
        for field in ["actual_start_time", "camera_end_time", "actual_end_time"]:
            if field in values and values[field] is not None:
                assert (
                    values[field].tzinfo is not None
                ) == is_offset_aware, (
                    "Unable to mix timezone offset-aware and -naive datetimes"
                )
        return values

    def get_time_ratio(self) -> Optional[float]:
        """If self.camera_end_time and self.actual_end_time are set, gets the camera
        time to actual time ratio. Otherwise, None

        Returns
        -------
        camera_time_to_actual_time_ratio : float

        Examples
        --------
        >>> location_time = LocationTime(
        ...     camera_start_time="2021-07-15T14:00",
        ...     camera_end_time="2021-07-15T16:00",
        ...     actual_end_time="2021-07-15T15:00",
        ... )
        >>> location_time.get_time_ratio()
        2.0

        `camera_end_time` and `actual_end_time` must be set, or else None is returned
        >>> print(LocationTime(camera_start_time="2021-07-15T14:00").get_time_ratio())
        None
        """
        if (
            self.actual_start_time is None
            or self.camera_end_time is None
            or self.actual_end_time is None
        ):
            return None

        camera_elapsed_time = self.camera_end_time - self.camera_start_time
        actual_elapsed_time = self.actual_end_time - self.actual_start_time
        return camera_elapsed_time / actual_elapsed_time

    def corrector(
        self, camera_time_to_actual_time_ratio: Optional[float] = None
    ) -> DatetimeCorrector:
        """Returns a datetime corrector function, which takes an original camera-
        reported datetime as an argument, and returns a corrected datetime.
        If self.actual_start_time is None, then it is assumed that
        self.camera_start_time reflects the actual time. Generally it is advised to set
        the time of the camera just before it is placed out, which is the basis for this
        assumption.

        Parameters
        ----------
        camera_time_to_actual_time_ratio : Optional[float]
            The amount of time elapsed as reported by the camera divided by the actual
            amount of time elapsed. If None, then this is inferred from the fields of
            the LocationTime instance. In this case, both self.camera_end_time and
            self.actual_end_time must be set.

        Returns
        -------
        datetime_corrector : DatetimeCorrector

        Examples
        --------
        >>> location_time = LocationTime(camera_start_time="2021-07-15T14:00")
        >>> location_corrector = location_time.corrector(2.)
        >>> location_corrector(datetime(2021, 7, 15, 16, 0))
        datetime.datetime(2021, 7, 15, 15, 0)

        Also works with offset-aware datetimes
        >>> location_time = LocationTime(camera_start_time="2021-07-15T14:00+10")
        >>> location_corrector = location_time.corrector(2.)
        >>> location_corrector(datetime.fromisoformat("2021-07-15T17:00+11:00"))
        datetime.datetime(2021, 7, 15, 15, 0, tzinfo=datetime.timezone(datetime.timedelta(seconds=36000)))

        Raises an error if `camera_time_to_actual_time_ratio` cannot be determined
        >>> location_time = LocationTime(camera_start_time="2021-07-15T14:00")
        >>> location_corrector = location_time.corrector()
        Traceback (most recent call last):
        ...
        ValueError: Must set camera_time_to_actual_time_ratio or both end times
        """
        if camera_time_to_actual_time_ratio is None:
            camera_time_to_actual_time_ratio = self.get_time_ratio()

        if camera_time_to_actual_time_ratio is None:
            raise ValueError(
                "Must set camera_time_to_actual_time_ratio or both end times"
            )

        def datetime_corrector(datetime_original: datetime) -> datetime:
            # To quash some mypy errors but still allow proper type checking:
            assert isinstance(camera_time_to_actual_time_ratio, float)
            assert isinstance(self.actual_start_time, datetime)

            camera_elapsed_time = datetime_original - self.camera_start_time
            actual_elapsed_time = camera_elapsed_time / camera_time_to_actual_time_ratio
            return self.actual_start_time + actual_elapsed_time

        return datetime_corrector


class LocationTimeCollector(BaseModel):
    """Used to generate `SubDirDict` instances which map subdirectories to location
    strings or `DatetimeCorrector`s

    Parammeters
    -----------
    items: Dict[Path, LocationTime]

    Examples
    --------
    Here, `LocationTimeCollector` is instantiated with two `LocationTime` instances.
    The first has enough information to get a `float` from its `.get_time_rato` method,
    However the second doesn't.
    >>> lt_collector = LocationTimeCollector(items={
    ...     Path("data"): LocationTime(
    ...         camera_start_time="2021-07-15T14:00",
    ...         camera_end_time="2021-07-15T16:00",
    ...         actual_end_time="2021-07-15T15:00",
    ...     ),
    ...     Path("foo"): LocationTime(camera_start_time="2021-07-15T13:00"),
    ... })

    When calling the `.get_time_ratio` method on a `LocationTimeCollector` instance,
    the mean of all (excluding those who return `None`) `.get_time_ratio` results from
    all the `LocationTime`s is given.
    >>> lt_collector.get_time_ratio()
    2.0
    """

    items: Dict[Path, LocationTime]

    def get_time_ratio(self) -> Optional[float]:
        """Gets the mean of calling the `.get_time_ratio` method on each LocationTime
        in self.items

        Examples
        --------
        If there is not enough information to calculate a time ratio, `None` is returned
        >>> lt_collector = LocationTimeCollector(items={
        ...     Path("foo"): LocationTime(camera_start_time="2021-07-15T13:00"),
        ... })
        >>> print(lt_collector.get_time_ratio())
        None
        """
        time_ratios: List[float] = []
        for lt in self.items.values():
            time_ratio = lt.get_time_ratio()
            if time_ratio is not None:
                time_ratios.append(time_ratio)

        if len(time_ratios) == 0:
            return None
        return fsum(time_ratios) / len(time_ratios)

    def get_correctors(
        self, camera_time_to_actual_time_ratio: Optional[float] = None
    ) -> SubDirDict[DatetimeCorrector]:
        """Calls the `.corrector` method on each LocationTime in self.items to produce
        a SubDirDict of datetime_corrector functions.

        Parameters
        ----------
        camera_time_to_actual_time_ratio : Optional[float]
            The amount of time elapsed as reported by the camera divided by the actual
            amount of time elapsed. If None, then this is inferred by calling
            self.get_time_ratio(). If this returns None, a RuntimeError is raised.

        Returns
        -------
        datetime_correctors : SubDirDict[DatetimeCorrector]

        Examples
        --------
        >>> lt_collector = LocationTimeCollector(items={
        ...     Path("data"): LocationTime(
        ...         camera_start_time="2021-07-15T14:00",
        ...         camera_end_time="2021-07-15T16:00",
        ...         actual_end_time="2021-07-15T15:00",
        ...     ),
        ...     Path("foo"): LocationTime(camera_start_time="2021-07-15T13:00"),
        ... })
        >>> datetime_correctors = lt_collector.get_correctors(
        ...     camera_time_to_actual_time_ratio=2.0
        ... )
        >>> datetime_correctors["data"](datetime(2021, 7, 15, 18, 0))
        datetime.datetime(2021, 7, 15, 16, 0)

        If `camera_time_to_actual_time_ratio` is not set, it is calculated from
        the items in the `LocationTimeCollector`
        >>> datetime_correctors = lt_collector.get_correctors()
        >>> datetime_correctors["data"](datetime(2021, 7, 15, 18, 0))
        datetime.datetime(2021, 7, 15, 16, 0)

        But don't do this if there isn't enough information
        >>> lt_collector = LocationTimeCollector(items={
        ...     Path("foo"): LocationTime(camera_start_time="2021-07-15T13:00"),
        ... })
        >>> print(lt_collector.get_time_ratio())
        None
        >>> datetime_correctors = lt_collector.get_correctors()
        Traceback (most recent call last):
        ...
        RuntimeError: Unable to calculate camera-to-actual time ratio from time data
        """
        if camera_time_to_actual_time_ratio is None:
            camera_time_to_actual_time_ratio = self.get_time_ratio()

        if camera_time_to_actual_time_ratio is None:
            raise RuntimeError(
                "Unable to calculate camera-to-actual time ratio from time data"
            )

        datetime_correctors: SubDirDict[DatetimeCorrector] = SubDirDict()
        for directory, location_time in self.items.items():
            datetime_correctors[directory] = location_time.corrector(
                camera_time_to_actual_time_ratio
            )

        return datetime_correctors

    def get_location_dict(self) -> SubDirDict[Optional[str]]:
        """Returns a SubDirDict of location strings.

        Examples
        --------
        >>> lt_collector = LocationTimeCollector(items={
        ...     Path("data"): LocationTime(
        ...         camera_start_time="2021-07-15T14:00",
        ...         camera_end_time="2021-07-15T16:00",
        ...         actual_end_time="2021-07-15T15:00",
        ...         location="loc0",
        ...     ),
        ...     Path("foo"): LocationTime(
        ...         camera_start_time="2021-07-15T13:00",
        ...         location="loc1",
        ...     ),
        ...     Path("bar"): LocationTime(camera_start_time="2021-07-15T13:00"),
        ... })
        >>> lt_collector.get_location_dict()
        SubDirDict({Path('data'): 'loc0', Path('foo'): 'loc1', Path('bar'): None})
        """
        location_dict: SubDirDict[Optional[str]] = SubDirDict()
        for directory, location_time in self.items.items():
            location_dict[directory] = location_time.location

        return location_dict


class MaskMaker(BaseModel):
    shape: Tuple[PositiveInt, PositiveInt]
    mask_dilate: Optional[PositiveInt] = None

    def get_point_mask(self, point: PointShapeAttributes) -> Tensor:
        """Produces a mask Tensor for a given point.
        Raises a ValueError if point lies outise self.shape"""
        cx = array([int(point.cx)])
        cy = array([int(point.cy)])

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

    def get_circle_mask(self, circle: CircleShapeAttributes) -> Tensor:
        """Produces a mask Tensor for a given circle.
        Raises a ValueError if centre lies outise self.shape"""
        return self.get_point_mask(circle.as_point())

    def get_polyline_mask(self, polyline: PolylineShapeAttributes) -> Tensor:
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

    def get_mask(
        self,
        shape_attributes: Union[
            PointShapeAttributes, CircleShapeAttributes, PolylineShapeAttributes
        ],
    ) -> Tensor:
        if isinstance(shape_attributes, PointShapeAttributes):
            return self.get_point_mask(shape_attributes)
        elif isinstance(shape_attributes, CircleShapeAttributes):
            return self.get_circle_mask(shape_attributes)
        elif isinstance(shape_attributes, PolylineShapeAttributes):
            return self.get_polyline_mask(shape_attributes)

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
    ) -> None:
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
        """Returns true if bounding box is at least as tall as it is wide.

        Examples
        --------
        >>> BoundingBox(x0=1, y0=0, x1=11, y1=10).is_portrait()
        True
        >>> BoundingBox(x0=0, y0=0, x1=11, y1=10).is_portrait()
        False
        """
        return self.y1 - self.y0 >= self.x1 - self.x0

    def crop_image(self, image: Tensor) -> Tensor:
        """Returns a view of an image cropped to the BoundingBox

        Parameters
        ----------
        image : Tensor
            With shape [..., height, width]

        Returns
        -------
        Tensor
            with shape [..., self.y1 - self.y0, self.x1 - self.x0], assuming
            height <= self.y1 - self.y0 and width <= self.x1 - self.x0.

        Examples
        --------
        >>> box = BoundingBox(x0=7, x1=15, y0=3, y1=7)
        >>> grey_image = zeros(10, 20)
        >>> box.crop_image(grey_image).shape
        torch.Size([4, 8])
        >>> colour_image = zeros(3, 10, 20)
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


class TargetPredictionABC(BaseModel, ABC):
    boxes: List[BoundingBox]
    labels: List[PositiveInt]
    masks: List[Tensor]

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def all_fields_have_same_length(cls, values):
        try:
            length = len(values["boxes"])
            if not all(len(values[k]) == length for k in ["labels", "masks"]):
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

    def __len__(self):
        return len(self.labels)

    @abstractmethod
    def to_tensor_dict(self) -> Dict[str, Tensor]:
        """Send data to a dict of Tensors"""

    @classmethod
    @abstractmethod
    def from_tensor_dict(cls, tensor_dict: Dict[str, Tensor]) -> TargetPredictionABC:
        """Load Target or Prediction from tensor_dict"""


class Target(TargetPredictionABC):
    image_id: NonNegativeInt

    def to_tensor_dict(self) -> Dict[str, Tensor]:
        """Send data to a dict of Tensors"""
        return dict(
            boxes=tensor([[b.x0, b.y0, b.x1, b.y1] for b in self.boxes]),
            labels=tensor(self.labels),
            image_id=tensor([self.image_id]),
            masks=stack(self.masks),
        )

    @classmethod
    def from_tensor_dict(cls, tensor_dict: Dict[str, Tensor]) -> Target:
        """Load Target from tensor_dict"""
        return Target(
            boxes=[
                BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
                for x0, y0, x1, y1 in tensor_dict["boxes"]
            ],
            labels=[int(v) for v in tensor_dict["labels"]],
            masks=list(tensor_dict["masks"]),
            image_id=int(tensor_dict["image_id"]),
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
        Target(boxes=[], labels=[], masks=[], image_id=0)
        """
        return Target.construct(boxes=[], labels=[], image_id=image_id, masks=[])


class Prediction(TargetPredictionABC):
    scores: List[NonNegativeFloat]

    @root_validator
    def all_fields_have_same_length(cls, values):
        try:
            length = len(values["boxes"])
            if not all(len(values[k]) == length for k in ["labels", "masks", "scores"]):
                raise ValueError("Fields must have same length")
        except KeyError:
            raise ValueError("Invalid parameters given to Target")

        return values

    def to_tensor_dict(self) -> Dict[str, Tensor]:
        """Send data to a dict of Tensors"""
        return dict(
            boxes=tensor([[b.x0, b.y0, b.x1, b.y1] for b in self.boxes]),
            labels=tensor(self.labels),
            scores=tensor(self.scores),
            masks=stack(self.masks),
        )

    @classmethod
    def from_tensor_dict(cls, tensor_dict: Dict[str, Tensor]) -> Prediction:
        """Load Prediction from tensor_dict"""
        return Prediction(
            boxes=[
                BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
                for x0, y0, x1, y1 in tensor_dict["boxes"]
            ],
            labels=[int(v) for v in tensor_dict["labels"]],
            masks=list(tensor_dict["masks"]),
            scores=[int(v) for v in tensor_dict["scores"]],
        )

    @classmethod
    def empty(cls) -> Prediction:
        """Initialises an Prediction with no data.

        Returns
        -------
        Prediction

        Examples
        --------
        >>> Prediction.empty()
        Prediction(boxes=[], labels=[], masks=[], scores=[])
        """
        return Prediction.construct(boxes=[], labels=[], masks=[], scores=[])

    def filter_by_score(self, score_thresh: float) -> Prediction:
        """Returns a Prediction instance with items with scores below `score_thresh`
        removed.

        Parameters
        ----------
        score_thresh: float

        Returns
        -------
        Prediction
        """
        filtered_prediction = Prediction.empty()
        for i in range(len(self)):
            if self.scores[i] >= score_thresh:
                filtered_prediction.boxes.append(self.boxes[i])
                filtered_prediction.labels.append(self.labels[i])
                filtered_prediction.masks.append(self.masks[i])
                filtered_prediction.scores.append(self.scores[i])

        return filtered_prediction

    def get_subset_from_index(self, subset: List[NonNegativeInt]) -> Prediction:
        """Returns a Prediction instance from self with items indexed by the elements
        of `subset`.

        Parameters
        ----------
        subset : List[NonNegativeInt]
            List of indices

        Returns
        -------
        Prediction

        Examples
        --------
        >>> prediction = Prediction(
        ...     boxes=[
        ...         BoundingBox(x0=0, y0=0, x1=1, y1=1),
        ...         BoundingBox(x0=1, y0=1, x1=2, y1=2),
        ...     ],
        ...     labels=[1, 2],
        ...     masks=[zeros(2, 2), zeros(2, 2)],
        ...     scores=[0.0, 1.0],
        ... )
        >>> subset_prediction = prediction.get_subset_from_index([0])
        >>> subset_prediction.boxes == prediction.boxes[0:1]
        True
        >>> subset_prediction.labels == prediction.labels[0:1]
        True
        >>> len(subset_prediction.masks) == 1
        True
        >>> subset_prediction.scores == prediction.scores[0:1]
        True
        """
        filtered_prediction = Prediction.empty()
        for i in subset:
            filtered_prediction.boxes.append(self.boxes[i])
            filtered_prediction.labels.append(self.labels[i])
            filtered_prediction.masks.append(self.masks[i])
            filtered_prediction.scores.append(self.scores[i])

        return filtered_prediction


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


class CamfiDataset(BaseModel, Dataset):
    root: Path
    via_project: ViaProject
    crop: Optional[BoundingBox] = None

    inference_mode: bool = False

    # Only set if inference_mode = False
    mask_maker: Optional[MaskMaker] = None
    transform: Optional[ImageTransform] = None
    min_annotations: int = 0
    max_annotations: float = inf
    box_margin: PositiveInt = 10

    # Optionally exclude some files
    exclude: set = None  # type: ignore[assignment]

    # Automatically generated. No need to set.
    keys: List = None  # type: ignore[assignment]

    @validator("exclude", pre=True, always=True)
    def default_exclude(cls, v):
        if v is None:
            return set()
        return v

    @validator("transform")
    def only_set_transform_if_not_inference_mode(cls, v, values):
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v is None, "Only set if inference_mode=False"
        return v

    @validator("min_annotations")
    def only_set_min_annotations_if_not_inference_mode(cls, v, values):
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v is 0, "Only set if inference_mode=False"
        return v

    @validator("max_annotations")
    def only_set_max_annotations_if_not_inference_mode(cls, v, values):
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v is inf, "Only set if inference_mode=False"
        return v

    @validator("box_margin")
    def only_set_box_margin_if_not_inference_mode(cls, v, values):
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v is 10, "Only set if inference_mode=False"
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
        image = metadata.read_image(root=self.root)

        if self.crop is not None:
            image = self.crop.crop_image(image)

        if self.inference_mode:
            target = Target.empty(image_id=idx)
        else:  # Training mode
            boxes = metadata.get_bounding_boxes()
            for box in boxes:
                box.add_margin(self.box_margin)
            target = Target(
                boxes=boxes,
                labels=metadata.get_labels(),
                image_id=idx,
                masks=self.mask_maker.get_masks(metadata),  # type: ignore[union-attr]
            )

            if self.transform is not None:
                image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.keys)

    def metadata(self, idx: int) -> ViaMetadata:
        """Returns the ViaMetadata object of the image at idx."""
        return self.via_project.via_img_metadata[self.keys[idx]]
