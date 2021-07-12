from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from math import inf
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union

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
from torch import stack, tensor, Tensor, zeros
from torch.utils.data import Dataset
from torchvision.io import read_image

from camfi.util import smallest_enclosing_circle, dilate_idx


class ViaFileAttributes(BaseModel):
    datetime_corrected: Optional[datetime]
    datetime_original: Optional[datetime]
    exposure_time: Optional[PositiveFloat]
    location: Optional[str]
    pixel_x_dimension: Optional[PositiveInt]
    pixel_y_dimension: Optional[PositiveInt]


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


class PointShapeAttributes(ViaShapeAttributes):
    cx: NonNegativeFloat
    cy: NonNegativeFloat
    name: str = Field("point", regex=r"^point$")

    def get_bounding_box(self) -> BoundingBox:
        return BoundingBox(
            x0=int(self.cx), y0=int(self.cy), x1=int(self.cx) + 1, y1=int(self.cy) + 1
        )


class CircleShapeAttributes(ViaShapeAttributes):
    cx: NonNegativeFloat
    cy: NonNegativeFloat
    name: str = Field("circle", regex=r"^circle$")
    r: NonNegativeFloat

    def as_point(self):
        return PointShapeAttributes(cx=self.cx, cy=self.cy)

    def get_bounding_box(self) -> BoundingBox:
        return self.as_point().get_bounding_box()


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


class ViaRegion(BaseModel):
    region_attributes: ViaRegionAttributes
    shape_attributes: Union[
        PolylineShapeAttributes, CircleShapeAttributes, PointShapeAttributes
    ]

    def get_bounding_box(self):
        return self.shape_attributes.get_bounding_box()


class ViaMetadata(BaseModel):
    file_attributes: ViaFileAttributes
    filename: Path
    regions: List[ViaRegion]
    size: int = -1


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
        self, shape: Tuple[PositiveInt, PositiveInt], mask_dilate: Optional[PositiveInt]
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
    transforms: ImageTransform
    via_project: ViaProject
    crop: Optional[Tuple[int, int, int, int]] = None
    mask_makers: Dict[str, MaskMaker]
    min_annotations: int = 0
    max_annotations: float = inf
    inference_mode: bool = False
    exclude: Optional[set] = None

    class Config:
        arbitrary_types_allowed = True
