from abc import ABC, abstractmethod
from datetime import datetime
from math import inf
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    validator,
    constr,
    conint,
    confloat,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image


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


class PolylineShapeAttributes(ViaShapeAttributes):
    all_points_x: List[float]
    all_points_y: List[float]
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


class CircleShapeAttributes(ViaShapeAttributes):
    cx: float = Field(..., ge=0)
    cy: float = Field(..., ge=0)
    name: str = Field("circle", regex=r"^circle$")
    r: float = Field(..., ge=0)


class PointShapeAttributes(ViaShapeAttributes):
    cx: float = Field(..., ge=0)
    cy: float = Field(..., ge=0)
    name: str = Field("point", regex=r"^point$")


class ViaRegion(BaseModel):
    region_attributes: ViaRegionAttributes
    shape_attributes: Union[
        PolylineShapeAttributes, CircleShapeAttributes, PointShapeAttributes
    ]


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


class ImageTransform(BaseModel, ABC):
    """Abstract base class for transforms on images with segmentation data."""

    @abstractmethod
    def __call__(
        self, image: Tensor, target: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        pass


class TransformCompose(ImageTransform):
    def __init__(self, transforms: List[ImageTransform]):
        self.transforms = transforms

    def __call__(
        self, image: Tensor, target: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class TransformRandomHorizontalFlip(ImageTransform):
    def __init__(self, prob):
        self.prob = prob

    def __call__(
        self, image: Tensor, target: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class MaskMaker(BaseModel, ABC):
    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape

    @abstractmethod
    def get_mask(self, shape: ViaShapeAttributes) -> Tensor:
        pass


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
