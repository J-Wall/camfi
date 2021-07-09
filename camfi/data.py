from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, validator, constr, conint, confloat


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
    shape_attributes: ViaShapeAttributes


class ViaMetadata(BaseModel):
    file_attributes: ViaFileAttributes
    filename: Path
    regions: List[ViaRegion]
    size: int = -1
