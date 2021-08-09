"""Defines the ViaRegionAttributes BaseModel class.
"""

from typing import Optional

from pydantic import BaseModel

from camfi.util import Field


class ViaRegionAttributes(BaseModel):
    """Contains object annotation (region)-level metadata.

    Parameters
    ----------
    score : Optional[float]
        Score of annotation. This is only relevant for annotations which have been
        obtained automatically. Score should not be set for manual annotations.
    best_peak : Optional[int]
        Period of wingbeat in pixels.
    blur_length : Optional[float]
        Length of motion blur in pixels.
    snr : Optional[float]
        Signal-to-noise ratio of best peak.
    wb_freq_up : Optional[float]
        Wingbeat frequency estimate, assuming upward motion (and zero body-length).
    wb_freq_down : Optional[float]
        Wingbeat frequency estimate, assuming downward motion (and zero body-length).
    et_up : Optional[float]
        Corrected moth exposure time, assuming upward motion.
    et_dn : Optional[float]
        Corrected moth exposure time, assuming downward motion.
    """

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
