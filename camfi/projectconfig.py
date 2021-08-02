"""Provides camfi project config parser.
"""

from __future__ import annotations

from datetime import date, timezone
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional, Union, Sequence

import pandas as pd
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PositiveFloat,
    ValidationError,
    root_validator,
    validator,
)

from camfi.datamodel.locationtime import LocationTimeCollector
from camfi.datamodel.via import ViaProject
from camfi.datamodel.weather import LocationWeatherStationCollector
from camfi.util import Timezone
from camfi.wingbeat import WingbeatExtractorConfig, extract_all_wingbeats


class ParameterUnspecifiedError(Exception):
    """Base exception called when a parameter which needs to be specified is not."""


class ViaProjectUnspecifiedError(ParameterUnspecifiedError):
    """Raised by CamfiConfig.load_via_project."""


class PlaceUnspecifiedError(ParameterUnspecifiedError):
    """Raised when CamfiConfig.place is requested, but it is unspecified."""


class WingbeatExtractorConfigUnspecifiedError(ParameterUnspecifiedError):
    """Raised when a WingbeatExtractorConfig is needed, but was not supplied."""


class CameraConfigUnspecifiedError(ParameterUnspecifiedError):
    """Raised when a method requies a camera config but none was supplied."""


class CameraConfig(BaseModel):
    """Camera hardware-related configuration."""

    camera_time_to_actual_time_ratio: Optional[float] = Field(
        None, description="Used for correcting timestamps from an inaccurate clock."
    )
    line_rate: Optional[PositiveFloat] = Field(
        None, description="Rolling-shutter line rate of camera (lines per second)."
    )


class CamfiConfig(BaseModel):
    """Defines structure of Camfi's config.json files, and provides methods for loading
    data from various sources.
    """

    root: DirectoryPath = Field(
        Path(), description="Directory containing all images for the project."
    )
    via_project_file: Optional[FilePath] = Field(
        None, description="Path to file containing VIA project."
    )
    day_zero: Optional[date] = Field(
        None, description="Used as reference date for plotting etc."
    )
    output_tz: Timezone = Field(
        description="Sets a global timezone used for various analyses.",
        default_factory=lambda: Timezone("Z"),
    )
    camera: Optional[CameraConfig] = None
    time: Optional[LocationTimeCollector] = None
    place: Optional[LocationWeatherStationCollector] = None
    wingbeat_extraction: Optional[WingbeatExtractorConfig] = None

    @property
    def timestamp_zero(self) -> Optional[pd.Timestamp]:
        if self.day_zero is None:
            return None
        return pd.to_datetime(self.day_zero).tz_localize(self.output_tz._timezone)

    @cached_property
    def via_project(self) -> ViaProject:
        """Loads ViaProject from file. Raises ViaProjectUnspecifiedError if
        self.via_project is None.

        Returns
        -------
        via_project : ViaProject
            VIA project loaded from self.via_project file.
        """
        if self.via_project_file is None:
            raise ViaProjectUnspecifiedError
        return ViaProject.parse_file(self.via_project_file)

    class Config:
        json_encoders = {Timezone: str}
        keep_untouched = (cached_property,)
        schema_extra = {"description": "Camfi configuration."}

    @root_validator
    def check_all_loctions_defined(cls, values):
        if values.get("time") is None:
            return values
        default_place = LocationWeatherStationCollector(
            locations=[], weather_stations=[], location_weather_station_mapping={}
        )
        specified_locations = set(
            location.name for location in values.get("place", default_place).locations
        )
        for camera_placement in values["time"].camera_placements.values():
            if camera_placement.location not in specified_locations:
                raise ValidationError(
                    f"location {camera_placement.location} unspecified in place field."
                )

        return values

    def get_image_dataframe(self) -> pd.DataFrame:
        """Calls self.via_project.to_image_dataframe(tz=self.output_tz), returning the
        result."""
        if self.place is None:
            raise PlaceUnspecifiedError
        return self.via_project.to_image_dataframe(tz=self.output_tz._timezone)

    def get_weather_dataframe(self) -> pd.DataFrame:
        """Calls self.place.get_weather_dataframe(), returning the result."""
        if self.place is None:
            raise PlaceUnspecifiedError
        return self.place.get_weather_dataframe()

    def get_sun_time_dataframe(
        self, days: Union[str, Dict[str, Sequence[date]]]
    ) -> pd.DataFrame:
        """Calls self.place.get_sun_time_dataframe"""
        if self.place is None:
            raise PlaceUnspecifiedError
        if days == "images":
            image_df = self.get_image_dataframe()
            days = {}
            for location in self.place.locations:
                days[location.name] = image_df[image_df["location"] == location.name][
                    "datetime_corrected"
                ].dt.date.unique()

        elif days == "weather":
            weather_df = self.get_weather_dataframe()
            days = {}
            for location in self.place.locations:
                weather_station: str = self.place.location_weather_station_mapping[
                    location.name
                ]
                days[location.name] = weather_df.loc[weather_station].index.date

        if isinstance(days, dict):
            return self.place.get_sun_time_dataframe(days)

        raise TypeError(
            "days must be one of ['images' | 'weather' | Dict[str, Sequence[date]]]. "
            f"Got {days} of type {type(days)}."
        )

    def get_merged_dataframe(self) -> pd.DataFrame:
        if self.place is None:
            raise PlaceUnspecifiedError
        image_df = self.get_image_dataframe()
        image_df["date"] = pd.to_datetime(image_df["datetime_corrected"].dt.date)

        days = {}
        for location in self.place.locations:
            days[location.name] = image_df[image_df["location"] == location.name][
                "date"
            ].dt.date.unique()

        weather_sun_df = self.place.get_weather_sun_dataframe(days=days)

        merged_df = pd.merge(
            image_df, weather_sun_df, how="left", on=["location", "date"], sort=True
        )
        merged_df.set_index(["location", "date"], inplace=True)

        return merged_df

    def load_all_exif_metadata(self) -> None:
        """Calls self.via_project.load_all_exif_metadata with appropriate arguments,
        set by config. Operates in place.
        """
        time_ratio = None
        if self.camera is not None:
            time_ratio = self.camera.camera_time_to_actual_time_ratio

        location_dict = None
        datetime_correctors = None
        if self.time is not None:
            location_dict = self.time.get_location_dict()
            datetime_correctors = self.time.get_correctors(
                camera_time_to_actual_time_ratio=time_ratio
            )

        self.via_project.load_all_exif_metadata(
            root=self.root,
            location_dict=location_dict,
            datetime_correctors=datetime_correctors,
        )

    def extract_all_wingbeats(self) -> None:
        """Calls extract_all_wingbeats on self.via_project with parameters taken from
        configuration."""
        if self.wingbeat_extraction is None:
            raise WingbeatExtractorConfigUnspecifiedError
        if self.camera is None or self.camera.line_rate is None:
            raise CameraConfigUnspecifiedError
        extract_all_wingbeats(
            self.via_project,
            root=self.root,
            line_rate=self.camera.line_rate,
            **self.wingbeat_extraction.dict(),
        )
