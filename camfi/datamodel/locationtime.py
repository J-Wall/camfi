"""Provides classes for operating on location and time information for camera
placements. Depends on camfi.util."""

from __future__ import annotations

from datetime import datetime
from math import fsum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, root_validator, validator

from camfi.util import DatetimeCorrector, SubDirDict, Field


class LocationTime(BaseModel):
    """Defines camera placement data. Specifically, it includes camera placement time
    and camera retrieval time (both according to the camera's clock, and real time), as
    well as a string specifying the location the camera was placed. Provides a the
    .corrector method, which returns a DatetimeCorrector function for inferring
    real time from camera timestamps (useful if cameras have inaccurate clocks).

    Parameters
    ----------
    camera_start_time : datetime
        Timestamp from when the camera was placed (according to the camera's clock).
    actual_start_time : Optional[datetime]
        Actual timestamp from when the camera was placed (real time).
        Defaults to camera_start_time.
    camera_end_time : Optional[datetime]
        Timestamp from when the camera was retrieved (according to the camera's clock).
    actual_end_time : Optional[datetime]
        Actual timestamp from when the camera was retrieved (real time).
    location : Optional[str]
        String specifying the location the camera was placed in.
    """

    camera_start_time: datetime = Field(
        ..., description="Camera placement datetime (according to camera's clock)."
    )
    actual_start_time: Optional[datetime] = Field(
        None,
        description="Actual camera placement datetime (defaults to camera_start_time).",
    )
    camera_end_time: Optional[datetime] = Field(
        None, description="Camera retrieval time (according to camera's clock)."
    )
    actual_end_time: Optional[datetime] = Field(
        None, description="Actual camera retrieval time."
    )
    location: Optional[str] = Field(
        None, description="Name of location the camera was placed in."
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

        @staticmethod
        def schema_extra(schema: dict[str, Any], model: LocationTime) -> None:
            # Remove title from schema
            schema.pop("title", None)
            schema["description"] = "Camera placement data"

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
            (Time elapsed on cameras cloack) / (Real time elapsed)

        Examples
        --------
        >>> location_time = LocationTime(
        ...     camera_start_time="2021-07-15T14:00",
        ...     camera_end_time="2021-07-15T16:00",
        ...     actual_end_time="2021-07-15T15:00",
        ... )
        >>> location_time.get_time_ratio()
        2.0

        camera_end_time and actual_end_time must be set, or else None is returned.

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
        self.camera_start_time reflects the actual time. Generally it is advised to
        set the time of the camera just before it is placed out, which is the basis for
        this assumption.

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
            Function which maps camera time to real time.

        Examples
        --------
        >>> location_time = LocationTime(camera_start_time="2021-07-15T14:00")
        >>> location_corrector = location_time.corrector(2.)
        >>> location_corrector(datetime(2021, 7, 15, 16, 0))
        datetime.datetime(2021, 7, 15, 15, 0)

        Also works with offset-aware datetimes.

        >>> location_time = LocationTime(camera_start_time="2021-07-15T14:00+10")
        >>> location_corrector = location_time.corrector(2.)
        >>> location_corrector(datetime.fromisoformat("2021-07-15T17:00+11:00"))
        datetime.datetime(2021, 7, 15, 15, 0, tzinfo=datetime.timezone(datetime.timedelta(seconds=36000)))

        If self.camera_start_time is offset aware, but the argument to the
        DatetimeCorrector function is not, it will be assumed that the argument has the
        same offset as self.camera_start_time.

        >>> location_time = LocationTime(camera_start_time="2021-07-15T14:00+10")
        >>> location_corrector = location_time.corrector(2.)
        >>> location_corrector(datetime.fromisoformat("2021-07-15T16:00"))
        datetime.datetime(2021, 7, 15, 15, 0, tzinfo=datetime.timezone(datetime.timedelta(seconds=36000)))

        However, the converse raises a TypeError.

        >>> location_time = LocationTime(camera_start_time="2021-07-15T14:00")
        >>> location_corrector = location_time.corrector(2.)
        >>> location_corrector(datetime.fromisoformat("2021-07-15T16:00+10:00"))
        Traceback (most recent call last):
        ...
        TypeError: Cannot call offset-naive DatetimeCorrector with offset-aware datetime.

        Raises an error if camera_time_to_actual_time_ratio cannot be determined.

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

            if datetime_original.tzinfo is None:
                datetime_original = datetime_original.replace(
                    tzinfo=self.camera_start_time.tzinfo
                )
            elif self.camera_start_time.tzinfo is None:
                raise TypeError(
                    "Cannot call offset-naive DatetimeCorrector with offset-aware datetime."
                )

            camera_elapsed_time = datetime_original - self.camera_start_time
            actual_elapsed_time = camera_elapsed_time / camera_time_to_actual_time_ratio
            return self.actual_start_time + actual_elapsed_time

        return datetime_corrector


class LocationTimeCollector(BaseModel):
    """Used to generate SubDirDict instances which map subdirectories to location
    strings or DatetimeCorrectors.

    Parameters
    -----------
    camera_placements: dict[str, LocationTime]
        dictionary mapping directories to LocationTime instances. For example, you
        might have one LocationTime for each camera placement, and each camera placement
        also has it's own directory for images.

    Examples
    --------
    Here, LocationTimeCollector is instantiated with two LocationTime instances.
    The first has enough information to get a float from its .get_time_rato method,
    However the second doesn't.

    >>> lt_collector = LocationTimeCollector(camera_placements={
    ...     "data": LocationTime(
    ...         camera_start_time="2021-07-15T14:00",
    ...         camera_end_time="2021-07-15T16:00",
    ...         actual_end_time="2021-07-15T15:00",
    ...     ),
    ...     "foo": LocationTime(camera_start_time="2021-07-15T13:00"),
    ... })

    When calling the .get_time_ratio method on a LocationTimeCollector instance,
    the mean of all (excluding those who return None) .get_time_ratio results from
    all the LocationTimes is given.

    >>> lt_collector.get_time_ratio()
    2.0
    """

    camera_placements: dict[str, LocationTime] = Field(
        ...,
        description="Mapping from sub-directories containing images to LocationTime instances.",
    )

    class Config:
        schema_extra = {"description": "Contains data for multiple camera placements."}

    def get_time_ratio(self) -> Optional[float]:
        """Gets the mean of calling the .get_time_ratio method on each LocationTime
        in self.camera_placements.

        Returns
        -------
        camera_time_to_actual_time_ratio : float
            The mean of (Time elapsed on cameras cloack) / (Real time elapsed) for all
            camera placements (LocationTime instances) in self which have sufficent
            data to determine this value.

        Examples
        --------
        If there is not enough information to calculate a time ratio, None is
        returned.

        >>> lt_collector = LocationTimeCollector(camera_placements={
        ...     "foo": LocationTime(camera_start_time="2021-07-15T13:00"),
        ... })
        >>> print(lt_collector.get_time_ratio())
        None
        """
        time_ratios: list[float] = []
        for lt in self.camera_placements.values():
            time_ratio = lt.get_time_ratio()
            if time_ratio is not None:
                time_ratios.append(time_ratio)

        if len(time_ratios) == 0:
            return None
        return fsum(time_ratios) / len(time_ratios)

    def get_correctors(
        self, camera_time_to_actual_time_ratio: Optional[float] = None
    ) -> SubDirDict[DatetimeCorrector]:
        """Calls the .corrector method on each LocationTime in self.camera_placements to
        produce a SubDirDict of datetime_corrector functions.

        Parameters
        ----------
        camera_time_to_actual_time_ratio : Optional[float]
            The amount of time elapsed as reported by the camera divided by the actual
            amount of time elapsed. If None, then this is inferred by calling
            self.get_time_ratio(). If this returns None, a RuntimeError is raised.

        Returns
        -------
        datetime_correctors : SubDirDict[DatetimeCorrector]
            A mapping from subdirectories to DatetimeCorrector functions.

        Examples
        --------
        >>> lt_collector = LocationTimeCollector(camera_placements={
        ...     "data": LocationTime(
        ...         camera_start_time="2021-07-15T14:00",
        ...         camera_end_time="2021-07-15T16:00",
        ...         actual_end_time="2021-07-15T15:00",
        ...     ),
        ...     "foo": LocationTime(camera_start_time="2021-07-15T13:00"),
        ... })
        >>> datetime_correctors = lt_collector.get_correctors(
        ...     camera_time_to_actual_time_ratio=2.0
        ... )
        >>> datetime_correctors["data"](datetime(2021, 7, 15, 18, 0))
        datetime.datetime(2021, 7, 15, 16, 0)

        If camera_time_to_actual_time_ratio is not set, it is calculated from
        the camera_placements in the LocationTimeCollector

        >>> datetime_correctors = lt_collector.get_correctors()
        >>> datetime_correctors["data"](datetime(2021, 7, 15, 18, 0))
        datetime.datetime(2021, 7, 15, 16, 0)

        But don't do this if there isn't enough information

        >>> lt_collector = LocationTimeCollector(camera_placements={
        ...     "foo": LocationTime(camera_start_time="2021-07-15T13:00"),
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
        for directory, location_time in self.camera_placements.items():
            datetime_correctors[Path(directory)] = location_time.corrector(
                camera_time_to_actual_time_ratio
            )

        return datetime_correctors

    def get_location_dict(self) -> SubDirDict[Optional[str]]:
        """Produces a SubDirDict of location strings.

        Returns
        -------
        location_dict : SubDirDict
            A mapping from subdirectories to location strings.

        Examples
        --------
        >>> lt_collector = LocationTimeCollector(camera_placements={
        ...     "data": LocationTime(
        ...         camera_start_time="2021-07-15T14:00",
        ...         camera_end_time="2021-07-15T16:00",
        ...         actual_end_time="2021-07-15T15:00",
        ...         location="loc0",
        ...     ),
        ...     "foo": LocationTime(
        ...         camera_start_time="2021-07-15T13:00",
        ...         location="loc1",
        ...     ),
        ...     "bar": LocationTime(camera_start_time="2021-07-15T13:00"),
        ... })
        >>> lt_collector.get_location_dict()
        SubDirDict({Path('data'): 'loc0', Path('foo'): 'loc1', Path('bar'): None})
        """
        location_dict: SubDirDict[Optional[str]] = SubDirDict()
        for directory, location_time in self.camera_placements.items():
            location_dict[Path(directory)] = location_time.location

        return location_dict
