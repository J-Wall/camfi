"""Provides utilities for working with weather data, and for calculating the movement
of the sun from the persepcitve of specific locations.

Constants defined in this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These can all be overwritten by setting environment variables with the same name.

* ``SKYFIELD_DATA_DIR``. Path to directory to save skyfield ephemeris data. By default,
  ``~/skyfield-data`` will be used.
* ``CAMFI_EPHEMERIS``. Name of ephemeris file to use for calculating sunset and
  twilight. By default, ``de440s.bsp`` will be used. See the `choosing an ephemeris`_
  in the Skyfield documentation for possible other ephemeris files to use. Note that the
  ephemeris file will be loaded when this module is imported. The first time this
  happens, the ephemeris file will be downloaded (def4402.bsp is about 37 mb).

**Note:** For testing purposes,
a tiny abbreviated ephemeris file is included
in the Camfi git repository.
Without this, CI testing wouldn't work.
The included ephemeris limits
the date range to a few days in 2021,
so it should not be used when running Camfi normally.
To use the included ephemeris (for testing purposes only),
set the following environment variables
(assuming you are running the tests from the camfi repo root directory)::

    SKYFIELD_DATA_DIR="camfi/test/data"
    CAMFI_EPHEMERIS="test_ephem.bsp"

.. _choosing an ephemeris: https://rhodesmill.org/skyfield/planets.html#choosing-an-ephemeris
"""

from datetime import date, datetime, time, timedelta, timezone
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    FilePath,
    NonNegativeFloat,
    validator,
)
from skyfield.api import Loader, wgs84
from skyfield import almanac

from camfi.util import Timezone, Field


# Initialise skyfield
SKYFIELD_DATA_DIR = os.getenv(
    "SKYFIELD_DATA_DIR", str(Path("~/skyfield-data").expanduser())
)
CAMFI_EPHEMERIS = os.getenv("CAMFI_EPHEMERIS", "de440s.bsp")

_load = Loader(SKYFIELD_DATA_DIR)
ephemeris = _load(CAMFI_EPHEMERIS)
timescale = _load.timescale()

TWILIGHT_TRANSITIONS = {
    1: "astronomical_twilight_start",
    2: "nautical_twilight_start",
    3: "civil_twilight_start",
    4: "sunrise",
    5: "astronomical_twilight_end",
    6: "nautical_twilight_end",
    7: "civil_twilight_end",
    8: "sunset",
}


class Location(BaseModel):
    """Provides methods for working with locations.

    Parameters
    ----------
    name : str
        Name of location.
    lat : float
        Decimal latitude.
    lon : float
        Decimal longitude.
    elevation_m : NonNegativeFloat
        Elevation in metres.
    tz : timezone
        Timezone offset. Can be given as ISO8601 timezone offset str (e.g. 'Z' or
        '+10:00' or simply '+10').

    Examples
    --------
    >>> Location(
    ...     name="canberra",
    ...     lat=-35.293056,
    ...     lon=149.126944,
    ...     elevation_m=578,
    ...     tz="+10:00",
    ... )
    Location(name='canberra', lat=-35.293056, lon=149.126944, elevation_m=578.0, tz=Timezone(datetime.timezone(datetime.timedelta(seconds=36000))))
    >>> Location(
    ...     name="greenwich",
    ...     lat=51.48,
    ...     lon=0,
    ...     elevation_m=47,
    ...     tz="Z",
    ... )
    Location(name='greenwich', lat=51.48, lon=0.0, elevation_m=47.0, tz=Timezone(datetime.timezone.utc))
    >>> Location(
    ...     name="nyc",
    ...     lat=40.712778,
    ...     lon=-74.006111,
    ...     elevation_m=10,
    ...     tz="-05",
    ... )
    Location(name='nyc', lat=40.712778, lon=-74.006111, elevation_m=10.0, tz=Timezone(datetime.timezone(datetime.timedelta(days=-1, seconds=68400))))
    """

    name: str = Field(
        ..., description="Name of location. Used to link to camera placements."
    )
    lat: float = Field(..., description="Decimal latitude.")
    lon: float = Field(..., description="Decimal longitude.")
    elevation_m: NonNegativeFloat = Field(..., description="Elevation in metres.")
    tz: Timezone = Field(..., description="ISO8601 timezone offset.")

    class Config:
        schema_extra = {
            "description": "Contains spatial data on locations, including timezone."
        }

    @property
    def _dark_twilight_day(self):
        return almanac.dark_twilight_day(
            ephemeris, wgs84.latlon(self.lat, self.lon, elevation_m=self.elevation_m)
        )

    def _get_tz_aware_dt(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz._timezone)
        return dt

    def twilight_state(self, dt: datetime) -> int:
        """Gets the twilight state for the location at the specified time(s).

        The meanings of the returned integer values are

        0. Dark of night.
        1. Astronomical twilight.
        2. Nautical twilight.
        3. Civil twilight.
        4. Daytime.

        Parameters
        ----------
        dt : datetime
            datetime to evaluate. If timezone-naive, timezone will be taken from
            self.tz.

        Returns
        -------
        ts : int
            Twilight value.

        Examples
        --------
        >>> location = Location(
        ...     name="canberra",
        ...     lat=-35.293056,
        ...     lon=149.126944,
        ...     elevation_m=578,
        ...     tz="+10:00",
        ... )
        >>> location.twilight_state(datetime.fromisoformat("2021-07-28T12:00:00+10:00"))
        4
        >>> location.twilight_state(datetime.fromisoformat("2021-07-28T23:00:00+11:00"))
        0

        Timezone will be taken from self.tz if dt is timezone-naive.

        >>> location.twilight_state(datetime.fromisoformat("2021-07-28T12:00:00"))
        4

        Skyfield provides mapping from these numbers to strings.

        >>> almanac.TWILIGHTS[0]
        'Night'
        >>> almanac.TWILIGHTS[1]
        'Astronomical twilight'
        >>> almanac.TWILIGHTS[2]
        'Nautical twilight'
        >>> almanac.TWILIGHTS[3]
        'Civil twilight'
        >>> almanac.TWILIGHTS[4]
        'Day'
        """
        time = timescale.from_datetime(self._get_tz_aware_dt(dt))
        return int(self._dark_twilight_day(time))

    def twilight_states(self, datetimes: Sequence[datetime]) -> np.ndarray:
        """Like Location.twilight_state but operates on sequence of datetimes.

        Parameters
        ----------
        datetimes : Sequence[datetime]
            datetimes to evaluate. If timezone-naive, timezone will be taken from
            self.tz.

        Returns
        -------
        ts : np.ndarray
            Twilight values.

        Examples
        --------
        >>> location = Location(
        ...     name="canberra",
        ...     lat=-35.293056,
        ...     lon=149.126944,
        ...     elevation_m=578,
        ...     tz=timezone(timedelta(hours=10)),
        ... )
        >>> datetimes = [
        ...     datetime.fromisoformat("2021-07-28T12:00:00+10:00"),
        ...     datetime.fromisoformat("2021-07-28T23:00:00+11:00"),
        ...     datetime.fromisoformat("2021-07-28T12:00:00"),
        ... ]
        >>> location.twilight_states(datetimes)
        array([4, 0, 4])
        """
        datetimes = [self._get_tz_aware_dt(dt) for dt in datetimes]
        times = timescale.from_datetimes(datetimes)
        return self._dark_twilight_day(times)

    def search_sun_times(self, day: date) -> dict[str, datetime]:
        """Gets sunrise, sunset, and twilight times for a given date.

        Parameters
        ----------
        day : date
            Day to get times from.

        Returns
        -------
        twilight_times : dict[str, datetime]
            dictionary with keys "astronomical_twilight_start",
            "nautical_twilight_start", "civil_twilight_start", "sunrise", "sunset",
            "nautical_twilight_end", "civil_twilight_end", "astronomical_twilight_end".

        Examples
        --------
        >>> location = Location(
        ...     name="canberra",
        ...     lat=-35.293056,
        ...     lon=149.126944,
        ...     elevation_m=578,
        ...     tz=timezone(timedelta(hours=10)),
        ... )
        >>> day = date(2021, 7, 28)
        >>> tt = location.search_sun_times(day)

        The ordering of the transitions is as expected.

        >>> tt["astronomical_twilight_start"] < tt["nautical_twilight_start"]
        True
        >>> tt["nautical_twilight_start"] < tt["civil_twilight_start"]
        True
        >>> tt["civil_twilight_start"] < tt["sunrise"]
        True
        >>> tt["sunrise"] < tt["sunset"]
        True
        >>> tt["sunset"] < tt["civil_twilight_end"]
        True
        >>> tt["civil_twilight_end"] < tt["nautical_twilight_end"]
        True
        >>> tt["nautical_twilight_end"] < tt["astronomical_twilight_end"]
        True

        And all of the datetimes are on the correct day.

        >>> all(d.date() == day for d in tt.values())
        True
        """
        start_time = datetime.combine(date=day, time=time(0), tzinfo=self.tz._timezone)
        end_time = start_time + timedelta(days=1)
        t0 = timescale.from_datetime(start_time)
        t1 = timescale.from_datetime(end_time)

        times, twilight_types = almanac.find_discrete(t0, t1, self._dark_twilight_day)
        twilight_transitions = (
            np.roll(twilight_types, 1) > twilight_types
        ) * 5 + twilight_types

        twilight_times: dict[str, datetime] = {}
        for t, tt in zip(times, twilight_transitions):
            twilight_times[TWILIGHT_TRANSITIONS[tt]] = t.utc_datetime().astimezone(
                self.tz._timezone
            )

        return twilight_times

    def get_sun_time_dataframe(self, days: Sequence[date]) -> pd.DataFrame:
        """Calls self.search_sun_times on each day in days, and builds a DataFrame of
        sun times.

        Parameters
        ----------
        days : Sequence[date]
            Dates which will become index for dataframe.

        Returns
        -------
        sun_df : pd.DataFrame
            DataFrame indexed by location and date, with columns
            "astronomical_twilight_start", "nautical_twilight_start",
            "civil_twilight_start", "sunrise", "sunset", "nautical_twilight_end",
            "civil_twilight_end", "astronomical_twilight_end".

        Examples
        --------
        >>> location = Location(
        ...     name="canberra",
        ...     lat=-35.293056,
        ...     lon=149.126944,
        ...     elevation_m=578,
        ...     tz=timezone(timedelta(hours=10)),
        ... )
        >>> days = [date(2021, 7, 23), date(2021, 7, 24), date(2021, 7, 25)]
        >>> sun_df = location.get_sun_time_dataframe(days)
        >>> np.all(sun_df["sunset"] > sun_df["sunrise"])
        True
        >>> sun_df
                                 astronomical_twilight_start  ...        astronomical_twilight_end
        location date                                         ...
        canberra 2021-07-23 2021-07-23 05:36:52.178788+10:00  ... 2021-07-23 18:43:25.223475+10:00
                 2021-07-24 2021-07-24 05:36:20.903963+10:00  ... 2021-07-24 18:43:59.629041+10:00
                 2021-07-25 2021-07-25 05:35:48.170485+10:00  ... 2021-07-25 18:44:34.315154+10:00
        <BLANKLINE>
        [3 rows x 8 columns]
        """
        sun_times: dict[str, list[pd.Timestamp]] = {
            "astronomical_twilight_start": [],
            "nautical_twilight_start": [],
            "civil_twilight_start": [],
            "sunrise": [],
            "sunset": [],
            "nautical_twilight_end": [],
            "civil_twilight_end": [],
            "astronomical_twilight_end": [],
        }
        for day in days:
            sun_time_dict = self.search_sun_times(day)
            for key in sun_times.keys():
                sun_times[key].append(pd.Timestamp(sun_time_dict[key]))

        sun_times["date"] = [pd.Timestamp(day) for day in days]
        sun_times["location"] = [self.name for _ in range(len(days))]
        sun_df = pd.DataFrame(data=sun_times)
        sun_df.set_index(["location", "date"], inplace=True)
        return sun_df


class WeatherStation(BaseModel):
    """Contains information on a weather station.

    Parameters
    ----------
    location : Location
        Location of weather station.
    data_file : FilePath
        Path to csv file containing weather data from weather station.
    """

    location: Location = Field(..., description="Location of weather station.")
    data_file: FilePath = Field(
        ...,
        description=(
            "Path to csv file containing weather data from weather station. "
            "The firt 6 lines of the file are skipped, "
            "and the 7th should contain column headers. "
            "Should have one line per date. "
            "Minimally, the first column should be date, in YYYY-mm-dd format. "
        ),
    )

    class Config:
        schema_extra = {"description": "Contains information on a weather station."}

    def load_dataframe(self):
        """Loads weather data from self.data_file into a pd.DataFrame

        Returns
        -------
        weather_df : pd.DataFrame
            DataFrame with daily weather data, indexed by "weather_station" and "date".
        """
        weather_df = pd.read_csv(
            self.data_file, skiprows=5, header=0, parse_dates=["date"]
        )
        weather_df["weather_station"] = self.location.name
        weather_df.set_index(["weather_station", "date"], inplace=True)
        return weather_df


class LocationWeatherStationCollector(BaseModel):
    """Contains lists of Locations and Weather stations, and a mapping between them.

    Parameters
    ----------
    locations : list[Location]
        list of locations where cameras have been placed.
    weather_stations : list[WeatherStation]
        list of weather stations.
    location_weather_station_mapping : dict[str, str]
        A mapping between location names and weather_station names.
    """

    locations: list[Location] = Field(
        ..., description="list of locations where cameras have been placed."
    )
    weather_stations: list[WeatherStation] = Field(
        ..., description="list of weather stations."
    )
    location_weather_station_mapping: dict[str, str] = Field(
        ..., description="A mapping between location names and weather_station names."
    )

    class Config:
        schema_extra = {"description": "Defines Locations and Weather stations."}

    @validator("location_weather_station_mapping")
    def mapping_contains_all_locations(cls, v, values):
        if "locations" not in values:
            return v
        for location in values["locations"]:
            assert (
                location.name in v
            ), f"{location.name} missing from location_weather_station_mapping."
        return v

    @validator("location_weather_station_mapping")
    def all_weather_stations_included(cls, v, values):
        if "weather_stations" not in values:
            return v
        weather_station_location_names = set(
            ws.location.name for ws in values["weather_stations"]
        )
        for val in v.values():
            assert (
                val in weather_station_location_names
            ), f"Undefined weather station {val}. Either remove from mapping or define."
        return v

    def get_sun_time_dataframe(self, days: dict[str, Sequence[date]]) -> pd.DataFrame:
        """Calls .get_sun_time_dataframe on each location in self.locations, and builds
        a DataFrame of sun times.

        Parameters
        ----------
        days : dict[str, Sequence[date]]
            Mapping from location name to sequences of dates which will become index for
            the dataframe.

        Returns
        -------
        sun_df : pd.DataFrame
            DataFrame indexed by location and date, with columns
            "astronomical_twilight_start", "nautical_twilight_start",
            "civil_twilight_start", "sunrise", "sunset", "nautical_twilight_end",
            "civil_twilight_end", "astronomical_twilight_end".
        """
        sun_dfs = []
        for location in self.locations:
            dates = days[location.name]
            sun_dfs.append(location.get_sun_time_dataframe(dates))

        return pd.concat(sun_dfs)

    def get_weather_dataframe(self) -> pd.DataFrame:
        """Calls .load_dataframe() on each WeatherStation in self.weather_stations, and
        builds a DataFrame of weather data.

        Returns
        -------
        weather_df : pd.DataFrame
            DataFrame with daily weather data, indexed by "weather_station" and "date".
        """
        weather_dfs = []
        for weather_station in self.weather_stations:
            weather_dfs.append(weather_station.load_dataframe())

        return pd.concat(weather_dfs)

    def get_weather_sun_dataframe(
        self, days: Optional[dict[str, Sequence[date]]] = None
    ) -> pd.DataFrame:
        """Calls self.get_weather_dataframe and self.get_sun_time_dataframe, and merges
        the results into a single DataFrame.

        Parameters
        ----------
        days : Optional[dict[str, Sequence[date]]]
            Mapping from location name to sequences of dates which will become index for
            the dataframe. If None (default), this will be inferred from the weather
            dataframe.

        Returns
        -------
        weather_sun_df : pd.DataFrame
            Merged weather and sun time DataFrame.
        """
        # Get weather_df
        weather_df = self.get_weather_dataframe()

        # Infer days from self.locations and weather_df, if not already set.
        if days is None:
            days = {}
            for location in self.locations:
                weather_station: str = self.location_weather_station_mapping[
                    location.name
                ]
                days[location.name] = weather_df.loc[weather_station].index.date

        # Get sun_df
        sun_df = self.get_sun_time_dataframe(days)

        # Add "weather_station" column to sun_df to prepare for merge
        # Also add "lat", "lon", and "elevation_m" colmuns
        for location in self.locations:
            weather_station = self.location_weather_station_mapping[location.name]
            sun_df.loc[location.name, "weather_station"] = weather_station
            sun_df.loc[location.name, "lat"] = location.lat
            sun_df.loc[location.name, "lon"] = location.lon
            sun_df.loc[location.name, "elevation_m"] = location.elevation_m

        # Demote index to regular columns before merge to prevent "location" from
        # disappearing mysteriously.
        sun_df.reset_index(inplace=True)

        # Merge and reindex
        weather_sun_df = pd.merge(
            sun_df, weather_df, how="left", on=["weather_station", "date"], sort=True
        )
        weather_sun_df.set_index(["location", "date"], inplace=True)

        return weather_sun_df
