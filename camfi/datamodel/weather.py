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

.. _choosing an ephemeris: https://rhodesmill.org/skyfield/planets.html#choosing-an-ephemeris
"""

from datetime import date, datetime, time, timedelta, timezone
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from skyfield.api import Loader, wgs84
from skyfield import almanac


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


class Location:
    def __init__(
        self,
        name: str,
        lat: float,
        lon: float,
        elevation_m: float,
        tz: timezone,
    ):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.elevation_m = elevation_m
        self.tz = tz

        self._dark_twilight_day = almanac.dark_twilight_day(
            ephemeris, wgs84.latlon(self.lat, self.lon, elevation_m=self.elevation_m)
        )

    def _get_tz_aware_dt(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
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
        ...     tz=timezone(timedelta(hours=10)),
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

    def search_sun_times(self, day: date) -> Dict[str, datetime]:
        """Gets sunrise, sunset, and twilight times for a given date.

        Parameters
        ----------
        day : date
            Day to get times from.

        Returns
        -------
        twilight_times : Dict[str, datetime]
            Dictionary with keys "astronomical_twilight_start",
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
        start_time = datetime.combine(date=day, time=time(0), tzinfo=self.tz)
        end_time = start_time + timedelta(days=1)
        t0 = timescale.from_datetime(start_time)
        t1 = timescale.from_datetime(end_time)

        times, twilight_types = almanac.find_discrete(t0, t1, self._dark_twilight_day)
        twilight_transitions = (
            np.roll(twilight_types, 1) > twilight_types
        ) * 5 + twilight_types

        twilight_times: Dict[str, datetime] = {}
        for t, tt in zip(times, twilight_transitions):
            twilight_times[TWILIGHT_TRANSITIONS[tt]] = t.utc_datetime().astimezone(
                self.tz
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
            DataFrame indexed by days, and with columns "astronomical_twilight_start",
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
        >>> days = [date(2021, 7, 23), date(2021, 7, 24), date(2021, 7, 25)]
        >>> sun_df = location.get_sun_time_dataframe(days)
        >>> np.all(sun_df["sunset"] > sun_df["sunrise"])
        True
        >>> sun_df
                        astronomical_twilight_start  ...        astronomical_twilight_end
        date                                         ...
        2021-07-23 2021-07-23 05:36:52.178788+10:00  ... 2021-07-23 18:43:25.223475+10:00
        2021-07-24 2021-07-24 05:36:20.903963+10:00  ... 2021-07-24 18:43:59.629041+10:00
        2021-07-25 2021-07-25 05:35:48.170485+10:00  ... 2021-07-25 18:44:34.315154+10:00
        <BLANKLINE>
        [3 rows x 8 columns]
        """
        sun_times: Dict[str, List[pd.Timestamp]] = {
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
        sun_df = pd.DataFrame(data=sun_times)
        sun_df.set_index("date", inplace=True)
        return sun_df
