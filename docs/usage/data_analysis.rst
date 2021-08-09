Data analysis
=============

Including weather data
----------------------

If you are including weather data in your analyses,
then you need to specify ``place`` in your configuration file.
You will also need daily weather data
from each weather station
which is included in your study.
This should be a comma-delimited file
with at minimum the first column being "date"
with dates formatted as YYYY-mm-dd.
Currently, Camfi only accepts weather data files which
have a header of exactly 7 lines,
which is typical of the daily observation files
from the `Bureau of Meteorology`_.
This requirement may be dropped
(or become configurable)
in future versions of Camfi.

.. _Bureau of Meteorology: http://www.bom.gov.au/

Running camfi
-------------

Once the manual (or automatic) annotation is completed,
running
``camfi extract-wingbeats`` will analyse wingbeats from polyline-annotated moth
motion blurs to gain information about the wingbeat frequency of the moths
which produced those blurs.

Before proceeding, ensure you have:

1. Measured the rolling shutter line rate of you camera(s)
   (:doc:`notebooks/camera_calibration`)
2. A completed annotation project file
3. The image files used to produce the annotation file

To perform wingbeat extraction,
we need a configuration file
with
``wingbeat-extraction``
configured.
It is also adviced to have
``camera``,
``time``,
and
``place``
configured,
so that all relavant metadata is included
(see :ref:`example-configuration`).
Below we will assume that config.yml
is this example configuration file.

Once we have our configuration file,
running::

    $ camfi --config config.yml \
         --output project_with_wingbeats.json \
         load-exif extract-wingbeats write

Will insert wingbeat measurements
into the VIA project,
writing a new VIA project file,
"project_with_wingbeats.json",
which can then be used for futher analysis.

The wingbeat data can also be exported
to various types of tab-separated files
using the
``image-table``,
``region-table``,
and
``table``
camfi commands
(see :doc:`cli`).


Wingbeat analysis
-----------------

Once ``camfi extract-wingbeats`` has been run,
the output can be used for further analysis of
wingbeat frequency. For an example of such analysis, please refer to the
example :doc:`notebooks/wingbeat_analysis` notebook.

Insect activity analysis
------------------------

The annotation file with image metadata produced
by running ``camfi load-exif``
can be used directly for analysis of insect activity levels.
Please refer to the example
:doc:`notebooks/activity_analysis`
notebook for guidance on how
this analysis could be conducted.
