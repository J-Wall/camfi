Data analysis
=============

Running camfi
-------------

Once the manual (or automatic) annotation is completed, running
``camfi extract-wingbeats`` will analyse wingbeats from polyline-annotated moth
motion blurs to gain information about the wingbeat frequency of the moths
which produced those blurs.

Before proceeding, ensure you have:

1. Measured the rolling shutter line rate of you camera(s)
2. A completed annotation project file
3. The image files used to produce the annotation file

A general summary of usage is provided below::

   NAME
       camfi extract-wingbeats - Uses the camfi algorithm to measure the
       wingbeat frequency of annotated flying insect motion blurs in still
       images.

   SYNOPSIS
       camfi extract-wingbeats <flags>

   DESCRIPTION
       Uses the camfi algorithm to measure the wingbeat frequency of annotated
       flying insect motion blurs in still images.

   FLAGS
       --processes=PROCESSES
           Default: 1
           number of child processes to spawn
       --i=I
           Type: Optional[]
           Default: None
           path to input VIA project json file. Defaults to sys.stdin
       --o=O
           Type: Optional[]
           Default: None
           path to output file. Defaults to sys.stdout

       --line_rate=LINE_RATE
           Default: inf
           The line rate of the rolling shutter
       --scan_distance=SCAN_DISTANCE
           Default: 100
           Half width of analysis windows (half width of blurs)
       --max_dist=MAX_DIST
           Type: Optional[]
           Default: None
           Maximum number of columns to calculate autocorrelation over.
           Defaults to a half of the length of the image
       --supplementary_figures=SUPPLEMENTARY_FIGURES
           Type: Optional[]
           Default: None
           Directory in which to put supplementary figures (optional)

   EXAMPLE USAGE
       $ camfi extract-wingbeats \
           --i via_annotation_project_file_with_metadata.json \
           --line-rate 91813 \
   	   --scan-distance 100 \
   	   --supplementary-figures wingbeat_supplemantry_figures \
   	   --processes 8 \
   	   --o moth_wingbeats.csv

Running the above will produce a tab-separated file called ``moth_wingbeats.csv``
with the following columns:

1.  ``image_name``: relative path to image
2.  ``capture_time``: datetime in yyyy-mm-dd HH:MM:SS format
3.  ``annotation_idx``: index of annotation in image (arbitrary)
4.  ``best_peak``: period of wingbeat in pixels
5.  ``blur_length``: length of motion blur in pixels
6.  ``snr``: signal to noise ratio of best peak
7.  ``wb_freq_up``: wingbeat frequency estimate, assuming upward motion (and zero
    body-length)
8.  ``wb_freq_down``: wingbeat frequency estimate, assuming downward motion (and
    zero body-length)
9.  ``et_up``: corrected moth exposure time, assuming upward motion
10. ``et_dn``: corrected moth exposure time, assuming downward motion
11. ``period_up``: wingbeat period, assuming upward motion (and zero body-length)
12. ``period_dn``: wingbeat period, assuming downward motion (and zero
    body-length)
13. ``spec_dens``: comma separated values, with the spectral density array
    associated with the annotation


Wingbeat analysis
-----------------

Once ``camfi.py`` has been run, the output can be used for further analysis of
wingbeat frequency. For an example of such analysis, please refer to the
example `wingbeat analysis notebook`_.

.. _`wingbeat analysis notebook`: https://github.com/J-Wall/camfi/blob/main/examples/wingbeat_analysis.ipynb

Insect activity analysis
------------------------

The annotation file with image metadata produced in section B of this manual
can be used directly for analysis of insect activity levels. Please refer to
the example `activity analysis notebook`_ for guidance on how this analysis
could be conducted.

.. _`activity analysis notebook`: https://github.com/J-Wall/camfi/blob/main/examples/activity_analysis.ipynb

