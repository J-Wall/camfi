Setting up the cameras
======================

Rolling shutter calibration measurements
----------------------------------------

There are probably a number of possible ways to measure the rolling shutter
line rate of a camera, so feel free to use your imagination. Here is one way of
doing it:

1. Find a place you can make very dark (or wait until night time to do the
   measurement).

2. Make a device which spins a white line about its centre at a constant
   rotational velocity. For the line, We used a cardboard tube from a roll of
   paper towels with a line of paper taped to it. The tube was taped to the
   blades of a small desk fan. The inertia of the cardboard tube helped ensure
   that the line would rotate at a constant rotational velocity.

   .. _fig-calibration_setup:

   .. figure:: figures/calibration_setup.jpg
      :alt: Apparatus for measuring rolling shutter line rate
      :width: 100 %
      :align: center

      Apparatus for measuring rolling shutter line rate.

      ..

3. Measure the rotational velocity of the line by synchronising a strobe light
   to the rotations of the line. We found the Strobily_ Android app to be
   very useful for this. This will be easiest in a dark room.

4. Mount the camera you wish to measure facing the rotating line, ensuring the
   camera is steady.

5. Take (multiple) photos of the rotating line, under illumination by the
   camera's infra-red LED flash. If using wildlife cameras, it is recommended
   to do this buy using the camera's timed capture setting, so that it is not
   bumped while taking the photos.

6. Load the images onto your computer and follow the steps in the
   :doc:`notebooks/camera_calibration` notebook.

.. _Strobily: https://play.google.com/store/apps/details?id=com.tp77.StrobeAd


Camera settings
---------------

In general, the specific settings you use depend on the research question, but
our suggestion is to use a time-lapse function, rather than (or in addition to,
if available) passive infra-red (PIR) motion detection to trigger the camera.
This is because insects will not be detected by the PIR sensor.

Other settings are up to the user. We use the highest available quality setting
and have set the cameras to only take photos during the night.


Camera placement
----------------

The cameras should ideally be placed such that the background of the images is
more or less uniform (for example, at the sky), but again, this depends on the
research question.
