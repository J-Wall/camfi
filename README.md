# camfi
**C**amera-based **A**nalysis and **M**onitoring of **F**lying **I**nsects

# Installation

Installation is as easy as:

```
$ pip install camfi
```

Once you have installed camfi, you can run it from the command line:

```
$ camfi <command> <flags>
```

# Requirements

Camfi requires python 3.7 or greater.

See [requirements.txt](requirements.txt) for concrete dependencies.

Note: Installing using `$ pip install camfi` will only install the dependencies
for the command line tools. The example notebooks have some additional
dependencies, which can be installed by cloning the repository, and installing
from the requirements file:

```
$ git clone https://github.com/J-Wall/camfi.git
$ cd camfi
$ pip install -r requirements.txt
```

# A. Setting up the cameras

## Rolling shutter calibration measurements

There are probably a number of possible ways to measure the rolling shutter line
rate of a camera, so feel free to use your imagination. Here is one way of doing
it:

1. Find a place you can make very dark (or wait until night time to do the
   measurement).

2. Make a device which spins a white line about its centre at a constant
   rotational velocity. For the line, We used a cardboard tube from a roll of
   paper towels with a line of paper taped to it. The tube was taped to the
   blades of a small desk fan. The inertia of the cardboard tube helped ensure
   that the line would rotate at a constant rotational velocity.

   ![Apparatus for measuring rolling shutter line
   rate](https://raw.githubusercontent.com/J-Wall/camfi/main/manual_figures/calibration_setup.jpg)*Figure
   A1. Apparatus for measuring rolling shutter line rate.*

3. Measure the rotational velocity of the line by synchronising a strobe light
   to the rotations of the line. We found the Strobily Android app (available on
   the [Google Play
   Store](https://play.google.com/store/apps/details?id=com.tp77.StrobeAd)) to
   be very useful for this. This will be easiest in a dark room.

4. Mount the camera you wish to measure facing the rotating line, ensuring the
   camera is steady.

5. Take (multiple) photos of the rotating line, under illumination by the
   camera's infra-red LED flash. If using wildlife cameras, it is recommended to
   do this buy using the camera's timed capture setting, so that it is not
   bumped while taking the photos.

6. Load the images onto your computer and follow the steps in the [camera
   calibration notebook](examples/camera_calibration.ipynb).

## Camera settings

In general, the specific settings you use depend on the research question, but
our suggestion is to use a time-lapse function, rather than (or in addition to,
if available) passive infra-red (PIR) motion detection to trigger the camera.
This is because insects will not be detected by the PIR sensor.

Other settings are up to the user. We use the highest available quality setting
and have set the cameras to only take photos during the night.

## Camera placement

The cameras should ideally be placed such that the background of the images is
more or less uniform (for example, at the sky), but again, this depends on the
research question.

# B. Image annotation

## Setting up the VIA project file

1. Place all the images to be annotated into a single parent directory. They may
   be in sub-directories within this parent directory.

2. Before making the project file, it is necessary to produce a text file which
   contains the relative paths to all of the images to be annotated, one per
   line. It is suggested to generate this on the command line. For example, if
   your file structure is something like this:

   ```
   annotation_project            # Parent directory
   ├-- via.html
   ├-- moth_images_november      # Images may be contained within subdirs,
   │   ├-- 0001                  # and these subdirs can be arbitrarily
   │   │   ├-- 100MEDIA          # nested.
   │   │   │   ├-- DSCF0001.JPG
   │   │   │   └--...
   │   │   └-- 101MEDIA
   │   │       ├-- DSCF0001.JPG
   │   │       └--...
   │   └-- 0002
   │       ├-- 100MEDIA
   │       │   ├-- DSCF0001.JPG
   │       │   └--...
   │       └-- 101MEDIA
   │           ├-- DSCF0001.JPG
   │           └--...
   └-- moth_images_december
       ├-- 0001
       │   ├-- 100MEDIA
       │   │   ├-- DSCF0001.JPG
       │   │   └--...
       │   └-- 101MEDIA
       │       ├-- DSCF0001.JPG
       │       └--...
       └-- 0002
           ├-- 100MEDIA
           │   ├-- DSCF0001.JPG
           │   └--...
           └-- 101MEDIA
               ├-- DSCF0001.JPG
               └--...
   ```

   Then you could generate the required file by running the following bash
   command from within the `annotation_project` directory:

   ```
   $ ls -1 moth_images_*/*/*/*.JPG > annotation_image_filelist.txt
   ```

   Or if using fish-shell, one can easily be more flexible with regard to the
   exact directory structure:

   ```
   ls -1 moth_images_**.JPG > annotation_image_filelist.txt
   ```

   You will then end with a file called `annotation_image_filelist.txt` in the
   `annotation_project` directory, which starts something like this (note the
   exact paths should reflect your directory structure, rather than that of this
   arbitrary example):

   ```
   annotation_project$ head annotation_image_filelist.txt
   moth_images_november/0001/100MEDIA/DSCF0001.JPG
   moth_images_november/0001/100MEDIA/DSCF0002.JPG
   moth_images_november/0001/100MEDIA/DSCF0003.JPG
   moth_images_november/0001/100MEDIA/DSCF0004.JPG
   moth_images_november/0001/100MEDIA/DSCF0005.JPG
   moth_images_november/0001/100MEDIA/DSCF0006.JPG
   moth_images_november/0001/100MEDIA/DSCF0007.JPG
   moth_images_november/0001/100MEDIA/DSCF0008.JPG
   moth_images_november/0001/100MEDIA/DSCF0009.JPG
   moth_images_november/0001/100MEDIA/DSCF0010.JPG
   ```

   Note that this file contains relative paths to the images, not absolute
   paths. This is important for making the project portable between machines
   (for example if the annotation work is to be spread between multiple people).
   Once this file list has been generated, we are ready to set up the project in
   VIA.

3. Ensure "via.html" is placed into the parent directory containing the images
   to be annotated (`annotation_project` in the above example), then open it in
   you web browser by double-clicking it.

4. In VIA, under "Project" in the top menu bar (see Fig. B1), select "Add url or
   path from text file".

5. Navigate to the project directory, and select the file list we generated in
   step 2. The images should appear in the project pane on the left-hand side of
   the screen.

6. After checking that the image files were loaded properly, select "Save" under
   the "Project" menu. Leave the boxes checked, and name the project file as you
   wish. Your web browser will treat this as a file download. Once downloaded,
   you should move the project file into your annotation project directory.

## Performing the annotations in VIA

1. Open “via.html”, ensuring it is in the parent directory containing the images
   to be annotated.

2. In the top menu, click on Project, then choose Load (see the red oval to the
   left in below Fig. B1). Find your VIA project file, and click Open.

   ![Loading and navigating
   VIA](https://raw.githubusercontent.com/J-Wall/camfi/main/manual_figures/navigating_via.png)*Figure
   B1.  Loading and navigating VIA.*

3. If it is the first time that you work on the file, simply start with the
   first image. If you have already worked on the project file before and you
   have a saved version, scroll down to the last image that you were working on
   and click on it. You can now start working from that image.

4. You move between images (backwards and forwards) with the sideways arrows in
   the top menu (see the blue oval to the right in Fig. B1), or you can use the
   sideways arrows on your keyboard.

5. To zoom in and out, use the magnifying glass (+ or -, see the yellow oval in
   the upper right corner in Fig. B1).

6. To the left, you can find different Region shapes (see the red oval in
   Fig. B2). The only ones I have been using are the “Circular region shape”, the
   “Point region shape”, and the “Polyline region shape”.

   ![Region shapes in
   VIA](https://raw.githubusercontent.com/J-Wall/camfi/main/manual_figures/region_shapes.png)*Figure
   B2. Region shapes in VIA.*

   - Circular region shape: This shape can be used when you cannot see the
     whole moth (or the whole motion blur), e.g., when the moth is going out the
     edge of the image (see the moth in the upper right corner in Fig. B3), if
     another moth or object is covering it, or if you find it hard to see where
     the motion blur starts and ends. To draw a circle region, simply press single
     click and drag the mouse.

     ![Example of circle
     annotation](https://raw.githubusercontent.com/J-Wall/camfi/main/manual_figures/circle_annotation.png)*Figure
     B3. Example of circle annotation.*

   - Point region shape: This shape can be used when the moth is visible as a
     point (usually in brighter conditions; see the two moths in Fig. B4). There
     is not as much motion blur, because the sun has not set yet, meaning the
     camera used a shorter exposure time. It can also be used when the area of
     the moth is too small for the circular region shape to function. When this
     is the case, an error message will show up at the bottom of the screen. To
     define a point, press single click.

     ![Example of point
     annotations](https://raw.githubusercontent.com/J-Wall/camfi/main/manual_figures/point_annotation.png)*Figure
     B4. Example of point annotations.*

   - Polyline region shape: This shape should be used when the moth is
     visible as a line (due to motion blur). Often, you can see the flapping of the
     wings (see Fig. B3). To draw a polyline, single click on the start of the motion
     blur, and then at the end of the motion blur. To finish drawing the polyline,
     press “Enter” on the keyboard. It is important to make sure that the ends of
     the polyline annotations match up with the ends of the motion blur. Also
     important is to follow the line carefully - by clicking along the line several
     times - so that a bend is properly annotated (see the polyline in Fig. B5).

     ![Example of polyline
     annotation](https://raw.githubusercontent.com/J-Wall/camfi/main/manual_figures/polyline_annotation.png)*Figure
     B5. Example of polyline annotation.*

7. In the bottom left corner, you can find different Keyboard Shortcuts (see
   Fig. B6). There is an explanation to the right of each shortcut. Some of them can
   be quite helpful, e.g. how to “Delete selected regions”. You basically just
   click on the region shape and it becomes selected. Then you can delete it by
   clicking on the letter “d”. Some shapes can be harder than others to delete,
   e.g. polylines, simply because the lines are so thin. Just be patient, it will
   work eventually.

   ![Button to view keyboard shortcuts in
   VIA](https://raw.githubusercontent.com/J-Wall/camfi/main/manual_figures/keyboard_shortcuts.png)*Figure
   B6. Button to view keyboard shortcuts in VIA.*

8. Do not forget to save. Do this regularly, about every 30 min. You can find
   "Save" in the top menu under "Project" (see Fig. B1). It is recommended to
   save each time to a new file, with a file name based on which image in the
   project you are up to. This will mitigate the risk of file corruption
   problems and will aid in keeping track of progress.

### Examples of problematic images and FAQ

![Extremely busy
image](https://raw.githubusercontent.com/J-Wall/camfi/main/manual_figures/problematic_busy.png)*Figure
B7. Extremely busy image.*

**Q:** What to do when an image is extremely busy and it is difficult to tell moths
apart?

**A:** Make a note of it and do not spend too long trying to get it exactly right.
Make some judgement calls.

![Endpoints are hard to see in faint blurs of moths in the
background](https://raw.githubusercontent.com/J-Wall/camfi/main/manual_figures/problematic_background_moths_endpoints.png)*Figure
B8.  Endpoints are hard to see in faint blurs of moths in the background.*

**Q:** In this image, there are faint moths in the background, but it is hard to see
exactly where the image blur starts and ends. What to do?

**A:** The main thing is to be as consistent across images as possible. If you are
not confident about the start and end point you could use a circle annotation
(then it will be included in the abundance analysis, but not the wingbeat
analysis).

![Faint blurs of moths in the background have no obviously visible
wingbeat](https://raw.githubusercontent.com/J-Wall/camfi/main/manual_figures/problematic_background_moths_wingbeats.png)*Figure
B9.  Faint blurs of moths in the background have no obviously visible wingbeat.*

**Q:** In this image, the flapping of the wings is not very obvious for some of the
moths in the background. Should I track them anyway?

**A:** It is a bit of a judgement call. Ideally, we mark all moths and only moths.
Obviously, this can be a bit tricky. Many of the smooth streaks might be other
insects, spider web, or even rain. If you are not sure, mark it, and make a note
of which image it is in. If it looks really different from other moths in the
image, do not mark it. These instructions are a bit vague, but try to be
consistent. Some of the moths just has lower contrast against the sky, which
makes the flapping harder to see. If you zoom in, you might see it better. You
could also compare the speed (based on the length of the streak) of the moth to
other moths in the image.

## Loading the image metadata into the VIA project file

**Note:** It is recommended to only perform this steps after the annotation has
been completed. This is because including the image metadata in the VIA project
file increases the size of the file substantially, and since it is recommended
to save all incremental versions of the project file, this could become
cumbersome if the metadata is included from the start.

For insect activity analysis ([see example jupyter
notebook](examples/activity_analysis.ipynb)), we first need to load the image
metadata into the annotation project file. This can be done using `camfi
add-metadata`. Usage is simple, just provide the input file and desired output
file:

```
$ camfi add-metadata \
    --i via_annotation_project_file.json \
    --o via_annotation_project_file_with_metadata.json
```

There is one optional argument to `camfi add_metadata`, called
`--processes` which enables multiple processes to be spawned. This may improve
performance to a certain point, but eventually I/O will be the limiting factor
to how fast this can run (as each image file needs to be opened in order to
extract the metadata). For example, to run with 8 cores:

```
$ camfi add-metadata \
    --i via_annotation_project_file.json \
    --o via_annotation_project_file_with_metadata.json \
    --processes 8
```

# C. Automatic image annotation

Camfi can perform the annotation process for you automatically by using an
annotation model based of Mask R-CNN. We have trained the model on images of
Bogong moth motion blurs, and it gives good results. It is very simple to train
and training can be done on Google Colab, meaning that training can be done even
if you do not have access to GPU compute.

## Inference (performing automatic annotation)

If you have trained you model (see next section for how to do this), or if you
want to use the included model, you can use `camfiannotate` to automatically
annotate your images. Sample usage:

```
$ camfiannotate via_annotation_project_file.json \
    --o autoannotated.json \
    --model release \
    --crop=0,0,4608,3312 \  # To crop out trailcamera info overlay
    --device cuda \
    --backup-device cpu \
    --score-thresh 0.4
```

This will annotate all the images in the input VIA project file using the
release model, outputting a new VIA project file. The above options tell
`camfiannotate` to crop the images before running inference on them, and to use
cuda (a GPU) to run the inference.  By setting `--backup-device cpu`, we tell
the annotator to run inference on the CPU for images with fail on the GPU due to
memory constraints (inference on images with lots of motion blurs in them takes
up more memory). Finally, only annotations which have a score of at least 0.4
will be output. For model validation, it is recommended to set this to 0.
Besides, you can always filter these later, for example:

```
$ camfi filter \
    --i autoannotated.json \
    --o autoannotated_0.9.json \
    --by score \
    --minimum 0.9
```

If you want to use a different model, you can set `--model <filepath>`, where
`<filepath>` is the path to the model you want to use. Alternatively you can set
`--model latest`, which will check github for the latest released model.

## Training

To train the model, you will first need a set of manually annotated images (see
above for instructions to do this). Once you have this, you can select just the
images which have at least one annotation for training and validation purposes:

```
$ camfi remove-unannotated \
    --i via_annotation_project_file.json \
    --o annotated_only.json
```

If you are training on a server or Colab instance, you might want to package
these images up along with the annotations in a zip file for easy uploading of
only the data we actually need:

```
$ camfi zip-images \
    --i annotated_only.json \
    --o annotated_only.zip
```

To define a test set, we make a text file listing a random subset of the images.
This will make a test set of 50 images:

```
$ camfi filelist --i annotated_only.json --shuffle 1234 | head -n 50 \
    > test_set.txt
```

We are now ready for training. This can either be done from the command line
using the `traincamfiannotator` command. For help on how to use this, run:

```
$ traincamfiannotator --help
```

Alternatively you can train on Colab. Please refer to the [model training
example notebook](examples/camfi_autoannotator_training.ipynb).

## Validation

To validate our automatic annotation model, we need a VIA project file
containing manual annotations (e.g.
`via_annotation_project_file_with_metadata.json`), and a second VIA project file
containing only the images we want to run the validation on. For example, we may
want to run validation on on our test set (e.g. those in `test_set.txt`). To
make a test set VIA project file, we proceed in the same way as we did to create
our original VIA project file (i.e. run `via.html`, and select "Project" > "Add
url or path from text file". Then select `test_set.txt`, and save the project as
`test_set.json`).

To validate the automatic annotations, we need to first get them. Once we have
the VIA project file for the images we want to run validation on, we run
`camfiannotate` on it:

```
$ camfiannotate test_set.json \
    --o test_set_autoannotated.json \
    --crop=0,0,4608,3312 \  # To crop out trailcamera info overlay
    --device cuda \
    --backup-device cpu \
    --score-thresh 0.0  # Important to set this to 0.0 for validation
```

This gives us the automatic annotations. To validate these against
`via_annotation_project_file_with_metadata.json`, we run:

```
$ camfi validate-annotations \
    via_annotation_project_file_with_metadata.json \
    --i test_set.json \
    --o validation.json
```

This gives us `validation.json`, which contains the validation data. For an
example of how to interpret this data see the [automatic annotation evaluation
notebook](examples/annotation_evaluation.ipynb).

# D. Data analysis

## Running camfi

Once the manual annotation is completed, running `camfi extract-wingbeats` will
analyse wingbeats from polyline-annotated moth motion blurs to gain information
about the wingbeat frequency of the moths which produced those blurs.

Before proceeding, ensure you have:

1. Measured the rolling shutter line rate of you camera(s)
2. A completed annotation project file
3. The image files used to produce the annotation file

A general summary of usage is provided below:

```
NAME
    camfi extract-wingbeats - Uses the camfi algorithm to measure the wingbeat frequency of annotated flying insect motion blurs in still images.

SYNOPSIS
    camfi extract-wingbeats <flags>

DESCRIPTION
    Uses the camfi algorithm to measure the wingbeat frequency of annotated flying insect motion blurs in still images.

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
        Maximum number of columns to calculate autocorrelation over. Defaults to a half of the length of the image
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
```

Running the above will produce a tab-separated file called `moth_wingbeats.csv`
with the following columns:

1.  `image_name` : relative path to image
2.  `capture_time` : datetime in yyyy-mm-dd HH:MM:SS format
3.  `annotation_idx` : index of annotation in image (arbitrary)
4.  `best_peak` : period of wingbeat in pixels
5.  `blur_length` : length of motion blur in pixels
6.  `snr` : signal to noise ratio of best peak
7.  `wb_freq_up` : wingbeat frequency estimate, assuming upward motion (and zero
    body-length)
8.  `wb_freq_down` : wingbeat frequency estimate, assuming downward motion (and
    zero body-length)
9.  `et_up` : corrected moth exposure time, assuming upward motion
10. `et_dn` : corrected moth exposure time, assuming downward motion
11. `period_up` : wingbeat period, assuming upward motion (and zero body-length)
12. `period_dn` : wingbeat period, assuming downward motion (and zero
    body-length)
13. `spec_dens` : comma separated values, with the spectral density array
    associated with the annotation

## Wingbeat analysis

Once `camfi.py` has been run, the output can be used for further analysis of
wingbeat frequency. For an example of such analysis, please refer to the
[wingbeat analysis example notebook](examples/wingbeat_analysis.ipynb).

## Insect activity analysis

The annotation file with image metadata produced in section B of this manual can
be used directly for analysis of insect activity levels. Please refer to the
[activity analysis example notebook](examples/activity_analysis.ipynb) for
guidance on how this analysis could be conducted.
