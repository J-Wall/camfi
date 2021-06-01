Command-line interface
======================

Installing Camfi exposes number of tools to your command line. The help pages
for each of these can be viewed by running::

   $ <command> -- --help

or::

   $ <command> <subcommand> -- --help

In some cases ``$ <command> <subcommand> --help`` may be sufficient, but if
this causes an error try the ``-- --help`` syntax.

There are three commands included with Camfi:

1. ``camfi``: provides various utilities for woirking with annotation files
   through subcommands (shown below).

2. ``traincamfiannotator``: the command for training the camfi automatic
   annotator

3. ``camfiannotate``: the command for running inference (automatic annotation)


The ``camfi`` command
---------------------

Running ``$ camfi -- --help`` will show the global options for all ``camfi``
subcommands::

   NAME
       camfi - Provides utilities for working with camfi projects. Available
       subcommands are: add_metadata, download_model, extract_wingbeats,
       filelist, filter, merge_annotations, remove_unannotated,
       validate_annotations, zip_images

   SYNOPSIS
       camfi <flags>

   DESCRIPTION
       Provides utilities for working with camfi projects. Available
       subcommands are: add_metadata, download_model, extract_wingbeats,
       filelist, filter, merge_annotations, remove_unannotated,
       validate_annotations, zip_images

   FLAGS
       --processes=PROCESSES
           Type: int
           Default: 1
           number of child processes to spawn
       --i=I
           Type: Optional[typing.U...
           Default: None
           path to input VIA project json file. Defaults to sys.stdin
       --o=O
           Type: Optional[typing.U...
           Default: None
           path to output file. Defaults to sys.stdout

The help page ofr each subcommand is provided below

``camfi add_metadata``
^^^^^^^^^^^^^^^^^^^^^^

``$ camfi add_metadata -- --help``::

   NAME
       camfi add-metadata - Adds image (EXIF) metadata to VIA project by
       reading image files. Optionally spawns multiple processes (reading the
       images is usually I/O bound and can take some time).

   SYNOPSIS
       camfi add-metadata [EXIF_TAGS]...

   DESCRIPTION
       Adds image (EXIF) metadata to VIA project by reading image files.
       Optionally spawns multiple processes (reading the images is usually I/O
       bound and can take some time).

   POSITIONAL ARGUMENTS
       EXIF_TAGS
           Type: str


``camfi download_model``
^^^^^^^^^^^^^^^^^^^^^^^^

``$ camfi download_model -- --help``::

   NAME
       camfi download-model - Downloads a pretrained image annotation model,
       returning the path to the model.

   SYNOPSIS
       camfi download-model <flags>

   DESCRIPTION
       Downloads a pretrained image annotation model, returning the path to the model.

   FLAGS
       --model=MODEL
           Type: typing.Union[str, os.PathLike]
           Default: 'release'
           Name of model. Can be one of {"release", "latest"} or a url
           pointing to the model file on the internet. Alternatively, a path
           to an existing local file can be given, in which case the path is
           returned and nothing else is done.


``camfi extract_wingbeats``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``$ camfi extract_wingbeats -- --help``::

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
       --line_rate=LINE_RATE
           Type: float
           Default: inf
           The line rate of the rolling shutter
       --scan_distance=SCAN_DISTANCE
           Type: int
           Default: 100
           Half width of analysis windows (half width of blurs)
       --max_dist=MAX_DIST
           Type: Optional[typing.Unio...
           Default: None
           Maximum number of columns to calculate autocorrelation over.
           Defaults to a half of the length of the image
       --supplementary_figures=SUPPLEMENTARY_FIGURES
           Type: Optional[typing.Unio...
           Default: None
           Directory in which to put supplementary figures (optional)


``camfi filelist``
^^^^^^^^^^^^^^^^^^

``$ camfi filelist -- --help``::

   NAME
       camfi filelist - Lists the images in the input VIA project

   SYNOPSIS
       camfi filelist <flags>

   DESCRIPTION
       Lists the images in the input VIA project

   FLAGS
       --sort=SORT
           Type: bool
           Default: True
           If True, output is sorted lexigraphically. If False, order is arbitrary
       --shuffle=SHUFFLE
           Type: typing.Union[bool, int]
           Default: False
           If int, then the output is shuffled using `shuffle` as the seed. If
           True, then the output is shuffled using the system time as the seed.
           If False (default), do not shuffle. Shuffling occurs after sorting.
           For reproducability, set `sort=True`.


``camfi filter``
^^^^^^^^^^^^^^^^

``$ camfi filter -- --help``::

   NAME
       camfi filter - Filters VIA annotations by enforcing a minimum and/or
       maximum value for a numerical region attribute (eg. "score" which is
       defined during automatic automatic annotation)

   SYNOPSIS
       camfi filter BY <flags>

   DESCRIPTION
       Filters VIA annotations by enforcing a minimum and/or maximum value for
       a numerical region attribute (eg. "score" which is defined during
       automatic automatic annotation)

   POSITIONAL ARGUMENTS
       BY
           Type: str
           The region_attributes key to filter annotations by.

   FLAGS
       --minimum=MINIMUM
           Type: float
           Default: -inf
           The minimum value of the region attribute to pass the filter
       --maximum=MAXIMUM
           Type: float
           Default: inf
           The maximum value of the region attribute to pass the filter
       --mode=MODE
           Type: str
           Default: 'warn'
           One of {"pass", "fail", "raise", "warn"}. Defines how annotations
           missing the `by` region attribute are handled. "pass": These
           annotations pass the filter. "fail": These annotations are removed.
           "raise": A KeyError is raised if an annotation is missing the
           attribute. "warn": Like "pass" but a warning is printed to
           sys.stderr.

   NOTES
       You can also use flags syntax for POSITIONAL ARGUMENTS


``camfi merge_annotations``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``$ camfi merge_annotations -- --help``::

   NAME
       camfi merge-annotations - Takes a list of VIA project files and merges
       them into one. Ignores --i in favour of annotation_files.

   SYNOPSIS
       camfi merge-annotations [ANNOTATION_FILES]...

   DESCRIPTION
       Takes a list of VIA project files and merges them into one. Ignores --i
       in favour of annotation_files.

   POSITIONAL ARGUMENTS
       ANNOTATION_FILES
           Type: str
           list of VIA project json files to merge. Project and VIA settings
           are taken from the first file.


``camfi remove_unannotated``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``$ camfi remove_unannotated -- --help``::

   NAME
       camfi remove-unannotated - Removes image metadata from VIA project file
       for images which have no annotations.

   SYNOPSIS
       camfi remove-unannotated -

   DESCRIPTION
       Removes image metadata from VIA project file for images which have no
       annotations.


``camfi validate_annotations``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``$ camfi validate_annotations -- --help``::

   NAME
       camfi validate-annotations - Compares annotation file against a
       ground-truth annotation file for automatic annotation validation
       puposes.

   SYNOPSIS
       camfi validate-annotations GROUND_TRUTH <flags>

   DESCRIPTION
       Validation data is output to a json dict, which includes:

       all_ious: list of [iou, score] pairs
           iou is the Intersection over Union of the bounding boxes of true
           positives to their matched ground truth annotation. All matched
           annotations are included.
           score is the prediction score of the automatic annotation
       polyline_hausdorff_distances: list of [h_dist, score] pairs
           h_dist is the hausdorff distance of a true positive polyline
           annotation, where the annotation is matched to a polyline ground
           truth annotation. Only polyline annotations which matched to a
           polyline ground truth annotation are included.
           score is the prediction score of the automatic annotation
       length_differences: list of [l_diff, score] pairs
           l_diff is calculated as the length of a true positive polyline
           annotation minus the length of it's matched ground truth annotation.
           Only polyline annotations which matched to a polyline ground truth
           annotation are included.
           score is the prediction score of the automatic annotation
       true_positives: list of scores
           score is the prediction score of the automatic annotation
       false_positives: list of scores
           score is the prediction score of the automatic annotation
       false_negatives: int
           number of false negative annotations

   POSITIONAL ARGUMENTS
       GROUND_TRUTH
           Type: str
           Path to ground truth VIA annotations file. Should contain
           annotations for all images in input annotation file.

   FLAGS
       --iou_thresh=IOU_THRESH
           Type: float
           Default: 0.5
           Threshold of intersection-over-union of bounding boxes to be considered a match.

   NOTES
       You can also use flags syntax for POSITIONAL ARGUMENTS


``camfi zip_images``
^^^^^^^^^^^^^^^^^^^^

``$ camfi zip_images -- --help``::

   NAME
       camfi zip-images - Makes a zip archive of all the images in the provided
       VIA project file. If --i is set, then the annotation file itself will be
       included in the zip file.

   SYNOPSIS
       camfi zip-images <flags>

   DESCRIPTION
       Makes a zip archive of all the images in the provided VIA project file.
       If --i is set, then the annotation file itself will be included in the
       zip file.

   FLAGS
       Flags are accepted.


The ``traincamfiannotator`` command
-----------------------------------

``$ traincamfiannotator -- --help``::

   NAME
       traincamfiannotator

   SYNOPSIS
       traincamfiannotator <flags> [VIA_PROJECTS]...

   POSITIONAL ARGUMENTS
       VIA_PROJECTS
           Type: typing.Union[str, os.PathLike]
           Path(s) to VIA project files with annotated motion blurs for training on

   FLAGS
       --load_pretrained_model=LOAD_PRETRAINED_MODEL
           Type: typing.Union[str,...
           Path to model parameters file. If set, will load the pretrained parameters
       --img_dir=IMG_DIR
           Type: typing.Union[str,...
           Path to direcotry containing images. By default inferred from first
           element in
       --crop=CROP
           Type: t...
           Crop images before processing. By default no crop. Original camfi
           data uses --crop=0,0,4608,3312
       --point_r=POINT_R
           Type: int
           Margin added to the coordinates of annotations to determine the
           bounding box of the annotation
       --mask_dilate=MASK_DILATE
           Type: int
           Radius of dilation to apply to training masks
       --min_annotations=MIN_ANNOTATIONS
           Type: int
           Skip images which have fewer than min_annotations annotations. E.g.
           to only train on images with at least one annotation set
           `min_annotations=1`
       --max_annotations=MAX_ANNOTATIONS
           Type: float
           Skip images which have more than max_annotations annotations. Set
           this if you are running into memory issues when training on a GPU.
       --exclude=EXCLUDE
           Type: typing.Union[str,...
           Path to file containing list of images to exclude (one per line).
           E.g. to exclude a test set
       --device=DEVICE
           Type: str
           E.g. "cpu" or "cuda"
       --num_classes=NUM_CLASSES
           Type: int
           Number of target classes (including background)
       --batch_size=BATCH_SIZE
           Type: int
           Number of images to load at once
       --num_workers=NUM_WORKERS
           Type: int
           Number of worker processes for data loader to spawn
       --num_epochs=NUM_EPOCHS
           Type: int
           Number of epochs to train
       --outdir=OUTDIR
           Type: typing.Union[str, os.PathLike]
           Path to directory where to save model(s)
       --model_name=MODEL_NAME
           Type: typing.Union[str, NoneType]
           Identifier to include in model save file. By default the current
           date in YYYYmmdd format
       --save_intermediate=SAVE_INTERMEDIATE
           Type: bool
           If True, model is saved after each epoch


The ``camfiannotate`` command
-----------------------------

``$ camfiannotate -- --help``::

   NAME
       camfiannotate - Provides a cli for automatic annotation of camfi images.

   SYNOPSIS
       camfiannotate VIA_PROJECT <flags>

   DESCRIPTION
       Provides a cli for automatic annotation of camfi images.

   POSITIONAL ARGUMENTS
       VIA_PROJECT
           Type: typing.Union[str, os.PathLike]
           Path to via project json file

   FLAGS
       --model=MODEL
           Type: typing.Union[str, os.PathLike]
           Default: 'release'
           Either a path to state dict file which defines the segmentation
           model, or a url pointing to a model to download from the internet,
           or "release" or "latest". See `camfi download-model --help` for more
           information.
       --num_classes=NUM_CLASSES
           Type: int
           Default: 2
           Number of classes in the model. Must correspond with how model was trained
       --img_dir=IMG_DIR
           Type: Optiona...
           Default: None
           Path to direcotry containing images. By default inferred from via_project
       --crop=CROP
           Type: Optional[typing.Union[typing.Tuple[int, int, int, int], NoneType]]
           Default: None
           Crop images before processing. By default no crop. Original camfi
           data uses --crop=0,0,4608,3312
       --device=DEVICE
           Type: str
           Default: 'cpu'
           Specifies device to run inference on. Set to cuda to use gpu.
       --backup_device=BACKUP_DEVICE
           Type: Optional[typing.Unio...
           Default: None
           Specifies device to run inference on when a runtime error occurs
           while using device. Probably only makes sense to set this to cpu if
           device=cuda
       --split_angle=SPLIT_ANGLE
           Type: float
           Default: 15.0
           Approximate maximum angle between polyline segments in degrees.
       --poly_order=POLY_ORDER
           Type: int
           Default: 2
           Order of polynomial used for fitting motion blur paths.
       --endpoint_method=ENDPOINT_METHOD
           Type: typing.Tuple[str, typing.Any]
           Default: ('truncate', 10)
           Method to find endpoints of motion blurs. Currently implemented:
           --endpoint_method=truncate,n  (where n is a positive int)
           --endpoint_method=quantile,q  (where q is a float between 0. and 1.)
       --score_thresh=SCORE_THRESH
           Type: float
           Default: 0.4
           Score threshold between 0. and 1. for annotations
       --overlap_thresh=OVERLAP_THRESH
           Type: float
           Default: 0.4
           Minimum proportion of overlap between two instance segmentation
           masks to infer that one of the masks should be discarded
       --edge_thresh=EDGE_THRESH
           Type: int
           Default: 10
           Minimum distance an annotation has to be from the edge of the image
           before it is converted from polyline to circle
       --o=O
           Type: Optiona...
           Default: None
           Path to output file. Default is to output to stdout

   NOTES
       You can also use flags syntax for POSITIONAL ARGUMENTS
