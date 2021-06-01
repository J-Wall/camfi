Automatic image annotation
==========================

Camfi can perform the annotation process for you automatically by using an
annotation model based of Mask R-CNN. We have trained the model on images of
Bogong moth motion blurs, and it gives good results. It is very simple to train
and training can be done on Google Colab, meaning that training can be done
even if you do not have access to GPU compute.


Inference (performing automatic annotation)
-------------------------------------------

If you have trained you model (see next section for how to do this), or if you
want to use the included model, you can use ``camfiannotate`` to automatically
annotate your images. Sample usage::

   $ camfiannotate via_annotation_project_file.json \
       --o autoannotated.json \
       --model release \
       --crop=0,0,4608,3312 \  # To crop out trailcamera info overlay
       --device cuda \
       --backup-device cpu \
       --score-thresh 0.4

This will annotate all the images in the input VIA project file using the
release model, outputting a new VIA project file. The above options tell
``camfiannotate`` to crop the images before running inference on them, and to
use cuda (a GPU) to run the inference. By setting ``--backup-device cpu``, we
tell the annotator to run inference on the CPU for images with fail on the GPU
due to memory constraints (inference on images with lots of motion blurs in
them takes up more memory). Finally, only annotations which have a score of at
least 0.4 will be output. For model validation, it is recommended to set this
to 0. Besides, you can always filter these later, for example::

   $ camfi filter \
       --i autoannotated.json \
       --o autoannotated_0.9.json \
       --by score \
       --minimum 0.9

If you want to use a different model, you can set ``--model <filepath>``, where
``<filepath>`` is the path to the model you want to use. Alternatively you can
set ``--model latest``, which will check github for the latest released model.


Training
--------

To train the model, you will first need a set of manually annotated images (see
above for instructions to do this). Once you have this, you can select just the
images which have at least one annotation for training and validation
purposes::

   $ camfi remove-unannotated \
       --i via_annotation_project_file.json \
       --o annotated_only.json

If you are training on a server or Colab instance, you might want to package
these images up along with the annotations in a zip file for easy uploading of
only the data we actually need::

   $ camfi zip-images \
       --i annotated_only.json \
       --o annotated_only.zip

To define a test set, we make a text file listing a random subset of the
images.  This will make a test set of 50 images::

   $ camfi filelist --i annotated_only.json --shuffle 1234 | head -n 50 \
       > test_set.txt

We are now ready for training. This can either be done from the command line
using the ``traincamfiannotator`` command. For help on how to use this, run::

   $ traincamfiannotator --help

Alternatively you can train on Colab. Please refer to the example model
training notebook :doc:`notebooks/camfi_autoannotator_training`.


Validation
----------

To validate our automatic annotation model, we need a VIA project file
containing manual annotations (e.g.
``via_annotation_project_file_with_metadata.json``), and a second VIA project
file containing only the images we want to run the validation on. For example,
we may want to run validation on on our test set (e.g. those in
``test_set.txt``). To make a test set VIA project file, we proceed in the same
way as we did to create our original VIA project file (i.e. run ``via.html``,
and select "Project" > "Add url or path from text file". Then select
``test_set.txt``, and save the project as ``test_set.json``).

To validate the automatic annotations, we need to first get them. Once we have
the VIA project file for the images we want to run validation on, we run
``camfiannotate`` on it::

   $ camfiannotate test_set.json \
       --o test_set_autoannotated.json \
       --crop=0,0,4608,3312 \  # To crop out trailcamera info overlay
       --device cuda \
       --backup-device cpu \
       --score-thresh 0.0  # Important to set this to 0.0 for validation

This gives us the automatic annotations. To validate these against
``via_annotation_project_file_with_metadata.json``, we run::

   $ camfi validate-annotations \
       via_annotation_project_file_with_metadata.json \
       --i test_set.json \
       --o validation.json

This gives us ``validation.json``, which contains the validation data. For an
example of how to interpret this data see the example notebook
:doc:`notebooks/annotation_evaluation`.
