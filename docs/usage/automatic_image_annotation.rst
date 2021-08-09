Automatic image annotation
==========================

Camfi can perform
the annotation process
for you
automatically by using
an annotation model based
on Mask R-CNN.
We have trained
the model
on images
of Bogong moth motion blurs,
and it gives good results.

In addition to a Camfi :doc:`configuration file <configuration>`,
a valid VIA project file is required
to proceed
with any of
the following steps.
See :ref:`via-project-setup`
for instructions on making
this file.

Inference (performing automatic annotation)
-------------------------------------------

If you have trained
your model
(see :ref:`training` section for how to do this),
or if you want to use
the included model,
you can use ``camfi annotate`` to automatically
annotate your images.
``annotator.inference`` must be configured in the configuration file.
Example ``config.yml``
(assuming camfi is being run from
the examples directory in the camfi git repo)::

    root: data
    via_project_file: data/cabramurra_all_annotations.json
    annotator:
      crop:
        x0: 0
        y0: 0
        x1: 4608
        y1: 3312
      inference:
        output_path: data/cabramurra_autoannotated.json
        device: cuda
        backup_device: cpu

Then running::

    $ camfi --config config.yml annotate

Will annotate all the images in the input VIA project file
``data/cabramurra_alla_annotations.json``
using the
default "release" model, outputting a new VIA project file
called ``data/cabramurra_autoannotated.json``.
The above configuration options tell
``camfi`` to crop the images before running inference on them,
and to use cuda (a GPU) to run the inference.
By setting ``backup_device: cpu``,
we tell the annotator to
run inference on the CPU for images with fail on the GPU
due to memory constraints
(inference on images with
lots of motion blurs in them takes up more memory).

If you want to use a different model,
you can set ``model: <filepath>``,
where ``<filepath>`` is the path to the model you want to use.

.. _training:

Training
--------

To train the model,
you will first need a set of manually annotated images
(see :doc:`image_annotation` for instructions on how to do this).

To define a test set, we make a text file listing a random subset of the
images. We would like to inlcude only images with at least one annotation,
so the following should be included in our config.yml::

    filters:
      image_filters:
        min_annotations: 1

Then we run the following command
to make a test set of 50 images::

    $ camfi --config config.yml filter-images filelist \
         | shuf | head -n 50 > data/cabramurra_test_set.txt

We are now ready for training.
This can be done from the command line
using the ``camfi train`` command.
We need ``annotator.training``
to be configured in our config.yml::

    root: data
    via_project_file: data/cabramurra_all_annotations.json
    annotator:
      crop:
        x0: 0
        y0: 0
        x1: 4608
        y1: 3312
      training:
        mask_maker:
          shape:
          - 3312
          - 4608
          mask_dilate: 5
        min_annotations: 1
        test_set_file: data/cabramurra_test_set.txt
        device: cuda
        batch_size: 5
        num_workers: 2
        num_epochs: 20
        outdir: data
        save_intermediate: yes

Then we can run::

    $ camfi --config config.yml train

Which will train camfi's instance segmentation model
on the GPU,
saving after each epoch into the ``data`` directory.


Validation
----------

To validate our automatic annotation model,
we need a VIA project file containing manual annotations
(e.g. the one used to train the model)
and a second VIA project file containing
automatically aquired annotations
(aquired using the model we want to validate).

Validation requires ``annotator.validation`` to be configured.
You should also include the configuration used for training,
which will allow you to validate against
the "train" and "test" image sets,
as well as the "all" image set.
With the following config.yml::

    root: data
    via_project_file: data/cabramurra_all_annotations.json
    annotator:
      crop:
        x0: 0
        y0: 0
        x1: 4608
        y1: 3312
      training:
        mask_maker:
          shape:
          - 3312
          - 4608
          mask_dilate: 5
        min_annotations: 1
        test_set_file: data/cabramurra_test_set.txt
        device: cuda
        batch_size: 5
        num_workers: 2
        num_epochs: 20
        outdir: data
        save_intermediate: yes
      inference:
        output_path: data/cabramurra_autoannotated.json
        device: cuda
        backup_device: cpu
      validation:
        autoannotated_via_project_file: data/cabramurra_autoannotated.json
        image_sets:
        - all
        - test
        - train
        output_dir: data

and assuming we have already run training and inference,
we can then run::

    $ camfi --config config.yml validate

or we can conveniently run all three in one command::

    $ camfi --config config.yml train annotate validate

This will give us three validation files,
``data/validation.all.json``,
``data/validation.test.json``,
and
``data/validation.train.json``,
which contain the validation data.
For an example
of how to interpret this data
see the example notebook
:doc:`notebooks/annotation_evaluation`.
