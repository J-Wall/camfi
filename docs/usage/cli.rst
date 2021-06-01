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

1. ``camfi``: provides various utilities for working with annotation files
   through subcommands (shown below).

2. ``traincamfiannotator``: the command for training the camfi automatic
   annotator

3. ``camfiannotate``: the command for running inference (automatic annotation)


The ``camfi`` command
---------------------

Running ``$ camfi -- --help`` will show the global options for all ``camfi``
subcommands:

.. literalinclude:: helppages/camfi.txt

The help page of each subcommand is provided below

``camfi add_metadata``
^^^^^^^^^^^^^^^^^^^^^^

``$ camfi add_metadata -- --help``:

.. literalinclude:: helppages/add_metadata.txt


``camfi download_model``
^^^^^^^^^^^^^^^^^^^^^^^^

``$ camfi download_model -- --help``:

.. literalinclude:: helppages/download_model.txt


``camfi extract_wingbeats``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``$ camfi extract_wingbeats -- --help``:

.. literalinclude:: helppages/extract_wingbeats.txt


``camfi filelist``
^^^^^^^^^^^^^^^^^^

``$ camfi filelist -- --help``:

.. literalinclude:: helppages/filelist.txt


``camfi filter``
^^^^^^^^^^^^^^^^

``$ camfi filter -- --help``:

.. literalinclude:: helppages/filter.txt


``camfi merge_annotations``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``$ camfi merge_annotations -- --help``:

.. literalinclude:: helppages/merge_annotations.txt


``camfi remove_unannotated``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``$ camfi remove_unannotated -- --help``:

.. literalinclude:: helppages/remove_unannotated.txt


``camfi validate_annotations``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``$ camfi validate_annotations -- --help``:

.. literalinclude:: helppages/validate_annotations.txt


``camfi zip_images``
^^^^^^^^^^^^^^^^^^^^

``$ camfi zip_images -- --help``:

.. literalinclude:: helppages/zip_images.txt


The ``traincamfiannotator`` command
-----------------------------------

``$ traincamfiannotator -- --help``:

.. literalinclude:: helppages/traincamfiannotator.txt


The ``camfiannotate`` command
-----------------------------

``$ camfiannotate -- --help``:

.. literalinclude:: helppages/camfiannotate.txt
