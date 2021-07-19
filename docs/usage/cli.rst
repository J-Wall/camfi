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
