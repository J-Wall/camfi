Installation
============

Using a virtualenv
------------------

Given the number of dependencies of camfi, it is recommended to install it in a
virtualenv_ or conda environment (e.g. miniconda_). Please see the previous links if you need
help setting one of these up.

.. _virtualenv: https://virtualenv.pypa.io/en/latest/

.. _miniconda: https://docs.conda.io/en/latest/miniconda.html


Installation using pip
----------------------

Installation is as easy as::

    $ pip install camfi

Once you have installed camfi, you can run it from the command line::

    $ camfi <command> <flags>


Requirements
------------

Camfi requires python 3.9 or greater.

A copy of VGG Image Annotator (`VIA`_) Version 2
is also required in order to do manual annotation
of images,
and to set up VIA projects
for automatic annotation by Camfi.

.. _VIA: https://www.robots.ox.ac.uk/~vgg/software/via/

The `GEOS library`_ is also required by Shapely,
which is one of Camfi's dependencies.
This can be installed with conda if you don't have a system install::

    $ conda install -c conda-forge geos

.. _GEOS library: https://trac.osgeo.org/geos/

Concrete python library dependencies
for camfi are provided in requirements.txt_.
They are:

.. literalinclude:: ../../requirements.txt

.. _requirements.txt: https://github.com/J-Wall/camfi/blob/main/requirements.txt

Note: Installing using ``$ pip install camfi`` will will not necessarily
install the exact versions of the dependencies specified above.
If you are running into unusual errors,
try installing the concrete dependencies
of the version of Camfi
you are using.
For example,
you can simply
cloning the repository, and
install from the requirements file::

    $ git clone https://github.com/J-Wall/camfi.git
    $ cd camfi
    $ git checkout <version>  # set to whatever version you are using
    $ pip install -r requirements.txt


Development
-----------

If you want to develop Camfi,
you may want to install
the testing and documentation
building requirements::

    $ git clone https://github.com/J-Wall/camfi.git
    $ cd camfi
    $ pip install -r requirements.txt
    $ pip install -r docs/requirements.txt  # For documentation building
    $ pip install pip install pytest-cov pytest-mypy  # For testing
    $ pip install -e .  # Installs camfi in develop mode

If you are making any changes to
``camfi.datamodel.via_region_attributes.ViaRegionAttributes``,
then you should run the script
``camfi/datamodel/_region_filter_config_dynamic.py``
to rebuild ``camfi/datamodel/region_filter_config.py``
before each commit::

    $ python camfi/datamodel/_region_filter_config_dynamic.py
    $ git add camfi/datamodel/region_filter_config.py

This script has an additional dependency,
which can be installed with pip::

    $ pip install datamodel-code-generator
