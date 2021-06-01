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

Camfi requires python 3.7 or greater.

See requirements.txt_ for concrete dependencies.

.. _requirements.txt: https://github.com/J-Wall/camfi/blob/main/requirements.txt

Note: Installing using ``$ pip install camfi`` will only install the
dependencies for the command line tools. The example notebooks have some
additional dependencies, which can be installed by cloning the repository, and
installing from the requirements file::

$ git clone https://github.com/J-Wall/camfi.git
$ cd camfi
$ pip install -r requirements.txt
