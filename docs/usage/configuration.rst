Configuration
=============

Rather than relying on
command-line options or environment variables,
Camfi is configured using
a configuration file
which can be
either JSON or YAML
(actually, a subset of YAML
called `StrictYAML <https://hitchdev.com/strictyaml/>`_).
This makes
camfi experiments
much more reproducible.

It is suggested to create
a camfi configuration file
before heading out
into the field
to set up your cameras.
Then you can start putting
in some of the metadata
(e.g. camera placement info)
as you go.
This will make
your life a lot easier
when it comes time
to analyse your data.

The `JSON-Schema <https://json-schema.org/>`_ for
camfi configuration files is
:download:`available here <../source/config_schema.json>`.
It should be possible
to use the schema
to generate
a web IU form
for easily producing
json configuration files
for Camfi,
however this has not been implemented.
`raise an issue on GitHub <https://github.com/J-Wall/camfi/issues/new>`_
if you plan on using Camfi
and would like to use something like that
rather than a text editor.

Example configuration file
--------------------------

Below is
an :download:`example configuration file <../../examples/data/cabramurra_config.yml>`
in yaml format.
:download:`The same configuration <../../examples/data/cabramurra_config.json>`
is also available
in json format.

.. include:: ../../examples/data/cabramurra_config.yml
    :code: YAML


Configuration specification
---------------------------

What follows is the specification
for Camfi configuration files
in a human-readable
tabulated format.

.. jsonschema:: ../source/config_schema.json
    :lift_description:
    :lift_definitions:
    :auto_target:
    :auto_reference:
