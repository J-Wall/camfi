# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

api/modules.rst:
	sphinx-apidoc -f -o api .. ../setup.py ../camfiutils.py

usage/installation.rst:
	touch usage/installation.rst

usage/camera_setup.rst:
	touch usage/camera_setup.rst

usage/image_annotation.rst:
	touch usage/image_annotation.rst

usage/automatic_image_annotation.rst:
	touch usage/automatic_image_annotation.rst

usage/data_analysis.rst:
	touch usage/data_analysis.rst

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile api/modules.rst usage/installation.rst usage/camera_setup.rst usage/image_annotation.rst usage/automatic_image_annotation.rst usage/data_analysis.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
