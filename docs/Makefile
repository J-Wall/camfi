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

api/modules.rst: Makefile
	sphinx-apidoc -f -H API -o api .. ../setup.py ../camfiutils.py

usage/helppages/camfi.txt: Makefile build_cli.py
	python build_cli.py

notebooks = usage/notebooks/activity_analysis.rst usage/notebooks/annotation_evaluation.rst usage/notebooks/camera_calibration.rst usage/notebooks/camfi_autoannotator_training.rst usage/notebooks/wingbeat_analysis.rst

$(notebooks): usage/notebooks/%.rst: ../examples/%.ipynb
	cp $< usage/notebooks/
	jupyter nbconvert --to rst usage/notebooks/$*.ipynb
	rm usage/notebooks/$*.ipynb

usages = usage/installation.rst usage/camera_setup.rst usage/image_annotation.rst usage/automatic_image_annotation.rst usage/data_analysis.rst usage/cli.rst usage/helppages/camfi.txt

# Catch-all target: route multiple targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html latexpdf: Makefile conf.py ../requirements.txt api/modules.rst usage/notebooks/notebooks.rst $(usages) $(notebooks)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
