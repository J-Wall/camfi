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

source/modules.rst: Makefile
	SPHINX_APIDOC_OPTIONS='members,special-members,show-inheritance' sphinx-apidoc -f -M -H API -e -o source .. ../setup.py

source/config_schema.json: FORCE
	python -c "from camfi.projectconfig import CamfiConfig; print(CamfiConfig.schema_json(indent=2))" > $@

FORCE: ;

source/cabramurra_config.json: ../examples/data/cabramurra_config.yml
	python -c "from camfi.projectconfig import CamfiConfig; import os; os.chdir('../examples'); print(CamfiConfig.parse_yaml_file('data/cabramurra_config.yml').json(exclude_unset=True, indent=2))" > $@

notebooks = usage/notebooks/activity_analysis.rst usage/notebooks/annotation_evaluation.rst usage/notebooks/camera_calibration.rst usage/notebooks/camfi_autoannotator_training.rst usage/notebooks/wingbeat_analysis.rst

notebook_files = usage/notebooks/activity_analysis_files usage/notebooks/annotation_evaluation_files usage/notebooks/camera_calibration_files usage/notebooks/wingbeat_analysis_files

$(notebooks): usage/notebooks/%.rst: ../examples/%.ipynb
	cp $< usage/notebooks/
	jupyter nbconvert --to rst usage/notebooks/$*.ipynb
	rm usage/notebooks/$*.ipynb

usages = usage/installation.rst usage/camera_setup.rst usage/image_annotation.rst usage/automatic_image_annotation.rst usage/data_analysis.rst usage/cli.rst

# Catch-all target: route multiple targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html latexpdf: Makefile conf.py ../requirements.txt source/modules.rst usage/notebooks/notebooks.rst source/config_schema.json source/cabramurra_config.json $(usages) $(notebooks)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -f source/*.rst $(notebooks)
	rm -rf _build $(notebook_files)
