VENV_NAME?=.venv

USER_PYTHON ?= python3
VENV_PYTHON=${VENV_NAME}/bin/python

.PHONY = prepare-venv lint format clean

.DEFAULT_GOAL = install

prepare-venv: $(VENV_NAME)/bin/python

$(VENV_NAME)/bin/python:
	make clean && ${USER_PYTHON} -m venv $(VENV_NAME)

install: prepare-venv
	${VENV_PYTHON} -m pip install -U pip
	${VENV_PYTHON} -m pip install -r requirements.txt
	${VENV_PYTHON} -m pip install -e .

lint: install
	${VENV_PYTHON} -m ruff .

format: install
	${VENV_PYTHON} -m black .

clean:
	rm -rf .venv
	rm -rf .ruff_cache
	rm -rf opm_thesis.egg-info
	find . -name __pycache__ -type d -exec rm -r {} +
	find . -name .ipynb_checkpoints -type d -exec rm -r {} +
