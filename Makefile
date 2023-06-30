VENV_NAME=.venv

USER_PYTHON= python
VENV_PYTHON=$(VENV_NAME)\Scripts\python

.PHONY = prepare-venv install lint format clean

.DEFAULT_GOAL = install

prepare-venv: $(VENV_NAME)/Scripts/activate

$(VENV_NAME)/Scripts/activate:
	@if [ -d $(VENV_NAME) ]; then rmdir /S /Q $(VENV_NAME); else echo Directory $(VENV_NAME) does not exist.; fi
	@$(USER_PYTHON) -m venv $(VENV_NAME)


install: prepare-venv
	${VENV_PYTHON} -m pip install -U pip
	${VENV_PYTHON} -m pip install -r requirements.txt
	${VENV_PYTHON} -m pip install -e .

lint: install
	${VENV_PYTHON} -m ruff .

format: install
	${VENV_PYTHON} -m black .

clean:
	clean:
	@IF EXIST $(VENV_NAME) (rmdir /S /Q $(VENV_NAME)) else (echo Directory $(VENV_NAME) does not exist.)
	@rmdir /S /Q opm_thesis.egg-info
	@rmdir /S /Q .ruff_cache
	@FOR /D /R .\opm_thesis %%d IN (__pycache__) DO @IF EXIST "%%d" (rmdir /S /Q "%%d") -type d -exec rm -r {} +
