# opm-thesis

## Setup

### Installation

This command will create a virtuelenv and install all dependencies

`make install`

### Lint

This command will ensure all dependencies are installed and run ruff

`make lint`

### Format

This command will ensure all dependencies are installed and run black

`make format`

### Cleanup

This command will delete the virtualenv

`make clean`

## Development

- activate python environment: `source .venv/bin/activate`
- run python script: `python <filename.py> `, e.g. `python train.py`
- install new dependency: `pip install sklearn`
- add new dependency to requirements.txt
