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

## Docker

- build the docker container using `docker-compose build` (You need to make sure that docker has enough memory to build the image)
- start the jupyter lab docker container using `docker-compose up`
- Copy the link (incl. token) from the console and paste it into the browser

## Git LFS

- In order to use git lfs, please refer to the [official instructions](https://git-lfs.github.com/)
- Configure which files to store in Git LFS using the `git lfs track "*.file_ending"`command
