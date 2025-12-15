# https://github.com/casey/just

# Default recipe, it's run when just is invoked without a recipe
default:
    just --list --unsorted

# Run ruff formatting
format:
	uv run ruff format

# Run ruff linting and mypy type checking
lint:
	uv run ruff check --fix
	uv run mypy --ignore-missing-imports --install-types --non-interactive --package python_repo_template

# Build docker image
dockerize:
	docker build -t python-repo-template .

# Setup
setup:
	uv venv

# Use it like: just run METHOD. Possible METHODs: odw, digs, nsh, ncr  
run method:
	uv run src/train.py --config ./configs/train_{{method}}.yaml
