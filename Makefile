SHELL=/bin/bash
VENV = venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

venv:  ## Set up virtual environment
	python3 -m venv $(VENV)
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt
	$(VENV_BIN)/pip install -r requirements-lint.txt

.PHONY: build
build: venv  ## Compile and install Polars for development
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop

.PHONY: build-release
build-release: venv  ## Compile and install a faster Polars binary
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate && maturin develop --release

.PHONY: fmt
fmt: venv  ## Run autoformatting and linting
	$(VENV_BIN)/black .
	$(VENV_BIN)/blackdoc .
	$(VENV_BIN)/ruff .
	cargo fmt --all
	-dprint fmt	
#-$(VENV_BIN)/mypy

.PHONY: clippy
clippy:  ## Run clippy
	cargo clippy -- -D warnings

.PHONY: pre-commit
pre-commit: fmt clippy  ## Run all code quality checks

.PHONY: test
test: venv build-release  ## Run fast unittests
	$(VENV_BIN)/pytest -s tests