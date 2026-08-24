.PHONY: install run debug lint lint-strict clean re

ARGS ?=
MYPY_FLAGS=\
	--warn-return-any\
	--warn-unused-ignores\
	--ignore-missing-imports\
	--disallow-untyped-defs\
	--check-untyped-defs

install:
	@echo "Installing packages.."
	@uv sync

run:
	@uv run python -m src $(ARGS)

debug:
	@uv run python -m pdb -m src $(ARGS)

lint:
	@echo "Running flake8"
	@uv run python -m flake8 .

	@echo "Running mypy"
	@uv run python -m mypy . $(MYPY_FLAGS)

lint-strict:
	@echo "Running flake8"
	@uv run python -m flake8 .

	@echo "Running mypy (strict mode)"
	@uv run python -m mypy . --strict

clean:
	@echo "Cleaning Repo.."
	@rm -rf .mypy_cache
	@rm -rf ./src/__pycache__
	@rm -rf ./llm_sdk/llm_sdk/__pycache__

re: clean run
