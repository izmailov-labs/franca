.PHONY: install lint fmt typecheck test test-contract test-mock aimock cov encoding build clean all

# The aimock release the mock suite runs against. tests/mock/conftest.py carries the
# same default and reads this variable, so overriding it here overrides it there.
AIMOCK_VERSION ?= 1.42.0
AIMOCK_PORT ?= 4010
export AIMOCK_VERSION

install:  ## Sync the dev environment and install the pre-commit hooks
	uv sync --group dev
	uv run pre-commit install

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

fmt:  ## Autofix and format
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Strict type check (src and tests)
	uv run mypy

test:  ## Run the test suite (offline; live provider calls are deselected)
	uv run pytest

test-contract:  ## Live provider calls; needs real keys in the environment or .env
	uv run pytest -m contract tests/contract

test-mock:  ## Real HTTP against the aimock mock server; spawns it via npx unless AIMOCK_BASE_URL is set
	uv run pytest -m mock tests/mock

aimock:  ## Run aimock in the foreground on $(AIMOCK_PORT) with tests/mock/fixtures, reloading on edits
	AIMOCK_API_KEYS=franca-mock-key npx --yes --package @copilotkit/aimock@$(AIMOCK_VERSION) llmock --port $(AIMOCK_PORT) --fixtures tests/mock/fixtures --watch

cov:  ## Run tests with coverage (fail_under=90)
	uv run pytest --cov --cov-report=term-missing

encoding:  ## Fail on any open()/read_text() missing an explicit encoding=
	PYTHONWARNDEFAULTENCODING=1 uv run pytest -q --no-cov -W error::EncodingWarning

build:  ## Build the wheel and sdist into ./dist and validate the metadata
	rm -rf dist
	uv build --out-dir dist
	uvx twine check dist/*

clean:
	rm -rf dist build .coverage .coverage.* htmlcov
	rm -rf .pytest_cache .mypy_cache .ruff_cache

all: lint typecheck cov build
