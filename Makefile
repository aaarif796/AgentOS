install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

boundaries:
	tach check
	tach check-external

typecheck:
	mypy src tests

test:
	pytest

check: lint boundaries typecheck test

serve:
	agentos serve
