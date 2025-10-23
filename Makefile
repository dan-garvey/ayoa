.PHONY: setup run test demo clean format lint server test-quick

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

run:
	. .venv/bin/activate && python -m core.cli

server:
	. .venv/bin/activate && uvicorn core.api.server:app --reload --port 8081

test:
	. .venv/bin/activate && pytest -v --cov=core --cov-report=term-missing

test-quick:
	. .venv/bin/activate && pytest -x -q

demo:
	. .venv/bin/activate && python -m core.cli create

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf saves/*.json core.db

format:
	. .venv/bin/activate && black . && ruff check --fix .

lint:
	. .venv/bin/activate && ruff check . && mypy core
