install:
	pip install -e .[dev]

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache .mypy_cache dist/ build/ *.egg-info

test:
	pytest

format:
	black .

lint:
	ruff .