install:
	pip install -e .[dev]

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache .mypy_cache dist/ build/ *.egg-info
	find . -type f -name "*.duckdb" -exec rm -f {} +
	find . -type f -name "*.json" -exec rm -f {} +
	find . -type d -name "*chromaDB" -exec rm -r {} +

test:
	pytest

format:
	black .

lint:
	ruff .

load_data:
	python3 backend/loading_documents.py 

run_app:
	.venv/bin/streamlit run frontend/web_app.py