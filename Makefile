.PHONY: demo test lint clean setup generate-data

demo: generate-data
	streamlit run app.py

test:
	python -m pytest tests/ -v

lint:
	ruff check . && ruff format --check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true

setup:
	pip install -r requirements-dev.txt

generate-data:
	@test -f demo_data/ecommerce.csv || python demo_data/generate_demo_data.py
