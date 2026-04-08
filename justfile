set shell := ["bash", "-cu"]

dataset := "data/sift-128-euclidean.hdf5"
python_version := "3.14"

default:
    @just --list

download-data:
    bash scripts/download_data.sh

setup:
    uv python install {{ python_version }}
    uv venv --python {{ python_version }} .venv
    uv sync --group dev

ui:
    uv run streamlit run ann_app.py

lint:
    uv run ruff check .

pylint:
    uv run pylint ann_app.py ann_backend.py ann_faiss.py ann_pgvector.py tests/test_backend_contract.py tests/test_pgvector_backend.py

pyright:
    uv run pyright

lint-fix:
    uv run ruff check . --fix

format:
    uv run black .

check: format lint pylint pyright

fix: format lint-fix

test:
    uv run pytest

reset-pgvector:
    uv run python scripts/reset_pgvector.py

clean:
    rm -rf cache
    rm -rf __pycache__
    find . -name "*.pyc" -delete
