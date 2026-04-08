set shell := ["bash", "-cu"]

dataset := "data/sift-128-euclidean.hdf5"
python_version := "3.14"

default:
    @just --list

download-data:
    @mkdir -p data
    @if [ -f data/sift-128-euclidean.hdf5 ]; then \
        echo "Dataset already exists. Skipping download."; \
        exit 0; \
    fi
    @echo "Downloading SIFT1M dataset..."
    @if command -v curl >/dev/null 2>&1; then \
        curl -L -o data/sift-128-euclidean.hdf5 https://ann-benchmarks.com/sift-128-euclidean.hdf5; \
    elif command -v wget >/dev/null 2>&1; then \
        wget -O data/sift-128-euclidean.hdf5 https://ann-benchmarks.com/sift-128-euclidean.hdf5; \
    else \
        echo "Error: neither curl nor wget is installed."; \
        exit 1; \
    fi
    @echo "Download complete."

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

clean:
    rm -rf cache
    rm -rf __pycache__
    find . -name "*.pyc" -delete
