# ANN Index Playground

An interactive local app for explaining why vector indexes exist and how `Flat`, `HNSW`, and `IVF` behave differently on the SIFT1M dataset.

This repo was created as a companion app for a YouTube video. The goal is not to produce a full benchmark suite or a production retrieval system. The goal is to make ANN trade-offs visible and demoable by changing a few parameters and rerunning the comparison.

<img src="./screenshot.png" alt="ANN Index Playground UI" width="700" />

## What the app is for

The app is designed to help explain:

- why exact search is simple but expensive at larger scales
- why HNSW is often a strong default when recall matters
- why IVF is useful when you want faster build times and a tunable search surface
- how query-time tuning differs from build-time tuning
- how the same index can feel very different when you change `efSearch`, `nlist`, or `nprobe`
- how ANN trade-offs differ between `faiss` and `pgvector`

The UI is intentionally opinionated:

- it keeps the benchmark scope narrow
- it focuses on SIFT1M and Euclidean distance
- it favors side-by-side comparisons over raw charts
- it keeps previous runs visible so parameter changes are easy to compare

## What the app does

For each run, the app compares:

- `Flat`
- `HNSW`
- `IVF`

The app can run against two backends:

- `faiss`
- `pgvector`

It measures:

- average latency
- p95 latency
- recall@10
- relative speed versus Flat

Build time and index size are shown separately under the `HNSW` and `IVF` control groups, because those values depend on build-time settings and are otherwise repetitive across run history.

## Dataset

This app expects the ANN-Benchmarks SIFT1M HDF5 dataset at:

```text
data/sift-128-euclidean.hdf5
```

If you do not already have it, download it with:

```bash
just download-data
```

## Why SIFT1M and L2

This repo uses the classic SIFT1M benchmark because it is a well-known ANN dataset and keeps the story simple.

- dataset: `SIFT1M`
- distance metric: `L2` / Euclidean distance
- nearest-neighbor target: `k = 10`

That makes the app a good teaching tool for ANN index behavior, even though it is not intended to represent modern semantic embedding workloads directly.

## Setup

Install `just` if needed:

```bash
brew install just
```

This repo uses `uv` for:

- installing Python `3.14`
- creating the virtual environment
- syncing dependencies
- running local commands

The `just` commands in this repo assume `uv` is installed.

Then set up the project:

```bash
just setup
```

This repo targets Python `3.14` and uses `uv` to create and manage the environment.

If you prefer to skip `just`, the equivalent commands are:

```bash
uv python install 3.14
uv venv --python 3.14 .venv
uv sync --group dev
```

## Running the app

Launch the Streamlit UI with:

```bash
just ui
```

If you prefer to run it directly without `just`:

```bash
uv run streamlit run ann_app.py
```

## Backend selection

Backend choice is controlled through `.env`:

```env
ANN_BACKEND=faiss
```

Supported values:

- `faiss`
- `pgvector`

`faiss` is the default.

For `pgvector`, the app also reads:

```env
PGVECTOR_DATABASE_URL=postgresql:///ann_indexes_pgvector
PGVECTOR_ADMIN_DATABASE_URL=postgresql:///postgres
```

The admin database URL is used only to create the target database if it does not already exist.

## Using the pgvector backend

The pgvector backend is partly automatic, but not completely zero-setup.

The app will do these steps for you:

- create the target database if it does not exist
- run `CREATE EXTENSION IF NOT EXISTS vector` in that database
- create the required tables
- build the HNSW and IVFFlat indexes it needs

These machine-level prerequisites must already be true:

- PostgreSQL is installed
- the PostgreSQL server is running
- the `vector` extension is installed for that Postgres instance
- your local user can connect and create databases

On macOS with Homebrew, the setup looks like:

```bash
brew install postgresql@18
brew services start postgresql@18
brew install pgvector
```

Sanity-check the server:

```bash
psql postgres
```

Then set `.env`:

```env
ANN_BACKEND=pgvector
PGVECTOR_DATABASE_URL=postgresql:///ann_indexes_pgvector
PGVECTOR_ADMIN_DATABASE_URL=postgresql:///postgres
VECTOR_COUNT=1000000
INCLUDE_HNSW=true
INCLUDE_IVF=true
```

And launch the app normally:

```bash
just setup
just ui
```

If you want to force the pgvector backend to rebuild its tables and indexes from scratch, reset the app database first:

```bash
just reset-pgvector
```

That drops and recreates only the database pointed to by `PGVECTOR_DATABASE_URL`.

Important caveat:

- if `pgvector` was installed against a different Postgres version than the server you are running, extension creation can fail
- if your Postgres auth or local socket setup differs from the defaults, you may need to adjust the connection URLs in `.env`

## Vector count configuration

The number of indexed vectors is controlled through `.env` rather than hardcoded in the app:

```env
VECTOR_COUNT=1000000
```

If you change `VECTOR_COUNT`, restart the app so the new value is picked up.

This setting matters a lot:

- it changes index build time
- it changes index size
- it can meaningfully change latency and recall behavior

The app currently treats vector count as an app-level setting, not a live in-UI control, because cached index files are keyed by that value.

## Saved results

Each time you click `Run`, the app appends the current results to:

```text
data/results.csv
```

This file is intended as a simple inspection log so you can compare runs across:

- different parameter settings
- different app sessions
- different backends such as `faiss` and `pgvector`

Each saved row includes run metadata plus the per-index result values, such as:

- timestamp
- backend
- vectors
- queries
- index family
- config text
- average latency
- p95 latency
- recall
- speedup versus Flat

The CSV is not shown in the app UI anymore. Inspect it directly if you want a persistent record of previous runs.

## Caching behavior

Caching depends on the selected backend.

### FAISS

The app writes serialized indexes into `cache/`.

These caches are reused when the build-time configuration matches:

- `Flat`: keyed by vector count
- `HNSW`: keyed by vector count, `M`, and `efConstruction`
- `IVF`: keyed by vector count and `nlist`

That means these changes do not require a rebuild:

- `Queries`
- `Timing repeats`
- `Single-query sample size`
- `HNSW efSearch`
- `IVF nprobe`

These changes do require a rebuild:

- `VECTOR_COUNT`
- `HNSW M`
- `HNSW efConstruction`
- `IVF nlist`

### pgvector

The app stores its artifacts in PostgreSQL instead of on disk.

It creates:

- one exact `Flat` table
- one `HNSW` table with HNSW indexes
- one `IVF` table with IVFFlat indexes

This table split is deliberate. It avoids planner ambiguity when both approximate index types exist at once, so the app can compare `Flat`, `HNSW`, and `IVF` in the same UI without guessing which index PostgreSQL used.

The pgvector backend also creates a small metadata table to store:

- index build time
- index size

Build-time cache keys for pgvector are effectively:

- `HNSW`: vector count, `M`, `efConstruction`
- `IVF`: vector count, `nlist`

Query-time-only changes still avoid rebuilds:

- `Queries`
- `Timing repeats`
- `Single-query sample size`
- `HNSW efSearch`
- `IVF nprobe`

## Understanding the controls

### Search settings

- `Vectors`
  - read-only in the UI
  - comes from `.env`
- `Queries`
  - how many queries are used for the current comparison
  - more queries means slower runs but more stable averages
- `Timing repeats`
  - how many times the batch search is repeated for average timing
- `Single-query sample size`
  - how many individual queries are timed to estimate p95 latency

### HNSW

- `M`
  - graph connectivity
  - higher values generally improve recall but increase build time and index size
- `efConstruction`
  - build-time search breadth
  - higher values generally improve graph quality but make builds slower
- `efSearch`
  - query-time search breadth
  - higher values generally improve recall but increase latency

### IVF

- `nlist`
  - number of coarse partitions
  - higher values can improve selectivity but make build/training more expensive
- `nprobe`
  - number of partitions searched at query time
  - higher values generally improve recall but increase latency

## Reading the results

Each run is kept on screen so you can compare changes over time.

The result cards show:

- `Avg Latency`
- `Recall`
- `Vs Flat`
- `p95 Latency`

Recall is displayed as a percentage for readability.

`Vs Flat` is the relative speedup versus exact search:

- above `1.0x` means faster than Flat
- `1.0x` means the same as Flat
- below `1.0x` means slower than Flat

## Developer commands

```bash
just ui
just format
just lint
just pylint
just check
just test
just reset-pgvector
just fix
```

Direct `uv` equivalents:

```bash
uv run streamlit run ann_app.py
uv run black .
uv run ruff check .
uv run pylint ann_app.py ann_backend.py ann_faiss.py ann_pgvector.py tests/test_backend_contract.py tests/test_pgvector_backend.py
```

## Code layout

- [ann_app.py](./ann_app.py)
  - Streamlit UI, backend selection, and run history rendering
- [ann_backend.py](./ann_backend.py)
  - shared backend settings and the small backend contract used by the app
- [ann_faiss.py](./ann_faiss.py)
  - FAISS implementation, file-backed cache handling, and benchmark helpers
- [ann_pgvector.py](./ann_pgvector.py)
  - pgvector implementation, database/index setup, and PostgreSQL-backed search execution
- [pyproject.toml](./pyproject.toml)
  - project configuration, linting, and Python/tooling settings
- [justfile](./justfile)
  - convenience commands for setup and local development
- [tests/test_backend_contract.py](./tests/test_backend_contract.py)
  - FAISS backend contract tests on a tiny synthetic dataset
- [tests/test_pgvector_backend.py](./tests/test_pgvector_backend.py)
  - pgvector integration tests against a temporary PostgreSQL database

## Notes

- This app uses `faiss-cpu`, not a CUDA FAISS build.
- The pgvector backend requires PostgreSQL plus the `vector` extension to be installed and available to the server.
- Absolute timings are hardware-dependent.
- Relative behavior can still be useful for explanation, but should not be overclaimed as universally applicable.
