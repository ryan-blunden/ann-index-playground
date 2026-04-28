# ANN Index Playground

An interactive local app for explaining why vector indexes exist and how `Flat`, `HNSW`, and `IVF` behave differently on the SIFT1M dataset.

This repo was created as a companion app for a YouTube video. The goal is not to produce a full benchmark suite or a production retrieval system. The goal is to make ANN trade-offs visible and demoable by changing a few parameters and rerunning the comparison.
<img src="./screenshot.png" alt="ANN Index Playground UI" width="700" />

## Quick start

Install `just` and `uv` if needed:

```bash
brew install just uv
```

Set up the project:

```bash
just setup
just download-data
just ui
```

Direct `uv` equivalents:

```bash
uv python install 3.14
uv venv --python 3.14 .venv
uv sync --group dev
uv run streamlit run ann_app.py
```

The dataset is expected at:

```text
data/sift-128-euclidean.hdf5
```

## Backends

Select the backend in `.env`:

```env
ANN_BACKEND=faiss
```

Supported values:

- `faiss`
- `pgvector`
- `actian`

`faiss` is the default.

### FAISS

No extra service setup.

### pgvector

Prerequisites:

- PostgreSQL is installed and running
- the `vector` extension is installed for that Postgres instance
- your local user can connect and create databases

Example local setup on macOS:

```bash
brew install postgresql@18
brew services start postgresql@18
brew install pgvector
```

Example `.env`:

```env
ANN_BACKEND=pgvector
PGVECTOR_DATABASE_URL=postgresql:///ann_indexes_pgvector
PGVECTOR_ADMIN_DATABASE_URL=postgresql:///postgres
PGVECTOR_MAINTENANCE_WORK_MEM=512MB
VECTOR_COUNT=1000000
INCLUDE_HNSW=true
INCLUDE_IVF=true
```

Reset the pgvector database if needed:

```bash
just reset-pgvector
```

### Actian

Run the local service:

```bash
just actian-up
```

Or:

```bash
docker compose -f docker-compose.actian-vectorai.yml up -d
```

Install the official Python SDK:

```bash
pip install actian-vectorai-client
```

Example `.env`:

```env
ANN_BACKEND=actian
ACTIAN_VECTORAI_URL=localhost:6574
VECTOR_COUNT=1000000
INCLUDE_HNSW=true
INCLUDE_IVF=false
```

Actian note:

- gRPC server: `localhost:6574`
- LocalUI: `localhost:6575`
- Actian data is ephemeral in this repo; `just actian-up` starts with a clean store
- the current Actian release used by this app is HNSW-only
- IVF controls are hidden for Actian because the server rejects IVF collection creation

Stop the service:

```bash
just actian-down
```

## Configuration

Important `.env` settings:

```env
ANN_BACKEND=faiss
VECTOR_COUNT=1000000
INCLUDE_HNSW=true
INCLUDE_IVF=true
```

Actian-specific runs should keep `INCLUDE_IVF=false`.

Backend-specific settings:

```env
PGVECTOR_DATABASE_URL=postgresql:///ann_indexes_pgvector
PGVECTOR_ADMIN_DATABASE_URL=postgresql:///postgres
PGVECTOR_MAINTENANCE_WORK_MEM=512MB
ACTIAN_VECTORAI_URL=localhost:6574
```

Restart the app after changing `.env`.

## What the app shows

Each run compares:

- `Flat`
- `HNSW`
- `IVF`

Actian currently compares:

- `Flat`
- `HNSW`

Metrics:

- average single-query latency
- p95 latency
- `recall@10`
- relative speed versus `Flat`

Current shared UI defaults:

- `HNSW`: `M=32`, `efConstruction=200`, `efSearch=20`
- `IVF`: `nlist=1024`, `nprobe=12`

Actian uses the HNSW defaults only.

The controls are split into:

- build-time settings: `HNSW M`, `HNSW efConstruction`, `IVF nlist`
- query-time settings: `HNSW efSearch`, `IVF nprobe`

## Saved results and caching

Results are appended to:

```text
data/results.csv
```

FAISS caches serialized indexes in:

```text
cache/
```

For FAISS and pgvector, changing query-time settings does not require rebuilding indexes:

- `Queries`
- `Timing repeats`
- `Single-query sample size`
- `HNSW efSearch`
- `IVF nprobe`

Changing these does require rebuilds:

- `VECTOR_COUNT`
- `HNSW M`
- `HNSW efConstruction`
- `IVF nlist`

Actian note:

- the current Actian release does not support IVF collections in this app

## Developer commands

```bash
just ui
just actian-up
just actian-down
just format
just lint
just pylint
just pyright
just check
just fix
just test
just reset-pgvector
just clean
```

## Notes

- This app uses `faiss-cpu`, not a CUDA build.
- Absolute timings are hardware-dependent.
- The demo uses `SIFT1M`, Euclidean distance, and `k=10`.
