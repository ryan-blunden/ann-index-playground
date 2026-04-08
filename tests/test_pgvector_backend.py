from __future__ import annotations

# pylint: disable=redefined-outer-name
import uuid
from pathlib import Path

import psycopg
import pytest

from ann_backend import BackendSettings, RunConfig
from ann_pgvector import PgvectorBackend
from tests.test_backend_contract import (
    write_synthetic_dataset,
)


@pytest.fixture
def pgvector_backend(tmp_path: Path) -> PgvectorBackend:
    dataset_path = tmp_path / "tiny-sift.hdf5"
    write_synthetic_dataset(dataset_path)

    database_name = f"ann_pgvector_test_{uuid.uuid4().hex[:8]}"
    admin_url = "postgresql:///postgres"
    database_url = f"postgresql:///{database_name}"

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{database_name}"')

    settings = BackendSettings(
        dataset_path=dataset_path,
        cache_dir=tmp_path / "cache",
        vector_count=512,
        database_url=database_url,
        admin_database_url=admin_url,
        include_hnsw=True,
        include_ivf=True,
        default_hnsw_m=16,
        default_hnsw_ef_construction=80,
        default_ivf_nlist=64,
    )
    backend = PgvectorBackend(settings)

    yield backend

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (database_name,))
        conn.execute(f'DROP DATABASE IF EXISTS "{database_name}"')


def test_pgvector_initial_artifacts_are_built_with_progress(pgvector_backend: PgvectorBackend) -> None:
    progress_messages: list[str] = []

    assert pgvector_backend.initial_artifacts_exist() is False

    pgvector_backend.ensure_initial_artifacts(lambda update: progress_messages.append(update.message))

    assert pgvector_backend.initial_artifacts_exist() is True
    assert progress_messages
    assert any("Postgres" in message for message in progress_messages)
    assert any("HNSW" in message for message in progress_messages)
    assert any("IVF" in message for message in progress_messages)


def test_pgvector_cache_summaries_report_built_metadata(pgvector_backend: PgvectorBackend) -> None:
    assert pgvector_backend.hnsw_summary(16, 80) == "Cache not built yet."
    assert pgvector_backend.ivf_summary(64) == "Cache not built yet."

    pgvector_backend.ensure_initial_artifacts()

    assert "Build time:" in pgvector_backend.hnsw_summary(16, 80)
    assert "Index size:" in pgvector_backend.hnsw_summary(16, 80)
    assert "Build time:" in pgvector_backend.ivf_summary(64)
    assert "Index size:" in pgvector_backend.ivf_summary(64)


def test_pgvector_run_comparison_returns_expected_rows_and_metrics(pgvector_backend: PgvectorBackend) -> None:
    pgvector_backend.ensure_initial_artifacts()

    results, metadata = pgvector_backend.run_comparison(
        RunConfig(
            query_count=32,
            k=10,
            repeats=2,
            latency_sample_size=16,
            hnsw_m=16,
            hnsw_ef_construction=80,
            hnsw_ef_search=32,
            ivf_nlist=64,
            ivf_nprobe=8,
        )
    )

    assert list(results["family"]) == ["Flat", "HNSW", "IVF"]
    assert metadata["base_vectors"] == 512
    assert metadata["query_count"] == 32
    assert (results["avg_latency_ms"] > 0).all()
    assert (results["p95_latency_ms"] > 0).all()
    assert results.loc[results["family"] == "Flat", "recall_at_10"].iloc[0] == pytest.approx(1.0)
    assert (results["speedup_vs_flat"] > 0).all()
