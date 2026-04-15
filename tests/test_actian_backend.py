from __future__ import annotations

# pylint: disable=protected-access,redefined-outer-name
from collections.abc import Generator
from pathlib import Path

import pytest
from actian_vectorai import VectorAIClient

from ann_actian import ActianBackend
from ann_backend import BackendSettings, RunConfig
from tests.test_backend_contract import write_synthetic_dataset


@pytest.fixture
def actian_backend(tmp_path: Path) -> Generator[ActianBackend]:
    dataset_path = tmp_path / "tiny-sift.hdf5"
    write_synthetic_dataset(dataset_path)

    service_url = "localhost:50051"
    try:
        with VectorAIClient(service_url, timeout=2.0) as client:
            client.health_check()
    except Exception as exc:  # pragma: no cover - integration precondition
        pytest.skip(f"Actian VectorAI DB is not reachable at {service_url}: {exc}")

    settings = BackendSettings(
        dataset_path=dataset_path,
        cache_dir=tmp_path / "cache",
        vector_count=512,
        service_url=service_url,
        include_hnsw=True,
        include_ivf=True,
        default_hnsw_m=16,
        default_hnsw_ef_construction=80,
        default_ivf_nlist=64,
    )
    backend = ActianBackend(settings)

    yield backend

    with VectorAIClient(service_url, timeout=5.0) as client:
        default_ivf_nprobe = backend.default_ivf_nprobe(settings.default_ivf_nlist)
        for collection_name in [
            backend._flat_collection_name(),
            backend._hnsw_collection_name(settings.default_hnsw_m, settings.default_hnsw_ef_construction),
            backend._ivf_collection_name(settings.default_ivf_nlist, default_ivf_nprobe),
        ]:
            if client.collections.exists(collection_name):
                client.collections.delete(collection_name)


def test_actian_initial_artifacts_are_built_with_progress(actian_backend: ActianBackend) -> None:
    progress_messages: list[str] = []

    assert actian_backend.initial_artifacts_exist() is False

    actian_backend.ensure_initial_artifacts(lambda update: progress_messages.append(update.message))

    assert actian_backend.initial_artifacts_exist() is True
    assert progress_messages
    assert any("Actian" in message for message in progress_messages)
    assert any("Flat" in message for message in progress_messages)
    assert any("HNSW" in message for message in progress_messages)
    assert any("IVF" in message for message in progress_messages)


def test_actian_cache_summaries_report_built_metadata(actian_backend: ActianBackend) -> None:
    assert actian_backend.hnsw_summary(16, 80) == "Cache not built yet."
    assert actian_backend.ivf_summary(64, 8) == "Cache not built yet."

    actian_backend.ensure_initial_artifacts()

    assert "Prep time:" in actian_backend.hnsw_summary(16, 80)
    assert "Prep time:" in actian_backend.ivf_summary(64, 8)


def test_actian_run_comparison_returns_expected_rows_and_metrics(actian_backend: ActianBackend) -> None:
    actian_backend.ensure_initial_artifacts()

    results, metadata = actian_backend.run_comparison(
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
