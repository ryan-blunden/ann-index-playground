from __future__ import annotations

# pylint: disable=redefined-outer-name
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from ann_backend import BackendSettings, RunConfig
from ann_faiss import FaissBackend


def write_synthetic_dataset(path: Path, *, train_size: int = 512, query_size: int = 64, dimension: int = 16, k: int = 10) -> None:
    rng = np.random.default_rng(1234)
    train = rng.random((train_size, dimension), dtype=np.float32)
    test = rng.random((query_size, dimension), dtype=np.float32)

    distances = np.sum((test[:, None, :] - train[None, :, :]) ** 2, axis=2)
    neighbors = np.argsort(distances, axis=1)[:, :k].astype(np.int64)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=train)
        handle.create_dataset("test", data=test)
        handle.create_dataset("neighbors", data=neighbors)


@pytest.fixture
def faiss_backend(tmp_path: Path) -> FaissBackend:
    dataset_path = tmp_path / "tiny-sift.hdf5"
    write_synthetic_dataset(dataset_path)
    settings = BackendSettings(
        dataset_path=dataset_path,
        cache_dir=tmp_path / "cache",
        vector_count=512,
        include_hnsw=True,
        include_ivf=True,
        default_hnsw_m=16,
        default_hnsw_ef_construction=80,
        default_ivf_nlist=256,
    )
    return FaissBackend(settings)


def test_initial_artifacts_are_built_with_progress(faiss_backend: FaissBackend) -> None:
    progress_messages: list[str] = []

    assert faiss_backend.initial_artifacts_exist() is False

    faiss_backend.ensure_initial_artifacts(lambda update: progress_messages.append(update.message))

    assert faiss_backend.initial_artifacts_exist() is True
    assert progress_messages
    assert any("Flat" in message for message in progress_messages)
    assert any("HNSW" in message for message in progress_messages)
    assert any("IVF" in message for message in progress_messages)


def test_cache_summaries_report_built_metadata(faiss_backend: FaissBackend) -> None:
    assert faiss_backend.hnsw_summary(16, 80) == "Cache not built yet."
    assert faiss_backend.ivf_summary(256) == "Cache not built yet."

    faiss_backend.ensure_initial_artifacts()

    assert "Build time:" in faiss_backend.hnsw_summary(16, 80)
    assert "Index size:" in faiss_backend.hnsw_summary(16, 80)
    assert "Build time:" in faiss_backend.ivf_summary(256)
    assert "Index size:" in faiss_backend.ivf_summary(256)


def test_run_comparison_returns_expected_rows_and_metrics(faiss_backend: FaissBackend) -> None:
    faiss_backend.ensure_initial_artifacts()

    results, metadata = faiss_backend.run_comparison(
        RunConfig(
            query_count=32,
            k=10,
            repeats=3,
            latency_sample_size=16,
            hnsw_m=16,
            hnsw_ef_construction=80,
            hnsw_ef_search=32,
            ivf_nlist=256,
            ivf_nprobe=8,
        )
    )

    assert isinstance(results, pd.DataFrame)
    assert list(results["family"]) == ["Flat", "HNSW", "IVF"]
    assert metadata["base_vectors"] == 512
    assert metadata["query_count"] == 32
    assert (results["avg_latency_ms"] > 0).all()
    assert (results["p95_latency_ms"] > 0).all()
    assert results.loc[results["family"] == "Flat", "recall_at_10"].iloc[0] == pytest.approx(1.0)
    assert (results["speedup_vs_flat"] > 0).all()
