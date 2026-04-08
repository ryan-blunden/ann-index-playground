from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import faiss
import h5py
import numpy as np

try:
    import psutil as psutil_module
except ImportError:
    psutil_module = None


REQUIRED_DATASET_KEYS = ("train", "test", "neighbors")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def current_rss_mb() -> float | None:
    if psutil_module is None:
        return None
    try:
        return psutil_module.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except (OSError, AttributeError):
        return None


def load_sift_hdf5(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with h5py.File(dataset_path, "r") as handle:
            missing_keys = [key for key in REQUIRED_DATASET_KEYS if key not in handle]
            if missing_keys:
                raise ValueError(f"Dataset is missing required keys: {', '.join(missing_keys)}")

            xb = np.asarray(handle["train"], dtype="float32")
            xq = np.asarray(handle["test"], dtype="float32")
            gt = np.asarray(handle["neighbors"], dtype=np.int64)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to open dataset {dataset_path}. The file appears truncated or corrupt. Re-download the ANN-Benchmarks SIFT1M HDF5 file."
        ) from exc

    if xb.ndim != 2 or xq.ndim != 2 or gt.ndim != 2:
        raise ValueError("Expected 2D arrays for train, test, and neighbors.")
    if xb.shape[1] != xq.shape[1]:
        raise ValueError("Train and test vectors must have the same dimensionality.")

    return xb, xq, gt


def overlap_recall_at_k(pred_ids: np.ndarray, reference_ids: np.ndarray, k: int) -> float:
    correct = 0
    for pred_row, ref_row in zip(pred_ids[:, :k], reference_ids[:, :k], strict=False):
        correct += len(set(pred_row.tolist()) & set(ref_row.tolist()))
    return correct / (pred_ids.shape[0] * k)


def warmup_index(index: faiss.Index, queries: np.ndarray, k: int) -> None:
    warmup_count = min(100, len(queries))
    index.search(queries[:warmup_count], k)


def measure_batch_latency(index: faiss.Index, queries: np.ndarray, k: int, repeats: int) -> tuple[list[float], np.ndarray]:
    batch_run_seconds: list[float] = []
    last_ids: np.ndarray | None = None

    for _ in range(repeats):
        start = time.perf_counter()
        _, ids = index.search(queries, k)
        batch_run_seconds.append(time.perf_counter() - start)
        last_ids = ids

    assert last_ids is not None
    return batch_run_seconds, last_ids


def measure_single_query_latencies(index: faiss.Index, queries: np.ndarray, k: int, sample_size: int) -> list[float]:
    sampled_queries = queries[: min(sample_size, len(queries))]
    latencies_ms: list[float] = []
    for query in sampled_queries:
        single_query = np.ascontiguousarray(query.reshape(1, -1))
        start = time.perf_counter()
        index.search(single_query, k)
        latencies_ms.append((time.perf_counter() - start) * 1000)
    return latencies_ms


def evaluate_index(
    index: faiss.Index,
    queries: np.ndarray,
    reference_ids: np.ndarray,
    *,
    k: int,
    repeats: int,
    latency_sample_size: int,
) -> dict[str, Any]:
    warmup_index(index, queries, k)
    batch_run_seconds, last_ids = measure_batch_latency(index, queries, k, repeats)
    single_query_latencies_ms = measure_single_query_latencies(index, queries, k, latency_sample_size)
    avg_latency_ms = float(np.mean([(seconds / len(queries)) * 1000 for seconds in batch_run_seconds]))

    return {
        "ids": last_ids,
        "avg_latency_ms": avg_latency_ms,
        "p50_latency_ms": percentile(single_query_latencies_ms, 50),
        "p95_latency_ms": percentile(single_query_latencies_ms, 95),
        "qps_avg": len(queries) / float(np.mean(batch_run_seconds)),
        "recall_at_10": overlap_recall_at_k(last_ids, reference_ids, k),
        "latency_sample_size": len(single_query_latencies_ms),
    }


def build_flat(xb: np.ndarray, dimension: int) -> tuple[faiss.IndexFlatL2, float, float | None]:
    rss_before = current_rss_mb()
    start = time.perf_counter()
    index = faiss.IndexFlatL2(dimension)
    index.add(xb)  # pylint: disable=no-value-for-parameter
    build_time_s = time.perf_counter() - start
    rss_after = current_rss_mb()
    memory_delta_mb = None if rss_before is None or rss_after is None else rss_after - rss_before
    return index, build_time_s, memory_delta_mb


def build_hnsw(xb: np.ndarray, dimension: int, m: int, ef_construction: int) -> tuple[faiss.IndexHNSWFlat, float, float | None]:
    rss_before = current_rss_mb()
    start = time.perf_counter()
    index = faiss.IndexHNSWFlat(dimension, m)
    index.hnsw.efConstruction = ef_construction
    index.add(xb)  # pylint: disable=no-value-for-parameter
    build_time_s = time.perf_counter() - start
    rss_after = current_rss_mb()
    memory_delta_mb = None if rss_before is None or rss_after is None else rss_after - rss_before
    return index, build_time_s, memory_delta_mb


def build_ivf(xb: np.ndarray, dimension: int, nlist: int) -> tuple[faiss.IndexIVFFlat, float, float | None]:
    rss_before = current_rss_mb()
    start = time.perf_counter()
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index.train(xb)  # pylint: disable=no-value-for-parameter
    index.add(xb)  # pylint: disable=no-value-for-parameter
    build_time_s = time.perf_counter() - start
    rss_after = current_rss_mb()
    memory_delta_mb = None if rss_before is None or rss_after is None else rss_after - rss_before
    return index, build_time_s, memory_delta_mb
