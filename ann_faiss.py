from __future__ import annotations

import json
import os
import time
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import faiss
import h5py
import numpy as np
import pandas as pd

from ann_backend import BackendSettings, ProgressCallback, ProgressUpdate, RunConfig

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
    except OSError, AttributeError:
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
    index.search(queries[:warmup_count], k)  # pyright: ignore[reportCallIssue]


def measure_batch_latency(index: faiss.Index, queries: np.ndarray, k: int, repeats: int) -> tuple[list[float], np.ndarray]:
    batch_run_seconds: list[float] = []
    last_ids: np.ndarray | None = None

    for _ in range(repeats):
        start = time.perf_counter()
        _, ids = index.search(queries, k)  # pyright: ignore[reportCallIssue]
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
        index.search(single_query, k)  # pyright: ignore[reportCallIssue]
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
    index.add(xb)  # pylint: disable=no-value-for-parameter  # pyright: ignore[reportCallIssue]
    build_time_s = time.perf_counter() - start
    rss_after = current_rss_mb()
    memory_delta_mb = None if rss_before is None or rss_after is None else rss_after - rss_before
    return index, build_time_s, memory_delta_mb


def build_hnsw(xb: np.ndarray, dimension: int, m: int, ef_construction: int) -> tuple[faiss.IndexHNSWFlat, float, float | None]:
    rss_before = current_rss_mb()
    start = time.perf_counter()
    index = faiss.IndexHNSWFlat(dimension, m)
    hnsw_index = cast(faiss.IndexHNSWFlat, index)
    hnsw_index.hnsw.efConstruction = ef_construction  # pyright: ignore[reportAttributeAccessIssue]
    hnsw_index.add(xb)  # pylint: disable=no-value-for-parameter  # pyright: ignore[reportCallIssue]
    build_time_s = time.perf_counter() - start
    rss_after = current_rss_mb()
    memory_delta_mb = None if rss_before is None or rss_after is None else rss_after - rss_before
    return hnsw_index, build_time_s, memory_delta_mb


def build_ivf(xb: np.ndarray, dimension: int, nlist: int) -> tuple[faiss.IndexIVFFlat, float, float | None]:
    rss_before = current_rss_mb()
    start = time.perf_counter()
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index.train(xb)  # pylint: disable=no-value-for-parameter  # pyright: ignore[reportCallIssue]
    index.add(xb)  # pylint: disable=no-value-for-parameter  # pyright: ignore[reportCallIssue]
    build_time_s = time.perf_counter() - start
    rss_after = current_rss_mb()
    memory_delta_mb = None if rss_before is None or rss_after is None else rss_after - rss_before
    return index, build_time_s, memory_delta_mb


class FaissBackend:
    name = "faiss"

    def __init__(self, settings: BackendSettings) -> None:
        self.settings = settings
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return load_sift_hdf5(self.settings.dataset_path)

    @cached_property
    def dataset_fingerprint(self) -> str:
        stat = self.settings.dataset_path.stat()
        return f"{self.settings.dataset_path.stem}_{stat.st_size}_{int(stat.st_mtime)}"

    def available_nlist_options(self) -> list[int]:
        return [value for value in [256, 512, 1024, 2048, 4096, 8192] if value <= self.settings.vector_count]

    def initial_artifacts_exist(self) -> bool:
        return all(path.exists() for path in self._initial_artifact_paths())

    def ensure_initial_artifacts(self, progress: ProgressCallback | None = None) -> None:
        total_steps = 2 + int(self.settings.include_hnsw) + int(self.settings.include_ivf)
        current_step = 0

        def notify(message: str) -> None:
            nonlocal current_step
            current_step += 1
            if progress is not None:
                progress(ProgressUpdate(current_step=current_step, total_steps=total_steps, message=message))

        notify(f"Loading {self.settings.vector_count:,} vectors from SIFT1M...")
        xb_full, _, _ = self.dataset
        xb = np.ascontiguousarray(xb_full[: self.settings.vector_count])
        dimension = xb.shape[1]

        flat_loaded = self._flat_path().exists()
        notify(self._cache_step_label("Flat", flat_loaded))
        self._load_or_build_flat(xb, dimension)

        if self.settings.include_hnsw:
            hnsw_loaded = self._hnsw_path(self.settings.default_hnsw_m, self.settings.default_hnsw_ef_construction).exists()
            notify(self._cache_step_label("HNSW", hnsw_loaded))
            self._load_or_build_hnsw(xb, dimension, self.settings.default_hnsw_m, self.settings.default_hnsw_ef_construction)

        if self.settings.include_ivf:
            ivf_loaded = self._ivf_path(self.settings.default_ivf_nlist).exists()
            notify(self._cache_step_label("IVF", ivf_loaded))
            self._load_or_build_ivf(xb, dimension, self.settings.default_ivf_nlist)

    def hnsw_summary(self, m: int, ef_construction: int) -> str:
        return self._format_cache_summary(self._hnsw_path(m, ef_construction))

    def ivf_summary(self, nlist: int) -> str:
        return self._format_cache_summary(self._ivf_path(nlist))

    def run_comparison(self, config: RunConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
        xb_full, xq_full, _ = self.dataset
        xb = np.ascontiguousarray(xb_full[: self.settings.vector_count])
        xq = np.ascontiguousarray(xq_full[: config.query_count])
        dimension = xb.shape[1]

        placeholder_reference = np.zeros((len(xq), config.k), dtype=np.int64)

        flat_index, flat_build_s, flat_size_mb, _ = self._load_or_build_flat(xb, dimension)
        flat_search = evaluate_index(
            flat_index,
            xq,
            placeholder_reference,
            k=config.k,
            repeats=config.repeats,
            latency_sample_size=config.latency_sample_size,
        )
        exact_reference_ids = flat_search["ids"]

        flat_row = {
            "family": "Flat",
            "config_text": f"Exact baseline · queries={config.query_count}",
            "avg_latency_ms": flat_search["avg_latency_ms"],
            "p50_latency_ms": flat_search["p50_latency_ms"],
            "p95_latency_ms": flat_search["p95_latency_ms"],
            "recall_at_10": 1.0,
            "build_time_s": flat_build_s,
            "index_size_mb": flat_size_mb,
            "speedup_vs_flat": 1.0,
        }

        rows = [flat_row]

        if self.settings.include_hnsw:
            assert config.hnsw_m is not None
            assert config.hnsw_ef_construction is not None
            assert config.hnsw_ef_search is not None
            hnsw_index, hnsw_build_s, hnsw_size_mb, _ = self._load_or_build_hnsw(xb, dimension, config.hnsw_m, config.hnsw_ef_construction)
            hnsw_index.hnsw.efSearch = config.hnsw_ef_search  # pyright: ignore[reportAttributeAccessIssue]
            hnsw_search = evaluate_index(
                hnsw_index,
                xq,
                exact_reference_ids,
                k=config.k,
                repeats=config.repeats,
                latency_sample_size=config.latency_sample_size,
            )
            rows.append(
                {
                    "family": "HNSW",
                    "config_text": (
                        f"M={config.hnsw_m} · efC={config.hnsw_ef_construction} · " f"efS={config.hnsw_ef_search} · queries={config.query_count}"
                    ),
                    "avg_latency_ms": hnsw_search["avg_latency_ms"],
                    "p50_latency_ms": hnsw_search["p50_latency_ms"],
                    "p95_latency_ms": hnsw_search["p95_latency_ms"],
                    "recall_at_10": hnsw_search["recall_at_10"],
                    "build_time_s": hnsw_build_s,
                    "index_size_mb": hnsw_size_mb,
                    "speedup_vs_flat": flat_row["avg_latency_ms"] / hnsw_search["avg_latency_ms"],
                }
            )

        if self.settings.include_ivf:
            assert config.ivf_nlist is not None
            assert config.ivf_nprobe is not None
            ivf_index, ivf_build_s, ivf_size_mb, _ = self._load_or_build_ivf(xb, dimension, config.ivf_nlist)
            ivf_index.nprobe = config.ivf_nprobe  # pyright: ignore[reportAttributeAccessIssue]
            ivf_search = evaluate_index(
                ivf_index,
                xq,
                exact_reference_ids,
                k=config.k,
                repeats=config.repeats,
                latency_sample_size=config.latency_sample_size,
            )
            rows.append(
                {
                    "family": "IVF",
                    "config_text": f"nlist={config.ivf_nlist} · nprobe={config.ivf_nprobe} · queries={config.query_count}",
                    "avg_latency_ms": ivf_search["avg_latency_ms"],
                    "p50_latency_ms": ivf_search["p50_latency_ms"],
                    "p95_latency_ms": ivf_search["p95_latency_ms"],
                    "recall_at_10": ivf_search["recall_at_10"],
                    "build_time_s": ivf_build_s,
                    "index_size_mb": ivf_size_mb,
                    "speedup_vs_flat": flat_row["avg_latency_ms"] / ivf_search["avg_latency_ms"],
                }
            )

        results = pd.DataFrame(rows)
        metadata = {
            "base_vectors": self.settings.vector_count,
            "query_count": config.query_count,
            "k": config.k,
            "repeats": config.repeats,
            "latency_sample_size": config.latency_sample_size,
        }
        return results, metadata

    def _cache_step_label(self, index_name: str, loaded_from_cache: bool) -> str:
        action = "Loading cached" if loaded_from_cache else "Building"
        return f"{action} {index_name} index..."

    def _flat_path(self) -> Path:
        return self._cache_path(f"flat_{self.settings.vector_count}")

    def _hnsw_path(self, m: int, ef_construction: int) -> Path:
        return self._cache_path(f"hnsw_{self.settings.vector_count}_m{m}_efc{ef_construction}")

    def _ivf_path(self, nlist: int) -> Path:
        return self._cache_path(f"ivf_{self.settings.vector_count}_nlist{nlist}")

    def _initial_artifact_paths(self) -> list[Path]:
        paths = [self._flat_path()]
        if self.settings.include_hnsw:
            paths.append(self._hnsw_path(self.settings.default_hnsw_m, self.settings.default_hnsw_ef_construction))
        if self.settings.include_ivf:
            paths.append(self._ivf_path(self.settings.default_ivf_nlist))
        return paths

    def _cache_path(self, name: str) -> Path:
        return self.settings.cache_dir / f"{self.dataset_fingerprint}_{name}.index"

    def _metadata_path(self, index_path: Path) -> Path:
        return index_path.with_suffix(".json")

    def _write_cache_metadata(self, index_path: Path, build_time_s: float) -> dict[str, float]:
        payload = {
            "build_time_s": float(build_time_s),
            "index_size_mb": index_path.stat().st_size / (1024 * 1024),
        }
        self._metadata_path(index_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _read_cache_metadata(self, index_path: Path) -> dict[str, float]:
        path = self._metadata_path(index_path)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        payload = {
            "build_time_s": 0.0,
            "index_size_mb": index_path.stat().st_size / (1024 * 1024),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _format_cache_summary(self, index_path: Path) -> str:
        if not index_path.exists():
            return "Cache not built yet."

        metadata = self._read_cache_metadata(index_path)
        return f"Build time: {metadata['build_time_s']:.1f} s · Index size: {metadata['index_size_mb']:.0f} MB"

    def _load_or_build_flat(self, xb: np.ndarray, dimension: int) -> tuple[faiss.Index, float, float, bool]:
        path = self._flat_path()
        if path.exists():
            metadata = self._read_cache_metadata(path)
            return faiss.read_index(str(path)), metadata["build_time_s"], metadata["index_size_mb"], True

        index, build_time_s, _ = build_flat(xb, dimension)
        faiss.write_index(index, str(path))
        metadata = self._write_cache_metadata(path, build_time_s)
        return index, metadata["build_time_s"], metadata["index_size_mb"], False

    def _load_or_build_hnsw(self, xb: np.ndarray, dimension: int, m: int, ef_construction: int) -> tuple[faiss.Index, float, float, bool]:
        path = self._hnsw_path(m, ef_construction)
        if path.exists():
            metadata = self._read_cache_metadata(path)
            return faiss.read_index(str(path)), metadata["build_time_s"], metadata["index_size_mb"], True

        index, build_time_s, _ = build_hnsw(xb, dimension, m, ef_construction)
        faiss.write_index(index, str(path))
        metadata = self._write_cache_metadata(path, build_time_s)
        return index, metadata["build_time_s"], metadata["index_size_mb"], False

    def _load_or_build_ivf(self, xb: np.ndarray, dimension: int, nlist: int) -> tuple[faiss.Index, float, float, bool]:
        path = self._ivf_path(nlist)
        if path.exists():
            metadata = self._read_cache_metadata(path)
            return faiss.read_index(str(path)), metadata["build_time_s"], metadata["index_size_mb"], True

        index, build_time_s, _ = build_ivf(xb, dimension, nlist)
        faiss.write_index(index, str(path))
        metadata = self._write_cache_metadata(path, build_time_s)
        return index, metadata["build_time_s"], metadata["index_size_mb"], False
