from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from actian_vectorai.proto import actian_vectorai_points_pb2 as points_pb

from actian_vectorai import (
    ChannelClosedError,
    CollectionNotFoundError,
    Distance,
    HnswConfigDiff,
    IndexType,
    IvfConfigDiff,
    PointStruct,
    SearchParams,
    VectorAIClient,
    VectorAIConnectionError,
    VectorAIError,
    VectorParams,
)

from ann_backend import BackendSettings, ProgressCallback, ProgressUpdate, RunConfig, default_ivf_nprobe, recommended_ivf_nlist_options
from ann_faiss import load_sift_hdf5, overlap_recall_at_k, percentile


class ActianBackend:
    name = "actian"
    client_timeout_s = 180.0

    def __init__(self, settings: BackendSettings) -> None:
        # The current Actian release used by this app is HNSW-only.
        self.settings = replace(settings, include_ivf=False)
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.settings.service_url:
            raise ValueError("service_url is required for the Actian backend.")
        self.service_url = self.settings.service_url

    @cached_property
    def dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return load_sift_hdf5(self.settings.dataset_path)

    @cached_property
    def dimension(self) -> int:
        xb, _, _ = self.dataset
        return int(xb.shape[1])

    @cached_property
    def dataset_prefix(self) -> str:
        stat = self.settings.dataset_path.stat()
        digest = hashlib.md5(
            f"{self.settings.dataset_path.stem}:{stat.st_size}:{int(stat.st_mtime)}:{self.settings.vector_count}".encode()
        ).hexdigest()[:10]
        return f"ann_{self.settings.dataset_path.stem.replace('-', '_')[:12]}_{self.settings.vector_count}_{digest}"

    def available_nlist_options(self) -> list[int]:
        return recommended_ivf_nlist_options(self.settings.vector_count, max_option=1024)

    def default_ivf_nprobe(self, nlist: int) -> int:
        return default_ivf_nprobe(nlist)

    def initial_artifacts_exist(self) -> bool:
        try:
            with self._client() as client:
                if not self._collection_ready(client, self._flat_collection_name()):
                    return False
                if self.settings.include_hnsw and not self._collection_ready(
                    client,
                    self._hnsw_collection_name(self.settings.default_hnsw_m, self.settings.default_hnsw_ef_construction),
                ):
                    return False
                if self.settings.include_ivf and not self._collection_ready(
                    client,
                    self._ivf_collection_name(self.settings.default_ivf_nlist, self.default_ivf_nprobe(self.settings.default_ivf_nlist)),
                ):
                    return False
        except Exception:
            return False

        return all(path.exists() for path in self._initial_metadata_paths())

    def ensure_initial_artifacts(self, progress: ProgressCallback | None = None) -> None:
        total_steps = 2 + int(self.settings.include_hnsw) + int(self.settings.include_ivf)
        current_step = 0

        def notify(message: str) -> None:
            nonlocal current_step
            current_step += 1
            if progress is not None:
                progress(ProgressUpdate(current_step=current_step, total_steps=total_steps, message=message))

        notify("Connecting to Actian VectorAI DB...")
        with self._client() as client:
            client.health_check()

            notify(f"Loading {self.settings.vector_count:,} vectors into Flat collection...")
            self._ensure_flat_collection(client)

            if self.settings.include_hnsw:
                notify("Building HNSW collection...")
                self._ensure_hnsw_collection(client, self.settings.default_hnsw_m, self.settings.default_hnsw_ef_construction)

            if self.settings.include_ivf:
                notify("Building IVF collection...")
                self._ensure_ivf_collection(
                    client,
                    self.settings.default_ivf_nlist,
                    self.default_ivf_nprobe(self.settings.default_ivf_nlist),
                )

    def hnsw_summary(self, m: int, ef_construction: int) -> str:
        metadata_path = self._metadata_path(f"hnsw_m{m}_efc{ef_construction}")
        metadata = self._refresh_metadata_size(metadata_path, self._hnsw_collection_name(m, ef_construction))
        return self._format_cache_summary(metadata)

    def ivf_summary(self, nlist: int, nprobe: int | None = None) -> str:
        effective_nprobe = self.default_ivf_nprobe(nlist) if nprobe is None else nprobe
        metadata_path = self._metadata_path(f"ivf_nlist{nlist}_np{effective_nprobe}")
        metadata = self._refresh_metadata_size(metadata_path, self._ivf_collection_name(nlist, effective_nprobe))
        return self._format_cache_summary(metadata)

    def flat_summary(self) -> str:
        return ""

    def run_comparison(self, config: RunConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
        with self._client() as client:
            flat_metadata = self._ensure_flat_collection(client)

            xq_full = self.dataset[1]
            queries = np.ascontiguousarray(xq_full[: config.query_count], dtype=np.float32)

            flat_search = self._search_collection(
                client,
                collection_name=self._flat_collection_name(),
                queries=queries,
                k=config.k,
                repeats=config.repeats,
                latency_sample_size=config.latency_sample_size,
            )
            exact_reference_ids = flat_search["ids"]

            rows = [
                {
                    "family": "Flat",
                    "config_text": f"Exact baseline · queries={config.query_count}",
                    "avg_latency_ms": flat_search["avg_latency_ms"],
                    "p50_latency_ms": flat_search["p50_latency_ms"],
                    "p95_latency_ms": flat_search["p95_latency_ms"],
                    "recall_at_10": 1.0,
                    "build_time_s": flat_metadata["build_time_s"],
                    "index_size_mb": flat_metadata["index_size_mb"],
                    "speedup_vs_flat": 1.0,
                }
            ]

            if self.settings.include_hnsw:
                assert config.hnsw_m is not None
                assert config.hnsw_ef_construction is not None
                assert config.hnsw_ef_search is not None
                hnsw_metadata = self._ensure_hnsw_collection(client, config.hnsw_m, config.hnsw_ef_construction)
                hnsw_search = self._search_collection(
                    client,
                    collection_name=self._hnsw_collection_name(config.hnsw_m, config.hnsw_ef_construction),
                    queries=queries,
                    k=config.k,
                    repeats=config.repeats,
                    latency_sample_size=config.latency_sample_size,
                    search_params=SearchParams(hnsw_ef=config.hnsw_ef_search),
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
                        "recall_at_10": overlap_recall_at_k(hnsw_search["ids"], exact_reference_ids, config.k),
                        "build_time_s": hnsw_metadata["build_time_s"],
                        "index_size_mb": hnsw_metadata["index_size_mb"],
                        "speedup_vs_flat": rows[0]["avg_latency_ms"] / hnsw_search["avg_latency_ms"],
                    }
                )

            if self.settings.include_ivf:
                assert config.ivf_nlist is not None
                assert config.ivf_nprobe is not None
                ivf_metadata = self._ensure_ivf_collection(client, config.ivf_nlist, config.ivf_nprobe)
                # IVF collection rebuild/open happens on a fresh client during preparation.
                # Search with a fresh client too, otherwise the first query against a newly
                # built IVF variant can observe a stale view and return empty results.
                with self._client() as ivf_client:
                    ivf_search = self._search_collection(
                        ivf_client,
                        collection_name=self._ivf_collection_name(config.ivf_nlist, config.ivf_nprobe),
                        queries=queries,
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
                        "recall_at_10": overlap_recall_at_k(ivf_search["ids"], exact_reference_ids, config.k),
                        "build_time_s": ivf_metadata["build_time_s"],
                        "index_size_mb": ivf_metadata["index_size_mb"],
                        "speedup_vs_flat": rows[0]["avg_latency_ms"] / ivf_search["avg_latency_ms"],
                    }
                )

            metadata = {
                "base_vectors": self.settings.vector_count,
                "query_count": config.query_count,
                "k": config.k,
                "repeats": config.repeats,
                "latency_sample_size": config.latency_sample_size,
            }
            return pd.DataFrame(rows), metadata

    def _client(self) -> VectorAIClient:
        return VectorAIClient(self.service_url, timeout=self.client_timeout_s)

    def _collection_ready(self, client: VectorAIClient, collection_name: str) -> bool:
        if not client.collections.exists(collection_name):
            return False
        return client.vde.get_vector_count(collection_name) == self.settings.vector_count

    def _ensure_flat_collection(self, client: VectorAIClient) -> dict[str, float]:
        collection_name = self._flat_collection_name()
        metadata_path = self._metadata_path("flat")
        if self._collection_ready(client, collection_name) and metadata_path.exists():
            return self._read_metadata(metadata_path)

        start = time.perf_counter()
        self._recreate_collection(
            client,
            collection_name=collection_name,
            index_type=IndexType.INDEX_TYPE_FLAT,
        )
        self._upload_dataset(
            collection_name,
            recreate_collection=lambda recreate_client: self._recreate_collection(
                recreate_client,
                collection_name=collection_name,
                index_type=IndexType.INDEX_TYPE_FLAT,
            ),
        )
        build_time_s = time.perf_counter() - start
        with self._client() as post_client:
            metadata = self._write_metadata(metadata_path, build_time_s, self._collection_size_mb(post_client, collection_name))
        return metadata

    def _ensure_hnsw_collection(self, client: VectorAIClient, m: int, ef_construction: int) -> dict[str, float]:
        collection_name = self._hnsw_collection_name(m, ef_construction)
        metadata_path = self._metadata_path(f"hnsw_m{m}_efc{ef_construction}")
        if self._collection_ready(client, collection_name) and metadata_path.exists():
            return self._read_metadata(metadata_path)

        start = time.perf_counter()
        self._recreate_collection(
            client,
            collection_name=collection_name,
            index_type=IndexType.INDEX_TYPE_HNSW,
            hnsw_config=HnswConfigDiff(m=m, ef_construct=ef_construction),
        )
        self._upload_dataset(
            collection_name,
            recreate_collection=lambda recreate_client: self._recreate_collection(
                recreate_client,
                collection_name=collection_name,
                index_type=IndexType.INDEX_TYPE_HNSW,
                hnsw_config=HnswConfigDiff(m=m, ef_construct=ef_construction),
            ),
        )
        build_time_s = time.perf_counter() - start
        with self._client() as post_client:
            metadata = self._write_metadata(metadata_path, build_time_s, self._collection_size_mb(post_client, collection_name))
        return metadata

    def _ensure_ivf_collection(self, client: VectorAIClient, nlist: int, nprobe: int) -> dict[str, float]:
        collection_name = self._ivf_collection_name(nlist, nprobe)
        metadata_path = self._metadata_path(f"ivf_nlist{nlist}_np{nprobe}")
        if self._collection_ready(client, collection_name) and metadata_path.exists():
            return self._read_metadata(metadata_path)

        start = time.perf_counter()
        self._recreate_collection(
            client,
            collection_name=collection_name,
            index_type=IndexType.INDEX_TYPE_IVF_FLAT,
            ivf_config=IvfConfigDiff(
                nlist=nlist,
                nprobe=nprobe,
                training_sample_size=min(self.settings.vector_count, max(10_000, nlist * 64)),
            ),
        )
        self._upload_dataset(
            collection_name,
            recreate_collection=lambda recreate_client: self._recreate_collection(
                recreate_client,
                collection_name=collection_name,
                index_type=IndexType.INDEX_TYPE_IVF_FLAT,
                ivf_config=IvfConfigDiff(
                    nlist=nlist,
                    nprobe=nprobe,
                    training_sample_size=min(self.settings.vector_count, max(10_000, nlist * 64)),
                ),
            ),
        )
        with self._client() as post_client:
            # On the validated Actian VectorAI DB 1.0.0 image, IVF collections did not
            # reliably return results until the index was explicitly rebuilt and reopened.
            post_client.vde.rebuild_index(collection_name)
            post_client.vde.open_collection(collection_name)
        self._wait_for_collection_searchable(collection_name, reference_collection=self._flat_collection_name())
        build_time_s = time.perf_counter() - start
        with self._client() as post_client:
            metadata = self._write_metadata(metadata_path, build_time_s, self._collection_size_mb(post_client, collection_name))
        return metadata

    def _recreate_collection(
        self,
        client: VectorAIClient,
        *,
        collection_name: str,
        index_type: IndexType,
        hnsw_config: HnswConfigDiff | None = None,
        ivf_config: IvfConfigDiff | None = None,
    ) -> None:
        def perform(recreate_client: VectorAIClient) -> None:
            if recreate_client.collections.exists(collection_name):
                recreate_client.collections.delete(collection_name)
            recreate_client.collections.create(
                collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.Euclid),
                index_type=index_type,
                hnsw_config=hnsw_config,
                ivf_config=ivf_config,
            )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt == 1:
                    perform(client)
                else:
                    with self._client() as retry_client:
                        perform(retry_client)
                return
            except (ChannelClosedError, VectorAIConnectionError, VectorAIError) as exc:
                if isinstance(exc, VectorAIError) and exc.code not in {408, 500, 503}:
                    raise
                if attempt == max_attempts:
                    raise
                self._wait_for_service(timeout_s=30.0)
                time.sleep(min(5.0, 1.0 * attempt))
        raise RuntimeError(f"Actian collection recreation failed for {collection_name!r}.")

    def _upload_dataset(self, collection_name: str, recreate_collection: Any) -> None:
        xb = np.ascontiguousarray(self.dataset[0][: self.settings.vector_count], dtype=np.float32)
        batch_size = 1_024
        max_upload_attempts = 3
        for upload_attempt in range(1, max_upload_attempts + 1):
            restart_required = False
            for start_index in range(0, len(xb), batch_size):
                batch = [
                    PointStruct(id=start_index + offset, vector=vector.tolist())
                    for offset, vector in enumerate(xb[start_index : start_index + batch_size], start=0)
                ]
                try:
                    self._upsert_batch_with_retry(collection_name, batch, start_index)
                except CollectionNotFoundError:
                    if upload_attempt == max_upload_attempts:
                        raise
                    self._wait_for_service()
                    with self._client() as recreate_client:
                        recreate_collection(recreate_client)
                    restart_required = True
                    break
            if not restart_required:
                return

        raise RuntimeError(f"Actian dataset upload failed for collection {collection_name!r}.")

    def _upsert_batch_with_retry(self, collection_name: str, batch: list[PointStruct], start_index: int) -> None:
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                with self._client() as client:
                    req = points_pb.UpsertPoints(collection_name=collection_name)
                    req.points.extend(point.to_proto() for point in batch)

                    async def perform_upsert() -> Any:
                        return await client._async_client.points._stub.Upsert(req)

                    client._loop.run(perform_upsert())
                return
            except CollectionNotFoundError:
                raise
            except (ChannelClosedError, VectorAIConnectionError, VectorAIError) as exc:
                if isinstance(exc, VectorAIError) and exc.code not in {500, 503}:
                    raise
                if attempt == max_attempts:
                    raise
                self._wait_for_service()
                time.sleep(min(5.0, 0.75 * attempt))
        raise RuntimeError(f"Actian batch upsert failed unexpectedly at offset {start_index}.")

    def _wait_for_service(self, timeout_s: float = 20.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with self._client() as client:
                    client.health_check()
                return
            except (ChannelClosedError, VectorAIConnectionError, VectorAIError):
                time.sleep(0.5)
        raise RuntimeError("Actian VectorAI DB did not become healthy again after a transient disconnect.")

    def _wait_for_collection_searchable(
        self,
        collection_name: str,
        reference_collection: str | None = None,
        *,
        timeout_s: float = 45.0,
        k: int = 10,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        probe_queries = np.ascontiguousarray(self.dataset[1][:3], dtype=np.float32)
        while time.monotonic() < deadline:
            try:
                with self._client() as client:
                    if reference_collection is None:
                        results = client.points.search(collection_name, vector=probe_queries[0].tolist(), limit=1)
                        if results:
                            return
                    else:
                        total_overlap = 0
                        for query in probe_queries:
                            reference = client.points.search(reference_collection, vector=query.tolist(), limit=k)
                            candidate = client.points.search(collection_name, vector=query.tolist(), limit=k)
                            reference_ids = {int(result.id) for result in reference}
                            candidate_ids = {int(result.id) for result in candidate}
                            total_overlap += len(reference_ids & candidate_ids)
                        if total_overlap > 0:
                            return
            except (ChannelClosedError, VectorAIConnectionError, VectorAIError):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Actian collection {collection_name!r} did not become searchable after rebuild.")

    def _collection_size_mb(self, client: VectorAIClient, collection_name: str) -> float:
        stats = client.vde.get_stats(collection_name)
        total_bytes = int(stats.storage_bytes) + int(stats.index_memory_bytes)
        if total_bytes > 0:
            return total_bytes / (1024 * 1024)

        if self.settings.service_data_dir is not None:
            collection_dir = self.settings.service_data_dir / collection_name
            if collection_dir.exists():
                directory_bytes = sum(path.stat().st_size for path in collection_dir.rglob("*") if path.is_file())
                if directory_bytes > 0:
                    return directory_bytes / (1024 * 1024)

        return 0.0

    def _search_collection(
        self,
        client: VectorAIClient,
        *,
        collection_name: str,
        queries: np.ndarray,
        k: int,
        repeats: int,
        latency_sample_size: int,
        search_params: SearchParams | None = None,
    ) -> dict[str, Any]:
        self._warmup_queries(client, collection_name, queries, k, search_params)

        repeat_avg_ms: list[float] = []
        sample_latencies_ms: list[float] = []
        last_ids: np.ndarray | None = None

        for repeat in range(repeats):
            latencies_ms: list[float] = []
            ids_for_repeat: list[list[int]] = []
            for query_index, query in enumerate(queries):
                start = time.perf_counter()
                results = client.points.search(collection_name, vector=query.tolist(), limit=k, params=search_params)
                latency_ms = (time.perf_counter() - start) * 1000
                latencies_ms.append(latency_ms)

                ids = [int(result.id) for result in results]
                ids.extend([-1] * max(0, k - len(ids)))
                ids_for_repeat.append(ids[:k])

                if repeat == repeats - 1 and query_index < min(latency_sample_size, len(queries)):
                    sample_latencies_ms.append(latency_ms)

            repeat_avg_ms.append(float(np.mean(latencies_ms)))
            last_ids = np.asarray(ids_for_repeat, dtype=np.int64)

        assert last_ids is not None
        return {
            "ids": last_ids,
            "avg_latency_ms": float(np.mean(repeat_avg_ms)),
            "p50_latency_ms": percentile(sample_latencies_ms, 50),
            "p95_latency_ms": percentile(sample_latencies_ms, 95),
        }

    def _warmup_queries(
        self,
        client: VectorAIClient,
        collection_name: str,
        queries: np.ndarray,
        k: int,
        search_params: SearchParams | None,
    ) -> None:
        for query in queries[: min(10, len(queries))]:
            client.points.search(collection_name, vector=query.tolist(), limit=k, params=search_params)

    def _initial_metadata_paths(self) -> list[Path]:
        paths = [self._metadata_path("flat")]
        if self.settings.include_hnsw:
            paths.append(self._metadata_path(f"hnsw_m{self.settings.default_hnsw_m}_efc{self.settings.default_hnsw_ef_construction}"))
        if self.settings.include_ivf:
            default_nprobe = self.default_ivf_nprobe(self.settings.default_ivf_nlist)
            paths.append(self._metadata_path(f"ivf_nlist{self.settings.default_ivf_nlist}_np{default_nprobe}"))
        return paths

    def _metadata_path(self, suffix: str) -> Path:
        return self.settings.cache_dir / f"{self.dataset_prefix}_{suffix}.json"

    def _write_metadata(self, path: Path, build_time_s: float, index_size_mb: float) -> dict[str, float]:
        payload = {"build_time_s": float(build_time_s), "index_size_mb": float(index_size_mb)}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _read_metadata(self, path: Path) -> dict[str, float]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _refresh_metadata_size(self, path: Path, collection_name: str) -> dict[str, float] | None:
        if not path.exists():
            return None

        metadata = self._read_metadata(path)
        if metadata["index_size_mb"] > 0:
            return metadata

        if self.settings.service_data_dir is not None:
            collection_dir = self.settings.service_data_dir / collection_name
            if collection_dir.exists():
                directory_bytes = sum(file_path.stat().st_size for file_path in collection_dir.rglob("*") if file_path.is_file())
                if directory_bytes > 0:
                    metadata["index_size_mb"] = directory_bytes / (1024 * 1024)
                    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def _format_cache_summary(self, metadata: dict[str, float] | None) -> str:
        if metadata is None:
            return "Cache not built yet."
        return f"Prep time: {metadata['build_time_s']:.1f} s"

    def _flat_collection_name(self) -> str:
        return f"{self.dataset_prefix}_flat"

    def _hnsw_collection_name(self, m: int, ef_construction: int) -> str:
        return f"{self.dataset_prefix}_hnsw_m{m}_efc{ef_construction}"

    def _ivf_collection_name(self, nlist: int, nprobe: int) -> str:
        # The current validated server image appears to ignore query-time ivf_nprobe
        # overrides, so Actian IVF variants are materialized as separate collections.
        return f"{self.dataset_prefix}_ivf_nlist{nlist}_np{nprobe}"
