from __future__ import annotations

import hashlib
import time
from functools import cached_property
from typing import Any, cast

import numpy as np
import pandas as pd
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql

from ann_backend import BackendSettings, ProgressCallback, ProgressUpdate, RunConfig
from ann_faiss import load_sift_hdf5, overlap_recall_at_k, percentile


class PgvectorBackend:
    name = "pgvector"

    def __init__(self, settings: BackendSettings) -> None:
        self.settings = settings
        if not self.settings.database_url:
            raise ValueError("database_url is required for the pgvector backend.")
        if not self.settings.admin_database_url:
            raise ValueError("admin_database_url is required for the pgvector backend.")
        self.database_url = self.settings.database_url
        self.admin_database_url = self.settings.admin_database_url

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
        return [value for value in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] if value <= self.settings.vector_count]

    def initial_artifacts_exist(self) -> bool:
        if not self._database_exists():
            return False

        with self._target_connection() as conn:
            if not self._table_exists(conn, self._flat_table_name()):
                return False
            if (
                self.settings.include_hnsw
                and self._read_metadata(conn, self._hnsw_artifact_key(self.settings.default_hnsw_m, self.settings.default_hnsw_ef_construction))
                is None
            ):
                return False
            if self.settings.include_ivf and self._read_metadata(conn, self._ivf_artifact_key(self.settings.default_ivf_nlist)) is None:
                return False
            return True

    def ensure_initial_artifacts(self, progress: ProgressCallback | None = None) -> None:
        total_steps = 2 + int(self.settings.include_hnsw) + int(self.settings.include_ivf)
        current_step = 0

        def notify(message: str) -> None:
            nonlocal current_step
            current_step += 1
            if progress is not None:
                progress(ProgressUpdate(current_step=current_step, total_steps=total_steps, message=message))

        notify("Preparing Postgres database...")
        self._ensure_database()

        with self._target_connection() as conn:
            self._ensure_extension_and_metadata(conn)

            notify(f"Loading {self.settings.vector_count:,} vectors into Postgres...")
            self._ensure_flat_table(conn)

            if self.settings.include_hnsw:
                notify("Building HNSW index...")
                self._ensure_hnsw_index(conn, self.settings.default_hnsw_m, self.settings.default_hnsw_ef_construction)

            if self.settings.include_ivf:
                notify("Building IVF index...")
                self._ensure_ivf_index(conn, self.settings.default_ivf_nlist)

    def hnsw_summary(self, m: int, ef_construction: int) -> str:
        if not self._database_exists():
            return "Cache not built yet."

        with self._target_connection() as conn:
            self._ensure_extension_and_metadata(conn)
            metadata = self._read_metadata(conn, self._hnsw_artifact_key(m, ef_construction))
            if metadata is None:
                return "Cache not built yet."
            return self._format_index_summary(metadata)

    def ivf_summary(self, nlist: int) -> str:
        if not self._database_exists():
            return "Cache not built yet."

        with self._target_connection() as conn:
            self._ensure_extension_and_metadata(conn)
            metadata = self._read_metadata(conn, self._ivf_artifact_key(nlist))
            if metadata is None:
                return "Cache not built yet."
            return self._format_index_summary(metadata)

    def run_comparison(self, config: RunConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
        self._ensure_database()
        with self._target_connection() as conn:
            self._ensure_extension_and_metadata(conn)
            self._ensure_flat_table(conn)

            xq_full = self.dataset[1]
            queries = np.ascontiguousarray(xq_full[: config.query_count], dtype=np.float32)

            flat_search = self._search_table(
                conn,
                table_name=self._flat_table_name(),
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
                    "build_time_s": 0.0,
                    "index_size_mb": 0.0,
                    "speedup_vs_flat": 1.0,
                }
            ]

            if self.settings.include_hnsw:
                assert config.hnsw_m is not None
                assert config.hnsw_ef_construction is not None
                assert config.hnsw_ef_search is not None
                hnsw_metadata = self._ensure_hnsw_index(conn, config.hnsw_m, config.hnsw_ef_construction)
                hnsw_search = self._search_table(
                    conn,
                    table_name=self._hnsw_table_name(),
                    queries=queries,
                    k=config.k,
                    repeats=config.repeats,
                    latency_sample_size=config.latency_sample_size,
                    session_settings={"hnsw.ef_search": config.hnsw_ef_search},
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
                ivf_metadata = self._ensure_ivf_index(conn, config.ivf_nlist)
                ivf_search = self._search_table(
                    conn,
                    table_name=self._ivf_table_name(),
                    queries=queries,
                    k=config.k,
                    repeats=config.repeats,
                    latency_sample_size=config.latency_sample_size,
                    session_settings={"ivfflat.probes": config.ivf_nprobe},
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

    def _database_exists(self) -> bool:
        database_name = self._database_name()
        with psycopg.connect(self.admin_database_url, autocommit=True) as conn:
            result = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,)).fetchone()
            return result is not None

    def _ensure_database(self) -> None:
        database_name = self._database_name()
        with psycopg.connect(self.admin_database_url, autocommit=True) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,)).fetchone()
            if exists is None:
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))

    def _target_connection(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(self.database_url, autocommit=True)
        return conn

    def _ensure_extension_and_metadata(self, conn: psycopg.Connection[Any]) -> None:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("ALTER EXTENSION vector UPDATE")
        register_vector(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ann_index_metadata (
                artifact_key text PRIMARY KEY,
                build_time_s double precision NOT NULL,
                index_size_mb double precision NOT NULL
            )
            """)

    def _ensure_flat_table(self, conn: psycopg.Connection[Any]) -> None:
        table_name = self._flat_table_name()
        conn.execute(
            sql.SQL("CREATE UNLOGGED TABLE IF NOT EXISTS {} (id integer PRIMARY KEY, embedding vector({}))").format(
                sql.Identifier(table_name),
                sql.SQL(cast(Any, str(self.dimension))),
            )
        )

        row = conn.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name))).fetchone()
        assert row is not None
        row_count = int(row[0])
        if row_count == self.settings.vector_count:
            return

        conn.execute(sql.SQL("TRUNCATE {}").format(sql.Identifier(table_name)))
        xb = np.ascontiguousarray(self.dataset[0][: self.settings.vector_count], dtype=np.float32)

        with conn.cursor() as cur:
            copy_sql = sql.SQL("COPY {} (id, embedding) FROM STDIN").format(sql.Identifier(table_name))
            with cur.copy(copy_sql) as copy:
                for row_id, vector in enumerate(xb):
                    copy.write_row((row_id, Vector(vector.tolist())))

        conn.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(table_name)))

    def _ensure_hnsw_index(self, conn: psycopg.Connection[Any], m: int, ef_construction: int) -> dict[str, float]:
        self._ensure_family_table(conn, self._hnsw_table_name())
        artifact_key = self._hnsw_artifact_key(m, ef_construction)
        existing_metadata = self._read_metadata(conn, artifact_key)
        index_name = self._hnsw_index_name(m, ef_construction)
        if self._index_exists(conn, index_name):
            if existing_metadata is not None:
                return existing_metadata
            return self._write_metadata(conn, artifact_key, index_name, 0.0)

        start = time.perf_counter()
        conn.execute(
            sql.SQL("CREATE INDEX {} ON {} USING hnsw (embedding vector_l2_ops) WITH (m = {}, ef_construction = {})").format(
                sql.Identifier(index_name),
                sql.Identifier(self._hnsw_table_name()),
                sql.SQL(cast(Any, str(m))),
                sql.SQL(cast(Any, str(ef_construction))),
            )
        )
        build_time_s = time.perf_counter() - start
        conn.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(self._hnsw_table_name())))
        metadata = self._write_metadata(conn, artifact_key, index_name, build_time_s)
        return metadata

    def _ensure_ivf_index(self, conn: psycopg.Connection[Any], nlist: int) -> dict[str, float]:
        self._ensure_family_table(conn, self._ivf_table_name())
        artifact_key = self._ivf_artifact_key(nlist)
        existing_metadata = self._read_metadata(conn, artifact_key)
        index_name = self._ivf_index_name(nlist)
        if self._index_exists(conn, index_name):
            if existing_metadata is not None:
                return existing_metadata
            return self._write_metadata(conn, artifact_key, index_name, 0.0)

        start = time.perf_counter()
        conn.execute(
            sql.SQL("CREATE INDEX {} ON {} USING ivfflat (embedding vector_l2_ops) WITH (lists = {})").format(
                sql.Identifier(index_name),
                sql.Identifier(self._ivf_table_name()),
                sql.SQL(cast(Any, str(nlist))),
            )
        )
        build_time_s = time.perf_counter() - start
        conn.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(self._ivf_table_name())))
        metadata = self._write_metadata(conn, artifact_key, index_name, build_time_s)
        return metadata

    def _ensure_family_table(self, conn: psycopg.Connection[Any], table_name: str) -> None:
        if self._table_exists(conn, table_name):
            row = conn.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name))).fetchone()
            assert row is not None
            row_count = int(row[0])
            if row_count == self.settings.vector_count:
                return
            conn.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table_name)))

        conn.execute(
            sql.SQL("CREATE UNLOGGED TABLE {} AS TABLE {}").format(
                sql.Identifier(table_name),
                sql.Identifier(self._flat_table_name()),
            )
        )
        conn.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(table_name)))

    def _table_exists(self, conn: psycopg.Connection[Any], table_name: str) -> bool:
        result = conn.execute("SELECT to_regclass(%s)", (table_name,)).fetchone()
        return result is not None and result[0] is not None

    def _index_exists(self, conn: psycopg.Connection[Any], index_name: str) -> bool:
        result = conn.execute("SELECT to_regclass(%s)", (index_name,)).fetchone()
        return result is not None and result[0] is not None

    def _read_metadata(self, conn: psycopg.Connection[Any], artifact_key: str) -> dict[str, float] | None:
        row = conn.execute(
            "SELECT build_time_s, index_size_mb FROM ann_index_metadata WHERE artifact_key = %s",
            (artifact_key,),
        ).fetchone()
        if row is None:
            return None
        return {"build_time_s": float(row[0]), "index_size_mb": float(row[1])}

    def _write_metadata(self, conn: psycopg.Connection[Any], artifact_key: str, index_name: str, build_time_s: float) -> dict[str, float]:
        row = conn.execute("SELECT pg_relation_size(%s::regclass) / (1024.0 * 1024.0)", (index_name,)).fetchone()
        assert row is not None
        index_size_mb = float(row[0])
        conn.execute(
            """
            INSERT INTO ann_index_metadata (artifact_key, build_time_s, index_size_mb)
            VALUES (%s, %s, %s)
            ON CONFLICT (artifact_key) DO UPDATE
            SET build_time_s = EXCLUDED.build_time_s,
                index_size_mb = EXCLUDED.index_size_mb
            """,
            (artifact_key, build_time_s, index_size_mb),
        )
        return {"build_time_s": float(build_time_s), "index_size_mb": index_size_mb}

    def _format_index_summary(self, metadata: dict[str, float]) -> str:
        if metadata["build_time_s"] <= 0:
            return f"Build time: existing index · Index size: {metadata['index_size_mb']:.0f} MB"
        return f"Build time: {metadata['build_time_s']:.1f} s · Index size: {metadata['index_size_mb']:.0f} MB"

    def _search_table(
        self,
        conn: psycopg.Connection[Any],
        *,
        table_name: str,
        queries: np.ndarray,
        k: int,
        repeats: int,
        latency_sample_size: int,
        session_settings: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        query_sql = sql.SQL("SELECT id FROM {} ORDER BY embedding <-> %s LIMIT %s").format(sql.Identifier(table_name))

        with conn.cursor() as cur:
            for key, value in (session_settings or {}).items():
                cur.execute(sql.SQL("SET {} = {}").format(sql.SQL(cast(Any, key)), sql.SQL(cast(Any, str(value)))))

            self._warmup_queries(cur, query_sql, queries, k)

            repeat_avg_ms: list[float] = []
            sample_latencies_ms: list[float] = []
            last_ids: np.ndarray | None = None

            for repeat in range(repeats):
                latencies_ms: list[float] = []
                ids_for_repeat: list[list[int]] = []
                for query_index, query in enumerate(queries):
                    start = time.perf_counter()
                    cur.execute(query_sql, (Vector(query.tolist()), k))
                    ids = [row[0] for row in cur.fetchall()]
                    latency_ms = (time.perf_counter() - start) * 1000
                    latencies_ms.append(latency_ms)
                    ids_for_repeat.append(ids)
                    if repeat == repeats - 1 and query_index < min(latency_sample_size, len(queries)):
                        sample_latencies_ms.append(latency_ms)

                repeat_avg_ms.append(float(np.mean(latencies_ms)))
                last_ids = np.asarray(ids_for_repeat, dtype=np.int64)

            for key in session_settings or {}:
                cur.execute(sql.SQL("RESET {}").format(sql.SQL(cast(Any, key))))

        assert last_ids is not None
        return {
            "ids": last_ids,
            "avg_latency_ms": float(np.mean(repeat_avg_ms)),
            "p50_latency_ms": percentile(sample_latencies_ms, 50),
            "p95_latency_ms": percentile(sample_latencies_ms, 95),
        }

    def _warmup_queries(self, cur: psycopg.Cursor[Any], query_sql: Any, queries: np.ndarray, k: int) -> None:
        for query in queries[: min(10, len(queries))]:
            cur.execute(query_sql, (Vector(query.tolist()), k))
            cur.fetchall()

    def _database_name(self) -> str:
        return self.database_url.rsplit("/", maxsplit=1)[-1]

    def _flat_table_name(self) -> str:
        return f"{self.dataset_prefix}_flat"

    def _hnsw_table_name(self) -> str:
        return f"{self.dataset_prefix}_hnsw"

    def _ivf_table_name(self) -> str:
        return f"{self.dataset_prefix}_ivf"

    def _hnsw_index_name(self, m: int, ef_construction: int) -> str:
        return f"{self.dataset_prefix}_hnsw_m{m}_efc{ef_construction}"

    def _ivf_index_name(self, nlist: int) -> str:
        return f"{self.dataset_prefix}_ivf_nlist{nlist}"

    def _hnsw_artifact_key(self, m: int, ef_construction: int) -> str:
        return f"{self.dataset_prefix}:hnsw:{m}:{ef_construction}"

    def _ivf_artifact_key(self, nlist: int) -> str:
        return f"{self.dataset_prefix}:ivf:{nlist}"
