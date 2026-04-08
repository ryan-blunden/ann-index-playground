from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class ProgressUpdate:
    current_step: int
    total_steps: int
    message: str


ProgressCallback = Callable[[ProgressUpdate], None]


@dataclass(frozen=True)
class BackendSettings:
    dataset_path: Path
    cache_dir: Path
    vector_count: int
    database_url: str | None = None
    admin_database_url: str | None = None
    include_hnsw: bool = True
    include_ivf: bool = True
    default_hnsw_m: int = 32
    default_hnsw_ef_construction: int = 200
    default_ivf_nlist: int = 256


@dataclass(frozen=True)
class RunConfig:
    query_count: int
    k: int
    repeats: int
    latency_sample_size: int
    hnsw_m: int | None
    hnsw_ef_construction: int | None
    hnsw_ef_search: int | None
    ivf_nlist: int | None
    ivf_nprobe: int | None


class AnnBackend(Protocol):
    name: str

    def initial_artifacts_exist(self) -> bool: ...

    def ensure_initial_artifacts(self, progress: ProgressCallback | None = None) -> None: ...

    def available_nlist_options(self) -> list[int]: ...

    def hnsw_summary(self, m: int, ef_construction: int) -> str: ...

    def ivf_summary(self, nlist: int) -> str: ...

    def run_comparison(self, config: RunConfig) -> tuple[pd.DataFrame, dict[str, Any]]: ...
