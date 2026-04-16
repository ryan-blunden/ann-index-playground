from __future__ import annotations

import math
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
    pgvector_maintenance_work_mem: str | None = None
    service_url: str | None = None
    service_data_dir: Path | None = None
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
    settings: BackendSettings

    def initial_artifacts_exist(self) -> bool: ...

    def ensure_initial_artifacts(self, progress: ProgressCallback | None = None) -> None: ...

    def available_nlist_options(self) -> list[int]: ...

    def flat_summary(self) -> str: ...

    def hnsw_summary(self, m: int, ef_construction: int) -> str: ...

    def ivf_summary(self, nlist: int, nprobe: int | None = None) -> str: ...

    def run_comparison(self, config: RunConfig) -> tuple[pd.DataFrame, dict[str, Any]]: ...


def recommended_ivf_nlist_options(vector_count: int, *, max_option: int = 4096) -> list[int]:
    base_options = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    target = max(64, int(round(math.sqrt(vector_count))))
    lower_bound = max(64, target // 8)
    upper_bound = min(max_option, target * 4)
    options = [value for value in base_options if value <= vector_count and lower_bound <= value <= upper_bound]
    if not options:
        options = [value for value in base_options if value <= min(vector_count, max_option)]
    if not options:
        options = [max(1, min(vector_count, max_option))]
    return options


def recommended_ivf_nprobe(nlist: int) -> int:
    return max(1, int(math.sqrt(nlist)))
