from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import streamlit as st

from ann_core import build_flat, build_hnsw, build_ivf, evaluate_index, load_sift_hdf5

DATASET_PATH = Path("data/sift-128-euclidean.hdf5")
CACHE_DIR = Path("cache")
ENV_PATH = Path(".env")
CSS_PATH = Path(".streamlit/styles.css")
DEFAULT_HNSW_M = 32
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF_SEARCH = 64
DEFAULT_IVF_NLIST = 256
DEFAULT_IVF_NPROBE = 16
DEFAULT_QUERY_COUNT = 1_000
DEFAULT_VECTOR_COUNT = 1_000_000
DEFAULT_INCLUDE_HNSW = True
DEFAULT_INCLUDE_IVF = True


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


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def read_vector_count() -> int:
    raw_value = os.getenv("VECTOR_COUNT", str(DEFAULT_VECTOR_COUNT))
    try:
        value = int(raw_value.replace("_", "").replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"Invalid VECTOR_COUNT value: {raw_value!r}") from exc

    if value <= 0:
        raise ValueError(f"VECTOR_COUNT must be positive, got {value}.")
    return value


def read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {name} value: {raw_value!r}")


load_env_file()
VECTOR_COUNT = read_vector_count()
INCLUDE_HNSW = read_bool_env("INCLUDE_HNSW", DEFAULT_INCLUDE_HNSW)
INCLUDE_IVF = read_bool_env("INCLUDE_IVF", DEFAULT_INCLUDE_IVF)


def configure_page() -> None:
    st.set_page_config(
        page_title="ANN Interactive Demo",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_sift_hdf5(DATASET_PATH)


@st.cache_resource(show_spinner=False)
def dataset_fingerprint() -> str:
    stat = DATASET_PATH.stat()
    return f"{DATASET_PATH.stem}_{stat.st_size}_{int(stat.st_mtime)}"


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(name: str) -> Path:
    ensure_cache_dir()
    return CACHE_DIR / f"{dataset_fingerprint()}_{name}.index"


def metadata_path(index_path: Path) -> Path:
    return index_path.with_suffix(".json")


def write_cache_metadata(index_path: Path, build_time_s: float) -> dict[str, float]:
    payload = {
        "build_time_s": float(build_time_s),
        "index_size_mb": index_path.stat().st_size / (1024 * 1024),
    }
    metadata_path(index_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_cache_metadata(index_path: Path) -> dict[str, float]:
    path = metadata_path(index_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    payload = {
        "build_time_s": 0.0,
        "index_size_mb": index_path.stat().st_size / (1024 * 1024),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def initial_cache_paths() -> list[Path]:
    paths = [cache_path(f"flat_{VECTOR_COUNT}")]
    if INCLUDE_HNSW:
        paths.append(cache_path(f"hnsw_{VECTOR_COUNT}_m{DEFAULT_HNSW_M}_efc{DEFAULT_HNSW_EF_CONSTRUCTION}"))
    if INCLUDE_IVF:
        paths.append(cache_path(f"ivf_{VECTOR_COUNT}_nlist{DEFAULT_IVF_NLIST}"))
    return paths


def initial_cache_exists() -> bool:
    return all(path.exists() for path in initial_cache_paths())


def load_or_build_flat(xb: np.ndarray, dimension: int) -> tuple[faiss.Index, float, float, bool]:
    path = cache_path(f"flat_{VECTOR_COUNT}")
    if path.exists():
        metadata = read_cache_metadata(path)
        return faiss.read_index(str(path)), metadata["build_time_s"], metadata["index_size_mb"], True

    index, build_time_s, _ = build_flat(xb, dimension)
    faiss.write_index(index, str(path))
    metadata = write_cache_metadata(path, build_time_s)
    return index, metadata["build_time_s"], metadata["index_size_mb"], False


def load_or_build_hnsw(xb: np.ndarray, dimension: int, m: int, ef_construction: int) -> tuple[faiss.Index, float, float, bool]:
    path = cache_path(f"hnsw_{VECTOR_COUNT}_m{m}_efc{ef_construction}")
    if path.exists():
        metadata = read_cache_metadata(path)
        return faiss.read_index(str(path)), metadata["build_time_s"], metadata["index_size_mb"], True

    index, build_time_s, _ = build_hnsw(xb, dimension, m, ef_construction)
    faiss.write_index(index, str(path))
    metadata = write_cache_metadata(path, build_time_s)
    return index, metadata["build_time_s"], metadata["index_size_mb"], False


def load_or_build_ivf(xb: np.ndarray, dimension: int, nlist: int) -> tuple[faiss.Index, float, float, bool]:
    path = cache_path(f"ivf_{VECTOR_COUNT}_nlist{nlist}")
    if path.exists():
        metadata = read_cache_metadata(path)
        return faiss.read_index(str(path)), metadata["build_time_s"], metadata["index_size_mb"], True

    index, build_time_s, _ = build_ivf(xb, dimension, nlist)
    faiss.write_index(index, str(path))
    metadata = write_cache_metadata(path, build_time_s)
    return index, metadata["build_time_s"], metadata["index_size_mb"], False


def cache_step_label(index_name: str, loaded_from_cache: bool) -> str:
    action = "Loading cached" if loaded_from_cache else "Building"
    return f"{action} {index_name} index..."


def ensure_initial_cache(progress_bar: Any | None = None, status_text: Any | None = None) -> None:
    total_steps = 2 + int(INCLUDE_HNSW) + int(INCLUDE_IVF)
    current_step = 0

    def update(detail: str) -> None:
        nonlocal current_step
        current_step += 1
        if progress_bar is not None:
            progress_bar.progress(current_step / total_steps)
        if status_text is not None:
            status_text.write(detail)

    update(f"Loading the first {VECTOR_COUNT:,} vectors from SIFT1M...")
    xb_full, _, _ = load_dataset()
    xb = np.ascontiguousarray(xb_full[:VECTOR_COUNT])
    dimension = xb.shape[1]

    flat_loaded = cache_path(f"flat_{VECTOR_COUNT}").exists()
    update(cache_step_label("Flat", flat_loaded))
    load_or_build_flat(xb, dimension)

    if INCLUDE_HNSW:
        hnsw_loaded = cache_path(f"hnsw_{VECTOR_COUNT}_m{DEFAULT_HNSW_M}_efc{DEFAULT_HNSW_EF_CONSTRUCTION}").exists()
        update(cache_step_label("HNSW", hnsw_loaded))
        load_or_build_hnsw(xb, dimension, DEFAULT_HNSW_M, DEFAULT_HNSW_EF_CONSTRUCTION)

    if INCLUDE_IVF:
        ivf_loaded = cache_path(f"ivf_{VECTOR_COUNT}_nlist{DEFAULT_IVF_NLIST}").exists()
        update(cache_step_label("IVF", ivf_loaded))
        load_or_build_ivf(xb, dimension, DEFAULT_IVF_NLIST)


def available_nlist_options() -> list[int]:
    return [value for value in [256, 512, 1024, 2048, 4096, 8192] if value <= VECTOR_COUNT]


def format_cache_summary(index_path: Path) -> str:
    if not index_path.exists():
        return "Cache not built yet."

    metadata = read_cache_metadata(index_path)
    return f"Build time: {metadata['build_time_s']:.1f} s · Index size: {metadata['index_size_mb']:.0f} MB"


def to_card_html(row: pd.Series) -> str:
    family = row["family"]
    return f"""
    <div class="index-card">
      <div class="index-label">Index</div>
      <div class="index-title">{family}</div>
      <div class="index-config">{row['config_text']}</div>
      <div class="stat-grid">
        <div class="stat-box">
          <div class="stat-label">Avg Latency</div>
          <div class="stat-value">{row['avg_latency_ms']:.3f} ms</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Recall</div>
          <div class="stat-value">{row['recall_at_10'] * 100:.1f}%</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Vs Flat</div>
          <div class="stat-value">{row['speedup_vs_flat']:.2f}x</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">p95 Latency</div>
          <div class="stat-value">{row['p95_latency_ms']:.3f} ms</div>
        </div>
      </div>
    </div>
    """


def run_comparison(config: RunConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    xb_full, xq_full, _ = load_dataset()
    xb = np.ascontiguousarray(xb_full[:VECTOR_COUNT])
    xq = np.ascontiguousarray(xq_full[: config.query_count])
    dimension = xb.shape[1]

    placeholder_reference = np.zeros((len(xq), config.k), dtype=np.int64)

    flat_index, flat_build_s, flat_size_mb, _ = load_or_build_flat(xb, dimension)
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
        "config_text": "Exact baseline",
        "avg_latency_ms": flat_search["avg_latency_ms"],
        "p50_latency_ms": flat_search["p50_latency_ms"],
        "p95_latency_ms": flat_search["p95_latency_ms"],
        "recall_at_10": 1.0,
        "build_time_s": flat_build_s,
        "index_size_mb": flat_size_mb,
        "speedup_vs_flat": 1.0,
    }

    rows = [flat_row]

    if INCLUDE_HNSW:
        assert config.hnsw_m is not None
        assert config.hnsw_ef_construction is not None
        assert config.hnsw_ef_search is not None
        hnsw_index, hnsw_build_s, hnsw_size_mb, _ = load_or_build_hnsw(xb, dimension, config.hnsw_m, config.hnsw_ef_construction)
        hnsw_index.hnsw.efSearch = config.hnsw_ef_search
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
                "config_text": f"M={config.hnsw_m} · efC={config.hnsw_ef_construction} · efS={config.hnsw_ef_search}",
                "avg_latency_ms": hnsw_search["avg_latency_ms"],
                "p50_latency_ms": hnsw_search["p50_latency_ms"],
                "p95_latency_ms": hnsw_search["p95_latency_ms"],
                "recall_at_10": hnsw_search["recall_at_10"],
                "build_time_s": hnsw_build_s,
                "index_size_mb": hnsw_size_mb,
                "speedup_vs_flat": flat_row["avg_latency_ms"] / hnsw_search["avg_latency_ms"],
            }
        )

    if INCLUDE_IVF:
        assert config.ivf_nlist is not None
        assert config.ivf_nprobe is not None
        ivf_index, ivf_build_s, ivf_size_mb, _ = load_or_build_ivf(xb, dimension, config.ivf_nlist)
        ivf_index.nprobe = config.ivf_nprobe
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
                "config_text": f"nlist={config.ivf_nlist} · nprobe={config.ivf_nprobe}",
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
        "base_vectors": VECTOR_COUNT,
        "query_count": config.query_count,
        "k": config.k,
        "repeats": config.repeats,
        "latency_sample_size": config.latency_sample_size,
    }
    return results, metadata


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>ANN Index Playground</h1>
          <p class="hero-copy">Comparing index performance using the SIFT1M dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_controls() -> RunConfig | None:
    column_count = 1 + int(INCLUDE_HNSW) + int(INCLUDE_IVF)
    columns = st.columns(column_count, gap="large")
    top_left = columns[0]
    next_column_index = 1

    with top_left:
        st.markdown("### Settings")
        st.text_input(
            "Vectors",
            value=f"{VECTOR_COUNT:,}",
            disabled=True,
            help="Edit `VECTOR_COUNT` in `.env` and re-run the application to change the number of vectors.",
        )
        query_count = st.number_input(
            "Queries",
            min_value=100,
            max_value=10_000,
            value=DEFAULT_QUERY_COUNT,
            step=100,
        )
        repeats = st.number_input("Timing repeats", min_value=3, max_value=10, value=3, step=1)
        latency_sample_size = st.number_input(
            "Single-query sample size",
            min_value=32,
            max_value=1024,
            value=128,
            step=32,
        )

    hnsw_m: int | None = None
    hnsw_ef_construction: int | None = None
    hnsw_ef_search: int | None = None
    if INCLUDE_HNSW:
        with columns[next_column_index]:
            st.markdown("### HNSW")
            hnsw_m = st.slider(
                "M",
                min_value=8,
                max_value=64,
                value=DEFAULT_HNSW_M,
                step=4,
                help=(
                    "How many graph connections each vector keeps. Higher M usually improves recall, "
                    "but makes the index larger and slower to build."
                ),
            )
            hnsw_ef_construction = st.slider(
                "efConstruction",
                min_value=40,
                max_value=400,
                value=DEFAULT_HNSW_EF_CONSTRUCTION,
                step=20,
                help=(
                    "How much work HNSW does while building the graph. Higher values usually improve graph "
                    "quality and recall, but increase build time."
                ),
            )
            hnsw_ef_search = st.slider(
                "efSearch",
                min_value=8,
                max_value=256,
                value=DEFAULT_HNSW_EF_SEARCH,
                step=8,
                help="How many candidates HNSW explores at query time. Higher values usually improve recall, but increase query latency.",
            )
            hnsw_cache_summary = format_cache_summary(cache_path(f"hnsw_{VECTOR_COUNT}_m{int(hnsw_m)}_efc{int(hnsw_ef_construction)}"))
            st.markdown(f'<div class="cache-meta">{hnsw_cache_summary}</div>', unsafe_allow_html=True)
        next_column_index += 1

    ivf_nlist: int | None = None
    ivf_nprobe: int | None = None
    if INCLUDE_IVF:
        with columns[next_column_index]:
            st.markdown("### IVF")
            ivf_nlist = st.select_slider(
                "nlist",
                options=available_nlist_options(),
                value=DEFAULT_IVF_NLIST,
                help=(
                    "How many coarse partitions IVF creates. Higher nlist can make search more selective, "
                    "but training and building become more expensive and tuning matters more."
                ),
            )
            ivf_nprobe = st.select_slider(
                "nprobe",
                options=[1, 2, 4, 8, 16, 32, 64, 128],
                value=DEFAULT_IVF_NPROBE,
                help="How many IVF partitions are searched for each query. Higher nprobe usually improves recall, but increases query latency.",
            )
            ivf_cache_summary = format_cache_summary(cache_path(f"ivf_{VECTOR_COUNT}_nlist{int(ivf_nlist)}"))
            st.markdown(f'<div class="cache-meta">{ivf_cache_summary}</div>', unsafe_allow_html=True)

    run_clicked = st.button("Run", use_container_width=True, type="primary")

    if not run_clicked:
        return None

    if INCLUDE_IVF and ivf_nprobe is not None and ivf_nlist is not None and ivf_nprobe > ivf_nlist:
        st.error("`nprobe` cannot exceed `nlist`.")
        return None

    return RunConfig(
        query_count=int(query_count),
        k=10,
        repeats=int(repeats),
        latency_sample_size=int(latency_sample_size),
        hnsw_m=None if hnsw_m is None else int(hnsw_m),
        hnsw_ef_construction=None if hnsw_ef_construction is None else int(hnsw_ef_construction),
        hnsw_ef_search=None if hnsw_ef_search is None else int(hnsw_ef_search),
        ivf_nlist=None if ivf_nlist is None else int(ivf_nlist),
        ivf_nprobe=None if ivf_nprobe is None else int(ivf_nprobe),
    )


def render_cards(df: pd.DataFrame) -> None:
    cols = st.columns(len(df), gap="large")
    for column, (_, row) in zip(cols, df.iterrows(), strict=False):
        column.markdown(to_card_html(row), unsafe_allow_html=True)


def render_results(history: list[dict[str, Any]]) -> None:
    st.markdown('<div class="run-history">', unsafe_allow_html=True)
    total_runs = len(history)
    for run_index, entry in enumerate(history, start=1):
        st.markdown(
            (
                '<div class="run-block"><div class="run-block-header">'
                f'<div class="run-block-title">Run {total_runs - run_index + 1}</div>'
                "</div></div>"
            ),
            unsafe_allow_html=True,
        )
        render_cards(entry["results"])
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    configure_page()

    if not st.session_state.get("initial_cache_ready", False):
        if initial_cache_exists():
            st.session_state["initial_cache_ready"] = True
        else:
            st.markdown("## Preparing cache")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            ensure_initial_cache(progress_bar=progress_bar, status_text=status_text)
            st.session_state["initial_cache_ready"] = True
            st.rerun()

    render_header()

    config = render_controls()

    if config is not None:
        with st.spinner("Measuring search performance..."):
            results, metadata = run_comparison(config)
        history = st.session_state.setdefault("results_history", [])
        history.insert(0, {"results": results, "metadata": metadata})
        st.session_state["results_history"] = history[:8]

    if "results_history" in st.session_state:
        render_results(st.session_state["results_history"])


if __name__ == "__main__":
    main()
