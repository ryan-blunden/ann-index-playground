from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ann_backend import AnnBackend, BackendSettings, ProgressUpdate, RunConfig
from ann_faiss import FaissBackend
from ann_pgvector import PgvectorBackend

DATASET_PATH = Path("data/sift-128-euclidean.hdf5")
CACHE_DIR = Path("cache")
RESULTS_CSV_PATH = Path("data/results.csv")
ENV_PATH = Path(".env")
CSS_PATH = Path(".streamlit/styles.css")
DEFAULT_HNSW_M = 32
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF_SEARCH = 64
DEFAULT_IVF_NLIST = 256
DEFAULT_IVF_NPROBE = 16
DEFAULT_QUERY_COUNT = 1_000
DEFAULT_PGVECTOR_QUERY_COUNT = 500
DEFAULT_VECTOR_COUNT = 1_000_000
DEFAULT_INCLUDE_HNSW = True
DEFAULT_INCLUDE_IVF = True
DEFAULT_BACKEND = "faiss"
DEFAULT_PGVECTOR_DATABASE_URL = "postgresql:///ann_indexes_pgvector"
DEFAULT_PGVECTOR_ADMIN_DATABASE_URL = "postgresql:///postgres"


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


def read_backend_name() -> str:
    return os.getenv("ANN_BACKEND", DEFAULT_BACKEND).strip().lower()


def read_env_value(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


load_env_file()
VECTOR_COUNT = read_vector_count()
INCLUDE_HNSW = read_bool_env("INCLUDE_HNSW", DEFAULT_INCLUDE_HNSW)
INCLUDE_IVF = read_bool_env("INCLUDE_IVF", DEFAULT_INCLUDE_IVF)
BACKEND_NAME = read_backend_name()
PGVECTOR_DATABASE_URL = read_env_value("PGVECTOR_DATABASE_URL", DEFAULT_PGVECTOR_DATABASE_URL)
PGVECTOR_ADMIN_DATABASE_URL = read_env_value("PGVECTOR_ADMIN_DATABASE_URL", DEFAULT_PGVECTOR_ADMIN_DATABASE_URL)


def build_backend() -> AnnBackend:
    settings = BackendSettings(
        dataset_path=DATASET_PATH,
        cache_dir=CACHE_DIR,
        vector_count=VECTOR_COUNT,
        database_url=PGVECTOR_DATABASE_URL,
        admin_database_url=PGVECTOR_ADMIN_DATABASE_URL,
        include_hnsw=INCLUDE_HNSW,
        include_ivf=INCLUDE_IVF,
        default_hnsw_m=DEFAULT_HNSW_M,
        default_hnsw_ef_construction=DEFAULT_HNSW_EF_CONSTRUCTION,
        default_ivf_nlist=DEFAULT_IVF_NLIST,
    )
    if BACKEND_NAME == "faiss":
        return FaissBackend(settings)
    if BACKEND_NAME == "pgvector":
        return PgvectorBackend(settings)
    raise ValueError(f"Unsupported ANN_BACKEND value: {BACKEND_NAME!r}")


BACKEND = build_backend()


def configure_page() -> None:
    st.set_page_config(
        page_title="ANN Interactive Demo",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


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


def backend_display_name() -> str:
    if BACKEND_NAME == "pgvector":
        return "pgvector"
    return "FAISS"


def default_query_count() -> int:
    if BACKEND_NAME == "pgvector":
        return DEFAULT_PGVECTOR_QUERY_COUNT
    return DEFAULT_QUERY_COUNT


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-topline">
            <h1>ANN Index Playground</h1>
            <div class="backend-badge">{backend_display_name()}</div>
          </div>
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
            value=default_query_count(),
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
            hnsw_cache_summary = BACKEND.hnsw_summary(int(hnsw_m), int(hnsw_ef_construction))
            st.markdown(f'<div class="cache-meta">{hnsw_cache_summary}</div>', unsafe_allow_html=True)
        next_column_index += 1

    ivf_nlist: int | None = None
    ivf_nprobe: int | None = None
    if INCLUDE_IVF:
        with columns[next_column_index]:
            st.markdown("### IVF")
            ivf_nlist = st.select_slider(
                "nlist",
                options=BACKEND.available_nlist_options(),
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
            ivf_cache_summary = BACKEND.ivf_summary(int(ivf_nlist))
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


def normalize_saved_results(saved_results: pd.DataFrame) -> pd.DataFrame:
    normalized = saved_results.copy()

    if "timestamp" not in normalized.columns and "timestamp_utc" in normalized.columns:
        normalized = normalized.rename(columns={"timestamp_utc": "timestamp"})
    if "run_id" in normalized.columns:
        normalized = normalized.drop(columns=["run_id"])
    if "timestamp" in normalized.columns:
        parsed = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
        if parsed.notna().any():
            local_values = parsed.dt.tz_convert(datetime.now().astimezone().tzinfo)
            formatted = local_values.dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            normalized["timestamp"] = formatted.where(parsed.notna(), normalized["timestamp"])

    return normalized


def append_results_to_csv(results: pd.DataFrame, metadata: dict[str, Any]) -> None:
    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    rows = results.copy()
    rows.insert(0, "timestamp", run_timestamp)
    rows.insert(1, "backend", BACKEND.name)
    rows.insert(2, "vectors", metadata["base_vectors"])
    rows.insert(3, "queries", metadata["query_count"])
    rows.insert(4, "k", metadata["k"])
    rows.insert(5, "repeats", metadata["repeats"])
    rows.insert(6, "latency_sample_size", metadata["latency_sample_size"])

    if RESULTS_CSV_PATH.exists():
        existing_results = normalize_saved_results(pd.read_csv(RESULTS_CSV_PATH))
        combined = pd.concat([existing_results, rows], ignore_index=True, sort=False)
        combined.to_csv(RESULTS_CSV_PATH, index=False)
        return

    rows.to_csv(RESULTS_CSV_PATH, index=False)


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
        if BACKEND.initial_artifacts_exist():
            st.session_state["initial_cache_ready"] = True
        else:
            st.markdown("## Preparing cache")
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(update: ProgressUpdate) -> None:
                progress_bar.progress(update.current_step / update.total_steps)
                status_text.write(update.message)

            BACKEND.ensure_initial_artifacts(progress=update_progress)
            st.session_state["initial_cache_ready"] = True
            st.rerun()

    render_header()

    config = render_controls()

    if config is not None:
        with st.spinner("Measuring search performance..."):
            results, metadata = BACKEND.run_comparison(config)
        append_results_to_csv(results, metadata)
        history = st.session_state.setdefault("results_history", [])
        history.insert(0, {"results": results, "metadata": metadata})
        st.session_state["results_history"] = history[:8]

    if "results_history" in st.session_state:
        render_results(st.session_state["results_history"])


if __name__ == "__main__":
    main()
