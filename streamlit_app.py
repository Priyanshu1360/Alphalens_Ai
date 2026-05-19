import time
import os

import matplotlib.pyplot as plt
import streamlit as st

from check_setup import build_report
from main import run_pipeline
from src.config import Config
from src.generator import generate_answer
from src.retriever import dense_search, hybrid_search, sparse_search

# Keep UI resilient by default even if shell env has strict remote flags set.
Config.QDRANT_FALLBACK_TO_LOCAL_ON_ERROR = True
Config.EMBEDDING_FALLBACK_TO_LOCAL_ON_ERROR = True

FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
ACCURATE_LLM_MODEL = os.getenv("ACCURATE_LLM_MODEL", "llama-3.3-70b-versatile")


def _default_mode():
    mode = str(Config.RETRIEVER_DEFAULT_MODE).lower()
    return mode if mode in {"hybrid", "dense", "sparse"} else "hybrid"


def _retrieve(query, mode, limit, prefetch_limit, rerank):
    if mode == "dense":
        return dense_search(query, limit=limit, rerank=rerank)
    if mode == "sparse":
        return sparse_search(query, limit=limit, rerank=rerank)
    return hybrid_search(
        query,
        limit=limit,
        prefetch_limit=prefetch_limit,
        rerank=rerank,
    )


def _retrieve_with_resilience(query, mode, limit, prefetch_limit, rerank):
    try:
        return _retrieve(query, mode, limit, prefetch_limit, rerank), None
    except Exception as exc:
        message = str(exc)
        if "Remote Qdrant connection failed" not in message:
            raise
        Config.QDRANT_FALLBACK_TO_LOCAL_ON_ERROR = True
        results = _retrieve(query, mode, limit, prefetch_limit, rerank)
        return results, (
            "Remote Qdrant was unavailable. Automatically retried with local fallback."
        )


def _pick_llm_model(query, strategy):
    strategy = (strategy or "auto").lower()
    if strategy == "fast":
        return FAST_LLM_MODEL, "fast"
    if strategy == "accurate":
        return ACCURATE_LLM_MODEL, "accurate"

    query_l = (query or "").lower()
    complex_terms = [
        "compare",
        "analyze",
        "analysis",
        "trend",
        "drivers",
        "explain",
        "detailed",
        "risk",
        "impact",
        "why",
    ]
    is_complex = len(query_l) > 90 or any(term in query_l for term in complex_terms)
    if is_complex:
        return ACCURATE_LLM_MODEL, "auto->accurate"
    return FAST_LLM_MODEL, "auto->fast"


def _show_sources(results, sources_limit):
    st.subheader("Sources")
    if not results:
        st.info("No retrieval results found.")
        return

    for item in results[: max(0, sources_limit)]:
        source_title = (
            f"[{item.get('rank', '-')}] "
            f"{item.get('company', 'unknown')} | {item.get('file_name', 'unknown')}"
        )
        with st.expander(source_title, expanded=False):
            st.write(
                {
                    "year": item.get("year"),
                    "quarter": item.get("quarter"),
                    "report_type": item.get("report_type"),
                    "score": item.get("score"),
                    "rerank_score": item.get("rerank_score"),
                    "coherence_score": item.get("coherence_score"),
                    "chunk_quality": item.get("chunk_quality"),
                }
            )
            snippet = (item.get("snippet") or "").strip()
            if snippet:
                st.caption("Snippet")
                st.write(snippet)


def _two_line_summary(query, answer, results):
    text = (answer or "").strip()
    if not text:
        return f"Summary:\nNo answer generated for '{query}'."

    lines = [line.strip("-* ").strip() for line in text.splitlines() if line.strip()]
    sentence_chunks = []

    for line in lines:
        parts = [part.strip() for part in line.split(".") if part.strip()]
        for part in parts:
            sentence = part if part.endswith(".") else f"{part}."
            sentence_chunks.append(sentence)
            if len(sentence_chunks) >= 2:
                break
        if len(sentence_chunks) >= 2:
            break

    if not sentence_chunks:
        sentence_chunks = [text[:160].strip()]
    if len(sentence_chunks) == 1:
        top_source = results[0].get("file_name") if results else "available sources"
        sentence_chunks.append(f"Based on {top_source}.")

    return f"Summary:\n{sentence_chunks[0]}\n{sentence_chunks[1]}"


def _query_needs_plot(query_text):
    text = (query_text or "").lower()
    plot_terms = [
        "plot",
        "graph",
        "chart",
        "pie chart",
        "bar chart",
        "line chart",
        "visualize",
        "visualise",
        "show graph",
        "show chart",
    ]
    return any(term in text for term in plot_terms)


def _plot_results(results, max_items=10):
    if not results:
        st.info("No results to plot.")
        return

    top = results[: max(1, max_items)]
    labels = []
    retrieval_scores = []
    rerank_scores = []
    coherence_scores = []

    for item in top:
        rank = item.get("rank", "-")
        company = item.get("company", "unknown")
        label = f"{rank}:{company}"
        labels.append(label)
        retrieval_scores.append(float(item.get("score") or 0.0))
        rerank_scores.append(float(item.get("rerank_score") or 0.0))
        coherence_scores.append(float(item.get("coherence_score") or 0.0))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(labels))
    width = 0.25
    ax.bar([i - width for i in x], retrieval_scores, width=width, label="retrieval_score")
    ax.bar(x, rerank_scores, width=width, label="rerank_score")
    ax.bar([i + width for i in x], coherence_scores, width=width, label="coherence_score")
    ax.set_title("Top Retrieval Results")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_results_pie(results, max_items=8):
    if not results:
        return

    top = results[: max(1, max_items)]
    labels = []
    values = []

    for item in top:
        rank = item.get("rank", "-")
        company = item.get("company", "unknown")
        file_name = item.get("file_name", "unknown")
        labels.append(f"{rank}:{company} ({file_name})")
        values.append(float(item.get("score") or 0.0))

    total = sum(values)
    if total <= 0:
        st.info("Pie chart skipped: scores are all zero.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    ax.set_title("Retrieval Score Share (Top Results)")
    ax.axis("equal")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Financial Filings RAG", layout="wide")
    st.title("Financial Filings RAG")
    st.caption("Query SEC filings with hybrid retrieval and LLM answer generation.")

    with st.sidebar:
        st.header("Settings")
        strict_remote = st.checkbox(
            "Strict remote Qdrant (disable fallback)",
            value=False,
            help="When enabled, query will fail if remote Qdrant is unreachable.",
        )
        Config.QDRANT_FALLBACK_TO_LOCAL_ON_ERROR = not strict_remote
        mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "sparse"], index=["hybrid", "dense", "sparse"].index(_default_mode()))
        model_strategy = st.selectbox(
            "LLM strategy",
            ["auto", "fast", "accurate"],
            index=0,
            help=(
                "auto: chooses model from query complexity | "
                "fast: lower latency | accurate: better reasoning quality"
            ),
        )
        limit = st.number_input("Top-K results", min_value=1, max_value=50, value=int(Config.ASK_DEFAULT_LIMIT), step=1)
        prefetch_limit = st.number_input(
            "Hybrid prefetch limit",
            min_value=1,
            max_value=200,
            value=int(Config.HYBRID_PREFETCH_LIMIT),
            step=1,
        )
        rerank = st.checkbox("Enable reranking", value=bool(Config.RERANK_ENABLED))
        sources_limit = st.number_input(
            "Sources to display",
            min_value=1,
            max_value=20,
            value=int(Config.ASK_DEFAULT_SOURCES_LIMIT),
            step=1,
        )
        st.divider()
        if st.button("Check Setup", use_container_width=True):
            with st.spinner("Checking setup..."):
                report = build_report()
            st.json(report)
        if st.button("DB Quick Status", use_container_width=True):
            try:
                probe = _retrieve("Amazon revenue 2024", mode="hybrid", limit=1, prefetch_limit=10, rerank=True)
                st.success(f"DB query ok. Retrieved {len(probe)} result(s).")
                if probe:
                    st.write(
                        {
                            "company": probe[0].get("company"),
                            "file_name": probe[0].get("file_name"),
                            "year": probe[0].get("year"),
                            "score": probe[0].get("score"),
                        }
                    )
            except Exception as exc:
                st.error(f"DB query failed: {exc}")
        if st.button("Rebuild Index", use_container_width=True):
            with st.spinner("Running ingestion pipeline... this may take several minutes."):
                result = run_pipeline()
            st.success("Index rebuild completed.")
            st.json(result)

    query = st.text_area(
        "Ask a question",
        placeholder="Example: Summarize Amazon revenue trend in 2024 with key drivers and cite sources.",
        height=120,
    )

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if st.button("Run Query", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            started = time.time()
            with st.spinner("Retrieving and generating answer..."):
                try:
                    selected_model, selection_mode = _pick_llm_model(query.strip(), model_strategy)
                    Config.LLM_MODEL = selected_model
                    results, fallback_notice = _retrieve_with_resilience(
                        query=query.strip(),
                        mode=mode,
                        limit=int(limit),
                        prefetch_limit=int(prefetch_limit),
                        rerank=bool(rerank),
                    )
                    generation = generate_answer(query.strip(), results)
                    elapsed = round(time.time() - started, 2)
                    run_result = {
                        "query": query.strip(),
                        "mode": mode,
                        "llm_model": selected_model,
                        "llm_selection_mode": selection_mode,
                        "show_plot": _query_needs_plot(query.strip()),
                        "elapsed_seconds": elapsed,
                        "results": results,
                        "generation": generation,
                    }
                    st.session_state["history"].append(run_result)
                    if fallback_notice:
                        st.warning(fallback_notice)
                except Exception as exc:
                    st.error(f"Run failed: {exc}")

    if st.session_state["history"]:
        latest = st.session_state["history"][-1]
        st.subheader("Answer")
        answer_text = latest["generation"].get("answer") or "No answer generated."
        st.write(answer_text)
        if latest["generation"].get("llm_error"):
            st.warning(f"LLM error: {latest['generation']['llm_error']}")
        st.caption(
            f"Mode: {latest['mode']} | LLM: {latest.get('llm_model')} "
            f"({latest.get('llm_selection_mode')}) | Results: {len(latest['results'])} "
            f"| Time: {latest['elapsed_seconds']}s"
        )
        if latest.get("show_plot"):
            st.subheader("Result Graph")
            _plot_results(latest["results"], max_items=10)
            _plot_results_pie(latest["results"], max_items=8)
        st.success(_two_line_summary(latest.get("query"), answer_text, latest.get("results", [])))
        _show_sources(latest["results"], int(sources_limit))

        with st.expander("Raw JSON", expanded=False):
            st.json(
                {
                    "query": latest["query"],
                    "mode": latest["mode"],
                    "llm_model": latest.get("llm_model"),
                    "llm_selection_mode": latest.get("llm_selection_mode"),
                    "show_plot": latest.get("show_plot"),
                    "elapsed_seconds": latest["elapsed_seconds"],
                    "generation_mode": latest["generation"].get("generation_mode"),
                    "llm_error": latest["generation"].get("llm_error"),
                    "results_count": len(latest["results"]),
                }
            )


if __name__ == "__main__":
    main()
