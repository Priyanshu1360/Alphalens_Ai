import time
import os
import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import streamlit as st

from check_setup import build_report
from main import run_pipeline
from src.config import Config
from src.generator import generate_answer
from src.retriever import dense_search, hybrid_search, sparse_search

try:
    from src.chat_history_store import ChatHistoryStore
except Exception:  # pragma: no cover - optional dependency fallback
    ChatHistoryStore = None

# Keep UI resilient by default even if shell env has strict remote flags set.
Config.QDRANT_FALLBACK_TO_LOCAL_ON_ERROR = True
Config.EMBEDDING_FALLBACK_TO_LOCAL_ON_ERROR = True

FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
ACCURATE_LLM_MODEL = os.getenv("ACCURATE_LLM_MODEL", "llama-3.3-70b-versatile")
LOGGER = logging.getLogger("finance_rag_streamlit")

chat_history_store = None
if ChatHistoryStore and Config.CHAT_HISTORY_ENABLED and Config.POSTGRES_URL:
    try:
        chat_history_store = ChatHistoryStore(Config.POSTGRES_URL)
    except Exception as exc:
        LOGGER.warning("Chat history DB disabled in Streamlit: %s", exc)


def _safe_store_chat_history(
    *,
    query: str,
    answer: str,
    thread_id: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
):
    if chat_history_store is None:
        return
    try:
        chat_history_store.add_entry(
            route="streamlit-query",
            query=query,
            answer=answer,
            thread_id=thread_id,
            metadata=metadata or {},
        )
    except Exception as exc:
        LOGGER.warning("Failed to persist Streamlit chat history: %s", exc)


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


def _show_sources(results, sources_limit, *, show_title=True):
    if show_title:
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


def _inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(1200px 500px at 10% -10%, #1a2c4a 0%, rgba(26,44,74,0) 55%),
                radial-gradient(1000px 420px at 95% 0%, #3a2330 0%, rgba(58,35,48,0) 50%),
                linear-gradient(180deg, #0b1220 0%, #0e1628 45%, #0b1220 100%);
        }
        [data-testid="stAppViewContainer"] {
            background: transparent;
        }
        [data-testid="stSidebar"] {
            background: #0f1a2d;
        }
        [data-testid="stSidebar"] * {
            color: #d7e3ff !important;
        }
        [data-testid="stHeader"] {
            background: rgba(11, 18, 32, 0.75);
        }
        .app-title {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            color: #e8f0ff;
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            text-align: center;
            color: #a8badf;
            margin-bottom: 1.2rem;
        }
        .search-wrap {
            max-width: 760px;
            margin: 0 auto;
            padding: 0.2rem 0.4rem 0.1rem 0.4rem;
            border: 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
            backdrop-filter: none;
        }
        .search-dock {
            position: sticky;
            bottom: 0.7rem;
            z-index: 40;
            padding-top: 0.4rem;
        }
        .loader-wrap {
            border: 1px solid #30496f;
            border-radius: 14px;
            padding: 0.8rem;
            background: #13233b;
            margin-bottom: 0.6rem;
            max-width: 520px;
        }
        .bot {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }
        .bot-head {
            width: 38px;
            height: 38px;
            border: 2px solid #8ab4ff;
            border-radius: 10px;
            position: relative;
            background: #1a3155;
            animation: pulse 1.1s infinite ease-in-out;
        }
        .bot-head::before,
        .bot-head::after {
            content: "";
            position: absolute;
            width: 6px;
            height: 6px;
            background: #b9d3ff;
            border-radius: 999px;
            top: 11px;
        }
        .bot-head::before { left: 9px; }
        .bot-head::after { right: 9px; }
        .bot-mouth {
            position: absolute;
            left: 10px;
            bottom: 8px;
            width: 14px;
            height: 3px;
            background: #b9d3ff;
            border-radius: 6px;
        }
        .bot-text {
            color: #d6e6ff;
            font-weight: 600;
        }
        [data-testid="stChatMessage"] {
            background: rgba(17, 30, 51, 0.62);
            border: 1px solid #253f66;
            border-radius: 12px;
        }
        .stAlert {
            background: rgba(17, 30, 51, 0.92);
            color: #d7e3ff;
            border: 1px solid #2c4f80;
        }
        .stExpander {
            border: 1px solid #2a4061;
            border-radius: 10px;
            background: rgba(14, 24, 40, 0.75);
        }
        .stMarkdown, .stText, label, p, li, span, div {
            color: #d7e3ff;
        }
        .stTextInput input {
            background: #0f1b2e !important;
            color: #e5efff !important;
            border: 1px solid #34527e !important;
        }
        .stButton button, .stFormSubmitButton button {
            background: linear-gradient(180deg, #2b4d80 0%, #223f69 100%) !important;
            color: #e9f2ff !important;
            border: 1px solid #3e6497 !important;
        }
        @keyframes pulse {
            0% { transform: translateY(0px); opacity: 0.75; }
            50% { transform: translateY(-2px); opacity: 1; }
            100% { transform: translateY(0px); opacity: 0.75; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_loading_bot(target):
    target.markdown(
        """
        <div class="loader-wrap">
          <div class="bot">
            <div class="bot-head"><div class="bot-mouth"></div></div>
            <div class="bot-text">Alphalens AI is thinking...</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_copilot_actions(query, idx):
    st.caption("Copilot Actions")
    c1, c2, c3 = st.columns(3)
    if c1.button("Explain Simply", key=f"copilot_simple_{idx}", use_container_width=True):
        st.session_state["pending_query"] = (
            f"{query}\n\nExplain this in simple language with 5 short bullet points."
        )
        st.rerun()
    if c2.button("Find Risks", key=f"copilot_risks_{idx}", use_container_width=True):
        st.session_state["pending_query"] = (
            f"{query}\n\nList key risks and potential impact in concise bullets."
        )
        st.rerun()
    if c3.button("Compare YoY", key=f"copilot_yoy_{idx}", use_container_width=True):
        st.session_state["pending_query"] = (
            f"{query}\n\nCompare year-over-year changes and highlight major drivers."
        )
        st.rerun()


def _render_assistant_turn(item, sources_limit, idx, *, show_copilot=False):
    answer_text = item["generation"].get("answer") or "No answer generated."
    st.info(_two_line_summary(item.get("query"), answer_text, item.get("results", [])))
    st.write(answer_text)

    if item["generation"].get("llm_error"):
        st.warning(f"LLM error: {item['generation']['llm_error']}")

    st.caption(
        f"Thread: {item.get('thread_id')} | Mode: {item['mode']} | LLM: {item.get('llm_model')} "
        f"({item.get('llm_selection_mode')}) | Results: {len(item['results'])} "
        f"| Time: {item['elapsed_seconds']}s"
    )
    if show_copilot:
        _render_copilot_actions(item.get("query", ""), idx)

    with st.expander("Sources", expanded=False):
        _show_sources(item["results"], int(sources_limit), show_title=False)

    if item.get("show_plot"):
        with st.expander("Charts", expanded=False):
            _plot_results(item["results"], max_items=10)
            _plot_results_pie(item["results"], max_items=8)

    with st.expander("Raw JSON", expanded=False):
        st.json(
            {
                "query": item["query"],
                "mode": item["mode"],
                "thread_id": item.get("thread_id"),
                "llm_model": item.get("llm_model"),
                "llm_selection_mode": item.get("llm_selection_mode"),
                "show_plot": item.get("show_plot"),
                "elapsed_seconds": item["elapsed_seconds"],
                "generation_mode": item["generation"].get("generation_mode"),
                "llm_error": item["generation"].get("llm_error"),
                "results_count": len(item["results"]),
            }
        )


def main():
    st.set_page_config(page_title="Alphalens AI", layout="wide")
    _inject_styles()
    st.markdown('<div class="app-title">Alphalens AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Copilot-style chat for financial filings using your existing RAG pipeline.</div>',
        unsafe_allow_html=True,
    )

    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "thread_counter" not in st.session_state:
        st.session_state["thread_counter"] = 1
    if not str(st.session_state.get("thread_id", "")).strip():
        st.session_state["thread_id"] = str(st.session_state["thread_counter"])
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = ""

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
        st.text_input(
            "Thread ID (numeric, ascending)",
            value=str(st.session_state["thread_id"]),
            disabled=True,
            help="Auto-managed as 1, 2, 3...",
        )
        st.caption(f"Current thread sequence: {st.session_state['thread_counter']}")
        if st.button("New Chat", use_container_width=True):
            st.session_state["history"] = []
            st.session_state["thread_counter"] = int(st.session_state["thread_counter"]) + 1
            st.session_state["thread_id"] = str(st.session_state["thread_counter"])
            st.rerun()
        if chat_history_store is not None:
            st.caption("Chat history DB: connected")
        else:
            st.caption("Chat history DB: not connected")
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

    for idx, item in enumerate(st.session_state["history"]):
        with st.chat_message("user"):
            st.write(item["query"])
        with st.chat_message("assistant"):
            _render_assistant_turn(
                item,
                sources_limit,
                idx,
                show_copilot=(idx == len(st.session_state["history"]) - 1),
            )

    centered_query = ""
    centered_submit = False
    with st.container():
        st.markdown('<div class="search-dock">', unsafe_allow_html=True)
        st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
        c_left, c_mid, c_right = st.columns([1, 3.8, 1])
        with c_mid:
            with st.form("center_search_form", clear_on_submit=True):
                centered_query = st.text_input(
                    "Ask",
                    value="",
                    label_visibility="collapsed",
                    placeholder="Ask anything about filings...",
                )
                centered_submit = st.form_submit_button("Search", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if centered_submit and centered_query.strip():
        query = centered_query.strip()
    elif str(st.session_state.get("pending_query", "")).strip():
        query = str(st.session_state.get("pending_query", "")).strip()
        st.session_state["pending_query"] = ""
    else:
        query = ""

    if not query:
        return

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        started = time.time()
        status_placeholder = st.empty()
        _render_loading_bot(status_placeholder)
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
                    "thread_id": str(st.session_state.get("thread_id")),
                    "mode": mode,
                    "llm_model": selected_model,
                    "llm_selection_mode": selection_mode,
                    "show_plot": _query_needs_plot(query.strip()),
                    "elapsed_seconds": elapsed,
                    "results": results,
                    "generation": generation,
                }
                st.session_state["history"].append(run_result)
                _safe_store_chat_history(
                    query=run_result["query"],
                    answer=str(generation.get("answer") or ""),
                    thread_id=run_result.get("thread_id"),
                    metadata={
                        "mode": mode,
                        "llm_model": selected_model,
                        "llm_selection_mode": selection_mode,
                        "results_count": len(results),
                        "elapsed_seconds": elapsed,
                        "generation_mode": generation.get("generation_mode"),
                        "llm_error": generation.get("llm_error"),
                        "fallback_notice": fallback_notice,
                    },
                )
                if fallback_notice:
                    st.warning(fallback_notice)
                status_placeholder.empty()
                st.rerun()
            except Exception as exc:
                status_placeholder.empty()
                st.error(f"Run failed: {exc}")


if __name__ == "__main__":
    main()
