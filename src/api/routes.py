import os
import time
import base64
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.agent_graph import AgentWorkflow
from src.utils.cache_memory import ExactMatchCache, SemanticCache
from src.utils.config import Config
from src.ingestion.ingestion_service import ingest_all_documents, ingest_single_pdf
from src.utils.mcp_client import MCPClient
from src.utils.rag_service import default_mode, run_rag_query

try:
    from src.utils.chat_history_store import ChatHistoryStore
except Exception:  # pragma: no cover - optional dependency fallback
    ChatHistoryStore = None

import numpy as np

def _sanitize_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, (list, tuple, set)):
        return [_sanitize_dict(i) for i in d]
    elif type(d).__module__ == 'numpy':
        return d.item()
    return d

app = FastAPI(title="Finance RAG Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOGGER = logging.getLogger("finance_rag_api")

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
ENABLE_SEMANTIC_CACHE = os.getenv("ENABLE_SEMANTIC_CACHE", "false").lower() in {
    "1",
    "true",
    "yes",
}

exact_cache = ExactMatchCache(ttl_seconds=CACHE_TTL_SECONDS)
semantic_cache = SemanticCache(
    ttl_seconds=CACHE_TTL_SECONDS,
    similarity_threshold=Config.SEMANTIC_CACHE_THRESHOLD,
)


mcp_client = MCPClient(
    server_url=os.getenv("MCP_SERVER_URL", ""),
    timeout_seconds=float(os.getenv("MCP_TIMEOUT_SECONDS", "5")),
    enable_mock_tools=os.getenv("MCP_ENABLE_MOCK_TOOLS", "true").lower()
    in {"1", "true", "yes"},
    cache_ttl_seconds=int(os.getenv("MCP_CACHE_TTL_SECONDS", "300")),
)
agent_workflow = AgentWorkflow(exact_cache=exact_cache, semantic_cache=semantic_cache, mcp_client=mcp_client)

chat_history_store = None
if ChatHistoryStore and Config.CHAT_HISTORY_ENABLED and Config.POSTGRES_URL:
    try:
        chat_history_store = ChatHistoryStore(Config.POSTGRES_URL)
    except Exception as exc:
        LOGGER.warning("Chat history DB disabled: %s", exc)


def _safe_store_chat_history(
    *,
    route: str,
    query: str,
    answer: str,
    thread_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    if chat_history_store is None:
        return
        
    def _do_store():
        try:
            chat_history_store.add_entry(
                route=route,
                query=query,
                answer=answer,
                thread_id=thread_id,
                metadata=metadata or {},
            )
        except Exception as exc:
            LOGGER.warning("Failed to persist chat history: %s", exc)

    threading.Thread(target=_do_store, daemon=True).start()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field(default_factory=default_mode)
    limit: int = Field(default=Config.ASK_DEFAULT_LIMIT, ge=1, le=50)
    prefetch_limit: int = Field(default=Config.HYBRID_PREFETCH_LIMIT, ge=1, le=200)
    rerank: bool = True
    thread_id: Optional[str] = None


class AgentQueryRequest(QueryRequest):
    thread_id: Optional[str] = None


class IngestRequest(BaseModel):
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    company: str = ""
    pdf_base64: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/status")
def status() -> Dict[str, Any]:
    # We always return is_ready=True here because FastEmbed manages its own internal cache
    # differently than the standard HuggingFace Hub directory.
    return {
        "status": "ok",
        "model": os.getenv("EMBEDDING_MODEL", "fastembed"),
        "downloaded_mb": 1300,
        "is_ready": True
    }


@app.get("/mcp-tools")
def list_mcp_tools() -> Dict[str, Any]:
    return {"tools": mcp_client.discover_tools()}


@app.get("/cache/stats")
def cache_stats() -> Dict[str, Any]:
    payload = {"exact_match_cache": exact_cache.stats()}
    if semantic_cache is not None:
        payload["semantic_cache"] = semantic_cache.stats()
    return payload


@app.get("/chat-history")
def get_chat_history(thread_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    if chat_history_store is None:
        return {"enabled": False, "items": []}
    try:
        items = chat_history_store.list_entries(thread_id=thread_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"enabled": True, "items": items}


@app.post("/ingest")
def ingest(payload: Optional[IngestRequest] = None):
    try:
        payload = payload or IngestRequest()
        company_name = (payload.company or "uploaded").strip().lower()

        if payload.pdf_base64:
            target_dir = Path(Config.PDF_BASE_PATH).resolve() / company_name
            target_dir.mkdir(parents=True, exist_ok=True)
            file_name = Path(payload.file_name or "uploaded.pdf").name
            target_path = target_dir / file_name
            target_path.write_bytes(base64.b64decode(payload.pdf_base64))
            result = ingest_single_pdf(str(target_path), company=company_name)
            ingest_mode = "single_pdf_base64"
        elif payload.file_path:
            source_path = Path(payload.file_path).resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")

            target_dir = Path(Config.PDF_BASE_PATH).resolve() / company_name
            target_dir.mkdir(parents=True, exist_ok=True)
            file_name = Path(payload.file_name or source_path.name).name
            target_path = target_dir / file_name
            if source_path != target_path:
                target_path.write_bytes(source_path.read_bytes())
            else:
                target_path = source_path
            result = ingest_single_pdf(str(target_path), company=company_name)
            ingest_mode = "single_pdf_path"
        else:
            result = ingest_all_documents()
            ingest_mode = "full_rebuild"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    exact_cache.invalidate()
    if semantic_cache is not None:
        semantic_cache.invalidate()

    return {
        "status": "ok",
        "mode": ingest_mode,
        "cache_invalidated": True,
        **result,
    }


@app.post("/query")
def query(payload: QueryRequest) -> Dict[str, Any]:
    started = time.perf_counter()
    from src.utils.guardrails import apply_input_guardrails
    try:
        query_text = apply_input_guardrails(payload.query.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
        
    if not query_text:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    cached = exact_cache.get(query_text)
    if cached:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        response_payload = {
            "query": query_text,
            "mode": payload.mode,
            "cache_hit": True,
            "cache_type": "exact_match",
            "answer": cached.get("answer"),
            "results": cached.get("docs", []),
            "elapsed_ms": elapsed_ms,
        }
        _safe_store_chat_history(
            route="query",
            query=query_text,
            answer=str(response_payload.get("answer") or ""),
            thread_id=payload.thread_id,
            metadata={
                "mode": payload.mode,
                "cache_hit": True,
                "cache_type": "exact_match",
                "elapsed_ms": elapsed_ms,
            },
        )
        return response_payload

    if semantic_cache is not None:
        semantic_cached = semantic_cache.get(query_text)
        if semantic_cached:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            exact_cache.set(
                query_text,
                semantic_cached.get("answer", ""),
                semantic_cached.get("docs", []),
            )
            response_payload = {
                "query": query_text,
                "mode": payload.mode,
                "cache_hit": True,
                "cache_type": "semantic",
                "semantic_similarity": semantic_cached.get("semantic_similarity"),
                "answer": semantic_cached.get("answer"),
                "results": semantic_cached.get("docs", []),
                "elapsed_ms": elapsed_ms,
            }
            _safe_store_chat_history(
                route="query",
                query=query_text,
                answer=str(response_payload.get("answer") or ""),
                thread_id=payload.thread_id,
                metadata={
                    "mode": payload.mode,
                    "cache_hit": True,
                    "cache_type": "semantic",
                    "semantic_similarity": response_payload.get("semantic_similarity"),
                    "elapsed_ms": elapsed_ms,
                },
            )
            return response_payload

    try:
        result = run_rag_query(
            query=query_text,
            mode=payload.mode,
            limit=payload.limit,
            prefetch_limit=payload.prefetch_limit,
            rerank=payload.rerank,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    from src.utils.classifier import classify_intent
    if classify_intent(query_text) != "conversational":
        exact_cache.set(query_text, result.get("answer", ""), result.get("results", []))
        if semantic_cache is not None:
            semantic_cache.set(
                query_text,
                result.get("answer", ""),
                result.get("results", []),
            )

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    response_payload = {
        **result,
        "cache_hit": False,
        "elapsed_ms": elapsed_ms,
    }
    _safe_store_chat_history(
        route="query",
        query=query_text,
        answer=str(response_payload.get("answer") or ""),
        thread_id=payload.thread_id,
        metadata={
            "mode": payload.mode,
            "cache_hit": False,
            "generation_mode": response_payload.get("generation_mode"),
            "results_count": len(response_payload.get("results", [])),
            "elapsed_ms": elapsed_ms,
        },
    )
    return _sanitize_dict(response_payload)


@app.post("/agent-query")
def agent_query(payload: AgentQueryRequest) -> Dict[str, Any]:
    started = time.perf_counter()
    from src.utils.guardrails import apply_input_guardrails
    try:
        query_text = apply_input_guardrails(payload.query.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
        
    if not query_text:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    try:
        state = agent_workflow.run(
            query=query_text,
            mode=payload.mode,
            limit=payload.limit,
            prefetch_limit=payload.prefetch_limit,
            rerank=payload.rerank,
            thread_id=payload.thread_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Write to caches if successful
    final_ans = state.get("final_answer", "")
    docs = state.get("retrieved_docs", [])
    
    from src.utils.classifier import classify_intent
    intent = classify_intent(query_text)
    
    if final_ans and intent != "conversational":
        exact_cache.set(query_text, final_ans, docs)
        if semantic_cache is not None:
            semantic_cache.set(query_text, final_ans, docs)

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    route_reason = "conversational" if intent == "conversational" else state.get("route_reason")
    
    response_payload = {
        "query": query_text,
        "retrieved_docs": state.get("retrieved_docs", []),
        "cache_hit": bool(state.get("cache_hit", False)),
        "mcp_results": state.get("mcp_results", []),
        "final_answer": state.get("final_answer", ""),
        "route_reason": route_reason,
        "run_log": state.get("run_log", []),
        "elapsed_ms": elapsed_ms,
        "langgraph_enabled": agent_workflow.langgraph_enabled,
    }
    _safe_store_chat_history(
        route="agent-query",
        query=query_text,
        answer=str(response_payload.get("final_answer") or ""),
        thread_id=payload.thread_id,
        metadata={
            "mode": payload.mode,
            "cache_hit": response_payload.get("cache_hit"),
            "retrieved_docs_count": len(response_payload.get("retrieved_docs", [])),
            "mcp_results_count": len(response_payload.get("mcp_results", [])),
            "langgraph_enabled": response_payload.get("langgraph_enabled"),
            "elapsed_ms": elapsed_ms,
        },
    )
    return _sanitize_dict(response_payload)
