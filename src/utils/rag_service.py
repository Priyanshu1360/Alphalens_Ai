from typing import Any, Dict, List, Optional

from src.utils.config import Config
from src.llm.llm_client import generate_answer, generate_conversational_answer
from src.utils.classifier import classify_intent
from src.retrieval.retriever import dense_search, hybrid_search, sparse_search


def default_mode() -> str:
    mode = str(Config.RETRIEVER_DEFAULT_MODE).lower()
    return mode if mode in {"hybrid", "dense", "sparse"} else "hybrid"


def retrieve_documents(
    query: str,
    mode: Optional[str] = None,
    limit: Optional[int] = None,
    prefetch_limit: Optional[int] = None,
    rerank: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    retrieval_mode = (mode or default_mode()).lower()

    if retrieval_mode == "dense":
        return dense_search(query, limit=limit, rerank=rerank)
    if retrieval_mode == "sparse":
        return sparse_search(query, limit=limit, rerank=rerank)
    return hybrid_search(
        query,
        limit=limit,
        prefetch_limit=prefetch_limit,
        rerank=rerank,
    )


def run_rag_query(
    query: str,
    mode: Optional[str] = None,
    limit: Optional[int] = None,
    prefetch_limit: Optional[int] = None,
    rerank: Optional[bool] = None,
) -> Dict[str, Any]:
    intent = classify_intent(query)
    if intent == "conversational":
        generation = generate_conversational_answer(query)
        return {
            "query": query,
            "mode": (mode or default_mode()).lower(),
            "results": [],
            "answer": generation.get("answer"),
            "generation_mode": generation.get("generation_mode"),
            "llm_error": generation.get("llm_error"),
        }

    results = retrieve_documents(
        query=query,
        mode=mode,
        limit=limit,
        prefetch_limit=prefetch_limit,
        rerank=rerank,
    )
    generation = generate_answer(query, results)
    return {
        "query": query,
        "mode": (mode or default_mode()).lower(),
        "results": results,
        "answer": generation.get("answer"),
        "generation_mode": generation.get("generation_mode"),
        "llm_error": generation.get("llm_error"),
    }
