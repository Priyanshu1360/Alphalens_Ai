import argparse
import json
import time

from src.config import Config
from src.generator import generate_answer
from src.retriever import hybrid_search


def _evaluate_query(query, limit):
    rerank_on = hybrid_search(query, limit=limit, rerank=True)
    rerank_off = hybrid_search(query, limit=limit, rerank=False)

    gen = generate_answer(query, rerank_on)

    def _top(result):
        if not result:
            return None
        item = result[0]
        return {
            "company": item.get("company"),
            "file_name": item.get("file_name"),
            "score": item.get("score"),
            "rerank_score": item.get("rerank_score"),
        }

    return {
        "query": query,
        "rerank_on_top1": _top(rerank_on),
        "rerank_off_top1": _top(rerank_off),
        "generation_mode": gen.get("generation_mode"),
        "llm_error": gen.get("llm_error"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval reranking and generation status.")
    parser.add_argument(
        "--limit",
        type=int,
        default=Config.EVALUATE_DEFAULT_LIMIT,
        help="Retrieval limit per query",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Custom query (repeat --query multiple times)",
    )
    args = parser.parse_args()

    queries = args.queries if args.queries else Config.EVALUATE_DEFAULT_QUERIES
    started = time.time()
    rows = []

    for query in queries:
        rows.append(_evaluate_query(query, args.limit))

    report = {
        "queries_count": len(queries),
        "duration_seconds": round(time.time() - started, 2),
        "rows": rows,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
