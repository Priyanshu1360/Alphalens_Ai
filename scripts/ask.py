import argparse
import json

from src.utils.config import Config
from src.llm.llm_client import generate_answer
from src.retrieval.retriever import dense_search, hybrid_search, sparse_search


def _default_mode():
    mode = str(Config.RETRIEVER_DEFAULT_MODE).lower()
    return mode if mode in {"hybrid", "dense", "sparse"} else "hybrid"


def _parse_args():
    parser = argparse.ArgumentParser(description="Ask a question over indexed filings.")
    parser.add_argument("query", help="Natural language query text")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "dense", "sparse"],
        default=_default_mode(),
        help="Retrieval mode",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=Config.ASK_DEFAULT_LIMIT,
        help="Max number of retrieval results",
    )
    parser.add_argument(
        "--sources-limit",
        type=int,
        default=Config.ASK_DEFAULT_SOURCES_LIMIT,
        help="Number of sources to include in output",
    )
    parser.add_argument(
        "--prefetch-limit",
        type=int,
        default=None,
        help="Prefetch limit for hybrid mode",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking step",
    )
    return parser.parse_args()


def _retrieve(args):
    if args.mode == "dense":
        return dense_search(args.query, limit=args.limit, rerank=not args.no_rerank)
    if args.mode == "sparse":
        return sparse_search(args.query, limit=args.limit, rerank=not args.no_rerank)
    return hybrid_search(
        args.query,
        limit=args.limit,
        prefetch_limit=args.prefetch_limit,
        rerank=not args.no_rerank,
    )


def main():
    args = _parse_args()
    results = _retrieve(args)
    generation = generate_answer(args.query, results)
    output = {
        "query": args.query,
        "mode": args.mode,
        "results_count": len(results),
        "generation_mode": generation.get("generation_mode"),
        "answer": generation.get("answer"),
        "llm_error": generation.get("llm_error"),
        "sources": [
            {
                "rank": item.get("rank"),
                "company": item.get("company"),
                "file_name": item.get("file_name"),
                "year": item.get("year"),
                "quarter": item.get("quarter"),
                "score": item.get("score"),
                "rerank_score": item.get("rerank_score"),
            }
            for item in results[: max(0, args.sources_limit)]
        ],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
