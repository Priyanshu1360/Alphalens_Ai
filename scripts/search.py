import argparse
import json

from src.utils.config import Config
from src.retrieval.retriever import dense_search, hybrid_search, sparse_search


def _default_mode():
    mode = str(Config.RETRIEVER_DEFAULT_MODE).lower()
    return mode if mode in {"hybrid", "dense", "sparse"} else "hybrid"


def _default_limit():
    return None if Config.SEARCH_DEFAULT_LIMIT <= 0 else Config.SEARCH_DEFAULT_LIMIT


def _parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval on Qdrant.")
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
        default=_default_limit(),
        help="Max number of results",
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


def main():
    args = _parse_args()

    if args.mode == "dense":
        results = dense_search(
            args.query,
            limit=args.limit,
            rerank=not args.no_rerank,
        )
    elif args.mode == "sparse":
        results = sparse_search(
            args.query,
            limit=args.limit,
            rerank=not args.no_rerank,
        )
    else:
        results = hybrid_search(
            args.query,
            limit=args.limit,
            prefetch_limit=args.prefetch_limit,
            rerank=not args.no_rerank,
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
