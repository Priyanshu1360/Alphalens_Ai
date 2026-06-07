import argparse

from src.utils.config import Config
from src.llm.llm_client import generate_answer
from src.retrieval.retriever import dense_search, hybrid_search, sparse_search


def _default_mode():
    mode = str(Config.RETRIEVER_DEFAULT_MODE).lower()
    return mode if mode in {"hybrid", "dense", "sparse"} else "hybrid"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive RAG chat over indexed filings.",
    )
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
        help="Number of sources to show",
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


def _retrieve(query, args):
    if args.mode == "dense":
        return dense_search(query, limit=args.limit, rerank=not args.no_rerank)
    if args.mode == "sparse":
        return sparse_search(query, limit=args.limit, rerank=not args.no_rerank)
    return hybrid_search(
        query,
        limit=args.limit,
        prefetch_limit=args.prefetch_limit,
        rerank=not args.no_rerank,
    )


def _print_sources(results, sources_limit):
    show_count = max(0, sources_limit)
    if show_count == 0:
        return

    top = results[:show_count]
    if not top:
        print("Sources: none")
        return

    print("Sources:")
    for item in top:
        rank = item.get("rank")
        company = item.get("company")
        file_name = item.get("file_name")
        year = item.get("year")
        quarter = item.get("quarter")
        score = item.get("score")
        print(
            f"  [{rank}] {company} | {file_name} | year={year} "
            f"quarter={quarter} score={score}"
        )


def main():
    args = _parse_args()
    print("RAG chat started. Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("\nQ: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        try:
            results = _retrieve(query, args)
            generation = generate_answer(query, results)
        except Exception as exc:
            print(f"A: Error: {exc}")
            continue

        answer = generation.get("answer") or "No answer generated."
        print(f"A: {answer}")
        if generation.get("llm_error"):
            print(f"LLM error: {generation.get('llm_error')}")
        _print_sources(results, args.sources_limit)


if __name__ == "__main__":
    main()
