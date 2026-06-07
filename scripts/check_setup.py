import json
import sys

from src.utils.config import Config
from src.embeddings.embedder import close_qdrant_client, get_qdrant_client


def _safe_collection_count(client, collection_name):
    try:
        return client.count(collection_name=collection_name, exact=True).count
    except Exception as exc:
        return f"unavailable: {exc}"


def _check_qdrant():
    try:
        client = get_qdrant_client()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    try:
        collection_names = [c.name for c in client.get_collections().collections]
        exists = bool(Config.QDRANT_COLLECTION and Config.QDRANT_COLLECTION in collection_names)
        points = (
            _safe_collection_count(client, Config.QDRANT_COLLECTION)
            if exists
            else 0
        )
        return {
            "status": "ok",
            "endpoint": Config.QDRANT_PATH or str(Config.QDRANT_LOCAL_PATH),
            "target_collection": Config.QDRANT_COLLECTION,
            "target_collection_exists": exists,
            "target_collection_points": points,
            "available_collections": collection_names,
        }
    finally:
        close_qdrant_client(client)


def build_report():
    return {
        "pipeline": {
            "steps": [
                "load_pdfs",
                "create_chunks",
                "clean_chunks",
                "embed_and_store_chunks",
                "hybrid_search/dense_search/sparse_search",
                "generate_answer",
            ]
        },
        "models": {
            "embedding_backend": Config.EMBEDDING_BACKEND,
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.LLM_MODEL,
            "llm_enabled": Config.LLM_ENABLED,
        },
        "qdrant": _check_qdrant(),
    }


def main():
    report = build_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["qdrant"].get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
