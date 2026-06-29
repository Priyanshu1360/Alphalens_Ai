import re
import warnings
import concurrent.futures

from qdrant_client import models

from src.utils.config import Config
from src.embeddings.embedder import (
    close_qdrant_client,
    get_dense_query_vector,
    get_qdrant_client,
    get_sparse_query_vector,
)


def _fusion_from_config():
    value = str(Config.HYBRID_FUSION).upper()
    if value == "DBSF":
        return models.Fusion.DBSF
    return models.Fusion.RRF


def _format_point(point):
    payload = point.payload or {}
    text = str(payload.get("text", ""))
    snippet = text[: Config.RETRIEVAL_SNIPPET_CHARS]

    return {
        "id": point.id,
        "score": float(point.score),
        "retrieval_score": float(point.score),
        "chunk_type": payload.get("chunk_type"),
        "company": payload.get("company"),
        "report_type": payload.get("report_type"),
        "year": payload.get("year"),
        "quarter": payload.get("quarter"),
        "coherence_score": payload.get("coherence_score"),
        "chunk_quality": payload.get("chunk_quality"),
        "file_name": payload.get("file_name"),
        "snippet": snippet,
        "payload": payload,
    }


def _hybrid_prefetch(query_text, prefetch_limit):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(get_dense_query_vector, query_text)
        sparse_future = executor.submit(get_sparse_query_vector, query_text)
        
        dense_vector = dense_future.result()
        sparse_vector = sparse_future.result()

    return [
        models.Prefetch(
            query=dense_vector,
            using=Config.QDRANT_DENSE_VECTOR_NAME,
            limit=prefetch_limit,
        ),
        models.Prefetch(
            query=sparse_vector,
            using=Config.QDRANT_SPARSE_VECTOR_NAME,
            limit=prefetch_limit,
        ),
    ]


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
QUARTER_PATTERN = re.compile(r"\bq([1-4])\b", re.IGNORECASE)
_FILTER_INDEXES_READY = False
_BOOTSTRAP_ATTEMPTED = False


def _tokenize(text):
    return TOKEN_PATTERN.findall((text or "").lower())


def _detect_companies(query_text):
    tokens = _tokenize(query_text)
    aliases = Config.RETRIEVER_COMPANY_ALIASES or {}
    seen = set()
    companies = []

    for token in tokens:
        company = aliases.get(token)
        if not company or company in seen:
            continue
        seen.add(company)
        companies.append(company)

    return companies


def _detect_report_type(query_text):
    normalized = (query_text or "").lower()
    if "10-k" in normalized or "10k" in normalized:
        return "10-k"
    if "10-q" in normalized or "10q" in normalized:
        return "10-q"
    if "8-k" in normalized or "8k" in normalized:
        return "8-k"
    return None


def _detect_year(query_text):
    match = YEAR_PATTERN.search(query_text or "")
    if not match:
        return None
    return match.group(1)


def _detect_quarter(query_text):
    match = QUARTER_PATTERN.search(query_text or "")
    if not match:
        return None
    return match.group(1)


def _build_query_filter(query_text):
    conditions = []

    companies = _detect_companies(query_text)
    if len(companies) == 1:
        conditions.append(
            models.FieldCondition(
                key="company",
                match=models.MatchValue(value=companies[0]),
            )
        )
    elif len(companies) > 1:
        conditions.append(
            models.FieldCondition(
                key="company",
                match=models.MatchAny(any=companies),
            )
        )

    year = _detect_year(query_text)
    if year:
        conditions.append(
            models.FieldCondition(
                key="year",
                match=models.MatchValue(value=year),
            )
        )

    quarter = _detect_quarter(query_text)
    if quarter:
        conditions.append(
            models.FieldCondition(
                key="quarter",
                match=models.MatchValue(value=quarter),
            )
        )

    report_type = _detect_report_type(query_text)
    if report_type:
        conditions.append(
            models.FieldCondition(
                key="report_type",
                match=models.MatchValue(value=report_type),
            )
        )

    if not conditions:
        return None
    return models.Filter(must=conditions)


def _ensure_filter_indexes(client):
    global _FILTER_INDEXES_READY
    if _FILTER_INDEXES_READY:
        return

    index_fields = Config.RETRIEVER_FILTER_INDEX_FIELDS or []
    for field_name in index_fields:
        try:
            client.create_payload_index(
                collection_name=Config.QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception:
            continue

    _FILTER_INDEXES_READY = True


def _normalize(values):
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def _quality_bonus(label):
    if label == "high":
        return 1.0
    if label == "medium":
        return 0.6
    if label == "low":
        return 0.2
    return 0.0


_CROSS_ENCODER_MODEL = None
_CROSS_ENCODER_UNAVAILABLE = False


def _get_cross_encoder_model():
    global _CROSS_ENCODER_MODEL, _CROSS_ENCODER_UNAVAILABLE

    if _CROSS_ENCODER_MODEL is not None:
        return _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_UNAVAILABLE:
        return None

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        warnings.warn(
            "sentence-transformers is required for cross-encoder reranking. "
            "Falling back to heuristic reranking.",
            RuntimeWarning,
        )
        _CROSS_ENCODER_UNAVAILABLE = True
        return None

    try:
        _CROSS_ENCODER_MODEL = CrossEncoder(
            Config.RERANK_MODEL,
            device=Config.RERANK_DEVICE,
            max_length=Config.RERANK_MAX_LENGTH,
        )
    except Exception as exc:
        warnings.warn(
            f"Unable to load cross-encoder reranker '{Config.RERANK_MODEL}': {exc}. "
            "Falling back to heuristic reranking.",
            RuntimeWarning,
        )
        _CROSS_ENCODER_UNAVAILABLE = True
        return None

    return _CROSS_ENCODER_MODEL


def _heuristic_rerank_results(query_text, results, limit):
    if not results:
        return []

    query_tokens = set(_tokenize(query_text))
    retrieval_norm = _normalize([item.get("retrieval_score", 0.0) for item in results])

    reranked = []
    for index, item in enumerate(results):
        snippet_tokens = set(_tokenize(item.get("snippet", "")))
        if query_tokens and snippet_tokens:
            lexical_overlap = len(query_tokens & snippet_tokens) / len(query_tokens)
        else:
            lexical_overlap = 0.0

        coherence = item.get("coherence_score")
        if isinstance(coherence, (int, float)):
            coherence_value = float(coherence)
        else:
            coherence_value = 0.0

        quality_value = _quality_bonus(item.get("chunk_quality"))
        rerank_score = (
            Config.RERANK_RETRIEVAL_WEIGHT * retrieval_norm[index]
            + Config.RERANK_LEXICAL_WEIGHT * lexical_overlap
            + Config.RERANK_COHERENCE_WEIGHT * coherence_value
            + Config.RERANK_QUALITY_WEIGHT * quality_value
        )

        updated = dict(item)
        updated["rerank_score"] = round(rerank_score, 6)
        updated["score"] = round(rerank_score, 6)
        updated["rerank_backend"] = "heuristic"
        reranked.append(updated)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    final = _deduplicate_results(reranked)[:limit]
    for rank, item in enumerate(final, start=1):
        item["rank"] = rank
    return final


def _cross_encoder_rerank_results(query_text, results, limit):
    if not results:
        return []

    model = _get_cross_encoder_model()
    if model is None:
        if Config.RERANK_FALLBACK_TO_HEURISTIC:
            return _heuristic_rerank_results(query_text, results, limit)
        return results[:limit]

    pairs = []
    for item in results:
        payload = item.get("payload") or {}
        candidate_text = str(payload.get("text") or item.get("snippet", ""))
        pairs.append((query_text, candidate_text))

    try:
        scores = model.predict(
            pairs,
            batch_size=Config.RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
    except TypeError:
        scores = model.predict(pairs, batch_size=Config.RERANK_BATCH_SIZE)
    except Exception as exc:
        warnings.warn(
            f"Cross-encoder reranking failed ({exc}). Falling back to heuristic reranking.",
            RuntimeWarning,
        )
        if Config.RERANK_FALLBACK_TO_HEURISTIC:
            return _heuristic_rerank_results(query_text, results, limit)
        return results[:limit]

    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    elif isinstance(scores, (int, float)):
        scores = [float(scores)]
    else:
        scores = list(scores)

    scores = [float(score) for score in scores]
    normalized_scores = _normalize(scores)

    reranked = []
    for item, raw_score, normalized_score in zip(results, scores, normalized_scores):
        updated = dict(item)
        updated["cross_encoder_score"] = round(raw_score, 6)
        updated["rerank_score"] = round(normalized_score, 6)
        updated["score"] = round(normalized_score, 6)
        updated["rerank_backend"] = "cross_encoder"
        reranked.append(updated)

    reranked.sort(key=lambda item: item["cross_encoder_score"], reverse=True)
    final = _deduplicate_results(reranked)[:limit]
    for rank, item in enumerate(final, start=1):
        item["rank"] = rank
    return final


def _deduplicate_results(results, threshold=0.5):
    if not results:
        return []
    
    unique_results = []
    seen_tokens = []
    
    for item in results:
        text = str(item.get("payload", {}).get("text") or item.get("snippet", ""))
        tokens = set(_tokenize(text))
        
        is_duplicate = False
        if tokens:
            for seen in seen_tokens:
                # Jaccard similarity
                overlap = len(tokens & seen) / len(tokens | seen)
                if overlap > threshold:
                    is_duplicate = True
                    break
                    
        if not is_duplicate:
            unique_results.append(item)
            seen_tokens.append(tokens)
            
    return unique_results


def _rerank_results(query_text, results, limit, use_rerank):
    if not use_rerank or not results:
        return _deduplicate_results(results)[:limit]

    backend = str(Config.RERANK_BACKEND or "cross_encoder").strip().lower()
    if backend in {"cross_encoder", "cross-encoder", "crossencoder"}:
        return _cross_encoder_rerank_results(query_text, results, limit)
    return _heuristic_rerank_results(query_text, results, limit)


def _ensure_collection_exists(client):
    global _BOOTSTRAP_ATTEMPTED

    if not Config.QDRANT_COLLECTION:
        raise ValueError("Qdrant collection name is missing in config")
    try:
        exists = client.collection_exists(Config.QDRANT_COLLECTION)
    except Exception:
        exists = False

    if (
        not exists
        and Config.AUTO_BOOTSTRAP_INDEX_ON_QUERY
        and not _BOOTSTRAP_ATTEMPTED
    ):
        _BOOTSTRAP_ATTEMPTED = True
        _bootstrap_index()
        try:
            exists = client.collection_exists(Config.QDRANT_COLLECTION)
        except Exception:
            exists = False

        if not exists:
            close_qdrant_client(client)
            client = get_qdrant_client()
            try:
                exists = client.collection_exists(Config.QDRANT_COLLECTION)
            except Exception:
                exists = False

    if not exists:
        raise ValueError(
            f"Collection '{Config.QDRANT_COLLECTION}' not found. "
            "Run ingestion first with `python main.py` on the same Qdrant instance."
        )

    return client


def _bootstrap_index():
    from src.chunking.chunker import create_chunks
    from src.utils.cleaner import clean_chunks
    from src.embeddings.embedder import embed_and_store_chunks
    from src.ingestion.pdf_loader import load_pdfs

    warnings.warn(
        "Collection missing. Running one-time ingestion bootstrap before retrieval.",
        RuntimeWarning,
    )
    docs = load_pdfs()
    chunks = create_chunks(docs)
    cleaned = clean_chunks(chunks)
    embed_and_store_chunks(cleaned)


def hybrid_search(query_text, limit=None, prefetch_limit=None, rerank=None):
    if not query_text or not str(query_text).strip():
        raise ValueError("query_text cannot be empty")

    if limit is None:
        limit = Config.RETRIEVAL_TOP_K
    if prefetch_limit is None:
        prefetch_limit = Config.HYBRID_PREFETCH_LIMIT
    if rerank is None:
        rerank = Config.RERANK_ENABLED

    fetch_limit = max(limit, Config.RERANK_CANDIDATES) if rerank else limit
    query_filter = _build_query_filter(query_text)

    client = get_qdrant_client()
    try:
        client = _ensure_collection_exists(client)
        if query_filter is not None:
            _ensure_filter_indexes(client)
        response = client.query_points(
            collection_name=Config.QDRANT_COLLECTION,
            prefetch=_hybrid_prefetch(query_text, prefetch_limit),
            query=models.FusionQuery(fusion=_fusion_from_config()),
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,
        )
        formatted = [_format_point(point) for point in response.points]
        return _rerank_results(query_text, formatted, limit, rerank)
    finally:
        close_qdrant_client(client)


def dense_search(query_text, limit=None, rerank=None):
    if not query_text or not str(query_text).strip():
        raise ValueError("query_text cannot be empty")

    if limit is None:
        limit = Config.RETRIEVAL_TOP_K
    if rerank is None:
        rerank = Config.RERANK_ENABLED

    fetch_limit = max(limit, Config.RERANK_CANDIDATES) if rerank else limit
    query_filter = _build_query_filter(query_text)

    client = get_qdrant_client()
    try:
        client = _ensure_collection_exists(client)
        if query_filter is not None:
            _ensure_filter_indexes(client)
        response = client.query_points(
            collection_name=Config.QDRANT_COLLECTION,
            query=get_dense_query_vector(query_text),
            using=Config.QDRANT_DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,
        )
        formatted = [_format_point(point) for point in response.points]
        return _rerank_results(query_text, formatted, limit, rerank)
    finally:
        close_qdrant_client(client)


def sparse_search(query_text, limit=None, rerank=None):
    if not query_text or not str(query_text).strip():
        raise ValueError("query_text cannot be empty")

    if limit is None:
        limit = Config.RETRIEVAL_TOP_K
    if rerank is None:
        rerank = Config.RERANK_ENABLED

    fetch_limit = max(limit, Config.RERANK_CANDIDATES) if rerank else limit
    query_filter = _build_query_filter(query_text)

    client = get_qdrant_client()
    try:
        client = _ensure_collection_exists(client)
        if query_filter is not None:
            _ensure_filter_indexes(client)
        response = client.query_points(
            collection_name=Config.QDRANT_COLLECTION,
            query=get_sparse_query_vector(query_text),
            using=Config.QDRANT_SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,
        )
        formatted = [_format_point(point) for point in response.points]
        return _rerank_results(query_text, formatted, limit, rerank)
    finally:
        close_qdrant_client(client)
