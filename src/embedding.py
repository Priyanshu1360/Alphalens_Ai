import hashlib
import math
import re
import warnings
from collections import Counter
from pathlib import Path

from openai import OpenAI
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config import Config


_OPENAI_EMBEDDINGS_DISABLED = False
_SENTENCE_TRANSFORMER_MODEL = None
_SENTENCE_TRANSFORMER_DISABLED = False
_REMOTE_QDRANT_DISABLED = False
_IN_MEMORY_QDRANT_CLIENT = None
_FORCE_IN_MEMORY_QDRANT = False


def _get_openai_client():
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing in environment variables")
    kwargs = {"api_key": Config.OPENAI_API_KEY}
    if Config.OPENAI_BASE_URL:
        kwargs["base_url"] = Config.OPENAI_BASE_URL
    return OpenAI(**kwargs)


def _get_sentence_transformer_model():
    global _SENTENCE_TRANSFORMER_MODEL
    if _SENTENCE_TRANSFORMER_MODEL is not None:
        return _SENTENCE_TRANSFORMER_MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for EMBEDDING_BACKEND=sentence_transformers. "
            "Install dependencies from requirements.txt."
        ) from exc

    kwargs = {
        "device": Config.EMBEDDING_DEVICE,
        "trust_remote_code": Config.EMBEDDING_TRUST_REMOTE_CODE,
    }
    _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(
        Config.EMBEDDING_MODEL,
        **kwargs,
    )
    return _SENTENCE_TRANSFORMER_MODEL


def _embed_texts_sentence_transformers(texts, batch_size):
    model = _get_sentence_transformer_model()
    encoded = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=Config.EMBEDDING_NORMALIZE,
        show_progress_bar=False,
    )
    if hasattr(encoded, "tolist"):
        return encoded.tolist()
    return [list(vector) for vector in encoded]


def _resolve_local_qdrant_path():
    local_path = str(Config.QDRANT_LOCAL_PATH or "").strip()
    if not local_path or local_path == ":memory:":
        return ":memory:"

    resolved = Path(local_path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def _get_in_memory_qdrant_client():
    global _IN_MEMORY_QDRANT_CLIENT
    if _IN_MEMORY_QDRANT_CLIENT is None:
        _IN_MEMORY_QDRANT_CLIENT = QdrantClient(location=":memory:")
    return _IN_MEMORY_QDRANT_CLIENT


def _get_local_qdrant_client():
    try:
        local_path = _resolve_local_qdrant_path()
    except Exception as exc:
        if not Config.QDRANT_LOCAL_FALLBACK_TO_MEMORY_ON_ERROR:
            raise
        warnings.warn(
            f"Unable to prepare local Qdrant path ({exc}). "
            "Falling back to in-memory Qdrant for this run.",
            RuntimeWarning,
        )
        return _get_in_memory_qdrant_client()
    if local_path == ":memory:":
        return _get_in_memory_qdrant_client()

    try:
        return QdrantClient(path=local_path)
    except Exception as exc:
        if Config.QDRANT_LOCAL_RECOVERY_PATH_ON_ERROR:
            recovery_path = str(Path(f"{local_path}_recovery").resolve())
            try:
                Path(recovery_path).mkdir(parents=True, exist_ok=True)
                warnings.warn(
                    "Local Qdrant storage failed at "
                    f"'{local_path}' ({exc}). Retrying with recovery path '{recovery_path}'.",
                    RuntimeWarning,
                )
                return QdrantClient(path=recovery_path)
            except Exception:
                pass

        if not Config.QDRANT_LOCAL_FALLBACK_TO_MEMORY_ON_ERROR:
            raise
        warnings.warn(
            "Local Qdrant storage failed at "
            f"'{local_path}' ({exc}). Falling back to in-memory Qdrant for this run.",
            RuntimeWarning,
        )
        return _get_in_memory_qdrant_client()


def _get_qdrant_client():
    global _REMOTE_QDRANT_DISABLED

    if _FORCE_IN_MEMORY_QDRANT:
        return _get_in_memory_qdrant_client()

    if Config.QDRANT_PATH and not _REMOTE_QDRANT_DISABLED:
        client = QdrantClient(
            url=Config.QDRANT_PATH,
            api_key=Config.QDRANT_API_KEY,
            timeout=Config.QDRANT_TIMEOUT_SECONDS,
        )
        try:
            client.get_collections()
            return client
        except Exception as exc:
            close_qdrant_client(client)
            if not Config.QDRANT_FALLBACK_TO_LOCAL_ON_ERROR:
                raise ConnectionError(
                    "Remote Qdrant connection failed. "
                    "Set QDRANT_FALLBACK_TO_LOCAL_ON_ERROR=true to allow automatic local fallback. "
                    f"Details: {exc}"
                ) from exc
            _REMOTE_QDRANT_DISABLED = True
            warnings.warn(
                f"Remote Qdrant connection failed, falling back to local storage: {exc}",
                RuntimeWarning,
            )

    return _get_local_qdrant_client()


def get_qdrant_client():
    return _get_qdrant_client()


def close_qdrant_client(client):
    if client is None:
        return
    if client is _IN_MEMORY_QDRANT_CLIENT:
        return
    try:
        client.close()
    except Exception:
        pass


def _chunk_list(items, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def _make_point_id(chunk):
    raw_id = "|".join(
        [
            str(chunk.get("file_path", "")),
            str(chunk.get("chunk_type", "")),
            str(chunk.get("chunk_index", "")),
            str(chunk.get("table_index", "")),
            str(chunk.get("text", "")),
        ]
    )
    digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def _normalize_vector(vector):
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _distance_from_config():
    mapping = {
        "COSINE": Distance.COSINE,
        "DOT": Distance.DOT,
        "EUCLID": Distance.EUCLID,
        "MANHATTAN": Distance.MANHATTAN,
    }
    return mapping.get(str(Config.QDRANT_DISTANCE).upper(), Distance.COSINE)


def _normalize_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return round(score, 4)


def _quality_from_score(score):
    if score >= Config.QUALITY_HIGH_THRESHOLD:
        return "high"
    if score >= Config.QUALITY_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


SPARSE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _build_sparse_vector(text):
    tokens = SPARSE_TOKEN_PATTERN.findall((text or "").lower())
    if not tokens:
        return SparseVector(indices=[], values=[])

    counts = Counter(tokens)
    max_count = max(counts.values())
    hashed = {}

    for token, count in counts.items():
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % Config.SPARSE_HASH_SPACE
        hashed[index] = hashed.get(index, 0.0) + (count / max_count)

    sorted_items = sorted(hashed.items(), key=lambda item: item[0])
    indices = [item[0] for item in sorted_items]
    values = [round(item[1], 6) for item in sorted_items]
    return SparseVector(indices=indices, values=values)


def _local_embed_text(text, dimension=None):
    if dimension is None:
        dimension = Config.LOCAL_EMBEDDING_DIMENSION
    vector = [0.0] * dimension
    tokens = text.lower().split()

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        vector[index] += sign * weight

    return _normalize_vector(vector)


def _local_embeddings_for_texts(texts):
    return [_local_embed_text(text) for text in texts]


def embed_texts(texts, batch_size=None):
    global _OPENAI_EMBEDDINGS_DISABLED, _SENTENCE_TRANSFORMER_DISABLED

    if not texts:
        return []
    if batch_size is None:
        batch_size = Config.EMBEDDING_BATCH_SIZE

    backend = str(Config.EMBEDDING_BACKEND or "").strip().lower()

    if backend in {"sentence_transformers", "sentence-transformers", "hf", "huggingface"}:
        if _SENTENCE_TRANSFORMER_DISABLED:
            return _local_embeddings_for_texts(texts)
        try:
            return _embed_texts_sentence_transformers(texts, batch_size)
        except Exception as exc:
            _SENTENCE_TRANSFORMER_DISABLED = True
            if not Config.EMBEDDING_FALLBACK_TO_LOCAL_ON_ERROR:
                raise
            warnings.warn(
                f"Sentence-transformers embedding failed ({exc}). "
                "Falling back to local hash embeddings for this run.",
                RuntimeWarning,
            )
            return _local_embeddings_for_texts(texts)

    if backend in {"local", "local_hash"}:
        return _local_embeddings_for_texts(texts)

    if backend not in {"openai", ""}:
        raise ValueError(
            f"Unsupported EMBEDDING_BACKEND='{Config.EMBEDDING_BACKEND}'. "
            "Use one of: sentence_transformers, openai, local_hash."
        )

    if (
        not Config.OPENAI_EMBEDDINGS_ENABLED
        or not Config.OPENAI_API_KEY
        or _OPENAI_EMBEDDINGS_DISABLED
    ):
        return _local_embeddings_for_texts(texts)

    client = _get_openai_client()
    embeddings = []
    warned_once = False

    for batch in _chunk_list(texts, batch_size):
        try:
            response = client.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)
        except openai.RateLimitError as exc:
            _OPENAI_EMBEDDINGS_DISABLED = True
            if not Config.OPENAI_FALLBACK_TO_LOCAL_ON_ERROR:
                raise
            if not warned_once:
                warnings.warn(
                    f"OpenAI embedding quota/rate-limit issue detected ({exc}). "
                    "Falling back to local embeddings for this run.",
                    RuntimeWarning,
                )
                warned_once = True
            embeddings.extend(_local_embed_text(text) for text in batch)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            _OPENAI_EMBEDDINGS_DISABLED = True
            if not Config.OPENAI_FALLBACK_TO_LOCAL_ON_ERROR:
                raise
            if not warned_once:
                warnings.warn(
                    f"OpenAI embedding auth/permission issue detected ({exc}). "
                    "Falling back to local embeddings for this run.",
                    RuntimeWarning,
                )
                warned_once = True
            embeddings.extend(_local_embed_text(text) for text in batch)
        except Exception as exc:
            if getattr(exc, "status_code", None) in {401, 403}:
                _OPENAI_EMBEDDINGS_DISABLED = True
            if not Config.OPENAI_FALLBACK_TO_LOCAL_ON_ERROR:
                raise
            if not warned_once:
                warnings.warn(
                    f"OpenAI embedding call failed ({exc}). "
                    "Falling back to local embeddings for this run.",
                    RuntimeWarning,
                )
                warned_once = True
            embeddings.extend(_local_embed_text(text) for text in batch)

    return embeddings


def get_dense_query_vector(query_text):
    if not query_text or not str(query_text).strip():
        raise ValueError("query_text cannot be empty")
    return embed_texts([query_text], batch_size=1)[0]


def get_sparse_query_vector(query_text):
    if not query_text or not str(query_text).strip():
        raise ValueError("query_text cannot be empty")
    return _build_sparse_vector(query_text)


def _extract_vector_size(vectors_config):
    if vectors_config is None:
        return None

    if hasattr(vectors_config, "size"):
        return vectors_config.size

    if isinstance(vectors_config, dict):
        if "size" in vectors_config:
            return vectors_config["size"]
        for value in vectors_config.values():
            nested_size = _extract_vector_size(value)
            if nested_size is not None:
                return nested_size

    return None


def _is_empty_vectors_config(vectors_config):
    return isinstance(vectors_config, dict) and not vectors_config


def _is_named_dense_config(vectors_config, expected_name, expected_size):
    if not isinstance(vectors_config, dict):
        return False
    dense_cfg = vectors_config.get(expected_name)
    if dense_cfg is None:
        return False
    return _extract_vector_size(dense_cfg) == expected_size


def _has_named_sparse_config(sparse_vectors_config, expected_name):
    if sparse_vectors_config is None:
        return False
    if isinstance(sparse_vectors_config, dict):
        return expected_name in sparse_vectors_config
    try:
        return expected_name in sparse_vectors_config
    except TypeError:
        return False


def _is_local_client(client):
    inner_client = getattr(client, "_client", None)
    if inner_client is None:
        return False
    return inner_client.__class__.__module__.startswith("qdrant_client.local")


def _create_collection_schema(client, vector_size):
    client.create_collection(
        collection_name=Config.QDRANT_COLLECTION,
        vectors_config={
            Config.QDRANT_DENSE_VECTOR_NAME: VectorParams(
                size=vector_size,
                distance=_distance_from_config(),
            )
        },
        sparse_vectors_config={
            Config.QDRANT_SPARSE_VECTOR_NAME: SparseVectorParams()
        },
    )


def _create_hybrid_collection(client, vector_size):
    global _FORCE_IN_MEMORY_QDRANT

    try:
        _create_collection_schema(client, vector_size)
        return client
    except Exception as exc:
        if not (_is_local_client(client) and Config.QDRANT_LOCAL_FALLBACK_TO_MEMORY_ON_ERROR):
            raise
        warnings.warn(
            "Local Qdrant collection creation failed "
            f"({exc}). Falling back to in-memory Qdrant for this run.",
            RuntimeWarning,
        )
        close_qdrant_client(client)
        _FORCE_IN_MEMORY_QDRANT = True
        memory_client = _get_in_memory_qdrant_client()
        _create_collection_schema(memory_client, vector_size)
        return memory_client


def ensure_collection(vector_size):
    if not Config.QDRANT_COLLECTION:
        raise ValueError("Collection_name is missing in environment variables")

    client = _get_qdrant_client()
    collections = client.get_collections().collections
    collection_names = {collection.name for collection in collections}

    if Config.QDRANT_COLLECTION not in collection_names:
        client = _create_hybrid_collection(client, vector_size)
    else:
        collection_info = client.get_collection(Config.QDRANT_COLLECTION)
        current_vectors = collection_info.config.params.vectors
        current_sparse = getattr(collection_info.config.params, "sparse_vectors", None)

        if _is_empty_vectors_config(current_vectors):
            try:
                current_count = client.count(
                    collection_name=Config.QDRANT_COLLECTION,
                    exact=True,
                ).count
            except Exception:
                current_count = None

            if current_count == 0 or Config.QDRANT_RECREATE_ON_SCHEMA_MISMATCH:
                client.delete_collection(collection_name=Config.QDRANT_COLLECTION)
                client = _create_hybrid_collection(client, vector_size)
                return client

            raise ValueError(
                f"Collection '{Config.QDRANT_COLLECTION}' has empty vectors config and appears incompatible "
                "with hybrid mode. Set QDRANT_RECREATE_ON_SCHEMA_MISMATCH=true to recreate automatically."
            )

        dense_ok = _is_named_dense_config(
            current_vectors,
            Config.QDRANT_DENSE_VECTOR_NAME,
            vector_size,
        )
        sparse_ok = _has_named_sparse_config(
            current_sparse,
            Config.QDRANT_SPARSE_VECTOR_NAME,
        )

        if not dense_ok or not sparse_ok:
            if Config.QDRANT_RECREATE_ON_SCHEMA_MISMATCH:
                client.delete_collection(collection_name=Config.QDRANT_COLLECTION)
                client = _create_hybrid_collection(client, vector_size)
                return client
            raise ValueError(
                f"Collection '{Config.QDRANT_COLLECTION}' is not in hybrid format "
                f"(dense='{Config.QDRANT_DENSE_VECTOR_NAME}', sparse='{Config.QDRANT_SPARSE_VECTOR_NAME}'). "
                "Set QDRANT_RECREATE_ON_SCHEMA_MISMATCH=true or recreate collection manually."
            )

    return client


def store_embeddings(chunks, embeddings, batch_size=None):
    if not chunks or not embeddings:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if batch_size is None:
        batch_size = Config.EMBEDDING_BATCH_SIZE

    client = ensure_collection(len(embeddings[0]))
    inserted_count = 0

    try:
        for chunk_batch, embedding_batch in zip(
            _chunk_list(chunks, batch_size),
            _chunk_list(embeddings, batch_size),
        ):
            points = []

            for chunk, embedding in zip(chunk_batch, embedding_batch):
                payload = dict(chunk)
                normalized_score = _normalize_score(payload.get("coherence_score"))
                if normalized_score is None:
                    normalized_score = 0.0
                payload["coherence_score"] = normalized_score
                payload["chunk_quality"] = _quality_from_score(normalized_score)
                sparse_vector = _build_sparse_vector(payload.get("text", ""))
                points.append(
                    PointStruct(
                        id=_make_point_id(chunk),
                        vector={
                            Config.QDRANT_DENSE_VECTOR_NAME: embedding,
                            Config.QDRANT_SPARSE_VECTOR_NAME: sparse_vector,
                        },
                        payload=payload,
                    )
                )
            client.upsert(
                collection_name=Config.QDRANT_COLLECTION,
                points=points,
            )
            inserted_count += len(points)
    finally:
        close_qdrant_client(client)

    return inserted_count


def embed_and_store_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)
    inserted_count = store_embeddings(chunks, embeddings)
    return inserted_count


def get_qdrant_quality_report(batch_size=None):
    if batch_size is None:
        batch_size = Config.QDRANT_SCROLL_BATCH_SIZE
    client = _get_qdrant_client()

    total = 0
    quality_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    scores = []
    next_offset = None

    try:
        while True:
            points, next_offset = client.scroll(
                collection_name=Config.QDRANT_COLLECTION,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )

            if not points:
                break

            total += len(points)
            for point in points:
                payload = point.payload or {}
                quality = payload.get("chunk_quality", "unknown")
                if quality not in quality_counts:
                    quality = "unknown"
                quality_counts[quality] += 1

                normalized_score = _normalize_score(payload.get("coherence_score"))
                if normalized_score is not None:
                    scores.append(normalized_score)

            if next_offset is None:
                break
    finally:
        close_qdrant_client(client)

    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    min_score = round(min(scores), 4) if scores else 0.0
    max_score = round(max(scores), 4) if scores else 0.0

    return {
        "db_points_total": total,
        "db_quality_high": quality_counts["high"],
        "db_quality_medium": quality_counts["medium"],
        "db_quality_low": quality_counts["low"],
        "db_quality_unknown": quality_counts["unknown"],
        "db_avg_coherence_score": avg_score,
        "db_min_coherence_score": min_score,
        "db_max_coherence_score": max_score,
    }
