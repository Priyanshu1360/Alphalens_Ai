import json
import os
from dotenv import load_dotenv

load_dotenv()


def _get_env(*names, default=None):
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            return value
    return default


def _get_int(*names, default):
    raw_value = _get_env(*names, default=None)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_float(*names, default):
    raw_value = _get_env(*names, default=None)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _get_bool(*names, default=False):
    raw_value = _get_env(*names, default=None)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_list(*names, default=None, separator=","):
    raw_value = _get_env(*names, default=None)
    if raw_value is None:
        return list(default) if default is not None else []
    values = [item.strip() for item in raw_value.split(separator) if item.strip()]
    if values:
        return values
    return list(default) if default is not None else []


def _get_dict(*names, default=None):
    raw_value = _get_env(*names, default=None)
    if raw_value is None:
        return dict(default) if default is not None else {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return dict(default) if default is not None else {}
    if isinstance(parsed, dict):
        normalized = {}
        for key, value in parsed.items():
            if key is None or value is None:
                continue
            normalized[str(key).lower()] = str(value).lower()
        return normalized
    return dict(default) if default is not None else {}


class Config:
    OPENAI_API_KEY = _get_env("OPENAI_API_KEY", "XAI_API_KEY", "GROQ_API_KEY")
    OPENAI_BASE_URL = _get_env("OPENAI_BASE_URL", "XAI_BASE_URL", "GROQ_BASE_URL")
    OPENAI_EMBEDDINGS_ENABLED = _get_bool("OPENAI_EMBEDDINGS_ENABLED", default=True)
    OPENAI_FALLBACK_TO_LOCAL_ON_ERROR = _get_bool(
        "OPENAI_FALLBACK_TO_LOCAL_ON_ERROR",
        default=True,
    )
    EMBEDDING_FALLBACK_TO_LOCAL_ON_ERROR = _get_bool(
        "EMBEDDING_FALLBACK_TO_LOCAL_ON_ERROR",
        default=True,
    )
    EMBEDDING_BACKEND = _get_env("EMBEDDING_BACKEND", default="sentence_transformers")
    EMBEDDING_MODEL = _get_env(
        "EMBEDDING_MODEL",
        default="FinLang/finance-embeddings-investopedia",
    )
    EMBEDDING_NORMALIZE = _get_bool("EMBEDDING_NORMALIZE", default=True)
    EMBEDDING_DEVICE = _get_env("EMBEDDING_DEVICE", default="cpu")
    EMBEDDING_TRUST_REMOTE_CODE = _get_bool("EMBEDDING_TRUST_REMOTE_CODE", default=False)
    LLM_MODEL = _get_env("LLM_MODEL", default="gpt-4o-mini")
    LLM_ENABLED = _get_bool("LLM_ENABLED", default=True)
    QDRANT_API_KEY = _get_env(
        "QDRANT_API_KEY",
        "Qdrant_API_key",
        "QADRANT_API_KEY",
        "QADRANT_APIKEY",
    )
    QDRANT_PATH = _get_env(
        "QDRANT_PATH",
        "QDRANT_URL",
        "QDRANT_ENDPOINT",
        "Cluster_Endpoint",
        "QADRANT_PATH",
        "QADRANT_URL",
        "QADRANT_ENDPOINT",
    )
    QDRANT_COLLECTION = _get_env(
        "QDRANT_COLLECTION",
        "Collection_name",
        "QADRANT_COLLECTION",
    )
    QDRANT_LOCAL_PATH = _get_env("QDRANT_LOCAL_PATH", default=":memory:")
    QDRANT_TIMEOUT_SECONDS = _get_float("QDRANT_TIMEOUT_SECONDS", default=30.0)
    QDRANT_FALLBACK_TO_LOCAL_ON_ERROR = _get_bool(
        "QDRANT_FALLBACK_TO_LOCAL_ON_ERROR",
        default=True,
    )
    QDRANT_LOCAL_FALLBACK_TO_MEMORY_ON_ERROR = _get_bool(
        "QDRANT_LOCAL_FALLBACK_TO_MEMORY_ON_ERROR",
        default=True,
    )
    QDRANT_LOCAL_RECOVERY_PATH_ON_ERROR = _get_bool(
        "QDRANT_LOCAL_RECOVERY_PATH_ON_ERROR",
        default=True,
    )
    QDRANT_DENSE_VECTOR_NAME = _get_env("QDRANT_DENSE_VECTOR_NAME", default="dense")
    QDRANT_SPARSE_VECTOR_NAME = _get_env("QDRANT_SPARSE_VECTOR_NAME", default="sparse")
    QDRANT_RECREATE_ON_SCHEMA_MISMATCH = _get_bool(
        "QDRANT_RECREATE_ON_SCHEMA_MISMATCH",
        default=False,
    )
    QDRANT_DISTANCE = _get_env("QDRANT_DISTANCE", default="COSINE")

    PDF_BASE_PATH = _get_env("PDF_BASE_PATH", default="Data/raw")
    PDF_GLOB_PATTERN = _get_env("PDF_GLOB_PATTERN", default="*.pdf")

    TEXT_CHUNK_SIZE = _get_int("TEXT_CHUNK_SIZE", default=300)
    TEXT_CHUNK_OVERLAP = _get_int("TEXT_CHUNK_OVERLAP", default=50)
    TABLE_CHUNK_SIZE = _get_int("TABLE_CHUNK_SIZE", default=120)
    TABLE_CHUNK_OVERLAP = _get_int("TABLE_CHUNK_OVERLAP", default=20)

    EMBEDDING_BATCH_SIZE = _get_int("EMBEDDING_BATCH_SIZE", default=100)
    QDRANT_SCROLL_BATCH_SIZE = _get_int("QDRANT_SCROLL_BATCH_SIZE", default=512)
    LOCAL_EMBEDDING_DIMENSION = _get_int("LOCAL_EMBEDDING_DIMENSION", default=1536)
    SPARSE_HASH_SPACE = _get_int("SPARSE_HASH_SPACE", default=2000003)
    RETRIEVAL_TOP_K = _get_int("RETRIEVAL_TOP_K", default=10)
    HYBRID_PREFETCH_LIMIT = _get_int("HYBRID_PREFETCH_LIMIT", default=50)
    HYBRID_FUSION = _get_env("HYBRID_FUSION", default="RRF")
    RETRIEVAL_SNIPPET_CHARS = _get_int("RETRIEVAL_SNIPPET_CHARS", default=240)
    RERANK_ENABLED = _get_bool("RERANK_ENABLED", default=True)
    RERANK_CANDIDATES = _get_int("RERANK_CANDIDATES", default=40)
    RERANK_RETRIEVAL_WEIGHT = _get_float("RERANK_RETRIEVAL_WEIGHT", default=0.45)
    RERANK_LEXICAL_WEIGHT = _get_float("RERANK_LEXICAL_WEIGHT", default=0.35)
    RERANK_COHERENCE_WEIGHT = _get_float("RERANK_COHERENCE_WEIGHT", default=0.15)
    RERANK_QUALITY_WEIGHT = _get_float("RERANK_QUALITY_WEIGHT", default=0.05)
    GENERATION_CONTEXT_CHUNKS = _get_int("GENERATION_CONTEXT_CHUNKS", default=4)
    GENERATION_CONTEXT_CHARS = _get_int("GENERATION_CONTEXT_CHARS", default=900)
    GENERATION_FALLBACK_TOP_SOURCES = _get_int("GENERATION_FALLBACK_TOP_SOURCES", default=3)
    GENERATOR_SYSTEM_PROMPT = _get_env(
        "GENERATOR_SYSTEM_PROMPT",
        default=(
            "You are a financial filing assistant. Answer only from the provided context. "
            "If context is insufficient, explicitly say so. Keep answer concise and include source numbers."
        ),
    )
    GENERATOR_OUTPUT_INSTRUCTION = _get_env(
        "GENERATOR_OUTPUT_INSTRUCTION",
        default="Return a short answer with 2-4 bullet points and source refs like [1], [2].",
    )

    RETRIEVER_DEFAULT_MODE = _get_env("RETRIEVER_DEFAULT_MODE", default="hybrid")
    AUTO_BOOTSTRAP_INDEX_ON_QUERY = _get_bool(
        "AUTO_BOOTSTRAP_INDEX_ON_QUERY",
        default=True,
    )
    ASK_DEFAULT_LIMIT = _get_int("ASK_DEFAULT_LIMIT", default=6)
    ASK_DEFAULT_SOURCES_LIMIT = _get_int("ASK_DEFAULT_SOURCES_LIMIT", default=1)
    SEARCH_DEFAULT_LIMIT = _get_int("SEARCH_DEFAULT_LIMIT", default=0)
    EVALUATE_DEFAULT_LIMIT = _get_int("EVALUATE_DEFAULT_LIMIT", default=5)
    EVALUATE_DEFAULT_QUERIES = _get_list(
        "EVALUATE_DEFAULT_QUERIES",
        separator="||",
        default=[
            "What was Amazon's revenue trend in 2024?",
            "What do Apple filings say about gross margin changes in 2024?",
            "What did Meta say about AI spending in 2024?",
            "Summarize Google Cloud performance in 2024 filings.",
        ],
    )

    RETRIEVER_COMPANY_ALIASES = _get_dict(
        "RETRIEVER_COMPANY_ALIASES",
        default={
            "amazon": "amazon",
            "amzn": "amazon",
            "apple": "apple",
            "aapl": "apple",
            "meta": "meta",
            "facebook": "meta",
            "fb": "meta",
            "google": "google",
            "alphabet": "google",
            "goog": "google",
            "googl": "google",
        },
    )
    RETRIEVER_FILTER_INDEX_FIELDS = _get_list(
        "RETRIEVER_FILTER_INDEX_FIELDS",
        default=["company", "year", "quarter", "report_type"],
    )

    QUALITY_HIGH_THRESHOLD = _get_float("QUALITY_HIGH_THRESHOLD", default=0.75)
    QUALITY_MEDIUM_THRESHOLD = _get_float("QUALITY_MEDIUM_THRESHOLD", default=0.55)

    TEXT_TARGET_WORDS = _get_float("TEXT_TARGET_WORDS", default=80.0)
    TABLE_TARGET_WORDS = _get_float("TABLE_TARGET_WORDS", default=60.0)
    TEXT_UNIQUE_RATIO_TARGET = _get_float("TEXT_UNIQUE_RATIO_TARGET", default=0.55)
    TABLE_UNIQUE_RATIO_TARGET = _get_float("TABLE_UNIQUE_RATIO_TARGET", default=0.65)
