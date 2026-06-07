import re

from src.utils.config import Config


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, value))


def _safe_positive(value, fallback):
    return value if value > 0 else fallback


def _tokenize(text):
    return WORD_PATTERN.findall(text.lower())


def _sentence_similarity(a, b):
    a_tokens = set(_tokenize(a))
    b_tokens = set(_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _text_chunk_coherence(text):
    words = _tokenize(text)
    word_count = len(words)
    if word_count == 0:
        return 0.0

    text_target_words = _safe_positive(Config.TEXT_TARGET_WORDS, 80.0)
    text_unique_target = _safe_positive(Config.TEXT_UNIQUE_RATIO_TARGET, 0.55)

    length_score = _clamp(word_count / text_target_words)
    unique_ratio = len(set(words)) / word_count
    diversity_score = _clamp(
        1.0
        - abs(unique_ratio - text_unique_target)
        / text_unique_target
    )

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) >= 2:
        sims = []
        for i in range(len(sentences) - 1):
            sims.append(_sentence_similarity(sentences[i], sentences[i + 1]))
        connection_score = _clamp((sum(sims) / len(sims)) * 4.0)
    else:
        connection_score = 0.5

    score = (
        0.35 * length_score
        + 0.35 * diversity_score
        + 0.30 * connection_score
    )
    return round(_clamp(score), 4)


def _table_chunk_coherence(text):
    words = _tokenize(text)
    word_count = len(words)
    if word_count == 0:
        return 0.0

    table_target_words = _safe_positive(Config.TABLE_TARGET_WORDS, 60.0)
    table_unique_target = _safe_positive(Config.TABLE_UNIQUE_RATIO_TARGET, 0.65)

    length_score = _clamp(word_count / table_target_words)
    lines = [line for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if "|" in line]

    if table_lines:
        col_counts = [line.count("|") + 1 for line in table_lines]
        same_cols = sum(1 for c in col_counts if c == col_counts[0]) / len(col_counts)
        structure_score = _clamp(same_cols)
    else:
        structure_score = 0.4

    unique_ratio = len(set(words)) / word_count
    diversity_score = _clamp(
        1.0
        - abs(unique_ratio - table_unique_target)
        / table_unique_target
    )

    score = (
        0.40 * length_score
        + 0.40 * structure_score
        + 0.20 * diversity_score
    )
    return round(_clamp(score), 4)


def get_coherence_score(text, chunk_type="text"):
    if chunk_type == "table":
        return _table_chunk_coherence(text)
    return _text_chunk_coherence(text)


def get_quality_label(score):
    if score >= Config.QUALITY_HIGH_THRESHOLD:
        return "high"
    if score >= Config.QUALITY_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def clean_chunks(chunks):
    cleaned_chunks = []

    for chunk in chunks:
        cleaned_text = clean_text(chunk.get("text", ""))
        if not cleaned_text:
            continue

        cleaned_chunk = dict(chunk)
        cleaned_chunk["text"] = cleaned_text
        cleaned_chunk["coherence_score"] = get_coherence_score(
            cleaned_text,
            cleaned_chunk.get("chunk_type", "text"),
        )
        cleaned_chunk["chunk_quality"] = get_quality_label(
            cleaned_chunk["coherence_score"]
        )
        cleaned_chunks.append(cleaned_chunk)

    return cleaned_chunks
