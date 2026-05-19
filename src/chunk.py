from src.config import Config


def chunk_text(text, size=None, overlap=None):
    if size is None:
        size = Config.TEXT_CHUNK_SIZE
    if overlap is None:
        overlap = Config.TEXT_CHUNK_OVERLAP

    if not text:
        return []
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _base_metadata(doc):
    return {
        "company": doc.get("company"),
        "report_type": doc.get("report_type"),
        "year": doc.get("year"),
        "quarter": doc.get("quarter"),
        "file_name": doc.get("file_name"),
        "file_path": doc.get("file_path"),
    }


def create_chunks(
    docs,
    text_chunk_size=None,
    text_overlap=None,
    table_chunk_size=None,
    table_overlap=None,
):
    if text_chunk_size is None:
        text_chunk_size = Config.TEXT_CHUNK_SIZE
    if text_overlap is None:
        text_overlap = Config.TEXT_CHUNK_OVERLAP
    if table_chunk_size is None:
        table_chunk_size = Config.TABLE_CHUNK_SIZE
    if table_overlap is None:
        table_overlap = Config.TABLE_CHUNK_OVERLAP

    all_chunks = []

    for doc in docs:
        metadata = _base_metadata(doc)

        text_chunks = chunk_text(
            doc.get("text", ""),
            size=text_chunk_size,
            overlap=text_overlap,
        )

        for index, chunk in enumerate(text_chunks, start=1):
            all_chunks.append(
                {
                    "text": chunk,
                    "chunk_type": "text",
                    "chunk_index": index,
                    **metadata,
                }
            )

        for table_index, table_text in enumerate(doc.get("tables", []), start=1):
            table_chunks = chunk_text(
                table_text,
                size=table_chunk_size,
                overlap=table_overlap,
            )

            for part_index, chunk in enumerate(table_chunks, start=1):
                all_chunks.append(
                    {
                        "text": chunk,
                        "chunk_type": "table",
                        "chunk_index": part_index,
                        "table_index": table_index,
                        **metadata,
                    }
                )

    return all_chunks
