from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber

from src.chunking.chunker import create_chunks
from src.utils.cleaner import clean_chunks
from src.embeddings.embedder import embed_and_store_chunks, get_qdrant_quality_report
from src.ingestion.pdf_loader import _extract_page_content, extract_metadata, load_pdfs


def _doc_from_pdf_path(pdf_path: Path, company: Optional[str] = None) -> Dict[str, Any]:
    text_parts: List[str] = []
    tables: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text, page_tables = _extract_page_content(page)
            if page_text:
                text_parts.append(page_text)
            if page_tables:
                tables.extend(page_tables)

    metadata = extract_metadata(pdf_path.name)
    return {
        "text": "\n\n".join(text_parts).strip(),
        "tables": tables,
        "company": company or pdf_path.parent.name,
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "report_type": metadata["report_type"],
        "year": metadata["year"],
        "quarter": metadata["quarter"],
    }


def ingest_documents(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunks = create_chunks(docs)
    cleaned_chunks = clean_chunks(chunks)
    vectors_inserted = embed_and_store_chunks(cleaned_chunks)
    db_quality = get_qdrant_quality_report()
    return {
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunks_cleaned": len(cleaned_chunks),
        "vectors_inserted": vectors_inserted,
        **db_quality,
    }


def ingest_all_documents() -> Dict[str, Any]:
    docs = load_pdfs()
    return ingest_documents(docs)


def ingest_single_pdf(pdf_path: str, company: Optional[str] = None) -> Dict[str, Any]:
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    doc = _doc_from_pdf_path(path, company=company)
    return ingest_documents([doc])
