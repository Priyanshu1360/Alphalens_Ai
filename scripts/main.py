from src.chunking.chunker import create_chunks
from src.utils.cleaner import clean_chunks
from src.embeddings.embedder import embed_and_store_chunks, get_qdrant_quality_report
from src.ingestion.pdf_loader import load_pdfs


def run_pipeline():
    docs = load_pdfs()
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


if __name__ == "__main__":
    result = run_pipeline()
    print("Ingestion pipeline completed successfully")
    print(f"documents_loaded: {result['documents_loaded']}")
    print(f"chunks_created: {result['chunks_created']}")
    print(f"chunks_cleaned: {result['chunks_cleaned']}")
    print(f"vectors_inserted: {result['vectors_inserted']}")
    print(f"db_points_total: {result['db_points_total']}")
    print(f"db_quality_high: {result['db_quality_high']}")
    print(f"db_quality_medium: {result['db_quality_medium']}")
    print(f"db_quality_low: {result['db_quality_low']}")
    print(f"db_quality_unknown: {result['db_quality_unknown']}")
