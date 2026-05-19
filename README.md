# My RAG Pipeline

RAG pipeline for financial filing PDFs with:
- chunking and cleaning
- hybrid retrieval on Qdrant (dense + sparse)
- reranking
- answer generation with Groq-compatible OpenAI client

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Update secrets in `.env`:
- `GROQ_API_KEY`
- `QDRANT_API_KEY`
- `QDRANT_PATH`

3. Verify pipeline/model/Qdrant connection:

```bash
python check_setup.py
```

## Run

1. Build/refresh index:

```bash
python main.py
```

2. Retrieval only:

```bash
python search.py "What did Apple say about gross margin in 2024?" --mode hybrid --limit 5
```

3. Retrieval + generation:

```bash
python ask.py "Summarize Amazon revenue trend in 2024 with key drivers and cite sources." --mode hybrid --limit 6
```

4. Evaluation report:

```bash
python evaluate.py --limit 3
```

## Notes

- Keep `.env` private and rotate leaked keys.
- If remote Qdrant is unavailable and you want local fallback, set `QDRANT_FALLBACK_TO_LOCAL_ON_ERROR=true`.
- For fully offline ingestion, set `EMBEDDING_BACKEND=local_hash`.
- If `QDRANT_LOCAL_PATH=:memory:`, index is process-local; `ask.py/search.py/chat.py` auto-bootstrap the index on first query.
- Config accepts both `QDRANT_*` and `QADRANT_*` variable names for connection settings.
