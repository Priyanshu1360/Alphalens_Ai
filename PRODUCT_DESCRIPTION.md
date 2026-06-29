# Alphalens AI Product Description

## 🌟 Vision
Alphalens AI is an advanced Document Intelligence and RAG platform built for production environments. It is engineered to ingest complex financial documents, securely embed them into a cloud vector database, and provide highly accurate, hallucination-free answers using Large Language Models (LLMs).

## 🛡️ Core Capabilities
The platform features a **hybrid retrieval system** combining dense vector search and sparse lexical search (BM25) with heuristic reranking. It is protected by a strong governance layer powered by **Guardrails AI** that actively guards against prompt injections and masks Personally Identifiable Information (PII) like SSNs, emails, and phone numbers. 

The architecture is split into high-performance micro-services, ensuring lightning-fast inferences, secure cloud storage, and an ultra-modern user interface.

---

## 🏗️ Production Architecture Stages

The system is organized into scalable, modular stages following enterprise best practices:

### 1. Ingestion Stage (Kaggle Cloud Compute)
Heavy document ingestion is offloaded to a Kaggle GPU environment to prevent local compute bottlenecks.
*   **Parsing & Chunking**: 28+ massive financial PDFs are parsed using `docling` and `pdfplumber`, cleaned, and split into semantically meaningful text and tabular chunks.
*   **Embedding Extraction**: Text chunks are embedded using GPU-accelerated embedding models (`BAAI/bge-large-en-v1.5`).
*   **BM25 Lexical Processing**: Sparse vectors are generated simultaneously to capture exact keyword matches.
*   **Cloud Push**: The processed vectors are streamed directly into Qdrant Cloud.

### 2. Vector Storage (Qdrant Cloud)
*   **Managed Database**: All dense and sparse vectors are hosted securely in a managed **Qdrant Cloud Cluster**.
*   **High-Availability**: Eliminates local database size constraints, allowing instant cross-origin queries via API Keys.

### 3. Backend Orchestration (FastAPI)
The central nervous system of the RAG pipeline (`src/api`).
*   **Hybrid Retrieval (`src/retrieval`)**: Employs Hybrid Search (Dense + Sparse) with Reciprocal Rank Fusion (RRF). Uses a lightning-fast **Heuristic Reranker** to drop latency to <0.1s while maintaining top-tier relevance.
*   **LLM Generation & Agents (`src/llm`, `src/utils/agent_graph.py`)**: Uses **LangGraph** to route queries. Conversational queries are handled quickly, while complex financial queries trigger a deep-reasoning agent.
*   **Dual-Layer Caching**: A unified caching layer (`ExactMatchCache` + `SemanticCache`) provides instant 0.04s responses for repeated queries and handles spelling typos gracefully.
*   **Security & Guardrails**: Validates outputs via `Guardrails AI` to prevent unauthorized PII leaks and malicious prompt injections.

### 4. Frontend UI (Next.js)
A premium, interactive web dashboard (`frontend/`).
*   **Modern Stack**: Built on React 19, Next.js 16 (Turbopack), and Tailwind CSS.
*   **Interactive Chat**: Real-time streaming UI, markdown rendering, and embedded mermaid flowcharts that visualize the backend agent's reasoning process.
