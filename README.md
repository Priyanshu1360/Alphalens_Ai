# Alphalens AI (Financial RAG Pipeline)

A robust, production-ready Retrieval-Augmented Generation (RAG) pipeline and Copilot-style UI designed for querying, analyzing, and chatting with financial filing PDFs.

## 🌟 Project Description

Alphalens AI is an advanced Document Intelligence and RAG platform. It is engineered to ingest complex financial documents, embed them into a vector database, and provide highly accurate, hallucination-free answers using Large Language Models (LLMs). 

The platform features a **hybrid retrieval system** combining dense vector search and sparse lexical search (BM25) with cross-encoder/heuristic reranking. It is protected by a strong governance layer powered by **Guardrails AI** that actively guards against prompt injections and masks Personally Identifiable Information (PII) like SSNs, emails, and phone numbers.

---

## 🏗️ Architecture Overview

The system is organized into modular components following enterprise best practices:

*   **Ingestion & Chunking (`src/ingestion`, `src/chunking`)**: PDFs are parsed using `pdfplumber`, cleaned, and split into semantically meaningful text and tabular chunks.
*   **Embeddings & Vector DB (`src/embeddings`, `src/vectordb`)**: Chunks are embedded using local models (`sentence-transformers`) and stored in **Qdrant** for high-performance similarity search.
*   **Retrieval (`src/retrieval`)**: Employs Hybrid Search (Dense + Sparse) with Reciprocal Rank Fusion (RRF). Uses a lightning-fast **Heuristic Reranker** to drop latency to <0.1s while maintaining top-tier relevance.
*   **LLM Generation & Agents (`src/llm`, `src/utils/agent_graph.py`)**: Uses **LangGraph** to route queries. Conversational queries are handled quickly, while complex financial queries trigger a deep-reasoning agent.
*   **Dual-Layer Caching**: A unified caching layer (`ExactMatchCache` + `SemanticCache`) provides instant 0.04s responses for repeated queries and handles spelling typos gracefully.
*   **Matplotlib Charts**: Features a smart backend interceptor that converts LLM JSON output into highly accurate, base64-encoded `Matplotlib` charts (Bar, Line, Pie) rendered instantly in the chat.
*   **Backend API (`src/api`)**: A **FastAPI** backend that exposes endpoints for querying, agent workflows, and document ingestion.
*   **Frontend UI (`frontend/`)**: A premium, interactive **Next.js** dashboard featuring execution flow visualizations and dynamic data rendering.

### Project Structure
```text
my_rag/
├── README.md               - Project description, setup guide, architecture
├── requirements.txt        - List of all Python dependencies
├── .env                    - Stores API keys and database credentials (Ignored by Git)
├── config.yaml             - Configuration for models, chunk sizes, and DB settings
├── frontend/               - Next.js modern web interface
├── scripts/                - Standalone CLI utilities for ingestion, search, and eval
├── src/
│   ├── api/                - API endpoints (FastAPI)
│   ├── chunking/           - Text and table chunking logic
│   ├── embeddings/         - Convert chunks to dense/sparse vectors
│   ├── ingestion/          - PDF loading and data ingestion logic
│   ├── llm/                - LLM API clients and generators
│   ├── prompts/            - System prompts and generation instructions
│   ├── retrieval/          - Hybrid search and reranking logic
│   ├── vectordb/           - Vector database client operations (Qdrant)
│   └── utils/              - Helper functions (Caching, Guardrails, Graph Agents, Plotting)
├── tests/                  - Unit and integration tests
└── logs/                   - Application logs
```

---

## ⚙️ Setup Guide

### 1. Prerequisites
- Python 3.9 or higher
- Node.js & npm (for the Next.js frontend)
- Qdrant Cluster (Cloud or Local Docker)
- PostgreSQL (Optional, required for persistent chat history)

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone <repository_url>
cd my_rag

# Install Backend dependencies
pip install -r requirements.txt

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

### 3. Environment Variables (`.env`)
Create a `.env` file in the root directory and configure your keys. *Do not commit this file to version control.*

```env
# LLM Providers (Compatible with OpenAI SDK, e.g. Groq, Together, xAI)
GROQ_API_KEY=your_api_key_here

# Qdrant Vector Database
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_PATH=https://your-cluster-url.qdrant.io
QDRANT_COLLECTION=financial-filings

# Database (For Chat History)
POSTGRES_URL=postgresql://user:password@localhost:5432/rag_db
```

### 4. Verify Setup
Run the diagnostic script to ensure the LLM connects and Qdrant is reachable:
```bash
python scripts/check_setup.py
```

---

## 🚀 How to Run

### 🖥️ Option 1: Full-Stack Launch (Recommended)
Launch both the FastAPI backend and Next.js frontend simultaneously using the provided script.

**Windows (PowerShell):**
```powershell
.\run_fullstack.ps1
```
```

### 🔌 Option 2: FastAPI Backend Server
Start the backend server to expose the REST API endpoints (`/health`, `/ingest`, `/query`, `/agent-query`).

```bash
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000
```
*Access the Swagger documentation at `http://localhost:8000/docs`.*

### 🛠️ Option 3: Command Line Interface (CLI)

**1. Ingest Documents / Rebuild Index:**
Places PDFs in `Data/raw/` and run:
```bash
python main.py
```

**2. Ask a Question (End-to-End Generation):**
```bash
python ask.py "Summarize Amazon's revenue trend in 2024 and cite sources." --mode hybrid --limit 6
```

**3. Test Retrieval Only (No Generation):**
```bash
python search.py "What did Apple say about gross margin in 2024?" --mode hybrid --limit 5
```

**4. Run Automated Evaluations:**
```bash
python evaluate.py --limit 3
```

---

## 🛡️ Key Features

- **Hybrid Retrieval:** Dense + Sparse (BM25) vector search for maximizing recall.
- **Intent Classification:** Regex-based query routing distinguishes between casual conversation and complex data queries.
- **Security Guardrails (Powered by Guardrails AI):** Structured validation layer that actively blocks prompt injection attacks and automatically masks PII (SSN, Email) using custom low-latency validators.
- **Strict Scope Generation:** Bounded generation forces the LLM to answer *only* from the retrieved context to prevent hallucinations.
- **Self-Reflective RAG (Self-RAG):** Autonomous evaluation loops that grade outputs in real-time. If retrieved documents are irrelevant, or if the generated answer contains hallucinations, it triggers a **rewrite node** to optimize the query and retry. A **Safety Valve** caps reflection loops to prevent runaway API costs.
- **Latency Optimizations:** Features a robust Semantic Cache bypass for repeat queries (~50ms latency) and delegates Self-RAG Quality Gates to lightning-fast 8B micro-models.
- **Persistent Memory:** Thread storage backed by PostgreSQL.

---

## 🏗️ System Architecture Workflow

```mermaid
graph TD
    classDef user fill:#2C3E50,stroke:#fff,stroke-width:2px,color:#fff;
    classDef security fill:#E74C3C,stroke:#C0392B,stroke-width:2px,color:#fff;
    classDef router fill:#F39C12,stroke:#D35400,stroke-width:2px,color:#fff;
    classDef retrieval fill:#3498DB,stroke:#2980B9,stroke-width:2px,color:#fff;
    classDef generate fill:#2ECC71,stroke:#27AE60,stroke-width:2px,color:#fff;
    classDef reflect fill:#9B59B6,stroke:#8E44AD,stroke-width:2px,color:#fff;
    classDef cache fill:#1ABC9C,stroke:#16A085,stroke-width:2px,color:#fff;

    User([User Query]) ::: user
    Guardrails["🛡️ Guardrails AI<br>PII Masking & Prompt Injection Block"] ::: security
    User --> Guardrails
    
    SemanticCache{"⚡ Semantic Cache"} ::: cache
    Guardrails --> SemanticCache
    SemanticCache -- "Cache Hit" --> Output([Final Answer]) ::: user
    
    Router{"🛣️ Intent Classifier"} ::: router
    SemanticCache -- "Cache Miss" --> Router
    
    ConvLLM["💬 Conversational LLM"] ::: generate
    Router -- "Casual Chat" --> ConvLLM
    ConvLLM --> Output([Final Answer]) ::: user
    
    HybridSearch["🔍 Hybrid Retrieval<br>Qdrant"] ::: retrieval
    Router -- "Financial Query" --> HybridSearch
    
    CrossEncoder["📊 Cross-Encoder Reranker"] ::: retrieval
    HybridSearch --> CrossEncoder
    
    DocGrader{"🤔 Grade Documents"} ::: reflect
    CrossEncoder --> DocGrader
    
    Rewrite["✍️ Rewrite Query"] ::: reflect
    DocGrader -- "Irrelevant" --> Rewrite
    Rewrite --> HybridSearch
    
    Generate["🧠 LLM Generation"] ::: generate
    DocGrader -- "Relevant" --> Generate
    
    GenGrader{"🧐 Grade Generation"} ::: reflect
    Generate --> GenGrader
    
    GenGrader -- "Failed" --> Rewrite
    GenGrader -- "Passed" --> Output
    Rewrite -.->|Safety Valve| Generate
```
