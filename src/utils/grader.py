import json
import logging
from typing import List, Dict, Any

from src.utils.config import Config
from src.llm.llm_client import _get_llm_clients

LOGGER = logging.getLogger("grader")


def _run_json_grader(prompt: str, user_content: str, fallback_value: bool = True) -> bool:
    """Helper to run a fast JSON-mode LLM call for grading."""
    last_exc = None
    try:
        clients = _get_llm_clients()
        for client in clients:
            try:
                response = client.chat.completions.create(
                    model=Config.GRADER_LLM_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                parsed = json.loads(content)
                return bool(parsed.get("valid", fallback_value))
            except Exception as e:
                last_exc = e
                continue
    except Exception as e:
        last_exc = e
    LOGGER.error(f"Grader LLM failed: {last_exc}")
    return fallback_value  # Fail-open to avoid breaking the pipeline entirely


def grade_documents(query: str, docs: List[Dict[str, Any]]) -> bool:
    """Checks if the retrieved documents are relevant to the query."""
    if not docs:
        return False
    
    # We can use the fast reranker score as a primary filter to save LLM calls
    # Threshold lowered to 0.05: heuristic reranker scores are in 0.0–0.5 range,
    # so 0.2 was incorrectly discarding valid docs before the LLM grader could evaluate them.
    max_score = max([float(d.get("rerank_score") or d.get("score") or 0) for d in docs])
    if max_score < 0.05:
        return False

    prompt = (
        "You are a grader assessing relevance of retrieved documents to a user question. "
        "Return JSON with a single key 'valid' set to true or false. "
        "It is true if ANY document contains keywords or semantic meaning related to the user question "
        "(e.g., if the user asks for 'revenue', documents mentioning 'sales' are highly relevant). "
        "Ignore all markdown formatting and focus on the text content."
    )
    
    doc_text = "\n\n".join([
        str(d.get("payload", {}).get("text") or d.get("snippet", ""))[:1200]
        for d in docs
    ])
    user_content = f"Question: {query}\n\nDocuments:\n{doc_text}"
    
    return _run_json_grader(prompt, user_content)


def grade_hallucination(answer: str, docs: List[Dict[str, Any]]) -> bool:
    """Checks if the generated answer is grounded in the documents (no hallucinations)."""
    if not docs or not answer:
        return True # Nothing to grade
        
    prompt = (
        "You are a grader assessing whether an AI generation is grounded in a set of retrieved facts. "
        "Return JSON with a single key 'valid' set to true or false. "
        "It is true if the KEY FACTUAL CLAIMS in the generation are supported by the facts. "
        "Ignore introductory phrases, hedging language, or formatting — focus only on whether the "
        "core facts and numbers are backed by the retrieved context. "
        "If the answer says it cannot find information, that is also valid (return true)."
    )
    
    doc_text = "\n\n".join([
        str(d.get("payload", {}).get("text") or d.get("snippet", ""))[:1200]
        for d in docs
    ])
    user_content = f"Facts:\n{doc_text}\n\nGeneration: {answer}"
    
    return _run_json_grader(prompt, user_content)


def grade_answer_relevance(query: str, answer: str) -> bool:
    """Checks if the generated answer actually resolves the user's question."""
    if not answer:
        return False
        
    prompt = (
        "You are an AI grader. Does the generated answer discuss the same general topic as the question? "
        "Return JSON with a single key 'valid' set to true or false. "
        "If the question is about Amazon revenue, and the answer discusses Amazon sales/revenue, it is valid (true). "
        "If the question is about Meta AI, and the answer discusses Meta AI or infrastructure, it is valid (true). "
        "Ignore formatting and charts. Output true if it is on-topic."
    )
    
    user_content = f"Question: {query}\n\nAnswer: {answer}"
    
    return _run_json_grader(prompt, user_content)


def rewrite_query(query: str) -> str:
    """Rewrites a query to be better optimized for retrieval after a failure."""
    last_exc = None
    try:
        clients = _get_llm_clients()
        prompt = (
            "You are a search query optimizer. The user's previous search failed to find relevant information. "
            "Rewrite the user's query to be a better semantic search query. "
            "Extract key entities and relationships. Do not include introductory text, just return the optimized query string."
        )
        for client in clients:
            try:
                response = client.chat.completions.create(
                    model=Config.GRADER_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Original query: {query}"},
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_exc = e
                continue
    except Exception as e:
        last_exc = e
    LOGGER.error(f"Rewrite LLM failed: {last_exc}")
    return query
