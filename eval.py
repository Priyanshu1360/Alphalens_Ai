import asyncio
import json
from typing import Dict, Any

from src.utils.config import Config
from src.utils.rag_service import run_rag_query
from src.utils.grader import grade_documents, grade_hallucination, grade_answer_relevance

def evaluate_system():
    queries = Config.EVALUATE_DEFAULT_QUERIES
    results = []
    
    total_queries = len(queries)
    context_relevance_pass = 0
    hallucination_pass = 0
    answer_relevance_pass = 0
    
    for query in queries:
        print(f"Evaluating: {query}")
        try:
            # 1. Run RAG Query
            response = run_rag_query(query)
            docs = response.get("results", [])
            answer = response.get("answer", "")
            
            # 2. Grade
            is_relevant_context = grade_documents(query, docs)
            is_grounded = grade_hallucination(answer, docs)
            is_relevant_answer = grade_answer_relevance(query, answer)
            
            if is_relevant_context:
                context_relevance_pass += 1
            if is_grounded:
                hallucination_pass += 1
            if is_relevant_answer:
                answer_relevance_pass += 1
                
            results.append({
                "query": query,
                "context_relevance": is_relevant_context,
                "groundedness": is_grounded,
                "answer_relevance": is_relevant_answer
            })
        except Exception as e:
            print(f"Error evaluating '{query}': {e}")
            
    summary = {
        "total": total_queries,
        "context_relevance_score": context_relevance_pass / total_queries if total_queries > 0 else 0,
        "groundedness_score": hallucination_pass / total_queries if total_queries > 0 else 0,
        "answer_relevance_score": answer_relevance_pass / total_queries if total_queries > 0 else 0,
        "details": results
    }
    
    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("Evaluation complete. Results saved to eval_results.json")

if __name__ == "__main__":
    evaluate_system()
