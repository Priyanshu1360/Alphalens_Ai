import logging
import os
import json
import re
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

from src.utils.cache_memory import ExactMatchCache, normalize_query
from src.llm.llm_client import generate_answer
from src.utils.mcp_client import MCPClient
from src.utils.rag_service import retrieve_documents
from src.utils.config import Config
from src.utils.grader import grade_documents, grade_hallucination, grade_answer_relevance, rewrite_query
from src.utils.guardrails import apply_input_guardrails

LOGGER = logging.getLogger("agent_graph")

try:
    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except Exception:  # pragma: no cover - version compatibility
        from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency fallback
    LANGGRAPH_AVAILABLE = False
    InMemorySaver = None
    END = "__end__"
    START = "__start__"
    StateGraph = None


class AgentState(TypedDict, total=False):
    query: str
    mode: str
    limit: int
    prefetch_limit: int
    rerank: bool
    query_normalized: str
    retrieved_docs: List[Dict[str, Any]]
    cache_hit: bool
    cached_response: Dict[str, Any]
    mcp_results: List[Dict[str, Any]]
    final_answer: str
    chart_json_data: str
    next_action: str
    route_reason: str
    needs_retrieve: bool
    needs_mcp: bool
    needs_direct: bool
    cache_checked: bool
    run_log: List[str]
    reflection_count: int


def _append_log(state: AgentState, message: str):
    state.setdefault("run_log", [])
    state["run_log"].append(message)
    LOGGER.info(message)


def _detect_company(query: str) -> str:
    lower = (query or "").lower()
    for name in ["amazon", "apple", "meta", "google", "alphabet", "microsoft"]:
        if name in lower:
            return "google" if name == "alphabet" else name
    return "apple"


def _detect_ticker(query: str) -> str:
    lower = (query or "").lower()
    mapping = {
        "apple": "AAPL",
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "microsoft": "MSFT",
    }
    for name, ticker in mapping.items():
        if name in lower:
            return ticker
    for symbol in ["AAPL", "AMZN", "GOOGL", "META", "MSFT"]:
        if symbol.lower() in lower:
            return symbol
    return "AAPL"


def _detect_year(query: str) -> str:
    match = re.search(r"\b(20\d{2})\b", query or "")
    return match.group(1) if match else ""


class AgentWorkflow:
    def __init__(self, exact_cache: ExactMatchCache, semantic_cache: Any, mcp_client: MCPClient):
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.mcp_client = mcp_client
        self.langgraph_enabled = LANGGRAPH_AVAILABLE
        self._graph = self._build_graph()

    def supervisor_node(self, state: AgentState) -> Dict[str, Any]:
        from src.utils.classifier import classify_intent
        query = state.get("query", "")
        
        if classify_intent(query) == "conversational":
            _append_log(state, "supervisor_node: conversational intent detected")
            return {
                "query_normalized": normalize_query(query),
                "needs_retrieve": False,
                "needs_mcp": False,
                "needs_direct": True,
                "cache_checked": True,  # Bypass cache to avoid bad cached entries
                "cache_hit": False,
                "route_reason": "conversational",
                "reflection_count": state.get("reflection_count") or 0,
            }

        normalized = normalize_query(query)
        lower = normalized

        needs_mcp = any(
            token in lower
            for token in ["stock price", "price", "ratio", "calculate", "yoy", "sec"]
        )
        needs_retrieve = not any(
            token in lower for token in ["general knowledge", "without filings"]
        )
        needs_direct = not needs_retrieve and not needs_mcp

        _append_log(
            state,
            f"supervisor_node: needs_retrieve={needs_retrieve}, needs_mcp={needs_mcp}, needs_direct={needs_direct}",
        )
        return {
            "query_normalized": normalized,
            "needs_retrieve": needs_retrieve,
            "needs_mcp": needs_mcp,
            "needs_direct": needs_direct,
            "cache_checked": False,
            "route_reason": "supervisor_decision",
            "reflection_count": state.get("reflection_count") or 0,
        }

    def router_node(self, state: AgentState) -> Dict[str, Any]:
        if not state.get("cache_checked"):
            next_action = "cache"
            reason = "cache_not_checked"
        elif state.get("cache_hit"):
            next_action = "generate"
            reason = "cache_hit"
        elif state.get("needs_retrieve") and not state.get("retrieved_docs"):
            next_action = "retrieve"
            reason = "need_rag_context"
        elif state.get("needs_mcp") and not state.get("mcp_results"):
            next_action = "mcp"
            reason = "need_mcp_context"
        elif state.get("needs_direct"):
            next_action = "generate"
            reason = "direct_answer"
        else:
            next_action = "generate"
            reason = "ready_to_generate"

        _append_log(state, f"router_node: next_action={next_action} reason={reason}")
        return {"next_action": next_action, "route_reason": reason}

    def cache_check_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        
        # 1. Check Exact Cache first (Zero overhead)
        cached = self.exact_cache.get(query)
        if cached:
            _append_log(state, f"cache_check_node: exact_cache_hit=True")
            return {
                "cache_checked": True,
                "cache_hit": True,
                "cached_response": cached,
            }
            
        # 2. Check Semantic Cache (Requires embedding overhead)
        if hasattr(self.semantic_cache, 'get'):
            cached = self.semantic_cache.get(query)
            if cached:
                _append_log(state, f"cache_check_node: semantic_cache_hit=True")
                # Backfill exact cache for future
                self.exact_cache.set(query, cached.get("answer", ""), cached.get("docs", []))
                return {
                    "cache_checked": True,
                    "cache_hit": True,
                    "cached_response": cached,
                }

        _append_log(state, f"cache_check_node: cache_hit=False")
        return {
            "cache_checked": True,
            "cache_hit": False,
            "cached_response": {},
        }

    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        mode = state.get("mode") or "hybrid"
        limit = state.get("limit")
        prefetch_limit = state.get("prefetch_limit")
        rerank = state.get("rerank")
        docs = retrieve_documents(
            query=query,
            mode=mode,
            limit=limit,
            prefetch_limit=prefetch_limit,
            rerank=rerank,
        )
        _append_log(state, f"retrieve_node: retrieved_docs={len(docs)}")
        return {"retrieved_docs": docs}

    def grade_documents_node(self, state: AgentState) -> Dict[str, Any]:
        """Self-RAG: Grade retrieved documents for relevance."""
        if not Config.SELF_RAG_ENABLED:
            return {"next_action": "continue_to_router"}

        docs = state.get("retrieved_docs", [])
        query = state.get("query", "")
        count = state.get("reflection_count", 0)

        is_relevant = grade_documents(query, docs)
        
        if is_relevant or count >= Config.MAX_REFLECTION_LOOPS:
            _append_log(state, f"grade_documents_node: relevant={is_relevant}, proceed")
            return {"next_action": "continue_to_router"}
        
        _append_log(state, f"grade_documents_node: documents irrelevant. Triggering rewrite.")
        return {"next_action": "rewrite"}

    def rewrite_node(self, state: AgentState) -> Dict[str, Any]:
        """Self-RAG: Rewrite query if retrieval or generation failed."""
        query = state.get("query", "")
        count = state.get("reflection_count", 0)
        
        new_query = rewrite_query(query)
        
        # Apply security guardrails to the AI's rewritten query to maintain governance
        try:
            safe_query = apply_input_guardrails(new_query)
        except Exception:
            safe_query = query # fallback to original if guardrail trips
            
        _append_log(state, f"rewrite_node: rewriting query. Count={count+1}")
        return {
            "query": safe_query,
            "reflection_count": count + 1,
            "retrieved_docs": [], # Clear docs to force re-retrieval
            "cache_checked": False # Re-check cache for new query
        }

    def mcp_call_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        year = _detect_year(query)
        company = _detect_company(query)
        ticker = _detect_ticker(query)

        requests: List[Dict[str, Any]] = []
        if any(token in query.lower() for token in ["stock", "price", "market"]):
            requests.append(
                {"tool_name": "get_stock_price", "arguments": {"symbol": ticker}}
            )
        if any(token in query.lower() for token in ["10-k", "10k", "filing", "sec"]):
            requests.append(
                {
                    "tool_name": "fetch_sec_filing",
                    "arguments": {
                        "company": company,
                        "year": year,
                        "report_type": "10-k",
                    },
                }
            )
        if any(token in query.lower() for token in ["ratio", "yoy", "change", "calculate"]):
            requests.append(
                {
                    "tool_name": "calculate_ratio",
                    "arguments": {
                        "numerator": 120.0,
                        "denominator": 100.0,
                        "metric_name": "change_ratio",
                    },
                }
            )

        results = self.mcp_client.call_tools_parallel(requests) if requests else []
        _append_log(state, f"mcp_call_node: tool_calls={len(results)}")
        return {"mcp_results": results}

    def generate_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")

        if state.get("cache_hit"):
            cached = state.get("cached_response") or {}
            answer = cached.get("answer") or "No cached answer available."
            _append_log(state, "generate_node: served_from_cache")
            return {"final_answer": answer}

        docs = state.get("retrieved_docs", [])
        
        from src.utils.classifier import classify_intent
        if classify_intent(query) == "conversational":
            from src.llm.llm_client import generate_conversational_answer
            generation = generate_conversational_answer(query)
            answer = generation.get("answer") or ""
        else:
            generation = generate_answer(query, docs)
            answer = generation.get("answer") or ""

        mcp_results = state.get("mcp_results") or []
        if mcp_results:
            mcp_lines = []
            for item in mcp_results:
                tool_name = item.get("tool_name", "unknown_tool")
                result = item.get("result")
                mcp_lines.append(f"{tool_name}: {result}")
            answer = (
                f"{answer}\n\nAdditional MCP context:\n- "
                + "\n- ".join(mcp_lines)
            ).strip()

        # Matplotlib injection logic
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", answer, re.DOTALL)
        if json_match:
            try:
                chart_data = json.loads(json_match.group(1))
                if "chart_type" in chart_data and "labels" in chart_data and "values" in chart_data:
                    # Remove the JSON block from the answer entirely so the user doesn't see it
                    # but store it in state for the AI grader to read.
                    json_text = json_match.group(0)
                    state["chart_json_data"] = json_text
                    answer = answer[:json_match.start()] + answer[json_match.end():]
                    
                    # Generate the chart
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ctype = chart_data.get("chart_type", "bar").lower()
                    labels = chart_data["labels"]
                    values = chart_data["values"]
                    title = chart_data.get("title", "")
                    
                    if ctype == "pie":
                        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
                        ax.axis('equal')
                    elif ctype == "line":
                        ax.plot(labels, values, marker='o', linewidth=2, color='#4da6ff')
                        ax.set_ylabel("Value")
                        plt.xticks(rotation=45, ha='right')
                    else: # default to bar
                        ax.bar(labels, values, color='#4da6ff')
                        ax.set_ylabel("Value")
                        plt.xticks(rotation=45, ha='right')
                    
                    if title:
                        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
                    
                    plt.tight_layout()
                    
                    # Save to base64
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', transparent=True, dpi=120)
                    plt.close(fig)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    
                    # Append image to markdown
                    answer += f"\n\n![Matplotlib Chart](data:image/png;base64,{img_base64})\n\n"
            except Exception as e:
                _append_log(state, f"generate_node: matplotlib error: {e}")
                pass

        # We defer saving to cache until it passes generation grading
        _append_log(
            state,
            f"generate_node: generated_answer_chars={len(answer)} mcp_used={len(mcp_results)}",
        )
        return {"final_answer": answer}

    def grade_generation_node(self, state: AgentState) -> Dict[str, Any]:
        """Self-RAG: Grade generated answer for hallucinations and relevance."""
        query = state.get("query", "")
        from src.utils.classifier import classify_intent
        
        # Completely bypass grading and caching for conversational intents
        if classify_intent(query) == "conversational":
            _append_log(state, "grade_generation_node: Skipping grading and caching for conversational intent.")
            return {"next_action": "end"}

        if not Config.SELF_RAG_ENABLED or state.get("cache_hit"):
            # If from cache or self-rag disabled, skip grading and save to cache if needed
            if not state.get("cache_hit"):
                self.exact_cache.set(state.get("query", ""), state.get("final_answer", ""), state.get("retrieved_docs", []))
                if hasattr(self.semantic_cache, 'set'):
                    self.semantic_cache.set(state.get("query", ""), state.get("final_answer", ""), state.get("retrieved_docs", []))
            return {"next_action": "end"}

        query = state.get("query", "")
        answer = state.get("final_answer", "")
        docs = state.get("retrieved_docs", [])
        count = state.get("reflection_count", 0)

        # Strip massive base64 images before passing to the Grader LLM to prevent TPM quota exhaustion
        clean_answer = re.sub(r"!\[.*?\]\(data:image/.*?;base64,[A-Za-z0-9+/=]+\)", "[Chart Image omitted for grading]", answer)

        if "chart_json_data" in state:
            _append_log(state, "grade_generation_node: Chart present, skipping grading to minimize latency.")
            self.exact_cache.set(query, answer, docs)
            if hasattr(self.semantic_cache, 'set'):
                self.semantic_cache.set(query, answer, docs)
            return {"next_action": "end"}

        is_grounded = grade_hallucination(clean_answer, docs)
        is_relevant = grade_answer_relevance(query, clean_answer)

        if (is_grounded and is_relevant) or count >= Config.MAX_REFLECTION_LOOPS:
            _append_log(state, f"grade_generation_node: grounded={is_grounded}, relevant={is_relevant}. Done.")
            self.exact_cache.set(query, answer, docs)
            if hasattr(self.semantic_cache, 'set'):
                self.semantic_cache.set(query, answer, docs)
            return {"next_action": "end"}
        
        _append_log(state, f"grade_generation_node: Failed check (grounded={is_grounded}, relevant={is_relevant}). Rewriting.")
        return {"next_action": "rewrite"}

    def _route_after_router(
        self, state: AgentState
    ) -> Literal["cache_check_node", "retrieve_node", "mcp_node", "generate_node"]:
        action = state.get("next_action")
        if action == "cache":
            return "cache_check_node"
        if action == "retrieve":
            return "retrieve_node"
        if action == "mcp":
            return "mcp_node"
        return "generate_node"

    def _route_after_cache(self, state: AgentState) -> Literal["router_node", "generate_node"]:
        if state.get("cache_hit"):
            return "generate_node"
        return "router_node"

    def _route_after_doc_grading(self, state: AgentState) -> Literal["rewrite_node", "router_node"]:
        if state.get("next_action") == "rewrite":
            return "rewrite_node"
        return "router_node"

    def _route_after_generation_grading(self, state: AgentState) -> Literal["rewrite_node", END]:
        if state.get("next_action") == "rewrite":
            return "rewrite_node"
        return END

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE:
            return None

        builder = StateGraph(AgentState)
        builder.add_node("supervisor_node", self.supervisor_node)
        builder.add_node("router_node", self.router_node)
        builder.add_node("cache_check_node", self.cache_check_node)
        builder.add_node("retrieve_node", self.retrieve_node)
        builder.add_node("grade_documents_node", self.grade_documents_node)
        builder.add_node("rewrite_node", self.rewrite_node)
        builder.add_node("mcp_call_node", self.mcp_call_node)
        builder.add_node("mcp_node", self.mcp_call_node)
        builder.add_node("generate_node", self.generate_node)
        builder.add_node("grade_generation_node", self.grade_generation_node)

        builder.add_edge(START, "supervisor_node")
        builder.add_edge("supervisor_node", "router_node")

        builder.add_conditional_edges("router_node", self._route_after_router)
        builder.add_conditional_edges("cache_check_node", self._route_after_cache)
        builder.add_edge("retrieve_node", "grade_documents_node")
        
        # Self-RAG conditional logic after retrieval
        builder.add_conditional_edges("grade_documents_node", self._route_after_doc_grading)
        builder.add_edge("rewrite_node", "router_node") # Send back to router with new query
        
        builder.add_edge("mcp_node", "router_node")
        builder.add_edge("mcp_call_node", "router_node")
        
        # Self-RAG conditional logic after generation
        builder.add_edge("generate_node", "grade_generation_node")
        builder.add_conditional_edges("grade_generation_node", self._route_after_generation_grading)

        use_checkpoint = os.getenv("LANGGRAPH_CHECKPOINT_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        if use_checkpoint and InMemorySaver is not None:
            checkpointer = InMemorySaver()
            return builder.compile(checkpointer=checkpointer)
        return builder.compile()

    def run(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 6,
        prefetch_limit: int = 50,
        rerank: bool = True,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        initial_state: AgentState = {
            "query": query,
            "mode": mode,
            "limit": limit,
            "prefetch_limit": prefetch_limit,
            "rerank": rerank,
            "retrieved_docs": [],
            "cache_hit": False,
            "mcp_results": [],
            "final_answer": "",
            "run_log": [],
            "reflection_count": 0,
        }

        if self._graph is None:
            state = dict(initial_state)
            state.update(self.supervisor_node(state))
            while True:
                state.update(self.router_node(state))
                action = state.get("next_action")
                if action == "cache":
                    state.update(self.cache_check_node(state))
                    if state.get("cache_hit"):
                        state.update(self.generate_node(state))
                        state.update(self.grade_generation_node(state))
                        if state.get("next_action") != "rewrite":
                            break
                        state.update(self.rewrite_node(state))
                elif action == "retrieve":
                    state.update(self.retrieve_node(state))
                    state.update(self.grade_documents_node(state))
                    if state.get("next_action") == "rewrite":
                        state.update(self.rewrite_node(state))
                elif action == "mcp":
                    state.update(self.mcp_call_node(state))
                else:
                    state.update(self.generate_node(state))
                    state.update(self.grade_generation_node(state))
                    if state.get("next_action") != "rewrite":
                        break
                    state.update(self.rewrite_node(state))
            return state

        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
        result = self._graph.invoke(initial_state, config=config)
        return result
