import logging
import os
import re
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict

from src.cache_memory import ExactMatchCache, normalize_query
from src.generator import generate_answer
from src.mcp_client import MCPClient
from src.rag_service import retrieve_documents

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
    next_action: str
    route_reason: str
    needs_retrieve: bool
    needs_mcp: bool
    needs_direct: bool
    cache_checked: bool
    run_log: List[str]


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
    def __init__(self, cache: ExactMatchCache, mcp_client: MCPClient):
        self.cache = cache
        self.mcp_client = mcp_client
        self.langgraph_enabled = LANGGRAPH_AVAILABLE
        self._graph = self._build_graph()

    def supervisor_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
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
        cached = self.cache.get(query)
        hit = cached is not None
        _append_log(state, f"cache_check_node: cache_hit={hit}")
        return {
            "cache_checked": True,
            "cache_hit": hit,
            "cached_response": cached or {},
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

        self.cache.set(query, answer, docs)
        _append_log(
            state,
            f"generate_node: generated_answer_chars={len(answer)} mcp_used={len(mcp_results)}",
        )
        return {"final_answer": answer}

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

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE:
            return None

        builder = StateGraph(AgentState)
        builder.add_node("supervisor_node", self.supervisor_node)
        builder.add_node("router_node", self.router_node)
        builder.add_node("cache_check_node", self.cache_check_node)
        builder.add_node("retrieve_node", self.retrieve_node)
        builder.add_node("mcp_call_node", self.mcp_call_node)
        builder.add_node("mcp_node", self.mcp_call_node)
        builder.add_node("generate_node", self.generate_node)

        builder.add_edge(START, "supervisor_node")
        builder.add_edge("supervisor_node", "router_node")

        builder.add_conditional_edges("router_node", self._route_after_router)
        builder.add_conditional_edges("cache_check_node", self._route_after_cache)
        builder.add_edge("retrieve_node", "router_node")
        builder.add_edge("mcp_node", "router_node")
        builder.add_edge("mcp_call_node", "router_node")
        builder.add_edge("generate_node", END)

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
                        break
                elif action == "retrieve":
                    state.update(self.retrieve_node(state))
                elif action == "mcp":
                    state.update(self.mcp_call_node(state))
                else:
                    state.update(self.generate_node(state))
                    break
            return state

        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
        result = self._graph.invoke(initial_state, config=config)
        return result
