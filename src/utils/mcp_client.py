import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from src.utils.rag_service import retrieve_documents


class MCPClient:
    def __init__(
        self,
        server_url: Optional[str] = None,
        timeout_seconds: float = 5.0,
        enable_mock_tools: bool = True,
        cache_ttl_seconds: int = 300,
    ):
        self.server_url = (server_url or "").rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.enable_mock_tools = enable_mock_tools
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))
        self._tool_cache: Dict[str, Dict[str, Any]] = {}

    def _cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        item = self._tool_cache.get(key)
        if not item:
            return None
        if (time.time() - item["timestamp"]) > self.cache_ttl_seconds:
            self._tool_cache.pop(key, None)
            return None
        return dict(item["value"])

    def _write_cache(self, key: str, value: Dict[str, Any]):
        self._tool_cache[key] = {"value": dict(value), "timestamp": time.time()}

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.server_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def _get_json(self, path: str) -> Dict[str, Any]:
        url = f"{self.server_url}{path}"
        request = urllib.request.Request(url=url, method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def discover_tools(self) -> List[Dict[str, Any]]:
        if self.server_url:
            try:
                remote = self._get_json("/tools")
                tools = remote.get("tools", []) if isinstance(remote, dict) else []
                if tools:
                    return tools
            except Exception:
                pass
        return self._mock_tools() if self.enable_mock_tools else []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        key = self._cache_key(tool_name, arguments)
        cached = self._read_cache(key)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        if self.server_url:
            try:
                payload = {"tool_name": tool_name, "arguments": arguments}
                result = self._post_json("/call", payload)
                response = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "provider": "mcp_server",
                    "cache_hit": False,
                }
                self._write_cache(key, response)
                return response
            except (urllib.error.URLError, TimeoutError, ValueError):
                pass
            except Exception:
                pass

        if not self.enable_mock_tools:
            return {
                "tool_name": tool_name,
                "arguments": arguments,
                "error": "MCP server unavailable and mock tools disabled",
                "provider": "none",
                "cache_hit": False,
            }

        result = self._call_mock_tool(tool_name, arguments)
        response = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "provider": "mock_mcp",
            "cache_hit": False,
        }
        self._write_cache(key, response)
        return response

    def call_tools_parallel(self, tool_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not tool_requests:
            return []

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(4, len(tool_requests))) as executor:
            future_to_request = {
                executor.submit(
                    self.call_tool,
                    request.get("tool_name", ""),
                    request.get("arguments", {}),
                ): request
                for request in tool_requests
            }
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "tool_name": request.get("tool_name"),
                            "arguments": request.get("arguments", {}),
                            "error": str(exc),
                            "provider": "none",
                            "cache_hit": False,
                        }
                    )
        return results

    def _mock_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_stock_price",
                "description": "Return recent stock price for a ticker symbol",
                "input_schema": {"symbol": "string"},
            },
            {
                "name": "fetch_sec_filing",
                "description": "Retrieve filing excerpts for company + year + report type",
                "input_schema": {
                    "company": "string",
                    "year": "string",
                    "report_type": "string",
                },
            },
            {
                "name": "calculate_ratio",
                "description": "Compute ratio from numerator and denominator",
                "input_schema": {
                    "numerator": "number",
                    "denominator": "number",
                    "metric_name": "string",
                },
            },
        ]

    def _call_mock_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "get_stock_price":
            symbol = str(arguments.get("symbol", "AAPL")).upper()
            baseline = {
                "AAPL": 192.45,
                "MSFT": 414.88,
                "GOOGL": 176.12,
                "AMZN": 184.37,
                "META": 498.03,
            }
            price = baseline.get(symbol, 150.0)
            return {
                "symbol": symbol,
                "price": price,
                "currency": "USD",
                "as_of": time.strftime("%Y-%m-%d"),
            }

        if tool_name == "fetch_sec_filing":
            company = str(arguments.get("company", "")).strip().lower()
            year = str(arguments.get("year", "")).strip()
            report_type = str(arguments.get("report_type", "10-k")).strip().lower()
            query = f"{company} {report_type} {year}".strip()
            docs = retrieve_documents(query=query, mode="hybrid", limit=3, rerank=True)
            return {
                "company": company,
                "year": year,
                "report_type": report_type,
                "snippets": [
                    {
                        "file_name": doc.get("file_name"),
                        "year": doc.get("year"),
                        "quarter": doc.get("quarter"),
                        "snippet": doc.get("snippet"),
                    }
                    for doc in docs
                ],
            }

        if tool_name == "calculate_ratio":
            numerator = float(arguments.get("numerator", 0.0))
            denominator = float(arguments.get("denominator", 1.0))
            metric_name = str(arguments.get("metric_name", "ratio"))
            value = None if denominator == 0.0 else (numerator / denominator)
            return {
                "metric_name": metric_name,
                "numerator": numerator,
                "denominator": denominator,
                "value": value,
            }

        raise ValueError(f"Unsupported MCP tool: {tool_name}")
