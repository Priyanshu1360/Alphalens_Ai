import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from src.embeddings.embedder import embed_texts


def normalize_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


class ExactMatchCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _is_expired(self, created_at: float) -> bool:
        return (time.time() - created_at) > self.ttl_seconds

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = normalize_query(query)
        with self._lock:
            value = self._store.get(key)
            if not value:
                self._misses += 1
                return None
            if self._is_expired(value["timestamp"]):
                self._store.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return dict(value)

    def set(self, query: str, answer: str, docs: Optional[List[Dict[str, Any]]] = None):
        key = normalize_query(query)
        payload = {
            "answer": answer,
            "docs": list(docs or []),
            "timestamp": time.time(),
        }
        with self._lock:
            self._store[key] = payload

    def invalidate(self, query: Optional[str] = None):
        with self._lock:
            if query is None:
                self._store.clear()
                return
            self._store.pop(normalize_query(query), None)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            ratio = (self._hits / total) if total else 0.0
            return {
                "entries": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(ratio, 4),
                "ttl_seconds": self.ttl_seconds,
            }


class SemanticCache:
    def __init__(self, ttl_seconds: int = 3600, similarity_threshold: float = 0.9):
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.similarity_threshold = float(similarity_threshold)
        self._items: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _is_expired(self, created_at: float) -> bool:
        return (time.time() - created_at) > self.ttl_seconds

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _prune_expired(self):
        self._items = [
            item for item in self._items if not self._is_expired(item["timestamp"])
        ]

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        normalized = normalize_query(query)
        if not normalized:
            self._misses += 1
            return None
        query_vector = embed_texts([normalized], batch_size=1)[0]

        with self._lock:
            self._prune_expired()
            best: Optional[Tuple[float, Dict[str, Any]]] = None
            for item in self._items:
                score = self._cosine(query_vector, item["vector"])
                if best is None or score > best[0]:
                    best = (score, item)
            if best is None or best[0] < self.similarity_threshold:
                self._misses += 1
                return None
            self._hits += 1
            result = dict(best[1]["value"])
            result["semantic_similarity"] = round(best[0], 4)
            return result

    def set(self, query: str, answer: str, docs: Optional[List[Dict[str, Any]]] = None):
        normalized = normalize_query(query)
        if not normalized:
            return
        vector = embed_texts([normalized], batch_size=1)[0]
        item = {
            "query": normalized,
            "vector": vector,
            "value": {
                "answer": answer,
                "docs": list(docs or []),
            },
            "timestamp": time.time(),
        }
        with self._lock:
            self._prune_expired()
            self._items.append(item)

    def invalidate(self):
        with self._lock:
            self._items = []

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            ratio = (self._hits / total) if total else 0.0
            return {
                "entries": len(self._items),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(ratio, 4),
                "ttl_seconds": self.ttl_seconds,
                "similarity_threshold": self.similarity_threshold,
            }
