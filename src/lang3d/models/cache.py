"""Semantic cache for LLM API responses.

Caches responses based on a hash of the prompt/messages to avoid redundant
API calls for identical or near-identical requests (e.g., repeated part
library queries, standard parameter generation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached response."""

    response: Any
    created_at: float
    hits: int = 0


class SemanticCache:
    """Simple LRU cache for LLM responses.

    Keyed on a deterministic hash of the input messages and tools.
    Supports TTL-based expiration and capacity limits.

    Usage::

        cache = SemanticCache(ttl_seconds=3600, max_entries=256)
        key = cache.make_key(messages, tools)
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = llm.chat(messages, tools)
        cache.put(key, result)
        return result
    """

    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        max_entries: int = 256,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 0,
    ) -> str:
        """Create a deterministic cache key from request parameters."""
        parts = [model, str(temperature), str(max_tokens)]
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}:{content}")
            else:
                parts.append(f"{role}:{json.dumps(content, sort_keys=True, default=str)}")
        if tools:
            for t in tools:
                parts.append(json.dumps(t, sort_keys=True, default=str))
        blob = "|".join(parts)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> Any | None:
        """Retrieve a cached response, or None if not found / expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if time.monotonic() - entry.created_at > self.ttl_seconds:
                del self._store[key]
                self._misses += 1
                return None

            # Promote to most-recently-used
            self._store.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            logger.debug("Cache hit: %s (hits=%d)", key[:12], entry.hits)
            return entry.response

    def put(self, key: str, response: Any) -> None:
        """Store a response in the cache, evicting LRU if at capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = CacheEntry(response=response, created_at=time.monotonic())
            # Evict oldest entries if over capacity
            while len(self._store) > self.max_entries:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("Cache eviction: %s", evicted_key[:12])

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "ttl_seconds": self.ttl_seconds,
        }


# Module-level singleton for global use
_global_cache: SemanticCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> SemanticCache:
    """Get or create the global LLM response cache (thread-safe)."""
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = SemanticCache()
    return _global_cache
