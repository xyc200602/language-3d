"""Tests for models/cache.py — SemanticCache."""

from __future__ import annotations

import time

import pytest

from lang3d.models.cache import SemanticCache, get_cache


class TestMakeKey:
    def test_deterministic(self):
        msgs = [{"role": "user", "content": "hello"}]
        key1 = SemanticCache.make_key(msgs, model="gpt-4")
        key2 = SemanticCache.make_key(msgs, model="gpt-4")
        assert key1 == key2

    def test_different_content(self):
        k1 = SemanticCache.make_key([{"role": "user", "content": "hello"}])
        k2 = SemanticCache.make_key([{"role": "user", "content": "world"}])
        assert k1 != k2

    def test_different_model(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = SemanticCache.make_key(msgs, model="gpt-4")
        k2 = SemanticCache.make_key(msgs, model="glm-4")
        assert k1 != k2

    def test_with_tools(self):
        msgs = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        k1 = SemanticCache.make_key(msgs, tools=tools)
        k2 = SemanticCache.make_key(msgs, tools=None)
        assert k1 != k2

    def test_different_temperature(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = SemanticCache.make_key(msgs, temperature=0.0)
        k2 = SemanticCache.make_key(msgs, temperature=0.7)
        assert k1 != k2


class TestCacheGetPut:
    def test_put_and_get(self):
        cache = SemanticCache()
        key = SemanticCache.make_key([{"role": "user", "content": "test"}])
        cache.put(key, "response_data")
        assert cache.get(key) == "response_data"

    def test_miss_returns_none(self):
        cache = SemanticCache()
        assert cache.get("nonexistent") is None

    def test_overwrite(self):
        cache = SemanticCache()
        key = "test_key"
        cache.put(key, "v1")
        cache.put(key, "v2")
        assert cache.get(key) == "v2"


class TestTTLExpiration:
    def test_expired_entry(self):
        cache = SemanticCache(ttl_seconds=0.01)  # 10ms TTL
        cache.put("key", "value")
        time.sleep(0.05)
        assert cache.get("key") is None

    def test_not_expired(self):
        cache = SemanticCache(ttl_seconds=60.0)
        cache.put("key", "value")
        assert cache.get("key") == "value"


class TestCapacity:
    def test_eviction(self):
        cache = SemanticCache(max_entries=3)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        cache.put("k4", "v4")  # should evict k1
        assert cache.get("k1") is None
        assert cache.get("k4") == "v4"
        assert cache.size == 3

    def test_lru_order(self):
        cache = SemanticCache(max_entries=2)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        # Access k1 to make it recently used
        cache.get("k1")
        # Adding k3 should evict k2 (LRU)
        cache.put("k3", "v3")
        assert cache.get("k1") == "v1"
        assert cache.get("k2") is None
        assert cache.get("k3") == "v3"


class TestClear:
    def test_clear_empties_cache(self):
        cache = SemanticCache()
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.clear()
        assert cache.size == 0
        assert cache.get("k1") is None


class TestStats:
    def test_stats(self):
        cache = SemanticCache()
        cache.put("k1", "v1")
        cache.get("k1")   # hit
        cache.get("k1")   # hit
        cache.get("x")    # miss
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)


class TestHitRate:
    def test_zero_when_empty(self):
        cache = SemanticCache()
        assert cache.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        cache = SemanticCache()
        cache.put("k1", "v1")
        cache.get("k1")  # hit
        cache.get("x")   # miss
        assert cache.hit_rate == pytest.approx(0.5)


class TestGetCache:
    def test_singleton(self):
        c1 = get_cache()
        c2 = get_cache()
        assert c1 is c2

    def test_returns_cache(self):
        c = get_cache()
        assert isinstance(c, SemanticCache)
