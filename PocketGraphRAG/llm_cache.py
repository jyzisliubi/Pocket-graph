"""
LLM 响应缓存层（v0.3.7：对标 LightRAG v1.4.3 Redis cache）

对相同 (system_prompt, user_prompt, model, temperature, max_tokens) 的 LLM 调用
返回缓存结果，避免重复调用付费 API。

两种后端：
  - InMemoryCache（默认）：进程内 LRU + TTL，零依赖
  - RedisCache：跨进程/跨机器共享缓存，需 pip install redis

配置：
  POCKET_LLM_CACHE=1                 # 启用（默认关闭）
  POCKET_LLM_CACHE_BACKEND=memory    # memory / redis
  POCKET_LLM_CACHE_TTL=3600          # 缓存有效期秒数
  POCKET_LLM_CACHE_MAX_SIZE=1000     # InMemoryCache 最大条目数
  POCKET_LLM_CACHE_REDIS_URL=redis://localhost:6379/0
  POCKET_LLM_CACHE_ROLES=query       # 缓存的角色（* 表示全部）
"""

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Optional


LLM_CACHE_ENABLED = os.environ.get("POCKET_LLM_CACHE", "").lower() in (
    "1", "true", "yes", "on",
)
LLM_CACHE_BACKEND = os.environ.get("POCKET_LLM_CACHE_BACKEND", "memory").lower()
LLM_CACHE_TTL = int(os.environ.get("POCKET_LLM_CACHE_TTL", "3600"))
LLM_CACHE_MAX_SIZE = int(os.environ.get("POCKET_LLM_CACHE_MAX_SIZE", "1000"))
LLM_CACHE_REDIS_URL = os.environ.get(
    "POCKET_LLM_CACHE_REDIS_URL", "redis://localhost:6379/0"
)
LLM_CACHE_ROLES = tuple(
    r.strip()
    for r in os.environ.get("POCKET_LLM_CACHE_ROLES", "query").split(",")
    if r.strip()
)


def _cache_key(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """生成缓存 key（SHA256 哈希）"""
    payload = json.dumps(
        {
            "s": system_prompt,
            "u": user_prompt,
            "m": model or "",
            "t": float(temperature),
            "max": int(max_tokens),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    """LLM 缓存抽象基类"""

    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: int = LLM_CACHE_TTL) -> None:
        raise NotImplementedError

    def clear(self) -> int:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError


class InMemoryCache(LLMCache):
    """进程内 LRU + TTL 缓存，线程安全"""

    def __init__(self, max_size: int = LLM_CACHE_MAX_SIZE, ttl: int = LLM_CACHE_TTL):
        self._store: OrderedDict = OrderedDict()
        self._timestamps: dict = {}
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            ts = self._timestamps.get(key, 0)
            if self._ttl > 0 and (time.time() - ts) > self._ttl:
                self._store.pop(key, None)
                self._timestamps.pop(key, None)
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]

    def set(self, key: str, value: str, ttl: int = LLM_CACHE_TTL) -> None:
        with self._lock:
            self._store[key] = value
            self._timestamps[key] = time.time()
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                k, _ = self._store.popitem(last=False)
                self._timestamps.pop(k, None)

    def clear(self) -> int:
        with self._lock:
            n = len(self._store)
            self._store.clear()
            self._timestamps.clear()
            return n

    def stats(self) -> dict:
        with self._lock:
            return {
                "backend": "memory",
                "size": len(self._store),
                "max_size": self._max_size,
                "ttl": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
            }


class RedisCache(LLMCache):
    """Redis 缓存后端，需 pip install redis"""

    _CLIENT = None

    def __init__(self, redis_url: str = LLM_CACHE_REDIS_URL, ttl: int = LLM_CACHE_TTL):
        self._redis_url = redis_url
        self._ttl = ttl
        self._prefix = "pocket:llm:"
        if RedisCache._CLIENT is None:
            import redis
            RedisCache._CLIENT = redis.Redis.from_url(redis_url, decode_responses=True)
        self._client = RedisCache._CLIENT

    def get(self, key: str) -> Optional[str]:
        try:
            val = self._client.get(self._prefix + key)
            return val if val is not None else None
        except Exception:
            return None

    def set(self, key: str, value: str, ttl: int = LLM_CACHE_TTL) -> None:
        try:
            self._client.setex(self._prefix + key, ttl or self._ttl, value)
        except Exception:
            pass

    def clear(self) -> int:
        try:
            keys = self._client.keys(self._prefix + "*")
            if keys:
                self._client.delete(*keys)
            return len(keys)
        except Exception:
            return 0

    def stats(self) -> dict:
        try:
            return {
                "backend": "redis",
                "size": len(self._client.keys(self._prefix + "*")),
                "ttl": self._ttl,
                "redis_url": self._redis_url,
            }
        except Exception as e:
            return {"backend": "redis", "error": str(e)}


_cache_instance: Optional[LLMCache] = None
_cache_lock = threading.Lock()


def get_cache() -> Optional[LLMCache]:
    """获取全局缓存实例（单例）。未启用返回 None，Redis 不可用降级到内存。"""
    global _cache_instance
    if not LLM_CACHE_ENABLED:
        return None
    if _cache_instance is not None:
        return _cache_instance
    with _cache_lock:
        if _cache_instance is not None:
            return _cache_instance
        if LLM_CACHE_BACKEND == "redis":
            try:
                _cache_instance = RedisCache()
            except ImportError:
                _cache_instance = InMemoryCache()
            except Exception:
                _cache_instance = InMemoryCache()
        else:
            _cache_instance = InMemoryCache()
        return _cache_instance


def should_cache_role(role: str) -> bool:
    """判断该角色的 LLM 调用是否应缓存"""
    if not LLM_CACHE_ENABLED:
        return False
    if "*" in LLM_CACHE_ROLES:
        return True
    return role in LLM_CACHE_ROLES


def get_cached_or_call(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    role: str,
    caller_fn,
):
    """缓存包装器：先查缓存，未命中则调用 caller_fn 并写入缓存"""
    cache = get_cache()
    if cache is None or not should_cache_role(role):
        return caller_fn()

    key = _cache_key(system_prompt, user_prompt, model, temperature, max_tokens)
    cached = cache.get(key)
    if cached is not None:
        return cached

    result = caller_fn()
    if result is not None:
        cache.set(key, result)
    return result
