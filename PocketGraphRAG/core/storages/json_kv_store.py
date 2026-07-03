"""JSON 文件键值存储后端（默认实现）

零外部依赖，适合中小规模 KV 数据（文档原文、chunk→doc_id 映射、抽取缓存）。
数据以单个 JSON 文件持久化，加载时全量读入内存。

对于大规模 KV 数据（百万级），请实现 RedisKVStorage 或 PostgresKVStorage。

用法::

    from PocketGraphRAG.core.storages import JsonKVStorage

    # 纯内存模式
    store = JsonKVStorage()
    store.upsert("doc_001", {"text": "盗梦空间...", "source": "manual.pdf"})

    # 持久化模式
    store = JsonKVStorage(path="data/doc_store.json")
    store.upsert("doc_001", {"text": "..."})
    store.save()  # 写入 data/doc_store.json
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

from .base import KVStore


class JsonKVStorage(KVStore):
    """内存 dict + JSON 文件持久化的键值存储。

    线程安全（读写均加锁）。支持纯内存模式（不传 path）和持久化模式（传 path）。

    持久化文件格式：
        单个 JSON 对象，key 为存储键，value 为 dict 记录。
        示例：{"doc_001": {"text": "...", "source": "a.pdf"}, ...}
    """

    def __init__(self, path: Optional[str] = None):
        """
        Args:
            path: 持久化文件路径。传 None 则纯内存（save/load 无效）。
                  传路径则构造时自动 load（若文件存在）。
        """
        self._data: Dict[str, dict] = {}
        self._path = path
        self._lock = threading.RLock()

        if path and os.path.exists(path):
            self.load(path)

    # ==========================
    # KVStore 抽象接口实现
    # ==========================

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            v = self._data.get(key)
            return dict(v) if v is not None else None  # 返回副本，避免外部修改

    def upsert(self, key: str, value: dict) -> bool:
        if not isinstance(value, dict):
            raise TypeError(f"value 必须是 dict，收到 {type(value).__name__}")
        with self._lock:
            is_new = key not in self._data
            self._data[key] = dict(value)
            return is_new

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def get_by_ids(self, keys: List[str]) -> List[Optional[dict]]:
        with self._lock:
            return [self.get(k) for k in keys]

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

    def save(self, path: Optional[str] = None) -> None:
        """持久化到 JSON 文件。

        Args:
            path: 目标路径。None 则用构造时传入的 path；都为 None 则报错。
        """
        target = path or self._path
        if not target:
            raise ValueError("save 需要 path：请传入 path 参数或在构造时指定 path")
        with self._lock:
            # 先写临时文件再 rename，保证原子性
            tmp = target + ".tmp"
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, target)
            self._path = target

    def load(self, path: str) -> None:
        """从 JSON 文件加载（会清空当前内存数据）"""
        with self._lock:
            if not os.path.exists(path):
                raise FileNotFoundError(f"KV 存储文件不存在: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(
                    f"KV 存储文件格式错误：期望 JSON 对象，收到 {type(data).__name__}"
                )
            self._data = {k: v for k, v in data.items() if isinstance(v, dict)}
            self._path = path

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    # ==========================
    # 便利方法
    # ==========================

    def items(self) -> List[tuple]:
        """返回 [(key, value), ...] 列表（副本）"""
        with self._lock:
            return [(k, dict(v)) for k, v in self._data.items()]

    def clear(self) -> int:
        """清空所有条目。返回被清除的条目数"""
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n
