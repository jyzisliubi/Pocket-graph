"""存储后端工厂 —— 按配置字符串创建对应后端实例

设计：业务代码通过 backend 名称创建存储实例，不感知具体实现类。
默认 backend="faiss" / "memory"，无需任何外部依赖。

环境变量：
  POCKET_VECTOR_BACKEND=faiss|chroma|pgvector   默认 faiss
  POCKET_GRAPH_BACKEND=memory|neo4j              默认 memory
  POCKET_KV_BACKEND=json|redis                   默认 json
"""

from __future__ import annotations

import os
from typing import Optional

from .base import GraphStore, KVStore, VectorStore
from .faiss_store import FAISSVectorStore
from .in_memory_graph import InMemoryGraphStore
from .json_kv_store import JsonKVStorage


def get_vector_store(
    backend: Optional[str] = None,
    model=None,
    dimension: int = 512,
    **kwargs,
) -> VectorStore:
    """按 backend 名称创建向量存储实例。

    Args:
        backend: 后端名称，None 则读 POCKET_VECTOR_BACKEND 环境变量，默认 "faiss"
        model: SentenceTransformer 模型实例（FAISS 检索文本时需要）
        dimension: 向量维度（FAISS 初始化用）
        **kwargs: 透传给具体后端的参数

    Returns:
        VectorStore 实例
    """
    backend = backend or os.environ.get("POCKET_VECTOR_BACKEND", "faiss").lower()

    if backend == "faiss":
        return FAISSVectorStore(model=model, dimension=dimension, **kwargs)
    elif backend == "chroma":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(**kwargs)
    elif backend == "pgvector":
        from .pgvector_store import PgVectorStore

        return PgVectorStore(embedding_dim=dimension, **kwargs)
    else:
        raise ValueError(f"未知向量后端: {backend}。可选: faiss / chroma / pgvector")


def get_graph_store(
    backend: Optional[str] = None,
    entity_relations: Optional[dict] = None,
    reverse_relations: Optional[dict] = None,
    **kwargs,
) -> GraphStore:
    """按 backend 名称创建图存储实例。

    Args:
        backend: 后端名称，None 则读 POCKET_GRAPH_BACKEND 环境变量，默认 "memory"
        entity_relations: 初始三元组（仅 memory 后端使用）
        reverse_relations: 反向关系（仅 memory 后端使用）
        **kwargs: 透传给具体后端的参数

    Returns:
        GraphStore 实例
    """
    backend = backend or os.environ.get("POCKET_GRAPH_BACKEND", "memory").lower()

    if backend == "memory":
        return InMemoryGraphStore(
            entity_relations=entity_relations,
            reverse_relations=reverse_relations,
            **kwargs,
        )
    elif backend == "neo4j":
        from .neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(
            entity_relations=entity_relations,
            reverse_relations=reverse_relations,
            **kwargs,
        )
    else:
        raise ValueError(f"未知图后端: {backend}。可选: memory / neo4j")


def get_kv_store(
    backend: Optional[str] = None,
    path: Optional[str] = None,
    **kwargs,
) -> KVStore:
    """按 backend 名称创建键值存储实例。

    用于存储文档原文、chunk→doc_id 映射、抽取缓存等结构化 KV 数据。

    Args:
        backend: 后端名称，None 则读 POCKET_KV_BACKEND 环境变量，默认 "json"
        path: 持久化文件路径（json 后端）。None 则纯内存
        **kwargs: 透传给具体后端的参数

    Returns:
        KVStore 实例
    """
    backend = backend or os.environ.get("POCKET_KV_BACKEND", "json").lower()

    if backend == "json":
        return JsonKVStorage(path=path, **kwargs)
    elif backend == "redis":
        # TODO: 实现 RedisKVStorage
        raise NotImplementedError(
            "Redis 后端尚未实现。欢迎贡献：参考 JsonKVStorage 实现完整接口。"
        )
    else:
        raise ValueError(f"未知 KV 后端: {backend}。可选: json / redis")


__all__ = [
    "get_vector_store",
    "get_graph_store",
    "get_kv_store",
]
