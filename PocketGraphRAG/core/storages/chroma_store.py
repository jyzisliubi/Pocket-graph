"""Chroma 向量存储后端（可选）

需安装：pip install chromadb

特性：
  - 持久化到本地 SQLite，无需额外服务
  - 自带 embedding 函数（可选），也支持外部传入向量
  - 适合中等规模（百万级）场景

当前状态：骨架实现，未完成。如需启用，请实现 TODO 标记的方法。
推荐迁移路径：先用 FAISSVectorStore 跑通，再换 ChromaVectorStore 验证一致性。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .base import VectorStore


class ChromaVectorStore(VectorStore):
    """Chroma 向量存储（骨架）。

    TODO:
      - 实现 add/search/remove_by_entity/save/load
      - 支持 collection 命名（按 dataset 隔离）
      - 元数据过滤（按 entity 字段查询）
    """

    def __init__(
        self,
        collection_name: str = "pocketgraphrag",
        persist_directory: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        try:
            import chromadb  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ChromaVectorStore 需要 chromadb。请安装：pip install chromadb"
            ) from e

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_dim = embedding_dim
        # TODO: 初始化 client + collection
        raise NotImplementedError(
            "ChromaVectorStore 尚未实现。欢迎贡献：参考 FAISSVectorStore 实现完整接口。"
        )

    def add(self, items: List[Tuple[str, dict]], embeddings) -> int:
        raise NotImplementedError

    def search(self, query_vec=None, top_k: int = 5, query: str = None):
        raise NotImplementedError

    def remove_by_entity(self, entity: str) -> int:
        raise NotImplementedError

    def save(self, path: str) -> None:
        # Chroma 自带持久化，save 是 no-op
        pass

    def load(self, path: str) -> None:
        # Chroma 自带持久化，load 是 no-op
        pass

    def __len__(self) -> int:
        raise NotImplementedError
