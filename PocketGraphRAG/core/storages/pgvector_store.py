"""pgvector 向量存储后端（可选）

需安装：pip install "psycopg[binary]" pgvector

特性：
  - 复用 PostgreSQL 的成熟运维（备份、复制、权限）
  - 支持 HNSW / IVFFlat 索引
  - 适合企业级生产部署

当前状态：骨架实现，未完成。如需启用，请实现 TODO 标记的方法。
"""

from __future__ import annotations

from typing import List, Tuple

from .base import VectorStore


class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector 向量存储（骨架）。

    TODO:
      - 实现 add/search/remove_by_entity/save/load
      - 自动建表 + 建索引（HNSW）
      - 连接池管理
      - 事务支持
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost:5432/pocketgraphrag",
        table_name: str = "pocket_chunks",
        embedding_dim: int = 512,
    ):
        try:
            import pgvector  # noqa: F401
            import psycopg  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PgVectorStore 需要 psycopg 和 pgvector。"
                "请安装：pip install 'psycopg[binary]' pgvector"
            ) from e

        self.dsn = dsn
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        # TODO: 初始化连接 + 建表 + 建索引
        raise NotImplementedError(
            "PgVectorStore 尚未实现。欢迎贡献：参考 FAISSVectorStore 实现完整接口。"
        )

    def add(self, items: List[Tuple[str, dict]], embeddings) -> int:
        raise NotImplementedError

    def search(self, query_vec=None, top_k: int = 5, query: str = None):
        raise NotImplementedError

    def remove_by_entity(self, entity: str) -> int:
        raise NotImplementedError

    def save(self, path: str) -> None:
        # pgvector 自带持久化（PostgreSQL），save 是 no-op
        pass

    def load(self, path: str) -> None:
        # pgvector 自带持久化，load 是 no-op
        pass

    def __len__(self) -> int:
        raise NotImplementedError
