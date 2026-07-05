"""pgvector 向量存储后端（可选）

需安装：pip install "psycopg[binary]" pgvector

特性：
  - 复用 PostgreSQL 的成熟运维（备份、复制、权限）
  - 支持 HNSW / IVFFlat 索引
  - 适合企业级生产部署
  - 支持 entity → chunk_ids 倒排索引（O(1) 按实体删除）

对标 LightRAG 的 NanoVectorDB，但用 PostgreSQL 替代内存向量库，
实现企业级持久化、并发安全、ACID 事务。

配置：
  POCKET_PG_DSN=postgresql://user:pass@host:5432/dbname
  POCKET_PG_TABLE=pocket_chunks
  POCKET_PG_INDEX=hnsw  # 或 ivfflat
"""

from __future__ import annotations

import os
import uuid
from typing import List, Optional, Tuple

from .base import VectorStore


class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector 向量存储。

    自动建表、建索引，支持 HNSW（默认）和 IVFFlat 索引。
    用 entity → chunk_ids 倒排索引实现 O(1) 按实体删除。
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        table_name: str = "pocket_chunks",
        embedding_dim: int = 512,
        index_type: str = "hnsw",
    ):
        try:
            import pgvector  # noqa: F401
            from pgvector.psycopg import register_vector  # noqa: F401
            import psycopg  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PgVectorStore 需要 psycopg 和 pgvector。"
                "请安装：pip install 'psycopg[binary]' pgvector"
            ) from e

        self.dsn = dsn or os.environ.get(
            "POCKET_PG_DSN", "postgresql://localhost:5432/pocketgraphrag"
        )
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.index_type = index_type.lower()
        self._conn = None
        self._texts: List[str] = []
        self._metadatas: List[dict] = []
        self._init_db()

    def _get_conn(self):
        """获取连接（惰性初始化）"""
        if self._conn is None or self._conn.closed:
            import psycopg

            self._conn = psycopg.connect(self.dsn, autocommit=False)
            # 注册 pgvector 类型
            from pgvector.psycopg import register_vector

            register_vector(self._conn)
        return self._conn

    def _init_db(self):
        """建表 + 建索引（幂等）"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # 启用 pgvector 扩展
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                # 建表
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id UUID PRIMARY KEY,
                        entity TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.embedding_dim})
                    )
                    """
                )
                # entity → chunk 倒排索引（用于 O(1) 按实体删除）
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_entity "
                    f"ON {self.table_name} (entity)"
                )
                # 向量索引
                if self.index_type == "hnsw":
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_vec "
                        f"ON {self.table_name} USING hnsw (embedding vector_cosine_ops)"
                    )
                elif self.index_type == "ivfflat":
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_vec "
                        f"ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)"
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def add(self, items: List[Tuple[str, dict]], embeddings) -> int:
        """批量添加文本块。

        Args:
            items: [(text, metadata), ...]
            embeddings: 向量矩阵 (n, dim)，与 items 等长
        """
        if not items:
            return 0
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                count = 0
                for (text, meta), emb in zip(items, embeddings):
                    entity = meta.get("entity", "") if isinstance(meta, dict) else ""
                    row_id = uuid.uuid4()
                    import json

                    cur.execute(
                        f"INSERT INTO {self.table_name} "
                        f"(id, entity, text, metadata, embedding) "
                        f"VALUES (%s, %s, %s, %s, %s)",
                        (row_id, entity, text, json.dumps(meta), emb),
                    )
                    count += 1
                    # 同步内存缓存
                    self._texts.append(text)
                    self._metadatas.append(meta)
            conn.commit()
            return count
        except Exception:
            conn.rollback()
            raise

    def search(
        self, query_vec=None, top_k: int = 5, query: str = None
    ) -> List[Tuple[str, float, dict]]:
        """向量检索 top_k 个最相似文本块。

        Returns:
            [(text, score, metadata), ...] 按相似度降序
        """
        if query_vec is None:
            return []
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT text, 1 - (embedding <=> %s) AS score, metadata "
                    f"FROM {self.table_name} "
                    f"ORDER BY embedding <=> %s "
                    f"LIMIT %s",
                    (query_vec, query_vec, top_k),
                )
                rows = cur.fetchall()
                import json

                return [
                    (row[0], float(row[1]), row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {})
                    for row in rows
                ]
        except Exception:
            conn.rollback()
            raise

    def remove_by_entity(self, entity: str) -> int:
        """删除某实体对应的所有条目（增量重建场景）。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE entity = %s", (entity,)
                )
                deleted = cur.rowcount
            conn.commit()
            # 同步内存缓存（重建）
            self._texts = [t for t, m in zip(self._texts, self._metadatas)
                          if m.get("entity") != entity]
            self._metadatas = [m for m in self._metadatas if m.get("entity") != entity]
            return deleted
        except Exception:
            conn.rollback()
            raise

    def save(self, path: str) -> None:
        """pgvector 后端数据持久化在数据库中，save 为 no-op。

        path 参数仅为接口兼容，实际数据由 PostgreSQL 管理。
        """
        # PostgreSQL 自持久化，无需额外操作
        pass

    def load(self, path: str) -> None:
        """从数据库重新加载内存缓存（texts/metadatas）。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT text, metadata FROM {self.table_name} ORDER BY id"
                )
                rows = cur.fetchall()
                import json

                self._texts = [row[0] for row in rows]
                self._metadatas = [
                    row[1] if isinstance(row[1], dict) else json.loads(row[1]) if row[1] else {}
                    for row in rows
                ]
        except Exception:
            conn.rollback()
            raise

    def clear(self) -> int:
        """清空所有数据。返回删除条数。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_name}")
                deleted = cur.rowcount
            conn.commit()
            self._texts = []
            self._metadatas = []
            return deleted
        except Exception:
            conn.rollback()
            raise

    @property
    def texts(self) -> List[str]:
        return self._texts

    @property
    def metadatas(self) -> List[dict]:
        return self._metadatas

    def __len__(self) -> int:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cur.fetchone()[0]
        except Exception:
            conn.rollback()
            return len(self._texts)

    def close(self):
        """关闭连接"""
        if self._conn and not self._conn.closed:
            self._conn.close()
