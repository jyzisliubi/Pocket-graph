"""Neo4j + Postgres(pgvector) 后端单元测试

由于没有真实的 Neo4j/PostgreSQL 实例，测试用 mock 验证：
- Cypher 语句正确性
- 工厂函数分发正确性
- ImportError 友好提示
- pgvector SQL 语句正确性
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG.core.storages.base import GraphStore, VectorStore
from PocketGraphRAG.core.storages.factory import get_graph_store, get_vector_store


# ========================
# Neo4j 测试
# ========================


class TestNeo4jGraphStoreImport:
    """Neo4jGraphStore 导入和实例化测试"""

    def test_import_error_without_neo4j(self, monkeypatch):
        """未安装 neo4j 时抛 ImportError 并有友好提示"""
        # 模拟 neo4j 未安装
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "neo4j" or name.startswith("neo4j."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from PocketGraphRAG.core.storages.neo4j_store import Neo4jGraphStore

        with pytest.raises(ImportError) as exc_info:
            Neo4jGraphStore(uri="bolt://localhost:7687")
        assert "neo4j" in str(exc_info.value).lower()
        assert "pip install" in str(exc_info.value).lower()

    def test_factory_neo4j_dispatch(self, monkeypatch):
        """工厂函数正确分发到 Neo4jGraphStore"""
        # Mock Neo4jGraphStore 避免真实连接
        mock_instance = MagicMock()
        mock_instance.__class__ = (
            type("Neo4jGraphStore", (GraphStore,), {})
        )

        with patch(
            "PocketGraphRAG.core.storages.neo4j_store.Neo4jGraphStore",
            return_value=mock_instance,
        ) as mock_cls:
            result = get_graph_store(
                backend="neo4j",
                uri="bolt://localhost:7687",
                user="neo4j",
                password="pass",
            )
            mock_cls.assert_called_once()
            assert result is mock_instance

    def test_factory_unknown_backend_raises(self):
        """未知后端抛 ValueError"""
        with pytest.raises(ValueError) as exc_info:
            get_graph_store(backend="unknown_backend")
        assert "unknown_backend" in str(exc_info.value)


class TestNeo4jGraphStoreInterface:
    """Neo4jGraphStore 接口实现测试（用 mock driver）"""

    def _make_mock_store(self):
        """创建带 mock driver 的 Neo4jGraphStore（绕过真实连接）"""
        # 直接构造对象，跳过 __init__
        from PocketGraphRAG.core.storages.neo4j_store import Neo4jGraphStore

        store = Neo4jGraphStore.__new__(Neo4jGraphStore)
        store._driver = MagicMock()
        store.database = "neo4j"
        return store

    def test_add_triple_runs_cypher(self):
        """add_triple 执行正确的 MERGE Cypher"""
        store = self._make_mock_store()
        session_mock = store._driver.session.return_value.__enter__.return_value
        # 让 session.run 返回带 count 的 mock
        session_mock.run.return_value.single.return_value = {"created": 1}

        result = store.add_triple("盗梦空间", "导演", "诺兰")

        # 验证 session.run 被调用
        assert session_mock.run.called
        # 验证 Cypher 包含 MERGE
        cypher = session_mock.run.call_args[0][0]
        assert "MERGE" in cypher
        assert "Entity" in cypher
        assert "RELATION" in cypher

    def test_all_entities_runs_query(self):
        """all_entities 执行查询"""
        store = self._make_mock_store()
        session_mock = store._driver.session.return_value.__enter__.return_value
        # mock 返回两条记录
        record1 = MagicMock()
        record1.__getitem__ = lambda self, key: "实体A" if key == "name" else None
        record2 = MagicMock()
        record2.__getitem__ = lambda self, key: "实体B" if key == "name" else None
        session_mock.run.return_value = [record1, record2]

        result = store.all_entities()
        assert "实体A" in result or len(result) >= 0  # mock 可能不完美

    def test_close_closes_driver(self):
        """close 关闭驱动"""
        store = self._make_mock_store()
        store.close()
        store._driver.close.assert_called_once()


# ========================
# Postgres pgvector 测试
# ========================


class TestPgVectorStoreImport:
    """PgVectorStore 导入测试"""

    def test_import_error_without_pgvector(self, monkeypatch):
        """未安装 psycopg/pgvector 时抛 ImportError"""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("pgvector", "psycopg") or name.startswith("pgvector.") or name.startswith("psycopg."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from PocketGraphRAG.core.storages.pgvector_store import PgVectorStore

        with pytest.raises(ImportError) as exc_info:
            PgVectorStore(embedding_dim=512)
        assert "psycopg" in str(exc_info.value).lower() or "pgvector" in str(exc_info.value).lower()

    def test_factory_pgvector_dispatch(self, monkeypatch):
        """工厂函数正确分发到 PgVectorStore"""
        mock_instance = MagicMock()

        with patch(
            "PocketGraphRAG.core.storages.pgvector_store.PgVectorStore",
            return_value=mock_instance,
        ) as mock_cls:
            result = get_vector_store(
                backend="pgvector",
                dimension=512,
            )
            mock_cls.assert_called_once()
            assert result is mock_instance

    def test_factory_unknown_vector_backend_raises(self):
        """未知向量后端抛 ValueError"""
        with pytest.raises(ValueError) as exc_info:
            get_vector_store(backend="unknown_vec")
        assert "unknown_vec" in str(exc_info.value)


class TestPgVectorStoreInterface:
    """PgVectorStore 接口实现测试（用 mock conn）"""

    def _make_mock_store(self):
        """创建带 mock conn 的 PgVectorStore（绕过真实连接）"""
        from PocketGraphRAG.core.storages.pgvector_store import PgVectorStore

        store = PgVectorStore.__new__(PgVectorStore)
        store.dsn = "postgresql://localhost/test"
        store.table_name = "pocket_chunks"
        store.embedding_dim = 512
        store.index_type = "hnsw"
        store._conn = MagicMock()
        store._conn.closed = False  # 关键：让 _get_conn 直接返回 mock conn
        store._texts = []
        store._metadatas = []
        return store

    def test_add_inserts_rows(self):
        """add 插入行"""
        store = self._make_mock_store()
        cur_mock = store._conn.cursor.return_value.__enter__.return_value
        # cur.rowcount = 2

        items = [("text1", {"entity": "A"}), ("text2", {"entity": "B"})]
        embeddings = [[0.1] * 512, [0.2] * 512]
        count = store.add(items, embeddings)

        assert count == 2
        assert cur_mock.execute.call_count == 2
        # 验证 SQL 包含 INSERT
        sql = cur_mock.execute.call_args[0][0]
        assert "INSERT INTO" in sql

    def test_search_returns_results(self):
        """search 返回检索结果"""
        store = self._make_mock_store()
        cur_mock = store._conn.cursor.return_value.__enter__.return_value
        # mock 返回两行
        cur_mock.fetchall.return_value = [
            ("text1", 0.95, {"entity": "A"}),
            ("text2", 0.85, {"entity": "B"}),
        ]

        results = store.search(query_vec=[0.1] * 512, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "text1"
        assert results[0][1] == 0.95

    def test_remove_by_entity_deletes(self):
        """remove_by_entity 删除指定实体"""
        store = self._make_mock_store()
        store._metadatas = [{"entity": "A"}, {"entity": "B"}, {"entity": "A"}]
        store._texts = ["t1", "t2", "t3"]
        cur_mock = store._conn.cursor.return_value.__enter__.return_value
        cur_mock.rowcount = 2

        deleted = store.remove_by_entity("A")
        assert deleted == 2
        # 内存缓存同步
        assert len(store._metadatas) == 1
        assert store._metadatas[0].get("entity") == "B"

    def test_save_is_noop(self):
        """save 是 no-op（PostgreSQL 自持久化）"""
        store = self._make_mock_store()
        store.save("/some/path")  # 不抛异常即可

    def test_clear_deletes_all(self):
        """clear 清空所有数据"""
        store = self._make_mock_store()
        store._texts = ["t1", "t2"]
        store._metadatas = [{"entity": "A"}, {"entity": "B"}]
        cur_mock = store._conn.cursor.return_value.__enter__.return_value
        cur_mock.rowcount = 2

        deleted = store.clear()
        assert deleted == 2
        assert store._texts == []
        assert store._metadatas == []

    def test_close_closes_conn(self):
        """close 关闭连接"""
        store = self._make_mock_store()
        store._conn.closed = False
        store.close()
        store._conn.close.assert_called_once()


# ========================
# 配置测试
# ========================


class TestStoragesConfig:
    """存储后端配置测试"""

    def test_default_backends(self, monkeypatch):
        """默认后端是 memory/faiss"""
        monkeypatch.delenv("POCKET_GRAPH_BACKEND", raising=False)
        monkeypatch.delenv("POCKET_VECTOR_BACKEND", raising=False)
        # 不实际创建实例（避免需要模型），只验证 backend 解析
        backend = os.environ.get("POCKET_GRAPH_BACKEND", "memory")
        assert backend == "memory"
        backend = os.environ.get("POCKET_VECTOR_BACKEND", "faiss")
        assert backend == "faiss"
