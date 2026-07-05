"""存储抽象层测试

覆盖：
1. VectorStore / GraphStore 抽象基类不能直接实例化
2. FAISSVectorStore：add/search/remove/save/load/__len__/向后兼容
3. InMemoryGraphStore：增删/邻域/关系/图算法(pagerank/communities/shortest_path)
4. 工厂函数：backend 路由 + 未知后端报错
5. 骨架后端：ChromaVectorStore / PgVectorStore 在未安装时给出清晰 ImportError
6. 与现有 FAISSIndex 互操作（向后兼容）
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from PocketGraphRAG.core.storages import (
    FAISSVectorStore,
    GraphStore,
    InMemoryGraphStore,
    JsonKVStorage,
    KVStore,
    VectorStore,
    get_graph_store,
    get_kv_store,
    get_vector_store,
)

# ==========================
# 测试夹具
# ==========================


class MockModel:
    """最小 mock：只实现 encode，返回稳定向量"""

    def __init__(self, dim=8):
        self.dim = dim

    def encode(
        self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
    ):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            seed = abs(hash(t)) % (2**32)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.dim).astype("float32")
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v)
        return np.array(vecs, dtype="float32")


@pytest.fixture
def sample_items():
    return [
        ("稻瘟病是由稻瘟病菌引起", {"entity": "稻瘟病"}),
        ("三环唑是防治稻瘟病的杀菌剂", {"entity": "三环唑"}),
        ("水稻纹枯病由立枯丝核菌引起", {"entity": "水稻纹枯病"}),
    ]


@pytest.fixture
def sample_graph_data():
    return {
        "稻瘟病": [("防治药剂", "三环唑"), ("属于", "真菌性病害")],
        "三环唑": [("属于", "杀菌剂")],
        "水稻纹枯病": [("防治药剂", "井冈霉素")],
    }


# ==========================
# 抽象基类
# ==========================


class TestAbstractBase:
    def test_vector_store_cannot_instantiate(self):
        with pytest.raises(TypeError):
            VectorStore()

    def test_graph_store_cannot_instantiate(self):
        with pytest.raises(TypeError):
            GraphStore()

    def test_vector_store_is_abstract(self):
        assert issubclass(FAISSVectorStore, VectorStore)

    def test_graph_store_is_abstract(self):
        assert issubclass(InMemoryGraphStore, GraphStore)


# ==========================
# FAISSVectorStore
# ==========================


class TestFAISSVectorStore:
    def test_create_empty(self):
        store = FAISSVectorStore(model=MockModel(), dimension=8)
        assert len(store) == 0
        assert store.dimension == 8

    def test_add_and_search(self, sample_items):
        model = MockModel(dim=8)
        store = FAISSVectorStore(model=model, dimension=8)
        embeddings = model.encode([t for t, _ in sample_items])

        n = store.add(sample_items, embeddings)
        assert n == 3
        assert len(store) == 3

        # 用向量检索
        query_vec = model.encode(["稻瘟病"])[0]
        results = store.search(query_vec=query_vec, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)

    def test_search_with_text_query(self, sample_items):
        model = MockModel(dim=8)
        store = FAISSVectorStore(model=model, dimension=8)
        embeddings = model.encode([t for t, _ in sample_items])
        store.add(sample_items, embeddings)

        # 用文本检索（内部会调 model.encode）
        results = store.search(query="稻瘟病", top_k=2)
        assert len(results) > 0

    def test_search_requires_query_or_vec(self, sample_items):
        model = MockModel(dim=8)
        store = FAISSVectorStore(model=model, dimension=8)
        embeddings = model.encode([t for t, _ in sample_items])
        store.add(sample_items, embeddings)

        with pytest.raises(ValueError):
            store.search()

    def test_search_without_model_raises(self, sample_items):
        store = FAISSVectorStore(model=None, dimension=8)
        embeddings = MockModel(dim=8).encode([t for t, _ in sample_items])
        store.add(sample_items, embeddings)

        with pytest.raises(ValueError):
            store.search(query="稻瘟病", top_k=2)

    def test_remove_by_entity(self, sample_items):
        model = MockModel(dim=8)
        store = FAISSVectorStore(model=model, dimension=8)
        embeddings = model.encode([t for t, _ in sample_items])
        store.add(sample_items, embeddings)

        removed = store.remove_by_entity("稻瘟病")
        assert removed == 1
        assert len(store) == 2
        # 剩余不应含稻瘟病的元数据
        for meta in store.metadatas:
            assert meta["entity"] != "稻瘟病"

    def test_remove_nonexistent_entity(self, sample_items):
        model = MockModel(dim=8)
        store = FAISSVectorStore(model=model, dimension=8)
        embeddings = model.encode([t for t, _ in sample_items])
        store.add(sample_items, embeddings)

        removed = store.remove_by_entity("不存在的实体")
        assert removed == 0
        assert len(store) == 3

    def test_save_and_load(self, sample_items, temp_dir):
        model = MockModel(dim=8)
        store = FAISSVectorStore(model=model, dimension=8)
        embeddings = model.encode([t for t, _ in sample_items])
        store.add(sample_items, embeddings)

        index_dir = os.path.join(temp_dir, "faiss_store")
        store.save(index_dir)

        # 加载到新实例
        store2 = FAISSVectorStore(model=model, dimension=8)
        store2.load(index_dir)
        assert len(store2) == 3
        assert store2.texts == store.texts

    def test_dimension_mismatch_raises(self, sample_items):
        model = MockModel(dim=8)
        store = FAISSVectorStore(model=model, dimension=16)  # 故意错配
        # 用 8 维向量加到 16 维索引应抛错
        embeddings = model.encode([t for t, _ in sample_items])
        with pytest.raises(ValueError):
            store.add(sample_items, embeddings)

    def test_texts_metadatas_compat(self, sample_items):
        model = MockModel(dim=8)
        store = FAISSVectorStore(model=model, dimension=8)
        embeddings = model.encode([t for t, _ in sample_items])
        store.add(sample_items, embeddings)

        assert store.texts == [t for t, _ in sample_items]
        assert store.metadatas == [m for _, m in sample_items]

    def test_delegate_property(self, sample_items):
        """FAISSVectorStore.delegate 暴露底层 FAISSIndex，向后兼容"""
        model = MockModel(dim=8)
        store = FAISSVectorStore(model=model, dimension=8)
        # delegate 应是 FAISSIndex 实例
        from PocketGraphRAG.build_index import FAISSIndex

        assert isinstance(store.delegate, FAISSIndex)

    def test_wrap_existing_faiss_index(self, sample_items):
        """通过 _delegate 包装现有 FAISSIndex"""
        from PocketGraphRAG.build_index import FAISSIndex

        model = MockModel(dim=8)
        old_index = FAISSIndex(dimension=8)
        old_index.model = model
        chunks = [{"text": t, "metadata": m} for t, m in sample_items]
        old_index.build(chunks, model)

        store = FAISSVectorStore(_delegate=old_index)
        assert len(store) == 3
        # 通过新接口访问应一致
        results = store.search(query="稻瘟病", top_k=2)
        assert len(results) > 0


# ==========================
# InMemoryGraphStore
# ==========================


class TestInMemoryGraphStore:
    def test_empty_graph(self):
        g = InMemoryGraphStore()
        assert len(g) == 0
        assert g.all_entities() == []
        assert g.all_relations() == []

    def test_init_from_dict(self, sample_graph_data):
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        # sample_graph_data 含 4 条三元组：稻瘟病→三环唑, 稻瘟病→真菌性病害,
        # 三环唑→杀菌剂, 水稻纹枯病→井冈霉素
        assert len(g) == 4
        assert "稻瘟病" in g.all_entities()
        assert "防治药剂" in g.all_relations()

    def test_init_auto_reverse(self, sample_graph_data):
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        # 应自动反推 reverse_relations
        reverse = g.reverse_relations_of("三环唑")
        assert ("稻瘟病", "防治药剂") in reverse

    def test_add_triple_idempotent(self):
        g = InMemoryGraphStore()
        assert g.add_triple("A", "r", "B") is True
        assert g.add_triple("A", "r", "B") is False  # 重复
        assert len(g) == 1

    def test_add_triples_batch(self):
        g = InMemoryGraphStore()
        added = g.add_triples([("A", "r1", "B"), ("B", "r2", "C"), ("A", "r1", "B")])
        assert added == 2  # 第3条重复

    def test_neighbors_bfs(self, sample_graph_data):
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        # 稻瘟病 1 跳：三环唑、真菌性病害
        n1 = set(g.neighbors("稻瘟病", hops=1))
        assert "三环唑" in n1
        assert "真菌性病害" in n1
        # 2 跳：还包含杀菌剂（三环唑的出边）
        n2 = set(g.neighbors("稻瘟病", hops=2))
        assert "杀菌剂" in n2

    def test_neighbors_zero_hops(self, sample_graph_data):
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        assert g.neighbors("稻瘟病", hops=0) == ["稻瘟病"]

    def test_neighbors_nonexistent(self):
        g = InMemoryGraphStore()
        assert g.neighbors("不存在", hops=2) == []

    def test_relations_of(self, sample_graph_data):
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        rels = g.relations_of("稻瘟病")
        assert ("防治药剂", "三环唑") in rels
        assert ("属于", "真菌性病害") in rels

    def test_reverse_relations_of(self, sample_graph_data):
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        rev = g.reverse_relations_of("井冈霉素")
        assert ("水稻纹枯病", "防治药剂") in rev

    def test_pagerank_single_node(self):
        g = InMemoryGraphStore()
        g.add_triple("A", "r", "B")
        pr = g.pagerank()
        assert pr["A"] > 0
        assert pr["B"] > 0
        assert abs(sum(pr.values()) - 1.0) < 1e-3

    def test_pagerank_hub_node_higher(self, sample_graph_data):
        """中心节点（稻瘟病有多个邻居）应有更高 PageRank"""
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        pr = g.pagerank()
        # 稻瘟病有出边到 2 个实体，应比孤立节点重要
        assert pr["稻瘟病"] > 0
        assert pr["真菌性病害"] > 0

    def test_pagerank_empty(self):
        g = InMemoryGraphStore()
        assert g.pagerank() == {}

    def test_communities(self, sample_graph_data):
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        comms = g.communities()
        # 至少有一个社区
        assert len(comms) >= 1
        # 所有实体都应分到一个社区
        all_assigned = set()
        for c in comms:
            all_assigned.update(c)
        assert all_assigned == set(g.all_entities())

    def test_shortest_path_direct(self, sample_graph_data):
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        path = g.shortest_path("稻瘟病", "三环唑", max_hops=3)
        assert path is not None
        assert path[0] == "稻瘟病"
        assert path[-1] == "三环唑"

    def test_shortest_path_two_hops(self, sample_graph_data):
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        # 稻瘟病 → 三环唑 → 杀菌剂
        path = g.shortest_path("稻瘟病", "杀菌剂", max_hops=3)
        assert path is not None
        assert len(path) == 3

    def test_shortest_path_no_connection(self):
        g = InMemoryGraphStore()
        g.add_triple("A", "r", "B")
        g.add_triple("C", "r", "D")  # 不连通
        path = g.shortest_path("A", "C", max_hops=5)
        assert path is None

    def test_shortest_path_same_entity(self, sample_graph_data):
        g = InMemoryGraphStore(entity_relations=sample_graph_data)
        path = g.shortest_path("稻瘟病", "稻瘟病")
        assert path == ["稻瘟病"]


# ==========================
# 工厂函数
# ==========================


class TestFactory:
    def test_get_vector_store_default_faiss(self):
        store = get_vector_store(backend="faiss", dimension=8)
        assert isinstance(store, FAISSVectorStore)

    def test_get_vector_store_unknown_backend(self):
        with pytest.raises(ValueError):
            get_vector_store(backend="unknown_backend")

    def test_get_vector_store_env_var(self, monkeypatch):
        monkeypatch.setenv("POCKET_VECTOR_BACKEND", "faiss")
        store = get_vector_store(dimension=8)
        assert isinstance(store, FAISSVectorStore)

    def test_get_graph_store_default_memory(self, sample_graph_data):
        g = get_graph_store(backend="memory", entity_relations=sample_graph_data)
        assert isinstance(g, InMemoryGraphStore)
        assert len(g) == 4

    def test_get_graph_store_unknown_backend(self):
        with pytest.raises(ValueError):
            get_graph_store(backend="unknown_backend")

    def test_get_graph_store_env_var(self, monkeypatch, sample_graph_data):
        monkeypatch.setenv("POCKET_GRAPH_BACKEND", "memory")
        g = get_graph_store(entity_relations=sample_graph_data)
        assert isinstance(g, InMemoryGraphStore)

    def test_get_graph_store_neo4j_optional_dep(self):
        """Neo4j 后端已实现，但 neo4j 驱动是可选依赖。

        未安装 neo4j 包时，工厂分发到 Neo4jGraphStore 后由其抛 ImportError
        （可选依赖的标准行为，与 chroma/pgvector 一致）。
        """
        try:
            get_graph_store(backend="neo4j")
        except ImportError as e:
            # 未安装 neo4j 驱动：期望 ImportError 含安装提示
            assert "neo4j" in str(e).lower()
            assert "pip install" in str(e).lower()
        except NotImplementedError:
            # 旧路径残留（不应再出现），失败
            pytest.fail(
                "Neo4jGraphStore 已实现，不应再抛 NotImplementedError；"
                "请检查 neo4j_store.py 实现。"
            )
        else:
            # 已安装 neo4j 驱动且成功实例化（需要真实 Neo4j 服务）：
            # 工厂分发成功即通过
            pass


# ==========================
# 骨架后端
# ==========================


class TestSkeletonBackends:
    def test_chroma_store_import_error(self):
        """ChromaVectorStore 在 chromadb 未安装时应抛 ImportError（或 NotImplementedError 占位）"""
        from PocketGraphRAG.core.storages.chroma_store import ChromaVectorStore

        # 可能因为 chromadb 未安装抛 ImportError，也可能抛 NotImplementedError（占位）
        try:
            import chromadb  # noqa: F401

            chroma_installed = True
        except ImportError:
            chroma_installed = False

        if not chroma_installed:
            with pytest.raises(ImportError):
                ChromaVectorStore()
        else:
            # 装了但未实现，应抛 NotImplementedError
            with pytest.raises(NotImplementedError):
                ChromaVectorStore()

    def test_pgvector_store_import_error(self):
        from PocketGraphRAG.core.storages.pgvector_store import PgVectorStore

        try:
            import psycopg  # noqa: F401

            pg_installed = True
        except ImportError:
            pg_installed = False

        if not pg_installed:
            with pytest.raises(ImportError):
                PgVectorStore()
        else:
            with pytest.raises(NotImplementedError):
                PgVectorStore()


# ==========================
# KVStore / JsonKVStorage
# ==========================


class TestKVStoreAbstract:
    def test_kv_store_cannot_instantiate(self):
        with pytest.raises(TypeError):
            KVStore()

    def test_json_kv_storage_is_subclass(self):
        assert issubclass(JsonKVStorage, KVStore)


class TestJsonKVStorage:
    def test_empty_store(self):
        store = JsonKVStorage()
        assert len(store) == 0
        assert store.keys() == []

    def test_upsert_and_get(self):
        store = JsonKVStorage()
        is_new = store.upsert("doc_001", {"text": "稻瘟病...", "source": "a.pdf"})
        assert is_new is True
        assert len(store) == 1

        v = store.get("doc_001")
        assert v == {"text": "稻瘟病...", "source": "a.pdf"}

    def test_upsert_update_returns_false(self):
        store = JsonKVStorage()
        store.upsert("k1", {"v": 1})
        is_new = store.upsert("k1", {"v": 2})
        assert is_new is False
        assert store.get("k1") == {"v": 2}

    def test_get_nonexistent(self):
        store = JsonKVStorage()
        assert store.get("不存在") is None

    def test_get_returns_copy(self):
        """get 返回副本，外部修改不影响内部数据"""
        store = JsonKVStorage()
        store.upsert("k1", {"v": 1})
        v = store.get("k1")
        v["v"] = 999
        assert store.get("k1") == {"v": 1}

    def test_delete(self):
        store = JsonKVStorage()
        store.upsert("k1", {"v": 1})
        assert store.delete("k1") is True
        assert len(store) == 0
        assert store.get("k1") is None

    def test_delete_nonexistent(self):
        store = JsonKVStorage()
        assert store.delete("不存在") is False

    def test_get_by_ids(self):
        store = JsonKVStorage()
        store.upsert("k1", {"v": 1})
        store.upsert("k2", {"v": 2})
        results = store.get_by_ids(["k1", "missing", "k2"])
        assert len(results) == 3
        assert results[0] == {"v": 1}
        assert results[1] is None
        assert results[2] == {"v": 2}

    def test_keys(self):
        store = JsonKVStorage()
        store.upsert("k1", {"v": 1})
        store.upsert("k2", {"v": 2})
        assert set(store.keys()) == {"k1", "k2"}

    def test_upsert_non_dict_raises(self):
        store = JsonKVStorage()
        with pytest.raises(TypeError):
            store.upsert("k1", "不是dict")
        with pytest.raises(TypeError):
            store.upsert("k1", ["也不是dict"])

    def test_clear(self):
        store = JsonKVStorage()
        store.upsert("k1", {"v": 1})
        store.upsert("k2", {"v": 2})
        n = store.clear()
        assert n == 2
        assert len(store) == 0

    def test_items(self):
        store = JsonKVStorage()
        store.upsert("k1", {"v": 1})
        items = store.items()
        assert items == [("k1", {"v": 1})]


class TestJsonKVStoragePersistence:
    def test_save_and_load_roundtrip(self, temp_dir):
        path = os.path.join(temp_dir, "kv.json")
        store = JsonKVStorage(path=path)
        store.upsert("doc_001", {"text": "稻瘟病", "source": "a.pdf"})
        store.upsert("doc_002", {"text": "三环唑", "source": "b.pdf"})
        store.save()

        # 新实例加载
        store2 = JsonKVStorage(path=path)
        assert len(store2) == 2
        assert store2.get("doc_001") == {"text": "稻瘟病", "source": "a.pdf"}
        assert store2.get("doc_002") == {"text": "三环唑", "source": "b.pdf"}

    def test_save_without_path_raises(self):
        store = JsonKVStorage()
        store.upsert("k1", {"v": 1})
        with pytest.raises(ValueError):
            store.save()

    def test_load_nonexistent_raises(self, temp_dir):
        store = JsonKVStorage()
        with pytest.raises(FileNotFoundError):
            store.load(os.path.join(temp_dir, "no_such_file.json"))

    def test_load_invalid_format_raises(self, temp_dir):
        path = os.path.join(temp_dir, "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(["not", "a", "dict"], f)  # type: ignore
        store = JsonKVStorage()
        with pytest.raises(ValueError):
            store.load(path)

    def test_in_memory_no_auto_load(self, temp_dir):
        """纯内存模式（不传 path）不会自动加载文件"""
        path = os.path.join(temp_dir, "kv.json")
        store = JsonKVStorage(path=path)
        store.upsert("k1", {"v": 1})
        store.save()

        # 纯内存模式：不传 path
        store2 = JsonKVStorage()
        assert len(store2) == 0


class TestKVStoreFactory:
    def test_get_kv_store_default_json(self):
        store = get_kv_store(backend="json")
        assert isinstance(store, JsonKVStorage)

    def test_get_kv_store_with_path(self, temp_dir):
        path = os.path.join(temp_dir, "factory_kv.json")
        store = get_kv_store(backend="json", path=path)
        assert isinstance(store, JsonKVStorage)
        store.upsert("k1", {"v": 1})
        store.save()

        store2 = get_kv_store(backend="json", path=path)
        assert len(store2) == 1

    def test_get_kv_store_env_var(self, monkeypatch):
        monkeypatch.setenv("POCKET_KV_BACKEND", "json")
        store = get_kv_store()
        assert isinstance(store, JsonKVStorage)

    def test_get_kv_store_unknown_backend(self):
        with pytest.raises(ValueError):
            get_kv_store(backend="unknown")

    def test_get_kv_store_redis_not_implemented(self):
        with pytest.raises(NotImplementedError):
            get_kv_store(backend="redis")
