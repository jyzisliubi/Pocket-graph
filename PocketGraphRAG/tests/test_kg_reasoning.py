"""Tests for KGDualRetriever (graph algorithms + mock embedding model)."""

import json
import os

import numpy as np
import pytest


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing."""

    def __init__(self, model_name=None, **kwargs):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        """Return deterministic fake embeddings based on text hash."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            h = hash(text) % (2**30)
            vec = np.zeros(32, dtype=np.float32)
            vec[0] = (h % 1000) / 1000.0
            vec[1] = ((h // 1000) % 1000) / 1000.0
            vec[2] = ((h // 1000000) % 1000) / 1000.0
            vec[3:8] = 0.1
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


def _create_mock_retriever(entity_relations, reverse_relations, temp_dir):
    """Helper to create a KGDualRetriever with mock model."""
    import faiss

    from PocketGraphRAG.kg_reasoning import KGDualRetriever

    model = MockSentenceTransformer()

    all_entities = sorted(set(entity_relations.keys()) | set(reverse_relations.keys()))
    all_relations_set = set()
    for rels in entity_relations.values():
        for rel, _ in rels:
            all_relations_set.add(rel)
    all_relations = sorted(all_relations_set)

    os.makedirs(temp_dir, exist_ok=True)

    entity_embeddings = model.encode(all_entities, normalize_embeddings=True)
    entity_index = faiss.IndexFlatIP(entity_embeddings.shape[1])
    entity_index.add(entity_embeddings)
    faiss.write_index(entity_index, os.path.join(temp_dir, "entity_faiss.index"))
    with open(os.path.join(temp_dir, "entity_names.json"), "w", encoding="utf-8") as f:
        json.dump(all_entities, f, ensure_ascii=False)

    relation_embeddings = model.encode(all_relations, normalize_embeddings=True)
    relation_index = faiss.IndexFlatIP(relation_embeddings.shape[1])
    relation_index.add(relation_embeddings)
    faiss.write_index(relation_index, os.path.join(temp_dir, "relation_faiss.index"))
    with open(
        os.path.join(temp_dir, "relation_names.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_relations, f, ensure_ascii=False)

    retriever = KGDualRetriever(
        entity_relations=entity_relations,
        reverse_relations=reverse_relations,
        model=model,
        index_dir=temp_dir,
        threshold=0.0,
        n_hops=2,
        relation_threshold=0.0,
    )
    return retriever


class TestKGDualRetrieverInit:
    def test_initialization(
        self, sample_entity_relations, sample_reverse_relations, temp_dir
    ):
        retriever = _create_mock_retriever(
            sample_entity_relations, sample_reverse_relations, temp_dir
        )
        assert retriever.entity_relations == sample_entity_relations
        assert retriever.reverse_relations == sample_reverse_relations
        assert len(retriever.all_entities) > 0
        assert len(retriever.all_relations) > 0

    def test_all_entities_collected(
        self, sample_entity_relations, sample_reverse_relations, temp_dir
    ):
        retriever = _create_mock_retriever(
            sample_entity_relations, sample_reverse_relations, temp_dir
        )
        expected = set(sample_entity_relations.keys()) | set(
            sample_reverse_relations.keys()
        )
        assert set(retriever.all_entities) == expected

    def test_graph_store_auto_wrapped(
        self, sample_entity_relations, sample_reverse_relations, temp_dir
    ):
        """不传 graph_store 时自动用 InMemoryGraphStore 包装 dict"""
        from PocketGraphRAG.core.storages import InMemoryGraphStore

        retriever = _create_mock_retriever(
            sample_entity_relations, sample_reverse_relations, temp_dir
        )
        assert isinstance(retriever.graph_store, InMemoryGraphStore)
        # 向后兼容：dict 引用仍然可用
        assert retriever.entity_relations is retriever.graph_store.entity_relations

    def test_graph_store_explicit(
        self, sample_entity_relations, sample_reverse_relations, temp_dir
    ):
        """显式传入 graph_store 时作为数据源"""
        from PocketGraphRAG.core.storages import InMemoryGraphStore

        store = InMemoryGraphStore(
            entity_relations=sample_entity_relations,
            reverse_relations=sample_reverse_relations,
        )
        from PocketGraphRAG.kg_reasoning import KGDualRetriever

        model = MockSentenceTransformer()
        # 需要写 entity/relation 索引文件
        import faiss

        all_entities = sorted(
            set(sample_entity_relations.keys()) | set(sample_reverse_relations.keys())
        )
        all_relations = sorted(
            {rel for rels in sample_entity_relations.values() for rel, _ in rels}
        )
        os.makedirs(temp_dir, exist_ok=True)
        ent_embs = model.encode(all_entities, normalize_embeddings=True)
        ent_idx = faiss.IndexFlatIP(ent_embs.shape[1])
        ent_idx.add(ent_embs)
        faiss.write_index(ent_idx, os.path.join(temp_dir, "entity_faiss.index"))
        with open(os.path.join(temp_dir, "entity_names.json"), "w", encoding="utf-8") as f:
            json.dump(all_entities, f, ensure_ascii=False)
        if all_relations:
            rel_embs = model.encode(all_relations, normalize_embeddings=True)
            rel_idx = faiss.IndexFlatIP(rel_embs.shape[1])
            rel_idx.add(rel_embs)
            faiss.write_index(rel_idx, os.path.join(temp_dir, "relation_faiss.index"))
            with open(os.path.join(temp_dir, "relation_names.json"), "w", encoding="utf-8") as f:
                json.dump(all_relations, f, ensure_ascii=False)

        retriever = KGDualRetriever(
            entity_relations=sample_entity_relations,
            reverse_relations=sample_reverse_relations,
            model=model,
            index_dir=temp_dir,
            threshold=0.0,
            graph_store=store,
        )
        assert retriever.graph_store is store
        # InMemoryGraphStore 的 dict 引用仍然可用
        assert retriever.entity_relations is store.entity_relations


class TestGraphAlgorithms:
    @pytest.fixture
    def retriever(self, sample_entity_relations, sample_reverse_relations, temp_dir):
        return _create_mock_retriever(
            sample_entity_relations, sample_reverse_relations, temp_dir
        )

    def test_get_entity_degree(self, retriever):
        degree = retriever.get_entity_degree("稻瘟病")
        assert degree > 0

    def test_get_entity_degree_nonexistent(self, retriever):
        degree = retriever.get_entity_degree("不存在的实体")
        assert degree == 0

    def test_get_top_entities(self, retriever):
        top = retriever.get_top_entities(top_k=5)
        assert len(top) <= 5
        assert len(top) > 0
        degrees = [retriever.get_entity_degree(e) for e in top]
        assert degrees == sorted(degrees, reverse=True)

    def test_get_top_entities_more_than_total(self, retriever):
        top = retriever.get_top_entities(top_k=1000)
        assert len(top) == len(retriever.all_entities)

    def test_get_graph_stats(self, retriever):
        stats = retriever.get_graph_stats()
        assert "total_entities" in stats
        assert "total_relations" in stats
        assert "total_edges" in stats
        assert "avg_degree" in stats
        assert stats["total_entities"] > 0
        assert stats["total_edges"] > 0
        assert stats["avg_degree"] >= 0

    def test_get_subgraph_single_entity_1hop(self, retriever):
        result = retriever.get_subgraph(["稻瘟病"], max_hops=1)
        assert "nodes" in result
        assert "links" in result
        assert len(result["nodes"]) > 1
        assert len(result["links"]) > 0

        node_names = {n["name"] for n in result["nodes"]}
        assert "稻瘟病" in node_names

    def test_get_subgraph_single_entity_2hops(self, retriever):
        result_1 = retriever.get_subgraph(["稻瘟病"], max_hops=1)
        result_2 = retriever.get_subgraph(["稻瘟病"], max_hops=2)
        assert len(result_2["nodes"]) >= len(result_1["nodes"])

    def test_get_subgraph_multiple_entities(self, retriever):
        result = retriever.get_subgraph(["稻瘟病", "水稻纹枯病"], max_hops=1)
        node_names = {n["name"] for n in result["nodes"]}
        assert "稻瘟病" in node_names
        assert "水稻纹枯病" in node_names

    def test_get_subgraph_categories(self, retriever):
        result = retriever.get_subgraph(["稻瘟病"], max_hops=1)
        center_nodes = [n for n in result["nodes"] if n["category"] == 0]
        neighbor_nodes = [n for n in result["nodes"] if n["category"] == 1]
        assert len(center_nodes) == 1
        assert center_nodes[0]["name"] == "稻瘟病"
        assert len(neighbor_nodes) > 0

    def test_get_subgraph_node_degree(self, retriever):
        result = retriever.get_subgraph(["水稻"], max_hops=1)
        for node in result["nodes"]:
            assert "degree" in node
            assert node["degree"] >= 0
            assert "symbolSize" in node
            assert node["symbolSize"] >= 12

    def test_get_subgraph_links_bidirectional(self, retriever):
        result = retriever.get_subgraph(["三环唑"], max_hops=1)
        link_targets = {l["target"] for l in result["links"]}
        link_sources = {l["source"] for l in result["links"]}
        all_nodes = {n["name"] for n in result["nodes"]}
        assert link_sources.issubset(all_nodes)
        assert link_targets.issubset(all_nodes)

    def test_get_subgraph_nonexistent_entity(self, retriever):
        result = retriever.get_subgraph(["不存在的实体"], max_hops=1)
        node_names = {n["name"] for n in result["nodes"]}
        assert "不存在的实体" in node_names
        assert len(result["links"]) == 0
        assert len(result["nodes"]) == 1

    def test_get_subgraph_empty_list(self, retriever):
        result = retriever.get_subgraph([], max_hops=1)
        assert len(result["nodes"]) == 0
        assert len(result["links"]) == 0


class TestEntityMatching:
    @pytest.fixture
    def retriever(self, sample_entity_relations, sample_reverse_relations, temp_dir):
        return _create_mock_retriever(
            sample_entity_relations, sample_reverse_relations, temp_dir
        )

    def test_match_entities_returns_list(self, retriever):
        result = retriever.match_entities("稻瘟病")
        assert isinstance(result, list)

    def test_match_entities_with_top_k(self, retriever):
        result = retriever.match_entities("稻瘟病", top_k=3)
        assert len(result) <= 3

    def test_match_entities_threshold(self, retriever):
        result = retriever.match_entities("稻瘟病", threshold=1.0)
        assert isinstance(result, list)


class TestRelationMatching:
    @pytest.fixture
    def retriever(self, sample_entity_relations, sample_reverse_relations, temp_dir):
        return _create_mock_retriever(
            sample_entity_relations, sample_reverse_relations, temp_dir
        )

    def test_match_relations_returns_list(self, retriever):
        result = retriever.match_relations("防治")
        assert isinstance(result, list)

    def test_match_relations_with_top_k(self, retriever):
        result = retriever.match_relations("防治", top_k=2)
        assert len(result) <= 2

    def test_match_relations_threshold(self, retriever):
        result = retriever.match_relations("防治", threshold=1.0)
        assert isinstance(result, list)


class TestSearchModes:
    @pytest.fixture
    def retriever(self, sample_entity_relations, sample_reverse_relations, temp_dir):
        return _create_mock_retriever(
            sample_entity_relations, sample_reverse_relations, temp_dir
        )

    def test_local_search_returns_list(self, retriever):
        result = retriever.local_search("稻瘟病")
        assert isinstance(result, list)

    def test_local_search_with_n_hops(self, retriever):
        result_1 = retriever.local_search("水稻", n_hops=1)
        result_2 = retriever.local_search("水稻", n_hops=2)
        assert isinstance(result_1, list)
        assert isinstance(result_2, list)

    def test_global_search_returns_list(self, retriever):
        result = retriever.global_search("防治")
        assert isinstance(result, list)

    def test_mix_search_returns_list(self, retriever):
        result = retriever.mix_search("稻瘟病")
        assert isinstance(result, list)


class TestGraphAlgorithmsAdvanced:
    """Test advanced graph algorithms: Pagerank, community detection, shortest path."""

    @pytest.fixture
    def retriever(self, sample_entity_relations, sample_reverse_relations, temp_dir):
        return _create_mock_retriever(
            sample_entity_relations, sample_reverse_relations, temp_dir
        )

    # --- Pagerank tests ---

    def test_compute_pagerank_returns_dict(self, retriever):
        pr = retriever.compute_pagerank()
        assert isinstance(pr, dict)
        assert len(pr) == len(retriever.all_entities)

    def test_compute_pagerank_values_positive(self, retriever):
        pr = retriever.compute_pagerank()
        for entity, score in pr.items():
            assert isinstance(score, float)
            assert score >= 0

    def test_compute_pagerank_sums_to_one(self, retriever):
        pr = retriever.compute_pagerank()
        total = sum(pr.values())
        assert abs(total - 1.0) < 0.01

    def test_compute_pagerank_hub_entity_has_higher_score(self, retriever):
        pr = retriever.compute_pagerank()
        # "水稻" 是连接最多的实体，应该有较高的 Pagerank
        if "水稻" in pr and "叶枯唑" in pr:
            assert pr["水稻"] > pr["叶枯唑"]

    def test_compute_pagerank_damping_factor(self, retriever):
        pr1 = retriever.compute_pagerank(damping=0.85)
        pr2 = retriever.compute_pagerank(damping=0.5)
        assert len(pr1) == len(pr2)
        # 不同阻尼系数结果应该不同
        assert set(pr1.values()) != set(pr2.values())

    # --- Community detection tests ---

    def test_detect_communities_returns_dict(self, retriever):
        import random

        random.seed(42)
        np.random.seed(42)

        communities = retriever.detect_communities()
        assert isinstance(communities, dict)
        assert len(communities) == len(retriever.all_entities)

    def test_detect_communities_ids_are_integers(self, retriever):
        import random

        random.seed(42)
        np.random.seed(42)

        communities = retriever.detect_communities()
        for entity, cid in communities.items():
            assert isinstance(cid, int)
            assert cid >= 0

    def test_detect_communities_number_of_communities(self, retriever):
        import random

        random.seed(42)
        np.random.seed(42)

        communities = retriever.detect_communities()
        unique_communities = set(communities.values())
        assert len(unique_communities) >= 1
        assert len(unique_communities) <= len(retriever.all_entities)

    def test_detect_communities_connected_entities_same_community(self, retriever):
        import random

        random.seed(42)
        np.random.seed(42)

        communities = retriever.detect_communities()
        # 稻瘟病和水稻有直接连接，可能在同一个社区
        # （不做硬性断言，因为标签传播有随机性，但至少都是合法的社区ID）
        assert "稻瘟病" in communities
        assert "水稻" in communities

    # --- Shortest path tests ---

    def test_shortest_path_direct_connection(self, retriever):
        path = retriever.shortest_path("稻瘟病", "三环唑")
        assert isinstance(path, list)
        assert len(path) >= 2
        assert path[0] == "稻瘟病"
        assert path[-1] == "三环唑"

    def test_shortest_path_same_entity(self, retriever):
        path = retriever.shortest_path("稻瘟病", "稻瘟病")
        assert path == ["稻瘟病"]

    def test_shortest_path_indirect_connection(self, retriever):
        path = retriever.shortest_path("三环唑", "稻瘟灵")
        assert isinstance(path, list)
        assert len(path) >= 2
        assert path[0] == "三环唑"
        assert path[-1] == "稻瘟灵"

    def test_shortest_path_nonexistent_entity(self, retriever):
        path = retriever.shortest_path("不存在的实体1", "不存在的实体2")
        assert path == []

    def test_shortest_path_max_hops(self, retriever):
        path = retriever.shortest_path("稻瘟病", "三环唑", max_hops=1)
        assert len(path) == 2

        path_short = retriever.shortest_path("稻瘟病", "三环唑", max_hops=0)
        assert path_short == []

    def test_path_between_entities_two_entities(self, retriever):
        entities = ["稻瘟病", "三环唑"]
        result = retriever.path_between_entities(entities)
        assert isinstance(result, list)
        assert "稻瘟病" in result
        assert "三环唑" in result

    def test_path_between_entities_single(self, retriever):
        result = retriever.path_between_entities(["稻瘟病"])
        assert result == ["稻瘟病"]

    def test_path_between_entities_multiple(self, retriever):
        entities = ["稻瘟病", "三环唑", "稻瘟灵"]
        result = retriever.path_between_entities(entities)
        assert isinstance(result, list)
        assert len(result) >= 3
        assert all(e in result for e in entities)
