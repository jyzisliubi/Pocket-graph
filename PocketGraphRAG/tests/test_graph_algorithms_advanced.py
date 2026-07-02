"""高级图算法单元测试 - 个性化 Pagerank、中心性、连接组件、聚类系数"""

import pytest


def _make_retriever(entity_relations):
    """辅助函数：创建测试用的 KGDualRetriever 实例"""
    from PocketGraphRAG.kg_reasoning import KGDualRetriever

    reverse_relations = {}
    ret = KGDualRetriever.__new__(KGDualRetriever)
    ret.entity_relations = entity_relations
    ret.reverse_relations = reverse_relations
    ret.all_entities = sorted(
        set(entity_relations.keys()) | set(reverse_relations.keys())
    )
    ret.entity_idx = {e: i for i, e in enumerate(ret.all_entities)}
    ret.idx_entity = {i: e for i, e in enumerate(ret.all_entities)}
    ret._adj_cache = None
    ret.threshold = 0.5
    ret.n_hops = 2
    ret.relation_threshold = 0.3
    return ret


@pytest.fixture
def retriever_advanced():
    """创建一个有更复杂结构的 KGDualRetriever 测试实例"""
    entity_relations = {
        "A": [("关系1", "B"), ("关系2", "C")],
        "B": [("关系1", "A"), ("关系3", "D")],
        "C": [("关系2", "A"), ("关系4", "D"), ("关系5", "E")],
        "D": [("关系3", "B"), ("关系4", "C"), ("关系6", "E")],
        "E": [("关系5", "C"), ("关系6", "D"), ("关系7", "F")],
        "F": [("关系7", "E")],
        "G": [],
    }
    return _make_retriever(entity_relations)


class TestPersonalizedPageRank:
    def test_ppr_single_seed(self, retriever_advanced):
        """测试单个种子实体的个性化 Pagerank"""
        ppr = retriever_advanced.personalized_pagerank(["A"])

        # 种子实体 A 分数应该最高
        assert ppr["A"] > ppr["B"]
        assert ppr["A"] > ppr["C"]
        # 距离种子越近的节点分数越高（整体趋势）
        assert ppr["B"] > ppr["E"]  # B 是 1 跳，E 是 2 跳
        assert ppr["C"] > ppr["F"]  # C 是 1 跳，F 是 3 跳
        # 孤立节点 G 分数为 0
        assert ppr["G"] == pytest.approx(0.0)
        # 所有分数加起来约等于 1（归一化）
        assert abs(sum(ppr.values()) - 1.0) < 0.01

    def test_ppr_multiple_seeds(self, retriever_advanced):
        """测试多个种子实体"""
        ppr = retriever_advanced.personalized_pagerank(["A", "F"])

        # A 和 F 都是种子，分数应该都比较高
        assert ppr["A"] > 0
        assert ppr["F"] > 0
        # E 是 F 的邻居也是 C/D 的邻居，应该也有较高分数
        assert ppr["E"] > ppr["B"]

    def test_ppr_invalid_seed(self, retriever_advanced):
        """测试无效种子实体"""
        ppr = retriever_advanced.personalized_pagerank(["不存在的实体"])
        assert all(v == 0.0 for v in ppr.values())

    def test_ppr_empty_seed(self, retriever_advanced):
        """测试空种子列表"""
        ppr = retriever_advanced.personalized_pagerank([])
        assert all(v == 0.0 for v in ppr.values())


class TestCentrality:
    def test_degree_centrality(self, retriever_advanced):
        """测试度中心性"""
        centrality = retriever_advanced.degree_centrality()

        # C 连接了 A, D, E → 度数最高
        assert centrality["C"] > centrality["A"]
        assert centrality["C"] > centrality["B"]
        # F 和 G 度数最低
        assert centrality["G"] == pytest.approx(0.0)
        assert centrality["F"] < centrality["E"]

    def test_degree_centrality_range(self, retriever_advanced):
        """测试度中心性范围在 0-1 之间"""
        centrality = retriever_advanced.degree_centrality()
        for v in centrality.values():
            assert 0.0 <= v <= 1.0

    def test_closeness_centrality(self, retriever_advanced):
        """测试接近中心性"""
        centrality = retriever_advanced.closeness_centrality(max_hops=5)

        # C 和 D 位于图中心，接近中心性应该更高
        assert centrality["C"] > centrality["A"]
        assert centrality["D"] > centrality["F"]
        # 孤立节点 G 中心性为 0
        assert centrality["G"] == pytest.approx(0.0)

    def test_betweenness_centrality_approx(self, retriever_advanced):
        """测试介数中心性（近似）"""
        centrality = retriever_advanced.betweenness_centrality_approx(k=None)

        # E 是连接 F 到其他节点的必经之路，介数应该较高
        assert centrality["E"] > centrality["F"]
        # C 和 D 在图中间，介数也应该比较高
        assert centrality["C"] >= 0
        assert centrality["D"] >= 0
        # 孤立节点和叶子节点介数为 0
        assert centrality["G"] == pytest.approx(0.0)

    def test_betweenness_centrality_sampled(self, retriever_advanced):
        """测试采样版本的介数中心性"""
        centrality = retriever_advanced.betweenness_centrality_approx(k=3)
        # 采样版本也应该返回合理结果
        assert len(centrality) == 7
        for v in centrality.values():
            assert v >= 0.0


class TestConnectedComponents:
    def test_connected_components(self, retriever_advanced):
        """测试连接组件检测"""
        components = retriever_advanced.connected_components()

        # 应该有 2 个连通分量：A-F 是一个大的，G 是孤立的
        assert len(components) == 2
        # 按大小排序
        assert len(components[0]) == 6  # A, B, C, D, E, F
        assert len(components[1]) == 1  # G

        # 第一个分量包含 A-F
        main_component = set(components[0])
        assert "A" in main_component
        assert "B" in main_component
        assert "F" in main_component
        assert "G" not in main_component

        # 第二个分量只有 G
        assert components[1] == ["G"]

    def test_fully_connected(self):
        """测试完全连通图"""

        entity_relations = {
            "A": [("r", "B"), ("r", "C")],
            "B": [("r", "A"), ("r", "C")],
            "C": [("r", "A"), ("r", "B")],
        }
        ret = _make_retriever(entity_relations)
        components = ret.connected_components()
        assert len(components) == 1
        assert len(components[0]) == 3

    def test_all_isolated(self):
        """测试所有节点都是孤立的"""

        entity_relations = {
            "A": [],
            "B": [],
            "C": [],
        }
        ret = _make_retriever(entity_relations)
        components = ret.connected_components()
        assert len(components) == 3
        assert all(len(c) == 1 for c in components)


class TestClusteringCoefficient:
    def test_local_clustering_triangle(self):
        """测试三角形图的局部聚类系数"""

        entity_relations = {
            "A": [("r", "B"), ("r", "C")],
            "B": [("r", "A"), ("r", "C")],
            "C": [("r", "A"), ("r", "B")],
        }
        ret = _make_retriever(entity_relations)
        clustering = ret.clustering_coefficient()

        # 三角形中每个节点的聚类系数都是 1.0
        # 每个节点有 2 个邻居，邻居之间有 1 条边，最大可能 1 条边
        assert clustering["A"] == pytest.approx(1.0)
        assert clustering["B"] == pytest.approx(1.0)
        assert clustering["C"] == pytest.approx(1.0)

    def test_local_clustering_line(self):
        """测试直线图的局部聚类系数"""

        entity_relations = {
            "A": [("r", "B")],
            "B": [("r", "A"), ("r", "C")],
            "C": [("r", "B"), ("r", "D")],
            "D": [("r", "C")],
        }
        ret = _make_retriever(entity_relations)
        clustering = ret.clustering_coefficient()

        # 端点只有 1 个邻居，聚类系数为 0
        assert clustering["A"] == pytest.approx(0.0)
        assert clustering["D"] == pytest.approx(0.0)
        # 中间节点 B 和 C 各有 2 个邻居，但邻居之间没有边
        assert clustering["B"] == pytest.approx(0.0)
        assert clustering["C"] == pytest.approx(0.0)

    def test_global_clustering_triangle(self):
        """测试三角形图的全局聚类系数"""

        entity_relations = {
            "A": [("r", "B"), ("r", "C")],
            "B": [("r", "A"), ("r", "C")],
            "C": [("r", "A"), ("r", "B")],
        }
        ret = _make_retriever(entity_relations)
        gcc = ret.global_clustering_coefficient()
        # 完全三角形图的全局聚类系数为 1.0
        assert gcc == pytest.approx(1.0)

    def test_global_clustering_line(self):
        """测试直线图的全局聚类系数"""

        entity_relations = {
            "A": [("r", "B")],
            "B": [("r", "A"), ("r", "C")],
            "C": [("r", "B")],
        }
        ret = _make_retriever(entity_relations)
        gcc = ret.global_clustering_coefficient()
        # 直线图没有三角形，全局聚类系数为 0
        assert gcc == pytest.approx(0.0)

    def test_global_clustering_range(self, retriever_advanced):
        """测试全局聚类系数在 0-1 范围内"""
        gcc = retriever_advanced.global_clustering_coefficient()
        assert 0.0 <= gcc <= 1.0


class TestAdjacencyList:
    def test_build_adjacency_list(self, retriever_advanced):
        """测试邻接表构建"""
        adj = retriever_advanced._build_adjacency_list()

        # A 的邻居：B, C
        a_idx = retriever_advanced.entity_idx["A"]
        b_idx = retriever_advanced.entity_idx["B"]
        c_idx = retriever_advanced.entity_idx["C"]
        g_idx = retriever_advanced.entity_idx["G"]

        assert b_idx in adj[a_idx]
        assert c_idx in adj[a_idx]
        # 对称：A 也是 B 的邻居
        assert a_idx in adj[b_idx]
        # 孤立节点 G 没有邻居
        assert len(adj[g_idx]) == 0
