"""RRF 融合算法单元测试"""


class MockRetriever:
    """模拟 KG 检索器，用于测试 RAG 系统"""

    def __init__(self, entities=None):
        self.all_entities = entities or []
        self.entity_relations = {}
        self.reverse_relations = {}

    def match_entities(self, query):
        return []

    def match_relations(self, query):
        return []

    def local_search(self, query):
        return []

    def global_search(self, query):
        return []

    def mix_search(self, query):
        return []


class TestRRFFusion:
    def test_rrf_basic_fusion(self):
        """测试 RRF 基本融合逻辑：两个列表有重叠元素"""
        from PocketGraphRAG.rag_system import PocketGraphRAG

        rag = PocketGraphRAG.__new__(PocketGraphRAG)
        rag.rrf_k = 60

        list_a = [
            ("doc1", 0.9, {"entity": "A"}),
            ("doc2", 0.8, {"entity": "B"}),
            ("doc3", 0.7, {"entity": "C"}),
        ]
        list_b = [
            ("doc2", 0.95, {"entity": "B"}),
            ("doc4", 0.85, {"entity": "D"}),
            ("doc1", 0.75, {"entity": "A"}),
        ]

        result = rag._rrf_fusion([list_a, list_b], top_k=10)

        # doc1 和 doc2 在两个列表中都出现，分数应该更高
        result_texts = [r[0] for r in result]
        assert "doc1" in result_texts
        assert "doc2" in result_texts
        assert "doc3" in result_texts
        assert "doc4" in result_texts

        # doc2 在两个列表中排名都靠前，应该排第一或第二
        doc2_idx = result_texts.index("doc2")
        doc3_idx = result_texts.index("doc3")
        # doc2 出现在两个列表中（rank 1 和 rank 1），doc3 只在一个列表（rank 3）
        assert doc2_idx < doc3_idx

    def test_rrf_top_k(self):
        """测试 RRF 返回 top_k 个结果"""
        from PocketGraphRAG.rag_system import PocketGraphRAG

        rag = PocketGraphRAG.__new__(PocketGraphRAG)
        rag.rrf_k = 60

        list_a = [(f"doc{i}", 1.0 - i * 0.1, {"entity": f"E{i}"}) for i in range(10)]
        list_b = [
            (f"doc{i + 5}", 1.0 - i * 0.1, {"entity": f"E{i + 5}"}) for i in range(10)
        ]

        result = rag._rrf_fusion([list_a, list_b], top_k=5)
        assert len(result) == 5

    def test_rrf_empty_lists(self):
        """测试 RRF 空列表"""
        from PocketGraphRAG.rag_system import PocketGraphRAG

        rag = PocketGraphRAG.__new__(PocketGraphRAG)
        rag.rrf_k = 60

        result = rag._rrf_fusion([[], []], top_k=10)
        assert result == []

    def test_rrf_single_list(self):
        """测试 RRF 单个列表"""
        from PocketGraphRAG.rag_system import PocketGraphRAG

        rag = PocketGraphRAG.__new__(PocketGraphRAG)
        rag.rrf_k = 60

        lst = [
            ("doc1", 0.9, {"entity": "A"}),
            ("doc2", 0.8, {"entity": "B"}),
        ]

        result = rag._rrf_fusion([lst], top_k=10)
        assert len(result) == 2
        # 单个列表时，顺序应与原列表一致
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"

    def test_rrf_no_duplicates(self):
        """测试 RRF 完全不重叠的两个列表"""
        from PocketGraphRAG.rag_system import PocketGraphRAG

        rag = PocketGraphRAG.__new__(PocketGraphRAG)
        rag.rrf_k = 60

        list_a = [("doc1", 0.9, {"entity": "A"}), ("doc2", 0.8, {"entity": "B"})]
        list_b = [("doc3", 0.9, {"entity": "C"}), ("doc4", 0.8, {"entity": "D"})]

        result = rag._rrf_fusion([list_a, list_b], top_k=10)
        assert len(result) == 4

    def test_rrf_different_k_values(self):
        """测试不同的 RRF k 值"""
        from PocketGraphRAG.rag_system import PocketGraphRAG

        # k 小的时候，高排名的权重更大
        rag_small_k = PocketGraphRAG.__new__(PocketGraphRAG)
        rag_small_k.rrf_k = 10

        rag_big_k = PocketGraphRAG.__new__(PocketGraphRAG)
        rag_big_k.rrf_k = 100

        list_a = [("doc1", 0.9, {"entity": "A"}), ("doc2", 0.8, {"entity": "B"})]
        list_b = [("doc2", 0.9, {"entity": "B"}), ("doc3", 0.8, {"entity": "C"})]

        result_small = rag_small_k._rrf_fusion([list_a, list_b], top_k=3)
        result_big = rag_big_k._rrf_fusion([list_a, list_b], top_k=3)

        # doc2 在两个列表里都排第2，应该比只在一个列表排第1的分数高
        # k 越小，排名差带来的分数差越大
        assert result_small[0][0] == "doc1" or result_small[0][0] == "doc2"
        # 两种 k 值下 doc2 都应该排名靠前（因为出现在两个列表）
        small_texts = [r[0] for r in result_small]
        big_texts = [r[0] for r in result_big]
        assert "doc2" in small_texts[:2]
        assert "doc2" in big_texts[:2]

    def test_weighted_fusion(self):
        """测试简单加权融合"""
        from PocketGraphRAG.rag_system import PocketGraphRAG

        rag = PocketGraphRAG.__new__(PocketGraphRAG)

        list_a = [
            ("doc1", 0.9, {"entity": "A"}),
            ("doc2", 0.5, {"entity": "B"}),
        ]
        list_b = [
            ("doc2", 0.8, {"entity": "B"}),
            ("doc3", 0.7, {"entity": "C"}),
        ]

        result = rag._weighted_fusion([list_a, list_b], top_k=10)

        # _weighted_fusion 对重复 text 取较高分：doc2 = max(0.5, 0.8) = 0.8
        assert len(result) == 3
        # doc1 分数 0.9 最高
        assert result[0][0] == "doc1"
        # doc2 取较高分 0.8，应排在 doc3 (0.7) 之前
        assert result[1][0] == "doc2"
        assert result[2][0] == "doc3"
