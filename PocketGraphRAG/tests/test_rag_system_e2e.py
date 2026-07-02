"""PocketGraphRAG 主类端到端测试 + Bug 回归测试

覆盖范围：
1. Bug 回归：vector_weight 并发污染（H4）、incremental_index NameError
2. PocketGraphRAG 主类：构造、检索、问答、流式问答
3. _basic_retrieve 6 种 search_mode 分支
4. _merge_results / _retrieve_by_entities / _retrieve_pure_kg 融合逻辑
5. 并发安全：多线程下 vector_weight 不互相污染
6. 异常路径：无 LLM、空索引、无 KG、参数边界
7. 增量索引：append 持久化路径（Bug 2 回归）
"""

from __future__ import annotations

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

import numpy as np

# ==========================
# Mock 模型（避免下载 BGE）
# ==========================


class MockSentenceTransformer:
    """模拟 SentenceTransformer，返回固定维度随机向量。

    同名文本编码一致（用 hash 做种子），保证 match_entities 的余弦相似度逻辑可测。
    """

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(
        self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
    ):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            # 同名文本稳定编码
            seed = abs(hash(t)) % (2**32)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.dim).astype("float32")
            if normalize_embeddings:
                norm = np.linalg.norm(v) + 1e-8
                v = v / norm
            vecs.append(v)
        return np.array(vecs, dtype="float32")


# ==========================
# 测试夹具：构建一个完整的 PocketGraphRAG 实例（绕过 _load_index）
# ==========================


def _build_test_chunks():
    """构造测试用 chunks（entity-level chunking 风格）"""
    return [
        {
            "text": "稻瘟病是由稻瘟病菌引起的真菌性病害",
            "metadata": {"entity": "稻瘟病"},
        },
        {"text": "三环唑是防治稻瘟病的常用杀菌剂", "metadata": {"entity": "三环唑"}},
        {"text": "稻瘟灵也可用于防治稻瘟病", "metadata": {"entity": "稻瘟灵"}},
        {"text": "水稻纹枯病由立枯丝核菌引起", "metadata": {"entity": "水稻纹枯病"}},
        {"text": "井冈霉素防治水稻纹枯病效果良好", "metadata": {"entity": "井冈霉素"}},
        {"text": "水稻是我国主要粮食作物", "metadata": {"entity": "水稻"}},
    ]


def _build_test_entity_relations():
    return {
        "稻瘟病": [
            ("属于", "真菌性病害"),
            ("防治药剂", "三环唑"),
            ("防治药剂", "稻瘟灵"),
        ],
        "三环唑": [("属于", "杀菌剂")],
        "稻瘟灵": [("属于", "杀菌剂")],
        "水稻纹枯病": [("属于", "真菌性病害"), ("防治药剂", "井冈霉素")],
        "井冈霉素": [("属于", "抗生素类杀菌剂")],
        "水稻": [("病害", "稻瘟病"), ("病害", "水稻纹枯病")],
    }


def _build_test_reverse_relations():
    reverse = {}
    for head, rels in _build_test_entity_relations().items():
        for rel, tail in rels:
            reverse.setdefault(tail, []).append((head, rel))
    return reverse


def _make_rag_instance(
    search_mode: str = "mix", vector_weight: float = 0.4
):
    """构造一个绕过 _load_index 的 PocketGraphRAG 实例，所有依赖已注入"""
    from PocketGraphRAG.build_index import FAISSIndex
    from PocketGraphRAG.rag_system import PocketGraphRAG

    rag = PocketGraphRAG.__new__(PocketGraphRAG)
    rag.top_k = 5
    rag.use_multihop = False
    rag.use_conversation = False
    rag.search_mode = search_mode
    rag.data_path = "/tmp/test_triples.txt"
    rag.use_pagerank = True
    rag.pagerank_weight = 0.3
    rag.fusion_strategy = "rrf"
    rag.rrf_k = 60
    rag.use_hyde = False
    rag.use_query_router = False
    rag._query_router = None
    rag.use_self_check = False
    rag.use_schema = False
    rag._schema = None
    rag.vector_weight = vector_weight
    rag._reranker_model = None
    rag.kg_retriever = None
    rag._pagerank_scores = None
    rag._community_data = None
    rag.conversation = None
    rag.model = MockSentenceTransformer(dim=8)

    # 构建索引
    index = FAISSIndex(dimension=8)
    index.model = rag.model
    chunks = _build_test_chunks()
    # 直接构建，跳过模型调用
    texts = [c["text"] for c in chunks]
    embeddings = rag.model.encode(texts)
    index.index = __import__("faiss").IndexFlatIP(8)
    index.index.add(embeddings)
    index.texts = texts
    index.metadatas = [c["metadata"] for c in chunks]
    index._embeddings = embeddings
    index.dimension = 8
    # 重建 entity→chunk_ids 倒排表：_retrieve_pure_kg / _retrieve_by_entities
    # 走 get_chunks_by_entities() O(1) 查表，缺这张表会返回空导致测试 StopIteration。
    index._rebuild_entity_index()
    rag.index = index

    # 构建 KG 检索器 mock
    rag.kg_retriever = _make_mock_kg_retriever()
    rag._pagerank_scores = {"稻瘟病": 0.25, "三环唑": 0.15, "水稻": 0.20}

    return rag


def _make_mock_kg_retriever():
    """构造 mock KGDualRetriever"""
    retriever = MagicMock()
    er = _build_test_entity_relations()
    rr = _build_test_reverse_relations()

    retriever.entity_relations = er
    retriever.reverse_relations = rr

    def match_entities(query, top_k=5, threshold=None, return_scores=False):
        # 简单关键词匹配
        all_ents = list(er.keys())
        matched = [e for e in all_ents if e in query or query in e][:3]
        if return_scores:
            # 精确子串匹配记为 1.0，模拟真实行为
            return [(e, 1.0) for e in matched]
        return matched

    def match_relations(query):
        all_rels = {r for rels in er.values() for r, _ in rels}
        return [r for r in all_rels if r in query][:3]

    def local_search(query):
        seeds = match_entities(query)
        # 模拟 BFS 扩展：种子 + 邻域
        expanded = []
        for s in seeds:
            for r, t in er.get(s, []):
                if t not in seeds and t not in expanded:
                    expanded.append(t)
        return seeds + expanded

    def global_search(query):
        matched_rels = match_relations(query)
        result = []
        for head, rels in er.items():
            for r, t in rels:
                if r in matched_rels:
                    if head not in result:
                        result.append(head)
                    if t not in result:
                        result.append(t)
        return result

    def mix_search(query):
        return list(set(local_search(query)) | set(global_search(query)))

    retriever.match_entities.side_effect = match_entities
    retriever.match_relations.side_effect = match_relations
    retriever.local_search.side_effect = local_search
    retriever.global_search.side_effect = global_search
    retriever.mix_search.side_effect = mix_search
    retriever.compute_pagerank.return_value = {
        "稻瘟病": 0.25,
        "三环唑": 0.15,
        "水稻": 0.20,
    }

    return retriever


# ==========================
# Bug 1 回归：vector_weight 并发污染
# ==========================


class TestBug1VectorWeightConcurrency:
    """H4 修复回归：retrieve(answer) 调用不能修改 self.vector_weight"""

    def test_retrieve_does_not_modify_vector_weight(self):
        """retrieve 传入 vector_weight 不应改变实例属性"""
        rag = _make_rag_instance(search_mode="mix", vector_weight=0.4)
        original_vw = rag.vector_weight

        rag.retrieve("稻瘟病", vector_weight=0.8, search_mode="vector")

        assert rag.vector_weight == original_vw, "vector_weight 不应被修改"

    def test_retrieve_with_zero_vector_weight(self):
        rag = _make_rag_instance(search_mode="mix", vector_weight=0.4)
        rag.retrieve("稻瘟病", vector_weight=0.0, search_mode="vector")
        assert rag.vector_weight == 0.4

    def test_retrieve_with_full_vector_weight(self):
        rag = _make_rag_instance(search_mode="mix", vector_weight=0.4)
        rag.retrieve("稻瘟病", vector_weight=1.0, search_mode="vector")
        assert rag.vector_weight == 0.4

    def test_retrieve_clamps_out_of_range(self):
        rag = _make_rag_instance(search_mode="mix", vector_weight=0.4)
        # 超出范围的值应被 clamp，不抛异常
        rag.retrieve("稻瘟病", vector_weight=1.5, search_mode="vector")
        assert rag.vector_weight == 0.4
        rag.retrieve("稻瘟病", vector_weight=-0.5, search_mode="vector")
        assert rag.vector_weight == 0.4

    def test_concurrent_retrieve_no_pollution(self):
        """多线程并发 retrieve 不同的 vector_weight，实例属性应保持不变"""
        rag = _make_rag_instance(search_mode="vector", vector_weight=0.4)
        original_vw = rag.vector_weight
        errors = []

        def worker(vw):
            try:
                for _ in range(20):
                    rag.retrieve("稻瘟病", vector_weight=vw, search_mode="vector")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(vw,))
            for vw in [0.1, 0.5, 0.9, 0.0, 1.0]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发执行出错: {errors}"
        assert rag.vector_weight == original_vw, "并发后实例属性被污染"

    def test_mix_mode_uses_effective_vw(self):
        """mix 模式下传入 vector_weight 应实际生效（影响融合权重），且不改实例属性"""
        rag = _make_rag_instance(search_mode="mix", vector_weight=0.5)
        original_vw = rag.vector_weight

        # vector_weight=0.0 表示完全偏向 KG，向量结果权重为 0
        results_low_vw, _ = rag.retrieve("稻瘟病", vector_weight=0.0, search_mode="mix")
        # vector_weight=1.0 表示完全偏向向量
        results_high_vw, _ = rag.retrieve(
            "稻瘟病", vector_weight=1.0, search_mode="mix"
        )

        assert rag.vector_weight == original_vw, "实例属性被修改"
        # 两种权重下都应该有结果
        assert len(results_low_vw) > 0
        assert len(results_high_vw) > 0


# ==========================
# Bug 2 回归：incremental_index NameError
# ==========================


def _build_minimal_index(data_path: str, model, index_dir: str):
    """手动构建最小可用的索引目录（绕过 build_index_with_data 的真实模型加载）。

    覆盖 incremental_index.add_triples_incremental 需要的所有文件：
    - faiss.index / texts.json / metadatas.json / embeddings.npy  （FAISSIndex.load 用）
    - entity_faiss.index / entity_names.json                       （_append_embedding_index 用）
    - relation_faiss.index / relation_names.json                   （_append_embedding_index 用）
    - triples_manifest.json                                        （load_manifest 用）
    """
    import faiss

    from PocketGraphRAG.build_index import FAISSIndex
    from PocketGraphRAG.data_processor import KGProcessor
    from PocketGraphRAG.incremental_index import build_manifest_from_data, save_manifest

    os.makedirs(index_dir, exist_ok=True)

    # 1. 加载三元组并生成 chunks
    processor = KGProcessor(data_path)
    processor.load_triples()
    chunks = processor.process()

    # 2. 构建 FAISS 主索引
    index = FAISSIndex()
    index.build(chunks, model)
    index.save(index_dir)

    # 3. 构建实体嵌入索引
    entities = sorted(processor.entity_relations.keys())
    if entities:
        embs = model.encode(entities, normalize_embeddings=True)
        embs = np.array(embs, dtype="float32")
        ent_index = faiss.IndexFlatIP(embs.shape[1])
        ent_index.add(embs)
        faiss.write_index(ent_index, os.path.join(index_dir, "entity_faiss.index"))
        with open(
            os.path.join(index_dir, "entity_names.json"), "w", encoding="utf-8"
        ) as f:
            import json

            json.dump(entities, f, ensure_ascii=False)

    # 4. 构建关系嵌入索引
    relations = sorted({r for _, r, _ in processor.triples})
    if relations:
        embs = model.encode(relations, normalize_embeddings=True)
        embs = np.array(embs, dtype="float32")
        rel_index = faiss.IndexFlatIP(embs.shape[1])
        rel_index.add(embs)
        faiss.write_index(rel_index, os.path.join(index_dir, "relation_faiss.index"))
        with open(
            os.path.join(index_dir, "relation_names.json"), "w", encoding="utf-8"
        ) as f:
            import json

            json.dump(relations, f, ensure_ascii=False)

    # 5. 写 manifest
    save_manifest(index_dir, build_manifest_from_data(data_path))


class TestBug2IncrementalNameError:
    """incremental_index.py append 阶段 r 未定义 bug 回归"""

    def test_add_triples_persists_to_data_path(self, temp_dir):
        """新增三元组应正确 append 到 data_path（不抛 NameError）"""
        from PocketGraphRAG.incremental_index import add_triples_incremental

        # 准备初始索引
        data_path = os.path.join(temp_dir, "triples.txt")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write("稻瘟病|防治药剂|三环唑\n")
            f.write("三环唑|属于|杀菌剂\n")

        model = MockSentenceTransformer(dim=8)
        index_dir = os.path.join(temp_dir, "index")
        _build_minimal_index(data_path, model, index_dir)

        # 增量添加新三元组
        stats = add_triples_incremental(
            new_triples=[("新病害X", "防治药剂", "新药剂Y")],
            model=model,
            index_dir=index_dir,
            data_path=data_path,
        )

        # 关键断言：不抛 NameError，且新三元组被持久化
        assert stats["new_triples"] == 1
        assert stats["skipped_duplicates"] == 0

        with open(data_path, encoding="utf-8") as f:
            content = f.read()
        assert "新病害X" in content
        assert "新药剂Y" in content
        assert "防治药剂" in content

    def test_add_triples_with_special_chars(self, temp_dir):
        """三元组含特殊字符（如 |）应被清理，不破坏格式，不抛 NameError"""
        from PocketGraphRAG.incremental_index import add_triples_incremental

        data_path = os.path.join(temp_dir, "triples.txt")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write("实体A|关系|实体B\n")

        model = MockSentenceTransformer(dim=8)
        index_dir = os.path.join(temp_dir, "index")
        _build_minimal_index(data_path, model, index_dir)

        # 三元组中的 | 应被清理
        stats = add_triples_incremental(
            new_triples=[("实|体C", "关|系", "实|体D")],
            model=model,
            index_dir=index_dir,
            data_path=data_path,
        )

        assert stats["new_triples"] == 1
        with open(data_path, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        # 最后一行应该正好有 2 个 | 分隔符（清理过）
        last_line = lines[-1]
        assert last_line.count("|") == 2

    def test_add_duplicate_triples_skipped(self, temp_dir):
        """重复三元组应被 manifest 跳过，不写文件"""
        from PocketGraphRAG.incremental_index import add_triples_incremental

        data_path = os.path.join(temp_dir, "triples.txt")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write("稻瘟病|防治药剂|三环唑\n")

        model = MockSentenceTransformer(dim=8)
        index_dir = os.path.join(temp_dir, "index")
        _build_minimal_index(data_path, model, index_dir)

        # 同一三元组应被跳过
        stats = add_triples_incremental(
            new_triples=[("稻瘟病", "防治药剂", "三环唑")],
            model=model,
            index_dir=index_dir,
            data_path=data_path,
        )

        assert stats["skipped_duplicates"] == 1
        assert stats["new_triples"] == 0

        # 文件应只有一行
        with open(data_path, encoding="utf-8") as f:
            lines = [l for l in f.read().split("\n") if l.strip()]
        assert len(lines) == 1


# ==========================
# _basic_retrieve 6 种 search_mode
# ==========================


class TestBasicRetrieveSearchModes:
    """覆盖 _basic_retrieve 的 6 种 search_mode 分支"""

    def test_vector_mode(self):
        rag = _make_rag_instance(search_mode="vector")
        results, kg_path = rag._basic_retrieve("稻瘟病", top_k=3)
        assert len(results) <= 3
        assert kg_path["search_type"] == "vector"
        assert kg_path["seed_entities"] == []

    def test_local_mode(self):
        rag = _make_rag_instance(search_mode="local")
        results, kg_path = rag._basic_retrieve("稻瘟病", top_k=5)
        assert kg_path["search_type"] == "local"
        # 稻瘟病在 query 中，应被 match_entities 命中
        assert "稻瘟病" in kg_path["seed_entities"]

    def test_global_mode(self):
        rag = _make_rag_instance(search_mode="global")
        results, kg_path = rag._basic_retrieve("防治药剂", top_k=5)
        assert kg_path["search_type"] == "global"
        # "防治药剂" 是关系名，应被 match_relations 命中
        assert "防治药剂" in kg_path["matched_relations"]

    def test_mix_mode(self):
        rag = _make_rag_instance(search_mode="mix")
        results, kg_path = rag._basic_retrieve("稻瘟病", top_k=5)
        assert kg_path["search_type"] == "mix"
        assert "稻瘟病" in kg_path["seed_entities"]

    def test_mix_mode_filters_vector_results_by_seed_entities_first(self):
        rag = _make_rag_instance(search_mode="mix")
        rag.kg_retriever.match_entities.side_effect = lambda query, **kw: [("稻瘟病", 1.0)] if kw.get("return_scores") else ["稻瘟病"]
        rag.kg_retriever.match_entities_by_relation_value.side_effect = lambda query: []
        rag.kg_retriever.match_relations.side_effect = lambda query: []
        rag.kg_retriever.local_search.side_effect = lambda query: ["稻瘟病", "异稻缘蝽"]
        rag.kg_retriever.global_search.side_effect = lambda query: ["稻瘟病", "异稻缘蝽"]
        rag.kg_retriever.personalized_pagerank.side_effect = (
            lambda seed_entities: {"稻瘟病": 0.9, "异稻缘蝽": 0.8}
        )
        with patch.object(
            rag.index,
            "search",
            return_value=[
                ("偏题文本", 0.92, {"entity": "异稻缘蝽"}),
                ("稻瘟病相关文本", 0.90, {"entity": "稻瘟病"}),
            ],
        ):
            results, kg_path = rag._basic_retrieve("稻瘟病有什么症状？", top_k=5)

        entities = [meta.get("entity") for _, _, meta in results]
        assert kg_path["seed_entities"] == ["稻瘟病"]
        assert "异稻缘蝽" not in entities
        assert "稻瘟病" in entities

    def test_kg_only_mode(self):
        rag = _make_rag_instance(search_mode="kg_only")
        results, kg_path = rag._basic_retrieve("稻瘟病", top_k=5)
        assert kg_path["search_type"] == "kg_only"
        assert "稻瘟病" in kg_path["seed_entities"]

    def test_unknown_mode_fallback_to_vector(self):
        """无效 search_mode 应抛 ValueError（BUG #7：不再静默回退）"""
        rag = _make_rag_instance(search_mode="unknown_mode")
        with pytest.raises(ValueError, match="无效的 search_mode"):
            rag._basic_retrieve("稻瘟病", top_k=3)

    def test_vector_mode_returns_scored_results(self):
        rag = _make_rag_instance(search_mode="vector")
        results, _ = rag._basic_retrieve("稻瘟病", top_k=3)
        for text, score, meta in results:
            assert isinstance(text, str)
            assert isinstance(score, float)
            assert "entity" in meta

    def test_local_mode_lazy_loads_kg_retriever(self):
        """vector 模式实例化后切到 local，应懒加载 kg_retriever（H3 修复）"""
        rag = _make_rag_instance(search_mode="vector")
        rag.kg_retriever = None  # 模拟未加载
        # local 模式应触发懒加载
        with patch.object(rag, "_load_kg_retriever") as mock_load:
            mock_load.side_effect = lambda: setattr(
                rag, "kg_retriever", _make_mock_kg_retriever()
            )
            rag._basic_retrieve("稻瘟病", top_k=3, search_mode="local")
            mock_load.assert_called_once()


# ==========================
# 融合逻辑
# ==========================


class TestFusionLogic:
    def test_merge_results_rrf(self):
        rag = _make_rag_instance(search_mode="mix")
        rag.fusion_strategy = "rrf"
        vector_results = [
            ("doc1", 0.9, {"entity": "A"}),
            ("doc2", 0.8, {"entity": "B"}),
        ]
        kg_results = [("doc2", 1.5, {"entity": "B"}), ("doc3", 1.0, {"entity": "C"})]
        merged = rag._merge_results(vector_results, kg_results, top_k=5)
        # doc2 在两个列表都出现，应排前
        assert merged[0][0] == "doc2"
        assert len(merged) == 3

    def test_merge_results_weighted(self):
        rag = _make_rag_instance(search_mode="mix")
        rag.fusion_strategy = "weighted"
        vector_results = [("doc1", 0.9, {"entity": "A"})]
        kg_results = [("doc2", 1.5, {"entity": "B"})]
        merged = rag._merge_results(vector_results, kg_results, top_k=5)
        assert len(merged) == 2

    def test_merge_results_respects_vector_weight_param(self):
        """_merge_results 接受 vector_weight 参数，影响融合权重（H4）"""
        rag = _make_rag_instance(search_mode="mix")
        rag.fusion_strategy = "weighted"
        rag.vector_weight = 0.5

        vector_results = [("doc1", 1.0, {"entity": "A"})]
        kg_results = [("doc2", 1.0, {"entity": "B"})]

        # vw=1.0：完全偏向向量，doc1 应排前
        merged_high_vw = rag._merge_results(
            vector_results, kg_results, top_k=5, vector_weight=1.0
        )
        assert merged_high_vw[0][0] == "doc1"
        # 实例属性不变
        assert rag.vector_weight == 0.5

        # vw=0.0：完全偏向 KG，doc2 应排前
        merged_low_vw = rag._merge_results(
            vector_results, kg_results, top_k=5, vector_weight=0.0
        )
        assert merged_low_vw[0][0] == "doc2"
        assert rag.vector_weight == 0.5

    def test_retrieve_by_entities_empty_returns_vector(self):
        """空实体列表应回退到纯向量检索"""
        rag = _make_rag_instance(search_mode="local")
        results = rag._retrieve_by_entities("稻瘟病", [], top_k=3)
        assert len(results) <= 3

    def test_retrieve_by_entities_with_pagerank(self):
        rag = _make_rag_instance(search_mode="local")
        results = rag._retrieve_by_entities("稻瘟病", ["稻瘟病", "三环唑"], top_k=5)
        assert len(results) > 0

    def test_retrieve_by_entities_filters_irrelevant_vector_noise(self):
        rag = _make_rag_instance(search_mode="local")
        with patch.object(
            rag.index,
            "search",
            return_value=[
                ("偏题文本", 0.92, {"entity": "异稻缘蝽"}),
                ("稻瘟病相关文本", 0.90, {"entity": "稻瘟病"}),
                ("三环唑相关文本", 0.89, {"entity": "三环唑"}),
            ],
        ):
            results = rag._retrieve_by_entities(
                "稻瘟病有什么症状？", ["稻瘟病", "三环唑"], top_k=5
            )

        entities = [meta.get("entity") for _, _, meta in results]
        assert "异稻缘蝽" not in entities
        assert "稻瘟病" in entities

    def test_merge_results_keeps_top_vector_when_all_vector_hits_filtered_out(self):
        rag = _make_rag_instance(search_mode="mix")
        filtered = rag._filter_vector_results(
            "完全未知问题",
            [("偏题文本", 0.92, {"entity": "异稻缘蝽"})],
            allowed_entities=["稻瘟病"],
        )
        assert len(filtered) == 1
        assert filtered[0][2]["entity"] == "异稻缘蝽"

    def test_entity_match_uses_focus_terms_instead_of_full_question(self):
        rag = _make_rag_instance(search_mode="mix")
        assert rag._entity_matches_query("稻瘟病有什么症状？", "稻瘟病") is True
        assert rag._entity_matches_query("稻瘟病有什么症状？", "Ⅱ优633") is False

    def test_filter_results_for_symptom_question_drops_variety_and_medication_noise(self):
        rag = _make_rag_instance(search_mode="mix")
        results = [
            (
                "叶片出现褐色病斑，严重时穗颈枯死。防治时可喷施三环唑。",
                0.92,
                {"entity": "水稻穗颈瘟病"},
            ),
            (
                "【稻瘟病】\n药剂：50%多菌灵可湿性粉剂\n用量：200克/亩",
                0.90,
                {"entity": "稻瘟病"},
            ),
            (
                "【Ⅱ优633】\n产量表现：丰产性较好\n抗稻瘟病：感稻瘟病",
                0.89,
                {"entity": "Ⅱ优633"},
            ),
        ]

        filtered = rag._filter_results_for_question(
            "稻瘟病有什么症状？", "symptom", results
        )

        entities = [meta["entity"] for _, _, meta in filtered]
        assert entities == ["水稻穗颈瘟病", "稻瘟病"]

    def test_filter_results_for_symptom_question_keeps_focus_entity_but_prioritizes_symptom_evidence(self):
        rag = _make_rag_instance(search_mode="mix")
        results = [
            (
                "【水稻苗瘟病】\n症状表现：叶片出现褐色或黑褐色病斑\n症状表现：严重时导致幼苗枯死",
                0.92,
                {"entity": "水稻苗瘟病"},
            ),
            (
                "【稻瘟病】\n药剂：50%多菌灵可湿性粉剂\n用量：200克/亩",
                0.90,
                {"entity": "稻瘟病"},
            ),
            (
                "【Ⅱ优633】\n产量表现：丰产性较好\n抗稻瘟病：感稻瘟病",
                0.89,
                {"entity": "Ⅱ优633"},
            ),
        ]

        filtered = rag._filter_results_for_question(
            "稻瘟病有什么症状？", "symptom", results
        )

        entities = [meta["entity"] for _, _, meta in filtered]
        assert entities == ["水稻苗瘟病", "稻瘟病"]

    def test_filter_results_for_list_question_prefers_disease_entities(self):
        rag = _make_rag_instance(search_mode="mix")
        results = [
            (
                "15%三环唑可湿性粉剂30-40克/亩，分蘖末期至孕穗初期施药。",
                0.92,
                {"entity": "三环唑"},
            ),
            (
                "三环唑可用于防治稻瘟病。",
                0.91,
                {"entity": "稻瘟病"},
            ),
            (
                "三环唑可用于防治纹枯病。",
                0.90,
                {"entity": "纹枯病"},
            ),
        ]

        filtered = rag._filter_results_for_question(
            "三环唑可以防治哪些病害？", "list", results
        )

        entities = [meta["entity"] for _, _, meta in filtered]
        assert entities == ["稻瘟病", "纹枯病"]

    def test_promote_relation_target_results_prefers_relation_value_entities_for_disease_list_query(self):
        rag = _make_rag_instance(search_mode="mix")
        results = [
            ("15%三环唑可湿性粉剂30-40克/亩。", 0.92, {"entity": "三环唑"}),
            ("分蘖末期至孕穗初期施药。", 0.91, {"entity": "三环唑预防"}),
        ]

        with patch.object(
            rag,
            "_retrieve_pure_kg",
            return_value=[
                ("三环唑可用于防治稻瘟病。", 0.95, {"entity": "稻瘟病"}),
                ("三环唑可用于防治纹枯病。", 0.94, {"entity": "纹枯病"}),
            ],
        ):
            promoted = rag._promote_relation_target_results(
                "三环唑可以防治哪些病害？",
                "list",
                results,
                {"relation_value_entities": ["稻瘟病", "纹枯病"]},
                top_k=4,
            )

        entities = [meta["entity"] for _, _, meta in promoted]
        assert entities[:2] == ["稻瘟病", "纹枯病"]

    def test_retrieve_pure_kg_seed_priority(self):
        """种子实体的分数应高于扩展实体"""
        rag = _make_rag_instance(search_mode="kg_only")
        results = rag._retrieve_pure_kg(
            entity_names=["稻瘟病", "三环唑", "水稻"],
            seed_entities=["稻瘟病"],
            top_k=10,
        )
        # 找出稻瘟病对应的 chunk，应在三环唑/水稻之前
        texts = [r[0] for r in results]
        idx_blast = next(i for i, t in enumerate(texts) if "稻瘟病" in t and "由" in t)
        idx_others = [
            i for i, t in enumerate(texts) if "三环唑" in t or "水稻是我国" in t
        ]
        if idx_others:
            assert idx_blast < min(idx_others), "种子实体应排前"

    def test_retrieve_pure_kg_empty(self):
        rag = _make_rag_instance(search_mode="kg_only")
        assert rag._retrieve_pure_kg([], [], top_k=5) == []


# ==========================
# retrieve 主入口
# ==========================


class TestRetrieveMain:
    def test_retrieve_default_search_mode(self):
        rag = _make_rag_instance(search_mode="mix")
        results, kg_path = rag.retrieve("稻瘟病")
        assert kg_path["search_type"] in ("mix", "multihop")

    def test_retrieve_override_search_mode(self):
        rag = _make_rag_instance(search_mode="mix")
        _, kg_path = rag.retrieve("稻瘟病", search_mode="vector")
        assert kg_path["search_type"] == "vector"

    def test_retrieve_top_k_limit(self):
        rag = _make_rag_instance(search_mode="vector")
        results, _ = rag.retrieve("稻瘟病", top_k=2)
        assert len(results) <= 2

    def test_retrieve_top_k_zero_fallback(self):
        """top_k=None 应回退到实例默认值"""
        rag = _make_rag_instance(search_mode="vector")
        rag.top_k = 3
        results, _ = rag.retrieve("稻瘟病", top_k=None)
        assert len(results) <= 3

    def test_retrieve_reranker_off_by_default(self):
        rag = _make_rag_instance(search_mode="vector")
        results, _ = rag.retrieve("稻瘟病")
        # use_reranker=None 默认 False，不应调用 reranker
        # 仅验证不抛异常即可

    def test_retrieve_reranker_explicit_off(self):
        rag = _make_rag_instance(search_mode="vector")
        results, _ = rag.retrieve("稻瘟病", use_reranker=False)
        assert len(results) > 0


# ==========================
# answer 主入口
# ==========================


class TestAnswerMain:
    def test_answer_no_llm_returns_retrieval_only(self):
        """无 LLM 时应返回纯检索结果"""
        rag = _make_rag_instance(search_mode="mix")
        with patch("PocketGraphRAG.rag_system.has_llm", return_value=False):
            result = rag.answer("稻瘟病怎么防治？")
        assert "answer" in result
        assert "sources" in result
        assert "pipeline_info" in result
        assert "检索到的相关知识" in result["answer"]
        assert len(result["sources"]) > 0

    def test_answer_with_mock_llm(self):
        rag = _make_rag_instance(search_mode="mix")
        with (
            patch("PocketGraphRAG.rag_system.has_llm", return_value=True),
            patch(
                "PocketGraphRAG.rag_system.call_llm",
                return_value="三环唑可防治稻瘟病[1][2]",
            ),
        ):
            result = rag.answer("稻瘟病怎么防治？")
        assert "三环唑可防治稻瘟病[1][2]" in result["answer"]
        assert "## 结论" not in result["answer"]
        assert len(result["sources"]) > 0
        assert result["pipeline_info"]["search_mode"] == "mix"
        assert result["pipeline_info"]["response_mode"] == "llm_standardized"
        assert result["pipeline_info"]["question_type"] == "method"

    def test_answer_pipeline_info_fields(self):
        rag = _make_rag_instance(search_mode="mix")
        with patch("PocketGraphRAG.rag_system.has_llm", return_value=False):
            result = rag.answer("稻瘟病")
        info = result["pipeline_info"]
        # 必须包含所有声明字段
        assert "multihop_used" in info
        assert "search_mode" in info
        assert "kg_entities_matched" in info
        assert "query_rewritten" in info
        assert "hyde_used" in info
        assert "query_routed" in info
        assert "self_check_used" in info
        assert "self_check_level" in info
        assert "top_k" in info
        assert "vector_weight" in info
        assert "reranker_used" in info
        assert "kg_path" in info

    def test_answer_returns_effective_vw_not_instance(self):
        """answer 的 pipeline_info.vector_weight 应是 effective_vw，不改实例属性"""
        rag = _make_rag_instance(search_mode="mix")
        rag.vector_weight = 0.4
        with patch("PocketGraphRAG.rag_system.has_llm", return_value=False):
            result = rag.answer("稻瘟病", vector_weight=0.8)
        assert result["pipeline_info"]["vector_weight"] == 0.8
        assert rag.vector_weight == 0.4  # 实例属性不变

    def test_answer_search_mode_override(self):
        rag = _make_rag_instance(search_mode="mix")
        with patch("PocketGraphRAG.rag_system.has_llm", return_value=False):
            result = rag.answer("稻瘟病", search_mode="vector")
        assert result["pipeline_info"]["search_mode"] == "vector"

    def test_answer_top_k_override(self):
        rag = _make_rag_instance(search_mode="vector")
        with patch("PocketGraphRAG.rag_system.has_llm", return_value=False):
            result = rag.answer("稻瘟病", top_k=2)
        assert result["pipeline_info"]["top_k"] == 2
        assert len(result["sources"]) <= 2


class TestAnswerFallback:
    def test_answer_uses_structured_fallback_when_llm_returns_empty_refusal(self):
        rag = _make_rag_instance(search_mode="mix")
        with (
            patch("PocketGraphRAG.rag_system.has_llm", return_value=True),
            patch(
                "PocketGraphRAG.rag_system.call_llm",
                return_value="知识库中未找到相关信息。",
            ),
        ):
            result = rag.answer("稻瘟病怎么防治？")

        assert "我目前只能根据命中的知识确认这些信息" in result["answer"]
        assert "稻瘟病" in result["answer"]
        assert result["pipeline_info"]["response_mode"] == "retrieval_fallback"
        assert result["pipeline_info"]["fallback_reason"] == "llm_empty_or_generic"
        assert result["sources"], "fallback 仍应保留 sources"

    def test_answer_marks_refusal_bucket_for_unmatched_kg_query(self):
        rag = _make_rag_instance(search_mode="mix")
        with patch.object(
            rag,
            "retrieve",
            return_value=(
                [("irrelevant text", 0.8, {"entity": "x"})],
                {
                    "search_type": "mix",
                    "seed_entities": [],
                    "matched_relations": [],
                    "expanded_entities": [],
                },
            ),
        ):
            result = rag.answer("完全不相关的问题xyz")

        assert result["pipeline_info"]["refused"] is True
        assert result["pipeline_info"]["failure_bucket"] == "no_entity_or_relation_hit"

    def test_answer_stream_emits_structured_fallback_chunk(self):
        rag = _make_rag_instance(search_mode="mix")
        with (
            patch("PocketGraphRAG.rag_system.has_llm", return_value=True),
            patch(
                "PocketGraphRAG.rag_system.call_llm",
                return_value=iter(["", ""]),
            ),
        ):
            chunks = list(rag.answer_stream("稻瘟病怎么防治？"))

        final_chunk = [c for c in chunks if "chunk" in c][-1]
        info_chunk = next(
            c
            for c in chunks
            if c.get("pipeline_info", {}).get("response_mode") == "retrieval_fallback"
        )
        assert "我目前只能根据命中的知识确认这些信息" in final_chunk["full_answer"]
        assert info_chunk["pipeline_info"]["response_mode"] == "retrieval_fallback"

    def test_answer_uses_structured_fallback_when_llm_answer_has_no_citations(self):
        rag = _make_rag_instance(search_mode="mix")
        with (
            patch("PocketGraphRAG.rag_system.has_llm", return_value=True),
            patch(
                "PocketGraphRAG.rag_system.call_llm",
                return_value="三环唑可防治稻瘟病",
            ),
        ):
            result = rag.answer("稻瘟病怎么防治？")

        assert result["pipeline_info"]["response_mode"] == "retrieval_fallback"
        assert result["pipeline_info"]["fallback_reason"] == "missing_citations"
        assert "我目前只能根据命中的知识确认这些信息" in result["answer"]
        assert "## 检索到的结构化结果" not in result["answer"]


class TestAnswerStandardization:
    def test_answer_standardizes_cited_method_answer(self):
        rag = _make_rag_instance(search_mode="mix")
        with (
            patch("PocketGraphRAG.rag_system.has_llm", return_value=True),
            patch(
                "PocketGraphRAG.rag_system.call_llm",
                return_value="三环唑可用于防治稻瘟病[1][2]。分蘖末期至孕穗初期可施药[2]。",
            ),
        ):
            result = rag.answer("稻瘟病怎么防治？")

        assert result["pipeline_info"]["response_mode"] == "llm_standardized"
        assert result["pipeline_info"]["question_type"] == "method"
        assert "三环唑可用于防治稻瘟病[1][2]" in result["answer"]
        assert "分蘖末期至孕穗初期可施药[2]" in result["answer"]
        assert "## 结论" not in result["answer"]

    def test_answer_strips_template_headers_from_cited_llm_output(self):
        rag = _make_rag_instance(search_mode="mix")
        with (
            patch("PocketGraphRAG.rag_system.has_llm", return_value=True),
            patch(
                "PocketGraphRAG.rag_system.call_llm",
                return_value=(
                    "结论\n三环唑可用于防治稻瘟病[1][2]。\n\n"
                    "关键事实\n分蘖末期至孕穗初期可施药[2]。"
                ),
            ),
        ):
            result = rag.answer("稻瘟病怎么防治？")

        assert result["pipeline_info"]["response_mode"] == "llm_standardized"
        assert "结论" not in result["answer"]
        assert "关键事实" not in result["answer"]
        assert "三环唑可用于防治稻瘟病[1][2]。" in result["answer"]
        assert "分蘖末期至孕穗初期可施药[2]。" in result["answer"]

    def test_answer_standardization_keeps_generic_query_lightweight(self):
        rag = _make_rag_instance(search_mode="mix")
        with (
            patch("PocketGraphRAG.rag_system.has_llm", return_value=True),
            patch(
                "PocketGraphRAG.rag_system.call_llm",
                return_value="稻瘟病是一种真菌性病害[1]。",
            ),
        ):
            result = rag.answer("稻瘟病是什么？")

        assert result["pipeline_info"]["question_type"] == "definition"
        assert "稻瘟病是一种真菌性病害[1]。" in result["answer"]
        assert "## 结论" not in result["answer"]

    def test_answer_promotes_relation_value_entities_for_disease_list_query(self):
        rag = _make_rag_instance(search_mode="mix")
        raw_results = [
            ("15%三环唑可湿性粉剂30-40克/亩。", 0.92, {"entity": "三环唑"}),
            ("分蘖末期至孕穗初期施药。", 0.91, {"entity": "三环唑预防"}),
        ]
        kg_path = {
            "search_type": "mix",
            "seed_entities": ["三环唑"],
            "matched_relations": ["防治措施"],
            "expanded_entities": ["三环唑预防"],
            "relation_value_entities": ["稻瘟病", "纹枯病"],
        }
        with (
            patch("PocketGraphRAG.rag_system.has_llm", return_value=False),
            patch.object(rag, "retrieve", return_value=(raw_results, kg_path)),
            patch.object(
                rag,
                "_retrieve_pure_kg",
                return_value=[
                    ("三环唑可用于防治稻瘟病。", 0.95, {"entity": "稻瘟病"}),
                    ("三环唑可用于防治纹枯病。", 0.94, {"entity": "纹枯病"}),
                ],
            ),
        ):
            result = rag.answer("三环唑可以防治哪些病害？")

        assert [src["entity"] for src in result["sources"][:2]] == ["稻瘟病", "纹枯病"]
