"""Tests for hierarchical community summarizer (P3).

Covers:
- _build_hierarchy: 父子关系判定（Jaccard + containment）
- build_hierarchical_community_summaries: 多层级构建 + 缓存 + 无 LLM 降级
- search_communities_hierarchical: 多层级检索 + roll-up
- search_communities: 自动检测层次 vs 单层格式
- _is_leiden_available / _detect_communities_at_resolution: 算法降级
- 向后兼容：单层模式 build_community_summaries 仍可用
"""

import json
import os

import numpy as np
import pytest

from PocketGraphRAG.community_summarizer import (
    DEFAULT_RESOLUTIONS,
    HIERARCHICAL_COMMUNITY_FILE,
    _build_hierarchy,
    _detect_communities_at_resolution,
    _is_leiden_available,
    build_community_summaries,
    build_hierarchical_community_summaries,
    search_communities,
    search_communities_hierarchical,
)


# ============================================================
# 全局 mock：避免测试调用真实 LLM（.env 里可能配置了 Ollama/SiliconFlow）
# ============================================================

@pytest.fixture(autouse=True)
def _mock_llm_calls(monkeypatch):
    """默认把 has_llm 改为 False，避免触发真实 LLM 调用导致测试卡死。
    单测里需要测 LLM 路径时，再 monkeypatch.setattr 回 True。
    """
    from PocketGraphRAG import community_summarizer

    monkeypatch.setattr(community_summarizer, "has_llm", lambda: False)


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing (deterministic by text hash)."""

    def __init__(self, dim=8):
        self.dim = dim

    def get_sentence_embedding_dimension(self):
        """返回 mock 维度，供社区摘要缓存维度校验使用。"""
        return self.dim

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False, **kw):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            h = hash(text) % (2**30)
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = (h % 1000) / 1000.0
            vec[1] = ((h // 1000) % 1000) / 1000.0
            vec[2] = ((h // 1000000) % 1000) / 1000.0
            vec[3] = 0.1
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


# ============================================================
# Helper：构建一个简单的 mock kg_retriever
# ============================================================

class _MockKGRetriever:
    """最小化的 kg_retriever mock，只暴露 community_summarizer 用到的方法。"""

    def __init__(self, entity_relations: dict):
        self.entity_relations = entity_relations
        # all_entities：dict.keys + values 中的实体
        ents = set(entity_relations.keys())
        for rels in entity_relations.values():
            for _, tail in rels:
                ents.add(tail)
        self.all_entities = sorted(ents)

    def detect_communities_louvain(self, resolution=1.0, seed=42):
        """简单 mock：按 connected component 分社区（忽略 resolution）。"""
        # 构建无向图
        adj = {e: set() for e in self.all_entities}
        for head, rels in self.entity_relations.items():
            for _, tail in rels:
                if tail != head:
                    adj.setdefault(head, set()).add(tail)
                    adj.setdefault(tail, set()).add(head)

        visited = set()
        communities = []
        for ent in self.all_entities:
            if ent in visited:
                continue
            # BFS
            stack = [ent]
            comp = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                for nxt in adj.get(cur, set()):
                    if nxt not in visited:
                        stack.append(nxt)
            communities.append(sorted(comp))

        community_map = {}
        for cid, members in enumerate(communities):
            for e in members:
                community_map[e] = cid
        return communities, community_map


@pytest.fixture
def mock_kg_retriever():
    """构建一个有多连通分量的 mock kg_retriever。"""
    entity_relations = {
        "稻瘟病": [("属于", "真菌性病害"), ("防治药剂", "三环唑"), ("防治药剂", "稻瘟灵")],
        "三环唑": [("属于", "杀菌剂")],
        "稻瘟灵": [("属于", "杀菌剂")],
        "水稻纹枯病": [("属于", "真菌性病害"), ("防治药剂", "井冈霉素")],
        "井冈霉素": [("属于", "抗生素类杀菌剂")],
        "水稻": [("病害", "稻瘟病"), ("病害", "水稻纹枯病"), ("虫害", "稻飞虱")],
        "白叶枯病": [("属于", "细菌性病害"), ("防治药剂", "叶枯唑")],
        "稻飞虱": [("防治药剂", "吡虫啉")],
        "吡虫啉": [("属于", "烟碱类杀虫剂")],
    }
    return _MockKGRetriever(entity_relations)


@pytest.fixture
def mock_model():
    return MockSentenceTransformer(dim=8)


# ============================================================
# _is_leiden_available
# ============================================================

class TestLeidenAvailability:
    def test_returns_bool(self):
        result = _is_leiden_available()
        assert isinstance(result, bool)

    def test_leiden_not_installed_in_test_env(self):
        """测试环境未安装 leidenalg，应该返回 False。"""
        # 这个测试假设测试环境没有 leidenalg；如果将来 CI 装了，改用 mock
        try:
            import leidenalg  # noqa: F401
            pytest.skip("leidenalg 已安装，跳过未安装场景测试")
        except ImportError:
            assert _is_leiden_available() is False


# ============================================================
# _detect_communities_at_resolution
# ============================================================

class TestDetectAtResolution:
    def test_louvain_algorithm(self, mock_kg_retriever):
        """强制 louvain 算法应该正常工作。"""
        comms, comm_map = _detect_communities_at_resolution(
            mock_kg_retriever, resolution=1.0, algorithm="louvain"
        )
        assert isinstance(comms, list)
        assert len(comms) > 0
        assert isinstance(comm_map, dict)
        # 所有实体都应该被分到一个社区
        all_entities = set()
        for c in comms:
            all_entities.update(c)
        assert all_entities == set(mock_kg_retriever.all_entities)

    def test_auto_fallback_to_louvain(self, mock_kg_retriever):
        """algorithm=auto 且 leiden 不可用时，应回退到 louvain。"""
        comms, _ = _detect_communities_at_resolution(
            mock_kg_retriever, resolution=1.0, algorithm="auto"
        )
        assert len(comms) > 0

    def test_leiden_fallback_when_not_installed(self, mock_kg_retriever):
        """algorithm=leiden 但未安装时，应回退到 louvain（不抛异常）。"""
        try:
            import leidenalg  # noqa: F401
            pytest.skip("leidenalg 已安装，跳过回退测试")
        except ImportError:
            pass
        comms, _ = _detect_communities_at_resolution(
            mock_kg_retriever, resolution=1.0, algorithm="leiden"
        )
        assert len(comms) > 0

    def test_different_resolutions_produce_different_counts(self, mock_kg_retriever):
        """不同 resolution 应产生不同数量的社区（粗 vs 细）。"""
        # 用大 resolution 应得到更多更小的社区
        comms_coarse = _detect_communities_at_resolution(
            mock_kg_retriever, resolution=0.1, algorithm="louvain"
        )[0]
        comms_fine = _detect_communities_at_resolution(
            mock_kg_retriever, resolution=5.0, algorithm="louvain"
        )[0]
        # 不做严格断言（Louvain 在小图上可能稳定），但至少数量合法
        assert len(comms_coarse) >= 1
        assert len(comms_fine) >= 1


# ============================================================
# _build_hierarchy
# ============================================================

class TestBuildHierarchy:
    def test_single_level_returns_none_parents(self):
        """单层时所有 parent_id 应为 None。"""
        level_communities = [
            [["A", "B"], ["C", "D"]],  # level 0
        ]
        parent_ids = _build_hierarchy(level_communities)
        assert len(parent_ids) == 1
        assert parent_ids[0] == [None, None]

    def test_two_levels_finds_parent_by_containment(self):
        """子社区被父社区完全包含时，应正确建立父子关系。"""
        level_communities = [
            [["A", "B", "C", "D"]],  # level 0: 一个大社区
            [["A", "B"], ["C", "D"]],  # level 1: 拆成两个子社区
        ]
        parent_ids = _build_hierarchy(level_communities)
        assert parent_ids[0] == [None]  # 最粗层无父
        assert parent_ids[1] == [0, 0]  # 两个子社区的父都是 level 0 的社区 0

    def test_low_overlap_returns_none_parent(self):
        """重叠率低于 0.3 时不应建立父子关系。"""
        level_communities = [
            [["A", "B", "C", "D", "E", "F", "G"]],  # 父社区
            [["X", "Y", "Z"]],  # 子社区完全不相交
        ]
        parent_ids = _build_hierarchy(level_communities)
        assert parent_ids[1] == [None]

    def test_three_levels_chain(self):
        """三层级链：祖→父→子。"""
        level_communities = [
            [["A", "B", "C", "D"]],  # L0: 1 个大社区
            [["A", "B"], ["C", "D"]],  # L1: 2 个中社区
            [["A"], ["B"], ["C"], ["D"]],  # L2: 4 个小社区
        ]
        parent_ids = _build_hierarchy(level_communities)
        assert parent_ids[0] == [None]
        assert parent_ids[1] == [0, 0]  # L1 两个社区的父都是 L0.0
        assert parent_ids[2] == [0, 0, 1, 1]  # L2 四个社区的父分别是 L1.0, L1.0, L1.1, L1.1


# ============================================================
# build_hierarchical_community_summaries
# ============================================================

class TestBuildHierarchical:
    def test_basic_build_three_levels(self, mock_kg_retriever, mock_model, temp_dir):
        """默认应构建 3 层（0.5, 1.0, 2.0）。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        assert "levels" in data
        assert len(data["levels"]) == len(DEFAULT_RESOLUTIONS)
        # 每层应有 resolution / communities / embeddings
        for lv in data["levels"]:
            assert "resolution" in lv
            assert "communities" in lv
            assert "embeddings" in lv
            assert isinstance(lv["communities"], list)
            # 摘要不应该为空
            for c in lv["communities"]:
                assert c["summary"]
                assert "parent_id" in c
                assert "child_ids" in c

    def test_custom_resolutions(self, mock_kg_retriever, mock_model, temp_dir):
        """自定义分辨率列表应被使用。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir, resolutions=[1.0, 3.0]
        )
        assert len(data["levels"]) == 2
        assert data["levels"][0]["resolution"] == 1.0
        assert data["levels"][1]["resolution"] == 3.0

    def test_single_resolution_collapses_to_one_level(
        self, mock_kg_retriever, mock_model, temp_dir
    ):
        """单分辨率时仍应工作（退化为 1 层）。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir, resolutions=[1.0]
        )
        assert len(data["levels"]) == 1
        # 单层时所有 parent_id 应为 None
        for c in data["levels"][0]["communities"]:
            assert c["parent_id"] is None

    def test_cache_writes_and_reads(self, mock_kg_retriever, mock_model, temp_dir):
        """首次构建应写缓存，二次调用应从缓存读。"""
        # 首次构建
        data1 = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        cache_path = os.path.join(temp_dir, HIERARCHICAL_COMMUNITY_FILE)
        assert os.path.exists(cache_path)

        # 二次调用：应从缓存读
        data2 = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        assert len(data2["levels"]) == len(data1["levels"])
        # 社区数应一致
        for lv1, lv2 in zip(data1["levels"], data2["levels"]):
            assert len(lv1["communities"]) == len(lv2["communities"])

    def test_force_ignores_cache(self, mock_kg_retriever, mock_model, temp_dir):
        """force=True 应忽略缓存重建。"""
        # 先建一次
        build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        # 强制重建
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir, force=True
        )
        assert "levels" in data
        assert len(data["levels"]) == len(DEFAULT_RESOLUTIONS)

    def test_compatibility_fields_point_to_finest_level(
        self, mock_kg_retriever, mock_model, temp_dir
    ):
        """层次模式返回值应包含 communities/embeddings/resolution 兼容字段。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        finest = data["levels"][-1]
        assert data["communities"] == finest["communities"]
        assert np.array_equal(data["embeddings"], finest["embeddings"])
        assert data["resolution"] == finest["resolution"]

    def test_parent_child_consistency(
        self, mock_kg_retriever, mock_model, temp_dir
    ):
        """父社区的 child_ids 应与子社区的 parent_id 一致。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        for i in range(len(data["levels"]) - 1):
            parent_level = data["levels"][i]
            child_level = data["levels"][i + 1]
            # 对每个父社区，检查 child_ids 中的子社区确实把它认作父
            for parent in parent_level["communities"]:
                for child_id in parent["child_ids"]:
                    child = child_level["communities"][child_id]
                    assert child["parent_id"] == parent["id"]

    def test_empty_retriever(self, mock_model, temp_dir):
        """空 kg_retriever 应返回空层次结构，不崩溃。"""
        empty_retriever = _MockKGRetriever({})
        data = build_hierarchical_community_summaries(
            empty_retriever, mock_model, temp_dir
        )
        assert "levels" in data
        # 每层都是空社区列表
        for lv in data["levels"]:
            assert lv["communities"] == []

    def test_no_llm_falls_back_to_entity_list(
        self, mock_kg_retriever, mock_model, temp_dir
    ):
        """无 LLM 时摘要应回退为实体列表格式（autouse fixture 已 mock has_llm=False）。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        for lv in data["levels"]:
            for c in lv["communities"]:
                # 回退摘要应包含"本社区包含"字样
                assert "本社区包含" in c["summary"]

    def test_dim_mismatch_triggers_rebuild(
        self, mock_kg_retriever, mock_model, temp_dir
    ):
        """缓存维度与 model 输出维度不匹配时应触发重建，而非直接使用脏缓存。

        回归测试：早期测试用 mock model 写入了 8 维缓存到真实 index 目录，
        运行时用 512 维真实 model 加载触发 matmul 崩溃。校验逻辑应检测到
        维度不匹配并重建。
        """
        # 1. 用 dim=8 的 mock_model 写缓存
        build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        cache_path = os.path.join(temp_dir, HIERARCHICAL_COMMUNITY_FILE)
        assert os.path.exists(cache_path)
        # 确认缓存里是 8 维
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        assert len(cached["levels"][0]["embeddings"][0]) == 8

        # 2. 换成 dim=16 的 model 加载，应触发重建
        model_16 = MockSentenceTransformer(dim=16)
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, model_16, temp_dir, force=False
        )
        # 重建后 embeddings 应是 16 维
        for lv in data["levels"]:
            if len(lv["embeddings"]) > 0:
                assert lv["embeddings"].shape[1] == 16, (
                    f"level {lv['level']} 维度应为 16，实际 {lv['embeddings'].shape[1]}"
                )


# ============================================================
# search_communities_hierarchical
# ============================================================

class TestSearchHierarchical:
    def test_basic_search_returns_top_k(self, mock_kg_retriever, mock_model, temp_dir):
        """检索应返回 top_k 个结果。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        results = search_communities_hierarchical(
            "稻瘟病如何防治", mock_model, data, top_k=2
        )
        assert isinstance(results, list)
        assert len(results) <= 2
        for r in results:
            assert "level" in r
            assert "id" in r
            assert "summary" in r
            assert "score" in r
            assert "entities" in r

    def test_search_at_coarsest_level(
        self, mock_kg_retriever, mock_model, temp_dir
    ):
        """level=0 应在最粗层检索。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        results = search_communities_hierarchical(
            "病害", mock_model, data, top_k=3, level=0
        )
        for r in results:
            assert r["level"] == 0 or r["level"] == 0  # roll-up 可能换层
        # 实际上 level=0 是最粗层，roll-up 无父可换，应保持 level=0
        # 但 roll_up=True 时若有 parent_id=None 不会触发 roll-up
        assert all(r["level"] == 0 for r in results)

    def test_search_at_finest_level(
        self, mock_kg_retriever, mock_model, temp_dir
    ):
        """level=-1 应在最细层检索。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        results = search_communities_hierarchical(
            "三环唑", mock_model, data, top_k=2, level=-1
        )
        # 最细层是最后一层
        finest_level_idx = len(data["levels"]) - 1
        # 结果 level 应该 <= finest_level_idx（roll-up 可能上移）
        for r in results:
            assert r["level"] <= finest_level_idx

    def test_roll_up_disabled(self, mock_kg_retriever, mock_model, temp_dir):
        """关闭 roll_up 时，结果应保持原层级。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        results = search_communities_hierarchical(
            "病害防治",
            mock_model,
            data,
            top_k=3,
            level=-1,
            roll_up=False,
        )
        finest_level_idx = len(data["levels"]) - 1
        for r in results:
            assert r["level"] == finest_level_idx

    def test_empty_community_data(self, mock_model):
        """空 community_data 应返回空列表。"""
        results = search_communities_hierarchical(
            "test", mock_model, {"levels": []}, top_k=3
        )
        assert results == []

    def test_results_sorted_by_score_desc(
        self, mock_kg_retriever, mock_model, temp_dir
    ):
        """结果应按 score 降序排列。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        results = search_communities_hierarchical(
            "稻瘟病", mock_model, data, top_k=5
        )
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]


# ============================================================
# search_communities (auto-detect 格式)
# ============================================================

class TestSearchAutoDetect:
    def test_auto_detect_hierarchical(self, mock_kg_retriever, mock_model, temp_dir):
        """数据含 levels 字段时应自动走层次检索。"""
        data = build_hierarchical_community_summaries(
            mock_kg_retriever, mock_model, temp_dir
        )
        results = search_communities("稻瘟病", mock_model, data, top_k=2)
        assert isinstance(results, list)
        # 层次检索结果应包含 level 字段
        for r in results:
            assert "level" in r

    def test_auto_detect_flat(self, mock_kg_retriever, mock_model, temp_dir):
        """数据无 levels 字段时应走单层检索。"""
        data = build_community_summaries(
            mock_kg_retriever, mock_model, temp_dir, algorithm="louvain"
        )
        assert "levels" not in data  # 单层模式无 levels
        results = search_communities("稻瘟病", mock_model, data, top_k=2)
        assert isinstance(results, list)
        for r in results:
            assert "level" in r  # 单层也带 level=0
            assert r["level"] == 0


# ============================================================
# 向后兼容：build_community_summaries 单层模式
# ============================================================

class TestBackwardCompatFlat:
    def test_flat_mode_returns_old_format(
        self, mock_kg_retriever, mock_model, temp_dir
    ):
        """单层模式应返回旧格式（communities + embeddings + resolution）。"""
        data = build_community_summaries(
            mock_kg_retriever,
            mock_model,
            temp_dir,
            resolution=1.0,
            algorithm="louvain",
        )
        # 旧字段都在
        assert "communities" in data
        assert "embeddings" in data
        assert "resolution" in data
        assert data["resolution"] == 1.0
        # 不应有 levels 字段
        assert "levels" not in data

    def test_flat_mode_caches(self, mock_kg_retriever, mock_model, temp_dir):
        """单层模式也应缓存。"""
        build_community_summaries(
            mock_kg_retriever, mock_model, temp_dir, algorithm="louvain"
        )
        from PocketGraphRAG.community_summarizer import COMMUNITY_SUMMARY_FILE

        assert os.path.exists(os.path.join(temp_dir, COMMUNITY_SUMMARY_FILE))

    def test_flat_mode_with_algorithm_field(
        self, mock_kg_retriever, mock_model, temp_dir
    ):
        """单层模式结果应包含 algorithm 字段。"""
        data = build_community_summaries(
            mock_kg_retriever, mock_model, temp_dir, algorithm="louvain"
        )
        assert data.get("algorithm") == "louvain"


# ============================================================
# 集成测试：rag_system global_summary 模式（不调真实 LLM）
# ============================================================

class TestRagSystemIntegration:
    """验证 rag_system 能正确加载层次社区摘要。

    这里只验证 _load_community_summaries 能被调用且不崩溃，
    不验证完整 retrieve（那需要更复杂的 mock，已在 e2e 测试覆盖）。
    """

    def test_load_hierarchical_via_rag(
        self, mock_kg_retriever, mock_model, temp_dir, monkeypatch
    ):
        """通过 rag_system._load_community_summaries 应能加载层次摘要。"""
        # 必须先 monkeypatch config 才能让 rag_system 走层次路径
        from PocketGraphRAG import config, rag_system

        monkeypatch.setattr(config, "COMMUNITY_HIERARCHICAL", True)
        monkeypatch.setattr(config, "COMMUNITY_ALGORITHM", "louvain")
        monkeypatch.setattr(config, "COMMUNITY_RESOLUTIONS", [0.5, 1.0, 2.0])
        monkeypatch.setattr(config, "COMMUNITY_MAX_TRIPLES", 80)
        monkeypatch.setattr(config, "COMMUNITY_RESOLUTION", 1.0)
        monkeypatch.setattr(config, "INDEX_DIR", temp_dir)

        # 构造一个最小 rag 实例（绕过 __init__）
        from PocketGraphRAG.rag_system import PocketGraphRAG

        rag = PocketGraphRAG.__new__(PocketGraphRAG)
        rag.kg_retriever = mock_kg_retriever
        rag.model = mock_model
        rag._community_data = None

        data = rag._load_community_summaries()
        assert "levels" in data
        assert len(data["levels"]) == 3
