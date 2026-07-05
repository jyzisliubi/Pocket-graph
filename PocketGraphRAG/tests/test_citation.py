"""Citation 引用功能单元测试（对标 LightRAG 2025.03 Citation 功能）

验证 Source 模型的 citation_id 字段 + 答案中 [1][2] 标注的回溯能力。
"""

import pytest

from PocketGraphRAG.api_server import Source, QAResponse, PipelineInfo, KGPathInfo
from PocketGraphRAG.rag_system import _extract_citation_ids


# ==========================
# 1. _extract_citation_ids 提取引用编号
# ==========================


class TestExtractCitationIds:
    """引用编号提取测试"""

    def test_single_citation(self):
        assert _extract_citation_ids("根据信息[1]可知") == ["1"]

    def test_multiple_citations(self):
        assert _extract_citation_ids("综合[1]和[2]的信息") == ["1", "2"]

    def test_consecutive_citations(self):
        assert _extract_citation_ids("来源[1][2]都提到") == ["1", "2"]

    def test_no_citation(self):
        assert _extract_citation_ids("没有引用的文本") == []

    def test_empty_text(self):
        assert _extract_citation_ids("") == []

    def test_citation_at_end(self):
        assert _extract_citation_ids("某事实[3]") == ["3"]

    def test_mixed_text(self):
        text = "阿甘正传[1]是1994年的电影，主演是汤姆·汉克斯[2]。"
        ids = _extract_citation_ids(text)
        assert "1" in ids
        assert "2" in ids


# ==========================
# 2. Source 模型 citation_id 字段
# ==========================


class TestSourceCitationId:
    """Source 模型 citation_id 字段测试"""

    def test_default_none(self):
        s = Source(entity="阿甘正传", text="剧情简介", score=0.9)
        assert s.citation_id is None

    def test_set_citation_id(self):
        s = Source(entity="阿甘正传", text="剧情简介", score=0.9, citation_id=1)
        assert s.citation_id == 1

    def test_serialization(self):
        s = Source(entity="阿甘正传", text="剧情简介", score=0.9, citation_id=1)
        d = s.model_dump()
        assert d["citation_id"] == 1
        assert d["entity"] == "阿甘正传"

    def test_deserialization(self):
        d = {"entity": "盗梦空间", "text": "科幻片", "score": 0.8, "citation_id": 2}
        s = Source(**d)
        assert s.citation_id == 2


# ==========================
# 3. QAResponse 集成 citation
# ==========================


class TestQAResponseCitation:
    """QAResponse 完整 citation 集成测试"""

    def test_response_with_citation_ids(self):
        sources = [
            Source(entity="阿甘正传", text="1994年电影", score=0.9, citation_id=1),
            Source(entity="肖申克的救赎", text="也是1994年", score=0.85, citation_id=2),
        ]
        resp = QAResponse(
            answer="阿甘正传[1]和肖申克的救赎[2]都是1994年的经典电影。",
            sources=sources,
            pipeline_info=PipelineInfo(kg_path=KGPathInfo()),
            effective_query="1994年的电影",
        )
        # 验证 answer 中的 [1] [2] 能映射回 sources
        citation_ids = _extract_citation_ids(resp.answer)
        assert "1" in citation_ids
        assert "2" in citation_ids
        # 每个 citation_id 都能在 sources 中找到对应
        for cid in citation_ids:
            matched = [s for s in resp.sources if s.citation_id == int(cid)]
            assert len(matched) == 1, f"citation_id {cid} 找不到对应 source"

    def test_sources_ordered_by_citation_id(self):
        """sources 的 citation_id 应按顺序 1, 2, 3..."""
        sources = [
            Source(entity="A", text="a", score=0.9, citation_id=1),
            Source(entity="B", text="b", score=0.8, citation_id=2),
            Source(entity="C", text="c", score=0.7, citation_id=3),
        ]
        resp = QAResponse(answer="test", sources=sources)
        ids = [s.citation_id for s in resp.sources]
        assert ids == [1, 2, 3]


# ==========================
# 4. citation 回溯集成测试（模拟 rag_system 行为）
# ==========================


class TestCitationTraceability:
    """引用回溯集成测试"""

    def test_simulated_rag_response(self):
        """模拟 RAG 系统返回的带引用答案"""
        # 模拟检索结果（text, score, meta）
        results = [
            ("阿甘正传是1994年的美国电影，由罗伯特·泽米吉斯执导。", 0.95, {"entity": "阿甘正传"}),
            ("肖申克的救赎也是1994年的电影，弗兰克·德拉邦特执导。", 0.88, {"entity": "肖申克的救赎"}),
            ("盗梦空间是2010年的科幻片。", 0.75, {"entity": "盗梦空间"}),
        ]

        # 模拟 rag_system.answer 构建 sources 的逻辑
        sources = [
            {
                "entity": meta["entity"],
                "score": float(score),
                "text": text,
                "source_type": "vector",
                "citation_id": idx + 1,
            }
            for idx, (text, score, meta) in enumerate(results)
        ]

        # 模拟 LLM 生成的带引用答案
        answer = "1994年的经典电影包括阿甘正传[1]和肖申克的救赎[2]。盗梦空间[3]则是2010年的。"

        # 验证：答案中每个 [N] 都能映射到 sources[N-1]
        citation_ids = _extract_citation_ids(answer)
        assert len(citation_ids) == 3

        for cid in citation_ids:
            idx = int(cid) - 1
            assert 0 <= idx < len(sources)
            assert sources[idx]["citation_id"] == int(cid)

    def test_citation_id_matches_source_order(self):
        """citation_id 应与 sources 列表顺序一致（1-based）"""
        results = [
            ("文本1", 0.9, {"entity": "实体1"}),
            ("文本2", 0.8, {"entity": "实体2"}),
            ("文本3", 0.7, {"entity": "实体3"}),
        ]
        sources = [
            {
                "entity": meta["entity"],
                "score": float(score),
                "text": text,
                "citation_id": idx + 1,
            }
            for idx, (text, score, meta) in enumerate(results)
        ]
        # 验证 citation_id 与列表索引的对应关系
        for i, s in enumerate(sources):
            assert s["citation_id"] == i + 1
            assert s["entity"] == f"实体{i + 1}"
