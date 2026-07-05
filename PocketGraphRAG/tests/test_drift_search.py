"""DRIFT Search 单元测试

测试三阶段流程：Primer → Drift 迭代 → Output
以及降级路径（无 LLM、解析失败等）
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG.drift_search import (
    DriftIteration,
    DriftResult,
    _parse_json_response,
    drift_iteration,
    drift_primer,
    drift_search,
)


class TestParseJsonResponse:
    """_parse_json_response 解析逻辑测试"""

    def test_parse_plain_json(self):
        data = _parse_json_response('{"key": "value"}')
        assert data == {"key": "value"}

    def test_parse_markdown_json_block(self):
        data = _parse_json_response('```json\n{"key": "value"}\n```')
        assert data == {"key": "value"}

    def test_parse_markdown_plain_block(self):
        data = _parse_json_response('```\n{"key": "value"}\n```')
        assert data == {"key": "value"}

    def test_parse_invalid_json_returns_none(self):
        assert _parse_json_response("not json") is None

    def test_parse_none_returns_none(self):
        assert _parse_json_response(None) is None

    def test_parse_empty_returns_none(self):
        assert _parse_json_response("") is None
        assert _parse_json_response("   ") is None


class TestDriftPrimer:
    """Primer 阶段测试"""

    def test_primer_no_llm_degrades_gracefully(self):
        """无 LLM 时应返回空后续查询（触发调用方降级）"""
        with patch("PocketGraphRAG.drift_search.has_llm", return_value=False):
            community_fn = MagicMock(return_value=[{"summary": "社区1"}])
            answer, followups, communities = drift_primer(
                "测试查询", community_fn, top_k_communities=2
            )
        assert answer == ""
        assert followups == []
        assert len(communities) == 1

    def test_primer_with_llm_generates_followups(self):
        """有 LLM 时应生成中间答案和后续查询"""
        mock_response = '''{
            "intermediate_answer": "初步答案",
            "followup_queries": ["查询1", "查询2"]
        }'''
        with patch("PocketGraphRAG.drift_search.has_llm", return_value=True), \
             patch("PocketGraphRAG.drift_search.call_llm", return_value=mock_response):
            community_fn = MagicMock(return_value=[{"summary": "社区1"}])
            answer, followups, communities = drift_primer(
                "测试查询", community_fn, top_k_communities=2, n_followup=3
            )
        assert answer == "初步答案"
        assert len(followups) == 2
        assert followups == ["查询1", "查询2"]

    def test_primer_limits_followup_count(self):
        """后续查询数量应被 n_followup 限制"""
        mock_response = '''{
            "intermediate_answer": "答案",
            "followup_queries": ["q1", "q2", "q3", "q4", "q5"]
        }'''
        with patch("PocketGraphRAG.drift_search.has_llm", return_value=True), \
             patch("PocketGraphRAG.drift_search.call_llm", return_value=mock_response):
            community_fn = MagicMock(return_value=[])
            _, followups, _ = drift_primer(
                "查询", community_fn, n_followup=2
            )
        assert len(followups) == 2

    def test_primer_handles_llm_parse_failure(self):
        """LLM 响应解析失败时应降级"""
        with patch("PocketGraphRAG.drift_search.has_llm", return_value=True), \
             patch("PocketGraphRAG.drift_search.call_llm", return_value="invalid json"):
            community_fn = MagicMock(return_value=[{"summary": "社区"}])
            answer, followups, _ = drift_primer("查询", community_fn)
        assert answer == ""
        assert followups == []

    def test_primer_handles_empty_communities(self):
        """社区搜索返回空时应仍能工作"""
        mock_response = '''{
            "intermediate_answer": "答案",
            "followup_queries": ["q1"]
        }'''
        with patch("PocketGraphRAG.drift_search.has_llm", return_value=True), \
             patch("PocketGraphRAG.drift_search.call_llm", return_value=mock_response):
            community_fn = MagicMock(return_value=[])
            answer, followups, communities = drift_primer(
                "查询", community_fn
            )
        assert len(communities) == 0
        assert answer == "答案"
        assert len(followups) == 1


class TestDriftIteration:
    """Drift 迭代阶段测试"""

    def test_iteration_collects_chunks(self):
        """迭代应收集后续查询检索到的文本块"""
        chunks = [("文本块1", 0.9, {"entity": "A"}), ("文本块2", 0.8, {"entity": "B"})]
        kg_info = {"seed_entities": ["A", "B"]}
        local_fn = MagicMock(return_value=(chunks, kg_info))

        with patch("PocketGraphRAG.drift_search.has_llm", return_value=True), \
             patch(
                 "PocketGraphRAG.drift_search.call_llm",
                 return_value='{"intermediate_answer": "更新答案", "followup_queries": []}',
             ):
            result = drift_iteration(
                query="原始查询",
                followup_queries=["后续查询1"],
                local_retrieve_fn=local_fn,
                intermediate_answer="初步答案",
                iteration=1,
            )
        assert result.iteration == 1
        assert len(result.retrieved_chunks) == 2
        assert "A" in result.seed_entities
        assert result.intermediate_answer == "更新答案"

    def test_iteration_deduplicates_chunks(self):
        """相同文本块应去重"""
        chunk1 = ("相同文本", 0.9, {"entity": "A"})
        chunk2 = ("相同文本", 0.8, {"entity": "B"})  # 前100字符相同
        local_fn = MagicMock(
            side_effect=[
                ([chunk1], {"seed_entities": ["A"]}),
                ([chunk2], {"seed_entities": ["B"]}),
            ]
        )

        with patch("PocketGraphRAG.drift_search.has_llm", return_value=False):
            result = drift_iteration(
                query="查询",
                followup_queries=["q1", "q2"],
                local_retrieve_fn=local_fn,
                intermediate_answer="",
                iteration=1,
            )
        assert len(result.retrieved_chunks) == 1  # 去重后只剩1个

    def test_iteration_no_evidence_keeps_old_answer(self):
        """无证据时应保留旧中间答案"""
        local_fn = MagicMock(return_value=([], {"seed_entities": []}))

        with patch("PocketGraphRAG.drift_search.has_llm", return_value=True):
            result = drift_iteration(
                query="查询",
                followup_queries=["q1"],
                local_retrieve_fn=local_fn,
                intermediate_answer="旧答案",
                iteration=1,
            )
        assert result.intermediate_answer == "旧答案"
        assert len(result.retrieved_chunks) == 0


class TestDriftSearch:
    """DRIFT 主流程测试"""

    def test_drift_no_followups_returns_early(self):
        """Primer 无后续查询时应早退"""
        with patch("PocketGraphRAG.drift_search.drift_primer") as mock_primer:
            mock_primer.return_value = ("初步答案", [], [{"summary": "社区"}])
            community_fn = MagicMock()
            local_fn = MagicMock()

            result = drift_search(
                "查询", community_fn, local_fn, max_iterations=2
            )

        assert result.primer_answer == "初步答案"
        assert result.final_answer == "初步答案"
        assert len(result.iterations) == 0
        local_fn.assert_not_called()  # 无后续查询不应调用局部检索

    def test_drift_full_flow_with_iterations(self):
        """完整 DRIFT 流程：Primer + 2 轮迭代"""
        with patch("PocketGraphRAG.drift_search.drift_primer") as mock_primer, \
             patch("PocketGraphRAG.drift_search.drift_iteration") as mock_iter:
            mock_primer.return_value = ("初步答案", ["q1", "q2"], [])
            # 第1轮：生成1个新查询
            iter1 = DriftIteration(
                iteration=1,
                intermediate_answer="更新1",
                followup_queries=["q3"],
                retrieved_chunks=[("chunk1", 0.9, {})],
                seed_entities=["A"],
            )
            # 第2轮：无新查询，结束
            iter2 = DriftIteration(
                iteration=2,
                intermediate_answer="最终答案",
                followup_queries=[],
                retrieved_chunks=[("chunk2", 0.8, {})],
                seed_entities=["B"],
            )
            mock_iter.side_effect = [iter1, iter2]

            result = drift_search(
                "查询",
                MagicMock(),
                MagicMock(),
                max_iterations=2,
            )

        assert result.total_iterations == 2
        assert result.final_answer == "最终答案"
        assert len(result.all_chunks) == 2
        assert set(result.all_entities) == {"A", "B"}

    def test_drift_stops_when_no_new_queries(self):
        """后续查询为空时应停止迭代"""
        with patch("PocketGraphRAG.drift_search.drift_primer") as mock_primer, \
             patch("PocketGraphRAG.drift_search.drift_iteration") as mock_iter:
            mock_primer.return_value = ("答案", ["q1"], [])
            iter1 = DriftIteration(
                iteration=1,
                intermediate_answer="答案",
                followup_queries=[],  # 无新查询
                retrieved_chunks=[],
                seed_entities=[],
            )
            mock_iter.return_value = iter1

            result = drift_search(
                "查询", MagicMock(), MagicMock(), max_iterations=5
            )

        assert result.total_iterations == 1  # 只迭代1轮就停止
        mock_iter.assert_called_once()

    def test_drift_respects_max_iterations(self):
        """应尊重 max_iterations 上限"""
        with patch("PocketGraphRAG.drift_search.drift_primer") as mock_primer, \
             patch("PocketGraphRAG.drift_search.drift_iteration") as mock_iter:
            mock_primer.return_value = ("答案", ["q1"], [])
            # 每轮都生成新查询，模拟无限迭代
            def make_iter(n):
                return DriftIteration(
                    iteration=n,
                    intermediate_answer=f"答案{n}",
                    followup_queries=["q_new"],
                    retrieved_chunks=[],
                    seed_entities=[],
                )
            mock_iter.side_effect = [make_iter(1), make_iter(2), make_iter(3)]

            result = drift_search(
                "查询", MagicMock(), MagicMock(), max_iterations=2
            )

        assert result.total_iterations == 2  # 不超过 max_iterations

    def test_drift_deduplicates_chunks(self):
        """最终结果应去重 chunks"""
        with patch("PocketGraphRAG.drift_search.drift_primer") as mock_primer, \
             patch("PocketGraphRAG.drift_search.drift_iteration") as mock_iter:
            mock_primer.return_value = ("", ["q1"], [])
            chunk = ("相同文本块", 0.9, {})
            iter1 = DriftIteration(
                iteration=1,
                intermediate_answer="答案",
                followup_queries=[],
                retrieved_chunks=[chunk],
                seed_entities=["A"],
            )
            iter2 = DriftIteration(
                iteration=2,
                intermediate_answer="答案",
                followup_queries=[],
                retrieved_chunks=[chunk],  # 相同 chunk
                seed_entities=["A"],
            )
            mock_iter.side_effect = [iter1, iter2]

            result = drift_search(
                "查询", MagicMock(), MagicMock(), max_iterations=2
            )

        assert len(result.all_chunks) == 1  # 去重后只剩1个


class TestDriftDataStructures:
    """数据结构测试"""

    def test_drift_iteration_default_fields(self):
        it = DriftIteration(iteration=1, intermediate_answer="ans")
        assert it.iteration == 1
        assert it.intermediate_answer == "ans"
        assert it.followup_queries == []
        assert it.retrieved_chunks == []
        assert it.seed_entities == []

    def test_drift_result_default_fields(self):
        r = DriftResult(primer_answer="p", final_answer="f")
        assert r.primer_answer == "p"
        assert r.final_answer == "f"
        assert r.iterations == []
        assert r.all_chunks == []
        assert r.all_entities == []
        assert r.total_iterations == 0

    def test_drift_result_total_iterations_property(self):
        r = DriftResult(primer_answer="", final_answer="")
        r.iterations = [
            DriftIteration(iteration=1, intermediate_answer="a1"),
            DriftIteration(iteration=2, intermediate_answer="a2"),
        ]
        assert r.total_iterations == 2
