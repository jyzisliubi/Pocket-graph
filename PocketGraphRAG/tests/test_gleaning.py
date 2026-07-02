"""Gleaning 多轮抽取单元测试

测试 _parse_triples_result 解析器 + extract_triples_from_text 的 gleaning 循环。
使用 mock LLM，不依赖真实 API。
"""

import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG.kg_extractor import (
    Triple,
    _GLEANING_PROMPT,
    _parse_triples_result,
    extract_triples_from_text,
)


# ==========================
# _parse_triples_result 解析器
# ==========================


class TestParseTriplesResult:
    def test_parse_valid_json(self):
        result = json.dumps(
            {
                "triples": [
                    {
                        "head": "稻瘟病",
                        "relation": "致病菌",
                        "tail": "稻瘟病菌",
                        "confidence": 0.95,
                        "evidence": "由稻瘟病菌引起",
                    }
                ]
            },
            ensure_ascii=False,
        )
        triples = _parse_triples_result(result)
        assert len(triples) == 1
        assert triples[0].head == "稻瘟病"
        assert triples[0].relation == "致病菌"
        assert triples[0].tail == "稻瘟病菌"
        assert triples[0].confidence == 0.95

    def test_parse_empty_result(self):
        assert _parse_triples_result("") == []
        assert _parse_triples_result(None) == []

    def test_parse_markdown_wrapped(self):
        result = '```json\n{"triples": [{"head": "A", "relation": "r", "tail": "B", "confidence": 0.9}]}\n```'
        triples = _parse_triples_result(result)
        assert len(triples) == 1
        assert triples[0].head == "A"

    def test_parse_with_surrounding_text(self):
        result = '以下是抽取结果：\n{"triples": [{"head": "A", "relation": "r", "tail": "B", "confidence": 0.9}]}\n解析完成。'
        triples = _parse_triples_result(result)
        assert len(triples) == 1

    def test_parse_empty_triples_list(self):
        result = '{"triples": []}'
        triples = _parse_triples_result(result)
        assert len(triples) == 0

    def test_parse_invalid_entity_filtered(self):
        """空实体/超长实体被过滤"""
        result = json.dumps(
            {
                "triples": [
                    {"head": "", "relation": "r", "tail": "B", "confidence": 0.9},
                    {
                        "head": "A",
                        "relation": "r",
                        "tail": "x" * 300,
                        "confidence": 0.9,
                    },
                    {"head": "A", "relation": "r", "tail": "B", "confidence": 0.9},
                ]
            },
            ensure_ascii=False,
        )
        triples = _parse_triples_result(result)
        assert len(triples) == 1  # 只有第三条有效

    def test_parse_confidence_clamped(self):
        result = json.dumps(
            {
                "triples": [
                    {"head": "A", "relation": "r", "tail": "B", "confidence": 1.5},
                    {"head": "C", "relation": "r", "tail": "D", "confidence": -0.3},
                ]
            },
            ensure_ascii=False,
        )
        triples = _parse_triples_result(result)
        assert len(triples) == 2
        assert triples[0].confidence == 1.0  # clamped
        assert triples[1].confidence == 0.0  # clamped

    def test_parse_with_schema_normalization(self):
        from PocketGraphRAG.schema import RelationSchema

        schema = RelationSchema()
        result = json.dumps(
            {
                "triples": [
                    {
                        "head": "稻瘟病",
                        "relation": "症状",
                        "tail": "病斑",
                        "confidence": 0.9,
                    }
                ]
            },
            ensure_ascii=False,
        )
        triples = _parse_triples_result(result, schema=schema)
        assert len(triples) == 1
        assert triples[0].relation == "症状表现"  # 归一化

    def test_parse_delimiter_fallback(self):
        """JSON 解析失败时回退到 delimiter 格式"""
        result = "这不是JSON\n<|#|>稻瘟病|致病菌|稻瘟病菌|0.9|由稻瘟病菌引起<|#|>"
        triples = _parse_triples_result(result)
        assert len(triples) == 1
        assert triples[0].head == "稻瘟病"


# ==========================
# Gleaning 循环（mock LLM）
# ==========================


def _make_llm_response(triples_data):
    """构造 LLM 返回的 JSON 字符串"""
    return json.dumps({"triples": triples_data}, ensure_ascii=False)


class TestGleaningLoop:
    """测试 extract_triples_from_text 的 gleaning_steps 参数"""

    def test_gleaning_steps_zero_no_extra_calls(self):
        """gleaning_steps=0 时不触发追问（向后兼容）"""
        initial = _make_llm_response(
            [
                {
                    "head": "稻瘟病",
                    "relation": "致病菌",
                    "tail": "稻瘟病菌",
                    "confidence": 0.95,
                }
            ]
        )
        call_count = [0]

        def mock_call_llm(*args, **kwargs):
            call_count[0] += 1
            return initial

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.call_llm", side_effect=mock_call_llm),
        ):
            triples = extract_triples_from_text("测试文本", gleaning_steps=0)

        assert len(triples) == 1
        assert call_count[0] == 1  # 只调用了 1 次（首轮），无 gleaning

    def test_gleaning_adds_new_triples(self):
        """gleaning_steps=1 时追问一轮，合并新增三元组"""
        first_response = _make_llm_response(
            [
                {
                    "head": "稻瘟病",
                    "relation": "致病菌",
                    "tail": "稻瘟病菌",
                    "confidence": 0.95,
                }
            ]
        )
        gleaning_response = _make_llm_response(
            [
                {
                    "head": "稻瘟病",
                    "relation": "症状表现",
                    "tail": "病斑",
                    "confidence": 0.9,
                }
            ]
        )

        responses = [first_response, gleaning_response]
        call_idx = [0]

        def mock_call_llm(*args, **kwargs):
            if call_idx[0] < len(responses):
                r = responses[call_idx[0]]
                call_idx[0] += 1
                return r
            return ""

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.call_llm", side_effect=mock_call_llm),
        ):
            triples = extract_triples_from_text("测试文本", gleaning_steps=1)

        assert len(triples) == 2  # 首轮 1 条 + gleaning 1 条
        heads = {t.head for t in triples}
        assert "稻瘟病" in heads
        relations = {t.relation for t in triples}
        assert "致病菌" in relations
        assert "症状表现" in relations

    def test_gleaning_dedup_duplicates(self):
        """gleaning 返回重复三元组时去重"""
        first_response = _make_llm_response(
            [
                {
                    "head": "稻瘟病",
                    "relation": "致病菌",
                    "tail": "稻瘟病菌",
                    "confidence": 0.95,
                }
            ]
        )
        gleaning_response = _make_llm_response(
            [
                {
                    "head": "稻瘟病",
                    "relation": "致病菌",
                    "tail": "稻瘟病菌",
                    "confidence": 0.9,
                },  # 重复
                {
                    "head": "三环唑",
                    "relation": "防治",
                    "tail": "稻瘟病",
                    "confidence": 0.88,
                },  # 新增
            ]
        )

        responses = [first_response, gleaning_response]
        call_idx = [0]

        def mock_call_llm(*args, **kwargs):
            if call_idx[0] < len(responses):
                r = responses[call_idx[0]]
                call_idx[0] += 1
                return r
            return ""

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.call_llm", side_effect=mock_call_llm),
        ):
            triples = extract_triples_from_text("测试文本", gleaning_steps=1)

        assert len(triples) == 2  # 去重后 2 条（不是 3 条）

    def test_gleaning_empty_response_stops_early(self):
        """gleaning 返回空列表时提前结束"""
        first_response = _make_llm_response(
            [
                {
                    "head": "稻瘟病",
                    "relation": "致病菌",
                    "tail": "稻瘟病菌",
                    "confidence": 0.95,
                }
            ]
        )
        gleaning_response = _make_llm_response([])  # 空列表

        responses = [first_response, gleaning_response, gleaning_response]
        call_idx = [0]

        def mock_call_llm(*args, **kwargs):
            if call_idx[0] < len(responses):
                r = responses[call_idx[0]]
                call_idx[0] += 1
                return r
            return ""

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.call_llm", side_effect=mock_call_llm),
        ):
            triples = extract_triples_from_text("测试文本", gleaning_steps=3)

        assert len(triples) == 1  # 只有首轮的 1 条
        # gleaning 第1轮返回空 → added=0 → 提前结束，不应有第2、3轮
        assert call_idx[0] == 2  # 首轮 + gleaning第1轮 = 2 次调用

    def test_gleaning_multiple_rounds(self):
        """gleaning_steps=2 执行两轮追问"""
        first = _make_llm_response(
            [{"head": "A", "relation": "r1", "tail": "B", "confidence": 0.9}]
        )
        glean1 = _make_llm_response(
            [{"head": "C", "relation": "r2", "tail": "D", "confidence": 0.85}]
        )
        glean2 = _make_llm_response(
            [{"head": "E", "relation": "r3", "tail": "F", "confidence": 0.8}]
        )

        responses = [first, glean1, glean2]
        call_idx = [0]

        def mock_call_llm(*args, **kwargs):
            if call_idx[0] < len(responses):
                r = responses[call_idx[0]]
                call_idx[0] += 1
                return r
            return ""

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.call_llm", side_effect=mock_call_llm),
        ):
            triples = extract_triples_from_text("测试文本", gleaning_steps=2)

        assert len(triples) == 3  # 3 轮各 1 条
        assert call_idx[0] == 3  # 首轮 + 2 轮 gleaning = 3 次

    def test_gleaning_prompt_format(self):
        """_GLEANING_PROMPT 能正确 format"""
        formatted = _GLEANING_PROMPT.format(
            entities="稻瘟病、三环唑",
            n_triples=3,
            text="测试原文",
        )
        assert "稻瘟病、三环唑" in formatted
        assert "3" in formatted
        assert "测试原文" in formatted

    def test_gleaning_no_llm_returns_empty(self):
        """未配置 LLM 时返回空列表"""
        with patch("PocketGraphRAG.llm.has_llm", return_value=False):
            triples = extract_triples_from_text("测试文本", gleaning_steps=2)
        assert triples == []

    def test_gleaning_first_round_empty_no_gleaning(self):
        """首轮抽取为空时不触发 gleaning"""
        first_response = _make_llm_response([])
        call_count = [0]

        def mock_call_llm(*args, **kwargs):
            call_count[0] += 1
            return first_response

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.call_llm", side_effect=mock_call_llm),
        ):
            triples = extract_triples_from_text("测试文本", gleaning_steps=3)

        assert triples == []
        assert call_count[0] == 1  # 只有首轮调用，不触发 gleaning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
