"""Multi-Model Fusion 单元测试

覆盖 extract_triples_multi_model 的核心逻辑：
- union 策略：并集 + 去重
- intersect 策略：交集
- 模型失败容错（单模型异常不影响整体）
- 单模型路径（models 为空时降级到 extract_triples_from_text）
- _extract_model_override 全局变量正确设置和清理
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG import llm as _llm
from PocketGraphRAG.kg_extractor import (
    Triple,
    extract_triples_multi_model,
)


def _make_triple(h, r, t, conf=0.9):
    return Triple(head=h, relation=r, tail=t, confidence=conf, evidence="", source_chunk=0)


class TestExtractTriplesMultiModelBasic:
    """基础功能测试"""

    def test_empty_models_falls_back_to_single(self):
        """models 为空时降级到单模型抽取"""
        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            mock_extract.return_value = [_make_triple("A", "r", "B")]
            triples, stats = extract_triples_multi_model(
                text="test",
                models=[],
            )
        assert len(triples) == 1
        assert stats == {}
        mock_extract.assert_called_once()

    def test_models_not_enough_falls_back(self):
        """models 只有一个时仍走融合路径（记录 stats）"""
        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            mock_extract.return_value = [_make_triple("A", "r", "B")]
            triples, stats = extract_triples_multi_model(
                text="test",
                models=["qwen-flash"],
            )
        assert len(triples) == 1
        assert stats == {"qwen-flash": 1}

    def test_union_strategy_dedup(self):
        """union 策略：相同三元组去重"""
        t1 = _make_triple("盗梦空间", "导演", "诺兰", conf=0.95)
        t2 = _make_triple("盗梦空间", "导演", "诺兰", conf=0.9)
        t3 = _make_triple("盗梦空间", "主演", "迪卡普里奥", conf=0.9)

        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            # 模型1 抽到 t1, t3；模型2 抽到 t2, t3
            mock_extract.side_effect = [[t1, t3], [t2, t3]]
            triples, stats = extract_triples_multi_model(
                text="test",
                models=["qwen-flash", "qwen-max"],
                strategy="union",
            )
        # t1/t2 重复（去重后保留一个），t3 两次都抽到但去重后保留一个
        assert len(triples) == 2
        assert stats == {"qwen-flash": 2, "qwen-max": 2}
        # 验证全局变量已清理
        assert _llm._extract_model_override is None

    def test_intersect_strategy(self):
        """intersect 策略：只保留所有模型都抽到的"""
        t1 = _make_triple("A", "r", "B")
        t2 = _make_triple("C", "r", "D")
        t3 = _make_triple("E", "r", "F")

        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            # 模型1: t1, t2, t3；模型2: t2, t3
            mock_extract.side_effect = [[t1, t2, t3], [t2, t3]]
            triples, stats = extract_triples_multi_model(
                text="test",
                models=["m1", "m2"],
                strategy="intersect",
            )
        # 只有 t2, t3 在两个模型中都出现
        keys = {t.key() for t in triples}
        assert keys == {t2.key(), t3.key()}
        assert stats == {"m1": 3, "m2": 2}


class TestExtractTriplesMultiModelErrorHandling:
    """错误处理测试"""

    def test_single_model_failure_does_not_break(self):
        """单个模型抽取失败不影响其他模型"""
        t1 = _make_triple("A", "r", "B")

        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            # 模型1 抛异常，模型2 正常返回
            mock_extract.side_effect = [RuntimeError("API 超时"), [t1]]
            triples, stats = extract_triples_multi_model(
                text="test",
                models=["bad-model", "good-model"],
                strategy="union",
            )
        assert len(triples) == 1
        assert stats == {"bad-model": 0, "good-model": 1}
        # 全局变量在异常后也要清理
        assert _llm._extract_model_override is None

    def test_all_models_fail_returns_empty(self):
        """所有模型都失败时返回空列表"""
        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            mock_extract.side_effect = [
                RuntimeError("m1 failed"),
                RuntimeError("m2 failed"),
            ]
            triples, stats = extract_triples_multi_model(
                text="test",
                models=["m1", "m2"],
                strategy="union",
            )
        assert triples == []
        assert stats == {"m1": 0, "m2": 0}


class TestExtractModelOverride:
    """全局变量 _extract_model_override 的设置/清理测试"""

    def test_override_set_during_extract(self):
        """抽取期间 _extract_model_override 被设置为当前模型"""
        captured = []

        def fake_extract(text, chunk_index, temperature, schema, gleaning_steps):
            # 记录调用时的 override 值
            captured.append(_llm._extract_model_override)
            return [_make_triple("A", "r", "B")]

        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text",
            side_effect=fake_extract,
        ):
            extract_triples_multi_model(
                text="test",
                models=["m1", "m2"],
                strategy="union",
            )
        assert captured == ["m1", "m2"]

    def test_override_cleared_after_success(self):
        """成功完成后清理"""
        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            mock_extract.return_value = [_make_triple("A", "r", "B")]
            extract_triples_multi_model(
                text="test",
                models=["m1", "m2"],
            )
        assert _llm._extract_model_override is None

    def test_override_cleared_after_exception(self):
        """异常时也要清理"""
        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            mock_extract.side_effect = RuntimeError("fail")
            extract_triples_multi_model(
                text="test",
                models=["m1"],
            )
        assert _llm._extract_model_override is None


class TestFusionStrategiesAdvanced:
    """融合策略边界情况"""

    def test_union_with_completely_disjoint(self):
        """union 完全不相交：直接合并"""
        t1 = _make_triple("A", "r", "B")
        t2 = _make_triple("C", "r", "D")

        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            mock_extract.side_effect = [[t1], [t2]]
            triples, stats = extract_triples_multi_model(
                text="test",
                models=["m1", "m2"],
                strategy="union",
            )
        assert len(triples) == 2

    def test_intersect_with_no_overlap_returns_empty(self):
        """intersect 完全不相交：返回空"""
        t1 = _make_triple("A", "r", "B")
        t2 = _make_triple("C", "r", "D")

        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            mock_extract.side_effect = [[t1], [t2]]
            triples, stats = extract_triples_multi_model(
                text="test",
                models=["m1", "m2"],
                strategy="intersect",
            )
        assert triples == []

    def test_three_models_intersect(self):
        """三个模型的 intersect：必须三个都抽到"""
        t1 = _make_triple("A", "r", "B")  # 三模型共有
        t2 = _make_triple("C", "r", "D")  # 两模型共有
        t3 = _make_triple("E", "r", "F")  # 一模型独有

        with patch(
            "PocketGraphRAG.kg_extractor.extract_triples_from_text"
        ) as mock_extract:
            mock_extract.side_effect = [
                [t1, t2, t3],
                [t1, t2],
                [t1],
            ]
            triples, stats = extract_triples_multi_model(
                text="test",
                models=["m1", "m2", "m3"],
                strategy="intersect",
            )
        # 只有 t1 在三个模型中都出现
        assert len(triples) == 1
        assert triples[0].key() == t1.key()
