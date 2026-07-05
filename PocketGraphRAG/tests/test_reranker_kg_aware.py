"""KG-aware Reranker 单元测试"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG.core.reranker import (
    KG_AWARE_ENABLED,
    KG_AWARE_MAX_ENTITIES,
    Reranker,
    _build_kg_aware_passage_for_tuple,
    get_reranker,
)


class TestBuildKgAwarePassage:
    """_build_kg_aware_passage_for_tuple 测试"""

    def test_no_entities_returns_original(self):
        """无实体时返回原始文本"""
        text = "这是原始文本"
        meta = {"entity": ""}
        result = _build_kg_aware_passage_for_tuple(text, meta)
        assert result == text

    def test_with_single_entity(self):
        """单个实体注入"""
        text = "盗梦空间的导演是诺兰"
        meta = {"entity": "盗梦空间"}
        result = _build_kg_aware_passage_for_tuple(text, meta)
        assert "盗梦空间" in result
        assert "[关联实体]" in result
        assert result.startswith(text)

    def test_with_multiple_entities_list(self):
        """多个实体（list 格式）注入"""
        text = "测试文本"
        meta = {"entities": ["盗梦空间", "诺兰", "莱昂纳多"]}
        result = _build_kg_aware_passage_for_tuple(text, meta)
        assert "盗梦空间" in result
        assert "诺兰" in result
        assert "莱昂纳多" in result

    def test_max_entities_limit(self):
        """实体数量超过上限应截断"""
        text = "测试"
        meta = {"entities": [f"实体{i}" for i in range(20)]}
        result = _build_kg_aware_passage_for_tuple(text, meta)
        # 默认上限 5
        for i in range(KG_AWARE_MAX_ENTITIES):
            assert f"实体{i}" in result
        # 第 6 个不应该在
        assert f"实体{KG_AWARE_MAX_ENTITIES}" not in result

    def test_no_meta_dict(self):
        """meta 不是 dict 时返回原始文本"""
        text = "测试"
        result = _build_kg_aware_passage_for_tuple(text, None)
        assert result == text
        result = _build_kg_aware_passage_for_tuple(text, "not dict")
        assert result == text

    def test_kg_aware_disabled(self):
        """KG_AWARE_ENABLED=False 时返回原始文本"""
        with patch("PocketGraphRAG.core.reranker.KG_AWARE_ENABLED", False):
            text = "测试"
            meta = {"entity": "盗梦空间"}
            result = _build_kg_aware_passage_for_tuple(text, meta)
            assert result == text

    def test_entities_list_with_empty_items(self):
        """entities list 含空值应过滤"""
        text = "测试"
        meta = {"entities": ["", "盗梦空间", None, "诺兰"]}
        result = _build_kg_aware_passage_for_tuple(text, meta)
        assert "盗梦空间" in result
        assert "诺兰" in result


class TestRerankerKgAware:
    """Reranker 类的 KG-aware 行为测试"""

    def test_rerank_with_kg_aware(self):
        """rerank 应调用 KG-aware passage 构建"""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3]

        reranker = get_reranker()
        reranker._model = mock_model
        reranker._load_failed = False

        chunks = [
            {"text": "盗梦空间是诺兰的电影", "meta": {"entity": "盗梦空间"}},
            {"text": "其他无关文本", "meta": {}},
        ]
        result = reranker.rerank("盗梦空间", chunks, top_k=2)

        # 验证 mock_model.predict 被调用，且 pairs 中包含 KG-aware passage
        assert mock_model.predict.called
        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs) == 2
        # 第一个 chunk 应包含关联实体
        assert "盗梦空间" in pairs[0][1]
        assert "[关联实体]" in pairs[0][1]

        # 结果按 score 降序
        assert result[0]["rerank_score"] == 0.9

    def test_rerank_empty_chunks(self):
        """空 chunks 应返回空列表"""
        reranker = get_reranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_model_unavailable(self):
        """模型不可用时应返回原 chunks 前 top_k 个"""
        reranker = get_reranker()
        reranker._model = None
        reranker._load_failed = True

        chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        result = reranker.rerank("query", chunks, top_k=2)
        assert len(result) == 2
