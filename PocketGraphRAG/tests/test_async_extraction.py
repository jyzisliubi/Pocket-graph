"""异步并发抽取单元测试

测试 extract_triples_from_text_async 和 extract_knowledge_graph_async。
使用 mock acall_llm，不依赖真实 API。
不依赖 pytest-asyncio，用 asyncio.run 手动驱动。
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG.kg_extractor import (
    ExtractionResult,
    Triple,
    extract_knowledge_graph_async,
    extract_triples_from_text_async,
)


def _make_response(triples_data):
    return json.dumps({"triples": triples_data}, ensure_ascii=False)


def _run(coro):
    """同步运行异步函数（兼容已有 event loop 的场景）"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已有运行中的 loop，用 ensure_future + run_until_complete 不可行
            # 这种情况下测试环境异常，直接新建 loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    return loop.run_until_complete(coro)


# ==========================
# extract_triples_from_text_async
# ==========================


class TestExtractTriplesAsync:
    def test_basic_async_extraction(self):
        """基本异步抽取"""
        response = _make_response(
            [
                {
                    "head": "稻瘟病",
                    "relation": "致病菌",
                    "tail": "稻瘟病菌",
                    "confidence": 0.95,
                }
            ]
        )

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.acall_llm", new=AsyncMock(return_value=response)),
        ):
            triples = _run(extract_triples_from_text_async("测试文本"))

        assert len(triples) == 1
        assert triples[0].head == "稻瘟病"

    def test_no_llm_returns_empty(self):
        """未配置 LLM 时返回空"""
        with patch("PocketGraphRAG.llm.has_llm", return_value=False):
            triples = _run(extract_triples_from_text_async("测试文本"))
        assert triples == []

    def test_empty_llm_response(self):
        """LLM 返回空时返回空列表"""
        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.acall_llm", new=AsyncMock(return_value=None)),
        ):
            triples = _run(extract_triples_from_text_async("测试文本"))
        assert triples == []

    def test_gleaning_async(self):
        """异步 gleaning 循环"""
        first = _make_response(
            [{"head": "A", "relation": "r1", "tail": "B", "confidence": 0.9}]
        )
        glean = _make_response(
            [{"head": "C", "relation": "r2", "tail": "D", "confidence": 0.85}]
        )

        responses = [first, glean]
        call_idx = [0]

        async def mock_acall(*args, **kwargs):
            if call_idx[0] < len(responses):
                r = responses[call_idx[0]]
                call_idx[0] += 1
                return r
            return ""

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.acall_llm", new=mock_acall),
        ):
            triples = _run(
                extract_triples_from_text_async("测试文本", gleaning_steps=1)
            )

        assert len(triples) == 2
        assert call_idx[0] == 2  # 首轮 + 1 轮 gleaning


# ==========================
# extract_knowledge_graph_async
# ==========================


class TestExtractKnowledgeGraphAsync:
    def test_concurrency_parallel_extraction(self):
        """并发抽取多个 chunk"""
        # 用足够长的文本确保切成多个 chunk
        para = "这是水稻病害防治的详细描述。" * 50
        text = "\n\n".join([para, para, para])  # 3 个长段落

        call_count = [0]

        async def mock_acall(*args, **kwargs):
            call_count[0] += 1
            return _make_response(
                [
                    {
                        "head": f"实体{call_count[0]}",
                        "relation": "r",
                        "tail": "B",
                        "confidence": 0.9,
                    }
                ]
            )

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.acall_llm", new=mock_acall),
        ):
            result = _run(
                extract_knowledge_graph_async(text, verbose=False, concurrency=3)
            )

        assert isinstance(result, ExtractionResult)
        # 应该切成 >=3 个 chunk，每个返回 1 条三元组
        assert call_count[0] >= 3
        assert len(result.triples) >= 3

    def test_concurrency_1_serial(self):
        """concurrency=1 等价于串行"""
        response = _make_response(
            [{"head": "A", "relation": "r", "tail": "B", "confidence": 0.9}]
        )

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.acall_llm", new=AsyncMock(return_value=response)),
        ):
            result = _run(
                extract_knowledge_graph_async(
                    "测试文本。", verbose=False, concurrency=1
                )
            )

        assert len(result.triples) >= 1

    def test_empty_text_returns_empty(self):
        """空文本返回空结果"""
        with patch("PocketGraphRAG.llm.has_llm", return_value=True):
            result = _run(extract_knowledge_graph_async("", verbose=False))

        assert isinstance(result, ExtractionResult)
        assert len(result.triples) == 0

    def test_chunk_failure_continues(self):
        """单个 chunk 失败时其他 chunk 继续"""
        # 用长文本确保切成多个 chunk
        para = "这是水稻病害防治的详细描述。" * 50
        text = "\n\n".join([para, para])  # 2 个长段落

        call_count = [0]

        async def mock_acall(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("模拟 API 错误")
            return _make_response(
                [{"head": "C", "relation": "r", "tail": "D", "confidence": 0.9}]
            )

        with (
            patch("PocketGraphRAG.llm.has_llm", return_value=True),
            patch("PocketGraphRAG.llm.acall_llm", new=mock_acall),
        ):
            result = _run(
                extract_knowledge_graph_async(text, verbose=False, concurrency=2)
            )

        # 至少有 1 个 chunk 成功（失败的被跳过）
        assert call_count[0] >= 2
        assert len(result.triples) >= 1
        # 成功的 chunk 返回了 "C" 实体
        heads = {t.head for t in result.triples}
        assert "C" in heads


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
