"""4 种分块策略单元测试（对标 LightRAG 2026.05）

验证 fixed/recursive/paragraph/semantic 四种策略的切分行为和统一入口。
"""

import pytest

from PocketGraphRAG.kg_extractor import (
    chunk_with_strategy,
    fixed_chunk_text,
    recursive_chunk_text,
    paragraph_chunk_text,
    semantic_chunk_text,
)


# ==========================
# 测试文本
# ==========================

SAMPLE_SHORT = "这是一个短文本。"

SAMPLE_PARAGRAPHS = """第一段落。这是第一段的内容，包含多个句子。第二句话在这里。

第二段落。这是第二段的内容。它也有多个句子。最后一句。

第三段落。独立成段。"""

SAMPLE_LONG = "这是一个很长的句子。" * 200  # ~1800 字符


# ==========================
# 1. 统一入口 chunk_with_strategy
# ==========================


class TestChunkWithStrategy:
    """统一分块入口测试"""

    def test_empty_text_returns_empty(self):
        assert chunk_with_strategy("", "semantic") == []
        assert chunk_with_strategy("   ", "semantic") == []
        assert chunk_with_strategy(None, "semantic") == []  # type: ignore

    def test_unknown_strategy_falls_back_to_semantic(self):
        chunks = chunk_with_strategy(SAMPLE_SHORT, "unknown_strategy")
        assert len(chunks) == 1
        assert "短文本" in chunks[0]

    def test_case_insensitive(self):
        """策略名大小写不敏感"""
        c1 = chunk_with_strategy(SAMPLE_SHORT, "SEMANTIC")
        c2 = chunk_with_strategy(SAMPLE_SHORT, "Semantic")
        c3 = chunk_with_strategy(SAMPLE_SHORT, "semantic")
        assert c1 == c2 == c3

    def test_all_strategies_return_non_empty_for_valid_text(self):
        for strategy in ["fixed", "recursive", "paragraph", "semantic"]:
            chunks = chunk_with_strategy(SAMPLE_PARAGRAPHS, strategy, chunk_size=500)
            assert len(chunks) > 0, f"策略 {strategy} 返回空列表"


# ==========================
# 2. fixed_chunk_text
# ==========================


class TestFixedChunk:
    """固定大小切分测试"""

    def test_short_text_single_chunk(self):
        chunks = fixed_chunk_text(SAMPLE_SHORT, chunk_size=1200)
        assert len(chunks) == 1
        assert chunks[0] == SAMPLE_SHORT

    def test_long_text_multiple_chunks(self):
        chunks = fixed_chunk_text(SAMPLE_LONG, chunk_size=500)
        assert len(chunks) > 1
        # 每块不应超过 chunk_size 太多（句子边界对齐可能略超）
        for c in chunks:
            assert len(c) <= 700  # 允许句子边界对齐的少量超出

    def test_empty_text(self):
        assert fixed_chunk_text("", chunk_size=500) == []
        assert fixed_chunk_text("   ", chunk_size=500) == []

    def test_sentence_boundary_alignment(self):
        """长文本应在句子边界对齐"""
        text = "第一句话。第二句话。第三句话。第四句话。" * 50
        chunks = fixed_chunk_text(text, chunk_size=100)
        # 不应从句子中间切断（应在 。 处对齐）
        for c in chunks[:-1]:  # 最后一块可能以句子结尾
            # 至少应在块尾附近找到句号
            assert "。" in c

    def test_overlap(self):
        """overlap 参数应让相邻块有重叠"""
        text = "句子一。句子二。句子三。句子四。句子五。" * 20
        chunks_no_overlap = fixed_chunk_text(text, chunk_size=100, overlap=0)
        chunks_overlap = fixed_chunk_text(text, chunk_size=100, overlap=20)
        # 有 overlap 时块数应 >= 无 overlap
        assert len(chunks_overlap) >= len(chunks_no_overlap)


# ==========================
# 3. recursive_chunk_text
# ==========================


class TestRecursiveChunk:
    """递归字符切分测试"""

    def test_short_text_single_chunk(self):
        chunks = recursive_chunk_text(SAMPLE_SHORT, chunk_size=1200)
        assert len(chunks) == 1

    def test_paragraph_split(self):
        """多段落文本应优先按 \\n\\n 切分"""
        chunks = recursive_chunk_text(SAMPLE_PARAGRAPHS, chunk_size=50, min_chunk_size=10)
        assert len(chunks) >= 2  # 至少切分

    def test_long_text_multiple_chunks(self):
        chunks = recursive_chunk_text(SAMPLE_LONG, chunk_size=500, min_chunk_size=100)
        assert len(chunks) > 1

    def test_empty_text(self):
        assert recursive_chunk_text("", chunk_size=500) == []
        assert recursive_chunk_text("   ", chunk_size=500) == []

    def test_preserves_content(self):
        """切分后所有块拼接应包含原文所有非空白内容"""
        text = "段落一内容。段落二内容。段落三内容。"
        chunks = recursive_chunk_text(text, chunk_size=20, min_chunk_size=5)
        joined = "".join(chunks)
        # 去除分隔符后应包含所有原文内容
        for keyword in ["段落一", "段落二", "段落三"]:
            assert keyword in joined

    def test_min_chunk_size_merging(self):
        """小于 min_chunk_size 的块应被合并"""
        text = "短。短。短。短。" + "长内容" * 100
        chunks = recursive_chunk_text(text, chunk_size=100, min_chunk_size=30)
        # 不应有太多极小块
        tiny = [c for c in chunks if len(c) < 10]
        assert len(tiny) <= 1  # 最多最后一块可能是小块


# ==========================
# 4. paragraph_chunk_text
# ==========================


class TestParagraphChunk:
    """按段落切分测试"""

    def test_each_paragraph_one_chunk(self):
        """每个段落应独立成一个 chunk"""
        chunks = paragraph_chunk_text(SAMPLE_PARAGRAPHS, max_chunk_size=500)
        assert len(chunks) == 3  # 三个段落
        assert "第一段落" in chunks[0]
        assert "第二段落" in chunks[1]
        assert "第三段落" in chunks[2]

    def test_empty_text(self):
        assert paragraph_chunk_text("", max_chunk_size=500) == []
        assert paragraph_chunk_text("   ", max_chunk_size=500) == []

    def test_short_text_single_chunk(self):
        chunks = paragraph_chunk_text(SAMPLE_SHORT, max_chunk_size=500)
        assert len(chunks) == 1

    def test_long_paragraph_falls_back_to_sentences(self):
        """超长段落应回退到句子切分"""
        long_para = "这是第一句话。" * 100  # 单段落超长
        chunks = paragraph_chunk_text(long_para, max_chunk_size=100)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 200  # 不应超过 max_chunk_size 太多

    def test_no_paragraph_merge(self):
        """paragraph 策略不应合并相邻段落（与 semantic 不同）"""
        text = "短段一。\n\n短段二。\n\n短段三。"
        chunks = paragraph_chunk_text(text, max_chunk_size=500)
        assert len(chunks) == 3  # 三个独立段落，不合并


# ==========================
# 5. semantic_chunk_text（已有实现，验证回归）
# ==========================


class TestSemanticChunk:
    """语义切分回归测试"""

    def test_short_text_single_chunk(self):
        chunks = semantic_chunk_text(SAMPLE_SHORT, max_chunk_size=1200)
        assert len(chunks) == 1

    def test_empty_text(self):
        assert semantic_chunk_text("", max_chunk_size=500) == []
        assert semantic_chunk_text("   ", max_chunk_size=500) == []

    def test_merges_tiny_last_chunk(self):
        """最后一块太小时应尝试合并到前一块（当容量允许时）"""
        text = "长内容" * 50 + "\n\n短"  # 中等长度 + 极短段
        chunks = semantic_chunk_text(text, max_chunk_size=500, min_chunk_size=50)
        # 应至少返回 1 个块
        assert len(chunks) >= 1
        # 所有块拼接应包含原文内容
        joined = "".join(chunks)
        assert "长内容" in joined
        assert "短" in joined

    def test_long_paragraph_splits_by_sentence(self):
        """超长段落应按句子切分"""
        long_para = "这是第一句话。" * 100
        chunks = semantic_chunk_text(long_para, max_chunk_size=100, min_chunk_size=20)
        assert len(chunks) > 1


# ==========================
# 6. 策略对比测试
# ==========================


class TestStrategyComparison:
    """4 种策略行为对比"""

    def test_all_strategies_handle_same_text(self):
        """所有策略都应能处理相同文本不报错"""
        text = SAMPLE_PARAGRAPHS + "\n\n" + SAMPLE_LONG
        for strategy in ["fixed", "recursive", "paragraph", "semantic"]:
            chunks = chunk_with_strategy(text, strategy, chunk_size=500, min_chunk_size=100)
            assert len(chunks) > 0, f"策略 {strategy} 失败"
            # 所有块拼接应包含原文关键内容
            joined = "".join(chunks)
            assert "段落" in joined or "句子" in joined

    def test_paragraph_yields_more_chunks_for_structured_text(self):
        """结构化文本（多段落）paragraph 策略应产生更多块"""
        multi_para = "\n\n".join([f"段落{i}。" for i in range(10)])
        p_chunks = chunk_with_strategy(multi_para, "paragraph", chunk_size=500)
        s_chunks = chunk_with_strategy(multi_para, "semantic", chunk_size=500, min_chunk_size=50)
        # paragraph 不合并，应产生 >= semantic 的块数
        assert len(p_chunks) >= len(s_chunks)

    def test_fixed_respects_chunk_size_most_strictly(self):
        """fixed 策略应最严格遵守 chunk_size"""
        text = "句子。" * 300
        f_chunks = fixed_chunk_text(text, chunk_size=200)
        # fixed 块大小应相对均匀（标准差小）
        f_sizes = [len(c) for c in f_chunks]
        assert max(f_sizes) <= 250  # 句子边界对齐后略超
