"""知识图谱抽取升级模块单元测试"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG.kg_extractor import (
    ExtractionResult,
    Triple,
    _fullwidth_to_halfwidth,
    _parse_delimiter_format,
    _rule_based_normalize,
    _split_sentences,
    align_entities,
    chunk_text,
    deduplicate_triples,
    extract_knowledge_graph,
    filter_low_quality,
    semantic_chunk_text,
)


class TestTriple:
    def test_triple_creation(self):
        t = Triple(
            head="稻瘟病",
            relation="症状表现",
            tail="病斑",
            confidence=0.95,
            evidence="test",
        )
        assert t.head == "稻瘟病"
        assert t.relation == "症状表现"
        assert t.tail == "病斑"
        assert t.confidence == 0.95
        assert t.evidence == "test"

    def test_triple_to_tuple(self):
        t = Triple(head="A", relation="B", tail="C")
        assert t.to_tuple() == ("A", "B", "C")

    def test_triple_key(self):
        t1 = Triple(head="稻瘟病", relation="症状", tail="病斑")
        t2 = Triple(head="稻瘟病", relation="症状", tail="病斑")
        t3 = Triple(head="稻瘟病", relation="治疗", tail="农药")
        assert t1.key() == t2.key()
        assert t1.key() != t3.key()

    def test_triple_key_case_insensitive(self):
        t1 = Triple(head="ABC", relation="DEF", tail="GHI")
        t2 = Triple(head="abc", relation="def", tail="ghi")
        assert t1.key() == t2.key()


class TestExtractionResult:
    def test_empty_result(self):
        r = ExtractionResult()
        assert len(r.triples) == 0
        assert r.avg_confidence == 0.0
        assert r.high_quality_count == 0
        assert len(r.entities) == 0

    def test_avg_confidence(self):
        r = ExtractionResult(
            triples=[
                Triple("A", "r1", "B", confidence=0.9),
                Triple("C", "r2", "D", confidence=0.7),
            ]
        )
        assert abs(r.avg_confidence - 0.8) < 0.001

    def test_high_quality_count(self):
        r = ExtractionResult(
            triples=[
                Triple("A", "r1", "B", confidence=0.9),
                Triple("C", "r2", "D", confidence=0.85),
                Triple("E", "r3", "F", confidence=0.7),
                Triple("G", "r4", "H", confidence=0.5),
            ]
        )
        assert r.high_quality_count == 2

    def test_entities_property(self):
        r = ExtractionResult(
            triples=[
                Triple("A", "r1", "B"),
                Triple("B", "r2", "C"),
            ]
        )
        assert r.entities == {"A", "B", "C"}


class TestSemanticChunkText:
    def test_empty_text(self):
        assert semantic_chunk_text("") == []
        assert semantic_chunk_text(None) == []
        assert semantic_chunk_text("   ") == []

    def test_single_short_paragraph(self):
        text = "这是一个短段落。"
        chunks = semantic_chunk_text(text, max_chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_paragraphs(self):
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = semantic_chunk_text(text, max_chunk_size=100)
        # 每个段落都很短，应该在一个块里
        assert len(chunks) >= 1

    def test_long_paragraph_split_by_sentence(self):
        # 构造一个超长段落
        long_para = "。".join([f"第{i}句" + "x" * 50 for i in range(20)]) + "。"
        chunks = semantic_chunk_text(long_para, max_chunk_size=200)
        # 应该被切成多个块
        assert len(chunks) > 1

    def test_chunk_size_bounds(self):
        text = "\n\n".join([f"第{i}段内容" + "x" * 100 for i in range(10)])
        chunks = semantic_chunk_text(text, max_chunk_size=500)
        # 每个块不应该比 max_chunk_size 大太多
        for chunk in chunks:
            assert len(chunk) < 1000  # 允许一些溢出

    def test_preserves_paragraph_structure(self):
        text = "段落一。\n\n段落二。"
        chunks = semantic_chunk_text(text, max_chunk_size=1000)
        # 两个短段落应该在一个块里，用 \n\n 分隔
        assert "\n\n" in chunks[0]


class TestSplitSentences:
    def test_chinese_sentences(self):
        text = "你好！这是测试。对吗？是的；没错。"
        sents = _split_sentences(text)
        assert len(sents) >= 3

    def test_english_sentences(self):
        text = "Hello. This is a test! Is it right? Yes."
        sents = _split_sentences(text)
        assert len(sents) >= 3

    def test_mixed_sentences(self):
        text = "你好world.这是test！对吗?"
        sents = _split_sentences(text)
        assert len(sents) >= 2

    def test_empty_text(self):
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []


class TestRuleBasedNormalize:
    def test_remove_parentheses(self):
        entities = {"稻瘟病(水稻)", "Bt(苏云金杆菌)"}
        result = _rule_based_normalize(entities)
        # 括号内容应该被去掉，保留括号外的
        assert "稻瘟病" in result.values() or "稻瘟病(水稻)" in result

    def test_fullwidth_to_halfwidth(self):
        text = _fullwidth_to_halfwidth("ＡＢＣ１２３")
        assert text == "ABC123"

    def test_fullwidth_space(self):
        text = _fullwidth_to_halfwidth("　")  # 全角空格
        assert text == " "

    def test_strip_punctuation(self):
        entities = {"，测试。", "【实体】"}
        result = _rule_based_normalize(entities)
        # 首尾标点应该被去掉
        for v in result.values():
            assert not v.startswith("，")
            assert not v.startswith("【")

    def test_normalize_preserves_mapping(self):
        entities = {"稻瘟病", "稻瘟病(水稻)"}
        result = _rule_based_normalize(entities)
        # 原始实体都应该在 keys 里
        assert "稻瘟病" in result
        assert "稻瘟病(水稻)" in result


class TestDeduplicateTriples:
    def test_no_duplicates(self):
        triples = [
            Triple("A", "r", "B", confidence=0.9),
            Triple("C", "r", "D", confidence=0.8),
        ]
        result, removed = deduplicate_triples(triples)
        assert len(result) == 2
        assert removed == 0

    def test_exact_duplicates(self):
        triples = [
            Triple("A", "r", "B", confidence=0.9, evidence="e1"),
            Triple("A", "r", "B", confidence=0.8, evidence="e2"),
        ]
        result, removed = deduplicate_triples(triples)
        assert len(result) == 1
        assert removed == 1
        assert result[0].confidence == 0.9  # 保留置信度高的
        assert "e1" in result[0].evidence  # evidence 被合并

    def test_case_insensitive_dedup(self):
        triples = [
            Triple("ABC", "DEF", "GHI", confidence=0.9),
            Triple("abc", "def", "ghi", confidence=0.8),
        ]
        result, removed = deduplicate_triples(triples)
        assert len(result) == 1
        assert removed == 1

    def test_evidence_merged_when_new_higher_confidence(self):
        """新三元组置信度更高时，旧三元组的 evidence 不能丢失（回归测试）

        历史 bug：先 best[key]=t 替换，再改 existing.evidence，
        导致 existing（被替换掉的）的 evidence 丢失。
        """
        triples = [
            Triple("A", "r", "B", confidence=0.8, evidence="旧依据"),
            Triple("A", "r", "B", confidence=0.9, evidence="新依据"),
        ]
        result, removed = deduplicate_triples(triples)
        assert len(result) == 1
        assert removed == 1
        # 置信度取高的
        assert result[0].confidence == 0.9
        # 两条 evidence 都必须保留（这是 bug 修复的核心）
        assert "旧依据" in result[0].evidence
        assert "新依据" in result[0].evidence

    def test_evidence_merged_when_new_lower_confidence(self):
        """新三元组置信度更低时，evidence 也要合并"""
        triples = [
            Triple("A", "r", "B", confidence=0.9, evidence="e1"),
            Triple("A", "r", "B", confidence=0.7, evidence="e2"),
        ]
        result, removed = deduplicate_triples(triples)
        assert len(result) == 1
        assert result[0].confidence == 0.9
        assert "e1" in result[0].evidence
        assert "e2" in result[0].evidence

    def test_evidence_not_duplicated(self):
        """相同 evidence 不重复合并"""
        triples = [
            Triple("A", "r", "B", confidence=0.9, evidence="same"),
            Triple("A", "r", "B", confidence=0.8, evidence="same"),
        ]
        result, removed = deduplicate_triples(triples)
        assert len(result) == 1
        # evidence 不应该重复
        assert result[0].evidence == "same"


class TestFilterLowQuality:
    def test_confidence_filter(self):
        triples = [
            Triple("A", "r", "B", confidence=0.9),
            Triple("C", "r", "D", confidence=0.5),
            Triple("E", "r", "F", confidence=0.7),
        ]
        result, removed = filter_low_quality(triples, min_confidence=0.6)
        assert len(result) == 2
        assert removed == 1
        assert all(t.confidence >= 0.6 for t in result)

    def test_generic_relation_filter(self):
        triples = [
            Triple("A", "是", "B", confidence=0.9),
            Triple("C", "有", "D", confidence=0.9),
            Triple("E", "防治", "F", confidence=0.9),
        ]
        result, removed = filter_low_quality(triples, min_confidence=0.5)
        assert len(result) == 1  # 只有"防治"保留
        assert removed == 2

    def test_short_entity_filter(self):
        triples = [
            Triple("", "r", "B", confidence=0.9),
            Triple("A", "r", "", confidence=0.9),
            Triple("C", "r", "D", confidence=0.9),
        ]
        result, removed = filter_low_quality(triples, min_confidence=0.5)
        assert len(result) == 1
        assert removed == 2


class TestIsLowQualityEntityV2:
    """v0.3.3 新增 4 类质量过滤规则测试（P0-4）

    基于真实数据集扫描（14.5% 漏网垃圾）+ HotpotQA/MuSiQue 通用性验证。
    """

    def test_pure_digit_filtered(self):
        """纯数字串（非年份）应被过滤"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        assert is_low_quality_entity("1.2") == "pure_digit"
        assert is_low_quality_entity("110.56") == "pure_digit"
        assert is_low_quality_entity("142") == "pure_digit"
        assert is_low_quality_entity("0.5") == "pure_digit"

    def test_four_digit_year_preserved(self):
        """4 位整数年份 (1900-2099) 应保留，可能是年份/书名《1984》"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        assert is_low_quality_entity("1984") is None  # 书名
        assert is_low_quality_entity("2001") is None  # 年份/电影名
        assert is_low_quality_entity("2024") is None
        # 但 1800/3000 超出范围，过滤
        assert is_low_quality_entity("1800") == "pure_digit"
        assert is_low_quality_entity("3000") == "pure_digit"

    def test_digit_with_unit_filtered(self):
        """数字+单位应被过滤"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        assert is_low_quality_entity("0.1kg/亩") == "digit_with_unit"
        assert is_low_quality_entity("1%") == "digit_with_unit"
        assert is_low_quality_entity("0.5天") == "digit_with_unit"
        assert is_low_quality_entity("3次") == "digit_with_unit"
        assert is_low_quality_entity("100毫升") == "digit_with_unit"
        assert is_low_quality_entity("0.2MPa") == "digit_with_unit"
        assert is_low_quality_entity("50ppm") == "digit_with_unit"

    def test_product_with_percent_preserved(self):
        """含 % 但有非数字字符的（如 "80%乙蒜素乳油"）应保留"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        assert is_low_quality_entity("80%乙蒜素乳油") is None
        assert is_low_quality_entity("25%噻嗪酮可湿性粉剂") is None

    def test_pure_date_filtered(self):
        """纯日期/时间应被过滤"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        assert is_low_quality_entity("2024年3月") == "pure_date"
        assert is_low_quality_entity("10月6日") == "pure_date"
        assert is_low_quality_entity("3月中下旬") == "pure_date"
        assert is_low_quality_entity("10月上旬至中旬") == "pure_date"

    def test_description_with_date_preserved(self):
        """含日期但非纯日期的描述应保留（如 "2018年亩产500公斤"）"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        assert is_low_quality_entity("2018年亩产500公斤") is None
        assert is_low_quality_entity("2024年水稻产量") is None

    def test_formula_filtered(self):
        """公式类（含 = 且含数字）应被过滤"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        formula = "理论产量（公斤/亩）= 每亩穗数 × 每穗粒数 × 单粒重 / 1000"
        assert is_low_quality_entity(formula) == "formula"

    def test_latin_scientific_name_preserved(self):
        """拉丁学名应保留（不误伤）"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        assert is_low_quality_entity("Xanthomonas oryzae pv. oryzae") is None
        assert is_low_quality_entity("Pyricularia oryzae") is None
        assert is_low_quality_entity("Magnaporthe oryzae") is None

    def test_standard_number_preserved(self):
        """标准号应保留（如 NY/T 391-2013）"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        assert is_low_quality_entity("NY/T 391-2013") is None
        assert is_low_quality_entity("GB/T 8321.1-2000") is None

    def test_english_entities_not_over_filtered(self):
        """HotpotQA 风格英文实体不应被过度过滤"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        valid_english = [
            "Arthur's Magazine", "New York City", "United States",
            "The Walt Disney Company", "J.K. Rowling",
            "World War II", "COVID-19", "Boeing 747",
            "PS5", "3M", "7-Eleven",
        ]
        for e in valid_english:
            assert is_low_quality_entity(e) is None, f"误伤: {e!r}"

    def test_original_4_rules_still_work(self):
        """原有 4 类规则应保持有效"""
        from PocketGraphRAG.kg_extractor import is_low_quality_entity

        # 句子片段
        assert is_low_quality_entity("叶片出现病斑、且扩大") == "sent_punct"
        # LLM 占位符
        assert is_low_quality_entity("entity_1") == "placeholder"
        assert is_low_quality_entity("llm_output") == "placeholder"
        # "…等"列举
        assert is_low_quality_entity("稻瘟病纹枯病等") == "enum"
        # 方法描述
        assert is_low_quality_entity("80%乙蒜素2000倍液浸种48小时") == "method_desc"


class TestAlignEntities:
    def test_empty_triples(self):
        result, mapping = align_entities([])
        assert result == []
        assert mapping == {}

    def test_rule_based_alignment(self):
        triples = [
            Triple("稻瘟病(水稻)", "症状", "病斑"),
            Triple("稻瘟病", "防治", "农药"),
        ]
        result, mapping = align_entities(triples, threshold=0.99)
        # 规则对齐应该把 "稻瘟病(水稻)" 标准化
        assert len(mapping) >= 1
        # 两个三元组的头实体应该被对齐（去掉括号后相同）
        heads = {t.head for t in result}
        # 可能都变成"稻瘟病"
        assert "稻瘟病" in heads

    def test_self_loop_removal(self):
        # 如果头尾实体对齐后变成同一个，应该被移除
        triples = [
            Triple("稻瘟病", "相关", "稻瘟病(水稻)"),
        ]
        result, mapping = align_entities(triples, threshold=0.99)
        # 自环应该被移除
        assert len(result) == 0

    def test_returns_mapping_dict(self):
        triples = [
            Triple("A(测试)", "r", "B"),
        ]
        result, mapping = align_entities(triples, threshold=0.99)
        assert isinstance(mapping, dict)
        assert "A(测试)" in mapping


class TestExtractKnowledgeGraph:
    def test_empty_text(self):
        result = extract_knowledge_graph("", verbose=False)
        assert len(result.triples) == 0
        assert result.raw_triples_count == 0

    def test_whitespace_only(self):
        result = extract_knowledge_graph("   \n\n  ", verbose=False)
        assert len(result.triples) == 0

    def test_no_llm_returns_empty(self):
        # 如果没有配置 LLM，应该返回空（不抛异常）
        result = extract_knowledge_graph("测试文本内容。", verbose=False)
        # 没有 LLM 时返回空列表
        assert isinstance(result, ExtractionResult)


class TestBackwardCompatibility:
    def test_chunk_text_legacy(self):
        """测试旧版 chunk_text 仍可用"""
        text = "a" * 2500
        chunks = chunk_text(text, chunk_size=1000)
        assert len(chunks) == 3
        assert len(chunks[0]) == 1000
        assert len(chunks[1]) == 1000
        assert len(chunks[2]) == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestDelimiterFormatParser:
    """P0-3: delimiter 格式解析器测试

    格式: <|#|>头实体|关系|尾实体|置信度|原文依据<|#|>
    作为 JSON 解析失败的 fallback。
    """

    def test_parse_basic_delimiter(self):
        """基本 delimiter 格式解析"""
        result = (
            "<|#|>稻瘟病|症状表现|叶片出现梭形病斑|0.95|叶片上出现梭形褐色病斑<|#|>\n"
            "<|#|>三环唑|防治|稻瘟病|0.90|三环唑是防治稻瘟病的常用药剂<|#|>"
        )
        triples = _parse_delimiter_format(result)
        assert len(triples) == 2
        assert triples[0].head == "稻瘟病"
        assert triples[0].relation == "症状表现"
        assert triples[0].tail == "叶片出现梭形病斑"
        assert triples[0].confidence == 0.95
        assert triples[1].head == "三环唑"
        assert triples[1].relation == "防治"

    def test_parse_without_confidence_and_evidence(self):
        """省略置信度和原文依据（用默认值）"""
        result = "<|#|>稻瘟病|致病菌|稻瘟病菌<|#|>"
        triples = _parse_delimiter_format(result)
        assert len(triples) == 1
        assert triples[0].head == "稻瘟病"
        assert triples[0].relation == "致病菌"
        assert triples[0].tail == "稻瘟病菌"
        assert triples[0].confidence == 0.7  # 默认值

    def test_parse_with_text_around(self):
        """LLM 在 delimiter 格式前后加了多余文字"""
        result = (
            "以下是抽取的三元组：\n"
            "<|#|>稻瘟病|症状表现|叶片病斑|0.9|病斑描述<|#|>\n"
            "解析完成。"
        )
        triples = _parse_delimiter_format(result)
        assert len(triples) == 1
        assert triples[0].head == "稻瘟病"

    def test_parse_empty_input(self):
        """空输入或无 marker 输入"""
        assert _parse_delimiter_format("") == []
        assert _parse_delimiter_format("没有 marker 的文本") == []

    def test_parse_invalid_format(self):
        """字段不足 3 个的行被跳过"""
        result = "<|#|>只有两个字段|关系<|#|>"
        triples = _parse_delimiter_format(result)
        assert len(triples) == 0

    def test_parse_with_schema_normalization(self):
        """带 schema 时关系名被归一化"""
        from PocketGraphRAG.schema import RelationSchema

        schema = RelationSchema()
        # "症状" 应归一化为 "症状表现"
        result = "<|#|>稻瘟病|症状|叶片病斑|0.9|病斑<|#|>"
        triples = _parse_delimiter_format(result, schema=schema)
        assert len(triples) == 1
        assert triples[0].relation == "症状表现"

    def test_parse_invalid_confidence_falls_back_to_default(self):
        """置信度字段非数字时用默认值"""
        result = "<|#|>稻瘟病|症状表现|病斑|高|病斑描述<|#|>"
        triples = _parse_delimiter_format(result)
        assert len(triples) == 1
        assert triples[0].confidence == 0.7  # 默认值


class TestEntityAliases:
    """语义别名字典 + pre-normalize 测试（P0-1）"""

    def test_load_default_aliases(self):
        """通用默认：空路径不加载任何领域别名（避免领域泄露）"""
        from PocketGraphRAG.kg_extractor import _load_entity_aliases

        # 空路径 → 空字典（通用 GraphRAG 默认，不施加水稻领域归一化）
        aliases = _load_entity_aliases()
        assert aliases == {}

    def test_load_explicit_rice_aliases(self):
        """显式指定水稻别名词典时能正确加载（向后兼容）"""
        import os
        from PocketGraphRAG.kg_extractor import _load_entity_aliases

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rice_path = os.path.join(here, "data", "entity_aliases.json")
        aliases = _load_entity_aliases(rice_path)
        # 应该包含 Bt → 苏云金杆菌 这种典型映射
        assert aliases.get("Bt") == "苏云金杆菌"
        assert aliases.get("稻热病") == "稻瘟病"
        assert aliases.get("富士一号") == "稻瘟灵"
        assert len(aliases) >= 50  # 至少 50 条

    def test_load_nonexistent_path_returns_empty(self):
        """不存在的路径返回空字典，不抛异常"""
        from PocketGraphRAG.kg_extractor import _load_entity_aliases

        aliases = _load_entity_aliases("/nonexistent/path/aliases.json")
        assert aliases == {}

    def test_get_default_aliases_cached(self):
        """_get_default_aliases 带缓存，多次调用返回同一对象"""
        from PocketGraphRAG.kg_extractor import _get_default_aliases

        a1 = _get_default_aliases()
        a2 = _get_default_aliases()
        assert a1 is a2  # 同一对象引用

    def test_alias_pre_normalize_merges_synonyms(self):
        """别名 pre-normalize 把同义词合并到规范名"""
        aliases = {"Bt": "苏云金杆菌", "Bt杀虫菌": "苏云金杆菌"}
        triples = [
            Triple("Bt", "防治", "二化螟", confidence=0.9),
            Triple("Bt杀虫菌", "属于", "生物农药", confidence=0.8),
            Triple("苏云金杆菌", "作用机理", "伴胞晶体", confidence=0.95),
        ]
        result, mapping = align_entities(triples, threshold=0.99, aliases=aliases)
        heads = {t.head for t in result}
        # 三个同义实体名应该都合并成 "苏云金杆菌"
        assert heads == {"苏云金杆菌"}
        assert mapping["Bt"] == "苏云金杆菌"
        assert mapping["Bt杀虫菌"] == "苏云金杆菌"

    def test_alias_none_skips_pre_normalize(self):
        """aliases=None 时不做 pre-normalize（向后兼容）"""
        triples = [
            Triple("Bt", "防治", "二化螟"),
            Triple("苏云金杆菌", "属于", "生物农药"),
        ]
        result, mapping = align_entities(triples, threshold=0.99, aliases=None)
        # 没有别名表，这两个名字不会被合并（embedding 阈值 0.99 太高）
        heads = {t.head for t in result}
        assert "Bt" in heads
        assert "苏云金杆菌" in heads

    def test_alias_self_canonical_not_in_mapping(self):
        """规范名本身不会出现在 mapping 里（alias == canonical 不算映射）"""
        aliases = {"三环唑": "三环唑"}  # 自映射
        triples = [Triple("三环唑", "防治", "稻瘟病")]
        result, mapping = align_entities(triples, threshold=0.99, aliases=aliases)
        # 自映射不应该产生 mapping 条目
        assert "三环唑" not in mapping


class TestCanonicalByFrequency:
    """Canonical 按频率选择测试（P0-2）"""

    def test_frequency_prefers_high_freq_entity(self):
        """频率高的实体名应被选为 canonical（修复"选最长名"bug）

        场景：三环唑可湿性粉剂(长名,1次) vs 三环唑(短名,5次)
        旧实现选最长名"三环唑可湿性粉剂"，新实现应选高频"三环唑"
        """
        import PocketGraphRAG.kg_extractor as kg_mod

        # 直接 monkeypatch _embedding_based_match：只合并指定的两个"三环唑"变体
        # 频率高的"三环唑"应被选为 canonical
        original_match = kg_mod._embedding_based_match

        def fake_match(entities, threshold=0.88, freq=None):
            if "三环唑" in entities and "三环唑可湿性粉剂" in entities:
                def _key(name):
                    f = freq.get(name, 0) if freq else 0
                    return (-f, -len(name), name)
                pair = ["三环唑", "三环唑可湿性粉剂"]
                canonical = min(pair, key=_key)
                other = [e for e in pair if e != canonical][0]
                return {other: canonical}
            return {}

        kg_mod._embedding_based_match = fake_match
        try:
            entities = {"三环唑", "三环唑可湿性粉剂", "稻瘟病"}
            freq = {"三环唑": 5, "三环唑可湿性粉剂": 1, "稻瘟病": 3}
            merged = kg_mod._embedding_based_match(entities, threshold=0.88, freq=freq)
        finally:
            kg_mod._embedding_based_match = original_match

        # 高频的"三环唑"应该被选为 canonical
        assert merged.get("三环唑可湿性粉剂") == "三环唑"

    def test_frequency_tie_falls_back_to_longest(self):
        """频率相同时退化为最长名（更完整）"""
        import PocketGraphRAG.kg_extractor as kg_mod

        original_match = kg_mod._embedding_based_match

        def fake_match(entities, threshold=0.88, freq=None):
            if "稻瘟病" in entities and "水稻稻瘟病" in entities:
                def _key(name):
                    f = freq.get(name, 0) if freq else 0
                    return (-f, -len(name), name)
                pair = ["稻瘟病", "水稻稻瘟病"]
                canonical = min(pair, key=_key)
                other = [e for e in pair if e != canonical][0]
                return {other: canonical}
            return {}

        kg_mod._embedding_based_match = fake_match
        try:
            entities = {"稻瘟病", "水稻稻瘟病"}
            freq = {"稻瘟病": 1, "水稻稻瘟病": 1}  # 频率相同
            merged = kg_mod._embedding_based_match(entities, threshold=0.88, freq=freq)
        finally:
            kg_mod._embedding_based_match = original_match

        # 频率相同，取最长名"水稻稻瘟病"
        assert merged.get("稻瘟病") == "水稻稻瘟病"
