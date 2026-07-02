"""
Schema 驱动抽取单元测试

验证：
1. RelationSchema 归一化各种碎片化关系名
2. KGProcessor 带 schema 加载时关系名被归一化
3. 实际 rice 数据集的关系名覆盖率（归一化效果）
4. 向后兼容：无 schema 时行为不变
"""

import os
from collections import Counter

import pytest

from PocketGraphRAG.data_processor import KGProcessor
from PocketGraphRAG.schema import RelationSchema

# ========================
# 1. RelationSchema 归一化
# ========================


class TestRelationSchema:
    def setup_method(self):
        self.schema = RelationSchema()

    def test_synonym_dict_exact_match(self):
        """同义词字典精确匹配"""
        assert self.schema.normalize_relation("症状") == "症状表现"
        assert self.schema.normalize_relation("典型症状") == "症状表现"
        assert self.schema.normalize_relation("病原菌") == "致病菌"
        assert self.schema.normalize_relation("药剂防治") == "化学防治"

    def test_pattern_match_year_variants(self):
        """正则模式匹配带年份的变体"""
        assert self.schema.normalize_relation("2018年亩产") == "产量"
        assert self.schema.normalize_relation("2019年平均亩产") == "产量"
        assert self.schema.normalize_relation("2021年区域试验亩产") == "产量"
        assert self.schema.normalize_relation("2023年生产试验亩产") == "产量"

    def test_pattern_match_increment_variants(self):
        """正则模式匹配带编号的变体"""
        assert self.schema.normalize_relation("技术要点1") == "栽培要点"
        assert self.schema.normalize_relation("优势2") == "包含"
        assert self.schema.normalize_relation("剂量1") == "用法用量"
        assert self.schema.normalize_relation("药剂2") == "防治"

    def test_pattern_match_comparison_variants(self):
        """正则模式匹配比对照增产/减产变体"""
        assert self.schema.normalize_relation("比对照增产(2021)") == "产量"
        assert self.schema.normalize_relation("比对照减产") == "产量"
        assert self.schema.normalize_relation("比丰两优四号增产(2013)") == "产量"

    def test_pattern_match_resistance_variants(self):
        """正则模式匹配抗性类变体"""
        assert self.schema.normalize_relation("稻瘟病抗性") == "抗性"
        assert self.schema.normalize_relation("白叶枯病敏感性") == "抗性"
        assert self.schema.normalize_relation("中抗稻瘟病") == "抗性"
        assert self.schema.normalize_relation("耐热性等级") == "抗性"

    def test_whitelist_relation_unchanged(self):
        """白名单内的关系保持不变"""
        assert self.schema.normalize_relation("防治") == "防治"
        assert self.schema.normalize_relation("症状表现") == "症状表现"
        assert self.schema.normalize_relation("用法用量") == "用法用量"

    def test_unknown_relation_unchanged(self):
        """未匹配的关系保留原样"""
        assert self.schema.normalize_relation("某未知关系") == "某未知关系"
        assert self.schema.normalize_relation("自定义属性") == "自定义属性"

    def test_empty_relation(self):
        """空关系名处理"""
        assert self.schema.normalize_relation("") == ""
        assert self.schema.normalize_relation(None) is None

    def test_is_allowed(self):
        """白名单检查"""
        assert self.schema.is_allowed("防治") is True
        assert self.schema.is_allowed("症状表现") is True
        assert self.schema.is_allowed("某未知关系") is False

    def test_get_canonical_relations(self):
        """获取白名单列表"""
        rels = self.schema.get_canonical_relations()
        assert "防治" in rels
        assert "症状表现" in rels
        assert "产量" in rels
        assert len(rels) > 10

    def test_build_prompt_constraint(self):
        """prompt 约束文本生成"""
        text = self.schema.build_prompt_constraint()
        assert "Schema 约束" in text
        assert "防治" in text
        assert "症状表现" in text

    def test_stats_tracking(self):
        """归一化统计"""
        self.schema.reset_stats()
        self.schema.normalize_relation("症状")  # normalized
        self.schema.normalize_relation("防治")  # unchanged (白名单)
        self.schema.normalize_relation("某未知关系")  # unknown
        stats = self.schema.get_stats()
        assert stats["normalized"] == 1
        assert stats["unchanged"] == 1
        assert stats["unknown"] == 1


# ========================
# 2. 自定义 schema 加载
# ========================


class TestCustomSchema:
    def test_load_from_json(self, tmp_path):
        """从 JSON 文件加载自定义 schema"""
        schema_json = {
            "allowed_relations": ["自定义关系1", "自定义关系2"],
            "synonyms": {"原关系": "自定义关系1"},
            "patterns": [["\\d+年", "时间关系"]],
        }
        json_path = tmp_path / "schema.json"
        import json

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(schema_json, f, ensure_ascii=False)

        schema = RelationSchema(schema_path=str(json_path))
        assert schema.is_allowed("自定义关系1")
        assert schema.normalize_relation("原关系") == "自定义关系1"
        assert schema.normalize_relation("2024年") == "时间关系"

    def test_load_nonexistent_json_falls_back(self):
        """加载不存在的 JSON 文件回退到默认 schema"""
        schema = RelationSchema(schema_path="/nonexistent/path/schema.json")
        # 应回退到默认 schema
        assert schema.normalize_relation("症状") == "症状表现"


# ========================
# 3. KGProcessor 集成
# ========================


class TestKGProcessorWithSchema:
    def _write_triples(self, tmp_path, triples):
        """写入临时三元组文件"""
        path = tmp_path / "triples.txt"
        with open(path, "w", encoding="utf-8") as f:
            for head, rel, tail in triples:
                f.write(f"{head} | {rel} | {tail}\n")
        return str(path)

    def test_load_with_schema_normalizes_relations(self, tmp_path):
        """带 schema 加载时关系名被归一化"""
        triples = [
            ("稻瘟病", "症状", "叶片褐斑"),
            ("稻瘟病", "典型症状", "病斑"),
            ("三环唑", "每亩用量", "推荐用量"),
            ("品种A", "2018年亩产", "高产"),
            ("品种A", "2019年平均亩产", "稳产"),
        ]
        path = self._write_triples(tmp_path, triples)

        # 无 schema：关系名保持原样
        proc_no_schema = KGProcessor(path)
        proc_no_schema.load_triples()
        rels_no_schema = set(r for _, r, _ in proc_no_schema.triples)
        assert "症状" in rels_no_schema
        assert "2018年亩产" in rels_no_schema

        # 有 schema：关系名被归一化
        schema = RelationSchema()
        proc_with_schema = KGProcessor(path, schema=schema)
        proc_with_schema.load_triples()
        rels_with_schema = set(r for _, r, _ in proc_with_schema.triples)

        # 碎片化关系名应被归一化
        assert "症状" not in rels_with_schema
        assert "典型症状" not in rels_with_schema
        assert "2018年亩产" not in rels_with_schema
        assert "2019年平均亩产" not in rels_with_schema

        # 归一化后的标准关系名应存在
        assert "症状表现" in rels_with_schema
        assert "产量" in rels_with_schema
        assert "用法用量" in rels_with_schema

    def test_dedup_after_normalization(self, tmp_path):
        """归一化后相同关系应去重"""
        triples = [
            ("稻瘟病", "症状", "叶片褐斑"),
            ("稻瘟病", "典型症状", "叶片褐斑"),  # 归一化后与上一条重复
        ]
        path = self._write_triples(tmp_path, triples)

        schema = RelationSchema()
        proc = KGProcessor(path, schema=schema)
        proc.load_triples()

        # 归一化后两条三元组完全相同，应只保留一条
        assert len(proc.triples) == 1
        assert proc.triples[0][1] == "症状表现"

    def test_backward_compatible_no_schema(self, tmp_path):
        """无 schema 时行为不变（向后兼容）"""
        triples = [
            ("稻瘟病", "症状", "叶片褐斑"),
            ("三环唑", "防治", "稻瘟病"),
        ]
        path = self._write_triples(tmp_path, triples)

        proc = KGProcessor(path)  # 不传 schema
        proc.load_triples()
        assert len(proc.triples) == 2
        assert proc.triples[0][1] == "症状"  # 保持原样


# ========================
# 4. 真实数据集覆盖率测试
# ========================


class TestRealDatasetCoverage:
    """验证 schema 对真实 rice 数据集的归一化覆盖率"""

    @pytest.fixture
    def real_relations(self):
        """加载真实数据集的所有关系名

        优先使用 .original.bak（原始碎片化数据），验证 schema 对未归一化数据的覆盖率。
        若不存在则用当前 DATA_PATH（已归一化数据，覆盖率会较低）。
        """
        from PocketGraphRAG.config import DATA_PATH

        # 优先用原始备份（碎片化数据），验证 schema 归一化能力
        # 文件名规则：triples.txt → triples.original.bak
        data_dir = os.path.dirname(DATA_PATH)
        data_basename = os.path.splitext(os.path.basename(DATA_PATH))[0]
        original_bak = os.path.join(data_dir, data_basename + ".original.bak")
        data_path = original_bak if os.path.exists(original_bak) else DATA_PATH

        if not os.path.exists(data_path):
            pytest.skip("真实数据集不存在")
        rels = Counter()
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                rels[parts[1].strip()] += 1
        return rels

    def test_normalization_coverage(self, real_relations):
        """验证 schema 对真实数据关系名的归一化覆盖率

        覆盖率 = 被归一化(非 unchanged/原样)的关系种类数 / 总关系种类数
        目标：> 60%（碎片化关系名占多数时应大幅归并）
        """
        schema = RelationSchema()
        schema.reset_stats()

        normalized_kinds = set()
        total_kinds = set()
        normalized_triples = 0
        total_triples = sum(real_relations.values())

        for rel, count in real_relations.items():
            total_kinds.add(rel)
            normalized = schema.normalize_relation(rel)
            if normalized != rel:
                normalized_kinds.add(rel)
                normalized_triples += count

        kind_coverage = len(normalized_kinds) / len(total_kinds)
        triple_coverage = normalized_triples / total_triples

        # 关系种类归一化率应 > 50%（数据集有 1400+ 种碎片化关系名）
        assert kind_coverage > 0.5, (
            f"关系种类归一化覆盖率仅 {kind_coverage:.1%}，期望 > 50%。"
            f"共 {len(total_kinds)} 种关系，归一化 {len(normalized_kinds)} 种。"
        )
        # 三元组归一化率应 > 30%
        assert triple_coverage > 0.3, (
            f"三元组归一化覆盖率仅 {triple_coverage:.1%}，期望 > 30%。"
        )

    def test_relation_count_reduced(self, real_relations):
        """归一化后唯一关系种类数应大幅减少"""
        schema = RelationSchema()
        original_kinds = len(real_relations)
        normalized_kinds = set()
        for rel in real_relations:
            normalized_kinds.add(schema.normalize_relation(rel))
        # 归一化后种类数应减少 > 50%
        reduction = 1 - len(normalized_kinds) / original_kinds
        assert reduction > 0.5, (
            f"关系种类数仅减少 {reduction:.1%}，期望 > 50%。"
            f"原始 {original_kinds} 种 → 归一化后 {len(normalized_kinds)} 种。"
        )



class TestEntityTypes:
    """P1-2: 实体类型约束测试"""

    def test_default_entity_types_not_empty(self):
        from PocketGraphRAG.schema import RelationSchema, DEFAULT_ENTITY_TYPES
        assert len(DEFAULT_ENTITY_TYPES) > 0
        s = RelationSchema()
        assert s.entity_types == DEFAULT_ENTITY_TYPES

    def test_custom_entity_types(self):
        from PocketGraphRAG.schema import RelationSchema
        s = RelationSchema(entity_types=["病害", "药剂"])
        assert s.entity_types == ["病害", "药剂"]

    def test_build_prompt_constraint_contains_entity_types(self):
        from PocketGraphRAG.schema import RelationSchema
        s = RelationSchema()
        c = s.build_prompt_constraint()
        assert "实体类型提示" in c
        assert "人物" in c  # 通用默认实体类型
        assert "名词短语" in c

    def test_build_prompt_constraint_empty_entity_types(self):
        from PocketGraphRAG.schema import RelationSchema
        s = RelationSchema(entity_types=[])
        c = s.build_prompt_constraint()
        # entity_types 为空时不输出实体类型提示
        assert "实体类型提示" not in c

    def test_load_from_json_with_entity_types(self, tmp_path):
        import json
        from PocketGraphRAG.schema import RelationSchema

        schema_json = {
            "allowed_relations": ["防治"],
            "entity_types": ["病害A", "药剂B"],
            "synonyms": {},
            "patterns": [],
        }
        path = tmp_path / "test_schema.json"
        path.write_text(json.dumps(schema_json, ensure_ascii=False), encoding="utf-8")

        s = RelationSchema(schema_path=str(path))
        assert s.entity_types == ["病害A", "药剂B"]
        assert s.allowed_relations == ["防治"]
