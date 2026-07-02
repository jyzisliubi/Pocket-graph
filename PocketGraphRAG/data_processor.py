"""
知识图谱数据处理器

功能：
1. 解析知识图谱三元组文件（头实体|关系|尾实体）
2. 按实体分组，将相关三元组合并为自然语言文本块
3. 支持配置反向链接关系（如将"A 包含 B"反向补充到 B 的描述中）
4. 输出可直接用于向量化的文本块列表

这是 GraphRAG pipeline 的数据预处理核心模块。
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .config import AUTO_REVERSE_LINK
from .kg_extractor import is_low_quality_entity
from .logging_config import get_logger

logger = get_logger(__name__)

AUTO_REVERSE_KEYWORDS = [
    "防治",
    "治疗",
    "治愈",
    "抑制",
    "预防",
    "包含",
    "包括",
    "含有",
    "组成",
    "属于",
    "归类于",
    "分为",
    "用于",
    "作用于",
    "应用于",
    "引起",
    "导致",
    "产生",
    "造成",
    "治疗",
    "医治",
    "缓解",
    "控制",
    "分泌",
    "产生",
    "合成",
    "危害",
    "侵害",
    "感染",
    "传播",
    "传染",
    "扩散",
]


def auto_detect_reverse_relations(relations: Set[str]) -> Set[str]:
    """自动检测哪些关系适合建立反向链接

    基于关键词匹配：如果关系名包含方向性动词（如"防治"、"包含"、"用于"等），
    则认为适合建立反向链接。
    """
    if not AUTO_REVERSE_LINK:
        return set()

    reverse_rels = set()
    for rel in relations:
        for keyword in AUTO_REVERSE_KEYWORDS:
            if keyword in rel:
                reverse_rels.add(rel)
                break
    return reverse_rels


class KGProcessor:
    """知识图谱数据处理器"""

    def __init__(
        self,
        data_path: str,
        reverse_link_relations: set = None,
        relation_templates: dict = None,
        schema=None,
    ):
        """
        Args:
            data_path: 三元组数据文件路径
            reverse_link_relations: 需要建立反向链接的关系集合。
                为 None 时使用配置文件中的值，若配置也为空且 AUTO_REVERSE_LINK 开启则自动推断。
            relation_templates: 关系到自然语言的映射模板。为 None 时使用通用格式。
            schema: RelationSchema 实例。加载三元组时归一化关系名，
                解决碎片化关系名（"2018年亩产"/"平均亩产" → "产量"）导致
                match_relations 关键词匹配召回率低的问题。None 不归一化
        """
        self.data_path = data_path
        self.triples: List[Tuple[str, str, str]] = []
        # 正向索引：头实体 → [(关系, 尾实体)]
        self.entity_relations: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # 反向索引：尾实体 → [(头实体, 关系)]
        self.reverse_relations: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # 去重集合
        self.seen_triples = set()

        self._reverse_link_relations_input = reverse_link_relations
        self.reverse_link_relations: Set[str] = set()
        self.relation_templates = relation_templates or {}
        # Schema 驱动：关系名归一化器
        self.schema = schema

    def load_triples(self) -> List[Tuple[str, str, str]]:
        """从文件加载三元组，自动去重和清洗

        若构造时传入了 schema，会对关系名做归一化（碎片化 → 标准名），
        提升 match_relations 关键词匹配召回率。

        实体名质量清洗：剔除头/尾为句子片段、用法描述、LLM 占位符的三元组，
        避免下游 match_entities 把整句话当实体召回（污染检索）。
        """
        removed_low_quality = 0
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                head, relation, tail = [p.strip() for p in parts]
                if not head or not relation or not tail:
                    continue
                # 实体名质量清洗：拦截句子片段 / 用法描述 / LLM 占位符
                if is_low_quality_entity(head) or is_low_quality_entity(tail):
                    removed_low_quality += 1
                    continue
                # Schema 驱动：归一化关系名
                if self.schema:
                    relation = self.schema.normalize_relation(relation)
                triple_key = (head, relation, tail)
                if triple_key not in self.seen_triples:
                    self.seen_triples.add(triple_key)
                    self.triples.append(triple_key)
                    self.entity_relations[head].append((relation, tail))
                    self.reverse_relations[tail].append((head, relation))

        if removed_low_quality:
            logger.info(
                "实体名清洗：剔除 %s 条低质量三元组（句子片段/用法描述/占位符）",
                removed_low_quality,
            )

        # 自动推断反向链接关系
        if self._reverse_link_relations_input is not None:
            self.reverse_link_relations = set(self._reverse_link_relations_input)
        else:
            all_relations = set(r for _, r, _ in self.triples)
            self.reverse_link_relations = auto_detect_reverse_relations(all_relations)

        return self.triples

    def _format_relation(self, relation: str, tail: str) -> str:
        """将关系+尾实体转换为自然语言描述"""
        template = self.relation_templates.get(relation)
        if template:
            return template.format(tail=tail)
        return f"{relation}：{tail}"

    def _find_reverse_targets(self, entity: str) -> List[str]:
        """
        查找反向链接对象。
        """
        targets = []
        for head, relation in self.reverse_relations.get(entity, []):
            if relation in self.reverse_link_relations:
                targets.append(f"{head}（{relation}）")
        return targets

    def _needs_reverse_links(self, entity: str) -> bool:
        """判断一个实体是否需要补充反向链接信息"""
        for _, relation in self.reverse_relations.get(entity, []):
            if relation in self.reverse_link_relations:
                return True
        return False

    def process(self) -> List[Dict[str, str]]:
        """
        将知识图谱三元组转换为文本块。

        处理策略：
        1. 按头实体分组，将同一实体的所有属性合并为一个文本块
        2. 对药剂类实体，额外补充"可防治对象"信息
        3. 每个文本块包含完整的上下文，有利于 embedding 检索

        Returns:
            文本块列表，每个块包含 entity、text 和 metadata
        """
        if not self.triples:
            self.load_triples()

        return self.process_entities(list(self.entity_relations.keys()))

    def process_entities(self, entity_names: List[str]) -> List[Dict[str, str]]:
        """仅重建指定实体的文本块（增量索引用：只重编码受影响实体）。

        与 process() 使用完全相同的格式化逻辑，但只输出 entity_names 列出的实体。
        未在 entity_names 中的实体不会被处理。反向链接信息基于当前已加载的
        全量三元组计算，保证重建出的 chunk 与全量 process() 结果一致。

        Args:
            entity_names: 需要重建的实体名列表

        Returns:
            文本块列表（仅包含指定实体）
        """
        if not self.triples:
            self.load_triples()

        # 去重保序
        seen = set()
        target_entities = []
        for e in entity_names:
            if e not in seen and e in self.entity_relations:
                seen.add(e)
                target_entities.append(e)

        chunks = []
        for entity in target_entities:
            relations = self.entity_relations[entity]
            lines = [f"【{entity}】"]

            for relation, tail in relations:
                lines.append(self._format_relation(relation, tail))

            if self._needs_reverse_links(entity):
                targets = self._find_reverse_targets(entity)
                if targets:
                    target_text = "、".join(dict.fromkeys(targets))  # 去重保序
                    lines.append(f"关联对象：{target_text}")

            text = "\n".join(lines)
            chunks.append(
                {
                    "entity": entity,
                    "text": text,
                    "metadata": {
                        "entity": entity,
                        "num_triples": len(relations),
                        "has_reverse_links": self._needs_reverse_links(entity),
                    },
                }
            )

        return chunks

    def add_triples(
        self, triples: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """将新三元组合并到当前内存状态（增量索引用）。

        对 (head, relation, tail) 做去重，已存在的跳过。同时维护
        entity_relations / reverse_relations / seen_triples / triples。
        不会写磁盘，磁盘持久化由调用方负责。

        Args:
            triples: 待合并的三元组列表 [(head, relation, tail), ...]

        Returns:
            实际新增的三元组列表（去重后的）
        """
        added = []
        for head, relation, tail in triples:
            head = (head or "").strip()
            relation = (relation or "").strip()
            tail = (tail or "").strip()
            if not head or not relation or not tail:
                continue
            # 实体名质量清洗：增量追加也拦截低质量实体
            if is_low_quality_entity(head) or is_low_quality_entity(tail):
                continue
            if self.schema:
                relation = self.schema.normalize_relation(relation)
            key = (head, relation, tail)
            if key in self.seen_triples:
                continue
            self.seen_triples.add(key)
            self.triples.append(key)
            self.entity_relations[head].append((relation, tail))
            self.reverse_relations[tail].append((head, relation))
            added.append(key)

        # 新增三元组可能引入新的反向链接关系，重算一次
        if self._reverse_link_relations_input is None:
            all_relations = set(r for _, r, _ in self.triples)
            self.reverse_link_relations = auto_detect_reverse_relations(all_relations)

        return added

    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        if not self.triples:
            self.load_triples()

        all_relations = set(r for _, r, _ in self.triples)
        reverse_linked_entities = sum(
            1
            for entity in self.entity_relations.keys()
            if self._needs_reverse_links(entity)
        )

        return {
            "总三元组数": len(self.triples),
            "唯一头实体数": len(self.entity_relations),
            "唯一尾实体数": len(self.reverse_relations),
            "关系类型数": len(all_relations),
            "关系类型列表": sorted(all_relations),
            "含反向链接实体数": reverse_linked_entities,
        }

    def save_chunks(self, chunks: List[Dict], output_path: str):
        """保存处理后的文本块到 JSON 文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_chunks(chunks_path: str) -> List[Dict]:
        """从 JSON 文件加载文本块"""
        with open(chunks_path, encoding="utf-8") as f:
            return json.load(f)
