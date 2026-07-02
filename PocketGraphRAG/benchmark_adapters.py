"""公开 benchmark 数据集适配器

把公开 multi-hop QA 数据集转成 PocketGraphRAG 的 benchmark 格式，
让用户能在公开数据集上跑分，对标 LightRAG / GraphRAG 的评测。

支持的公开数据集：
  1. HotpotQA (distractor setting) —— 维基百科多跳问答
     下载: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
     字段: question, answer, supporting_facts, context
  2. MuSiQue —— 多跳推理问答（2-4 跳）
     下载: https://github.com/google-research/language/tree/master/language/musique

转换后的格式与 movie_kg_v1.json 兼容：
    {
      "version": "hotpotqa-v1.1",
      "questions": [
        {
          "id": "...",
          "question": "...",
          "type": "multi-hop",
          "difficulty": "medium",
          "expected_entities": [...],   # 从 supporting_facts 的 title 提取
          "expected_relations": [],
          "expected_answer_keywords": [...],  # 从 answer 提取
          "ground_truth": "..."        # answer 字段
        }
      ]
    }

用法::

    from PocketGraphRAG.benchmark_adapters import convert_hotpotqa

    # 转换 HotpotQA 数据集
    benchmark = convert_hotpotqa("hotpot_train_v1.1.json", max_questions=100)
    # 保存为项目 benchmark 格式
    benchmark.to_json("hotpotqa_100.json")

    # 用 eval_harness 跑分
    from PocketGraphRAG.eval_harness import load_benchmark, run_evaluation
    ds = load_benchmark("hotpotqa_100.json")
    report = run_evaluation(rag, ds, top_k=5)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PublicBenchmark:
    """公开数据集转换后的 benchmark 格式（与 movie_kg_v1.json 兼容）"""

    version: str
    source: str  # 数据集来源标识
    questions: List[dict] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        """保存为项目 benchmark 格式 JSON"""
        data = {
            "version": self.version,
            "source": self.source,
            "description": (
                f"由 {self.source} 数据集转换而来，"
                f"用于在公开数据集上评测 PocketGraphRAG。"
            ),
            "questions": self.questions,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self.questions)


# ==========================
# HotpotQA 适配器
# ==========================


def _extract_keywords(answer: str) -> List[str]:
    """从答案中提取关键词（用于 answer_keyword_hit 指标）

    英文按空格分词，过滤停用词和短词；
    中文整体作为一个关键词（无分词器依赖，保持答案完整匹配）。
    """
    if not answer:
        return []
    # 去标点
    cleaned = re.sub(r"[，。！？、,\.!?;:；：\"'()\[\]{}]", " ", answer)
    cleaned = cleaned.strip()
    if not cleaned:
        return []

    # 判断是否含中文
    has_chinese = any("\u4e00" <= c <= "\u9fff" for c in cleaned)
    if has_chinese:
        # 中文：整体作为一个关键词（避免分词依赖）
        return [cleaned] if len(cleaned) >= 2 else []

    # 英文：按空格分词，过滤短词和常见停用词
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
    }
    words = cleaned.split()
    keywords = [w for w in words if len(w) >= 3 and w.lower() not in stop_words]
    return keywords


def _normalize_entity_name(title: str) -> str:
    """规范化实体名（HotpotQA 的 title 常带下划线和括号）"""
    if not title:
        return ""
    # 下划线转空格
    name = title.replace("_", " ").strip()
    # 去掉括号内容
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name).strip()
    return name


def convert_hotpotqa(
    hotpot_path: str,
    max_questions: Optional[int] = None,
    only_bridge: bool = True,
) -> PublicBenchmark:
    """转换 HotpotQA 数据集为项目 benchmark 格式

    Args:
        hotpot_path: HotpotQA JSON 文件路径
        max_questions: 最多转换多少题（None=全部）。建议先跑 100 题试水。
        only_bridge: 只保留 bridge 类型（多跳推理），过滤 comparison 类型

    Returns:
        PublicBenchmark 实例

    Note:
        HotpotQA 的 expected_entities 从 supporting_facts 的 title 提取，
        这些是回答问题需要的关键实体。在 GraphRAG 评测中，这些实体应该
        出现在 KG 检索结果里（recall@k 指标）。
    """
    with open(hotpot_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for item in data:
        if max_questions and len(questions) >= max_questions:
            break

        q_type = item.get("type", "")
        if only_bridge and q_type != "bridge":
            continue

        question = item.get("question", "").strip()
        if not question:
            continue

        answer = item.get("answer", "").strip()
        if not answer or answer.lower() in ("yes", "no"):
            # yes/no 答案无法做关键词命中，跳过
            continue

        # 从 supporting_facts 提取期望实体
        supporting_facts = item.get("supporting_facts", [])
        expected_entities = []
        seen = set()
        for fact in supporting_facts:
            if isinstance(fact, list) and len(fact) >= 1:
                title = fact[0]
                name = _normalize_entity_name(title)
                if name and name not in seen:
                    seen.add(name)
                    expected_entities.append(name)

        if not expected_entities:
            continue

        # 从 context 提取所有实体（用于构建 KG 的文档源）
        # context 格式: [["title", ["sent1", "sent2", ...]], ...]
        context = item.get("context", [])
        context_titles = [
            _normalize_entity_name(c[0])
            for c in context
            if isinstance(c, list) and len(c) >= 1
        ]

        questions.append(
            {
                "id": item.get("_id", f"hotpot_{len(questions)}"),
                "question": question,
                "type": "multi-hop",
                "difficulty": q_type or "medium",
                "expected_entities": expected_entities,
                "expected_relations": [],  # HotpotQA 不标注关系
                "expected_answer_keywords": _extract_keywords(answer),
                "ground_truth": answer,
                # 额外字段：构建 KG 用的文档源
                "context_entities": context_titles,
                "context_paragraphs": [
                    " ".join(sents) for _, sents in context if isinstance(sents, list)
                ],
            }
        )

    return PublicBenchmark(
        version="hotpotqa-v1.1",
        source="hotpotqa",
        questions=questions,
    )


# ==========================
# MuSiQue 适配器
# ==========================


def convert_musique(
    musique_path: str,
    max_questions: Optional[int] = None,
) -> PublicBenchmark:
    """转换 MuSiQue 数据集为项目 benchmark 格式

    MuSiQue 是 2-4 跳的多跳推理问答数据集，比 HotpotQA 更难。

    Args:
        musique_path: MuSiQue JSON 文件路径
        max_questions: 最多转换多少题

    Returns:
        PublicBenchmark 实例
    """
    with open(musique_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for item in data:
        if max_questions and len(questions) >= max_questions:
            break

        question = item.get("question", "").strip()
        if not question:
            continue

        answer = item.get("answer", "").strip()
        if not answer:
            continue

        # MuSiQue 的 supporting_items 提供推理链
        supporting = item.get("supporting_items", [])
        expected_entities = []
        seen = set()
        for supp in supporting:
            # supp 格式: {"title": "...", "paragraph_id": ...}
            if isinstance(supp, dict):
                title = supp.get("title", "")
                name = _normalize_entity_name(title)
                if name and name not in seen:
                    seen.add(name)
                    expected_entities.append(name)

        # paragraphs 字段提供所有文档
        paragraphs = item.get("paragraphs", [])
        context_entities = [
            _normalize_entity_name(p.get("title", ""))
            for p in paragraphs
            if isinstance(p, dict)
        ]
        context_text = [
            p.get("paragraph_text", "") for p in paragraphs if isinstance(p, dict)
        ]

        questions.append(
            {
                "id": item.get("id", f"musique_{len(questions)}"),
                "question": question,
                "type": "multi-hop",
                "difficulty": "hard",  # MuSiQue 整体比 HotpotQA 难
                "expected_entities": expected_entities,
                "expected_relations": [],
                "expected_answer_keywords": _extract_keywords(answer),
                "ground_truth": answer,
                "context_entities": context_entities,
                "context_paragraphs": context_text,
            }
        )

    return PublicBenchmark(
        version="musique-v1",
        source="musique",
        questions=questions,
    )


# ==========================
# 便利函数：从转换后的 benchmark 构建 KG 文档源
# ==========================


def build_corpus_from_benchmark(benchmark: PublicBenchmark) -> List[str]:
    """从转换后的 benchmark 提取所有文档段落，用于构建 KG

    评测流程：
        1. convert_hotpotqa → PublicBenchmark
        2. build_corpus_from_benchmark → 文档段落列表
        3. 用 extract_knowledge_graph 从文档段落抽取 KG
        4. 构建索引
        5. run_evaluation 跑分

    Returns:
        文档段落列表（每个段落是一个 context_paragraph）
    """
    corpus = []
    for q in benchmark.questions:
        corpus.extend(q.get("context_paragraphs", []))
    return corpus


if __name__ == "__main__":
    # CLI: 转换数据集
    import argparse

    parser = argparse.ArgumentParser(
        description="转换公开数据集为 PocketGraphRAG benchmark 格式"
    )
    parser.add_argument("dataset", choices=["hotpotqa", "musique"], help="数据集类型")
    parser.add_argument("input", help="输入 JSON 文件路径")
    parser.add_argument(
        "-o", "--output", required=True, help="输出 benchmark JSON 路径"
    )
    parser.add_argument(
        "-n", "--max-questions", type=int, default=None, help="最多转换多少题"
    )
    args = parser.parse_args()

    if args.dataset == "hotpotqa":
        bench = convert_hotpotqa(args.input, max_questions=args.max_questions)
    else:
        bench = convert_musique(args.input, max_questions=args.max_questions)

    bench.to_json(args.output)
    print(f"转换完成: {len(bench)} 题 → {args.output}")
