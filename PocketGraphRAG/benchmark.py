"""
PocketGraphRAG Benchmark 评测模块

提供标准评测指标和工具，用于量化比较不同检索模式的效果：
- Recall@k: 正确实体出现在前 k 个结果中的比例
- Precision@k: 前 k 个结果中正确实体的比例
- MRR (Mean Reciprocal Rank): 第一个正确实体的排名倒数的平均值
- Hit Rate@k: 至少有一个正确实体出现在前 k 个结果中的问题比例
- Entity Coverage: 所有期望实体中被检索到的比例
- Answer Keyword Hit: 回答中包含期望关键词的比例

使用方式：
    from PocketGraphRAG.benchmark import BenchmarkEvaluator

    evaluator = BenchmarkEvaluator("path/to/benchmark.json")
    report = evaluator.evaluate(rag_system, top_k=8)
    report.print_summary()
    report.save("benchmark_report.json")
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class QuestionResult:
    """单个问题的评测结果"""

    question_id: str
    question: str
    question_type: str
    difficulty: str
    expected_entities: List[str]
    expected_relations: List[str]
    expected_keywords: List[str]

    retrieved_entities: List[str] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)
    answer: str = ""

    # 各项指标
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    entity_coverage: float = 0.0
    keyword_hit_rate: float = 0.0


@dataclass
class BenchmarkReport:
    """Benchmark 完整评测报告"""

    benchmark_name: str
    total_questions: int
    config_name: str
    search_mode: str
    use_multihop: bool
    use_pagerank: bool
    top_k: int

    results: List[QuestionResult] = field(default_factory=list)

    def compute_overall(self) -> Dict[str, float]:
        """计算总体指标"""
        if not self.results:
            return {}

        n = len(self.results)
        metrics = {
            "recall@1": sum(r.recall_at_1 for r in self.results) / n,
            "recall@3": sum(r.recall_at_3 for r in self.results) / n,
            "recall@5": sum(r.recall_at_5 for r in self.results) / n,
            "recall@10": sum(r.recall_at_10 for r in self.results) / n,
            "precision@1": sum(r.precision_at_1 for r in self.results) / n,
            "precision@3": sum(r.precision_at_3 for r in self.results) / n,
            "precision@5": sum(r.precision_at_5 for r in self.results) / n,
            "mrr": sum(r.mrr for r in self.results) / n,
            "hit_rate@1": sum(1 for r in self.results if r.hit_at_1) / n,
            "hit_rate@3": sum(1 for r in self.results if r.hit_at_3) / n,
            "hit_rate@5": sum(1 for r in self.results if r.hit_at_5) / n,
            "entity_coverage": sum(r.entity_coverage for r in self.results) / n,
            "keyword_hit_rate": sum(r.keyword_hit_rate for r in self.results) / n,
        }
        return metrics

    def compute_by_type(self) -> Dict[str, Dict[str, float]]:
        """按问题类型分组统计"""
        groups: Dict[str, List[QuestionResult]] = {}
        for r in self.results:
            groups.setdefault(r.question_type, []).append(r)

        type_metrics = {}
        for qtype, group_results in groups.items():
            n = len(group_results)
            type_metrics[qtype] = {
                "count": n,
                "recall@3": sum(r.recall_at_3 for r in group_results) / n,
                "recall@5": sum(r.recall_at_5 for r in group_results) / n,
                "mrr": sum(r.mrr for r in group_results) / n,
                "hit_rate@3": sum(1 for r in group_results if r.hit_at_3) / n,
                "keyword_hit_rate": sum(r.keyword_hit_rate for r in group_results) / n,
            }
        return type_metrics

    def compute_by_difficulty(self) -> Dict[str, Dict[str, float]]:
        """按难度分组统计"""
        groups: Dict[str, List[QuestionResult]] = {}
        for r in self.results:
            groups.setdefault(r.difficulty, []).append(r)

        diff_metrics = {}
        for diff, group_results in groups.items():
            n = len(group_results)
            diff_metrics[diff] = {
                "count": n,
                "recall@3": sum(r.recall_at_3 for r in group_results) / n,
                "recall@5": sum(r.recall_at_5 for r in group_results) / n,
                "mrr": sum(r.mrr for r in group_results) / n,
                "hit_rate@5": sum(1 for r in group_results if r.hit_at_5) / n,
            }
        return diff_metrics

    def print_summary(self, detailed: bool = False):
        """打印评测摘要"""
        overall = self.compute_overall()

        print("\n" + "=" * 70)
        print(f"  Benchmark Report: {self.benchmark_name}")
        print(
            f"  Config: {self.config_name} (mode={self.search_mode}, "
            f"multihop={self.use_multihop}, pagerank={self.use_pagerank})"
        )
        print(f"  Questions: {self.total_questions}  |  Top-K: {self.top_k}")
        print("=" * 70)

        print(f"\n  {'Metric':<25s} {'Score':>10s}")
        print(f"  {'-' * 25}-+-{'-' * 10}")
        for key in [
            "recall@1",
            "recall@3",
            "recall@5",
            "recall@10",
            "precision@1",
            "precision@3",
            "precision@5",
            "mrr",
            "hit_rate@3",
            "hit_rate@5",
            "entity_coverage",
            "keyword_hit_rate",
        ]:
            val = overall.get(key, 0)
            print(f"  {key:<25s} {val:>10.2%}")

        print("\n  --- By Question Type ---")
        by_type = self.compute_by_type()
        print(
            f"  {'Type':<20s} {'Count':>5s} {'R@3':>8s} {'R@5':>8s} {'MRR':>8s} {'Hit@5':>8s}"
        )
        print(
            f"  {'-' * 20}-+-{'-' * 5}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}"
        )
        for qtype, m in sorted(by_type.items()):
            print(
                f"  {qtype:<20s} {m['count']:>5d} "
                f"{m['recall@3']:>8.2%} {m['recall@5']:>8.2%} "
                f"{m['mrr']:>8.2%} {m['hit_rate@5']:>8.2%}"
            )

        if detailed:
            print("\n  --- Per Question ---")
            for r in self.results:
                print(
                    f"  [{r.question_id}] {r.question[:40]:<40s} "
                    f"R@5={r.recall_at_5:.0%} MRR={r.mrr:.2f}"
                )

        print()

    def save(self, filepath: str):
        """保存评测报告到 JSON"""
        data = {
            "benchmark_name": self.benchmark_name,
            "config_name": self.config_name,
            "search_mode": self.search_mode,
            "use_multihop": self.use_multihop,
            "use_pagerank": self.use_pagerank,
            "top_k": self.top_k,
            "total_questions": self.total_questions,
            "overall": self.compute_overall(),
            "by_type": self.compute_by_type(),
            "by_difficulty": self.compute_by_difficulty(),
            "results": [asdict(r) for r in self.results],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] 评测报告已保存: {filepath}")


class BenchmarkEvaluator:
    """Benchmark 评测器"""

    def __init__(self, benchmark_path: str):
        """加载评测集"""
        with open(benchmark_path, encoding="utf-8") as f:
            self.data = json.load(f)

        self.name = self.data.get("name", "benchmark")
        self.questions = self.data.get("questions", [])
        print(f"[OK] 加载评测集: {self.name} ({len(self.questions)} 个问题)")

    def evaluate(
        self,
        rag_system,
        config_name: str = "default",
        top_k: int = 8,
    ) -> BenchmarkReport:
        """对 RAG 系统运行完整评测

        Args:
            rag_system: PocketGraphRAG 实例
            config_name: 配置名称，用于报告标识
            top_k: 检索 top_k

        Returns:
            BenchmarkReport 评测报告
        """
        report = BenchmarkReport(
            benchmark_name=self.name,
            total_questions=len(self.questions),
            config_name=config_name,
            search_mode=getattr(rag_system, "search_mode", "unknown"),
            use_multihop=getattr(rag_system, "use_multihop", False),
            use_pagerank=getattr(rag_system, "use_pagerank", False),
            top_k=top_k,
        )

        for q_data in self.questions:
            result = self._evaluate_question(rag_system, q_data, top_k)
            report.results.append(result)

        return report

    def _evaluate_question(
        self,
        rag_system,
        q_data: Dict[str, Any],
        top_k: int,
    ) -> QuestionResult:
        """评测单个问题"""
        question = q_data["question"]
        expected_entities = q_data.get("expected_entities", [])
        expected_relations = q_data.get("expected_relations", [])
        expected_keywords = q_data.get("expected_answer_keywords", [])

        result = QuestionResult(
            question_id=q_data.get("id", ""),
            question=question,
            question_type=q_data.get("type", "unknown"),
            difficulty=q_data.get("difficulty", "medium"),
            expected_entities=expected_entities,
            expected_relations=expected_relations,
            expected_keywords=expected_keywords,
        )

        full_answer = ""
        sources = []

        try:
            for step in rag_system.answer_stream(question):
                if "chunk" in step:
                    full_answer += step["chunk"]
                if "sources" in step:
                    sources = step["sources"]
        except Exception as e:
            print(f"  [WARN] 问题 '{question[:30]}...' 评测失败: {e}")
            result.answer = f"<error: {e}>"
            return result

        result.answer = full_answer.strip()

        # 提取检索到的实体（从 sources 的 meta 中）
        retrieved_entities = []
        retrieved_scores = []
        for s in sources:
            entity = s.get("entity", "") if isinstance(s, dict) else ""
            score = float(s.get("score", 0)) if isinstance(s, dict) else 0.0
            if entity and entity not in retrieved_entities:
                retrieved_entities.append(entity)
                retrieved_scores.append(score)

        result.retrieved_entities = retrieved_entities
        result.retrieved_scores = retrieved_scores

        # 计算指标
        self._compute_retrieval_metrics(result, expected_entities, top_k)
        self._compute_keyword_metrics(result, expected_keywords)

        return result

    @staticmethod
    def _compute_retrieval_metrics(
        result: QuestionResult,
        expected_entities: List[str],
        top_k: int,
    ):
        """计算检索质量指标"""
        if not expected_entities:
            result.entity_coverage = 1.0
            result.mrr = 1.0
            for k in [1, 3, 5, 10]:
                setattr(result, f"recall_at_{k}", 1.0)
                setattr(result, f"precision_at_{k}", 1.0)
            result.hit_at_1 = True
            result.hit_at_3 = True
            result.hit_at_5 = True
            return

        retrieved = result.retrieved_entities
        n_expected = len(expected_entities)

        # 期望实体集合（小写归一化比较）
        expected_set = {e.lower() for e in expected_entities}

        # 找到第一个命中位置
        first_hit_rank = None
        for i, ent in enumerate(retrieved):
            if ent.lower() in expected_set:
                first_hit_rank = i + 1  # 1-based
                break

        # MRR
        if first_hit_rank:
            result.mrr = 1.0 / first_hit_rank
        else:
            result.mrr = 0.0

        # Recall@k 和 Precision@k
        for k in [1, 3, 5, 10]:
            top_k_entities = retrieved[:k]
            hits = sum(1 for e in top_k_entities if e.lower() in expected_set)
            recall = hits / n_expected if n_expected > 0 else 0.0
            precision = hits / k if k > 0 else 0.0
            setattr(result, f"recall_at_{k}", min(recall, 1.0))
            setattr(result, f"precision_at_{k}", precision)

        # Hit Rate
        result.hit_at_1 = any(e.lower() in expected_set for e in retrieved[:1])
        result.hit_at_3 = any(e.lower() in expected_set for e in retrieved[:3])
        result.hit_at_5 = any(e.lower() in expected_set for e in retrieved[:5])

        # Entity Coverage: 所有期望实体中有多少被检索到了
        total_hits = sum(1 for e in retrieved if e.lower() in expected_set)
        result.entity_coverage = min(total_hits / n_expected, 1.0)

    @staticmethod
    def _compute_keyword_metrics(
        result: QuestionResult,
        expected_keywords: List[str],
    ):
        """计算回答关键词命中率"""
        if not expected_keywords:
            result.keyword_hit_rate = 1.0
            return

        answer_lower = result.answer.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        result.keyword_hit_rate = hits / len(expected_keywords)


def compare_reports(reports: List[BenchmarkReport]) -> str:
    """横向对比多个评测报告，输出对比表格

    Args:
        reports: 多个 BenchmarkReport 对象

    Returns:
        对比表格字符串
    """
    if not reports:
        return "没有可对比的报告"

    overall_metrics = [
        "recall@3",
        "recall@5",
        "mrr",
        "hit_rate@3",
        "hit_rate@5",
        "entity_coverage",
        "keyword_hit_rate",
    ]

    lines = []
    lines.append("\n" + "=" * 90)
    lines.append(f"  Benchmark Comparison ({len(reports)} configs)")
    lines.append("=" * 90)

    header = f"  {'Metric':<25s}"
    for r in reports:
        header += f" {r.config_name:>15s}"
    lines.append(header)
    lines.append(f"  {'-' * 25}" + "-+-" + "-+-".join("-" * 15 for _ in reports))

    for metric in overall_metrics:
        row = f"  {metric:<25s}"
        for r in reports:
            val = r.compute_overall().get(metric, 0)
            row += f" {val:>14.2%}"
        lines.append(row)

    lines.append("\n")
    return "\n".join(lines)
