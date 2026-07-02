"""Benchmark 模块单元测试"""

import json
import os

import pytest

from PocketGraphRAG.benchmark import (
    BenchmarkEvaluator,
    BenchmarkReport,
    QuestionResult,
    compare_reports,
)


@pytest.fixture
def sample_benchmark_file(tmp_path):
    """创建一个示例评测集文件"""
    data = {
        "name": "test_benchmark",
        "version": "1.0",
        "questions": [
            {
                "id": "q1",
                "question": "稻瘟病有什么症状？",
                "expected_entities": ["稻瘟病"],
                "expected_relations": ["症状表现"],
                "expected_answer_keywords": ["斑点", "坏死"],
                "type": "single_fact",
                "difficulty": "easy",
            },
            {
                "id": "q2",
                "question": "三环唑能治什么病？",
                "expected_entities": ["三环唑"],
                "expected_relations": ["化学防治"],
                "expected_answer_keywords": ["稻瘟病"],
                "type": "reverse_link",
                "difficulty": "medium",
            },
        ],
    }
    f = tmp_path / "test_benchmark.json"
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(f)


class TestBenchmarkEvaluator:
    def test_load_benchmark(self, sample_benchmark_file):
        """测试加载评测集"""
        evaluator = BenchmarkEvaluator(sample_benchmark_file)
        assert evaluator.name == "test_benchmark"
        assert len(evaluator.questions) == 2

    def test_compute_retrieval_metrics_all_hit(self):
        """测试全部命中时的检索指标"""
        result = QuestionResult(
            question_id="q1",
            question="测试问题",
            question_type="single_fact",
            difficulty="easy",
            expected_entities=["稻瘟病", "纹枯病"],
            expected_relations=[],
            expected_keywords=[],
            retrieved_entities=["稻瘟病", "纹枯病", "稻飞虱"],
            retrieved_scores=[0.9, 0.8, 0.7],
        )
        BenchmarkEvaluator._compute_retrieval_metrics(
            result, ["稻瘟病", "纹枯病"], top_k=5
        )

        assert result.mrr == pytest.approx(1.0)
        assert result.recall_at_1 == pytest.approx(0.5)
        assert result.recall_at_3 == pytest.approx(1.0)
        assert result.recall_at_5 == pytest.approx(1.0)
        assert result.precision_at_1 == pytest.approx(1.0)
        assert result.precision_at_3 == pytest.approx(2 / 3)
        assert result.hit_at_1 is True
        assert result.hit_at_3 is True
        assert result.entity_coverage == pytest.approx(1.0)

    def test_compute_retrieval_metrics_partial_hit(self):
        """测试部分命中时的检索指标"""
        result = QuestionResult(
            question_id="q1",
            question="测试问题",
            question_type="single_fact",
            difficulty="easy",
            expected_entities=["稻瘟病", "纹枯病"],
            expected_relations=[],
            expected_keywords=[],
            retrieved_entities=["稻飞虱", "稻瘟病", "白叶枯病"],
            retrieved_scores=[0.9, 0.8, 0.7],
        )
        BenchmarkEvaluator._compute_retrieval_metrics(
            result, ["稻瘟病", "纹枯病"], top_k=5
        )

        assert result.mrr == pytest.approx(0.5)  # 第二个位置命中
        assert result.recall_at_3 == pytest.approx(0.5)
        assert result.hit_at_1 is False
        assert result.hit_at_3 is True

    def test_compute_retrieval_metrics_no_hit(self):
        """测试完全没命中时的检索指标"""
        result = QuestionResult(
            question_id="q1",
            question="测试问题",
            question_type="single_fact",
            difficulty="easy",
            expected_entities=["稻瘟病"],
            expected_relations=[],
            expected_keywords=[],
            retrieved_entities=["稻飞虱", "纹枯病"],
            retrieved_scores=[0.9, 0.8],
        )
        BenchmarkEvaluator._compute_retrieval_metrics(result, ["稻瘟病"], top_k=5)

        assert result.mrr == pytest.approx(0.0)
        assert result.recall_at_3 == pytest.approx(0.0)
        assert result.hit_at_5 is False
        assert result.entity_coverage == pytest.approx(0.0)

    def test_compute_retrieval_metrics_no_expected(self):
        """测试没有期望实体时，指标应为满分"""
        result = QuestionResult(
            question_id="q1",
            question="测试问题",
            question_type="single_fact",
            difficulty="easy",
            expected_entities=[],
            expected_relations=[],
            expected_keywords=[],
            retrieved_entities=["稻飞虱"],
            retrieved_scores=[0.9],
        )
        BenchmarkEvaluator._compute_retrieval_metrics(result, [], top_k=5)

        assert result.mrr == pytest.approx(1.0)
        assert result.recall_at_3 == pytest.approx(1.0)
        assert result.hit_at_5 is True
        assert result.entity_coverage == pytest.approx(1.0)

    def test_compute_keyword_metrics_all_hit(self):
        """测试关键词全部命中"""
        result = QuestionResult(
            question_id="q1",
            question="测试",
            question_type="test",
            difficulty="easy",
            expected_entities=[],
            expected_relations=[],
            expected_keywords=["稻瘟病", "斑点"],
        )
        result.answer = "稻瘟病的症状是叶片出现斑点和坏死"
        BenchmarkEvaluator._compute_keyword_metrics(result, ["稻瘟病", "斑点"])

        assert result.keyword_hit_rate == pytest.approx(1.0)

    def test_compute_keyword_metrics_partial_hit(self):
        """测试关键词部分命中"""
        result = QuestionResult(
            question_id="q1",
            question="测试",
            question_type="test",
            difficulty="easy",
            expected_entities=[],
            expected_relations=[],
            expected_keywords=["稻瘟病", "斑点", "坏死"],
        )
        result.answer = "稻瘟病会导致叶片坏死"
        BenchmarkEvaluator._compute_keyword_metrics(result, ["稻瘟病", "斑点", "坏死"])

        assert result.keyword_hit_rate == pytest.approx(2 / 3)

    def test_compute_keyword_metrics_no_keywords(self):
        """测试没有期望关键词时满分"""
        result = QuestionResult(
            question_id="q1",
            question="测试",
            question_type="test",
            difficulty="easy",
            expected_entities=[],
            expected_relations=[],
            expected_keywords=[],
        )
        result.answer = "随便什么回答"
        BenchmarkEvaluator._compute_keyword_metrics(result, [])

        assert result.keyword_hit_rate == pytest.approx(1.0)


class TestBenchmarkReport:
    def test_compute_overall(self):
        """测试计算总体指标"""
        report = BenchmarkReport(
            benchmark_name="test",
            total_questions=2,
            config_name="test_config",
            search_mode="vector",
            use_multihop=False,
            use_pagerank=False,
            top_k=5,
            results=[
                QuestionResult(
                    question_id="q1",
                    question="q1",
                    question_type="a",
                    difficulty="easy",
                    expected_entities=["e1"],
                    expected_relations=[],
                    expected_keywords=[],
                    recall_at_3=1.0,
                    mrr=1.0,
                    hit_at_3=True,
                    entity_coverage=1.0,
                    keyword_hit_rate=1.0,
                ),
                QuestionResult(
                    question_id="q2",
                    question="q2",
                    question_type="b",
                    difficulty="hard",
                    expected_entities=["e2"],
                    expected_relations=[],
                    expected_keywords=[],
                    recall_at_3=0.5,
                    mrr=0.5,
                    hit_at_3=True,
                    entity_coverage=0.5,
                    keyword_hit_rate=0.5,
                ),
            ],
        )

        overall = report.compute_overall()
        assert overall["recall@3"] == pytest.approx(0.75)
        assert overall["mrr"] == pytest.approx(0.75)
        assert overall["hit_rate@3"] == pytest.approx(1.0)
        assert overall["entity_coverage"] == pytest.approx(0.75)

    def test_compute_by_type(self):
        """测试按类型分组统计"""
        report = BenchmarkReport(
            benchmark_name="test",
            total_questions=3,
            config_name="test_config",
            search_mode="vector",
            use_multihop=False,
            use_pagerank=False,
            top_k=5,
            results=[
                QuestionResult(
                    question_id="q1",
                    question="q1",
                    question_type="single_fact",
                    difficulty="easy",
                    expected_entities=[],
                    expected_relations=[],
                    expected_keywords=[],
                    recall_at_3=1.0,
                    mrr=1.0,
                    hit_at_5=True,
                    keyword_hit_rate=1.0,
                ),
                QuestionResult(
                    question_id="q2",
                    question="q2",
                    question_type="single_fact",
                    difficulty="easy",
                    expected_entities=[],
                    expected_relations=[],
                    expected_keywords=[],
                    recall_at_3=0.8,
                    mrr=0.8,
                    hit_at_5=True,
                    keyword_hit_rate=0.8,
                ),
                QuestionResult(
                    question_id="q3",
                    question="q3",
                    question_type="comparison",
                    difficulty="hard",
                    expected_entities=[],
                    expected_relations=[],
                    expected_keywords=[],
                    recall_at_3=0.5,
                    mrr=0.3,
                    hit_at_5=False,
                    keyword_hit_rate=0.4,
                ),
            ],
        )

        by_type = report.compute_by_type()
        assert "single_fact" in by_type
        assert "comparison" in by_type
        assert by_type["single_fact"]["count"] == 2
        assert by_type["comparison"]["count"] == 1
        assert by_type["single_fact"]["recall@3"] == pytest.approx(0.9)

    def test_save_report(self, tmp_path):
        """测试保存报告"""
        report = BenchmarkReport(
            benchmark_name="test",
            total_questions=1,
            config_name="test_config",
            search_mode="vector",
            use_multihop=False,
            use_pagerank=False,
            top_k=5,
            results=[
                QuestionResult(
                    question_id="q1",
                    question="测试问题",
                    question_type="single_fact",
                    difficulty="easy",
                    expected_entities=["e1"],
                    expected_relations=[],
                    expected_keywords=[],
                    answer="回答内容",
                ),
            ],
        )

        filepath = str(tmp_path / "report.json")
        report.save(filepath)

        assert os.path.exists(filepath)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        assert data["benchmark_name"] == "test"
        assert data["config_name"] == "test_config"
        assert "overall" in data
        assert "by_type" in data
        assert len(data["results"]) == 1


class TestCompareReports:
    def test_compare_empty(self):
        """测试空报告对比"""
        assert "没有可对比" in compare_reports([])

    def test_compare_multiple(self):
        """测试多个报告对比"""
        report1 = BenchmarkReport(
            benchmark_name="test",
            total_questions=1,
            config_name="Config A",
            search_mode="vector",
            use_multihop=False,
            use_pagerank=False,
            top_k=5,
            results=[
                QuestionResult(
                    question_id="q1",
                    question="q1",
                    question_type="a",
                    difficulty="easy",
                    expected_entities=[],
                    expected_relations=[],
                    expected_keywords=[],
                    recall_at_3=0.8,
                    mrr=0.7,
                    hit_at_3=True,
                    entity_coverage=0.8,
                    keyword_hit_rate=0.9,
                ),
            ],
        )
        report2 = BenchmarkReport(
            benchmark_name="test",
            total_questions=1,
            config_name="Config B",
            search_mode="mix",
            use_multihop=False,
            use_pagerank=False,
            top_k=5,
            results=[
                QuestionResult(
                    question_id="q1",
                    question="q1",
                    question_type="a",
                    difficulty="easy",
                    expected_entities=[],
                    expected_relations=[],
                    expected_keywords=[],
                    recall_at_3=0.95,
                    mrr=0.9,
                    hit_at_3=True,
                    entity_coverage=0.95,
                    keyword_hit_rate=1.0,
                ),
            ],
        )

        result = compare_reports([report1, report2])
        assert "Config A" in result
        assert "Config B" in result
        assert "recall@3" in result
        assert "mrr" in result
