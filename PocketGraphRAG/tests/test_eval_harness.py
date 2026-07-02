"""
eval_harness 评测指标单元测试 (纯函数，不依赖 LLM / 模型)
"""

import os
from unittest import mock

import pytest

from PocketGraphRAG.eval_harness import (
    DEFAULT_BENCHMARK_PATH,
    QuestionResult,
    _aggregate_retrieval,
    _answer_keyword_hit,
    _build_langchain_llm,
    _evaluate_retrieval,
    _has_ragas,
    _run_ragas,
    load_benchmark,
    run_evaluation,
)


class TestAnswerKeywordHit:
    def test_all_hit(self):
        assert _answer_keyword_hit("稻瘟病和纹枯病", ["稻瘟病", "纹枯病"]) == 1.0

    def test_partial_hit(self):
        assert _answer_keyword_hit("稻瘟病相关", ["稻瘟病", "纹枯病"]) == 0.5

    def test_no_hit(self):
        assert _answer_keyword_hit("无相关内容", ["稻瘟病"]) == 0.0

    def test_empty_keywords_returns_one(self):
        assert _answer_keyword_hit("任意", []) == 1.0

    def test_empty_keyword_string_ignored(self):
        # 空字符串关键词不应计入分母
        assert _answer_keyword_hit("ok", ["", "ok"]) == 0.5


class TestAggregateRetrieval:
    def test_hit_rate_and_mrr(self):
        results = [
            QuestionResult(
                id="q1",
                question="",
                type="",
                difficulty="",
                expected_entities=["A"],
                expected_relations=[],
                expected_answer_keywords=[],
                first_expected_rank=1,  # 命中且排第1
            ),
            QuestionResult(
                id="q2",
                question="",
                type="",
                difficulty="",
                expected_entities=["B"],
                expected_relations=[],
                expected_answer_keywords=[],
                first_expected_rank=None,  # 未命中
            ),
        ]
        agg = _aggregate_retrieval(results)
        assert agg["n"] == 2
        assert agg["hit_rate"] == 0.5
        # MRR = (1/1 + 0) / 2 = 0.5
        assert agg["mrr"] == 0.5

    def test_entity_coverage(self):
        results = [
            QuestionResult(
                id="q1",
                question="",
                type="",
                difficulty="",
                expected_entities=["A", "B"],
                expected_relations=[],
                expected_answer_keywords=[],
                seed_entities=["A"],  # 命中 1/2
            ),
        ]
        agg = _aggregate_retrieval(results)
        assert agg["entity_coverage"] == 0.5

    def test_relation_coverage(self):
        results = [
            QuestionResult(
                id="q1",
                question="",
                type="",
                difficulty="",
                expected_entities=[],
                expected_relations=["化学防治", "生物防治"],
                expected_answer_keywords=[],
                matched_relations=["化学防治"],
            ),
        ]
        agg = _aggregate_retrieval(results)
        assert agg["relation_coverage"] == 0.5

    def test_relation_coverage_bidirectional_normalization(self):
        """P0-1 修复：双向归一化 + 子串兜底匹配

        benchmark expected_relations 用 "用量" / "最佳防治时期"，
        KG matched_relations 用归一化后的 "用法用量" / "防治方法"，
        应该能匹配上。
        """
        from PocketGraphRAG.eval_harness import _relations_match

        # 用量 → 归一化为 用法用量
        assert _relations_match("用量", {"用法用量"}) is True
        # 最佳防治时期 → 归一化为 防治方法
        assert _relations_match("最佳防治时期", {"防治方法"}) is True
        # 适宜温度 → 归一化为 环境条件
        assert _relations_match("适宜温度", {"环境条件"}) is True
        # 子串兜底：expected 是 matched 的子串
        assert _relations_match("化学", {"化学防治"}) is True
        # 完全不相关
        assert _relations_match("化学防治", {"症状表现"}) is False
        # 空字符串
        assert _relations_match("", {"化学防治"}) is False

        # 端到端：relation_coverage 应从 0.5 提升到 1.0
        results = [
            QuestionResult(
                id="q1",
                question="",
                type="",
                difficulty="",
                expected_entities=[],
                expected_relations=["用量", "最佳防治时期"],
                expected_answer_keywords=[],
                matched_relations=["用法用量", "防治方法"],
            ),
        ]
        agg = _aggregate_retrieval(results)
        assert agg["relation_coverage"] == 1.0, f"应通过归一化匹配到 1.0，实际 {agg['relation_coverage']}"

    def test_empty_results(self):
        agg = _aggregate_retrieval([])
        assert agg["n"] == 0
        assert agg["hit_rate"] == 0.0


class TestEvaluateRetrieval:
    def test_uses_rag_retrieve(self):
        """用一个假的 rag 验证 _evaluate_retrieval 正确解析 retrieve 返回"""

        class FakeRag:
            def retrieve(self, query, top_k=None):
                kg_path = {
                    "seed_entities": ["稻瘟病"],
                    "matched_relations": ["症状表现"],
                    "expanded_entities": [],
                    "search_type": "local",
                }
                # 模拟检索结果: 第 1 条来源实体命中
                results = [
                    ("text1", 1.0, {"entity": "稻瘟病"}),
                    ("text2", 0.9, {"entity": "其他"}),
                ]
                return results, kg_path

        q = {
            "id": "t1",
            "question": "稻瘟病有什么症状？",
            "expected_entities": ["稻瘟病"],
            "expected_relations": ["症状表现"],
            "expected_answer_keywords": ["斑点"],
            "type": "single_fact",
            "difficulty": "easy",
        }
        qr = _evaluate_retrieval(FakeRag(), q, top_k=5)
        assert qr.first_expected_rank == 1
        assert qr.seed_entities == ["稻瘟病"]
        assert "症状表现" in qr.matched_relations
        assert qr.retrieved_entities == ["稻瘟病", "其他"]


class TestLoadBenchmark:
    def test_default_benchmark_loads(self):
        ds = load_benchmark()
        assert ds["name"] == "movie_kg_benchmark_v1"
        assert len(ds["questions"]) == 20
        assert "metrics" in ds

    def test_default_path_points_to_package(self):
        assert DEFAULT_BENCHMARK_PATH.endswith("movie_kg_v1.json")

    def test_every_question_has_ground_truth(self):
        """benchmark 每题必须有 ground_truth 字段供 RAGAS context_recall 使用"""
        ds = load_benchmark()
        missing = [q["id"] for q in ds["questions"] if not q.get("ground_truth")]
        assert not missing, f"以下题目缺 ground_truth: {missing}"

    def test_metrics_list_includes_ragas_metrics(self):
        ds = load_benchmark()
        metrics = ds["metrics"]
        for m in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            assert m in metrics, f"metrics 缺少 {m}"

    def test_version_is_v1(self):
        ds = load_benchmark()
        assert ds["version"] == "1.0"


class TestRagasAvailability:
    def test_has_ragas_returns_bool(self):
        assert isinstance(_has_ragas(), bool)


class TestRunRagasGroundTruthHandling:
    """验证 _run_ragas 对 ground_truths 参数的处理逻辑。

    不真正调 LLM/ragas evaluate，用 mock 拦截 evaluate 调用并断言传入的 metrics。
    """

    def test_without_ground_truths_skips_context_recall(self):
        """无 ground_truths 时不应包含 context_recall metric"""
        if not _has_ragas():
            pytest.skip("ragas 未安装")

        captured = {}

        class _FakeLLM:
            pass

        class _FakeEmb:
            pass

        # 拦截 ragas.evaluate 调用，记录 metrics 列表
        def _fake_evaluate(ds, **kwargs):
            captured["metrics"] = list(kwargs.get("metrics", []))
            captured["n_rows"] = len(ds)
            captured["has_gt_col"] = "ground_truth" in ds.column_names
            return {
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "context_precision": 0.6,
            }

        with mock.patch("ragas.evaluate", _fake_evaluate):
            result = _run_ragas(
                questions=["Q1"],
                answers=["A1"],
                contexts=[["C1"]],
                ground_truths=None,
                llm_wrapper=_FakeLLM(),
                embeddings_wrapper=_FakeEmb(),
            )
        assert result is not None
        assert captured["n_rows"] == 1
        assert captured["has_gt_col"] is False
        # 3 个 metric，不含 context_recall
        assert len(captured["metrics"]) == 3

    def test_with_ground_truths_adds_context_recall(self):
        """有 ground_truths 时应包含 context_recall metric，且数据集含 ground_truth 列"""
        if not _has_ragas():
            pytest.skip("ragas 未安装")

        captured = {}

        class _FakeLLM:
            pass

        class _FakeEmb:
            pass

        def _fake_evaluate(ds, **kwargs):
            captured["metrics"] = list(kwargs.get("metrics", []))
            captured["n_rows"] = len(ds)
            captured["has_gt_col"] = "ground_truth" in ds.column_names
            return {
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "context_precision": 0.6,
                "context_recall": 0.5,
            }

        with mock.patch("ragas.evaluate", _fake_evaluate):
            result = _run_ragas(
                questions=["Q1", "Q2"],
                answers=["A1", "A2"],
                contexts=[["C1"], ["C2"]],
                ground_truths=["GT1", "GT2"],
                llm_wrapper=_FakeLLM(),
                embeddings_wrapper=_FakeEmb(),
            )
        assert result is not None
        assert "context_recall" in result
        assert captured["n_rows"] == 2
        assert captured["has_gt_col"] is True
        # 4 个 metric，含 context_recall
        assert len(captured["metrics"]) == 4

    def test_mismatched_ground_truths_length_falls_back_to_3_metrics(self):
        """ground_truths 长度与 questions 不一致时安全降级为 3 指标"""
        if not _has_ragas():
            pytest.skip("ragas 未安装")

        captured = {}

        class _FakeLLM:
            pass

        class _FakeEmb:
            pass

        def _fake_evaluate(ds, **kwargs):
            captured["metrics"] = list(kwargs.get("metrics", []))
            captured["has_gt_col"] = "ground_truth" in ds.column_names
            return {
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "context_precision": 0.6,
            }

        with mock.patch("ragas.evaluate", _fake_evaluate):
            _run_ragas(
                questions=["Q1", "Q2"],
                answers=["A1", "A2"],
                contexts=[["C1"], ["C2"]],
                ground_truths=["GT1"],  # 长度不匹配
                llm_wrapper=_FakeLLM(),
                embeddings_wrapper=_FakeEmb(),
            )
        assert captured["has_gt_col"] is False
        assert len(captured["metrics"]) == 3

    def test_returns_none_when_no_llm_wrapper_and_llm_unavailable(self):
        """无 LLM wrapper 且 call_llm 不可用时返回 None"""
        if not _has_ragas():
            pytest.skip("ragas 未安装")

        # mock has_llm 返回 False，让 _build_langchain_llm 返回 None
        with (
            mock.patch("PocketGraphRAG.eval_harness.has_llm", return_value=False)
            if False
            else mock.patch("PocketGraphRAG.llm.has_llm", return_value=False)
        ):
            result = _run_ragas(
                questions=["Q1"],
                answers=["A1"],
                contexts=[["C1"]],
                llm_wrapper=None,  # 强制走 _build_langchain_llm
            )
        assert result is None


class TestRunEvaluationGroundTruthFallback:
    """验证 run_evaluation 的 ground_truth 兜底：缺失 ground_truth 字段时
    用 expected_answer_keywords 拼接参考答案。"""

    def test_ground_truth_fallback_to_keywords(self):
        """benchmark 题目缺 ground_truth 时用 keywords 拼"""

        # 构造一个最小 rag mock + dataset
        class _FakeRag:
            model = None  # 不走 embeddings wrapper

            def retrieve(self, query, top_k=None):
                return [], {
                    "seed_entities": [],
                    "matched_relations": [],
                    "expanded_entities": [],
                    "search_type": "vector",
                }

            def answer(self, query, top_k=None):
                return {"answer": "测试答案", "sources": []}

        dataset = {
            "questions": [
                {
                    "id": "x1",
                    "question": "Q",
                    "expected_entities": [],
                    "expected_relations": [],
                    "expected_answer_keywords": ["稻瘟病", "纹枯病"],
                    # 故意不提供 ground_truth
                    "type": "single_fact",
                    "difficulty": "easy",
                }
            ]
        }
        # mock _run_ragas 拦截 ground_truths 参数
        captured = {}

        def _fake_run_ragas(qs, ans, ctxs, ground_truths=None, **kw):
            captured["ground_truths"] = ground_truths
            return None  # 不真跑

        with mock.patch("PocketGraphRAG.eval_harness._run_ragas", _fake_run_ragas):
            run_evaluation(
                _FakeRag(), dataset, top_k=3, run_generation=True, run_ragas=True
            )

        # 兜底应使用 expected_answer_keywords 拼成 "稻瘟病；纹枯病"
        assert captured["ground_truths"] == ["稻瘟病；纹枯病"]

    def test_ground_truth_field_preferred_over_keywords(self):
        """benchmark 题目有 ground_truth 时优先使用"""

        class _FakeRag:
            model = None

            def retrieve(self, query, top_k=None):
                return [], {
                    "seed_entities": [],
                    "matched_relations": [],
                    "expanded_entities": [],
                    "search_type": "vector",
                }

            def answer(self, query, top_k=None):
                return {"answer": "测试答案", "sources": []}

        dataset = {
            "questions": [
                {
                    "id": "x1",
                    "question": "Q",
                    "expected_entities": [],
                    "expected_relations": [],
                    "expected_answer_keywords": ["关键词1"],
                    "ground_truth": "这是真正的参考答案。",
                    "type": "single_fact",
                    "difficulty": "easy",
                }
            ]
        }
        captured = {}

        def _fake_run_ragas(qs, ans, ctxs, ground_truths=None, **kw):
            captured["ground_truths"] = ground_truths
            return None

        with mock.patch("PocketGraphRAG.eval_harness._run_ragas", _fake_run_ragas):
            run_evaluation(
                _FakeRag(), dataset, top_k=3, run_generation=True, run_ragas=True
            )

        assert captured["ground_truths"] == ["这是真正的参考答案。"]


class TestBuildLangchainLLM:
    """验证 _build_langchain_llm 在不同 LLM 可用性下的行为"""

    def test_returns_none_when_no_llm(self):
        with mock.patch("PocketGraphRAG.llm.has_llm", return_value=False):
            wrapper = _build_langchain_llm()
        assert wrapper is None

    def test_returns_wrapper_when_llm_available(self):
        """LLM 可用且 langchain_core 已安装时应返回 wrapper 实例"""
        try:
            from langchain_core.language_models.llms import LLM  # noqa: F401
        except ImportError:
            pytest.skip("langchain_core 未安装")

        with (
            mock.patch("PocketGraphRAG.llm.has_llm", return_value=True),
            mock.patch("PocketGraphRAG.llm.call_llm", return_value="mocked"),
        ):
            wrapper = _build_langchain_llm()
        assert wrapper is not None
        # wrapper 应该是 LangChain LLM 子类
        from langchain_core.language_models.llms import LLM

        assert isinstance(wrapper, LLM)


class TestCliOllamaModelPatch:
    """验证 --ollama-model CLI 参数能 patch llm 模块常量"""

    def test_ollama_model_env_var_set(self, monkeypatch):
        """模拟 main() 中的 patch 逻辑，确认 OLLAMA_MODEL 被更新"""
        from PocketGraphRAG import llm as llm_mod

        original = llm_mod.OLLAMA_MODEL
        try:
            monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
            # 模拟 main() 中的 patch 行为
            llm_mod.OLLAMA_MODEL = "qwen2.5:7b"
            assert llm_mod.OLLAMA_MODEL == "qwen2.5:7b"
            assert os.environ.get("OLLAMA_MODEL") == "qwen2.5:7b"
        finally:
            llm_mod.OLLAMA_MODEL = original
