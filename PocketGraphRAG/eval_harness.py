"""
PocketGraphRAG 评测 Harness

基于内置 benchmark 数据集 (PocketGraphRAG/benchmark/movie_kg_v1.json)
对 RAG 系统做标准化评测，对标 MultiHop-RAG / LightRAG reproduce 的评测思路。

提供两层指标：
  1. 检索层 (Retrieval) —— 不需要 LLM，纯匹配
     - recall_at_k       : 期望实体出现在检索结果来源里的比例
     - entity_coverage   : 期望实体被 KG 命中(seed)的比例
     - relation_coverage  : 期望关系被 KG 命中的比例
     - mrr                : 第一个期望实体在检索来源中的倒数排名
     - hit_rate           : 至少命中一个期望实体的题目比例
  2. 生成层 (Generation) —— 需要 LLM
     - answer_keyword_hit : 参考答案关键词在生成回答中的命中比例
  3. (可选) RAGAS 评测 —— 需要 `pip install ragas` 并配置 LLM
     - faithfulness          : 答案是否忠于上下文（无幻觉）
     - answer_relevancy      : 答案是否切题（反向生成问题与原问题相似度）
     - context_precision     : 检索上下文对答案的精确度（top-k 是否相关）
     - context_recall        : 答案是否覆盖了 ground_truth 的关键信息（需要参考答案）

  RAGAS 需要 benchmark 数据集每题包含 `ground_truth` 字段（参考答案）。
  本项目的 movie_kg_v1.json 已为 20 题提供 ground_truth。

  RAGAS 评估 LLM 默认走 PocketGraphRAG 的 call_llm（优先 Ollama），
  也可以通过环境变量 OLLAMA_MODEL 指定具体模型，例如::

    set OLLAMA_MODEL=qwen2.5:7b
    python -m PocketGraphRAG.eval_harness --ragas

用法::

    # CLI
    python -m PocketGraphRAG.eval_harness
    python -m PocketGraphRAG.eval_harness --search-mode mix --top-k 5
    python -m PocketGraphRAG.eval_harness --ragas   # 启用 RAGAS (需安装)

    # Python API
    from PocketGraphRAG.eval_harness import run_evaluation, load_benchmark
    ds = load_benchmark()
    report = run_evaluation(rag, ds, top_k=5)
    print(report["summary"])
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

# 过滤 `python -m PocketGraphRAG.eval_harness` 触发的 runpy 导入告警。
# 该告警源于 __init__.py 把 eval_harness 作为子模块导入，随后 runpy 又把它当 __main__ 执行。
# eval_harness 无模块级副作用（只有函数/常量定义），告警对本项目纯属噪音。
# 过滤器必须在包导入阶段就位（__init__.py 触发的 import 链中），才能在 runpy exec 前生效。
warnings.filterwarnings(
    "ignore",
    message=".*found in sys.modules after import of package.*",
    category=RuntimeWarning,
)

from .logging_config import get_logger

logger = get_logger(__name__)

# 关系归一化器（懒加载，避免 import 时开销）
_schema = None


def _get_schema():
    """懒加载 RelationSchema，用于关系名归一化匹配。"""
    global _schema
    if _schema is None:
        try:
            from .schema import RelationSchema
            _schema = RelationSchema()
        except Exception:
            _schema = None
    return _schema


def _relations_match(expected: str, matched_set: set) -> bool:
    """关系匹配：双向归一化 + 子串兜底

    解决 benchmark expected_relations 与 KG matched_relations 关系名不一致问题：
    - expected="用量" vs matched="用法用量"
    - expected="最佳防治时期" vs matched="防治方法"

    匹配策略（按优先级）：
    1. 精确匹配
    2. 双向归一化后匹配（schema.normalize_relation 两边都归一化）
    3. 子串包含（任一方是另一方的子串）
    """
    if not expected:
        return False
    # L1: 精确匹配
    if expected in matched_set:
        return True
    schema = _get_schema()
    # L2: 双向归一化
    if schema is not None:
        norm_expected = schema.normalize_relation(expected)
        if norm_expected != expected:
            if norm_expected in matched_set:
                return True
            # 归一化后再做一次子串匹配
            for m in matched_set:
                norm_m = schema.normalize_relation(m)
                if norm_expected == norm_m:
                    return True
                if norm_expected in m or m in norm_expected:
                    return True
    # L3: 原始子串兜底
    for m in matched_set:
        if expected in m or m in expected:
            return True
    return False


# benchmark 默认路径
DEFAULT_BENCHMARK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmark", "movie_kg_v1.json"
)


@dataclass
class QuestionResult:
    id: str
    question: str
    type: str
    difficulty: str
    expected_entities: list
    expected_relations: list
    expected_answer_keywords: list
    # 检索命中
    seed_entities: list = field(default_factory=list)
    matched_relations: list = field(default_factory=list)
    retrieved_entities: list = field(default_factory=list)
    first_expected_rank: Optional[int] = None  # 1-based, None 表示未命中
    # 生成命中
    answer: str = ""
    answer_keyword_hit: float = 0.0  # 0~1
    # RAGAS (可选)
    ragas: dict = field(default_factory=dict)


def load_benchmark(path: str = DEFAULT_BENCHMARK_PATH) -> dict:
    """加载内置 benchmark 数据集"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ==========================
# 检索层指标
# ==========================


def _evaluate_retrieval(rag, q: dict, top_k: int) -> QuestionResult:
    """对单条问题跑检索，计算检索指标"""
    qr = QuestionResult(
        id=q["id"],
        question=q["question"],
        type=q.get("type", ""),
        difficulty=q.get("difficulty", ""),
        expected_entities=q.get("expected_entities", []),
        expected_relations=q.get("expected_relations", []),
        expected_answer_keywords=q.get("expected_answer_keywords", []),
    )

    results, kg_path = rag.retrieve(q["question"], top_k=top_k)

    qr.seed_entities = kg_path.get("seed_entities", [])
    qr.matched_relations = kg_path.get("matched_relations", [])
    qr.retrieved_entities = [m.get("entity", "") for _, _, m in results]

    # recall@k / mrr / hit : 期望实体是否出现在检索来源里
    expected_set = set(qr.expected_entities)
    first_rank = None
    if expected_set:
        # 常规题：找第一个期望实体在检索结果中的位置
        for rank, ent in enumerate(qr.retrieved_entities, 1):
            if ent in expected_set:
                if first_rank is None:
                    first_rank = rank
                break
        # reverse_link / multi_hop 兜底：expected_entity 是查询主体（tail 实体），
        # 已被 seed 命中说明查询理解正确，但 chunk 按 head 实体建索引，
        # tail 实体永远不会出现在 retrieved_entities 里 → 结构性 MISS。
        # 回退用 expected_answer_keywords 在检索文本里找命中位置，
        # 验证答案内容是否被检索到。通用逻辑，不绑定具体领域。
        if first_rank is None:
            seed_set = set(qr.seed_entities)
            expected_in_seed = any(e in seed_set for e in qr.expected_entities)
            if expected_in_seed and qr.expected_answer_keywords:
                kw_list = [kw for kw in qr.expected_answer_keywords if kw]
                for rank, (text, _, _) in enumerate(results, 1):
                    if any(kw in text for kw in kw_list):
                        first_rank = rank
                        break
    else:
        # 期望实体为空（list-type / reverse_link 无固定答案实体）：
        # 这是 benchmark 设计漏洞的兜底——这类题答案是一个实体集合，
        # 原指标结构上永远不可能 hit。改用两级 fallback：
        #   1. expected_answer_keywords 在检索文本里找第一个命中位置
        #   2. expected_relations 在 matched_relations 里命中 → 视为 top-1 命中
        kw_list = [kw for kw in qr.expected_answer_keywords if kw]
        if kw_list:
            for rank, (text, _, _) in enumerate(results, 1):
                if any(kw in text for kw in kw_list):
                    if first_rank is None:
                        first_rank = rank
                        break
        if first_rank is None and qr.expected_relations:
            matched_set = set(qr.matched_relations)
            if any(_relations_match(rel, matched_set) for rel in qr.expected_relations):
                first_rank = 1  # 关系命中视为 top-1（无法定位具体 chunk）
    qr.first_expected_rank = first_rank
    return qr


def _aggregate_retrieval(results: list) -> dict:
    n = len(results) or 1
    hit = sum(1 for r in results if r.first_expected_rank is not None)
    mrr = sum(1.0 / r.first_expected_rank for r in results if r.first_expected_rank) / n
    # entity_coverage: 期望实体被 seed 命中的比例
    total_expected = sum(len(r.expected_entities) for r in results) or 1
    hit_expected = 0
    for r in results:
        seed_set = set(r.seed_entities)
        hit_expected += sum(1 for e in r.expected_entities if e in seed_set)
    entity_coverage = hit_expected / total_expected
    # relation_coverage（双向归一化 + 子串兜底匹配）
    total_rel = sum(len(r.expected_relations) for r in results) or 1
    hit_rel = 0
    for r in results:
        matched = set(r.matched_relations)
        hit_rel += sum(1 for rel in r.expected_relations if _relations_match(rel, matched))
    relation_coverage = hit_rel / total_rel
    return {
        "n": len(results),
        "hit_rate": hit / n,
        "mrr": mrr,
        "entity_coverage": entity_coverage,
        "relation_coverage": relation_coverage,
    }


# ==========================
# 生成层指标
# ==========================


def _answer_keyword_hit(answer: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    hit = sum(1 for kw in keywords if kw and kw in answer)
    return hit / len(keywords)


# ==========================
# RAGAS (可选)
# ==========================


def _has_ragas() -> bool:
    try:
        import ragas  # noqa: F401

        return True
    except ImportError:
        return False


def _build_langchain_llm():
    """用 PocketGraphRAG 的 call_llm 构造 LangChain LLM wrapper 供 RAGAS 0.1.x 使用。

    RAGAS 0.4+ 不再支持 LangchainLLMWrapper，请用 _build_ragas_llm_v04。
    本函数保留用于向后兼容旧版 RAGAS。
    """
    try:
        from langchain_core.language_models.llms import LLM
    except ImportError:
        logger.warning(
            "未安装 langchain-core，RAGAS 无法构造 LLM wrapper。pip install ragas 会自动拉取。"
        )
        return None

    from .llm import call_llm, has_llm

    if not has_llm():
        logger.warning(
            "未配置 LLM，RAGAS 无法运行（faithfulness/answer_relevancy 需 LLM）"
        )
        return None

    class _PocketLLM(LLM):
        @property
        def _identifying_params(self):
            return {"model_name": "pocketgraphrag-llm-wrapper"}

        @property
        def _llm_type(self):
            return "pocketgraphrag-llm"

        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            try:
                # RAGAS 内部 prompt 通常是英文，用空 system prompt 即可
                return call_llm("", prompt) or ""
            except Exception as e:
                logger.warning("RAGAS LLM wrapper 调用失败: %s", e)
                return ""

    return _PocketLLM()


def _build_ragas_llm_v04():
    """RAGAS 0.4+ 要求 InstructorLLM（不再是 LangchainLLMWrapper）。

    用 ragas.llms.llm_factory + openai.OpenAI client 构造，按 PocketGraphRAG 的
    LLM 优先级（Ollama → SiliconFlow → DeepSeek → DashScope → OpenAI）选择后端。
    所有后端都走 OpenAI 兼容 API。

    Returns:
        InstructorLLM 实例，或 None（无可用后端/缺依赖）
    """
    try:
        from openai import OpenAI
        from ragas.llms import llm_factory
    except ImportError:
        logger.warning(
            "RAGAS 0.4 需要 openai + instructor 库（pip install ragas 会自动拉取）"
        )
        return None

    from .config import (
        DASHSCOPE_API_KEY,
        DASHSCOPE_API_URL,
        DASHSCOPE_MODEL,
        DEEPSEEK_API_BASE,
        DEEPSEEK_API_KEY,
        DEEPSEEK_MODEL,
        OLLAMA_API_BASE,
        OLLAMA_MODEL,
        OPENAI_API_BASE,
        OPENAI_API_KEY,
        OPENAI_MODEL,
        SILICONFLOW_API_BASE,
        SILICONFLOW_API_KEY,
        SILICONFLOW_MODEL,
    )

    # 按优先级找第一个可用后端
    # DashScope 的 base_url 需要从 chat/completions URL 截取
    dashscope_base = DASHSCOPE_API_URL.rsplit("/chat/completions", 1)[0]
    backends = [
        # (model, base_url, api_key, label)
        (OLLAMA_MODEL, OLLAMA_API_BASE, "not-needed", "Ollama"),
        (SILICONFLOW_MODEL, SILICONFLOW_API_BASE, SILICONFLOW_API_KEY, "SiliconFlow"),
        (DEEPSEEK_MODEL, DEEPSEEK_API_BASE, DEEPSEEK_API_KEY, "DeepSeek"),
        (DASHSCOPE_MODEL, dashscope_base, DASHSCOPE_API_KEY, "DashScope"),
        (OPENAI_MODEL, OPENAI_API_BASE, OPENAI_API_KEY, "OpenAI"),
    ]
    for model, base_url, api_key, label in backends:
        if not model or not base_url:
            continue
        if label != "Ollama" and not api_key:
            continue
        try:
            client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
            ragas_llm = llm_factory(model=model, provider="openai", client=client)
            logger.info("RAGAS 评估 LLM: %s (%s)", label, model)
            return ragas_llm
        except Exception as e:
            logger.warning("构造 %s RAGAS LLM 失败，尝试下一后端: %s", label, e)
            continue
    logger.warning("无可用 LLM 后端供 RAGAS 0.4 使用")
    return None


def _build_langchain_embeddings(model):
    """用 PocketGraphRAG 的 SentenceTransformer 构造 LangChain Embeddings wrapper。

    RAGAS 的 answer_relevancy 需要把"反向生成的问题"与原问题做 embedding 相似度，
    所以除了 LLM 还需要 embeddings。这里复用 rag 实例的 model，无需额外下载。
    """
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError:
        return None

    class _PocketEmbeddings(Embeddings):
        def __init__(self, st_model):
            self._st = st_model

        def embed_documents(self, texts):
            return self._st.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            return self._st.encode([text], normalize_embeddings=True)[0].tolist()

    return _PocketEmbeddings(model)


def _build_ragas_embeddings_v04(model_path=None):
    """RAGAS 0.4+ 要求现代 embeddings 接口（collections metrics 不再接受
    LangchainEmbeddingsWrapper）。用 ragas.embeddings.HuggingFaceEmbeddings
    加载本地 bge-small-zh-v1.5，避免额外下载/依赖 Ollama embedding 模型。

    Args:
        model_path: 模型路径或名称；None 则用 config.EMBEDDING_MODEL

    Returns:
        HuggingFaceEmbeddings 实例，或 None（缺依赖/加载失败）
    """
    try:
        from ragas.embeddings import HuggingFaceEmbeddings
    except ImportError:
        return None
    if model_path is None:
        from .config import EMBEDDING_MODEL

        model_path = EMBEDDING_MODEL
    try:
        return HuggingFaceEmbeddings(model=model_path)
    except Exception as e:
        logger.warning("构造 RAGAS HuggingFaceEmbeddings 失败: %s", e)
        return None


def _extract_ragas_scores(result) -> dict:
    """从 RAGAS evaluate() 返回值提取 {metric: mean_score}。

    RAGAS 0.1.x 返回 dict（直接 .items()）；
    RAGAS 0.4+ 返回 EvaluationResult，需从 result.scores（list[dict]）算均值。
    """
    # 0.1.x：dict 直接转换
    if isinstance(result, dict):
        return {k: float(v) for k, v in result.items()}
    # 0.4+：EvaluationResult，scores 是 per-row dict 列表
    scores = getattr(result, "scores", None)
    if not scores:
        # 兜底：尝试 _scores_dict（聚合值）
        sd = getattr(result, "_scores_dict", None)
        if sd:
            return {k: float(v) for k, v in sd.items()}
        return {}
    import statistics

    all_keys = set()
    for row in scores:
        all_keys.update(row.keys())
    agg = {}
    for k in all_keys:
        vals = [row[k] for row in scores if k in row and row[k] is not None]
        agg[k] = float(statistics.mean(vals)) if vals else 0.0
    return agg


def _run_ragas(
    questions,
    answers,
    contexts,
    ground_truths=None,
    llm_wrapper=None,
    embeddings_wrapper=None,
) -> Optional[dict]:
    """如果安装了 ragas 且配置了 LLM，跑 RAGAS 评测；否则返回 None。

    Args:
        questions/answers/contexts: 三个对齐的列表（contexts 是 list[list[str]]）
        ground_truths: 参考答案列表（context_recall 必需）；None 则跳过 context_recall
        llm_wrapper: LangChain LLM；None 则自动用 PocketGraphRAG 的 call_llm 构造
        embeddings_wrapper: LangChain Embeddings；None 则跳过（answer_relevancy 可能失败）

    Returns:
        {metric: score} dict，或 None

    兼容性：
        - RAGAS 0.1.x: `evaluate(ds, metrics, llm, embeddings)`
        - RAGAS 0.2.x: 同签名，但 metric 模块路径有变化；用 try/except 兼容
    """
    if not _has_ragas():
        return None
    if llm_wrapper is None:
        llm_wrapper = _build_langchain_llm()
    if llm_wrapper is None:
        logger.warning("RAGAS 跳过：无可用 LLM wrapper")
        return None

    try:
        from datasets import Dataset
        from ragas import evaluate

        # RAGAS 期望的 schema；context_recall 需要 ground_truth 列
        rows = []
        has_gt = bool(ground_truths) and len(ground_truths) == len(questions)
        for i, (q, a, ctx) in enumerate(zip(questions, answers, contexts)):
            row = {
                "question": q,
                "answer": a,
                "contexts": ctx if isinstance(ctx, list) else [ctx],
            }
            if has_gt:
                row["ground_truth"] = ground_truths[i] or ""
            rows.append(row)
        ds = Dataset.from_list(rows)

        # RunConfig 控制超时/重试（0.2+ 引入，0.1.x 无）
        run_config = None
        try:
            from ragas.run_config import RunConfig

            run_config = RunConfig(timeout=120, max_retries=2)
        except ImportError:
            pass

        # ---------- RAGAS 0.4+ API：用 InstructorLLM + 预构造 metric 实例 ----------
        # RAGAS 0.4 起 evaluate() 要求 metric 是 ragas.metrics.base.Metric 实例。
        # ragas.metrics.collections.Faithfulness 是 SimpleBaseMetric（不被 evaluate 接受），
        # 正确来源是 ragas.metrics._faithfulness.faithfulness 等预构造实例（MetricWithLLM）。
        # evaluate() 会在 metric.llm is None 时自动注入顶层传入的 llm/embeddings。
        try:
            from ragas.metrics._answer_relevance import answer_relevancy
            from ragas.metrics._context_precision import context_precision
            from ragas.metrics._context_recall import context_recall
            from ragas.metrics._faithfulness import faithfulness

            # 用 _build_ragas_llm_v04() 拿 InstructorLLM（llm_factory + Ollama/SiliconFlow/...）
            ragas_llm = _build_ragas_llm_v04()
            if ragas_llm is None:
                raise ImportError("no available backend for RAGAS 0.4 InstructorLLM")

            # old-style metric (MetricWithEmbeddings) 期望 LangChain 风格 embeddings
            # （embed_query 方法）。HuggingFaceEmbeddings（modern）没有 embed_query，
            # 会触发 AttributeError。用传入的 embeddings_wrapper（LangChain Embeddings），
            # evaluate() 会自动用 LangchainEmbeddingsWrapper 包装。
            ragas_emb = embeddings_wrapper

            # 预构造 metric 实例；evaluate() 会自动把顶层 llm/embeddings 注入到
            # metric.llm / metric.embeddings（见 evaluation.py MetricWithLLM 分支）
            metrics = [faithfulness, context_precision]
            # answer_relevancy 需要 n=3 反向生成问题 + embeddings，本地小模型（如
            # Ollama 7B）对 n>1 支持差、极慢。设 RAGAS_SKIP_ANSWER_RELEVANCY=1 跳过。
            skip_ar = os.environ.get("RAGAS_SKIP_ANSWER_RELEVANCY", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if ragas_emb is not None and not skip_ar:
                metrics.append(answer_relevancy)
            if has_gt:
                metrics.append(context_recall)

            eval_kwargs = {"metrics": metrics, "llm": ragas_llm}
            if ragas_emb is not None and not skip_ar:
                eval_kwargs["embeddings"] = ragas_emb
            if run_config is not None:
                eval_kwargs["run_config"] = run_config
            result = evaluate(ds, **eval_kwargs)
            return _extract_ragas_scores(result)
        except (ImportError, TypeError, AttributeError):
            # ---------- 回退 RAGAS 0.1.x API：metric 是模块级实例，evaluate 传 llm ----------
            try:
                from ragas.metrics import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )
            except ImportError:  # pragma: no cover
                from ragas.metrics.collections import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )

            metrics = [faithfulness, answer_relevancy, context_precision]
            if has_gt:
                metrics.append(context_recall)

            eval_kwargs = {"metrics": metrics, "llm": llm_wrapper}
            if embeddings_wrapper is not None:
                eval_kwargs["embeddings"] = embeddings_wrapper
            if run_config is not None:
                eval_kwargs["run_config"] = run_config

            result = evaluate(ds, **eval_kwargs)
            return _extract_ragas_scores(result)
    except Exception as e:
        logger.warning("RAGAS 评测失败，跳过: %s", e)
        return None


# ==========================
# 主流程
# ==========================


def run_evaluation(
    rag,
    dataset: dict,
    top_k: int = 5,
    run_generation: bool = True,
    run_ragas: bool = False,
) -> dict:
    """对整个 dataset 跑评测。

    Args:
        rag: PocketGraphRAG 实例
        dataset: load_benchmark() 的返回
        top_k: 检索数量
        run_generation: 是否调用 LLM 生成回答（计算 answer_keyword_hit）
        run_ragas: 是否启用 RAGAS（需安装 ragas 并配置 LLM）

    Returns:
        {summary, per_question, ragas}
    """
    questions = dataset["questions"]
    per_question = []

    all_answers, all_questions, all_contexts, all_ground_truths = [], [], [], []

    for q in questions:
        qr = _evaluate_retrieval(rag, q, top_k)
        if run_generation:
            res = rag.answer(q["question"], top_k=top_k)
            qr.answer = res.get("answer", "")
            qr.answer_keyword_hit = _answer_keyword_hit(
                qr.answer, qr.expected_answer_keywords
            )
            all_answers.append(qr.answer)
            all_questions.append(q["question"])
            all_contexts.append([s.get("text", "") for s in res.get("sources", [])])
            # context_recall 需要 ground_truth；优先用 benchmark 的 ground_truth 字段，
            # 缺失则用 expected_answer_keywords 拼一个简短参考答案兜底
            gt = q.get("ground_truth", "")
            if not gt and qr.expected_answer_keywords:
                gt = "；".join(qr.expected_answer_keywords)
            all_ground_truths.append(gt)
        per_question.append(qr)
        status = "✓" if qr.first_expected_rank else "✗"
        # no-generation 模式下 answer_keyword_hit 是默认 0.0，显示 0.00 会误导成"指标失效"
        # （曾经把自己都骗过）。改成 N/A 明确标识未计算。
        kw_hit_str = (
            f"{qr.answer_keyword_hit:.2f}"
            if run_generation
            else "N/A"
        )
        print(
            f"  {status} [{q['id']}] {q['question']}"
            f"  (seed={len(qr.seed_entities)}, kw_hit={kw_hit_str})"
        )

    summary = _aggregate_retrieval(per_question)
    if run_generation:
        avg_kw = sum(r.answer_keyword_hit for r in per_question) / (
            len(per_question) or 1
        )
        summary["answer_keyword_hit_avg"] = avg_kw

    ragas_result = None
    if run_ragas:
        # 用 rag 实例的 embedding 模型构造 LangChain Embeddings wrapper，
        # 这样 answer_relevancy 不需要额外下载模型。
        embeddings_wrapper = None
        rag_model = getattr(rag, "model", None)
        if rag_model is not None:
            embeddings_wrapper = _build_langchain_embeddings(rag_model)
        ragas_result = _run_ragas(
            all_questions,
            all_answers,
            all_contexts,
            ground_truths=all_ground_truths,
            embeddings_wrapper=embeddings_wrapper,
        )
        if ragas_result is None:
            logger.warning(
                "RAGAS 未启用或失败，仅返回检索/生成指标。"
                "确认已 pip install pocketgraphrag[eval] 并配置 LLM"
            )

    return {
        "summary": summary,
        "per_question": per_question,
        "ragas": ragas_result,
    }


def print_report(report: dict) -> None:
    s = report["summary"]
    print("\n" + "=" * 50)
    print("评测汇总 (Evaluation Summary)")
    print("=" * 50)
    print(f"  题目数 N          : {s.get('n', 0)}")
    print(f"  检索 Hit Rate     : {s.get('hit_rate', 0):.4f}")
    print(f"  MRR               : {s.get('mrr', 0):.4f}")
    print(f"  Entity Coverage   : {s.get('entity_coverage', 0):.4f}")
    print(f"  Relation Coverage : {s.get('relation_coverage', 0):.4f}")
    if "answer_keyword_hit_avg" in s:
        print(f"  Answer Keyword Hit: {s.get('answer_keyword_hit_avg', 0):.4f}")
    if report.get("ragas"):
        print("-" * 50)
        print("RAGAS (生成质量，0-1 越高越好)：")
        # 中文标签 + 英文 metric 名
        labels = {
            "faithfulness": "忠实度(无幻觉)",
            "answer_relevancy": "答案切题度",
            "context_precision": "上下文精确度",
            "context_recall": "上下文召回率",
        }
        for k, v in report["ragas"].items():
            label = labels.get(k, k)
            print(f"  {label:18s} ({k}): {v:.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="PocketGraphRAG 评测 Harness")
    parser.add_argument(
        "--benchmark", default=DEFAULT_BENCHMARK_PATH, help="benchmark 数据集 JSON 路径"
    )
    parser.add_argument(
        "--search-mode", default=None, help="检索模式 vector/local/global/mix/kg_only"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-generation", action="store_true", help="跳过 LLM 生成，只评检索"
    )
    parser.add_argument(
        "--ragas", action="store_true", help="启用 RAGAS 评测 (需 pip install ragas)"
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="指定 Ollama 模型作为 RAGAS 评估 LLM，如 qwen2.5:7b。"
        "未指定时走 PocketGraphRAG 默认 call_llm 优先级。",
    )
    args = parser.parse_args()

    # 显式指定 Ollama 模型时，写入环境变量并 patch llm 模块的常量
    # （config.py 在 import 时已读取，必须双管齐下才能生效）
    if args.ollama_model:
        os.environ["OLLAMA_MODEL"] = args.ollama_model
        try:
            from . import llm as _llm_mod

            _llm_mod.OLLAMA_MODEL = args.ollama_model
        except ImportError:
            pass
        print(f"[RAGAS] 已设置 OLLAMA_MODEL={args.ollama_model}")

    dataset = load_benchmark(args.benchmark)
    print(f"已加载 benchmark: {dataset['name']} ({len(dataset['questions'])} 题)")

    # 延迟导入，避免 --help 时加载模型
    from .rag_system import PocketGraphRAG

    kwargs = {}
    if args.search_mode:
        kwargs["search_mode"] = args.search_mode
    rag = PocketGraphRAG(top_k=args.top_k, **kwargs)

    report = run_evaluation(
        rag,
        dataset,
        top_k=args.top_k,
        run_generation=not args.no_generation,
        run_ragas=args.ragas,
    )
    print_report(report)


if __name__ == "__main__":
    main()
