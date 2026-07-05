"""DRIFT Search: Dynamic Reasoning and Inference with Flexible Traversal

对标微软 GraphRAG 的 DRIFT 搜索策略，结合全局社区搜索和局部 KG 检索的
迭代细化能力，用于回答复杂多跳问题。

核心流程（三阶段）：
1. Primer 阶段：HyDE 生成假设性答案 → 召回相关社区摘要 → 生成中间答案 + N 个后续查询
2. Drift 阶段：后续查询并行执行局部检索（KG 实体 + 关系 + 文本块）→ 可迭代 K 轮
3. Output 阶段：汇总所有中间答案和局部证据 → 融合生成最终答案

与现有 multihop 的区别：
- multihop 是一次性查询分解，不基于中间结果迭代
- DRIFT 基于中间答案动态生成后续查询，迭代细化，能挖更深的多跳链路
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .llm import call_llm, has_llm
from .logging_config import get_logger

logger = get_logger(__name__)


# 默认配置（可通过环境变量覆盖）
DRIFT_MAX_ITERATIONS = 2
DRIFT_FOLLOWUP_PER_ITER = 3
DRIFT_LOCAL_TOP_K = 3
DRIFT_PRIMER_COMMUNITY_TOP_K = 3


PRIMER_PROMPT = """你是知识图谱问答专家。基于以下社区摘要和用户问题，完成两件事：

## 社区摘要
{community_context}

## 用户问题
{query}

## 任务
1. 基于社区摘要给出一个初步答案（即使信息不全也要尝试）
2. 生成 {n_followup} 个后续查询，用于在知识图谱中深挖细节

后续查询要求：
- 每个查询应该是简洁的事实型问题（如 "X 的导演是谁"）
- 针对初步答案中缺失的信息
- 覆盖问题的不同子方面

只输出 JSON，格式：
{{
  "intermediate_answer": "初步答案文本",
  "followup_queries": ["后续查询1", "后续查询2"]
}}
"""

DRIFT_ITERATION_PROMPT = """你是知识图谱问答专家。基于已收集的证据，继续深挖用户原始问题。

## 用户原始问题
{query}

## 当前已收集的证据
{evidence_context}

## 上一轮的中间答案
{intermediate_answer}

## 任务
1. 基于新证据更新中间答案
2. 如果仍有缺失信息，生成最多 {n_followup} 个新的后续查询；信息已足够则返回空列表

只输出 JSON，格式：
{{
  "intermediate_answer": "更新后的答案文本",
  "followup_queries": ["新查询1"]
}}
"""


@dataclass
class DriftIteration:
    """单轮 DRIFT 迭代的中间结果"""

    iteration: int
    intermediate_answer: str
    followup_queries: List[str] = field(default_factory=list)
    retrieved_chunks: List[Any] = field(default_factory=list)
    seed_entities: List[str] = field(default_factory=list)


@dataclass
class DriftResult:
    """DRIFT 搜索的完整结果"""

    primer_answer: str
    final_answer: str
    iterations: List[DriftIteration] = field(default_factory=list)
    all_chunks: List[Any] = field(default_factory=list)
    all_entities: List[str] = field(default_factory=list)

    @property
    def total_iterations(self) -> int:
        return len(self.iterations)


def _parse_json_response(text: Optional[str]) -> Optional[dict]:
    """解析 LLM 返回的 JSON（兼容 markdown 代码块包裹）"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("DRIFT JSON 解析失败: %s", e)
        return None


def drift_primer(
    query: str,
    community_search_fn: Callable[[str, int], List[dict]],
    top_k_communities: int = DRIFT_PRIMER_COMMUNITY_TOP_K,
    n_followup: int = DRIFT_FOLLOWUP_PER_ITER,
) -> Tuple[str, List[str], List[dict]]:
    """DRIFT Primer 阶段：社区搜索 + 中间答案生成

    Args:
        query: 用户原始查询
        community_search_fn: 社区摘要检索函数 (query, top_k) -> [{"summary": "..."}]
        top_k_communities: 召回的社区数量
        n_followup: 生成的后续查询数量

    Returns:
        (intermediate_answer, followup_queries, communities)
        无 LLM 时 followup_queries 为空，调用方降级到 multihop
    """
    communities = community_search_fn(query, top_k_communities)
    community_context = "\n\n".join(
        f"[社区 {i + 1}] {c.get('summary', c.get('text', ''))}"
        for i, c in enumerate(communities)
    )
    if not community_context.strip():
        community_context = "（无相关社区摘要）"

    if not has_llm():
        logger.info("DRIFT Primer: 无 LLM，降级到 multihop")
        return "", [], communities

    prompt = PRIMER_PROMPT.format(
        community_context=community_context,
        query=query,
        n_followup=n_followup,
    )
    response = call_llm(
        "你是知识图谱问答专家，只返回 JSON 格式结果。",
        prompt,
        temperature=0.1,
        max_tokens=800,
        role="query",
    )
    data = _parse_json_response(response)
    if not data:
        logger.warning("DRIFT Primer: LLM 响应解析失败")
        return "", [], communities

    intermediate_answer = str(data.get("intermediate_answer", "")).strip()
    followup_queries = [
        str(q).strip() for q in data.get("followup_queries", []) if str(q).strip()
    ][:n_followup]

    logger.info(
        "DRIFT Primer: 生成 %d 个后续查询, 中间答案 %d 字",
        len(followup_queries),
        len(intermediate_answer),
    )
    return intermediate_answer, followup_queries, communities


def drift_iteration(
    query: str,
    followup_queries: List[str],
    local_retrieve_fn: Callable[[str, int], Tuple[List[Any], dict]],
    intermediate_answer: str,
    iteration: int,
    n_followup: int = DRIFT_FOLLOWUP_PER_ITER,
    local_top_k: int = DRIFT_LOCAL_TOP_K,
) -> DriftIteration:
    """DRIFT 单轮迭代：执行后续查询 + 更新中间答案 + 生成新查询"""
    iter_result = DriftIteration(
        iteration=iteration,
        intermediate_answer=intermediate_answer,
        followup_queries=[],
        retrieved_chunks=[],
        seed_entities=[],
    )

    seen_chunk_keys = set()
    evidence_parts: List[str] = []

    for fq in followup_queries:
        chunks, kg_info = local_retrieve_fn(fq, local_top_k)
        for chunk in chunks:
            chunk_text = chunk[0] if isinstance(chunk, (list, tuple)) else str(chunk)
            key = chunk_text[:100] if isinstance(chunk_text, str) else ""
            if key and key not in seen_chunk_keys:
                seen_chunk_keys.add(key)
                iter_result.retrieved_chunks.append(chunk)
                evidence_parts.append(f"[{fq}] {chunk_text[:200]}")

        if isinstance(kg_info, dict):
            iter_result.seed_entities.extend(kg_info.get("seed_entities", []))

    if not has_llm() or not evidence_parts:
        return iter_result

    evidence_context = "\n".join(evidence_parts[:10])
    prompt = DRIFT_ITERATION_PROMPT.format(
        query=query,
        evidence_context=evidence_context,
        intermediate_answer=intermediate_answer or "（暂无）",
        n_followup=n_followup,
    )
    response = call_llm(
        "你是知识图谱问答专家，只返回 JSON 格式结果。",
        prompt,
        temperature=0.1,
        max_tokens=800,
        role="query",
    )
    data = _parse_json_response(response)
    if data:
        iter_result.intermediate_answer = str(
            data.get("intermediate_answer", intermediate_answer)
        ).strip()
        iter_result.followup_queries = [
            str(q).strip()
            for q in data.get("followup_queries", [])
            if str(q).strip()
        ][:n_followup]

    logger.info(
        "DRIFT 迭代 %d: 收集 %d 个文本块, 生成 %d 个新查询",
        iteration,
        len(iter_result.retrieved_chunks),
        len(iter_result.followup_queries),
    )
    return iter_result


def drift_search(
    query: str,
    community_search_fn: Callable[[str, int], List[dict]],
    local_retrieve_fn: Callable[[str, int], Tuple[List[Any], dict]],
    max_iterations: int = DRIFT_MAX_ITERATIONS,
    n_followup: int = DRIFT_FOLLOWUP_PER_ITER,
    primer_top_k: int = DRIFT_PRIMER_COMMUNITY_TOP_K,
    local_top_k: int = DRIFT_LOCAL_TOP_K,
) -> DriftResult:
    """DRIFT 搜索主入口（三阶段：Primer → Drift 迭代 → Output）

    Args:
        query: 用户原始查询
        community_search_fn: 社区摘要检索函数 (query, top_k) -> [{"summary": "..."}]
        local_retrieve_fn: 局部检索函数 (query, top_k) -> (chunks, kg_info)
        max_iterations: 最大迭代轮数
        n_followup: 每轮后续查询数量上限
        primer_top_k: Primer 阶段社区召回数量
        local_top_k: 每个后续查询的文本块检索数量

    Returns:
        DriftResult: 包含最终答案、所有迭代详情、所有收集的文本块
    """
    # Phase 1: Primer
    primer_answer, followup_queries, communities = drift_primer(
        query=query,
        community_search_fn=community_search_fn,
        top_k_communities=primer_top_k,
        n_followup=n_followup,
    )

    result = DriftResult(
        primer_answer=primer_answer,
        final_answer=primer_answer,
        iterations=[],
        all_chunks=[],
        all_entities=[],
    )

    if not followup_queries:
        result.final_answer = primer_answer or ""
        return result

    # Phase 2: Drift 迭代
    current_answer = primer_answer
    current_queries = followup_queries

    for i in range(1, max_iterations + 1):
        if not current_queries:
            break

        iter_result = drift_iteration(
            query=query,
            followup_queries=current_queries,
            local_retrieve_fn=local_retrieve_fn,
            intermediate_answer=current_answer,
            iteration=i,
            n_followup=n_followup,
            local_top_k=local_top_k,
        )

        result.iterations.append(iter_result)
        result.all_chunks.extend(iter_result.retrieved_chunks)
        result.all_entities.extend(iter_result.seed_entities)

        if iter_result.intermediate_answer:
            current_answer = iter_result.intermediate_answer
        current_queries = iter_result.followup_queries

    # Phase 3: Output
    result.final_answer = current_answer
    result.all_entities = list(set(result.all_entities))
    # 去重 chunks
    seen = set()
    unique_chunks = []
    for chunk in result.all_chunks:
        chunk_text = chunk[0] if isinstance(chunk, (list, tuple)) else str(chunk)
        key = chunk_text[:100] if isinstance(chunk_text, str) else ""
        if key and key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    result.all_chunks = unique_chunks

    logger.info(
        "DRIFT 完成: %d 轮迭代, %d 个文本块, %d 个实体",
        result.total_iterations,
        len(result.all_chunks),
        len(result.all_entities),
    )
    return result
