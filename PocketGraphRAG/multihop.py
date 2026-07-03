"""
多跳查询分解模块

对于复杂问题（如"三环唑可以防治哪些病害，各自的用量是多少？"），
单轮检索可能无法覆盖所有相关信息。本模块将复杂查询分解为多个子查询，
分别检索后合并结果，提升多跳问题的回答质量。

策略：
  1. 使用 LLM 将复杂查询分解为 2-4 个子查询
  2. 对每个子查询独立检索
  3. 合并去重所有检索结果
  4. 将完整上下文交给 LLM 生成最终回答
"""

import json

from .logging_config import get_logger

logger = get_logger(__name__)
from typing import List, Tuple

from .llm import call_llm

DECOMPOSE_PROMPT = """你是一个查询分解专家。请将用户的复杂问题分解为多个简单的子查询。

规则：
1. 如果问题已经很简单（只涉及一个实体/一个事实），直接返回原问题
2. 如果问题涉及多个方面或多个实体，分解为 2-4 个子查询
3. 每个子查询应该是独立可检索的
4. 返回 JSON 格式：{{"sub_queries": ["子查询1", "子查询2", ...]}}

示例：
问题："三环唑可以防治哪些病害，各自的用量是多少？"
回答：{{"sub_queries": ["三环唑可以防治哪些病害", "三环唑的用量"]}}

问题："盗梦空间和星际穿越的导演风格有什么区别？各自的主题是什么？"
回答：{{"sub_queries": ["盗梦空间导演风格", "星际穿越导演风格", "盗梦空间主题", "星际穿越主题"]}}

请分解以下问题：
{query}

只返回 JSON，不要其他内容："""


def decompose_query(query: str) -> List[str]:
    """调用 LLM 分解查询，使用统一 LLM 调用层"""
    prompt = DECOMPOSE_PROMPT.format(query=query)

    result = call_llm(
        "你是查询分解专家，只返回 JSON 格式的分解结果。",
        prompt,
        temperature=0.0,
        max_tokens=500,
    )

    if result:
        # 清洗可能包含的 Markdown 标记
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:-3].strip()
        elif result.startswith("```"):
            result = result[3:-3].strip()

        try:
            parsed = json.loads(result)
            return parsed.get("sub_queries", [query])
        except json.JSONDecodeError as e:
            logger.debug("多跳查询分解 JSON 解析失败，回退为原查询: %s", e)

    # LLM 不可用或解析失败时直接返回原查询
    return [query]


def multi_hop_retrieve(retrieve_fn, query: str, top_k: int = 5) -> List[Tuple]:
    """
    多跳检索：分解查询 -> 分别检索 -> 合并去重

    Args:
        retrieve_fn: 检索函数，接受 (query, top_k) 返回 [(text, score, meta), ...]
        query: 用户原始查询
        top_k: 每个子查询的检索数量

    Returns:
        合并去重后的检索结果
    """
    sub_queries = decompose_query(query)

    # 如果只有一个子查询（简单问题），直接检索
    if len(sub_queries) <= 1:
        return retrieve_fn(query, top_k)

    # 对每个子查询分别检索
    all_results = []
    seen_entities = set()

    for sq in sub_queries:
        results = retrieve_fn(sq, top_k)
        for text, score, meta in results:
            entity = meta.get("entity", "")
            if entity not in seen_entities:
                seen_entities.add(entity)
                all_results.append((text, score, meta))

    # 按分数排序
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[: top_k * 2]  # 返回更多结果供 LLM 使用
