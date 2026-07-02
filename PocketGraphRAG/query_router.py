"""查询路由器：自动选择最优检索模式。

核心动机：
  用户不知道该选 vector / local / global / mix / global_summary 哪个模式。
  但不同问题类型对检索模式有强偏好：
    - "稻瘟病有什么症状" → local（实体中心，BFS 邻域找症状实体）
    - "对比三环唑和稻瘟灵的效果" → mix（多实体 + 比较，需向量+KG）
    - "水稻病害防治体系概述" → global_summary（归纳，走社区摘要）
    - "稻瘟病是什么" → vector（简单事实，向量足够）
    - "稻瘟病和纹枯病的关系" → mix（跨实体关系）

  用一次轻量 LLM 调用做分类（temperature=0 保证稳定），无 LLM 时回退默认模式。

设计要点：
  - 分类缓存（同 query 不重复调用 LLM）
  - 失败/超时回退到调用方传入的 default_mode
  - 输出严格枚举，下游无需 try/except
"""

from __future__ import annotations

from typing import Optional

from .llm import call_llm, has_llm
from .logging_config import get_logger

logger = get_logger(__name__)

# 支持的检索模式（与 rag_system._basic_retrieve 分支保持一致）
SUPPORTED_MODES = ("vector", "local", "global", "mix", "kg_only", "global_summary")

ROUTER_PROMPT = """你是一个检索模式分类器。根据用户问题，判断应该用哪种知识库检索模式。

可选模式：
- vector：纯向量检索。适合简单事实查询（"X 是什么"、"X 的定义"）
- local：KG 实体匹配 + BFS 邻域扩展。适合围绕单一实体的问题（"X 有什么症状"、"X 的特征"、"X 的防治方法"）
- global：KG 关系关键词匹配。适合从关系出发的问题（"哪些病害会导致倒伏"）
- mix：向量 + KG 融合。适合多实体、比较、跨关系问题（"对比 A 和 B"、"A 和 B 的关系"、"A 如何影响 B"）
- global_summary：社区摘要归纳。适合宏观概述、体系性问题（"X 的整体防治体系"、"X 领域概览"）

判断依据（按优先级）：
1. 含"对比""比较""和...哪个"→ mix
2. 含"概述""体系""整体""总结""全局"→ global_summary
3. 含"哪些...会导致""哪些...有...特征"（从关系反查实体）→ global
4. 围绕单一实体的属性询问（症状/特征/防治/药剂）→ local
5. 简单定义/事实 → vector

只输出模式名（vector/local/global/mix/global_summary），不要任何解释。

用户问题：{question}

模式："""


class QueryRouter:
    """查询路由器：根据问题类型自动选择检索模式。

    用法：
        router = QueryRouter(default_mode="mix")
        mode = router.route("稻瘟病有什么症状")  # 返回 "local"
    """

    def __init__(self, default_mode: str = "mix"):
        """
        Args:
            default_mode: LLM 不可用或分类失败时的回退模式
        """
        self.default_mode = default_mode if default_mode in SUPPORTED_MODES else "mix"
        # 简单 LRU 缓存：query -> mode，避免重复调用
        self._cache: dict[str, str] = {}
        self._cache_max = 256

    def route(self, query: str) -> str:
        """根据 query 返回推荐检索模式。

        Args:
            query: 用户查询文本

        Returns:
            模式名（vector/local/global/mix/global_summary 等）
        """
        if not query or not query.strip():
            return self.default_mode

        # 1. 缓存命中
        cache_key = query.strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 2. 无 LLM 直接回退
        if not has_llm():
            return self.default_mode

        # 3. LLM 分类
        mode = self._classify(query)
        if mode is None:
            mode = self.default_mode

        # 4. 写入缓存
        if len(self._cache) >= self._cache_max:
            # 简单淘汰：清空一半（LRU 完整实现需要 OrderedDict，这里够用）
            keys = list(self._cache.keys())
            for k in keys[: self._cache_max // 2]:
                del self._cache[k]
        self._cache[cache_key] = mode
        return mode

    def _classify(self, query: str) -> Optional[str]:
        """调用 LLM 做模式分类，返回小写模式名或 None"""
        try:
            result = call_llm(
                "你是检索模式分类器，只输出模式名。",
                ROUTER_PROMPT.format(question=query),
                temperature=0.0,
                max_tokens=20,
            )
            if not result:
                return None
            # 容错：LLM 可能带多余文字，提取模式名
            result = result.strip().lower()
            for m in SUPPORTED_MODES:
                if m in result:
                    logger.debug("查询路由 [%s] -> %s", query[:30], m)
                    return m
            logger.debug("查询路由 [%s] LLM 返回无法解析: %s", query[:30], result)
            return None
        except Exception as e:
            logger.warning("查询路由分类失败，回退默认模式: %s", e)
            return None

    def clear_cache(self):
        """清空路由缓存（切换数据集后调用）"""
        self._cache.clear()
