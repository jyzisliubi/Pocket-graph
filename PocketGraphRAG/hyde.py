"""HyDE (Hypothetical Document Embeddings) 查询改写。

核心思想（Gao et al. 2022）：
  用户的短查询（如"稻瘟病怎么治"）与文档库中的长段落，在向量空间中语义距离往往较大，
  导致召回率低。HyDE 先让 LLM 针对问题"假想"一段答案文档，再用这段假设性文档去检索——
  因为假想文档与真实文档都是完整段落，语义空间更接近，匹配度显著提升。

适用边界：
  - 只对向量检索（vector / mix 模式的 vector 部分）有效
  - KG 实体/关系匹配仍用原 query（假想文档会引入噪声实体，降低 KG 命中）
  - multihop 模式已用 LLM 做查询分解，不再叠加 HyDE（避免多次 LLM 调用拖慢）
  - 无 LLM 时回退为原 query

参考：
  - Precise Zero-Shot Dense Retrieval without Relevance Labels, arXiv:2212.10496
  - LightRAG / LangChain 的 HyDE 实现
"""

from __future__ import annotations

from typing import Optional

from .llm import call_llm, has_llm
from .logging_config import get_logger

logger = get_logger(__name__)

HYDE_PROMPT = """请针对下面的问题，写一段 150-250 字的假设性答案段落。
要求：
- 像真实知识库文档一样写，用陈述句陈述事实，不要写"我认为"等主观语气
- 即使你不确定答案，也要写一段看起来合理的文档（这就是 HyDE 的关键：用假设性文档去检索真实文档）
- 不要分点，写成一段连贯的文字
- 语言与问题一致

问题：{question}

假设性答案文档："""


def generate_hypothetical_document(query: str) -> Optional[str]:
    """用 LLM 生成假设性答案文档。

    Args:
        query: 用户原始查询

    Returns:
        假设性文档文本；无 LLM 或调用失败时返回 None（调用方应回退为原 query）
    """
    if not has_llm():
        return None
    try:
        doc = call_llm(
            "你是知识库文档生成器，根据问题生成一段假设性答案文档用于检索增强。",
            HYDE_PROMPT.format(question=query),
            temperature=0.7,  # 略高温度增加多样性，让假想文档覆盖更多语义
            max_tokens=400,
        )
        if doc and len(doc.strip()) >= 20:
            logger.debug(
                "HyDE 生成假设性文档 (%s 字): %s...",
                len(doc),
                doc[:50].replace("\n", " "),
            )
            return doc.strip()
        logger.debug("HyDE 返回过短或为空，回退原查询")
        return None
    except Exception as e:
        logger.warning("HyDE 生成失败，回退原查询: %s", e)
        return None
