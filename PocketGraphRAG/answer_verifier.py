"""答案自检器：LLM 生成后校验答案是否被检索到的上下文支持。

核心动机：
  RAG 的最大风险是 LLM 编造知识库没有的内容（幻觉）。
  RAGAS 的 faithfulness 是"事后验尸"——评测时才知道有没有幻觉。
  这里做"实时止血"——生成后立即用一次 LLM 调用校验，发现幻觉就加警告前缀或重试。

设计要点：
  - 只校验"事实性陈述"，不校验常识/推理（LLM 区分得了）
  - 返回 support_level: full(全部支持) / partial(部分编造) / none(完全编造)
  - partial/none 时返回编造的具体句子，供调用方加警告
  - 失败/超时默认 full（不阻断主流程，避免自检拖垮可用性）
  - 无 LLM 直接返回 full

参考：
  - Self-RAG (Asai et al. 2023) 的 reflection token 思想
  - LangChain 的 SelfCheck / LLMChecker
"""

from __future__ import annotations

from .llm import call_llm, has_llm
from .logging_config import get_logger

logger = get_logger(__name__)

VERIFY_PROMPT = """你是答案事实校验员。你的任务是判断回答中的内容是否安全，而不是生硬地卡每个字。

判断原则：
- 区分“硬事实”和“软补白”
  - 硬事实：涉及具体领域知识（如病害名称、药剂、用量、防治方法、时期等）的结论
  - 软补白：连接词、语气词、常识性过渡（如"因此"、"需要注意"、"这类病害通常先观察叶片表现"）、自然句式重组
- 硬事实：必须能在"检索到的知识信息"中找到明确支持，或者是常识
- 软补白：允许使用，只要不编造新的硬事实即可
- 如果答案中存在编造的硬事实，标记为 unsupported

输出 JSON，格式：
{{
  "support_level": "full" 或 "partial" 或 "none",
  "unsupported_sentences": ["被识别为编造的硬事实句子..."],
  "is_common_knowledge": ["属于常识无需知识支持的句子..."]
}}

- full: 所有硬事实都被知识支持或属于常识
- partial: 有部分编造的硬事实
- none: 答案几乎完全编造

只输出 JSON，不要任何解释。

【检索到的知识信息】：
{context}

【答案】：
{answer}

校验结果："""


def verify_answer(answer: str, context: str) -> dict:
    """校验答案是否被上下文支持。

    Args:
        answer: LLM 生成的答案文本
        context: 检索到的知识上下文（_build_context 的输出）

    Returns:
        dict: {
            support_level: "full" | "partial" | "none",
            unsupported_sentences: list[str],  # 编造的句子
            is_common_knowledge: list[str],  # 常识性陈述
        }
        无 LLM 或校验失败时返回 {"support_level": "full", ...}（不阻断主流程）
    """
    if not has_llm():
        return {
            "support_level": "full",
            "unsupported_sentences": [],
            "is_common_knowledge": [],
        }

    if not answer or not answer.strip():
        return {
            "support_level": "full",
            "unsupported_sentences": [],
            "is_common_knowledge": [],
        }

    try:
        import json as _json

        result = call_llm(
            "你是答案事实校验员，只输出 JSON。",
            VERIFY_PROMPT.format(context=context, answer=answer),
            temperature=0.0,  # 校验必须稳定
            max_tokens=500,
        )
        if not result:
            logger.debug("自检返回空，默认通过")
            return {
                "support_level": "full",
                "unsupported_sentences": [],
                "is_common_knowledge": [],
            }

        # 容错：LLM 可能带 markdown 代码块，提取 JSON
        result = result.strip()
        if result.startswith("```"):
            # 去掉 ```json ... ```
            lines = result.split("\n")
            result = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[0]

        parsed = _json.loads(result)
        level = parsed.get("support_level", "full").lower().strip()
        if level not in ("full", "partial", "none"):
            level = "full"

        unsupported = parsed.get("unsupported_sentences", []) or []
        common = parsed.get("is_common_knowledge", []) or []

        logger.info(
            "答案自检: %s（无依据 %s 句，常识 %s 句）",
            level,
            len(unsupported),
            len(common),
        )
        return {
            "support_level": level,
            "unsupported_sentences": unsupported,
            "is_common_knowledge": common,
        }
    except Exception as e:
        logger.warning("答案自检失败，默认通过（不阻断主流程）: %s", e)
        return {
            "support_level": "full",
            "unsupported_sentences": [],
            "is_common_knowledge": [],
        }


def build_warning_prefix(verification: dict) -> str:
    """根据校验结果生成警告前缀。

    Args:
        verification: verify_answer 的返回值

    Returns:
        前缀字符串（可能为空）；调用方拼接到答案前面
    """
    level = verification.get("support_level", "full")
    unsupported = verification.get("unsupported_sentences", [])

    if level == "full":
        return ""
    if level == "none":
        return "⚠️ **自检提示**：以下回答未能从知识库中找到充分依据，可能存在不准确内容，请谨慎参考：\n\n"
    # partial
    if unsupported:
        preview = "、".join(unsupported[:3])
        return (
            f"⚠️ **自检提示**：以下回答中部分内容（如「{preview}」）"
            f"未能从知识库中找到依据，请结合权威资料核实：\n\n"
        )
    return "⚠️ **自检提示**：以下回答中部分内容未被知识库支持：\n\n"
