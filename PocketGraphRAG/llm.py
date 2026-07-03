"""
统一 LLM 调用层

所有模块通过此模块调用 LLM，避免重复实现 API 调用逻辑。
支持 SiliconFlow / DashScope / OpenAI 兼容 API，按优先级自动降级。

使用方式：
    from .llm import call_llm

    result = call_llm("系统提示", "用户提示")
    # 或指定参数
    result = call_llm("系统提示", "用户提示", temperature=0.0, max_tokens=500)
"""

import asyncio
import json
import queue
import threading
from typing import AsyncGenerator, Generator, Optional, Union

import requests

from .logging_config import get_logger

logger = get_logger(__name__)

from .config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_API_URL,
    DASHSCOPE_MODEL,
    DEEPSEEK_API_BASE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    FREELM_CN_API_BASE,
    FREELM_CN_API_KEY,
    FREELM_CN_MODEL,
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SILICONFLOW_API_BASE,
    SILICONFLOW_API_KEY,
    SILICONFLOW_MODEL,
)


def _stream_chunks(response, label: str = "API"):
    """从 SSE 响应中解析流式 chunk。

    内部异常被捕获并记录，不向调用方冒泡——生成器只会自然结束。
    避免流式中途连接断开 / JSON 解析失败导致上层 `for chunk in gen` 抛异常。
    """
    try:
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}) or {}
            content = delta.get("content")
            if content:
                yield content
    except Exception as e:
        # 流式中断：记录后让生成器自然结束，调用方只会看到"提前结束"
        logger.warning("[%s 流式中断] %s", label, e)


def _call_openai_compatible(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    label: str = "API",
    stream: bool = False,
) -> Union[Optional[str], Generator[str, None, None]]:
    """通用 OpenAI 兼容 API 调用。

    流式模式下，会先 peek 第一个有效 chunk 以验证连接真正可用；
    若连接失败或没有任何 chunk，返回 None 让上层 call_llm 降级到下一个后端。
    修复了旧版"流式返回生成器对象永远非 None 导致降级失效"的 bug。
    """
    import itertools

    url = f"{api_base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    # 1. 发起请求：连接 / HTTP 错误在此捕获，返回 None 触发降级
    #    超时走 config.LLM_TIMEOUT（默认 120s，Ollama 7B 本地推理较慢）
    try:
        from .config import LLM_TIMEOUT

        timeout = LLM_TIMEOUT
    except ImportError:
        timeout = 120  # 兜底，避免 config 循环导入时崩溃
    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=timeout, stream=stream
        )
        response.raise_for_status()
    except Exception as e:
        logger.warning("[%s 请求失败] %s", label, e)
        return None

    # 2. 非流式：直接解析 JSON
    if not stream:
        try:
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("[%s 解析响应失败] %s", label, e)
            return None

    # 3. 流式：peek 第一个 chunk 验证可用性，再拼接剩余 chunk
    gen = _stream_chunks(response, label=label)
    try:
        first = next(gen)
    except StopIteration:
        # 连接成功但没有任何 chunk（空生成），视作失败以触发降级
        logger.info("[%s 流式返回空，降级到下一个后端]", label)
        return None
    except Exception as e:
        logger.warning("[%s 流式首 chunk 失败] %s", label, e)
        return None
    return itertools.chain([first], gen)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    stream: bool = False,
    role: str = "query",
) -> Union[Optional[str], Generator[str, None, None]]:
    """
    统一 LLM 调用入口，按优先级依次尝试不同后端。

    优先级：Ollama → freellm-cn → SiliconFlow → DeepSeek → DashScope → OpenAI

    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        temperature: 生成温度
        max_tokens: 最大生成 token 数
        stream: 是否流式输出
        role: LLM 角色 "query"（问答）或 "extract"（KG 抽取）。
            当设置了 POCKET_EXTRACT_MODEL / POCKET_QUERY_MODEL 环境变量时，
            会用对应模型覆盖当前 provider 的默认模型，实现多角色差异化配置。
            未设置时使用统一模型（向后兼容）。

    Returns:
        LLM 生成的文本，所有后端均失败时返回 None
    """
    from .config import EXTRACT_MODEL, QUERY_MODEL
    # 根据 role 选择模型覆盖：extract 角色用 EXTRACT_MODEL，query 角色用 QUERY_MODEL
    model_override = EXTRACT_MODEL if role == "extract" else QUERY_MODEL

    # 优先尝试 Ollama 本地模型
    if OLLAMA_MODEL:
        result = _call_openai_compatible(
            OLLAMA_API_BASE,
            "",
            model_override or OLLAMA_MODEL,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            label="Ollama",
            stream=stream,
        )
        if result is not None:
            return result

    # 备选 freellm-cn（本地大模型服务网关，OpenAI 兼容）
    # 启动方式：运行 start_freellm_cn.bat / start_freellm_cn.sh
    if FREELM_CN_API_KEY:
        result = _call_openai_compatible(
            FREELM_CN_API_BASE,
            FREELM_CN_API_KEY,
            model_override or FREELM_CN_MODEL,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            label="freellm-cn",
            stream=stream,
        )
        if result is not None:
            return result

    # 备选 SiliconFlow
    if SILICONFLOW_API_KEY:
        result = _call_openai_compatible(
            SILICONFLOW_API_BASE,
            SILICONFLOW_API_KEY,
            model_override or SILICONFLOW_MODEL,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            label="SiliconFlow",
            stream=stream,
        )
        if result is not None:
            return result

    # 备选 DeepSeek
    if DEEPSEEK_API_KEY:
        result = _call_openai_compatible(
            DEEPSEEK_API_BASE,
            DEEPSEEK_API_KEY,
            model_override or DEEPSEEK_MODEL,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            label="DeepSeek",
            stream=stream,
        )
        if result is not None:
            return result

    # 备选 DashScope（通义千问）
    if DASHSCOPE_API_KEY:
        result = _call_openai_compatible(
            # DashScope 兼容 OpenAI 格式，base URL 需要去掉 /chat/completions 后缀
            DASHSCOPE_API_URL.rsplit("/chat/completions", 1)[0],
            DASHSCOPE_API_KEY,
            model_override or DASHSCOPE_MODEL,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            label="DashScope",
            stream=stream,
        )
        if result is not None:
            return result

    # 备选 OpenAI 兼容 API
    if OPENAI_API_KEY:
        result = _call_openai_compatible(
            OPENAI_API_BASE,
            OPENAI_API_KEY,
            model_override or OPENAI_MODEL,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            label="OpenAI",
            stream=stream,
        )
        if result is not None:
            return result

    if stream:
        return iter(())
    return None


# ==========================
# 异步 API（Async API）
# ==========================
# 提供 acall_llm / acall_llm_stream，与同步版一一对应。
# 通过 asyncio 把同步阻塞调用搬到线程池中执行，避免重复实现一遍 HTTP 逻辑，
# 也不引入额外依赖（无需 httpx / aiohttp）。
# 适用于 FastAPI / anyio 等 async 上下文。

if hasattr(asyncio, "to_thread"):
    # Python 3.9+
    _to_thread = asyncio.to_thread
else:  # pragma: no cover - Python 3.8 兼容

    async def _to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def acall_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> Optional[str]:
    """异步版 call_llm（非流式）。

    用法::

        answer = await acall_llm("系统提示", "用户提示")

    Returns:
        LLM 生成的文本，所有后端均失败时返回 None
    """
    return await _to_thread(
        call_llm, system_prompt, user_prompt, temperature, max_tokens, stream=False
    )


async def acall_llm_stream(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> AsyncGenerator[str, None]:
    """异步流式版 call_llm。

    在后台线程运行同步流式生成器，通过 asyncio.Queue 把 chunk 传回事件循环，
    避免阻塞 async 上下文。

    用法::

        async for chunk in acall_llm_stream("系统提示", "用户提示"):
            print(chunk, end="", flush=True)
    """
    sync_gen = call_llm(
        system_prompt, user_prompt, temperature, max_tokens, stream=True
    )
    if sync_gen is None:
        return

    buf: queue.Queue = queue.Queue()
    # 用唯一哨兵对象做结束标记，避免与可能的 None chunk 冲突
    _SENTINEL = object()
    _EXC_MARKER = object()

    def _produce():
        try:
            for chunk in sync_gen:
                buf.put(chunk)
        except Exception as e:  # 把异常也送回主线程
            buf.put((_EXC_MARKER, e))
        finally:
            buf.put(_SENTINEL)

    producer = threading.Thread(target=_produce, daemon=True)
    producer.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, buf.get)
        if item is _SENTINEL:
            break
        if isinstance(item, tuple) and item and item[0] is _EXC_MARKER:
            raise item[1]
        yield item


def has_llm() -> bool:
    """检查是否配置了至少一个 LLM 后端"""
    return bool(
        OLLAMA_MODEL
        or FREELM_CN_API_KEY
        or SILICONFLOW_API_KEY
        or DEEPSEEK_API_KEY
        or DASHSCOPE_API_KEY
        or OPENAI_API_KEY
    )


def get_active_provider() -> str:
    """获取当前激活的 LLM 提供商名称"""
    if OLLAMA_MODEL:
        return f"Ollama ({OLLAMA_MODEL})"
    if FREELM_CN_API_KEY:
        return f"freellm-cn ({FREELM_CN_MODEL})"
    if SILICONFLOW_API_KEY:
        return f"SiliconFlow ({SILICONFLOW_MODEL})"
    if DEEPSEEK_API_KEY:
        return f"DeepSeek ({DEEPSEEK_MODEL})"
    if DASHSCOPE_API_KEY:
        return f"DashScope ({DASHSCOPE_MODEL})"
    if OPENAI_API_KEY:
        return f"OpenAI ({OPENAI_MODEL})"
    return "纯检索模式（未配置 LLM API Key 或 Ollama）"
