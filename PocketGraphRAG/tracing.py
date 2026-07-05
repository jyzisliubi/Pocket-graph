"""Langfuse Tracing 集成模块

对标 microsoft/graphrag 的 OpenTelemetry 集成和 LightRAG 的 LangSmith 集成。
提供 LLM 调用、检索、抽取等关键环节的可观测性。

设计原则：
1. **可选依赖**：未安装 langfuse 时所有调用为 no-op，零开销
2. **零配置降级**：LANGFUSE_ENABLED=False 时所有调用为 no-op
3. **上下文管理器**：用 with 语句自动管理 span 生命周期
4. **线程安全**：每个 trace 用 contextvars 隔离，避免多请求串扰

使用方式::

    from .tracing import trace_llm_call, start_trace

    # 方式1：独立 LLM 调用 trace
    with trace_llm_call(name="qa_generation", model="qwen-max",
                        prompt="...", metadata={...}) as span:
        result = call_llm(...)
        span.set_output(result)

    # 方式2：完整请求 trace
    with start_trace("qa_request", user_id="user123") as trace:
        with trace.span("retrieval", metadata={"mode": "mix"}) as span:
            results = retrieve(...)
            span.set_output({"count": len(results)})
        with trace.span("generation") as span:
            answer = call_llm(...)
            span.set_output(answer)

Langfuse 上可观察：
- 每次问答的完整链路（检索 → 生成）
- LLM 调用的 prompt/completion/token 数/耗时
- 三元组抽取的 chunk 数和三元组数
- 多模型融合的每个模型抽取数量
"""

import contextlib
import os
import time
from typing import Any, Dict, Optional

from .config import (
    LANGFUSE_ENABLED,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)
from .logging_config import get_logger

logger = get_logger(__name__)


# ========================
# Langfuse 客户端（惰性初始化）
# ========================

_langfuse_client = None
_langfuse_available: Optional[bool] = None


def _get_langfuse_client():
    """惰性初始化 Langfuse 客户端。

    未安装 langfuse 或未配置 keys 时返回 None，所有 tracing 调用降级为 no-op。
    """
    global _langfuse_client, _langfuse_available

    if _langfuse_available is False:
        return None

    if _langfuse_client is not None:
        return _langfuse_client

    if not LANGFUSE_ENABLED:
        _langfuse_available = False
        return None

    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        logger.warning(
            "Langfuse 已启用但未配置 POCKET_LANGFUSE_PUBLIC_KEY/SECRET_KEY，"
            "tracing 降级为 no-op"
        )
        _langfuse_available = False
        return None

    try:
        from langfuse import Langfuse  # type: ignore

        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        _langfuse_available = True
        logger.info("Langfuse Tracing 已启用: %s", LANGFUSE_HOST)
        return _langfuse_client
    except ImportError:
        logger.info(
            "langfuse 未安装，tracing 降级为 no-op。"
            "安装: pip install langfuse 或 pip install 'pocketgraphrag[langfuse]'"
        )
        _langfuse_available = False
        return None
    except Exception as e:
        logger.warning("Langfuse 初始化失败，tracing 降级为 no-op: %s", e)
        _langfuse_available = False
        return None


def is_tracing_enabled() -> bool:
    """检查 tracing 是否启用（用于上层判断是否收集 metadata）"""
    return _get_langfuse_client() is not None


# ========================
# No-op Span（未启用时使用）
# ========================


class _NoOpSpan:
    """未启用 tracing 时的空 span，所有方法为 no-op"""

    def set_output(self, output: Any) -> None:
        pass

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        pass

    def set_level(self, level: str) -> None:
        pass

    def set_status(self, status: str) -> None:
        pass

    def set_error(self, error: Exception) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.set_error(exc_val)
        self.end()
        return False


class _NoOpTrace:
    """未启用 tracing 时的空 trace"""

    def span(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        input: Optional[Any] = None,
    ) -> _NoOpSpan:
        return _NoOpSpan()

    def generation(
        self,
        name: str,
        model: Optional[str] = None,
        prompt: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _NoOpSpan:
        return _NoOpSpan()

    def update(self, **kwargs: Any) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ========================
# Langfuse Span 包装器
# ========================


class _LangfuseSpan:
    """Langfuse span 包装器，统一接口"""

    def __init__(self, span: Any):
        self._span = span
        self._ended = False

    def set_output(self, output: Any) -> None:
        try:
            self._span.end(output=output)
            self._ended = True
        except Exception as e:
            logger.debug("Langfuse span.set_output 失败: %s", e)

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        try:
            self._span.update(metadata=metadata)
        except Exception as e:
            logger.debug("Langfuse span.set_metadata 失败: %s", e)

    def set_level(self, level: str) -> None:
        try:
            self._span.update(level=level)
        except Exception as e:
            logger.debug("Langfuse span.set_level 失败: %s", e)

    def set_status(self, status: str) -> None:
        try:
            self._span.update(status=status)
        except Exception as e:
            logger.debug("Langfuse span.set_status 失败: %s", e)

    def set_error(self, error: Exception) -> None:
        try:
            self._span.end(level="ERROR", status_message=str(error))
            self._ended = True
        except Exception as e:
            logger.debug("Langfuse span.set_error 失败: %s", e)

    def end(self) -> None:
        if not self._ended:
            try:
                self._span.end()
            except Exception as e:
                logger.debug("Langfuse span.end 失败: %s", e)
            self._ended = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.set_error(exc_val)
        self.end()
        return False


class _LangfuseTrace:
    """Langfuse trace 包装器"""

    def __init__(self, trace: Any):
        self._trace = trace

    def span(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        input: Optional[Any] = None,
    ) -> _LangfuseSpan:
        try:
            span = self._trace.span(name=name, metadata=metadata or {}, input=input)
            return _LangfuseSpan(span)
        except Exception as e:
            logger.debug("Langfuse span 创建失败: %s", e)
            return _NoOpSpan()

    def generation(
        self,
        name: str,
        model: Optional[str] = None,
        prompt: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _LangfuseSpan:
        try:
            gen = self._trace.generation(
                name=name,
                model=model,
                input=prompt,
                metadata=metadata or {},
            )
            return _LangfuseSpan(gen)
        except Exception as e:
            logger.debug("Langfuse generation 创建失败: %s", e)
            return _NoOpSpan()

    def update(self, **kwargs: Any) -> None:
        try:
            self._trace.update(**kwargs)
        except Exception as e:
            logger.debug("Langfuse trace.update 失败: %s", e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ========================
# 公共 API
# ========================


@contextlib.contextmanager
def start_trace(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    input: Optional[Any] = None,
):
    """启动一个 trace（顶层）。

    用法::

        with start_trace("qa_request", user_id="u123") as trace:
            with trace.span("retrieval") as span:
                results = retrieve(...)
                span.set_output({"count": len(results)})
            with trace.generation("llm_answer", model="qwen-max") as gen:
                answer = call_llm(...)
                gen.set_output(answer)

    未启用 Langfuse 时返回 NoOpTrace，所有方法为 no-op。
    """
    client = _get_langfuse_client()
    if client is None:
        yield _NoOpTrace()
        return

    try:
        trace = client.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            input=input,
        )
        yield _LangfuseTrace(trace)
    except Exception as e:
        logger.debug("Langfuse trace 创建失败，降级 no-op: %s", e)
        yield _NoOpTrace()


@contextlib.contextmanager
def trace_llm_call(
    name: str,
    model: Optional[str] = None,
    prompt: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
):
    """便捷函数：独立 LLM 调用的 trace（自动创建顶层 trace + generation）。

    用于不显式管理 trace 的简单场景，如 kg_extractor 中的 LLM 调用。
    """
    client = _get_langfuse_client()
    if client is None:
        yield _NoOpSpan()
        return

    try:
        trace = client.trace(name=name, metadata=metadata or {})
        gen = trace.generation(
            name=name,
            model=model,
            input=prompt,
            metadata=metadata or {},
        )
        yield _LangfuseSpan(gen)
    except Exception as e:
        logger.debug("Langfuse trace_llm_call 失败，降级 no-op: %s", e)
        yield _NoOpSpan()


@contextlib.contextmanager
def trace_span(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    input: Optional[Any] = None,
):
    """便捷函数：独立 span（无父 trace 时自动创建一个）。

    用于 retrieve/extract 等非 LLM 调用环节。
    """
    client = _get_langfuse_client()
    if client is None:
        yield _NoOpSpan()
        return

    try:
        trace = client.trace(name=name)
        span = trace.span(name=name, metadata=metadata or {}, input=input)
        yield _LangfuseSpan(span)
    except Exception as e:
        logger.debug("Langfuse trace_span 失败，降级 no-op: %s", e)
        yield _NoOpSpan()


def flush():
    """强制刷新所有 pending 的 trace 到 Langfuse。

    程序退出时调用，确保 trace 不丢失。
    """
    client = _get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:
        logger.debug("Langfuse flush 失败: %s", e)
