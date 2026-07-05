"""Langfuse Tracing 集成单元测试

覆盖 tracing 模块的核心逻辑：
- 未启用时所有调用为 no-op（_NoOpSpan/_NoOpTrace）
- is_tracing_enabled 在未配置时返回 False
- start_trace / trace_llm_call / trace_span 上下文管理器正确工作
- span 的 set_output/set_metadata/set_error 不抛异常
- 异常时 span 自动标记 ERROR
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG import tracing
from PocketGraphRAG.tracing import (
    _NoOpSpan,
    _NoOpTrace,
    flush,
    is_tracing_enabled,
    start_trace,
    trace_llm_call,
    trace_span,
)


class TestNoOpSpan:
    """_NoOpSpan 测试"""

    def test_all_methods_are_noop(self):
        """所有方法都是 no-op，不抛异常"""
        span = _NoOpSpan()
        span.set_output({"key": "value"})
        span.set_metadata({"meta": "data"})
        span.set_level("DEBUG")
        span.set_status("OK")
        span.set_error(RuntimeError("test error"))
        span.end()

    def test_context_manager_protocol(self):
        """支持 with 语句"""
        with _NoOpSpan() as span:
            span.set_output("test")
        # 不抛异常即可

    def test_context_manager_captures_error(self):
        """with 块内抛异常时自动 set_error（不吞异常）"""
        with pytest.raises(RuntimeError):
            with _NoOpSpan() as span:
                raise RuntimeError("test")


class TestNoOpTrace:
    """_NoOpTrace 测试"""

    def test_span_returns_noop(self):
        trace = _NoOpTrace()
        span = trace.span("test_span")
        assert isinstance(span, _NoOpSpan)

    def test_generation_returns_noop(self):
        trace = _NoOpTrace()
        gen = trace.generation("test_gen", model="qwen-max")
        assert isinstance(gen, _NoOpSpan)

    def test_update_is_noop(self):
        trace = _NoOpTrace()
        trace.update(name="updated")

    def test_context_manager(self):
        with _NoOpTrace() as trace:
            span = trace.span("test")
            span.set_output("ok")


class TestIsTracingEnabled:
    """is_tracing_enabled 测试"""

    def test_disabled_by_default(self, monkeypatch):
        """默认未启用"""
        # 重置客户端状态
        monkeypatch.setattr(tracing, "_langfuse_client", None)
        monkeypatch.setattr(tracing, "_langfuse_available", None)
        monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", False)
        assert is_tracing_enabled() is False

    def test_enabled_but_no_keys(self, monkeypatch):
        """启用但无 keys 也返回 False"""
        monkeypatch.setattr(tracing, "_langfuse_client", None)
        monkeypatch.setattr(tracing, "_langfuse_available", None)
        monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", True)
        monkeypatch.setattr(tracing, "LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setattr(tracing, "LANGFUSE_SECRET_KEY", "")
        assert is_tracing_enabled() is False

    def test_enabled_with_keys_but_no_langfuse_module(self, monkeypatch):
        """启用+有 keys 但 langfuse 未安装"""
        monkeypatch.setattr(tracing, "_langfuse_client", None)
        monkeypatch.setattr(tracing, "_langfuse_available", None)
        monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", True)
        monkeypatch.setattr(tracing, "LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setattr(tracing, "LANGFUSE_SECRET_KEY", "sk-test")
        # langfuse 未安装时会触发 ImportError
        assert is_tracing_enabled() is False


class TestStartTrace:
    """start_trace 上下文管理器测试"""

    def test_disabled_returns_noop_trace(self, monkeypatch):
        monkeypatch.setattr(tracing, "_langfuse_client", None)
        monkeypatch.setattr(tracing, "_langfuse_available", None)
        monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", False)
        with start_trace("test") as trace:
            assert isinstance(trace, _NoOpTrace)

    def test_disabled_span_is_noop(self, monkeypatch):
        monkeypatch.setattr(tracing, "_langfuse_client", None)
        monkeypatch.setattr(tracing, "_langfuse_available", None)
        monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", False)
        with start_trace("test") as trace:
            with trace.span("child") as span:
                assert isinstance(span, _NoOpSpan)
                span.set_output("ok")

    def test_disabled_generation_is_noop(self, monkeypatch):
        monkeypatch.setattr(tracing, "_langfuse_client", None)
        monkeypatch.setattr(tracing, "_langfuse_available", None)
        monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", False)
        with start_trace("test") as trace:
            with trace.generation("llm", model="qwen") as gen:
                assert isinstance(gen, _NoOpSpan)


class TestTraceLLMCall:
    """trace_llm_call 便捷函数测试"""

    def test_disabled_returns_noop_span(self, monkeypatch):
        monkeypatch.setattr(tracing, "_langfuse_client", None)
        monkeypatch.setattr(tracing, "_langfuse_available", None)
        monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", False)
        with trace_llm_call("test", model="qwen") as span:
            assert isinstance(span, _NoOpSpan)
            span.set_output("answer")

    def test_exception_propagates(self, monkeypatch):
        """with 块内异常正常传播"""
        monkeypatch.setattr(tracing, "_langfuse_client", None)
        monkeypatch.setattr(tracing, "_langfuse_available", None)
        monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", False)
        with pytest.raises(ValueError):
            with trace_llm_call("test"):
                raise ValueError("test error")


class TestTraceSpan:
    """trace_span 便捷函数测试"""

    def test_disabled_returns_noop(self, monkeypatch):
        monkeypatch.setattr(tracing, "_langfuse_client", None)
        monkeypatch.setattr(tracing, "_langfuse_available", None)
        monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", False)
        with trace_span("retrieval") as span:
            assert isinstance(span, _NoOpSpan)


class TestFlush:
    """flush 函数测试"""

    def test_disabled_flush_is_noop(self, monkeypatch):
        monkeypatch.setattr(tracing, "_langfuse_client", None)
        monkeypatch.setattr(tracing, "_langfuse_available", None)
        monkeypatch.setattr(tracing, "LANGFUSE_ENABLED", False)
        # 不抛异常即可
        flush()


class TestConfigLoading:
    """config.py 中 Langfuse 配置加载测试"""

    def test_config_values_exist(self):
        """config.py 正确导出 Langfuse 配置"""
        from PocketGraphRAG import config

        assert hasattr(config, "LANGFUSE_ENABLED")
        assert hasattr(config, "LANGFUSE_PUBLIC_KEY")
        assert hasattr(config, "LANGFUSE_SECRET_KEY")
        assert hasattr(config, "LANGFUSE_HOST")
        assert isinstance(config.LANGFUSE_ENABLED, bool)
        assert isinstance(config.LANGFUSE_HOST, str)
