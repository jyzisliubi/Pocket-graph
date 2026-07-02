"""
异步 LLM 调用层单元测试 (acall_llm / acall_llm_stream)

不依赖 pytest-asyncio：用 asyncio.run 驱动协程，保持零额外依赖。
"""

import asyncio
from unittest.mock import patch

import pytest

from PocketGraphRAG.llm import _to_thread, acall_llm, acall_llm_stream


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(agen):
    out = []
    async for chunk in agen:
        out.append(chunk)
    return out


class TestACallLLM:
    def test_acall_returns_string(self):
        """acall_llm 在配置了后端时返回字符串"""
        with patch("PocketGraphRAG.llm.call_llm", return_value="异步回答") as mock_call:
            result = _run(acall_llm("系统", "用户"))
            assert result == "异步回答"
            mock_call.assert_called_once()

    def test_acall_returns_none_when_no_llm(self):
        with patch("PocketGraphRAG.llm.call_llm", return_value=None):
            assert _run(acall_llm("系统", "用户")) is None

    def test_acall_forwards_params(self):
        """temperature / max_tokens / stream 透传到同步实现"""
        with patch("PocketGraphRAG.llm.call_llm", return_value="ok") as mock_call:
            _run(acall_llm("系统", "用户", temperature=0.5, max_tokens=128))
            args, kwargs = mock_call.call_args
            # acall_llm 把 system/user/temperature/max_tokens 作为位置参数、
            # stream=False 作为关键字参数传给底层 call_llm（经 _to_thread）
            assert args[0] == "系统"
            assert args[1] == "用户"
            assert args[2] == 0.5
            assert args[3] == 128
            assert kwargs.get("stream") is False  # stream=False 走 kwargs


class TestACallLLMStream:
    def test_stream_yields_chunks(self):
        def fake_sync_gen():
            yield "hello"
            yield " world"

        with patch("PocketGraphRAG.llm.call_llm", return_value=fake_sync_gen()):
            chunks = _run(_collect(acall_llm_stream("系统", "用户")))
            assert "".join(chunks) == "hello world"

    def test_stream_empty_when_no_llm(self):
        with patch("PocketGraphRAG.llm.call_llm", return_value=None):
            assert _run(_collect(acall_llm_stream("系统", "用户"))) == []

    def test_stream_propagates_exception(self):
        def fake_sync_gen():
            yield "a"
            raise RuntimeError("boom")

        with patch("PocketGraphRAG.llm.call_llm", return_value=fake_sync_gen()):
            with pytest.raises(RuntimeError, match="boom"):
                _run(_collect(acall_llm_stream("系统", "用户")))


class TestToThreadCompat:
    def test_to_thread_runs_in_executor(self):
        """_to_thread 在 3.8+ 上都可用"""

        async def _main():
            return await _to_thread(lambda x: x + 1, 41)

        assert _run(_main()) == 42

    def test_to_thread_returns_coroutine(self):
        """_to_thread(...) 必须返回可 await 的协程"""
        coro = _to_thread(lambda: 1)
        import inspect

        assert inspect.iscoroutine(coro)
        coro.close()  # 避免未 await 警告
