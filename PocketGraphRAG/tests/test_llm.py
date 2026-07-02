"""
LLM 统一调用层单元测试
"""

from unittest.mock import MagicMock, patch

from PocketGraphRAG.llm import (
    _call_openai_compatible,
    call_llm,
    get_active_provider,
    has_llm,
)


class TestCallOpenAICompatible:
    @patch("PocketGraphRAG.llm.requests.post")
    def test_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "测试回答"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = _call_openai_compatible(
            "https://api.example.com/v1",
            "test-key",
            "test-model",
            "系统提示",
            "用户提示",
        )
        assert result == "测试回答"

    @patch("PocketGraphRAG.llm.requests.post")
    def test_failure_returns_none(self, mock_post):
        mock_post.side_effect = Exception("网络错误")
        result = _call_openai_compatible(
            "https://api.example.com/v1",
            "test-key",
            "test-model",
            "系统提示",
            "用户提示",
        )
        assert result is None


class TestCallLLM:
    @patch("PocketGraphRAG.llm.SILICONFLOW_API_KEY", "test-key")
    @patch("PocketGraphRAG.llm._call_openai_compatible")
    def test_tries_siliconflow_first(self, mock_call):
        mock_call.return_value = "SiliconFlow 回答"
        result = call_llm("系统", "用户")
        assert result == "SiliconFlow 回答"
        mock_call.assert_called_once()

    @patch("PocketGraphRAG.llm.OLLAMA_MODEL", "")
    @patch("PocketGraphRAG.llm.SILICONFLOW_API_KEY", "")
    @patch("PocketGraphRAG.llm.DASHSCOPE_API_KEY", "")
    @patch("PocketGraphRAG.llm.OPENAI_API_KEY", "")
    def test_no_key_returns_none(self):
        result = call_llm("系统", "用户")
        assert result is None

    @patch("PocketGraphRAG.llm.SILICONFLOW_API_KEY", "key1")
    @patch("PocketGraphRAG.llm._call_openai_compatible")
    def test_fallback_on_failure(self, mock_call):
        """第一个后端失败时应尝试下一个"""
        mock_call.side_effect = [None, "DashScope 回答"]
        with patch("PocketGraphRAG.llm.DASHSCOPE_API_KEY", "key2"):
            result = call_llm("系统", "用户")
            assert result == "DashScope 回答"


class TestHasLLM:
    @patch("PocketGraphRAG.llm.OLLAMA_MODEL", "")
    @patch("PocketGraphRAG.llm.SILICONFLOW_API_KEY", "")
    @patch("PocketGraphRAG.llm.DASHSCOPE_API_KEY", "")
    @patch("PocketGraphRAG.llm.OPENAI_API_KEY", "")
    def test_no_key(self):
        assert has_llm() is False

    @patch("PocketGraphRAG.llm.OLLAMA_MODEL", "")
    @patch("PocketGraphRAG.llm.SILICONFLOW_API_KEY", "test")
    def test_has_key(self):
        assert has_llm() is True


class TestGetActiveProvider:
    @patch("PocketGraphRAG.llm.OLLAMA_MODEL", "")
    @patch("PocketGraphRAG.llm.SILICONFLOW_API_KEY", "test")
    def test_siliconflow(self):
        assert "SiliconFlow" in get_active_provider()

    @patch("PocketGraphRAG.llm.OLLAMA_MODEL", "")
    @patch("PocketGraphRAG.llm.SILICONFLOW_API_KEY", "")
    @patch("PocketGraphRAG.llm.DASHSCOPE_API_KEY", "test")
    def test_dashscope(self):
        assert "DashScope" in get_active_provider()

    @patch("PocketGraphRAG.llm.OLLAMA_MODEL", "")
    @patch("PocketGraphRAG.llm.SILICONFLOW_API_KEY", "")
    @patch("PocketGraphRAG.llm.DASHSCOPE_API_KEY", "")
    @patch("PocketGraphRAG.llm.OPENAI_API_KEY", "")
    def test_no_provider(self):
        assert "纯检索模式" in get_active_provider()
