"""Tests for ConversationMemory."""

from PocketGraphRAG.conversation import ConversationMemory


class MockLLMCaller:
    """Mock LLM caller function for testing rewrite_query.

    Acts as a callable: llm_caller(system_prompt, user_prompt, stream=False) -> str
    """

    def __init__(self, return_value="改写后的查询"):
        self.return_value = return_value
        self.calls = []

    def __call__(self, system_prompt, user_prompt, stream=False):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        if stream:

            def gen():
                yield {"chunk": self.return_value, "full_answer": self.return_value}
                yield {"full_answer": self.return_value, "done": True}

            return gen()
        return self.return_value


class TestConversationMemory:
    def test_initial_state(self):
        cm = ConversationMemory()
        assert cm.history == []
        assert cm.max_turns == 5

    def test_custom_max_turns(self):
        cm = ConversationMemory(max_turns=3)
        assert cm.max_turns == 3

    def test_add_user_message(self):
        cm = ConversationMemory()
        cm.add("user", "你好")
        assert len(cm.history) == 1
        assert cm.history[0]["role"] == "user"
        assert cm.history[0]["content"] == "你好"

    def test_add_assistant_message(self):
        cm = ConversationMemory()
        cm.add("assistant", "你好，有什么可以帮助你的？")
        assert len(cm.history) == 1
        assert cm.history[0]["role"] == "assistant"

    def test_add_multiple_messages(self):
        cm = ConversationMemory()
        cm.add("user", "问题1")
        cm.add("assistant", "回答1")
        cm.add("user", "问题2")
        cm.add("assistant", "回答2")
        assert len(cm.history) == 4

    def test_trim_exceeds_max_turns(self):
        cm = ConversationMemory(max_turns=2)
        for i in range(5):
            cm.add("user", f"问题{i}")
            cm.add("assistant", f"回答{i}")

        assert len(cm.history) == 4
        assert cm.history[0]["content"] == "问题3"
        assert cm.history[-1]["content"] == "回答4"

    def test_clear_history(self):
        cm = ConversationMemory()
        cm.add("user", "测试")
        cm.add("assistant", "回复")
        cm.clear()
        assert cm.history == []

    def test_get_context_string_empty(self):
        cm = ConversationMemory()
        assert cm.get_context_string() == ""

    def test_get_context_string_with_history(self):
        cm = ConversationMemory()
        cm.add("user", "稻瘟病怎么治？")
        cm.add("assistant", "可以用三环唑")

        ctx = cm.get_context_string()
        assert "用户" in ctx
        assert "助手" in ctx
        assert "稻瘟病" in ctx
        assert "三环唑" in ctx

    def test_rewrite_query_no_history_returns_original(self):
        cm = ConversationMemory()
        mock_llm = MockLLMCaller(return_value="稻瘟病怎么防治？")

        result = cm.rewrite_query("稻瘟病怎么防治？", mock_llm)
        assert result == "稻瘟病怎么防治？"
        assert len(mock_llm.calls) == 0

    def test_rewrite_query_with_history(self):
        cm = ConversationMemory()
        cm.add("user", "稻瘟病怎么防治？")
        cm.add("assistant", "可以使用三环唑等药剂。")

        mock_llm = MockLLMCaller(return_value="稻瘟病的用量是多少？")

        result = cm.rewrite_query("用量是多少？", mock_llm)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "用量是多少？"

    def test_rewrite_query_skips_complete_question_even_with_history(self):
        cm = ConversationMemory()
        cm.add("user", "三环唑可以防治哪些病害？")
        cm.add("assistant", "可以防治稻瘟病等病害。")

        mock_llm = MockLLMCaller(return_value="三环唑可以防治稻瘟病吗？如果有，它的症状是什么？")

        result = cm.rewrite_query("稻瘟病有什么症状？", mock_llm)
        assert result == "稻瘟病有什么症状？"
        assert len(mock_llm.calls) == 0

    def test_rewrite_query_calls_llm(self):
        cm = ConversationMemory()
        cm.add("user", "问题1")
        cm.add("assistant", "回答1")

        mock_llm = MockLLMCaller(return_value="改写结果")
        cm.rewrite_query("那用量是多少？", mock_llm)

        assert len(mock_llm.calls) == 1
        assert "那用量是多少？" in mock_llm.calls[0]["user"]
        assert "问题1" in mock_llm.calls[0]["user"]

    def test_rewrite_query_llm_returns_empty_fallback(self):
        cm = ConversationMemory()
        cm.add("user", "问题1")
        cm.add("assistant", "回答1")

        mock_llm = MockLLMCaller(return_value="")

        result = cm.rewrite_query("追问", mock_llm)
        assert result == "追问"
