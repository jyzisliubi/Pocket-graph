"""
Conversational Memory + Context Rewriting

Maintains conversation history and uses LLM to rewrite follow-up queries
with context from previous turns, enabling natural multi-turn dialogue.
"""

REWRITE_PROMPT = """你是一个查询改写助手。根据对话历史，将用户的追问改写成一个完整、独立的查询。

规则：
1. 如果追问依赖上文语境（如"导演呢"、"那诺兰呢"），补全为完整查询
2. 如果问题本身已完整，直接返回原问题
3. 只返回改写后的查询，不要解释

对话历史：
{history}

用户追问：{question}

改写后的完整查询："""


class ConversationMemory:
    """Manage conversation history and context-aware query rewriting."""

    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = []  # list of {"role": "user/assistant", "content": str}

    def add(self, role, content):
        """Add a message to history."""
        self.history.append({"role": role, "content": content})
        # Keep only last max_turns pairs
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def clear(self):
        """Clear conversation history."""
        self.history = []

    def get_context_string(self):
        """Format history as a readable string for LLM prompts."""
        if not self.history:
            return ""
        lines = []
        for msg in self.history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}：{msg['content'][:200]}")
        return "\n".join(lines)

    def _looks_like_follow_up(self, question):
        """Only rewrite when the new question clearly depends on prior turns."""
        q = (question or "").strip()
        if not q:
            return False

        direct_follow_up_phrases = {
            "用量是多少？",
            "剂量是多少？",
            "什么时候用？",
            "什么时候施药？",
            "还有哪些？",
            "区别是什么？",
            "那怎么治？",
            "那怎么办？",
            "那呢？",
        }
        if q in direct_follow_up_phrases:
            return True

        follow_up_prefixes = (
            "那",
            "那么",
            "这个",
            "这个病",
            "这种",
            "这类",
            "这些",
            "那些",
            "它",
            "它的",
            "其",
            "该",
            "该病",
            "前者",
            "后者",
        )
        if q.startswith(follow_up_prefixes):
            return True

        pronoun_markers = ("它", "其", "该", "这个", "那", "前者", "后者")
        follow_up_keywords = (
            "用量",
            "剂量",
            "时间",
            "时期",
            "多久",
            "区别",
            "哪个好",
            "怎么用",
            "如何用",
            "还有",
            "还能",
            "可以吗",
            "能不能",
            "是否",
            "症状呢",
        )
        return any(p in q for p in pronoun_markers) and any(
            k in q for k in follow_up_keywords
        )

    def rewrite_query(self, question, llm_caller):
        """Rewrite a follow-up question using conversation context.

        Args:
            question: user's follow-up question
            llm_caller: function(system_prompt, user_prompt) -> str or None

        Returns:
            Rewritten query string (falls back to original if LLM fails)
        """
        if not self.history:
            return question
        if not self._looks_like_follow_up(question):
            return question

        context = self.get_context_string()
        prompt = REWRITE_PROMPT.format(history=context, question=question)

        result = llm_caller(
            "你是一个查询改写助手，只返回改写后的查询文本。",
            prompt,
            stream=False,
        )

        if result and hasattr(result, "strip") and result.strip():
            rewritten = result.strip()
            # Sanity check: don't use rewritten query if it's too long or contains errors
            if len(rewritten) < 200 and rewritten != question:
                return rewritten

        return question
