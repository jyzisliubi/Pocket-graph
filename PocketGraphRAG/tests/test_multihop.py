"""Tests for multi-hop query decomposition."""

import json
from unittest.mock import patch


class TestDecomposeQuery:
    def test_simple_query_not_decomposed(self):
        """A simple single-entity query should return itself."""
        from PocketGraphRAG.multihop import decompose_query

        mock_response = json.dumps({"sub_queries": ["稻瘟病怎么防治？"]})
        with patch("PocketGraphRAG.multihop.call_llm", return_value=mock_response):
            result = decompose_query("稻瘟病怎么防治？")
            assert isinstance(result, list)
            assert len(result) >= 1
            assert "稻瘟病" in result[0]

    def test_complex_query_decomposed(self):
        """A complex query should be split into multiple sub-queries."""
        from PocketGraphRAG.multihop import decompose_query

        mock_response = json.dumps(
            {
                "sub_queries": [
                    "三环唑可以防治哪些病害",
                    "三环唑的用量是多少",
                ]
            }
        )
        with patch("PocketGraphRAG.multihop.call_llm", return_value=mock_response):
            result = decompose_query("三环唑可以防治哪些病害，各自的用量是多少？")
            assert isinstance(result, list)
            assert len(result) >= 2

    def test_returns_list_of_strings(self):
        """Result should always be a list of strings."""
        from PocketGraphRAG.multihop import decompose_query

        mock_response = json.dumps({"sub_queries": ["子查询1", "子查询2"]})
        with patch("PocketGraphRAG.multihop.call_llm", return_value=mock_response):
            result = decompose_query("复杂问题")
            assert all(isinstance(q, str) for q in result)

    def test_handles_malformed_json(self):
        """Should handle malformed LLM responses gracefully."""
        from PocketGraphRAG.multihop import decompose_query

        with patch("PocketGraphRAG.multihop.call_llm", return_value="not json"):
            result = decompose_query("测试问题")
            assert isinstance(result, list)
            assert len(result) >= 1
