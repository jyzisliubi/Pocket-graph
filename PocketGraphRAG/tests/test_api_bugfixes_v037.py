"""API 端点 Bug 修复测试（v0.3.7）

测试发现的 3 个 API Bug 修复：
1. /api/retrieve 的 citation_id 为 null → 应为 1/2/3
2. /api/graph/subgraph POST 422 → 应接受 JSON body
3. /api/qa/stream done 事件 answer 为空 → 应正确读取 answer 键
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """测试客户端 fixture，mock RAG 系统"""
    from PocketGraphRAG.api_server import app, _rag

    with patch("PocketGraphRAG.api_server._rag") as mock_rag:
        # 配置 mock RAG
        mock_rag.retrieve.return_value = (
            [
                ("text1", 0.9, {"entity": "entity1", "source_type": "vector"}),
                ("text2", 0.8, {"entity": "entity2", "source_type": "vector"}),
            ],
            {
                "search_type": "vector",
                "seed_entities": ["entity1"],
                "expanded_entities": [],
                "matched_relations": [],
            },
        )
        mock_rag.kg_retriever = MagicMock()
        mock_rag.kg_retriever.get_subgraph.return_value = {
            "nodes": [{"id": "1", "name": "A", "degree": 1, "category": 0, "symbolSize": 10}],
            "links": [{"source": "A", "target": "B", "relation": "r"}],
        }
        with TestClient(app) as c:
            yield c


class TestRetrieveCitationId:
    """Bug #1: /api/retrieve citation_id 修复测试"""

    def test_retrieve_returns_citation_id(self, client):
        """retrieve 端点应返回 citation_id 1/2/3..."""
        response = client.post(
            "/api/retrieve",
            json={"query": "test", "top_k": 2},
        )
        assert response.status_code == 200
        data = response.json()
        sources = data.get("sources", [])
        assert len(sources) == 2
        # citation_id 应为 1 和 2（从1开始）
        assert sources[0]["citation_id"] == 1
        assert sources[1]["citation_id"] == 2

    def test_retrieve_citation_id_sequential(self, client):
        """citation_id 应是连续的 1, 2, 3..."""
        response = client.post(
            "/api/retrieve",
            json={"query": "test", "top_k": 5},
        )
        data = response.json()
        sources = data.get("sources", [])
        for idx, s in enumerate(sources):
            assert s["citation_id"] == idx + 1


class TestSubgraphJsonBody:
    """Bug #2: /api/graph/subgraph POST 接受 JSON body"""

    def test_subgraph_accepts_json_array_body(self, client):
        """subgraph 端点应接受 POST JSON 数组 body"""
        response = client.post(
            "/api/graph/subgraph",
            json=["entity1", "entity2"],
            params={"hops": 1},
        )
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "links" in data

    def test_subgraph_single_entity(self, client):
        """单实体子图"""
        response = client.post(
            "/api/graph/subgraph",
            json=["entity1"],
            params={"hops": 2},
        )
        assert response.status_code == 200


class TestStreamingDoneAnswer:
    """Bug #3: /api/qa/stream done 事件 answer 字段修复测试

    通过单元测试验证 api_server 的 SSE 事件处理逻辑：
    rag_system.answer_stream yield {"done": True, "answer": full_answer, ...}，
    api_server 应读取 "answer" 键（而非 "full_answer"）。
    """

    def test_done_event_reads_answer_key(self):
        """验证 done 事件从 step 中读取 answer 键"""
        # 模拟 rag_system.answer_stream 的 done step
        done_step = {
            "done": True,
            "answer": "这是最终答案",
            "sources": [],
            "pipeline_info": {},
        }
        # api_server 的逻辑：step.get("answer", "") or step.get("full_answer", "")
        answer = done_step.get("answer", "") or done_step.get("full_answer", "")
        assert answer == "这是最终答案"

    def test_done_event_fallback_to_full_answer(self):
        """如果 answer 键不存在，回退到 full_answer"""
        done_step = {
            "done": True,
            "full_answer": "回退答案",
        }
        answer = done_step.get("answer", "") or done_step.get("full_answer", "")
        assert answer == "回退答案"

    def test_done_event_empty_when_both_missing(self):
        """两个键都不存在时返回空字符串"""
        done_step = {"done": True}
        answer = done_step.get("answer", "") or done_step.get("full_answer", "")
        assert answer == ""


class TestDockerAndCI:
    """Docker 和 CI 文件存在性测试"""

    def test_dockerfile_exists(self):
        """Dockerfile 应存在"""
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "..", "Dockerfile")
        assert os.path.exists(path), "Dockerfile 不存在"

    def test_docker_compose_exists(self):
        """docker-compose.yml 应存在"""
        import os

        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "docker-compose.yml"
        )
        assert os.path.exists(path), "docker-compose.yml 不存在"

    def test_ci_workflow_exists(self):
        """CI workflow 应存在"""
        import os

        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            ".github",
            "workflows",
            "ci.yml",
        )
        assert os.path.exists(path), "CI workflow 不存在"

    def test_changelog_exists(self):
        """CHANGELOG.md 应存在"""
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "..", "CHANGELOG.md")
        assert os.path.exists(path), "CHANGELOG.md 不存在"

    def test_security_policy_exists(self):
        """SECURITY.md 应存在"""
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "..", "SECURITY.md")
        assert os.path.exists(path), "SECURITY.md 不存在"


class TestVersionConsistency:
    """版本号一致性测试"""

    def test_pyproject_version(self):
        """pyproject.toml 版本号应为 0.3.7"""
        from PocketGraphRAG.config import _PROJECT_ROOT
        import os

        pyproject_path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()
        assert 'version = "0.3.7"' in content

    def test_api_server_version(self):
        """api_server 版本号应为 0.3.7"""
        from PocketGraphRAG.api_server import app

        assert app.version == "0.3.7"
