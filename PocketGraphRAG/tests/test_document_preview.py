"""文档预览端点测试（v0.3.7）

测试 ``GET /api/documents/{filename}/raw`` 端点：
- 正常预览 .txt / .md 文件
- 不存在的文件 → 404
- 路径穿越攻击 → 400
- 超大文件自动截断 → truncated=true
- OpenAPI schema 包含 Documents tag
"""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    """测试客户端 fixture，临时 USER_DOCS_DIR + mock RAG 系统"""
    # 在临时目录下创建测试文件
    (tmp_path / "hello.txt").write_text("Hello, PocketGraphRAG!", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Notes\n\nThis is a **markdown** doc.", encoding="utf-8")

    # 超大文件（> 50_000 字符）
    big_content = "A" * 60_000
    (tmp_path / "big.txt").write_text(big_content, encoding="utf-8")

    with patch("PocketGraphRAG.api_server._rag") as mock_rag:
        mock_rag.kg_retriever = None
        with patch("PocketGraphRAG.api_server.USER_DOCS_DIR", str(tmp_path)):
            from PocketGraphRAG.api_server import app
            with TestClient(app) as c:
                yield c


class TestDocumentPreview:
    """文档预览端点测试"""

    def test_preview_txt_file(self, client):
        """预览 .txt 文件应返回正确内容"""
        resp = client.get("/api/documents/hello.txt/raw")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "hello.txt"
        assert data["source_type"] == "txt"
        assert "Hello, PocketGraphRAG!" in data["content"]
        assert data["total_chars"] == len("Hello, PocketGraphRAG!")
        assert data["truncated"] is False

    def test_preview_markdown_file(self, client):
        """预览 .md 文件应返回正确内容"""
        resp = client.get("/api/documents/notes.md/raw")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "notes.md"
        assert data["source_type"] in ("md", "markdown")
        # DataImporter._import_markdown 会去掉 markdown 语法符号，只保留纯文本
        assert "Notes" in data["content"]
        assert "markdown" in data["content"].lower()
        assert data["truncated"] is False

    def test_preview_nonexistent_file_returns_404(self, client):
        """预览不存在的文件应返回 404"""
        resp = client.get("/api/documents/nonexistent.txt/raw")
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_preview_path_traversal_blocked(self, client):
        """路径穿越攻击应被阻止（返回 400）"""
        # 尝试 ../etc/passwd
        resp = client.get("/api/documents/..%2F..%2Fetc%2Fpasswd/raw")
        # basename 会剥离 ../，最终查找的是 passwd（不存在 → 404）或被 _safe_doc_path 拒绝（400）
        assert resp.status_code in (400, 404)

    def test_preview_large_file_truncated(self, client):
        """超大文件应自动截断并标记 truncated=true"""
        resp = client.get("/api/documents/big.txt/raw")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chars"] == 60_000
        assert data["truncated"] is True
        # 截断后的 content 不超过上限
        assert len(data["content"]) <= 50_000

    def test_preview_response_model_fields(self, client):
        """响应应包含所有 DocumentPreviewResponse 字段"""
        resp = client.get("/api/documents/hello.txt/raw")
        data = resp.json()
        required_fields = {
            "filename",
            "source_type",
            "title",
            "content",
            "total_chars",
            "truncated",
        }
        assert required_fields.issubset(set(data.keys()))


class TestOpenAPITags:
    """OpenAPI tags 分组测试（Swagger 文档专业化）"""

    def test_openapi_tags_defined(self, client):
        """OpenAPI schema 应定义 5 个 tag 分组"""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        tags = spec.get("tags", [])
        tag_names = {t["name"] for t in tags}
        expected = {"QA", "Graph", "Documents", "Settings", "System"}
        assert expected.issubset(tag_names), f"缺少 tags: {expected - tag_names}"

    def test_routes_have_tags(self, client):
        """所有 /api/* 路由都应分配到 tag 分组"""
        resp = client.get("/openapi.json")
        spec = resp.json()
        paths = spec.get("paths", {})
        untagged = []
        for path, methods in paths.items():
            if not path.startswith("/api/"):
                continue
            for method, info in methods.items():
                if method in ("get", "post", "delete", "put", "patch"):
                    if not info.get("tags"):
                        untagged.append(f"{method.upper()} {path}")
        assert not untagged, f"以下路由缺少 tags: {untagged}"

    def test_documents_routes_tagged_documents(self, client):
        """/api/documents/* 路由应标记为 Documents tag"""
        resp = client.get("/openapi.json")
        spec = resp.json()
        paths = spec.get("paths", {})
        doc_paths = [
            p for p in paths if p.startswith("/api/documents")
        ]
        assert len(doc_paths) >= 5, "应有至少 5 个 documents 路由"
        for path in doc_paths:
            for method, info in paths[path].items():
                if method in ("get", "post", "delete", "put", "patch"):
                    assert "Documents" in info.get("tags", []), (
                        f"{method.upper()} {path} 未标记 Documents tag"
                    )
