import importlib
from types import SimpleNamespace

from PocketGraphRAG import webapp
from PocketGraphRAG.incremental_index import load_doc_map


def test_handle_upload_returns_archived_doc_id(monkeypatch):
    fake_doc = SimpleNamespace(
        content="PocketGraphRAG 可以把文档变成知识图谱。",
        source_type="txt",
        source="demo.txt",
        metadata={},
    )

    monkeypatch.setattr(
        "PocketGraphRAG.data_importer.DataImporter.import_file",
        lambda self, filepath, image_mode="ocr": fake_doc,
    )
    monkeypatch.setattr(webapp, "_archive_doc", lambda **kwargs: "doc_upload_001")

    status, content, doc_id = webapp.handle_upload("demo.txt")

    assert "已读取" in status
    assert content == fake_doc.content
    assert doc_id == "doc_upload_001"


def test_handle_build_index_requires_passed_triples_instead_of_global_cache(
    monkeypatch,
):
    monkeypatch.setattr(webapp, "current_dataset", "example")
    monkeypatch.setattr(webapp, "extracted_triples", [("旧实体", "旧关系", "旧尾实体")])
    monkeypatch.setattr(webapp, "load_runtime_status", lambda: "runtime")
    monkeypatch.setattr(
        webapp, "load_recommended_questions_markdown", lambda dataset: f"qs:{dataset}"
    )

    result = webapp.handle_build_index([], "doc_stale")

    assert result[0] == "请先抽取三元组"
    assert result[1] == "example"
    assert webapp.extracted_triples == []


def test_handle_build_index_passes_doc_id_for_incremental_updates(
    tmp_path, monkeypatch
):
    index_dir = tmp_path / "index"
    user_data_dir = tmp_path / "user_data"
    user_triples_path = user_data_dir / "triples.txt"
    index_dir.mkdir()
    user_data_dir.mkdir()
    (index_dir / "faiss.index").write_text("ready", encoding="utf-8")

    captured = {}

    def fake_add_triples_incremental(
        new_triples,
        model,
        index_dir,
        data_path,
        reverse_link_relations=None,
        relation_templates=None,
        schema=None,
        doc_id=None,
    ):
        captured["new_triples"] = new_triples
        captured["model"] = model
        captured["index_dir"] = index_dir
        captured["data_path"] = data_path
        captured["doc_id"] = doc_id
        return {
            "new_triples": 1,
            "skipped_duplicates": 0,
            "new_entities": 2,
            "affected_entities": 0,
            "new_relations": 1,
            "total_chunks": 3,
        }

    monkeypatch.setattr(webapp, "current_dataset", "user")
    monkeypatch.setattr(webapp, "INDEX_DIR", str(index_dir))
    monkeypatch.setattr(webapp, "USER_DATA_DIR", str(user_data_dir))
    monkeypatch.setattr(webapp, "USER_TRIPLES_PATH", str(user_triples_path))
    monkeypatch.setattr(webapp, "_get_embedding_model", lambda: "fake-model")
    monkeypatch.setattr(webapp, "load_runtime_status", lambda: "runtime")
    monkeypatch.setattr(
        webapp, "load_recommended_questions_markdown", lambda dataset: f"qs:{dataset}"
    )
    monkeypatch.setattr(
        "PocketGraphRAG.incremental_index.add_triples_incremental",
        fake_add_triples_incremental,
    )

    result = webapp.handle_build_index([("实体A", "关联", "实体B")], "doc_user_001")

    assert "增量追加完成" in result[0]
    assert captured["new_triples"] == [("实体A", "关联", "实体B")]
    assert captured["model"] == "fake-model"
    assert captured["index_dir"] == str(index_dir)
    assert captured["data_path"] == str(user_triples_path)
    assert captured["doc_id"] == "doc_user_001"


def test_handle_build_index_records_doc_map_on_full_build(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    user_data_dir = tmp_path / "user_data"
    user_triples_path = user_data_dir / "triples.txt"
    index_dir.mkdir()
    user_data_dir.mkdir()

    monkeypatch.setattr(webapp, "current_dataset", "example")
    monkeypatch.setattr(webapp, "INDEX_DIR", str(index_dir))
    monkeypatch.setattr(webapp, "USER_DATA_DIR", str(user_data_dir))
    monkeypatch.setattr(webapp, "USER_TRIPLES_PATH", str(user_triples_path))
    monkeypatch.setattr(webapp, "load_runtime_status", lambda: "runtime")
    monkeypatch.setattr(
        webapp, "load_recommended_questions_markdown", lambda dataset: f"qs:{dataset}"
    )
    build_index_module = importlib.import_module("PocketGraphRAG.build_index")
    monkeypatch.setattr(build_index_module, "build_index_with_data", lambda path: None)

    result = webapp.handle_build_index([("实体1", "关系1", "实体2")], "doc_full_001")

    assert "索引构建完成" in result[0]
    assert user_triples_path.read_text(encoding="utf-8").strip() == "实体1 | 关系1 | 实体2"
    assert load_doc_map(str(index_dir))["doc_full_001"] == ["实体1|关系1|实体2"]


def test_handle_delete_doc_syncs_graph_when_doc_binding_exists(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    user_data_dir = tmp_path / "user_data"
    docs_dir.mkdir()
    index_dir.mkdir()
    user_data_dir.mkdir()

    monkeypatch.setattr(webapp, "USER_DOCS_DIR", str(docs_dir))
    monkeypatch.setattr(webapp, "DOCS_MANIFEST_PATH", str(docs_dir / "manifest.json"))
    monkeypatch.setattr(webapp, "INDEX_DIR", str(index_dir))
    monkeypatch.setattr(webapp, "USER_TRIPLES_PATH", str(user_data_dir / "triples.txt"))
    monkeypatch.setattr(webapp, "_get_embedding_model", lambda: "fake-model")

    (docs_dir / "doc_1.txt").write_text("content", encoding="utf-8")
    (index_dir / "faiss.index").write_text("ready", encoding="utf-8")
    (user_data_dir / "triples.txt").write_text("实体1 | 关系1 | 实体2\n", encoding="utf-8")
    webapp._save_docs_manifest(
        [
            {
                "id": "doc_1",
                "name": "demo.txt",
                "type": "txt",
                "source": "demo.txt",
                "chars": 7,
                "imported_at": "2026-06-29 18:00:00",
                "content_file": "doc_1.txt",
            }
        ]
    )

    captured = {}

    def fake_remove_document_incremental(
        doc_id,
        model,
        index_dir,
        data_path,
        reverse_link_relations=None,
        relation_templates=None,
        schema=None,
    ):
        captured["doc_id"] = doc_id
        captured["model"] = model
        captured["index_dir"] = index_dir
        captured["data_path"] = data_path
        return {
            "removed_triples": 1,
            "affected_entities": 1,
            "orphan_entities_removed": 0,
            "total_chunks": 1,
        }

    monkeypatch.setattr(
        "PocketGraphRAG.incremental_index.load_doc_map",
        lambda path: {"doc_1": ["实体1|关系1|实体2"]},
    )
    monkeypatch.setattr(
        "PocketGraphRAG.incremental_index.remove_document_incremental",
        fake_remove_document_incremental,
    )

    summary, _, rows = webapp.handle_delete_doc("doc_1")

    assert "已同步更新知识图谱" in summary
    assert captured["doc_id"] == "doc_1"
    assert captured["model"] == "fake-model"
    assert captured["index_dir"] == str(index_dir)
    assert captured["data_path"] == str(user_data_dir / "triples.txt")
    assert rows == []


def test_handle_delete_doc_keeps_fallback_hint_when_doc_binding_missing(
    tmp_path, monkeypatch
):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    user_data_dir = tmp_path / "user_data"
    docs_dir.mkdir()
    index_dir.mkdir()
    user_data_dir.mkdir()

    monkeypatch.setattr(webapp, "USER_DOCS_DIR", str(docs_dir))
    monkeypatch.setattr(webapp, "DOCS_MANIFEST_PATH", str(docs_dir / "manifest.json"))
    monkeypatch.setattr(webapp, "INDEX_DIR", str(index_dir))
    monkeypatch.setattr(webapp, "USER_TRIPLES_PATH", str(user_data_dir / "triples.txt"))

    (docs_dir / "doc_1.txt").write_text("content", encoding="utf-8")
    (index_dir / "faiss.index").write_text("ready", encoding="utf-8")
    (user_data_dir / "triples.txt").write_text("实体1 | 关系1 | 实体2\n", encoding="utf-8")
    webapp._save_docs_manifest(
        [
            {
                "id": "doc_1",
                "name": "demo.txt",
                "type": "txt",
                "source": "demo.txt",
                "chars": 7,
                "imported_at": "2026-06-29 18:00:00",
                "content_file": "doc_1.txt",
            }
        ]
    )

    monkeypatch.setattr("PocketGraphRAG.incremental_index.load_doc_map", lambda path: {})

    summary, _, rows = webapp.handle_delete_doc("doc_1")

    assert "不会自动更新已构建的知识图谱" in summary
    assert "尚未绑定到图谱映射" in summary
    assert rows == []
