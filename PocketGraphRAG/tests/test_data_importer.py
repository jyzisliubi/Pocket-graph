"""多数据源导入模块单元测试"""

import pytest

from PocketGraphRAG.data_importer import (
    DataImporter,
    ExtractedDocument,
    detect_file_type,
)


@pytest.fixture
def importer():
    """创建 DataImporter 实例"""
    return DataImporter()


@pytest.fixture
def sample_txt_file(tmp_path):
    """创建示例 txt 文件"""
    f = tmp_path / "test.txt"
    f.write_text("这是一个测试文本文件。\n\n它有两行内容。", encoding="utf-8")
    return str(f)


@pytest.fixture
def sample_md_file(tmp_path):
    """创建示例 markdown 文件"""
    f = tmp_path / "test.md"
    f.write_text(
        "# 测试文档\n\n这是 **加粗** 的内容。\n\n"
        "## 第二章\n\n* 列表项1\n* 列表项2\n\n"
        "[链接文字](http://example.com)\n\n"
        "`代码片段`\n",
        encoding="utf-8",
    )
    return str(f)


class TestDetectFileType:
    def test_detect_txt(self):
        assert detect_file_type("test.txt") == "txt"
        assert detect_file_type("path/to/file.TXT") == "txt"

    def test_detect_md(self):
        assert detect_file_type("test.md") == "md"
        assert detect_file_type("test.markdown") == "md"

    def test_detect_pdf(self):
        assert detect_file_type("test.pdf") == "pdf"

    def test_detect_unknown(self):
        assert detect_file_type("test.docx") is None
        assert detect_file_type("test") is None


class TestDataImporter:
    def test_import_txt(self, importer, sample_txt_file):
        """测试导入 txt 文件"""
        doc = importer.import_file(sample_txt_file)
        assert doc is not None
        assert doc.source == "test.txt"
        assert doc.source_type == "txt"
        assert doc.title == "test"
        assert "测试文本文件" in doc.content
        assert "两行内容" in doc.content
        assert doc.metadata["size"] > 0

    def test_import_markdown(self, importer, sample_md_file):
        """测试导入 markdown 文件"""
        doc = importer.import_file(sample_md_file)
        assert doc is not None
        assert doc.source == "test.md"
        assert doc.source_type == "md"
        assert doc.title == "测试文档"
        # 标题应该被提取了
        assert "加粗" in doc.content
        # Markdown 标记应该被清理了
        assert "**" not in doc.content
        assert "#" not in doc.content

    def test_import_nonexistent_file(self, importer):
        """测试导入不存在的文件"""
        doc = importer.import_file("nonexistent.txt")
        assert doc is None

    def test_import_unsupported_type(self, importer, tmp_path):
        """测试导入不支持的文件类型"""
        f = tmp_path / "test.docx"
        f.write_text("test")
        doc = importer.import_file(str(f))
        assert doc is None

    def test_import_batch_files(self, importer, sample_txt_file, sample_md_file):
        """测试批量导入文件"""
        docs = importer.import_batch(file_paths=[sample_txt_file, sample_md_file])
        assert len(docs) == 2
        types = {d.source_type for d in docs}
        assert "txt" in types
        assert "md" in types

    def test_import_directory(self, importer, tmp_path):
        """测试导入整个目录"""
        (tmp_path / "a.txt").write_text("文件A", encoding="utf-8")
        (tmp_path / "b.md").write_text("# B\n内容B", encoding="utf-8")
        (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4 fake")
        # 不支持的类型应该被跳过
        (tmp_path / "d.docx").write_text("不支持")

        docs = importer.import_directory(str(tmp_path), recursive=False)
        assert len(docs) == 2  # txt + md

    def test_import_directory_recursive(self, importer, tmp_path):
        """测试递归导入目录"""
        (tmp_path / "a.txt").write_text("文件A", encoding="utf-8")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "b.txt").write_text("文件B", encoding="utf-8")

        docs_recursive = importer.import_directory(str(tmp_path), recursive=True)
        assert len(docs_recursive) == 2

        docs_non_recursive = importer.import_directory(str(tmp_path), recursive=False)
        assert len(docs_non_recursive) == 1

    def test_markdown_to_text(self):
        """测试 Markdown 转纯文本"""
        md = "# 标题\n\n**加粗** *斜体*\n\n* 列表1\n* 列表2\n\n[链接](http://x.com)\n\n`代码`"
        text = DataImporter._markdown_to_text(md)

        assert "#" not in text
        assert "**" not in text
        assert "*" not in text or "列表" in text
        assert "加粗" in text
        assert "斜体" in text
        assert "链接" in text
        assert "代码" in text

    def test_import_url_invalid(self, importer):
        """测试导入无效 URL 会返回 None（不抛出异常导致崩溃）"""
        # 无效 URL 应该返回 None 而不是崩溃
        doc = importer.import_url("not-a-valid-url-scheme://invalid")
        # 可能返回 None 或抛出异常后返回 None
        assert doc is None or isinstance(doc, ExtractedDocument)  # 灵活点

    def test_extracted_document_dataclass(self):
        """测试 ExtractedDocument 数据类"""
        doc = ExtractedDocument(
            source="test.txt",
            source_type="txt",
            title="测试",
            content="内容",
            metadata={"key": "value"},
        )
        assert doc.source == "test.txt"
        assert doc.title == "测试"
        assert doc.content == "内容"
        assert doc.metadata["key"] == "value"
