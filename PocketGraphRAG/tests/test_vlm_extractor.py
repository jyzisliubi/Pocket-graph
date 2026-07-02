"""VLM 多模态抽取模块单元测试"""

import base64
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG.vlm_extractor import (
    _call_vlm_openai_compatible,
    encode_image_to_base64,
    get_image_mime_type,
    get_vlm_provider,
    has_vlm,
    is_image_file,
)


class TestImageUtils:
    def test_is_image_file_true(self):
        assert is_image_file("test.jpg")
        assert is_image_file("test.jpeg")
        assert is_image_file("test.png")
        assert is_image_file("test.webp")
        assert is_image_file("test.gif")
        assert is_image_file("test.bmp")
        assert is_image_file("test.tiff")
        assert is_image_file("test.tif")

    def test_is_image_file_false(self):
        assert not is_image_file("test.txt")
        assert not is_image_file("test.pdf")
        assert not is_image_file("test.md")
        assert not is_image_file("test.doc")

    def test_is_image_file_case_insensitive(self):
        assert is_image_file("test.JPG")
        assert is_image_file("test.PNG")
        assert is_image_file("test.JPEG")

    def test_get_image_mime_type(self):
        assert get_image_mime_type("test.jpg") == "image/jpeg"
        assert get_image_mime_type("test.jpeg") == "image/jpeg"
        assert get_image_mime_type("test.png") == "image/png"
        assert get_image_mime_type("test.gif") == "image/gif"
        assert get_image_mime_type("test.webp") == "image/webp"
        assert get_image_mime_type("test.bmp") == "image/bmp"
        assert get_image_mime_type("test.unknown") == "image/jpeg"  # 默认值

    def test_encode_image_to_base64_file_not_exists(self):
        result = encode_image_to_base64("/nonexistent/file.png")
        assert result is None

    def test_encode_image_to_base64_success(self):
        # 创建一个临时图片文件
        img_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwAE"
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            temp_path = f.name

        try:
            result = encode_image_to_base64(temp_path)
            assert result is not None
            assert isinstance(result, str)
            # 验证可以解码回去
            decoded = base64.b64decode(result)
            assert len(decoded) > 0
        finally:
            os.unlink(temp_path)


class TestHasVLM:
    def test_has_vlm_no_config(self):
        """has_vlm 不抛异常，返回 bool"""
        result = has_vlm()
        assert isinstance(result, bool)

    def test_get_vlm_provider_returns_string(self):
        result = get_vlm_provider()
        assert isinstance(result, str)
        assert len(result) > 0


class TestVLMCall:
    def test_call_vlm_openai_compatible_invalid_url(self):
        """调用失败时返回 None"""
        result = _call_vlm_openai_compatible(
            api_base="http://invalid.local:9999",
            api_key="test-key",
            model="test-model",
            image_base64="fake-base64",
            image_mime="image/png",
            prompt="test prompt",
            label="Test",
        )
        # 应该返回 None（连接失败）
        assert result is None


class TestDataImporterImageSupport:
    def test_data_importer_supports_images(self):
        """DataImporter 应该支持图片文件类型"""
        from PocketGraphRAG.data_importer import DataImporter

        DataImporter()  # 验证可以实例化
        # 验证支持的扩展名
        supported_exts = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
        for ext in supported_exts:
            assert is_image_file(f"test{ext}")

    def test_data_importer_import_image_method_exists(self):
        """_import_image 方法应该存在"""
        from PocketGraphRAG.data_importer import DataImporter

        assert hasattr(DataImporter, "_import_image")

    def test_data_importer_import_file_accepts_image_mode(self):
        """import_file 应该接受 image_mode 参数"""
        import inspect

        from PocketGraphRAG.data_importer import DataImporter

        sig = inspect.signature(DataImporter.import_file)
        assert "image_mode" in sig.parameters


class TestPlaywrightSupport:
    def test_data_importer_has_playwright_method(self):
        """_fetch_with_playwright 方法应该存在"""
        from PocketGraphRAG.data_importer import DataImporter

        assert hasattr(DataImporter, "_fetch_with_playwright")

    def test_import_url_accepts_use_playwright_param(self):
        """_import_url 应该接受 use_playwright 参数"""
        import inspect

        from PocketGraphRAG.data_importer import DataImporter

        sig = inspect.signature(DataImporter._import_url)
        assert "use_playwright" in sig.parameters

    def test_import_pdf_accepts_enable_ocr_param(self):
        """_import_pdf 应该接受 enable_ocr 参数"""
        import inspect

        from PocketGraphRAG.data_importer import DataImporter

        sig = inspect.signature(DataImporter._import_pdf)
        assert "enable_ocr" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
