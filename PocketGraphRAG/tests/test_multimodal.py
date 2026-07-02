"""多模态模块单元测试"""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_model():
    """创建一个 mock 的文本 embedding 模型（模拟纯文本模型）"""
    model = MagicMock(spec=["encode"])

    # 模拟文本编码
    def mock_encode(texts, **kwargs):
        n = len(texts)
        rng = np.random.RandomState(42)
        vecs = rng.randn(n, 128).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        return vecs

    model.encode = mock_encode
    return model


@pytest.fixture
def mock_clip_model():
    """创建一个 mock 的 CLIP 多模态模型"""
    model = MagicMock(spec=["encode", "encode_text", "encode_image"])

    def mock_encode_text(texts, **kwargs):
        n = len(texts)
        rng = np.random.RandomState(42)
        vecs = rng.randn(n, 128).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        return vecs

    def mock_encode_image(images, **kwargs):
        n = len(images)
        rng = np.random.RandomState(43)
        vecs = rng.randn(n, 128).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        return vecs

    model.encode = mock_encode_text
    model.encode_text = mock_encode_text
    model.encode_image = mock_encode_image
    return model


@pytest.fixture
def temp_index_dir(tmp_path):
    """创建临时索引目录"""
    d = tmp_path / "multimodal_index"
    d.mkdir(exist_ok=True)
    return str(d)


class TestMultimodalIndex:
    def test_init_empty(self, mock_model, temp_index_dir):
        """测试初始化空索引"""
        from PocketGraphRAG.multimodal import MultimodalIndex

        index = MultimodalIndex(temp_index_dir, model=mock_model)
        assert index.available is False
        assert index.image_index is None
        assert len(index.image_paths) == 0

    def test_add_and_search_by_text(self, mock_model, temp_index_dir):
        """测试添加图片并文本搜索"""
        from PocketGraphRAG.multimodal import MultimodalIndex

        index = MultimodalIndex(temp_index_dir, model=mock_model)

        # 创建临时图片文件（不需要真实图片，mock 会处理）
        img_paths = []
        for i in range(3):
            img_path = os.path.join(temp_index_dir, f"test_{i}.jpg")
            # 创建空文件
            with open(img_path, "w") as f:
                f.write("")
            img_paths.append(img_path)

        # 添加图片
        index.add_images("实体A", img_paths[:2])
        index.add_images("实体B", img_paths[2:])

        # 验证添加成功
        assert index.available is True
        assert len(index.image_paths) == 3
        assert "实体A" in index.entity_image_map
        assert "实体B" in index.entity_image_map
        assert len(index.entity_image_map["实体A"]) == 2
        assert len(index.entity_image_map["实体B"]) == 1

    def test_search_by_text_returns_results(self, mock_model, temp_index_dir):
        """测试文本搜图返回结果"""
        from PocketGraphRAG.multimodal import MultimodalIndex

        index = MultimodalIndex(temp_index_dir, model=mock_model)

        # 添加一些图片
        img_paths = []
        for i in range(5):
            img_path = os.path.join(temp_index_dir, f"img_{i}.jpg")
            with open(img_path, "w") as f:
                f.write("")
            img_paths.append(img_path)

        index.add_images("测试实体", img_paths)

        # 搜索
        results = index.search_by_text("测试", top_k=3, threshold=0.0)
        assert len(results) <= 3
        for entity, path, score in results:
            assert isinstance(entity, str)
            assert isinstance(path, str)
            assert isinstance(score, float)

    def test_get_entity_images(self, mock_model, temp_index_dir):
        """测试获取实体的图片"""
        from PocketGraphRAG.multimodal import MultimodalIndex

        index = MultimodalIndex(temp_index_dir, model=mock_model)

        img_paths = []
        for i in range(2):
            p = os.path.join(temp_index_dir, f"img_{i}.jpg")
            with open(p, "w") as f:
                f.write("")
            img_paths.append(p)

        index.add_images("实体X", img_paths)

        images = index.get_entity_images("实体X")
        assert len(images) == 2

        # 不存在的实体返回空列表
        assert index.get_entity_images("不存在的实体") == []

    def test_stats(self, mock_model, temp_index_dir):
        """测试统计信息"""
        from PocketGraphRAG.multimodal import MultimodalIndex

        index = MultimodalIndex(temp_index_dir, model=mock_model)
        stats = index.stats()
        assert stats["available"] is False
        assert stats["total_images"] == 0
        assert stats["total_entities_with_images"] == 0

    def test_save_and_load(self, mock_model, temp_index_dir):
        """测试保存和加载索引"""
        from PocketGraphRAG.multimodal import MultimodalIndex

        # 创建并添加
        index1 = MultimodalIndex(temp_index_dir, model=mock_model)

        img_paths = []
        for i in range(2):
            p = os.path.join(temp_index_dir, f"img_{i}.jpg")
            with open(p, "w") as f:
                f.write("")
            img_paths.append(p)

        index1.add_images("测试实体", img_paths)
        index1.save()

        # 重新加载
        index2 = MultimodalIndex(temp_index_dir, model=mock_model)
        assert index2.available is True
        assert len(index2.image_paths) == 2
        assert "测试实体" in index2.entity_image_map

    def test_search_by_image_not_available_when_empty(self, mock_model, temp_index_dir):
        """测试空索引时图搜图返回空"""
        from PocketGraphRAG.multimodal import MultimodalIndex

        index = MultimodalIndex(temp_index_dir, model=mock_model)
        results = index.search_by_image("nonexistent.jpg")
        assert results == []

    def test_nonexistent_image_skipped(self, mock_model, temp_index_dir):
        """测试不存在的图片会被跳过"""
        from PocketGraphRAG.multimodal import MultimodalIndex

        index = MultimodalIndex(temp_index_dir, model=mock_model)
        index.add_images("实体", ["不存在的图片.jpg"])

        # 应该没有添加成功
        assert len(index.image_paths) == 0


class TestIsMultimodalModel:
    def test_clip_models(self):
        """测试 CLIP 模型识别"""
        from PocketGraphRAG.multimodal import is_multimodal_model

        assert is_multimodal_model("clip-vit-base-patch32") is True
        assert is_multimodal_model("openai/clip-vit-large") is True
        assert is_multimodal_model("ViT-B/32") is True
        assert is_multimodal_model("multimodal-bert") is True

    def test_text_only_models(self):
        """测试纯文本模型识别"""
        from PocketGraphRAG.multimodal import is_multimodal_model

        assert is_multimodal_model("bge-small-zh") is False
        assert is_multimodal_model("text-embedding-3-small") is False
        assert is_multimodal_model("all-MiniLM-L6-v2") is False
