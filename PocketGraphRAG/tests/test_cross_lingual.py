"""bge-m3 / 跨语言检索 / 模型指纹 校验测试

覆盖：
1. config.py 的 EMBEDDING_MODEL_ALIASES 别名解析
2. faiss_store / factory 的 dimension=None 动态推断
3. build_index 的 embedding_model.json 指纹写入与校验
4. rag_system 的跨语言 query 检测
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========================
# 1. 别名解析
# ========================


class TestEmbeddingModelAliases:
    """EMBEDDING_MODEL_ALIASES 与 _resolve_embedding_model"""

    def test_alias_table_contains_bge_m3(self):
        from PocketGraphRAG.config import EMBEDDING_MODEL_ALIASES

        assert "bge-m3" in EMBEDDING_MODEL_ALIASES
        assert EMBEDDING_MODEL_ALIASES["bge-m3"] == "BAAI/bge-m3"

    def test_resolve_bge_m3_alias(self):
        from PocketGraphRAG.config import _resolve_embedding_model

        assert _resolve_embedding_model("bge-m3") == "BAAI/bge-m3"

    def test_resolve_full_name_passthrough(self):
        from PocketGraphRAG.config import _resolve_embedding_model

        # 完整名且本地不存在时应原样返回
        assert _resolve_embedding_model("BAAI/bge-m3") == "BAAI/bge-m3"

    def test_resolve_case_insensitive(self):
        from PocketGraphRAG.config import _resolve_embedding_model

        assert _resolve_embedding_model("BGE-M3") == "BAAI/bge-m3"
        assert _resolve_embedding_model("BGE-M3") == "BAAI/bge-m3"

    def test_resolve_unknown_passthrough(self):
        from PocketGraphRAG.config import _resolve_embedding_model

        assert _resolve_embedding_model("some-other-model") == "some-other-model"

    def test_resolve_empty(self):
        from PocketGraphRAG.config import _resolve_embedding_model

        assert _resolve_embedding_model("") == ""

    def test_resolve_local_path_priority(self, tmp_path, monkeypatch):
        """本地缓存路径优先于 HuggingFace 名称"""
        from PocketGraphRAG import config as cfg

        # 构造一个本地路径，让 _resolve_embedding_model 优先返回它
        # 用 monkeypatch 修改 _PROJECT_ROOT 让本地路径指向 tmp_path
        monkeypatch.setattr(cfg, "_PROJECT_ROOT", str(tmp_path))
        local_dir = tmp_path / "models" / "BAAI" / "bge-m3"
        local_dir.mkdir(parents=True)
        result = cfg._resolve_embedding_model("bge-m3")
        assert "models" in result
        assert "bge-m3" in result


# ========================
# 2. 动态维度（消除硬编码 512）
# ========================


class TestDynamicDimension:
    """FAISSVectorStore / factory 的 dimension=None 支持"""

    def test_faiss_store_accepts_none_dimension(self):
        from PocketGraphRAG.core.storages.faiss_store import FAISSVectorStore

        # 不传 dimension 应能用 None 构造（懒加载）
        store = FAISSVectorStore(model=None, dimension=None)
        assert store is not None

    def test_factory_accepts_none_dimension_faiss(self):
        from PocketGraphRAG.core.storages.factory import get_vector_store

        store = get_vector_store(backend="faiss", dimension=None)
        assert store is not None

    def test_factory_pgvector_none_dimension_without_model_raises(self):
        from PocketGraphRAG.core.storages.factory import get_vector_store

        with pytest.raises(ValueError) as exc_info:
            get_vector_store(backend="pgvector", dimension=None, model=None)
        assert "维度" in str(exc_info.value) or "dimension" in str(exc_info.value).lower()

    def test_factory_pgvector_none_dimension_with_model(self):
        """pgvector 后端从 model 推断维度"""
        from PocketGraphRAG.core.storages.factory import get_vector_store

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        # 实际 PgVectorStore 会尝试连接 DB，需要 mock
        with patch(
            "PocketGraphRAG.core.storages.pgvector_store.PgVectorStore"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            get_vector_store(backend="pgvector", dimension=None, model=mock_model)
            # 验证传入 PgVectorStore 的 embedding_dim=1024
            _, kwargs = mock_cls.call_args
            assert kwargs.get("embedding_dim") == 1024


# ========================
# 3. 模型指纹（embedding_model.json）
# ========================


class TestEmbeddingModelFingerprint:
    """FAISSIndex.save 写入指纹 / load 校验"""

    def test_save_writes_fingerprint(self, tmp_path):
        """save 应写入 embedding_model.json"""
        from PocketGraphRAG.build_index import FAISSIndex
        import faiss

        idx = FAISSIndex(dimension=8)
        idx.index = faiss.IndexFlatIP(8)
        idx.texts = ["hello"]
        idx.metadatas = [{"entity": "E1"}]
        idx._embeddings = np.zeros((1, 8), dtype="float32")

        idx.save(str(tmp_path))

        fp_path = tmp_path / "embedding_model.json"
        assert fp_path.exists()
        with open(fp_path, encoding="utf-8") as f:
            fp = json.load(f)
        assert "model" in fp
        assert "dimension" in fp
        assert fp["dimension"] == 8

    def test_load_dimension_mismatch_raises(self, tmp_path, monkeypatch):
        """维度不一致时 load 应抛 RuntimeError"""
        from PocketGraphRAG.build_index import FAISSIndex
        import faiss

        # 构造一个 8 维索引
        idx = FAISSIndex(dimension=8)
        idx.index = faiss.IndexFlatIP(8)
        idx.texts = ["hello"]
        idx.metadatas = [{"entity": "E1"}]
        idx._embeddings = np.zeros((1, 8), dtype="float32")
        idx.save(str(tmp_path))

        # 篡改指纹：把 dimension 改成 1024，模拟旧索引与新模型不匹配
        fp_path = tmp_path / "embedding_model.json"
        with open(fp_path, encoding="utf-8") as f:
            fp = json.load(f)
        fp["dimension"] = 1024
        with open(fp_path, "w", encoding="utf-8") as f:
            json.dump(fp, f)

        # mock 模型：当前模型 8 维
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 8

        with pytest.raises(RuntimeError) as exc_info:
            FAISSIndex.load(str(tmp_path), mock_model)
        assert "维度" in str(exc_info.value) or "不匹配" in str(exc_info.value)

    def test_load_dimension_match_success(self, tmp_path):
        """维度一致时正常加载"""
        from PocketGraphRAG.build_index import FAISSIndex
        import faiss

        idx = FAISSIndex(dimension=8)
        idx.index = faiss.IndexFlatIP(8)
        idx.texts = ["hello"]
        idx.metadatas = [{"entity": "E1"}]
        idx._embeddings = np.zeros((1, 8), dtype="float32")
        idx.save(str(tmp_path))

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 8

        loaded = FAISSIndex.load(str(tmp_path), mock_model)
        assert loaded.dimension == 8
        assert loaded.texts == ["hello"]

    def test_load_without_fingerprint_backward_compat(self, tmp_path):
        """无 embedding_model.json 时（旧索引）应能正常加载（向后兼容）"""
        from PocketGraphRAG.build_index import FAISSIndex
        import faiss

        idx = FAISSIndex(dimension=8)
        idx.index = faiss.IndexFlatIP(8)
        idx.texts = ["hello"]
        idx.metadatas = [{"entity": "E1"}]
        idx._embeddings = np.zeros((1, 8), dtype="float32")
        idx.save(str(tmp_path))

        # 删除指纹
        (tmp_path / "embedding_model.json").unlink()

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 8

        loaded = FAISSIndex.load(str(tmp_path), mock_model)
        assert loaded.dimension == 8

    def test_load_model_name_mismatch_only_warns(self, tmp_path, caplog):
        """模型名不一致但维度一致时仅 warning，不抛错"""
        from PocketGraphRAG.build_index import FAISSIndex
        import faiss
        import logging

        idx = FAISSIndex(dimension=8)
        idx.index = faiss.IndexFlatIP(8)
        idx.texts = ["hello"]
        idx.metadatas = [{"entity": "E1"}]
        idx._embeddings = np.zeros((1, 8), dtype="float32")
        idx.save(str(tmp_path))

        # 篡改模型名但保持维度
        fp_path = tmp_path / "embedding_model.json"
        with open(fp_path, encoding="utf-8") as f:
            fp = json.load(f)
        fp["model"] = "old-model-bge-small"
        with open(fp_path, "w", encoding="utf-8") as f:
            json.dump(fp, f)

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 8

        with caplog.at_level(logging.WARNING):
            loaded = FAISSIndex.load(str(tmp_path), mock_model)
        assert loaded.dimension == 8
        # 应有 warning 日志
        assert any(
            "Embedding 模型切换" in rec.message or "切换" in rec.message
            for rec in caplog.records
        )


# ========================
# 4. 跨语言 query 检测
# ========================


class TestCrossLingualHint:
    """PocketGraphRAG._cross_lingual_hint"""

    def test_english_query_with_zh_model_returns_hint(self, monkeypatch):
        from PocketGraphRAG.rag_system import PocketGraphRAG as RagCls
        import PocketGraphRAG.rag_system as rs_mod

        monkeypatch.setattr(rs_mod, "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

        instance = MagicMock()
        hint = RagCls._cross_lingual_hint(instance, "What is the capital of France?")
        assert hint is not None
        assert "bge-m3" in hint.lower() or "跨语言" in hint

    def test_chinese_query_with_zh_model_returns_none(self, monkeypatch):
        from PocketGraphRAG.rag_system import PocketGraphRAG as RagCls
        import PocketGraphRAG.rag_system as rs_mod

        monkeypatch.setattr(rs_mod, "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

        instance = MagicMock()
        hint = RagCls._cross_lingual_hint(instance, "法国的首都是哪里？")
        assert hint is None

    def test_english_query_with_bge_m3_model_returns_none(self, monkeypatch):
        """已用 bge-m3 时不应提示"""
        from PocketGraphRAG.rag_system import PocketGraphRAG as RagCls
        import PocketGraphRAG.rag_system as rs_mod

        monkeypatch.setattr(rs_mod, "EMBEDDING_MODEL", "BAAI/bge-m3")

        instance = MagicMock()
        hint = RagCls._cross_lingual_hint(instance, "What is the capital of France?")
        assert hint is None

    def test_short_query_returns_none(self, monkeypatch):
        """太短的 query 不判断"""
        from PocketGraphRAG.rag_system import PocketGraphRAG as RagCls
        import PocketGraphRAG.rag_system as rs_mod

        monkeypatch.setattr(rs_mod, "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

        instance = MagicMock()
        hint = RagCls._cross_lingual_hint(instance, "hi")
        assert hint is None

    def test_empty_query_returns_none(self, monkeypatch):
        from PocketGraphRAG.rag_system import PocketGraphRAG as RagCls
        import PocketGraphRAG.rag_system as rs_mod

        monkeypatch.setattr(rs_mod, "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

        instance = MagicMock()
        hint = RagCls._cross_lingual_hint(instance, "")
        assert hint is None

    def test_mixed_query_majority_chinese_returns_none(self, monkeypatch):
        """中文占多数的混合 query 不提示"""
        from PocketGraphRAG.rag_system import PocketGraphRAG as RagCls
        import PocketGraphRAG.rag_system as rs_mod

        monkeypatch.setattr(rs_mod, "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

        instance = MagicMock()
        # 中文占多数
        hint = RagCls._cross_lingual_hint(instance, "请告诉我 France 的首都城市是哪里")
        assert hint is None
