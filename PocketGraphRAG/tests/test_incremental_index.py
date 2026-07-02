"""Tests for incremental indexing (增量索引).

覆盖：
- 新实体增量 add：旧索引保留，新 chunk 可检索
- 已有实体获得新三元组：旧 chunk 被替换，新内容可检索
- 重复三元组去重：skipped_duplicates 正确，索引不变
- 新关系 append 到关系索引
- 旧索引（无 embeddings.npy / 无 manifest）向后兼容迁移
- manifest 持久化与重载
- reset_index 全量重置
- KGProcessor.process_entities / add_triples 单元测试

使用 MockSentenceTransformer 避免下载真实 BGE 模型。
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from PocketGraphRAG.build_index import FAISSIndex, build_index_with_data
from PocketGraphRAG.data_processor import KGProcessor
from PocketGraphRAG.incremental_index import (
    MANIFEST_FILENAME,
    add_triples_incremental,
    build_manifest_from_data,
    load_manifest,
    reset_index,
    save_manifest,
)


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing (deterministic 32-dim embeddings)."""

    def __init__(self, model_name=None, **kwargs):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            # 确定性 hash：相同文本 → 相同向量
            h = abs(hash(text)) % (2**30)
            vec = np.zeros(32, dtype=np.float32)
            vec[0] = (h % 1000) / 1000.0
            vec[1] = ((h // 1000) % 1000) / 1000.0
            vec[2] = ((h // 1000000) % 1000) / 1000.0
            vec[3:8] = 0.1
            # 加入文本长度特征，让不同长度文本向量有差异
            vec[8] = min(len(text) / 100.0, 1.0)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


@pytest.fixture
def mock_model():
    return MockSentenceTransformer()


@pytest.fixture
def base_triples_file(temp_dir):
    """基础三元组文件（水稻病害 demo 子集）"""
    triples = [
        "稻瘟病|属于|真菌性病害",
        "稻瘟病|防治药剂|三环唑",
        "三环唑|属于|杀菌剂",
        "水稻纹枯病|属于|真菌性病害",
        "水稻纹枯病|防治药剂|井冈霉素",
        "井冈霉素|属于|抗生素类杀菌剂",
    ]
    path = os.path.join(temp_dir, "base_triples.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(triples))
    return path


@pytest.fixture
def built_index(temp_dir, base_triples_file, mock_model):
    """构建好的基础索引（用 mock 模型）"""
    index_dir = os.path.join(temp_dir, "index")
    with patch(
        "PocketGraphRAG.build_index.SentenceTransformer", return_value=mock_model
    ):
        build_index_with_data(base_triples_file, index_dir=index_dir, run_tests=False)
    return index_dir


# ============================================================
# KGProcessor 新方法单元测试
# ============================================================


class TestKGProcessorNewMethods:
    def test_process_entities_returns_only_specified(self, sample_triples_file):
        proc = KGProcessor(sample_triples_file)
        proc.load_triples()
        all_chunks = proc.process()
        all_entities = [c["entity"] for c in all_chunks]

        # 只取前两个实体
        subset = proc.process_entities(all_entities[:2])
        assert len(subset) == 2
        assert [c["entity"] for c in subset] == all_entities[:2]

    def test_process_entities_dedup_preserves_order(self, sample_triples_file):
        proc = KGProcessor(sample_triples_file)
        proc.load_triples()
        all_entities = list(proc.entity_relations.keys())
        target = [all_entities[0], all_entities[0], all_entities[1]]  # 含重复

        subset = proc.process_entities(target)
        assert len(subset) == 2  # 重复被去重

    def test_process_entities_unknown_entity_skipped(self, sample_triples_file):
        proc = KGProcessor(sample_triples_file)
        proc.load_triples()
        subset = proc.process_entities(["不存在的实体"])
        assert subset == []

    def test_process_entities_format_matches_process(self, sample_triples_file):
        """process_entities 输出格式必须与 process() 一致"""
        proc = KGProcessor(sample_triples_file)
        proc.load_triples()
        all_chunks = {c["entity"]: c for c in proc.process()}

        for entity in all_chunks:
            single = proc.process_entities([entity])
            assert len(single) == 1
            assert single[0]["text"] == all_chunks[entity]["text"]
            assert (
                single[0]["metadata"]["num_triples"]
                == all_chunks[entity]["metadata"]["num_triples"]
            )

    def test_add_triples_merges_new(self, sample_triples_file):
        proc = KGProcessor(sample_triples_file)
        proc.load_triples()
        before = len(proc.triples)

        added = proc.add_triples([("新实体A", "关系X", "新实体B")])
        assert len(added) == 1
        assert len(proc.triples) == before + 1
        assert "新实体A" in proc.entity_relations
        assert "新实体B" in proc.reverse_relations

    def test_add_triples_dedup(self, sample_triples_file):
        proc = KGProcessor(sample_triples_file)
        proc.load_triples()
        before = len(proc.triples)

        # sample_triples_file 里有 "稻瘟病|属于|真菌性病害"
        added = proc.add_triples([("稻瘟病", "属于", "真菌性病害")])
        assert len(added) == 0
        assert len(proc.triples) == before

    def test_add_triples_skips_empty(self, sample_triples_file):
        proc = KGProcessor(sample_triples_file)
        proc.load_triples()
        added = proc.add_triples(
            [("", "关系", "尾"), ("头", "", "尾"), ("头", "关系", "")]
        )
        assert len(added) == 0


# ============================================================
# FAISSIndex 增量能力测试
# ============================================================


class TestFAISSIndexIncremental:
    def test_build_then_add_chunks(self, temp_dir, mock_model):
        """build 后 add_chunks，新 chunk 可检索"""
        idx = FAISSIndex()
        chunks = [
            {"text": "【实体A】关系：值1", "metadata": {"entity": "实体A"}},
            {"text": "【实体B】关系：值2", "metadata": {"entity": "实体B"}},
        ]
        idx.build(chunks, mock_model)
        assert len(idx.texts) == 2

        # 增量加一个新实体
        new_chunks = [{"text": "【实体C】关系：值3", "metadata": {"entity": "实体C"}}]
        added = idx.add_chunks(new_chunks, mock_model)
        assert added == 1
        assert len(idx.texts) == 3

        # embeddings 缓存对齐
        assert idx._embeddings.shape[0] == 3

        # 保存后重载，仍能检索到三个实体
        idx.save(temp_dir)
        reloaded = FAISSIndex.load(temp_dir, mock_model)
        assert len(reloaded.texts) == 3
        assert reloaded._embeddings.shape[0] == 3

    def test_remove_by_entity(self, temp_dir, mock_model):
        """remove_by_entity 删除指定实体的 chunk，其余保留"""
        idx = FAISSIndex()
        chunks = [
            {"text": "【实体A】关系：值1", "metadata": {"entity": "实体A"}},
            {"text": "【实体B】关系：值2", "metadata": {"entity": "实体B"}},
            {"text": "【实体C】关系：值3", "metadata": {"entity": "实体C"}},
        ]
        idx.build(chunks, mock_model)

        removed = idx.remove_by_entity("实体B")
        assert removed == 1
        assert len(idx.texts) == 2
        remaining_entities = [m["entity"] for m in idx.metadatas]
        assert "实体B" not in remaining_entities
        assert "实体A" in remaining_entities
        assert "实体C" in remaining_entities
        # embeddings 缓存同步
        assert idx._embeddings.shape[0] == 2
        # FAISS 索引总数同步
        assert idx.index.ntotal == 2

    def test_remove_nonexistent_entity(self, temp_dir, mock_model):
        idx = FAISSIndex()
        chunks = [{"text": "【实体A】关系：值1", "metadata": {"entity": "实体A"}}]
        idx.build(chunks, mock_model)
        removed = idx.remove_by_entity("不存在的实体")
        assert removed == 0
        assert len(idx.texts) == 1

    def test_remove_then_add_replaces_chunk(self, temp_dir, mock_model):
        """模拟受影响实体重建：先 remove 旧 chunk，再 add 新 chunk"""
        idx = FAISSIndex()
        chunks = [
            {"text": "【实体A】关系：旧值", "metadata": {"entity": "实体A"}},
        ]
        idx.build(chunks, mock_model)

        # remove 旧
        idx.remove_by_entity("实体A")
        # add 新（模拟实体获得新三元组后重建）
        idx.add_chunks(
            [
                {
                    "text": "【实体A】关系：新值\n关系2：值2",
                    "metadata": {"entity": "实体A"},
                }
            ],
            mock_model,
        )
        assert len(idx.texts) == 1
        assert "新值" in idx.texts[0]
        assert "旧值" not in idx.texts[0]

    def test_backward_compat_legacy_index_no_embeddings_npy(self, temp_dir, mock_model):
        """旧索引（无 embeddings.npy）load 时自动 reconstruct 迁移"""
        idx = FAISSIndex()
        chunks = [
            {"text": "【实体A】关系：值1", "metadata": {"entity": "实体A"}},
            {"text": "【实体B】关系：值2", "metadata": {"entity": "实体B"}},
        ]
        idx.build(chunks, mock_model)
        idx.save(temp_dir)

        # 删掉 embeddings.npy 模拟旧索引
        emb_path = os.path.join(temp_dir, "embeddings.npy")
        assert os.path.exists(emb_path)
        os.remove(emb_path)

        # 重载：应自动 reconstruct
        reloaded = FAISSIndex.load(temp_dir, mock_model)
        assert reloaded._embeddings is not None
        assert reloaded._embeddings.shape[0] == 2

        # 迁移后增量操作可用
        reloaded.add_chunks(
            [{"text": "【实体C】关系：值3", "metadata": {"entity": "实体C"}}],
            mock_model,
        )
        assert len(reloaded.texts) == 3
        assert reloaded._embeddings.shape[0] == 3


# ============================================================
# Manifest 测试
# ============================================================


class TestManifest:
    def test_save_and_load_manifest(self, temp_dir):
        m = {"a|b|c", "d|e|f"}
        save_manifest(temp_dir, m)
        loaded = load_manifest(temp_dir)
        assert loaded == m

    def test_load_manifest_missing_file(self, temp_dir):
        loaded = load_manifest(temp_dir)
        assert loaded == set()

    def test_build_manifest_from_data(self, base_triples_file):
        m = build_manifest_from_data(base_triples_file)
        assert len(m) == 6  # base_triples_file 有 6 条
        assert "稻瘟病|属于|真菌性病害" in m

    def test_build_manifest_from_missing_file(self, temp_dir):
        m = build_manifest_from_data(os.path.join(temp_dir, "nope.txt"))
        assert m == set()


# ============================================================
# add_triples_incremental 端到端测试
# ============================================================


class TestAddTriplesIncremental:
    def test_add_new_entity(self, built_index, base_triples_file, mock_model):
        """新增一个全新实体（不与现有重叠）→ 增量 add，旧索引保留"""
        # 基础索引有多少 chunk
        idx_before = FAISSIndex.load(built_index, mock_model)
        chunks_before = len(idx_before.texts)

        new_triples = [
            ("新病害X", "属于", "细菌性病害"),
            ("新病害X", "防治药剂", "新药剂Y"),
            ("新药剂Y", "属于", "新型杀菌剂"),
        ]
        stats = add_triples_incremental(
            new_triples,
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )

        assert stats["new_triples"] == 3
        assert stats["skipped_duplicates"] == 0
        assert (
            stats["new_entities"] >= 2
        )  # 新病害X, 新药剂Y (细菌性病害可能已存在作为尾实体)
        assert stats["affected_entities"] >= 0

        # chunk 数应增加（新实体产生新 chunk）
        idx_after = FAISSIndex.load(built_index, mock_model)
        assert len(idx_after.texts) > chunks_before

        # 旧实体仍在
        before_entities = {m["entity"] for m in idx_before.metadatas}
        after_entities = {m["entity"] for m in idx_after.metadatas}
        assert before_entities.issubset(after_entities)
        assert "新病害X" in after_entities

    def test_add_triple_to_existing_entity_replaces_chunk(
        self, built_index, base_triples_file, mock_model
    ):
        """给已有实体加新三元组 → 该实体 chunk 被替换（旧文本消失，新文本出现）"""
        idx_before = FAISSIndex.load(built_index, mock_model)
        # 找到"稻瘟病"的旧 chunk 文本
        idx_before_texts = {
            m["entity"]: t for t, m in zip(idx_before.texts, idx_before.metadatas)
        }
        old_blast_text = idx_before_texts["稻瘟病"]
        assert "三环唑" in old_blast_text

        # 给稻瘟病加一个新的防治药剂
        new_triples = [("稻瘟病", "防治药剂", "全新特效药Z")]
        stats = add_triples_incremental(
            new_triples,
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )

        assert stats["new_triples"] == 1
        assert stats["affected_entities"] >= 1  # 稻瘟病受影响

        # 重载验证：稻瘟病的 chunk 应包含新药剂，且仍包含旧内容
        idx_after = FAISSIndex.load(built_index, mock_model)
        after_texts = {
            m["entity"]: t for t, m in zip(idx_after.texts, idx_after.metadatas)
        }
        new_blast_text = after_texts["稻瘟病"]
        assert "全新特效药Z" in new_blast_text
        assert "三环唑" in new_blast_text  # 旧的还在（因为重建的是全量三元组）

        # 稻瘟病的 chunk 文本应该变了（更长）
        assert new_blast_text != old_blast_text

        # 稻瘟病仍只有 1 个 chunk（没有重复）
        blast_count = sum(1 for m in idx_after.metadatas if m["entity"] == "稻瘟病")
        assert blast_count == 1

    def test_duplicate_triple_skipped(self, built_index, base_triples_file, mock_model):
        """完全重复的三元组被跳过，索引不变"""
        idx_before = FAISSIndex.load(built_index, mock_model)
        chunks_before = len(idx_before.texts)

        # base 里有 "稻瘟病|属于|真菌性病害"
        dup = [("稻瘟病", "属于", "真菌性病害")]
        stats = add_triples_incremental(
            dup,
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )

        assert stats["new_triples"] == 0
        assert stats["skipped_duplicates"] == 1

        idx_after = FAISSIndex.load(built_index, mock_model)
        assert len(idx_after.texts) == chunks_before  # 索引未变

    def test_new_relation_appended(self, built_index, base_triples_file, mock_model):
        """新关系被追加到关系索引"""
        relation_names_path = os.path.join(built_index, "relation_names.json")
        with open(relation_names_path, encoding="utf-8") as f:
            relations_before = set(json.load(f))

        # 用一个全新的关系
        new_triples = [("新实体M", "完全全新的关系类型", "新实体N")]
        stats = add_triples_incremental(
            new_triples,
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )

        assert stats["new_relations"] >= 1
        with open(relation_names_path, encoding="utf-8") as f:
            relations_after = set(json.load(f))
        assert "完全全新的关系类型" in relations_after
        assert relations_before.issubset(relations_after)

    def test_manifest_written_and_contains_new(
        self, built_index, base_triples_file, mock_model
    ):
        """增量后 manifest 文件存在且包含新三元组"""
        new_triples = [("实体P", "关系Q", "实体R")]
        add_triples_incremental(
            new_triples,
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )

        manifest = load_manifest(built_index)
        assert "实体P|关系Q|实体R" in manifest

    def test_data_path_appended(self, built_index, base_triples_file, mock_model):
        """新三元组被 append 到 data_path"""
        with open(base_triples_file, encoding="utf-8") as f:
            lines_before = f.readlines()

        new_triples = [("实体S", "关系T", "实体U")]
        add_triples_incremental(
            new_triples,
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )

        with open(base_triples_file, encoding="utf-8") as f:
            lines_after = f.readlines()

        assert len(lines_after) == len(lines_before) + 1
        assert "实体S" in lines_after[-1]

    def test_second_incremental_add_uses_existing_manifest(
        self, built_index, base_triples_file, mock_model
    ):
        """第二次增量：manifest 已存在，不再从 data_path 重建"""
        # 第一次加
        add_triples_incremental(
            [("实体V1", "关系", "实体V2")],
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )
        manifest1 = load_manifest(built_index)

        # 第二次加（不同三元组）
        stats = add_triples_incremental(
            [("实体W1", "关系", "实体W2")],
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )
        manifest2 = load_manifest(built_index)

        assert stats["new_triples"] == 1
        assert manifest2 > manifest1  # 集合变大
        assert "实体W1|关系|实体W2" in manifest2

    def test_all_duplicates_returns_zero(
        self, built_index, base_triples_file, mock_model
    ):
        """全部重复时零计算返回"""
        # 取基础里已有的几条
        dups = [
            ("稻瘟病", "属于", "真菌性病害"),
            ("三环唑", "属于", "杀菌剂"),
        ]
        stats = add_triples_incremental(
            dups,
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )
        assert stats["new_triples"] == 0
        assert stats["skipped_duplicates"] == 2
        assert stats["new_entities"] == 0
        assert stats["new_relations"] == 0


# ============================================================
# reset_index 测试
# ============================================================


class TestResetIndex:
    def test_reset_rebuilds_from_data(self, built_index, base_triples_file, mock_model):
        """reset 清空并从 data_path 全量重建"""
        # 先增量加一些（污染索引）
        add_triples_incremental(
            [("临时实体", "临时关系", "临时尾")],
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )
        idx_after_add = FAISSIndex.load(built_index, mock_model)
        assert "临时实体" in {m["entity"] for m in idx_after_add.metadatas}

        # 注意：reset 用 data_path 重建，但 data_path 已被 append 了临时三元组
        # 所以 reset 后临时实体仍在（因为 data_path 里有）。先清掉 data_path 里的临时行
        # 这里测试 reset 本身：清索引文件 + 重建
        stats = reset_index(
            mock_model,
            index_dir=built_index,
            data_path=base_triples_file,
        )
        assert stats["total_triples"] > 0
        assert stats["total_chunks"] > 0

        # manifest 重建
        manifest = load_manifest(built_index)
        assert len(manifest) == stats["total_triples"]

        # 索引可正常加载检索
        idx = FAISSIndex.load(built_index, mock_model)
        assert idx.index.ntotal == stats["total_chunks"]


# ============================================================
# 向后兼容：旧索引（无 manifest）首次增量迁移
# ============================================================


class TestBackwardCompat:
    def test_legacy_index_no_manifest_migrates(
        self, temp_dir, base_triples_file, mock_model
    ):
        """旧索引无 manifest → 首次增量从 data_path 重建 manifest"""
        index_dir = os.path.join(temp_dir, "index")
        with patch(
            "PocketGraphRAG.build_index.SentenceTransformer", return_value=mock_model
        ):
            build_index_with_data(
                base_triples_file, index_dir=index_dir, run_tests=False
            )

        # 删除 manifest 模拟旧索引
        manifest_path = os.path.join(index_dir, MANIFEST_FILENAME)
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        # 也删 embeddings.npy 模拟更旧的版本
        emb_path = os.path.join(index_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            os.remove(emb_path)

        # 增量加：应自动迁移 manifest + embeddings 缓存
        stats = add_triples_incremental(
            [("迁移测试实体", "关系", "尾实体")],
            mock_model,
            index_dir=index_dir,
            data_path=base_triples_file,
        )
        assert stats["new_triples"] == 1
        # manifest 已重建
        assert os.path.exists(manifest_path)
        manifest = load_manifest(index_dir)
        # manifest 应包含 data_path 里所有原有三元组 + 新增的
        assert "迁移测试实体|关系|尾实体" in manifest
        assert "稻瘟病|属于|真菌性病害" in manifest  # 原有的也在
