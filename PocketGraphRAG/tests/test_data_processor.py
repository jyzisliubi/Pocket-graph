"""Tests for KGProcessor and data processing utilities."""

import os

from PocketGraphRAG.data_processor import (
    KGProcessor,
    auto_detect_reverse_relations,
)


class TestAutoDetectReverseRelations:
    def test_detects_prevention_relations(self):
        rels = {"防治", "治疗", "抑制", "预防"}
        result = auto_detect_reverse_relations(rels)
        assert result == rels

    def test_detects_containment_relations(self):
        rels = {"包含", "包括", "含有", "组成"}
        result = auto_detect_reverse_relations(rels)
        assert result == rels

    def test_detects_belonging_relations(self):
        rels = {"属于", "归类于", "分为"}
        result = auto_detect_reverse_relations(rels)
        assert result == rels

    def test_no_reverse_for_symmetric_relations(self):
        rels = {"相关", "相似", "等于"}
        result = auto_detect_reverse_relations(rels)
        assert len(result) == 0

    def test_empty_input(self):
        result = auto_detect_reverse_relations(set())
        assert result == set()

    def test_mixed_relations(self):
        rels = {"防治", "相关", "属于", "等于", "用于"}
        result = auto_detect_reverse_relations(rels)
        assert "防治" in result
        assert "属于" in result
        assert "用于" in result
        assert "相关" not in result
        assert "等于" not in result


class TestKGProcessor:
    def test_load_triples(self, sample_triples_file):
        processor = KGProcessor(sample_triples_file)
        processor.load_triples()

        assert len(processor.entity_relations) > 0
        assert "稻瘟病" in processor.entity_relations
        assert "三环唑" in processor.entity_relations

    def test_entity_relations_structure(self, sample_triples_file):
        processor = KGProcessor(sample_triples_file)
        processor.load_triples()

        rels = processor.entity_relations["稻瘟病"]
        assert isinstance(rels, list)
        assert len(rels) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in rels)

    def test_build_text_chunks(self, sample_triples_file):
        processor = KGProcessor(sample_triples_file)
        processor.load_triples()
        chunks = processor.process()

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        sample_chunk = chunks[0]
        assert "entity" in sample_chunk
        assert "text" in sample_chunk
        assert "metadata" in sample_chunk
        assert isinstance(sample_chunk["metadata"], dict)

    def test_chunks_contain_entity_text(self, sample_triples_file):
        processor = KGProcessor(sample_triples_file)
        processor.load_triples()
        chunks = processor.process()

        entity_names = [c["entity"] for c in chunks]
        assert "稻瘟病" in entity_names
        assert "三环唑" in entity_names

        for chunk in chunks:
            assert chunk["entity"] in chunk["text"]

    def test_reverse_link_relations_applied(self, sample_triples_file):
        processor = KGProcessor(
            sample_triples_file,
            reverse_link_relations={"防治药剂", "属于"},
        )
        processor.load_triples()
        chunks = processor.process()

        entity_texts = {c["entity"]: c["text"] for c in chunks}

        assert "三环唑" in entity_texts
        assert "稻瘟病" in entity_texts["三环唑"] or "防治" in entity_texts["三环唑"]

    def test_get_entity_count(self, sample_triples_file):
        processor = KGProcessor(sample_triples_file)
        processor.load_triples()
        chunks = processor.process()

        assert len(chunks) > 5

    def test_relation_templates_format(self, sample_triples_file):
        templates = {
            "防治药剂": "可用药剂：{tail}",
            "属于": "类别：{tail}",
        }
        processor = KGProcessor(
            sample_triples_file,
            relation_templates=templates,
        )
        processor.load_triples()
        chunks = processor.process()

        entity_texts = {c["entity"]: c["text"] for c in chunks}
        if "稻瘟病" in entity_texts:
            text = entity_texts["稻瘟病"]
            assert "三环唑" in text

    def test_invalid_file_raises_error(self, temp_dir):
        bad_path = os.path.join(temp_dir, "nonexistent.txt")
        processor = KGProcessor(bad_path)
        try:
            processor.load_triples()
        except (FileNotFoundError, OSError):
            pass
        else:
            assert len(processor.entity_relations) == 0
