"""公开 benchmark 数据集适配器单元测试

用构造的 mock 数据测试 HotpotQA / MuSiQue 适配器，不依赖真实数据集下载。
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG.benchmark_adapters import (
    PublicBenchmark,
    build_corpus_from_benchmark,
    convert_hotpotqa,
    convert_musique,
    _extract_keywords,
    _normalize_entity_name,
)


# ==========================
# 辅助函数
# ==========================


class TestHelpers:
    def test_extract_keywords_basic(self):
        # 中文整体作为一个关键词
        kws = _extract_keywords("稻瘟病是由真菌引起的")
        assert len(kws) == 1
        assert "稻瘟病是由真菌引起的" in kws[0]

    def test_extract_keywords_empty(self):
        assert _extract_keywords("") == []
        assert _extract_keywords(None) == []

    def test_extract_keywords_english(self):
        kws = _extract_keywords("The rice blast disease")
        assert "rice" in kws
        assert "blast" in kws
        assert "disease" in kws
        # 停用词被过滤
        assert "The" not in kws
        assert "is" not in kws

    def test_normalize_entity_name(self):
        assert _normalize_entity_name("Rice_blast") == "Rice blast"
        assert _normalize_entity_name("Entity (disambiguation)") == "Entity"
        assert _normalize_entity_name("") == ""

    def test_normalize_entity_name_multiple_parens(self):
        assert _normalize_entity_name("Foo (bar) (baz)") == "Foo"


# ==========================
# HotpotQA 适配器
# ==========================


class TestConvertHotpotQA:
    def test_basic_conversion(self, tmp_path):
        """基本转换"""
        hotpot_data = [
            {
                "_id": "q1",
                "question": "What disease affects rice?",
                "answer": "rice blast",
                "type": "bridge",
                "supporting_facts": [["Rice_blast", 0], ["Fungus", 1]],
                "context": [
                    ["Rice_blast", ["Rice blast is a disease."]],
                    ["Fungus", ["Fungus causes diseases."]],
                ],
            },
        ]
        path = tmp_path / "hotpot.json"
        path.write_text(json.dumps(hotpot_data), encoding="utf-8")

        bench = convert_hotpotqa(str(path))
        assert len(bench) == 1
        assert bench.version == "hotpotqa-v1.1"
        assert bench.source == "hotpotqa"

        q = bench.questions[0]
        assert q["id"] == "q1"
        assert q["question"] == "What disease affects rice?"
        assert q["type"] == "multi-hop"
        assert "Rice blast" in q["expected_entities"]
        assert "Fungus" in q["expected_entities"]
        assert q["ground_truth"] == "rice blast"
        assert "rice" in q["expected_answer_keywords"]

    def test_skip_comparison_type(self, tmp_path):
        """only_bridge=True 时跳过 comparison 类型"""
        hotpot_data = [
            {
                "_id": "q1",
                "question": "Compare A and B?",
                "answer": "A is better",
                "type": "comparison",
                "supporting_facts": [["A", 0]],
                "context": [["A", ["A desc"]]],
            },
        ]
        path = tmp_path / "hotpot.json"
        path.write_text(json.dumps(hotpot_data), encoding="utf-8")

        bench = convert_hotpotqa(str(path), only_bridge=True)
        assert len(bench) == 0  # comparison 被过滤

        # only_bridge=False 时保留
        bench2 = convert_hotpotqa(str(path), only_bridge=False)
        assert len(bench2) == 1

    def test_skip_yes_no_answer(self, tmp_path):
        """yes/no 答案被跳过"""
        hotpot_data = [
            {
                "_id": "q1",
                "question": "Is rice blast a disease?",
                "answer": "yes",
                "type": "bridge",
                "supporting_facts": [["Rice_blast", 0]],
                "context": [["Rice_blast", ["desc"]]],
            },
        ]
        path = tmp_path / "hotpot.json"
        path.write_text(json.dumps(hotpot_data), encoding="utf-8")

        bench = convert_hotpotqa(str(path))
        assert len(bench) == 0  # yes 答案被跳过

    def test_skip_no_supporting_facts(self, tmp_path):
        """没有 supporting_facts 的题目被跳过"""
        hotpot_data = [
            {
                "_id": "q1",
                "question": "Q?",
                "answer": "answer",
                "type": "bridge",
                "supporting_facts": [],
                "context": [["C", ["desc"]]],
            },
        ]
        path = tmp_path / "hotpot.json"
        path.write_text(json.dumps(hotpot_data), encoding="utf-8")

        bench = convert_hotpotqa(str(path))
        assert len(bench) == 0

    def test_max_questions_limit(self, tmp_path):
        """max_questions 限制转换数量"""
        hotpot_data = [
            {
                "_id": f"q{i}",
                "question": f"Q{i}?",
                "answer": f"answer{i}",
                "type": "bridge",
                "supporting_facts": [[f"Entity{i}", 0]],
                "context": [[f"Entity{i}", ["desc"]]],
            }
            for i in range(10)
        ]
        path = tmp_path / "hotpot.json"
        path.write_text(json.dumps(hotpot_data), encoding="utf-8")

        bench = convert_hotpotqa(str(path), max_questions=3)
        assert len(bench) == 3

    def test_to_json_roundtrip(self, tmp_path):
        """to_json 保存的文件可被 load_benchmark 读取"""
        hotpot_data = [
            {
                "_id": "q1",
                "question": "Q?",
                "answer": "answer",
                "type": "bridge",
                "supporting_facts": [["Entity1", 0]],
                "context": [["Entity1", ["desc"]]],
            },
        ]
        in_path = tmp_path / "hotpot.json"
        in_path.write_text(json.dumps(hotpot_data), encoding="utf-8")

        bench = convert_hotpotqa(str(in_path))
        out_path = str(tmp_path / "output.json")
        bench.to_json(out_path)

        # 验证文件格式
        with open(out_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["version"] == "hotpotqa-v1.1"
        assert len(loaded["questions"]) == 1
        assert "description" in loaded


# ==========================
# MuSiQue 适配器
# ==========================


class TestConvertMuSiQue:
    def test_basic_conversion(self, tmp_path):
        musique_data = [
            {
                "id": "m1",
                "question": "What is X?",
                "answer": "Y",
                "supporting_items": [{"title": "X"}],
                "paragraphs": [
                    {"title": "X", "paragraph_text": "X is something."},
                    {"title": "Z", "paragraph_text": "Z is other."},
                ],
            },
        ]
        path = tmp_path / "musique.json"
        path.write_text(json.dumps(musique_data), encoding="utf-8")

        bench = convert_musique(str(path))
        assert len(bench) == 1
        assert bench.version == "musique-v1"

        q = bench.questions[0]
        assert q["id"] == "m1"
        assert q["difficulty"] == "hard"
        assert "X" in q["expected_entities"]
        assert q["ground_truth"] == "Y"

    def test_skip_empty_answer(self, tmp_path):
        musique_data = [
            {
                "id": "m1",
                "question": "Q?",
                "answer": "",
                "supporting_items": [],
                "paragraphs": [],
            },
        ]
        path = tmp_path / "musique.json"
        path.write_text(json.dumps(musique_data), encoding="utf-8")

        bench = convert_musique(str(path))
        assert len(bench) == 0


# ==========================
# build_corpus_from_benchmark
# ==========================


class TestBuildCorpus:
    def test_build_corpus(self, tmp_path):
        hotpot_data = [
            {
                "_id": "q1",
                "question": "Q?",
                "answer": "answer",
                "type": "bridge",
                "supporting_facts": [["E1", 0], ["E2", 0]],
                "context": [
                    ["E1", ["E1 desc."]],
                    ["E2", ["E2 desc."]],
                ],
            },
        ]
        path = tmp_path / "hotpot.json"
        path.write_text(json.dumps(hotpot_data), encoding="utf-8")

        bench = convert_hotpotqa(str(path))
        corpus = build_corpus_from_benchmark(bench)
        assert len(corpus) == 2  # 2 个段落
        assert "E1 desc." in corpus
        assert "E2 desc." in corpus

    def test_empty_benchmark(self):
        bench = PublicBenchmark(version="test", source="test")
        assert build_corpus_from_benchmark(bench) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
