"""Shared fixtures for PocketGraphRAG tests."""

import os
import tempfile

import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_triples_file(temp_dir):
    """Create a sample triples file for testing."""
    triples = [
        "稻瘟病|属于|真菌性病害",
        "稻瘟病|防治药剂|三环唑",
        "稻瘟病|防治药剂|稻瘟灵",
        "三环唑|属于|杀菌剂",
        "稻瘟灵|属于|杀菌剂",
        "水稻纹枯病|属于|真菌性病害",
        "水稻纹枯病|防治药剂|井冈霉素",
        "井冈霉素|属于|抗生素类杀菌剂",
        "水稻|病害|稻瘟病",
        "水稻|病害|水稻纹枯病",
        "白叶枯病|属于|细菌性病害",
        "白叶枯病|防治药剂|叶枯唑",
        "水稻|虫害|稻飞虱",
        "稻飞虱|防治药剂|吡虫啉",
        "吡虫啉|属于|烟碱类杀虫剂",
    ]
    filepath = os.path.join(temp_dir, "test_triples.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(triples))
    return filepath


@pytest.fixture
def sample_entity_relations():
    """Sample entity_relations dict for KGDualRetriever tests."""
    return {
        "稻瘟病": [
            ("属于", "真菌性病害"),
            ("防治药剂", "三环唑"),
            ("防治药剂", "稻瘟灵"),
        ],
        "三环唑": [("属于", "杀菌剂")],
        "稻瘟灵": [("属于", "杀菌剂")],
        "水稻纹枯病": [("属于", "真菌性病害"), ("防治药剂", "井冈霉素")],
        "井冈霉素": [("属于", "抗生素类杀菌剂")],
        "水稻": [("病害", "稻瘟病"), ("病害", "水稻纹枯病"), ("虫害", "稻飞虱")],
        "白叶枯病": [("属于", "细菌性病害"), ("防治药剂", "叶枯唑")],
        "稻飞虱": [("防治药剂", "吡虫啉")],
        "吡虫啉": [("属于", "烟碱类杀虫剂")],
    }


@pytest.fixture
def sample_reverse_relations(sample_entity_relations):
    """Build reverse_relations from entity_relations."""
    reverse = {}
    for head, rels in sample_entity_relations.items():
        for rel, tail in rels:
            if tail not in reverse:
                reverse[tail] = []
            reverse[tail].append((head, rel))
    return reverse
