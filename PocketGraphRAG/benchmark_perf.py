"""
PocketGraphRAG 性能基准测试

测试图算法、检索等核心功能的性能。

使用方式：
    python -m PocketGraphRAG.benchmark_perf
"""

import random
import time
from collections import defaultdict


def benchmark_graph_algorithms(num_entities: int = 500, num_edges: int = 2000):
    """图算法性能基准测试"""

    from PocketGraphRAG.kg_reasoning import KGDualRetriever

    print("=" * 60)
    print("  Graph Algorithm Benchmark")
    print(f"  Entities: {num_entities}  Edges: {num_edges}")
    print("=" * 60)

    # 生成随机图
    random.seed(42)
    entity_relations = defaultdict(list)
    all_entities = [f"实体_{i}" for i in range(num_entities)]

    for _ in range(num_edges):
        a = random.choice(all_entities)
        b = random.choice(all_entities)
        if a != b:
            entity_relations[a].append(("关联", b))

    ret = KGDualRetriever.__new__(KGDualRetriever)
    ret.entity_relations = dict(entity_relations)
    ret.reverse_relations = {}
    ret.all_entities = sorted(set(entity_relations.keys()))
    ret.entity_idx = {e: i for i, e in enumerate(ret.all_entities)}
    ret.idx_entity = {i: e for i, e in enumerate(ret.all_entities)}
    ret._adj_cache = None
    ret.threshold = 0.5
    ret.n_hops = 2
    ret.relation_threshold = 0.3

    results = {}

    # 测试 Pagerank
    t0 = time.time()
    pr = ret.compute_pagerank()
    t1 = time.time()
    results["pagerank"] = t1 - t0
    print(f"  Pagerank:           {t1 - t0:.4f}s  ({len(pr)} entities)")

    # 测试个性化 Pagerank
    t0 = time.time()
    ret.personalized_pagerank([all_entities[0], all_entities[1]])
    t1 = time.time()
    results["personalized_pagerank"] = t1 - t0
    print(f"  Personalized Pagerank: {t1 - t0:.4f}s")

    # 测试社区发现
    t0 = time.time()
    communities = ret.detect_communities()
    t1 = time.time()
    results["community_detection"] = t1 - t0
    print(
        f"  Community Detection: {t1 - t0:.4f}s  ({len(set(communities.values()))} communities"
    )

    # 测试度中心性
    t0 = time.time()
    ret.degree_centrality()
    t1 = time.time()
    results["degree_centrality"] = t1 - t0
    print(f"  Degree Centrality:   {t1 - t0:.4f}s")

    # 测试接近中心性
    t0 = time.time()
    ret.closeness_centrality(max_hops=3)
    t1 = time.time()
    results["closeness_centrality"] = t1 - t0
    print(f"  Closeness Centrality: {t1 - t0:.4f}s")

    # 测试介数中心性（近似）
    t0 = time.time()
    ret.betweenness_centrality_approx(k=50)
    t1 = time.time()
    results["betweenness_centrality"] = t1 - t0
    print(f"  Betweenness Centrality (k=50): {t1 - t0:.4f}s")

    # 测试连接组件
    t0 = time.time()
    comps = ret.connected_components()
    t1 = time.time()
    results["connected_components"] = t1 - t0
    print(f"  Connected Components: {t1 - t0:.4f}s  ({len(comps)} components")

    # 测试聚类系数
    t0 = time.time()
    ret.clustering_coefficient()
    t1 = time.time()
    results["clustering_coefficient"] = t1 - t0
    print(f"  Clustering Coeff:    {t1 - t0:.4f}s")

    # 测试最短路径
    t0 = time.time()
    for _ in range(100):
        a = random.choice(all_entities)
        b = random.choice(all_entities)
        ret.shortest_path(a, b, max_hops=5)
    t1 = time.time()
    results["shortest_path_x100"] = t1 - t0
    print(f"  Shortest Path x100: {t1 - t0:.4f}s")

    # 测试 BFS 邻域扩展
    t0 = time.time()
    for _ in range(100):
        seed = random.choice(all_entities)
        # 用 shortest_path 来测试 BFS 性能
        ret.shortest_path(seed, random.choice(all_entities), max_hops=2)
    t1 = time.time()
    results["bfs_x100"] = t1 - t0
    print(f"  BFS Search x100: {t1 - t0:.4f}s")

    print()
    return results


def benchmark_index_building():
    """索引构建性能测试"""
    print("=" * 60)
    print("  Index Building Benchmark")
    print("=" * 60)

    from PocketGraphRAG.config import DATA_PATH
    from PocketGraphRAG.data_processor import DataProcessor

    t0 = time.time()
    processor = DataProcessor(DATA_PATH)
    processor.process()
    t1 = time.time()
    print(f"  Data Processing: {t1 - t0:.4f}s  ({len(processor.chunks)} chunks")
    print()


def main():
    print("\n" + "=" * 60)
    print("  PocketGraphRAG Performance Benchmark")
    print("=" * 60)
    print()

    # 小图测试
    print("\n--- Small Graph (100 entities, 300 edges) ---")
    benchmark_graph_algorithms(num_entities=100, num_edges=300)

    # 中图测试
    print("\n--- Medium Graph (500 entities, 2000 edges) ---")
    benchmark_graph_algorithms(num_entities=500, num_edges=2000)

    # 索引构建测试
    try:
        benchmark_index_building()
    except Exception as e:
        print(f"  [WARN] Index building benchmark skipped: {e}")

    print("=" * 60)
    print("  Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
