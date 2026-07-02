"""评测合并 KG (gleaning1 + gleaning2)"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["OLLAMA_MODEL"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
TRIPLES = os.path.join(HERE, "hotpotqa_50_triples_merged.txt")
BENCH = os.path.join(HERE, "hotpotqa_50.json")
INDEX_DIR = os.path.join(HERE, "hotpotqa_index_merged")
os.environ["POCKET_INDEX_DIR"] = INDEX_DIR
os.environ["POCKET_VECTOR_WEIGHT"] = "0.3"
os.environ["POCKET_FUSION_STRATEGY"] = "weighted"
os.environ["POCKET_PAGERANK_WEIGHT"] = "0.3"

from PocketGraphRAG.rag_system import PocketGraphRAG
from PocketGraphRAG.eval_harness import load_benchmark, run_evaluation, print_report
from PocketGraphRAG.build_index import build_index_with_data


def ensure_index():
    faiss_path = os.path.join(INDEX_DIR, "faiss.index")
    if os.path.exists(faiss_path):
        print(f"索引已存在: {INDEX_DIR}")
        return
    print(f"构建索引: {TRIPLES} -> {INDEX_DIR}")
    from PocketGraphRAG import config
    config.INDEX_DIR = INDEX_DIR
    build_index_with_data(TRIPLES, INDEX_DIR, run_tests=False)


def main():
    dataset = load_benchmark(BENCH)
    print(f"benchmark: {dataset['name']} ({len(dataset['questions'])} 题)")
    ensure_index()

    configs = [
        ("mix", 20),    # 最优配置
        ("mix", 25),    # 更大 top_k
    ]
    all_reports = {}
    for mode, top_k in configs:
        print(f"\n{'='*60}")
        print(f"检索模式: {mode}, top_k={top_k}")
        print(f"{'='*60}")
        rag = PocketGraphRAG(data_path=TRIPLES, search_mode=mode, top_k=top_k)
        t0 = time.time()
        report = run_evaluation(rag, dataset, top_k=top_k, run_generation=False)
        elapsed = time.time() - t0
        print(f"\n耗时: {elapsed:.1f}s")
        print_report(report)
        all_reports[f"{mode}_k{top_k}"] = report["summary"]

    print(f"\n{'='*60}")
    print("合并 KG vs gleaning1 KG 对比")
    print(f"{'='*60}")
    print(f"{'配置':<20} {'Hit Rate':>10} {'MRR':>10} {'Entity Cov':>12}")
    print("-" * 55)
    for k, s in all_reports.items():
        print(f"{k:<20} {s['hit_rate']:>10.4f} {s['mrr']:>10.4f} "
              f"{s['entity_coverage']:>12.4f}")
    print(f"\ngleaning1 基线:")
    print(f"  mix k=20 weighted:  Hit=0.8000  MRR=0.5365  EC=0.3200")

    out = os.path.join(HERE, "hotpotqa_merged_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
