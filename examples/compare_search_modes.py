"""对比 5 种检索模式的效果

PocketGraphRAG 支持 5 种检索模式：
- vector: 纯向量相似度
- local: KG 实体邻域
- global: KG 关系嵌入匹配
- mix: 向量 + KG 融合（默认）
- kg_only: 纯 KG（local + global）

这个脚本用同一个问题对比不同模式的检索结果，帮助理解各模式差异。

运行前请先构建索引：
    python -m PocketGraphRAG.build_index

然后运行：
    python examples/compare_search_modes.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG import PocketGraphRAG


def main():
    print("=" * 60)
    print("  PocketGraphRAG 检索模式对比")
    print("=" * 60)

    question = "盗梦空间讲了什么？诺兰还拍过哪些电影？"
    modes = ["vector", "local", "global", "mix", "kg_only"]

    print(f"\n问题：{question}")
    print(f"对比模式：{modes}")

    for mode in modes:
        print(f"\n{'─' * 60}")
        print(f"模式: {mode}")
        print(f"{'─' * 60}")

        try:
            rag = PocketGraphRAG(search_mode=mode)
            results, kg_path = rag.retrieve(question, top_k=5)

            print(f"检索到 {len(results)} 条结果：")
            for i, (text, score, meta) in enumerate(results[:3], 1):
                entity = meta.get("entity", "")
                print(f"  [{i}] score={score:.4f} entity={entity}")
                print(f"      text={text[:70]}...")

            # KG 路径信息
            seed = kg_path.get("seed_entities", [])
            matched = kg_path.get("matched_relations", [])
            if seed:
                print(f"  seed entities: {seed[:3]}")
            if matched:
                print(f"  matched relations: {matched[:3]}")
        except Exception as e:
            print(f"  错误: {e}")

    print(f"\n{'=' * 60}")
    print("  对比完成！kg_only 通常在领域 KG-RAG 场景下效果最佳")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
