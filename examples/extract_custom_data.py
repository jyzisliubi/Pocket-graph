"""从自定义文档抽取三元组并构建知识图谱

这个示例展示如何用 PocketGraphRAG 的 KG 抽取能力处理你自己的文档：
1. 准备一段文本（可以是技术文档、产品手册、研究笔记等）
2. 用 LLM 抽取三元组
3. 保存为标准格式
4. 构建索引

运行前需要配置 LLM（任选其一）：
    # Ollama 本地（推荐，免费）
    set POCKET_OLLAMA_MODEL=qwen2.5:7b
    # 或 SiliconFlow 云端
    set POCKET_SILICONFLOW_API_KEY=sk-xxx

然后运行：
    python examples/extract_custom_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PocketGraphRAG.kg_extractor import extract_knowledge_graph
from PocketGraphRAG.schema import RelationSchema


def main():
    print("=" * 60)
    print("  PocketGraphRAG 自定义文档抽取")
    print("=" * 60)

    # 示例文本（可替换为你自己的文档）
    sample_text = """
    水稻纹枯病是由立枯丝核菌引起的真菌性病害。
    主要危害叶鞘和叶片，严重时也能危害茎秆和穗部。
    病斑初为水渍状，后变为暗绿色至褐色的云纹状病斑。
    高温高湿条件下发病严重，适宜温度为 28-32℃，相对湿度 90% 以上。
    防治方法包括：农业防治（合理密植、浅水勤灌）、
    化学防治（井冈霉素、噻呋酰胺）、生物防治（枯草芽孢杆菌）。
    井冈霉素的推荐用量为每亩 50-100 毫升，兑水 30-50 公斤喷雾。
    """

    print(f"\n[1/4] 准备示例文本（{len(sample_text)} 字符）...")
    print(f"      内容：水稻纹枯病相关资料")

    # 用 schema 约束抽取（关系名归一化）
    print("\n[2/4] 初始化 Schema 约束...")
    schema = RelationSchema()
    print(f"      标准关系白名单：{schema.get_canonical_relations()[:8]}...")

    # 抽取三元组
    print("\n[3/4] 调用 LLM 抽取三元组...")
    result = extract_knowledge_graph(sample_text, schema=schema)

    print(f"\n      抽取到 {len(result.triples)} 条三元组：")
    for i, t in enumerate(result.triples, 1):
        print(f"      [{i}] {t.head} | {t.relation} | {t.tail}")
        print(f"          confidence={t.confidence:.2f} evidence={t.evidence[:40]}...")

    # 保存为标准格式
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "custom_data",
        "extracted_triples.txt",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n[4/4] 保存到 {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for t in result.triples:
            f.write(f"{t.head}|{t.relation}|{t.tail}\n")

    print(f"\n{'=' * 60}")
    print(f"  完成！共抽取 {len(result.triples)} 条三元组")
    print(f"  接下来可以构建索引：")
    print(f"    python -m PocketGraphRAG.cli build --data {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
