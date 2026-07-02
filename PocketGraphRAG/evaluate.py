"""
评测脚本 - PocketGraphRAG 消融实验

主要功能：
1. 定义评测问题集（覆盖单事实、反向链接、对比、多跳、开放等类型）。
2. 对比不同检索模式下的 RAG 回答质量（消融实验）。
3. 输出评测报告。

基线配置：
1. 基础 RAG（纯向量检索）
2. 纯 KG 推理（Local + Global，不走向量检索）
3. KG Local 检索（向量 + Local）
4. KG Mix 检索（向量 + Local + Global）
5. 完整模式（KG Mix + Multi-hop + 对话记忆）

使用方式：
    python -m PocketGraphRAG.evaluate
"""

import argparse
import json
import os
from datetime import datetime

from .llm import has_llm
from .rag_system import PocketGraphRAG

# ========================
# 评测问题集
# ========================
# 从 benchmark/ 目录加载 JSON 评测集，不再硬编码领域问题。
# 默认使用 movie_kg_v1.json（电影知识图谱评测集）。
_DEFAULT_BENCHMARK_FILENAME = "movie_kg_v1.json"


def _get_default_benchmark_path() -> str:
    """返回包内置 benchmark 目录下的默认评测集路径。"""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_dir, "benchmark", _DEFAULT_BENCHMARK_FILENAME)


def load_eval_questions(benchmark_path: str | None = None) -> list[dict]:
    """从 benchmark JSON 文件加载评测问题集。

    Args:
        benchmark_path: 自定义评测集 JSON 路径。None 时用默认 movie_kg_v1.json。

    Returns:
        评测问题列表，每项含 question / expected_entities / expected_relations / type 等字段。

    Raises:
        FileNotFoundError: 评测集文件不存在时提示使用 benchmark/movie_kg_v1.json。
    """
    path = benchmark_path or _get_default_benchmark_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"评测集不存在: {path}\n"
            f"请通过 --benchmark 指定评测集，或使用内置 benchmark/{_DEFAULT_BENCHMARK_FILENAME}"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    print(f"[OK] 加载评测集: {data.get('name', 'benchmark')} ({len(questions)} 个问题)")
    return questions


def run_evaluation(args):
    print("=" * 60)
    print("  PocketGraphRAG 消融实验")
    print("=" * 60)

    if not has_llm():
        print("[错误] 未配置 LLM API Key 或 Ollama，无法运行评测。")
        return

    # 加载评测问题集（从 benchmark JSON 文件）
    try:
        eval_questions = load_eval_questions(getattr(args, "benchmark", None))
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return

    results = []

    # 消融实验配置：5 种基线
    eval_configs = [
        {
            "name": "基础 RAG",
            "search_mode": "vector",
            "use_multihop": False,
            "use_conversation": False,
        },
        {
            "name": "纯 KG 推理",
            "search_mode": "kg_only",
            "use_multihop": False,
            "use_conversation": False,
        },
        {
            "name": "KG Local",
            "search_mode": "local",
            "use_multihop": False,
            "use_conversation": False,
        },
        {
            "name": "KG Mix",
            "search_mode": "mix",
            "use_multihop": False,
            "use_conversation": False,
        },
        {
            "name": "完整模式",
            "search_mode": "mix",
            "use_multihop": True,
            "use_conversation": True,
        },
    ]

    for config in eval_configs:
        print(f"\n{'=' * 60}")
        print(f"  评测配置: {config['name']}")
        print(f"{'=' * 60}")

        rag_system = PocketGraphRAG(
            search_mode=config["search_mode"],
            use_multihop=config["use_multihop"],
            use_conversation=config["use_conversation"],
            top_k=8,  # 使用更大的 top_k 以观察不同检索模式的差异
        )

        config_results = []
        for i, q_data in enumerate(eval_questions):
            question = q_data["question"]
            print(f"\n[{i + 1}/{len(eval_questions)}] Q: {question}")

            full_answer = ""
            sources = []
            pipeline_info = {}
            effective_query = question

            for step in rag_system.answer_stream(question):
                if "chunk" in step:
                    full_answer += step["chunk"]
                if "sources" in step:
                    sources = step["sources"]
                    pipeline_info = step["pipeline_info"]
                    effective_query = step["effective_query"]

            print(f"  A: {full_answer.strip()[:150]}...")
            print(f"  来源数: {len(sources)}", end="")
            extras = []
            search_mode = pipeline_info.get("search_mode", "vector")
            if search_mode != "vector":
                extras.append(f"KG-{search_mode}")
            kg_count = pipeline_info.get("kg_entities_matched", 0)
            if kg_count:
                extras.append(f"匹配{kg_count}实体")
            if pipeline_info.get("query_rewritten"):
                extras.append(f"改写→{effective_query}")
            if pipeline_info.get("multihop_used"):
                extras.append("Multi-hop")
            if extras:
                print(f"  | {' '.join(extras)}", end="")
            print()

            config_results.append(
                {
                    "question": question,
                    "answer": full_answer.strip(),
                    "sources": sources,
                    "pipeline_info": pipeline_info,
                    "effective_query": effective_query,
                    "expected_entities": q_data["expected_entities"],
                    "expected_relations": q_data["expected_relations"],
                    "type": q_data["type"],
                }
            )
        results.append({"config": config, "questions": config_results})

    # 输出报告
    output_filename = (
        f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'=' * 60}")
    print("  消融实验摘要")
    print(f"{'=' * 60}")
    print(f"  {'配置':15s} | {'平均来源数':>8s} | {'KG匹配实体':>10s} | 问题数")
    print(f"  {'-' * 15}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 4}")
    for config_result in results:
        name = config_result["config"]["name"]
        q_count = len(config_result["questions"])
        avg_sources = (
            sum(len(q["sources"]) for q in config_result["questions"]) / q_count
        )
        avg_kg_entities = (
            sum(
                q["pipeline_info"].get("kg_entities_matched", 0)
                for q in config_result["questions"]
            )
            / q_count
        )
        print(
            f"  {name:15s} | {avg_sources:8.1f} | {avg_kg_entities:10.1f} | {q_count}"
        )

    print(f"\n评测报告已保存至: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PocketGraphRAG 消融实验")
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        default=None,
        help="评测集 JSON 文件路径，默认使用内置电影知识图谱评测集 benchmark/movie_kg_v1.json",
    )
    args = parser.parse_args()
    run_evaluation(args)
