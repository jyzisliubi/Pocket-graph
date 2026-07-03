"""
PocketGraphRAG Benchmark 一键评测脚本

对比不同检索配置的效果，生成评测报告和对比表格。

使用方式：
    # 使用内置电影知识图谱评测集
    python -m PocketGraphRAG.run_benchmark

    # 使用自定义评测集
    python -m PocketGraphRAG.run_benchmark --benchmark path/to/your_benchmark.json

    # 指定 top_k
    python -m PocketGraphRAG.run_benchmark --top-k 10

    # 只评测部分配置
    python -m PocketGraphRAG.run_benchmark --configs vector,mix
"""

import argparse
import os
from datetime import datetime

from .benchmark import BenchmarkEvaluator, compare_reports
from .config import DATA_PATH
from .llm import has_llm
from .rag_system import PocketGraphRAG

# 默认评测配置组合
DEFAULT_CONFIGS = [
    {
        "name": "Vector Only",
        "search_mode": "vector",
        "use_multihop": False,
        "use_pagerank": False,
    },
    {
        "name": "KG Only",
        "search_mode": "kg_only",
        "use_multihop": False,
        "use_pagerank": False,
    },
    {
        "name": "KG Local",
        "search_mode": "local",
        "use_multihop": False,
        "use_pagerank": False,
    },
    {
        "name": "KG Global",
        "search_mode": "global",
        "use_multihop": False,
        "use_pagerank": False,
    },
    {
        "name": "KG Mix",
        "search_mode": "mix",
        "use_multihop": False,
        "use_pagerank": False,
    },
    {
        "name": "KG Mix + Pagerank",
        "search_mode": "mix",
        "use_multihop": False,
        "use_pagerank": True,
    },
    {
        "name": "Full (Mix+Multihop+Pagerank)",
        "search_mode": "mix",
        "use_multihop": True,
        "use_pagerank": True,
    },
]


def run_benchmark(args):
    print("=" * 70)
    print("  PocketGraphRAG Benchmark Runner")
    print("=" * 70)

    if not has_llm():
        print("[错误] 未配置 LLM API Key 或 Ollama，无法运行评测。")
        return

    # 确定评测集路径
    if args.benchmark:
        benchmark_path = args.benchmark
    else:
        # 使用内置电影知识图谱评测集
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_path = os.path.join(pkg_dir, "benchmark", "movie_kg_v1.json")

    if not os.path.exists(benchmark_path):
        print(f"[错误] 评测集不存在: {benchmark_path}")
        return

    # 加载评测集
    evaluator = BenchmarkEvaluator(benchmark_path)

    # 选择要评测的配置
    configs = DEFAULT_CONFIGS
    if args.configs:
        selected = [c.strip() for c in args.configs.split(",")]
        configs = [
            c
            for c in DEFAULT_CONFIGS
            if c["name"] in selected or c["search_mode"] in selected
        ]
        if not configs:
            print(f"[错误] 未找到匹配的配置: {args.configs}")
            return

    print(f"\n评测配置数: {len(configs)}")
    print(f"Top-K: {args.top_k}")
    print()

    reports = []
    for config in configs:
        print("-" * 70)
        print(f"  Running: {config['name']}")
        print("-" * 70)

        try:
            rag = PocketGraphRAG(
                search_mode=config["search_mode"],
                use_multihop=config["use_multihop"],
                use_conversation=False,
                use_pagerank=config["use_pagerank"],
                top_k=args.top_k,
                data_path=args.data_path or DATA_PATH,
            )

            report = evaluator.evaluate(
                rag,
                config_name=config["name"],
                top_k=args.top_k,
            )
            report.print_summary(detailed=args.detailed)
            reports.append(report)
        except Exception as e:
            print(f"  [错误] 配置 '{config['name']}' 运行失败: {e}")
            import traceback

            traceback.print_exc()

    # 输出对比报告
    if len(reports) > 1:
        print(compare_reports(reports))

    # 保存所有报告
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(output_dir, exist_ok=True)

    for report in reports:
        safe_name = (
            report.config_name.replace(" ", "_").replace("(", "").replace(")", "")
        )
        filepath = os.path.join(output_dir, f"{safe_name}.json")
        report.save(filepath)

    # 保存对比摘要
    if len(reports) > 1:
        summary = {
            "benchmark": reports[0].benchmark_name,
            "top_k": args.top_k,
            "configs": [],
        }
        for report in reports:
            summary["configs"].append(
                {
                    "name": report.config_name,
                    "search_mode": report.search_mode,
                    "use_multihop": report.use_multihop,
                    "use_pagerank": report.use_pagerank,
                    "overall": report.compute_overall(),
                }
            )

        summary_path = os.path.join(output_dir, "comparison_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            import json

            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] 对比摘要已保存: {summary_path}")

    print(f"\n[完成] 所有评测结果已保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="PocketGraphRAG Benchmark Runner - 对比不同检索配置的效果"
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        default=None,
        help="评测集 JSON 文件路径，默认使用内置电影知识图谱评测集 benchmark/movie_kg_v1.json",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=8,
        help="检索 top_k 值，默认 8",
    )
    parser.add_argument(
        "--configs",
        "-c",
        type=str,
        default=None,
        help="要评测的配置名称，用逗号分隔，如 vector,mix,full",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出目录，默认自动生成带时间戳的目录",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="三元组数据路径，默认使用配置中的 DATA_PATH",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="显示每个问题的详细结果",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
