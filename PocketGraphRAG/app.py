"""
CLI 命令行交互入口

使用方式：
    python -m PocketGraphRAG                # 交互式问答
    python -m PocketGraphRAG webui          # 启动 Web UI
    python -m PocketGraphRAG ask "问题"     # 单次问答
    python -m PocketGraphRAG eval           # 运行评测
    python -m PocketGraphRAG init           # 初始化索引
    python -m PocketGraphRAG doctor         # 环境诊断
"""

import argparse
import sys


def print_banner():
    print("=" * 60)
    print("  PocketGraphRAG 问答系统 v0.3.0")
    print("  上传资料 -> 本地知识图谱 -> 带证据问答")
    print("=" * 60)


def cmd_webui(args):
    from .webapp import main as webui_main
    webui_main()


def cmd_ask(args):
    rag = _build_rag(args)
    question = " ".join(args.question)
    if not question:
        print("请提供问题: pocketgraphrag ask \"你的问题\"")
        sys.exit(1)

    print(f"问题: {question}\n")
    collected = ""
    sources = []
    for step in rag.answer_stream(question):
        if "chunk" in step:
            chunk = step["chunk"]
            print(chunk, end="", flush=True)
            collected += chunk
        if "sources" in step:
            sources = step.get("sources", [])
    print()
    if sources:
        print(f"\n参考知识来源（共 {len(sources)} 条）:")
        for i, src in enumerate(sources, 1):
            print(f"  [{i}] {src['entity']}（相似度: {src['score']:.4f}）")


def cmd_eval(args):
    from .eval_harness import main as eval_main
    eval_main()


def cmd_init(args):
    from .data_processor import KGProcessor
    from .build_index import build_index
    import os
    from . import config

    processor = KGProcessor()
    triples = processor.load_triples()
    if not triples:
        print(f"[!] 未找到三元组数据: {config.DATA_PATH}")
        print(f"    请先上传文档或放入数据文件到 {config.DATA_PATH}")
        sys.exit(1)
    print(f"[init] 加载 {len(triples)} 条三元组")
    build_index()
    print(f"[init] 索引构建完成: {config.INDEX_DIR}")


def cmd_doctor(args):
    print_banner()
    import os
    from . import config

    checks = []
    try:
        import faiss
        checks.append(("FAISS", True, faiss.__version__))
    except ImportError:
        checks.append(("FAISS", False, "未安装"))

    try:
        import gradio
        checks.append(("Gradio", True, gradio.__version__))
    except ImportError:
        checks.append(("Gradio", False, "未安装"))

    try:
        import torch
        checks.append(("PyTorch", True, torch.__version__))
    except ImportError:
        checks.append(("PyTorch", False, "未安装"))

    try:
        from sentence_transformers import SentenceTransformer
        checks.append(("sentence-transformers", True, "OK"))
    except ImportError:
        checks.append(("sentence-transformers", False, "未安装"))

    triples_exist = os.path.exists(config.DATA_PATH)
    checks.append(("三元组文件", triples_exist, str(config.DATA_PATH)))

    from .llm import get_active_provider
    provider = get_active_provider()
    checks.append(("LLM Provider", True, provider))

    print("\n环境检查:\n")
    all_ok = True
    for name, ok, detail in checks:
        mark = "[OK]" if ok else "[!!]"
        if not ok:
            all_ok = False
        print(f"  {mark} {name:25s} {detail}")

    print()
    if all_ok:
        print("环境就绪，可以启动 WebUI 或开始问答。")
    else:
        print("部分依赖缺失，请检查后重试。")
    print()


def cmd_interactive(args):
    rag = _build_rag(args)
    print_banner()

    provider = _get_provider_name()
    print(f"  LLM: {provider}")
    features = []
    if rag.use_multihop:
        features.append("Multi-hop")
    if rag.search_mode != "mix":
        features.append(f"KG-{rag.search_mode}")
    if rag.use_conversation:
        features.append("对话记忆")
    print(f"  特性: {' + '.join(features) if features else '基础模式'}")
    print()
    print("输入你的问题（输入 q 退出，输入 c 清空对话历史）\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("q", "quit", "exit", "退出"):
            break
        if question.lower() in ("c", "clear", "清空"):
            rag.reset_conversation()
            print("[对话历史已清空]\n")
            continue
        if not question:
            continue

        print("\n[thinking...]\n")
        print("-" * 60)
        final_result = {}
        for step in rag.answer_stream(question):
            if "chunk" in step:
                print(step["chunk"], end="", flush=True)
            if "sources" in step:
                final_result = step
        print("\n" + "-" * 60)

        if "sources" in final_result:
            srcs = final_result["sources"]
            print(f"\n参考知识来源（共 {len(srcs)} 条）:")
            for i, src in enumerate(srcs, 1):
                print(f"  [{i}] {src['entity']}（相似度: {src['score']:.4f}）")
        print()


def _build_rag(args):
    from .rag_system import PocketGraphRAG
    return PocketGraphRAG(
        use_multihop=getattr(args, "multihop", False),
        search_mode=getattr(args, "search_mode", "mix"),
        use_conversation=True,
    )


def _get_provider_name():
    from .llm import get_active_provider
    return get_active_provider()


def main():
    parser = argparse.ArgumentParser(
        prog="pocketgraphrag",
        description="PocketGraphRAG - 轻量级 GraphRAG 框架",
    )
    parser.add_argument(
        "--multihop", action="store_true", help="启用多跳查询分解"
    )
    parser.add_argument(
        "--search-mode",
        default="mix",
        choices=["vector", "local", "global", "mix", "kg_only"],
        help="检索模式 (默认: mix)",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    subparsers.add_parser("webui", help="启动 Web UI")
    ask_p = subparsers.add_parser("ask", help="单次问答")
    ask_p.add_argument("question", nargs="+", help="你的问题")
    subparsers.add_parser("eval", help="运行评测套件")
    subparsers.add_parser("init", help="初始化/重建索引")
    subparsers.add_parser("doctor", help="环境诊断")
    subparsers.add_parser("shell", help="交互式问答 (默认)")

    args = parser.parse_args()

    cmd = args.command or "shell"
    dispatch = {
        "webui": cmd_webui,
        "ask": cmd_ask,
        "eval": cmd_eval,
        "init": cmd_init,
        "doctor": cmd_doctor,
        "shell": cmd_interactive,
    }
    fn = dispatch.get(cmd, cmd_interactive)
    fn(args)


if __name__ == "__main__":
    main()
