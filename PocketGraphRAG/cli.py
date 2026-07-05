"""
PocketGraphRAG CLI — 统一命令行入口

现代化命令行工具，对标 microsoft/graphrag 和 LightRAG 的 CLI 体验。

安装::

    pip install pocketgraphrag          # 核心
    pip install "pocketgraphrag[cli]"   # 含Typer CLI

使用::

    pocketgraphrag                      # 交互式问答 (shell)
    pocketgraphrag --help               # 查看所有命令
    pocketgraphrag doctor               # 环境诊断
    pocketgraphrag init                 # 环境检查 & 初始化向导
    pocketgraphrag ask "这部电影讲了什么？"  # 单次问答
    pocketgraphrag qa "问题" --stream   # 流式问答（别名）
    pocketgraphrag extract -i doc.txt   # 从文档抽取三元组
    pocketgraphrag build                # 构建索引
    pocketgraphrag serve web            # 启动 Gradio Web UI
    pocketgraphrag serve api            # 启动 FastAPI REST API
    pocketgraphrag eval                 # 运行评测套件
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError as exc:
    raise SystemExit(
        "Typer 未安装。请运行: pip install 'pocketgraphrag[cli]'\n"
        "或使用轻量入口: python -m PocketGraphRAG.app"
    ) from exc

app = typer.Typer(
    name="pocketgraphrag",
    help="PocketGraphRAG — 轻量级本地优先 GraphRAG 框架",
    no_args_is_help=False,
    add_completion=False,
)

serve_app = typer.Typer(help="启动 Web UI 或 REST API 服务", no_args_is_help=True)
app.add_typer(serve_app, name="serve")


# ========================
# 输出格式化工具
# ========================


def _print_banner():
    typer.echo("=" * 60)
    typer.echo("  PocketGraphRAG 问答系统 v0.3.0")
    typer.echo("  上传资料 -> 本地知识图谱 -> 带证据问答")
    typer.echo("=" * 60)


def _source_type_label(source_type: str) -> str:
    labels = {"kg": "KG", "vector": "向量", "community_summary": "社区摘要"}
    return labels.get(source_type or "", "")


def _format_source_line(idx: int, s: dict, max_score: float) -> str:
    """格式化单条来源，输出形如: [1] 实体A（相关度: 100%） [KG]"""
    raw = float(s.get("score", 0.0) or 0.0)
    if max_score > 0:
        pct = int(round(raw / max_score * 100))
    else:
        pct = 0
    pct = max(0, min(100, pct))
    entity = s.get("entity", "")
    type_label = _source_type_label(s.get("source_type", ""))
    type_suffix = f" [{type_label}]" if type_label else ""
    return f"  [{idx}] {entity}（相关度: {pct}%）{type_suffix}"


def _format_sources_block(sources: list[dict]) -> str:
    """把 sources 列表渲染成 📚 来源 区块。"""
    if not sources:
        return ""
    max_score = max((float(s.get("score", 0.0) or 0.0) for s in sources), default=0.0)
    lines = ["📚 来源（相关度为相对最高分比例）:"]
    for i, s in enumerate(sources, 1):
        lines.append(_format_source_line(i, s, max_score))
    return "\n".join(lines)


def _build_rag(search_mode: str = "mix", multihop: bool = False, top_k: int = 5):
    from .rag_system import PocketGraphRAG
    return PocketGraphRAG(search_mode=search_mode, use_multihop=multihop, top_k=top_k)


# ========================
# doctor — 环境诊断
# ========================


@app.command("doctor")
def doctor_cmd():
    """环境诊断：检查依赖、配置、索引、LLM 可用性。"""
    _print_banner()
    _run_environment_check(verbose=True)


def _run_environment_check(verbose: bool = True) -> dict:
    """运行环境检查，返回结果字典；verbose=True 时打印诊断信息。"""
    from . import config
    from .llm import get_active_provider, has_llm

    checks: list[tuple[str, bool, str]] = []

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
        import sentence_transformers
        checks.append(("sentence-transformers", True, "OK"))
    except ImportError:
        checks.append(("sentence-transformers", False, "未安装"))

    triples_exist = os.path.exists(config.DATA_PATH)
    checks.append(("数据文件", triples_exist, str(config.DATA_PATH)))

    index_exist = os.path.exists(os.path.join(config.INDEX_DIR, "faiss.index"))
    checks.append(("FAISS 索引", index_exist, str(config.INDEX_DIR)))

    llm_configured = has_llm()
    provider = get_active_provider()
    checks.append(("LLM", llm_configured, provider))

    if verbose:
        typer.echo("\n环境检查:\n")
        all_ok = True
        for name, ok, detail in checks:
            if ok:
                mark = typer.style("[OK]", fg=typer.colors.GREEN)
            else:
                mark = typer.style("[!!]", fg=typer.colors.RED)
                all_ok = False
            typer.echo(f"  {mark} {name:25s} {detail}")

        typer.echo(f"\n  数据路径: {config.DATA_PATH}")
        typer.echo(f"  索引目录: {config.INDEX_DIR}")
        typer.echo(f"  默认检索模式: {config.SEARCH_MODE}")
        typer.echo(f"  Embedding 模型: {config.EMBEDDING_MODEL}")
        typer.echo("")
        if all_ok:
            typer.secho("环境就绪，可以启动 WebUI 或开始问答。", fg=typer.colors.GREEN)
        else:
            typer.secho("部分依赖缺失或配置不完整，请检查后重试。", fg=typer.colors.YELLOW)
        typer.echo("")

    return {
        "checks": checks,
        "data_path": config.DATA_PATH,
        "index_dir": config.INDEX_DIR,
        "search_mode": config.SEARCH_MODE,
        "embedding_model": config.EMBEDDING_MODEL,
    }


# ========================
# init — 环境检查 & 配置向导
# ========================


@app.command("init")
def init_cmd(
    setup: bool = typer.Option(False, "--setup", help="交互式生成 .env 配置文件"),
):
    """环境检查 & 初始化向导：检查依赖/配置/索引状态，可选生成 .env。"""
    _print_banner()
    typer.echo("PocketGraphRAG 环境检查\n")
    env = _run_environment_check(verbose=True)

    if setup:
        _run_setup_wizard(env)


def _run_setup_wizard(env: dict):
    """交互式 .env 生成向导。"""
    cwd = Path.cwd()
    pkg_root = Path(__file__).parent.parent
    example_env = pkg_root / ".env.example"
    target_env = cwd / ".env"

    if target_env.exists():
        typer.secho(f"已存在 .env: {target_env}", fg=typer.colors.YELLOW)
        return

    if example_env.exists():
        shutil.copy(example_env, target_env)
        typer.secho(f"已从 .env.example 创建: {target_env}", fg=typer.colors.GREEN)
        typer.echo("请编辑该文件填入 API Key 后重新运行。")
    else:
        typer.secho("未找到 .env.example 模板，请手动创建 .env 文件。", fg=typer.colors.YELLOW)


# ========================
# build — 构建索引
# ========================


@app.command("build")
def build_cmd(
    data_path: Optional[str] = typer.Option(None, "--data", "-d", help="三元组数据文件路径"),
):
    """构建 FAISS 向量索引 + 实体/关系嵌入索引。"""
    from .build_index import build_index_with_data, build_index
    from .config import INDEX_DIR
    typer.secho("正在构建索引...", fg=typer.colors.YELLOW)
    if data_path:
        build_index_with_data(data_path, INDEX_DIR, run_tests=True)
    else:
        build_index()
    typer.secho("索引构建完成", fg=typer.colors.GREEN)


# ========================
# extract — 文档抽取三元组
# ========================


@app.command("extract")
def extract_cmd(
    input_path: str = typer.Option(..., "--input", "-i", help="输入文件路径 (txt/md/pdf/docx)"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="输出三元组文件路径"),
    min_confidence: float = typer.Option(0.6, "--min-confidence", help="最低置信度阈值"),
):
    """从文档自动抽取知识图谱三元组。"""
    from .kg_extractor import extract_knowledge_graph

    if not os.path.exists(input_path):
        typer.secho(f"文件不存在: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    output_path = output_path or input_path + ".triples.txt"
    typer.secho(f"正在从 {input_path} 抽取三元组...", fg=typer.colors.YELLOW)
    result = extract_knowledge_graph(text, min_confidence=min_confidence)

    with open(output_path, "w", encoding="utf-8") as f:
        for t in result.triples:
            f.write(f"{t.head}|{t.relation}|{t.tail}\n")

    typer.secho(
        f"抽取 {len(result.triples)} 条三元组 (高质量 {result.high_quality_count}, "
        f"平均置信度 {result.avg_confidence:.3f}) -> {output_path}",
        fg=typer.colors.GREEN,
    )


# ========================
# multi-extract — 多模型 KG 融合抽取（PocketGraphRAG 独有）
# ========================


@app.command("multi-extract")
def multi_extract_cmd(
    input_path: str = typer.Option(..., "--input", "-i", help="输入文件路径 (txt/md/pdf/docx)"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="输出三元组文件路径"),
    models: str = typer.Option(
        ...,
        "--models",
        "-m",
        help="逗号分隔的 LLM 模型名列表，如 qwen-flash,qwen-max",
    ),
    strategy: str = typer.Option(
        "union", "--strategy", "-s", help="融合策略: union (并集去重) / intersect (交集)"
    ),
    min_confidence: float = typer.Option(0.6, "--min-confidence", help="最低置信度阈值"),
):
    """多模型 KG 融合抽取：用多个 LLM 抽取同一份文档并融合，覆盖每个模型的盲点。

    \b
    PocketGraphRAG 独有技术，实测在 HotpotQA 上 Hit Rate 0.80 → 0.86（+6%）。
    相当于集成学习，成本低收益高。

    \b
    示例:
        pocketgraphrag multi-extract -i doc.txt -m qwen-flash,qwen-max
        pocketgraphrag multi-extract -i doc.txt -m qwen-flash,qwen-max --strategy intersect
    """
    from .kg_extractor import extract_triples_multi_model

    if not os.path.exists(input_path):
        typer.secho(f"文件不存在: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_list) < 2:
        typer.secho(
            "多模型融合至少需要 2 个模型，单个模型请使用 extract 命令",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    output_path = output_path or input_path + ".fusion.triples.txt"
    typer.secho(
        f"正在用 {len(model_list)} 个模型融合抽取: {model_list}",
        fg=typer.colors.YELLOW,
    )

    triples, stats = extract_triples_multi_model(
        text=text,
        models=model_list,
        strategy=strategy,
    )

    # 过滤低置信度
    triples = [t for t in triples if t.confidence >= min_confidence]

    with open(output_path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(f"{t.head}|{t.relation}|{t.tail}\n")

    typer.secho(
        f"\n融合完成: {len(triples)} 条三元组 (策略={strategy})",
        fg=typer.colors.GREEN,
    )
    typer.secho("各模型抽取数量:", fg=typer.colors.CYAN)
    for m, n in stats.items():
        typer.echo(f"  {m}: {n}")
    typer.secho(f"输出: {output_path}", fg=typer.colors.GREEN)


# ========================
# ask / qa — 单次问答
# ========================


def _do_ask(question: str, search_mode: str, multihop: bool, top_k: int, stream: bool):
    from . import config
    index_ready = os.path.exists(os.path.join(config.INDEX_DIR, "faiss.index"))
    typer.echo(f"当前数据路径: {config.DATA_PATH}")
    typer.echo(f"索引状态: {'已构建' if index_ready else '未构建'}")
    typer.echo(f"检索模式: {search_mode}")
    typer.echo("")
    typer.echo(f"问题: {question}\n")
    typer.secho("[thinking...]", fg=typer.colors.YELLOW)

    rag = _build_rag(search_mode=search_mode, multihop=multihop, top_k=top_k)

    if stream:
        sources = []
        pipeline_info: dict = {}
        for step in rag.answer_stream(question):
            if "chunk" in step:
                typer.echo(step["chunk"], nl=False)
            if "sources" in step:
                sources = step.get("sources", []) or []
            if "pipeline_info" in step:
                pipeline_info = step.get("pipeline_info", {}) or {}
        typer.echo("")
        info = pipeline_info
        if info.get("response_mode") == "retrieval_only":
            typer.secho("\n[注意] LLM 未连接，返回的是检索到的原始片段", fg=typer.colors.YELLOW)
        if sources:
            typer.echo(_format_sources_block(sources))
        kg_count = info.get("kg_entities_matched", 0)
        typer.secho(f"\n[KG 匹配实体数: {kg_count}]", fg=typer.colors.BLUE)
    else:
        result = rag.answer(question)
        typer.echo(result["answer"])
        info = result.get("pipeline_info", {})
        resp_mode = info.get("response_mode", "")
        if resp_mode in ("retrieval_fallback", "retrieval_only"):
            reason = info.get("fallback_reason", "")
            typer.secho("\n回答模式: 检索兜底" + (f"（{reason}）" if reason else ""), fg=typer.colors.YELLOW)
        sources = result.get("sources") or []
        if sources:
            typer.echo("")
            typer.echo(_format_sources_block(sources))
        kg_count = info.get("kg_entities_matched", 0)
        typer.secho(f"\n[KG 匹配实体数: {kg_count}]", fg=typer.colors.BLUE)


@app.command("ask")
def ask_cmd(
    question: list[str] = typer.Argument(..., help="要提问的问题（支持多词）"),
    search_mode: str = typer.Option("mix", "--search-mode", "-m", help="vector/local/global/mix/kg_only"),
    multihop: bool = typer.Option(False, "--multihop", help="启用多跳查询分解"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="检索返回条数"),
    stream: bool = typer.Option(False, "--stream/--no-stream", help="流式输出（默认关闭，与qa一致）"),
):
    """单次问答：提问后输出回答和来源即退出。"""
    q = " ".join(question).strip()
    if not q:
        typer.secho("请提供问题: pocketgraphrag ask \"你的问题\"", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    _do_ask(q, search_mode, multihop, top_k, stream)


@app.command("qa")
def qa_cmd(
    question: str = typer.Argument(..., help="要提问的问题"),
    search_mode: str = typer.Option("mix", "--search-mode", "-m"),
    multihop: bool = typer.Option(False, "--multihop"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    stream: bool = typer.Option(False, "--stream/--no-stream", help="流式输出（默认关闭）"),
):
    """单次问答（ask 的别名，保持与旧 CLI 兼容）。"""
    _do_ask(question, search_mode, multihop, top_k, stream)


# ========================
# shell (默认命令)
# ========================


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    multihop: bool = typer.Option(False, "--multihop", help="启用多跳查询分解"),
    search_mode: str = typer.Option("mix", "--search-mode", "-m", help="检索模式"),
):
    if ctx.invoked_subcommand is not None:
        return
    shell_cmd(multihop=multihop, search_mode=search_mode)


@app.command("shell")
def shell_cmd(
    multihop: bool = typer.Option(False, "--multihop", help="启用多跳查询分解"),
    search_mode: str = typer.Option("mix", "--search-mode", "-m", help="检索模式"),
):
    """交互式问答 REPL（默认命令，直接运行 pocketgraphrag 进入）。"""
    _print_banner()
    from .llm import get_active_provider
    rag = _build_rag(search_mode=search_mode, multihop=multihop, top_k=5)

    provider = get_active_provider()
    typer.echo(f"  LLM: {provider}")
    feats = []
    if rag.use_multihop:
        feats.append("Multi-hop")
    if rag.search_mode != "mix":
        feats.append(f"KG-{rag.search_mode}")
    if rag.use_conversation:
        feats.append("对话记忆")
    typer.echo(f"  特性: {' + '.join(feats) if feats else '基础模式'}")

    from . import config
    idx_ok = os.path.exists(os.path.join(config.INDEX_DIR, "faiss.index"))
    typer.echo(f"  索引: {'已就绪' if idx_ok else '未构建'}")
    typer.echo("")
    typer.echo("输入你的问题（输入 q 退出，输入 c 清空对话历史）")
    typer.echo("")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("")
            break
        if question.lower() in ("q", "quit", "exit", "退出"):
            break
        if question.lower() in ("c", "clear", "清空"):
            rag.reset_conversation()
            typer.secho("[对话历史已清空]", fg=typer.colors.YELLOW)
            typer.echo("")
            continue
        if not question:
            continue

        typer.echo("")
        typer.secho("[thinking...]", fg=typer.colors.YELLOW)
        typer.echo("-" * 60)
        final_result: dict = {}
        sources = []
        for step in rag.answer_stream(question):
            if "chunk" in step:
                typer.echo(step["chunk"], nl=False)
            if "sources" in step:
                final_result = step
                sources = step.get("sources", []) or []
        typer.echo("")
        typer.echo("-" * 60)

        if sources:
            typer.secho(f"\n参考知识来源（共 {len(sources)} 条）:", fg=typer.colors.CYAN)
            typer.echo(_format_sources_block(sources))
        info = final_result.get("pipeline_info", {})
        if info.get("response_mode") == "retrieval_only":
            typer.secho("\n[注意] LLM 未连接，返回的是检索到的原始片段", fg=typer.colors.YELLOW)
        typer.echo("")


# ========================
# serve — 启动服务
# ========================


def _launch_webapp(host: str, port: int, share: bool, listen: bool):
    """启动 Gradio Web UI（通过 webapp.main）。"""
    # 预检 gradio 依赖，避免直接抛 ModuleNotFoundError（BUG #8）
    try:
        import gradio  # noqa: F401
    except ImportError:
        typer.secho(
            "未安装 gradio，无法启动 Web UI。\n"
            "请运行: pip install 'pocketgraphrag[web]'\n"
            "或: pip install gradio",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if listen:
        host = "0.0.0.0"
    typer.secho(f"启动 Gradio Web UI @ http://{host}:{port}", fg=typer.colors.CYAN, bold=True)
    os.environ["POCKET_WEBUI_PORT"] = str(port)
    sys.argv = ["webapp", "--host", host, "--port", str(port)]
    if share:
        sys.argv.append("--share")
    if listen:
        sys.argv.append("--listen")
    from . import webapp
    webapp.main()


@serve_app.command("web")
def serve_web(
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址"),
    port: int = typer.Option(7860, "--port", help="监听端口"),
    share: bool = typer.Option(False, "--share", help="创建 Gradio 公共链接"),
    listen: bool = typer.Option(False, "--listen", help="监听 0.0.0.0（外部可访问）"),
):
    """启动 Gradio Web UI。"""
    _launch_webapp(host, port, share, listen)


@serve_app.command("api")
def serve_api(
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址"),
    port: int = typer.Option(8000, "--port", help="监听端口"),
):
    """启动 FastAPI REST API 服务。"""
    import uvicorn
    typer.secho(f"启动 FastAPI REST API @ http://{host}:{port}", fg=typer.colors.CYAN, bold=True)
    from .api_server import app as fastapi_app
    uvicorn.run(fastapi_app, host=host, port=port)


# ========================
# eval — 运行评测
# ========================


@app.command("eval")
def eval_cmd(
    benchmark: str = typer.Option(
        None, "--benchmark", "-b", help="benchmark 数据集 JSON 路径"
    ),
    search_mode: str = typer.Option(
        None, "--search-mode", "-m", help="检索模式 vector/local/global/mix/kg_only"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="返回结果数量"),
    no_generation: bool = typer.Option(
        False, "--no-generation", help="跳过 LLM 生成，只评检索"
    ),
):
    """运行评测套件（movie_kg_v1 benchmark）。"""
    import sys
    from .eval_harness import main as eval_main

    # 构造 sys.argv 传递给 eval_harness 的 argparse
    argv = ["eval"]
    if benchmark:
        argv += ["--benchmark", benchmark]
    if search_mode:
        argv += ["--search-mode", search_mode]
    argv += ["--top-k", str(top_k)]
    if no_generation:
        argv.append("--no-generation")
    original_argv = sys.argv
    sys.argv = argv
    try:
        eval_main()
    finally:
        sys.argv = original_argv


# ========================
# webui — serve web 快捷方式
# ========================


@app.command("webui", help="启动 Web UI（serve web 的快捷方式）")
def webui_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    share: bool = typer.Option(False, "--share"),
    listen: bool = typer.Option(False, "--listen"),
):
    _launch_webapp(host, port, share, listen)


def main():
    app()


if __name__ == "__main__":
    main()
