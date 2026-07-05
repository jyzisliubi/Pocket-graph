"""PocketGraphRAG Setup Wizard（对标 LightRAG `make env-base` 交互式配置）

首次使用 PocketGraphRAG 时引导用户完成 .env 配置：
1. LLM Provider 选择（Ollama / SiliconFlow / DeepSeek / DashScope / OpenAI / freellm-cn）
2. API Key 输入（脱敏显示）
3. 模型选择（带推荐默认）
4. Embedding 模型选择（bge-small-zh / bge-m3）
5. 存储后端选择（memory+faiss / neo4j / pgvector）
6. 可选：API 认证、Langfuse tracing、角色 LLM

使用方式：
    python -m PocketGraphRAG.setup_wizard

非交互模式（CI/批处理）：
    python -m PocketGraphRAG.setup_wizard --non-interactive --provider ollama
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# ==========================
# 配置项定义
# ==========================

PROVIDERS = [
    {
        "id": "ollama",
        "label": "Ollama（本地运行，无需 API Key）",
        "needs_key": False,
        "default_model": "qwen2.5:7b",
        "default_base": "http://localhost:11434/v1",
        "env_vars": {"model": "POCKET_OLLAMA_MODEL", "base": "POCKET_OLLAMA_API_BASE"},
    },
    {
        "id": "freellm-cn",
        "label": "freellm-cn（本地大模型网关，免费）",
        "needs_key": True,
        "default_model": "auto",
        "default_base": "http://localhost:8000/v1",
        "env_vars": {"model": "POCKET_FREELM_CN_MODEL", "base": "POCKET_FREELM_CN_API_BASE", "key": "POCKET_FREELM_CN_API_KEY"},
    },
    {
        "id": "siliconflow",
        "label": "SiliconFlow（硅基流动，有免费额度）",
        "needs_key": True,
        "default_model": "qwen-flash",
        "default_base": "https://api.siliconflow.cn/v1",
        "env_vars": {"model": "POCKET_SILICONFLOW_MODEL", "base": "POCKET_SILICONFLOW_API_BASE", "key": "POCKET_SILICONFLOW_API_KEY"},
    },
    {
        "id": "deepseek",
        "label": "DeepSeek（性价比高）",
        "needs_key": True,
        "default_model": "deepseek-chat",
        "default_base": "https://api.deepseek.com/v1",
        "env_vars": {"model": "POCKET_DEEPSEEK_MODEL", "base": "POCKET_DEEPSEEK_API_BASE", "key": "POCKET_DEEPSEEK_API_KEY"},
    },
    {
        "id": "dashscope",
        "label": "DashScope（阿里通义千问，推荐中文场景）",
        "needs_key": True,
        "default_model": "qwen-plus",
        "default_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_vars": {"model": "POCKET_DASHSCOPE_MODEL", "base": "POCKET_DASHSCOPE_API_BASE", "key": "POCKET_DASHSCOPE_API_KEY"},
    },
    {
        "id": "openai",
        "label": "OpenAI（GPT 系列）",
        "needs_key": True,
        "default_model": "gpt-4o-mini",
        "default_base": "https://api.openai.com/v1",
        "env_vars": {"model": "POCKET_OPENAI_MODEL", "base": "POCKET_OPENAI_API_BASE", "key": "POCKET_OPENAI_API_KEY"},
    },
]

EMBEDDING_MODELS = [
    {"id": "bge-small-zh", "label": "BAAI/bge-small-zh-v1.5（中文，512维，默认推荐）", "alias": "bge-small-zh"},
    {"id": "bge-m3", "label": "BAAI/bge-m3（多语言，1024维，跨语言检索推荐）", "alias": "bge-m3"},
    {"id": "bge-large-zh", "label": "BAAI/bge-large-zh-v1.5（中文，1024维，更高精度）", "alias": "bge-large-zh"},
]

STORAGE_BACKENDS = [
    {"id": "memory+faiss", "label": "内存图 + FAISS（默认，零依赖，<100万实体）"},
    {"id": "neo4j+faiss", "label": "Neo4j + FAISS（大规模图，需 Neo4j 服务）"},
    {"id": "memory+pgvector", "label": "内存图 + pgvector（PostgreSQL 向量库）"},
    {"id": "neo4j+pgvector", "label": "Neo4j + pgvector（生产级，全持久化）"},
]


# ==========================
# 交互式输入辅助
# ==========================

def _input(prompt: str, default: str = "") -> str:
    """带默认值的输入"""
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default


def _input_password(prompt: str) -> str:
    """密码输入（不回显，Windows 用 getpass）"""
    import getpass
    return getpass.getpass(f"{prompt}: ").strip()


def _choose(prompt: str, options: List[dict], default_idx: int = 0) -> dict:
    """选项菜单"""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = " *" if i == default_idx else ""
        print(f"  [{i + 1}] {opt['label']}{marker}")
    while True:
        choice = input(f"选择 (1-{len(options)}) [{default_idx + 1}]: ").strip()
        if not choice:
            return options[default_idx]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"  无效输入，请输入 1-{len(options)}")


def _confirm(prompt: str, default: bool = True) -> bool:
    """确认对话框"""
    suffix = " [Y/n]" if default else " [y/N]"
    val = input(f"{prompt}{suffix}: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "1", "true")


# ==========================
# Wizard 主流程
# ==========================

def run_wizard(non_interactive: bool = False, provider: str = "ollama") -> str:
    """运行配置向导，返回生成的 .env 内容

    Args:
        non_interactive: 非交互模式，使用默认值
        provider: 非交互模式下指定的 provider id

    Returns:
        .env 文件内容
    """
    lines: List[str] = [
        "# PocketGraphRAG 配置文件（由 setup_wizard 自动生成）",
        "# 手动编辑后需重启服务生效",
        "",
        "# =========================",
        "# LLM Provider 配置",
        "# =========================",
    ]

    if non_interactive:
        prov = next((p for p in PROVIDERS if p["id"] == provider), PROVIDERS[0])
        lines.append(f'{prov["env_vars"]["model"]}={prov["default_model"]}')
        lines.append(f'{prov["env_vars"]["base"]}={prov["default_base"]}')
        if prov["needs_key"]:
            lines.append(f'{prov["env_vars"]["key"]}=')
        lines.append("")
        lines.append("POCKET_EMBEDDING_MODEL=bge-small-zh")
        lines.append("")
        return "\n".join(lines)

    # 交互式
    print("=" * 60)
    print("  PocketGraphRAG Setup Wizard")
    print("  首次配置引导，所有配置可后续手动修改 .env")
    print("=" * 60)

    # 1. LLM Provider
    prov = _choose("选择 LLM Provider（推荐本地用 Ollama，中文场景用 DashScope）:", PROVIDERS, default_idx=0)
    lines.append(f'# Provider: {prov["id"]}')

    if prov["needs_key"]:
        api_key = _input_password(f'输入 {prov["id"]} API Key')
        if api_key:
            lines.append(f'{prov["env_vars"]["key"]}={api_key}')

    model = _input(f'模型名称', default=prov["default_model"])
    lines.append(f'{prov["env_vars"]["model"]}={model}')

    if "base" in prov["env_vars"]:
        api_base = _input('API Base URL', default=prov["default_base"])
        lines.append(f'{prov["env_vars"]["base"]}={api_base}')

    # 2. Embedding
    lines.append("")
    lines.append("# =========================")
    lines.append("# Embedding 模型")
    lines.append("# =========================")
    emb = _choose("选择 Embedding 模型（中文用 bge-small-zh，多语言用 bge-m3）:", EMBEDDING_MODELS, default_idx=0)
    lines.append(f"POCKET_EMBEDDING_MODEL={emb['alias']}")

    # 3. 存储后端
    lines.append("")
    lines.append("# =========================")
    lines.append("# 存储后端")
    lines.append("# =========================")
    storage = _choose("选择存储后端（<100万实体用默认即可）:", STORAGE_BACKENDS, default_idx=0)
    if "neo4j" in storage["id"]:
        lines.append("POCKET_GRAPH_STORE=neo4j")
        neo4j_uri = _input("Neo4j URI", default="bolt://localhost:7687")
        neo4j_user = _input("Neo4j 用户名", default="neo4j")
        neo4j_pass = _input_password("Neo4j 密码")
        lines.append(f"POCKET_NEO4J_URI={neo4j_uri}")
        lines.append(f"POCKET_NEO4J_USER={neo4j_user}")
        if neo4j_pass:
            lines.append(f"POCKET_NEO4J_PASSWORD={neo4j_pass}")
    if "pgvector" in storage["id"]:
        lines.append("POCKET_VECTOR_STORE=pgvector")
        pg_dsn = _input("PostgreSQL DSN", default="postgresql://user:pass@localhost:5432/pocketgraphrag")
        lines.append(f"POCKET_PGVECTOR_DSN={pg_dsn}")
        lines.append("POCKET_PGVECTOR_INDEX=hnsw")

    # 4. 可选：API 认证
    lines.append("")
    lines.append("# =========================")
    lines.append("# 可选：API 认证（公网部署必备）")
    lines.append("# =========================")
    if _confirm("是否启用 API Key 认证？", default=False):
        import secrets
        suggested = secrets.token_urlsafe(32)
        api_key = _input("API Key（留空自动生成）", default=suggested)
        lines.append(f"POCKET_API_KEYS={api_key}")

    # 5. 可选：角色 LLM
    if _confirm("是否配置角色特定 LLM（extract/query 用不同模型）？", default=False):
        lines.append("")
        lines.append("# =========================")
        lines.append("# 角色 LLM（extract 用强模型，query 用快模型）")
        lines.append("# =========================")
        extract_llm = _input("POCKET_EXTRACT_LLM (provider::model)", default=f'{prov["id"]}::{prov["default_model"]}')
        lines.append(f"POCKET_EXTRACT_LLM={extract_llm}")
        query_llm = _input("POCKET_QUERY_LLM (provider::model)", default=f'{prov["id"]}::{prov["default_model"]}')
        lines.append(f"POCKET_QUERY_LLM={query_llm}")

    # 6. 可选：Langfuse
    if _confirm("是否启用 Langfuse 追踪？", default=False):
        lines.append("")
        lines.append("# =========================")
        lines.append("# Langfuse Tracing")
        lines.append("# =========================")
        lines.append(f"POCKET_LANGFUSE_PUBLIC_KEY={_input('Langfuse public key')}")
        lines.append(f"POCKET_LANGFUSE_SECRET_KEY={_input_password('Langfuse secret key')}")
        lines.append(f"POCKET_LANGFUSE_HOST={_input('Langfuse host', default='https://cloud.langfuse.com')}")

    lines.append("")
    return "\n".join(lines)


def write_env_file(content: str, env_path: Optional[str] = None) -> str:
    """写入 .env 文件，返回路径"""
    if env_path is None:
        # 项目根目录（PocketGraphRAG/ 的上一级）
        here = Path(__file__).parent
        env_path = str(here.parent / ".env")

    # 备份现有 .env
    if os.path.exists(env_path):
        backup = env_path + ".bak"
        if not os.path.exists(backup):
            os.rename(env_path, backup)
            print(f"  已备份原 .env 到 {backup}")

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)
    return env_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PocketGraphRAG Setup Wizard")
    parser.add_argument("--non-interactive", action="store_true", help="非交互模式，使用默认值")
    parser.add_argument("--provider", default="ollama", choices=[p["id"] for p in PROVIDERS],
                        help="非交互模式下的 provider")
    parser.add_argument("--output", default=None, help="输出 .env 文件路径（默认 ./.env）")
    args = parser.parse_args()

    content = run_wizard(non_interactive=args.non_interactive, provider=args.provider)
    env_path = write_env_file(content, args.output)
    print(f"\n✓ 配置已写入: {env_path}")
    print(f"  下一步: python -m PocketGraphRAG.api_server")


if __name__ == "__main__":
    main()
