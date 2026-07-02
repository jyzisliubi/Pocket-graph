"""
LLM 设置管理器（Web UI 使用）

让用户在网页里直接配置 LLM provider / API Key / 模型名，
无需手动编辑 .env 文件，也无需重启服务。

功能：
- load_llm_settings()  读取当前 .env + config 的 LLM 配置
- save_llm_settings()  写入 .env 并热同步到 PocketGraphRAG.config 模块变量
- reset_rag_instance() 让下次问答重新初始化 RAG，使用新配置
- test_llm_connection() 调一次 call_llm 验证当前配置是否可用
- list_ollama_models()  列出本地 Ollama 已下载的模型

依赖：python-dotenv（核心依赖）
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from dotenv import dotenv_values, set_key
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "python-dotenv 未安装。请运行: pip install python-dotenv\n"
        "或 pip install -e '.[dev]'"
    ) from _e

from . import config

ENV_PATH = Path(config._PROJECT_ROOT) / ".env"

# 支持的 LLM Provider
PROVIDERS = {
    "ollama": {
        "label": "Ollama（本地离线，推荐）",
        "needs_key": False,
        "fields": ["OLLAMA_MODEL", "OLLAMA_API_BASE"],
        "default_model": "qwen2.5:7b",
        "default_base": "http://localhost:11434/v1",
        "help": "需先安装 Ollama (https://ollama.com/) 并拉取模型: ollama pull qwen2.5:7b",
    },
    "siliconflow": {
        "label": "SiliconFlow 硅基流动（国内访问快，有免费额度）",
        "needs_key": True,
        "fields": ["SILICONFLOW_API_KEY", "SILICONFLOW_MODEL"],
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "apply_url": "https://siliconflow.cn/",
    },
    "deepseek": {
        "label": "DeepSeek（推理能力强，有免费额度）",
        "needs_key": True,
        "fields": ["DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"],
        "default_model": "deepseek-chat",
        "apply_url": "https://platform.deepseek.com/",
    },
    "dashscope": {
        "label": "阿里云 DashScope 通义千问（支持图片 VLM）",
        "needs_key": True,
        "fields": ["DASHSCOPE_API_KEY", "DASHSCOPE_MODEL"],
        "default_model": "qwen-plus",
        "apply_url": "https://dashscope.console.aliyun.com/",
    },
    "openai": {
        "label": "OpenAI 或兼容接口",
        "needs_key": True,
        "fields": ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_MODEL"],
        "default_model": "gpt-3.5-turbo",
        "default_base": "https://api.openai.com/v1",
    },
}


def _read_env() -> dict:
    """读取 .env 文件（不存在则返回空 dict）"""
    if not ENV_PATH.exists():
        return {}
    return {k: v for k, v in dotenv_values(str(ENV_PATH)).items() if v is not None}


def detect_active_provider() -> str:
    """根据当前 config 判断激活的 provider（与 llm.py 的优先级一致）"""
    if config.OLLAMA_MODEL:
        return "ollama"
    if config.SILICONFLOW_API_KEY:
        return "siliconflow"
    if config.DEEPSEEK_API_KEY:
        return "deepseek"
    if config.DASHSCOPE_API_KEY:
        return "dashscope"
    if config.OPENAI_API_KEY:
        return "openai"
    return ""


def load_llm_settings() -> dict:
    """加载当前 LLM 配置（.env 优先，回退到 config 模块变量）"""
    env = _read_env()
    return {
        "provider": env.get("DEFAULT_LLM_PROVIDER", "") or detect_active_provider(),
        # Ollama
        "OLLAMA_MODEL": env.get("OLLAMA_MODEL", config.OLLAMA_MODEL),
        "OLLAMA_API_BASE": env.get("OLLAMA_API_BASE", config.OLLAMA_API_BASE),
        # SiliconFlow
        "SILICONFLOW_API_KEY": env.get(
            "SILICONFLOW_API_KEY", config.SILICONFLOW_API_KEY
        ),
        "SILICONFLOW_MODEL": env.get("SILICONFLOW_MODEL", config.SILICONFLOW_MODEL),
        # DeepSeek
        "DEEPSEEK_API_KEY": env.get("DEEPSEEK_API_KEY", config.DEEPSEEK_API_KEY),
        "DEEPSEEK_MODEL": env.get("DEEPSEEK_MODEL", config.DEEPSEEK_MODEL),
        # DashScope
        "DASHSCOPE_API_KEY": env.get("DASHSCOPE_API_KEY", config.DASHSCOPE_API_KEY),
        "DASHSCOPE_MODEL": env.get("DASHSCOPE_MODEL", config.DASHSCOPE_MODEL),
        # OpenAI
        "OPENAI_API_KEY": env.get("OPENAI_API_KEY", config.OPENAI_API_KEY),
        "OPENAI_API_BASE": env.get("OPENAI_API_BASE", config.OPENAI_API_BASE),
        "OPENAI_MODEL": env.get("OPENAI_MODEL", config.OPENAI_MODEL),
    }


def save_llm_settings(values: dict) -> None:
    """保存 LLM 配置到 .env 并热同步到 config 模块。

    Args:
        values: load_llm_settings() 返回的同结构 dict

    保存后会清空非当前 provider 的敏感字段，避免多个 key 同时生效造成混乱。
    """
    provider = (values.get("provider") or "").strip().lower()

    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ENV_PATH.exists():
        ENV_PATH.touch()

    # 先把所有 provider 的字段写入（空字符串也算写入，会覆盖旧值）
    all_fields = []
    for p, info in PROVIDERS.items():
        all_fields.extend(info["fields"])
    # 加上 DEFAULT_LLM_PROVIDER
    set_key(str(ENV_PATH), "DEFAULT_LLM_PROVIDER", provider)

    for field in all_fields:
        raw_val = values.get(field, "")
        val = str(raw_val).strip()
        # 对非当前 provider 的 API_KEY 字段清空，确保只有选中的 provider 生效
        info = _field_to_provider(field)
        if info and field.endswith("API_KEY") and info != provider:
            val = ""
        set_key(str(ENV_PATH), field, val)

    # 热同步到 config 模块变量（让 llm.py 立即读到新值）
    _sync_config_from_env()

    # 重置全局 RAG 实例，下次问答时重新初始化
    reset_rag_instance()


def _field_to_provider(field: str) -> Optional[str]:
    for p, info in PROVIDERS.items():
        if field in info["fields"]:
            return p
    return None


def _sync_config_from_env() -> None:
    """重新读取 .env 并同步到 PocketGraphRAG.config 模块变量"""
    env = _read_env()
    mapping = {
        "OLLAMA_MODEL": "OLLAMA_MODEL",
        "OLLAMA_API_BASE": "OLLAMA_API_BASE",
        "SILICONFLOW_API_KEY": "SILICONFLOW_API_KEY",
        "SILICONFLOW_MODEL": "SILICONFLOW_MODEL",
        "DEEPSEEK_API_KEY": "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL": "DEEPSEEK_MODEL",
        "DASHSCOPE_API_KEY": "DASHSCOPE_API_KEY",
        "DASHSCOPE_MODEL": "DASHSCOPE_MODEL",
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "OPENAI_API_BASE": "OPENAI_API_BASE",
        "OPENAI_MODEL": "OPENAI_MODEL",
    }
    for env_key, cfg_key in mapping.items():
        if env_key in env:
            setattr(config, cfg_key, env[env_key])


def reset_rag_instance() -> None:
    """重置 webapp 的全局 RAG 实例，下次问答时重新加载。

    这里用延迟导入避免循环依赖：webapp 依赖 settings_manager，
    settings_manager 不应直接依赖 webapp。
    """
    try:
        from . import webapp as _webapp

        _webapp.rag = None
    except ImportError:
        pass  # webapp 未导入时不报错（非错误，故不记日志）


def test_llm_connection(provider: str = None) -> tuple[bool, str]:
    """测试当前 LLM 配置是否可用。

    Args:
        provider: 指定 provider 测试；None 则用当前激活的

    Returns:
        (success, message)
    """
    from .llm import call_llm, get_active_provider, has_llm

    if not has_llm():
        return False, "未配置任何 LLM 后端。请在上方填写 API Key 或选择 Ollama。"

    active = get_active_provider()
    # 发一个最简单的 ping，max_tokens 给小一点省 token
    try:
        result = call_llm(
            "你是一个测试助手。",
            "请回复：pong",
            temperature=0.0,
            max_tokens=10,
        )
        if result:
            preview = result.strip()[:60]
            return True, f"✓ 连接成功！当前后端: {active}\n回复预览: {preview}"
        return (
            False,
            f"✗ 调用返回空。当前后端: {active}\n请检查 API Key 或 Ollama 是否在运行。",
        )
    except Exception as e:
        return False, f"✗ 调用失败: {e}\n当前后端: {get_active_provider()}"


def list_ollama_models(base_url: str = None) -> tuple[bool, list, str]:
    """列出本地 Ollama 已下载的模型。

    Returns:
        (success, model_list, message)
    """
    import requests

    base = base_url or config.OLLAMA_API_BASE
    # Ollama 原生 list 接口: GET /api/tags（不是 /v1 前缀）
    tags_url = base.rstrip("/").replace("/v1", "") + "/api/tags"
    try:
        resp = requests.get(tags_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        return True, models, f"找到 {len(models)} 个已下载模型"
    except Exception as e:
        return (
            False,
            [],
            (
                f"无法连接 Ollama ({tags_url}): {e}\n"
                "请确认: 1) 已安装 Ollama  2) ollama serve 正在运行  3) 地址正确"
            ),
        )


def pull_ollama_model(model_name: str, base_url: str = None):
    """从 Ollama 拉取（下载）模型，流式返回进度文本。

    用 Ollama 原生 /api/pull 接口，stream=true，逐行返回 JSON 状态。
    Gradio 端用 gr.Textbox 流式接收。

    用法（在 Gradio 里）::

        pull_btn.click(fn=pull_ollama_model, inputs=[model_tb, base_tb],
                       outputs=[progress_md])

    Yields:
        str: 进度文本片段（含换行）
    """
    import json as _json

    import requests

    model_name = (model_name or "").strip()
    if not model_name:
        yield "❌ 请先填写要拉取的模型名（例如 qwen2.5:7b）"
        return

    base = base_url or config.OLLAMA_API_BASE
    pull_url = base.rstrip("/").replace("/v1", "") + "/api/pull"

    yield f"🚀 开始拉取模型 `{model_name}`...\n"
    yield f"   目标: {pull_url}\n"
    yield "   ⏳ 首次拉取大模型可能需要几分钟到几十分钟（取决于网络与模型大小）\n\n"

    try:
        # stream=True 让 requests 逐行返回
        resp = requests.post(
            pull_url,
            json={"name": model_name, "stream": True},
            timeout=None,  # 拉模型可能很久，不设超时
            stream=True,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        yield (
            f"❌ 无法连接 Ollama 服务: {e}\n"
            "请确认:\n"
            "  1) 已安装 Ollama: https://ollama.com/\n"
            "  2) Ollama 服务在运行（终端执行 `ollama serve` 或已开机自启）\n"
            f"  3) API 地址正确: {pull_url}\n"
        )
        return
    except Exception as e:
        yield f"❌ 拉取请求失败: {e}\n"
        return

    last_status = ""
    last_pct = -1
    try:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = _json.loads(line.decode("utf-8"))
            except _json.JSONDecodeError:
                continue

            status = data.get("status", "")
            total = data.get("total")
            completed = data.get("completed")

            if status != last_status and status:
                yield f"▸ {status}\n"
                last_status = status

            # 拉取进度百分比
            if total and completed is not None and total > 0:
                pct = int(completed * 100 / total)
                if pct != last_pct and pct % 5 == 0:  # 每 5% 报一次，避免刷屏
                    # 简易进度条
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    yield f"\r   [{bar}] {pct}%  ({completed // 1024 // 1024}MB / {total // 1024 // 1024}MB)"
                    last_pct = pct
    finally:
        resp.close()

    yield "\n\n"
    # 拉取完成后列出已下载模型确认
    ok, models, msg = list_ollama_models(base_url)
    if ok and model_name in models:
        yield f"✅ 模型 `{model_name}` 拉取成功！现在可以在「LLM 设置」里选中它使用了。\n"
        yield f"已下载模型: {models}"
    elif ok:
        yield f"⚠️ 拉取流程已结束，但模型列表里没找到 `{model_name}`。\n"
        yield f"当前已下载模型: {models}"
    else:
        yield "✅ 拉取流程已结束。请刷新「列出本地已下载模型」确认。"


def is_ollama_running(base_url: str = None) -> bool:
    """快速检测 Ollama 服务是否在运行"""
    import requests

    base = base_url or config.OLLAMA_API_BASE
    tags_url = base.rstrip("/").replace("/v1", "") + "/api/tags"
    try:
        resp = requests.get(tags_url, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False
