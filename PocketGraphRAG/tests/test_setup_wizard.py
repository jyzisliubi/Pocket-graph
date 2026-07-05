"""Setup Wizard 单元测试（对标 LightRAG `make env-base`）"""

import os
import tempfile
from pathlib import Path

import pytest

from PocketGraphRAG.setup_wizard import (
    run_wizard,
    write_env_file,
    PROVIDERS,
    EMBEDDING_MODELS,
    STORAGE_BACKENDS,
)


# ==========================
# 1. 非交互模式
# ==========================


class TestNonInteractive:
    """非交互模式测试"""

    def test_ollama_default(self):
        content = run_wizard(non_interactive=True, provider="ollama")
        assert "POCKET_OLLAMA_MODEL" in content
        assert "qwen2.5:7b" in content
        assert "POCKET_EMBEDDING_MODEL=bge-small-zh" in content

    def test_dashscope_provider(self):
        content = run_wizard(non_interactive=True, provider="dashscope")
        assert "POCKET_DASHSCOPE_MODEL=qwen-plus" in content
        assert "POCKET_DASHSCOPE_API_KEY" in content

    def test_siliconflow_provider(self):
        content = run_wizard(non_interactive=True, provider="siliconflow")
        assert "POCKET_SILICONFLOW_MODEL=qwen-flash" in content
        assert "POCKET_SILICONFLOW_API_KEY" in content

    def test_unknown_provider_falls_back_to_ollama(self):
        content = run_wizard(non_interactive=True, provider="unknown")
        assert "POCKET_OLLAMA_MODEL" in content

    def test_all_providers_non_interactive(self):
        """所有 provider 都应能非交互生成配置"""
        for p in PROVIDERS:
            content = run_wizard(non_interactive=True, provider=p["id"])
            assert "POCKET_EMBEDDING_MODEL" in content
            # 应包含对应 provider 的环境变量
            assert p["env_vars"]["model"] in content


# ==========================
# 2. .env 文件写入
# ==========================


class TestWriteEnvFile:
    """.env 文件写入测试"""

    def test_write_new_env(self, tmp_path):
        env_path = str(tmp_path / ".env")
        content = "# test\nPOCKET_OLLAMA_MODEL=qwen2.5:7b\n"
        result = write_env_file(content, env_path)
        assert result == env_path
        assert os.path.exists(env_path)
        with open(env_path, encoding="utf-8") as f:
            assert f.read() == content

    def test_backup_existing_env(self, tmp_path):
        """已存在 .env 应备份为 .env.bak"""
        env_path = str(tmp_path / ".env")
        # 先写一个旧文件
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("# old config\n")
        # 再写新文件
        write_env_file("# new config\n", env_path)
        # 应有 .bak 备份
        assert os.path.exists(env_path + ".bak")
        with open(env_path + ".bak", encoding="utf-8") as f:
            assert "old config" in f.read()

    def test_no_double_backup(self, tmp_path):
        """已有 .env.bak 时不重复备份"""
        env_path = str(tmp_path / ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("# old\n")
        with open(env_path + ".bak", "w", encoding="utf-8") as f:
            f.write("# older backup\n")
        write_env_file("# new\n", env_path)
        # .bak 应保留最早的内容
        with open(env_path + ".bak", encoding="utf-8") as f:
            assert "older backup" in f.read()


# ==========================
# 3. 配置项定义完整性
# ==========================


class TestConfigDefinitions:
    """配置项定义测试"""

    def test_providers_have_required_fields(self):
        for p in PROVIDERS:
            assert "id" in p
            assert "label" in p
            assert "needs_key" in p
            assert "default_model" in p
            assert "env_vars" in p
            assert "model" in p["env_vars"]

    def test_embedding_models_have_aliases(self):
        for emb in EMBEDDING_MODELS:
            assert "id" in emb
            assert "label" in emb
            assert "alias" in emb

    def test_storage_backends_have_ids(self):
        for s in STORAGE_BACKENDS:
            assert "id" in s
            assert "label" in s

    def test_provider_ids_unique(self):
        ids = [p["id"] for p in PROVIDERS]
        assert len(ids) == len(set(ids))

    def test_default_provider_is_ollama(self):
        """默认第一个 provider 应是 ollama（本地优先）"""
        assert PROVIDERS[0]["id"] == "ollama"
