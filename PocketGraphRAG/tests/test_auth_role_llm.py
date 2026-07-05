"""API Key 认证 + 角色 LLM 配置 测试

覆盖：
1. config.py 的 API_KEYS / API_AUTH_ENABLED / 角色配置解析
2. api_server.py 的 _verify_api_key 多 key + Bearer 支持
3. config.py 的 get_role_llm_config 4 角色解析
4. llm.py 的 call_llm role 参数传递（mock 验证）
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_async(coro):
    """同步运行 async 函数（避免依赖 pytest-asyncio）"""
    return asyncio.get_event_loop().run_until_complete(coro)


# ========================
# 1. API Key 配置
# ========================


class TestApiKeysConfig:
    """API_KEYS / API_AUTH_ENABLED / API_PUBLIC_PATHS"""

    def test_default_disabled(self):
        """未配置 POCKET_API_KEYS 时认证禁用"""
        from PocketGraphRAG import config as cfg

        assert isinstance(cfg.API_AUTH_ENABLED, bool)
        assert isinstance(cfg.API_KEYS, list)

    def test_public_paths_includes_health(self):
        from PocketGraphRAG.config import API_PUBLIC_PATHS

        assert "/api/health" in API_PUBLIC_PATHS
        assert "/docs" in API_PUBLIC_PATHS
        assert "/" in API_PUBLIC_PATHS

    def test_protected_prefix(self):
        from PocketGraphRAG.config import API_PROTECTED_PREFIX

        assert API_PROTECTED_PREFIX == "/api/"


# ========================
# 2. _verify_api_key 认证逻辑
# ========================


class TestVerifyApiKey:
    """api_server._verify_api_key 多 key + Bearer 支持"""

    def _make_request(self, path: str, x_api_key=None, authorization=None):
        """构造 mock Request"""
        request = MagicMock()
        request.url.path = path
        return request, x_api_key, authorization

    def test_auth_disabled_allows_all(self, monkeypatch):
        """未启用认证时所有请求放行"""
        from PocketGraphRAG import api_server

        monkeypatch.setattr(api_server, "_AUTH_ENABLED", False)
        request, x_key, auth = self._make_request("/api/qa")
        result = _run_async(api_server._verify_api_key(request, x_key, auth))
        assert result is None

    def test_public_path_bypasses_auth(self, monkeypatch):
        """公开路径不需要认证"""
        from PocketGraphRAG import api_server

        monkeypatch.setattr(api_server, "_AUTH_ENABLED", True)
        monkeypatch.setattr(api_server, "_ALL_API_KEYS", {"secret123"})
        request, _, _ = self._make_request("/api/health")
        result = _run_async(api_server._verify_api_key(request, None, None))
        assert result is None

    def test_x_api_key_header_valid(self, monkeypatch):
        """X-API-Key 头匹配时通过"""
        from PocketGraphRAG import api_server

        monkeypatch.setattr(api_server, "_AUTH_ENABLED", True)
        monkeypatch.setattr(api_server, "_ALL_API_KEYS", {"secret123"})
        request, _, _ = self._make_request("/api/qa")
        result = _run_async(api_server._verify_api_key(request, "secret123", None))
        assert result == "secret123"

    def test_x_api_key_header_invalid(self, monkeypatch):
        """X-API-Key 头不匹配时 401"""
        from PocketGraphRAG import api_server
        from fastapi import HTTPException

        monkeypatch.setattr(api_server, "_AUTH_ENABLED", True)
        monkeypatch.setattr(api_server, "_ALL_API_KEYS", {"secret123"})
        request, _, _ = self._make_request("/api/qa")
        with pytest.raises(HTTPException) as exc_info:
            _run_async(api_server._verify_api_key(request, "wrong-key", None))
        assert exc_info.value.status_code == 401

    def test_bearer_token_valid(self, monkeypatch):
        """Authorization: Bearer <key> 格式支持"""
        from PocketGraphRAG import api_server

        monkeypatch.setattr(api_server, "_AUTH_ENABLED", True)
        monkeypatch.setattr(api_server, "_ALL_API_KEYS", {"secret123"})
        request, _, _ = self._make_request("/api/qa")
        result = _run_async(
            api_server._verify_api_key(request, None, "Bearer secret123")
        )
        assert result == "secret123"

    def test_bearer_token_invalid(self, monkeypatch):
        """Bearer token 错误时 401"""
        from PocketGraphRAG import api_server
        from fastapi import HTTPException

        monkeypatch.setattr(api_server, "_AUTH_ENABLED", True)
        monkeypatch.setattr(api_server, "_ALL_API_KEYS", {"secret123"})
        request, _, _ = self._make_request("/api/qa")
        with pytest.raises(HTTPException):
            _run_async(
                api_server._verify_api_key(request, None, "Bearer wrong-key")
            )

    def test_no_credentials_raises(self, monkeypatch):
        """已启用认证但无任何凭据时 401"""
        from PocketGraphRAG import api_server
        from fastapi import HTTPException

        monkeypatch.setattr(api_server, "_AUTH_ENABLED", True)
        monkeypatch.setattr(api_server, "_ALL_API_KEYS", {"secret123"})
        request, _, _ = self._make_request("/api/qa")
        with pytest.raises(HTTPException):
            _run_async(api_server._verify_api_key(request, None, None))

    def test_non_protected_path_bypasses(self, monkeypatch):
        """非 /api/ 前缀路径放行（静态资源）"""
        from PocketGraphRAG import api_server

        monkeypatch.setattr(api_server, "_AUTH_ENABLED", True)
        monkeypatch.setattr(api_server, "_ALL_API_KEYS", {"secret123"})
        request, _, _ = self._make_request("/static/foo.js")
        result = _run_async(api_server._verify_api_key(request, None, None))
        assert result is None

    def test_multiple_keys_supported(self, monkeypatch):
        """多 key 轮换场景"""
        from PocketGraphRAG import api_server

        monkeypatch.setattr(api_server, "_AUTH_ENABLED", True)
        monkeypatch.setattr(
            api_server, "_ALL_API_KEYS", {"key1", "key2", "key3"}
        )
        request, _, _ = self._make_request("/api/qa")
        for k in ["key1", "key2", "key3"]:
            result = _run_async(api_server._verify_api_key(request, k, None))
            assert result == k

    def test_bearer_case_insensitive_prefix(self, monkeypatch):
        """Bearer 前缀大小写不敏感"""
        from PocketGraphRAG import api_server

        monkeypatch.setattr(api_server, "_AUTH_ENABLED", True)
        monkeypatch.setattr(api_server, "_ALL_API_KEYS", {"secret123"})
        request, _, _ = self._make_request("/api/qa")
        result = _run_async(
            api_server._verify_api_key(request, None, "bearer secret123")
        )
        assert result == "secret123"


# ========================
# 3. 角色 LLM 配置
# ========================


class TestRoleLlmConfig:
    """get_role_llm_config 4 角色解析"""

    def test_default_returns_none(self):
        """未配置时返回 (None, None)"""
        from PocketGraphRAG.config import get_role_llm_config

        provider, model = get_role_llm_config("extract")
        assert provider is None
        assert model is None

    def test_full_config_format(self, monkeypatch):
        """<provider>::<model> 格式同时切 provider + model"""
        from PocketGraphRAG import config as cfg

        monkeypatch.setattr(cfg, "EXTRACT_LLM_CONFIG", "dashscope::qwen-max")
        provider, model = cfg.get_role_llm_config("extract")
        assert provider == "dashscope"
        assert model == "qwen-max"

    def test_model_only_format(self, monkeypatch):
        """仅 model 时 provider 为 None"""
        from PocketGraphRAG import config as cfg

        monkeypatch.setattr(cfg, "EXTRACT_MODEL", "qwen-max")
        provider, model = cfg.get_role_llm_config("extract")
        assert provider is None
        assert model == "qwen-max"

    def test_full_config_takes_priority_over_model_only(self, monkeypatch):
        """POCKET_EXTRACT_LLM 优先于 POCKET_EXTRACT_MODEL"""
        from PocketGraphRAG import config as cfg

        monkeypatch.setattr(cfg, "EXTRACT_LLM_CONFIG", "dashscope::qwen-max")
        monkeypatch.setattr(cfg, "EXTRACT_MODEL", "qwen-flash")
        provider, model = cfg.get_role_llm_config("extract")
        assert provider == "dashscope"
        assert model == "qwen-max"

    def test_4_roles_all_supported(self, monkeypatch):
        """4 个角色都能正确解析"""
        from PocketGraphRAG import config as cfg

        monkeypatch.setattr(cfg, "EXTRACT_LLM_CONFIG", "dashscope::qwen-max")
        monkeypatch.setattr(cfg, "QUERY_LLM_CONFIG", "dashscope::qwen-flash")
        monkeypatch.setattr(cfg, "KEYWORDS_LLM_CONFIG", "siliconflow::Qwen/Qwen2.5-7B")
        monkeypatch.setattr(cfg, "VLM_LLM_CONFIG", "dashscope::qwen-vl-max")

        for role, expected_model in [
            ("extract", "qwen-max"),
            ("query", "qwen-flash"),
            ("keywords", "Qwen/Qwen2.5-7B"),
            ("vlm", "qwen-vl-max"),
        ]:
            provider, model = cfg.get_role_llm_config(role)
            assert model == expected_model, f"role={role}"

    def test_unknown_role_returns_none(self):
        """未知角色返回 (None, None)"""
        from PocketGraphRAG.config import get_role_llm_config

        provider, model = get_role_llm_config("unknown_role")
        assert provider is None
        assert model is None

    def test_full_config_strips_whitespace(self, monkeypatch):
        """配置值两端空格被 strip"""
        from PocketGraphRAG import config as cfg

        monkeypatch.setattr(cfg, "EXTRACT_LLM_CONFIG", "  dashscope :: qwen-max  ")
        provider, model = cfg.get_role_llm_config("extract")
        assert provider == "dashscope"
        assert model == "qwen-max"


# ========================
# 4. call_llm role 参数传递
# ========================


class TestCallLlmRoleIntegration:
    """call_llm 集成 role 配置（mock 验证不实际调用 LLM）"""

    def _disable_all_providers(self, monkeypatch):
        """禁用所有 LLM provider，让 call_llm 返回 None。

        注意：llm.py 在模块顶部 from .config import 引用了这些常量，
        所以必须 patch llm 模块本身的引用，而非 config 模块。
        """
        from PocketGraphRAG import llm as llm_mod

        monkeypatch.setattr(llm_mod, "OLLAMA_MODEL", "")
        monkeypatch.setattr(llm_mod, "FREELM_CN_API_KEY", "")
        monkeypatch.setattr(llm_mod, "SILICONFLOW_API_KEY", "")
        monkeypatch.setattr(llm_mod, "DEEPSEEK_API_KEY", "")
        monkeypatch.setattr(llm_mod, "DASHSCOPE_API_KEY", "")
        monkeypatch.setattr(llm_mod, "OPENAI_API_KEY", "")

    def test_call_llm_with_query_role(self, monkeypatch):
        """query 角色应能调用"""
        self._disable_all_providers(monkeypatch)
        from PocketGraphRAG.llm import call_llm

        result = call_llm("system", "user", role="query")
        assert result is None

    def test_call_llm_with_extract_role(self, monkeypatch):
        """extract 角色应能调用"""
        self._disable_all_providers(monkeypatch)
        from PocketGraphRAG.llm import call_llm

        result = call_llm("system", "user", role="extract")
        assert result is None

    def test_call_llm_with_keywords_role(self, monkeypatch):
        """keywords 角色应能调用（新角色）"""
        self._disable_all_providers(monkeypatch)
        from PocketGraphRAG.llm import call_llm

        result = call_llm("system", "user", role="keywords")
        assert result is None

    def test_call_llm_with_vlm_role(self, monkeypatch):
        """vlm 角色应能调用（新角色）"""
        self._disable_all_providers(monkeypatch)
        from PocketGraphRAG.llm import call_llm

        result = call_llm("system", "user", role="vlm")
        assert result is None
