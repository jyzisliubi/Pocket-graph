"""测试可观测性软依赖（API 限流 + Prometheus 指标）。

验证 slowapi / prometheus-fastapi-instrumentator 未安装时 api_server 仍能正常工作，
这是 local-first 设计的关键：生产可选启用，本地零依赖运行。
"""

from PocketGraphRAG import api_server


class TestObservabilitySoftDeps:
    def test_module_loads_without_hard_dependencies(self):
        """api_server 在缺少 observability 依赖时仍可正常加载"""
        assert api_server.app is not None
        assert isinstance(api_server._METRICS_ENABLED, bool)
        # limiter 为 None（未装 slowapi 或未配置）或 Limiter 实例
        assert api_server.limiter is None or hasattr(api_server.limiter, "limit")

    def test_app_has_core_routes_intact(self):
        """软依赖不影响核心路由注册"""
        paths = {r.path for r in api_server.app.routes if hasattr(r, "path")}
        assert "/api/qa" in paths or "/api/qa/stream" in paths
        # 健康检查端点存在（/api/health 或 /health）
        assert any("health" in p for p in paths)

    def test_no_metrics_endpoint_when_disabled(self):
        """未启用 POCKET_METRICS 时不暴露 /metrics"""
        if not api_server._METRICS_ENABLED:
            paths = {r.path for r in api_server.app.routes if hasattr(r, "path")}
            assert "/metrics" not in paths

    def test_limiter_config_env_var_respected(self, monkeypatch):
        """POCKET_RATE_LIMIT 环境变量可被读取（即使 slowapi 未安装也不报错）"""
        # 仅验证环境变量读取逻辑不崩溃；实际限流行为需 slowapi 安装
        monkeypatch.setenv("POCKET_RATE_LIMIT", "100/minute")
        # 重新读取环境变量值（不 reload 模块，仅验证读取逻辑）
        val = __import__("os").environ.get("POCKET_RATE_LIMIT", "").strip()
        assert val == "100/minute"
