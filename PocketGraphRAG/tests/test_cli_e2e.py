"""PocketGraphRAG CLI 端到端测试

覆盖范围：
1. Bug 回归：typer.echo(fg=...) 致命崩溃（P4-A 修复）
   - 根因：typer.echo 不支持 fg 参数，旧代码 LLM 失败时 CLI 直接 TypeError 退出
   - 修复：改用 typer.secho
2. Citation sources 列表展示（P4-B）
3. Score 归一化百分比展示（P4-B）
4. init / build / extract 命令基础冒烟（不崩溃）

测试策略：
- 用 typer.testing.CliRunner 调用 cli.app
- mock PocketGraphRAG.rag_system.PocketGraphRAG 类，避免依赖真实索引 / LLM
- mock build_index.build_index / kg_extractor.extract_knowledge_graph

Typer 是可选依赖（pyproject.toml [cli] extra）。未安装时整个文件 skip。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

from PocketGraphRAG.cli import app  # noqa: E402


# ==========================
# Fixtures
# ==========================


@pytest.fixture
def runner():
    """Typer CliRunner。

    注：typer 0.20+ / click 8.2+ 移除了 mix_stderr 参数，这里用默认配置。
    """
    return CliRunner()


def _make_mock_rag(
    answer_text: str = "稻瘟病可用三环唑防治[1][2]。",
    sources=None,
    kg_entities_matched: int = 3,
):
    """构造一个 mock PocketGraphRAG 实例。

    返回值用于 patch 的 side_effect：每次实例化都返回同一个 mock 实例。
    """
    if sources is None:
        # 默认 sources：分数故意用 RRF 量级（0.01）以验证归一化
        sources = [
            {"entity": "稻瘟病", "score": 0.0162, "text": "稻瘟病是由..."},
            {"entity": "三环唑", "score": 0.0108, "text": "三环唑是..."},
            {"entity": "稻瘟灵", "score": 0.0080, "text": "稻瘟灵也可..."},
        ]

    instance = MagicMock()
    instance.answer.return_value = {
        "answer": answer_text,
        "sources": sources,
        "question": "稻瘟病怎么防治？",
        "effective_query": "稻瘟病怎么防治？",
        "pipeline_info": {
            "multihop_used": False,
            "search_mode": "mix",
            "kg_entities_matched": kg_entities_matched,
            "query_rewritten": False,
            "hyde_used": False,
            "query_routed": False,
            "reranker_used": False,
            "vector_weight": 0.4,
            "refused": False,
        },
    }
    # answer_stream 返回空生成器（stream 路径冒烟）
    instance.answer_stream.return_value = iter(
        [{"chunk": answer_text}]
    )
    return instance


# ==========================
# Bug 回归：typer.echo(fg=...) 崩溃
# ==========================


class TestCliQaNoCrash:
    """P4-A 回归：cli qa 命令在任何路径下都不应抛 TypeError。

    旧 bug：typer.echo(text, fg=typer.colors.BLUE) → TypeError，LLM 失败时必崩。
    修复：所有带颜色的输出改用 typer.secho。
    """

    def test_qa_with_sources_does_not_crash(self, runner):
        """有 sources 时 qa 命令正常退出（exit_code=0），不抛 TypeError。"""
        mock_instance = _make_mock_rag()
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "稻瘟病怎么防治？", "-m", "mix"])

        # 关键断言：不崩溃
        assert result.exit_code == 0, (
            f"CLI 崩溃，exit_code={result.exit_code}\n"
            f"stdout: {result.stdout}\n"
            f"exception: {result.exception}"
        )
        # 答案正文应出现
        assert "稻瘟病可用三环唑防治" in result.stdout

    def test_qa_without_sources_does_not_crash(self, runner):
        """无 sources（空列表）时 qa 命令也不崩，且不渲染 Sources 区块。"""
        mock_instance = _make_mock_rag(sources=[])
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "测试问题", "-m", "mix"])

        assert result.exit_code == 0, (
            f"CLI 崩溃: {result.exception}"
        )
        assert "📚 来源" not in result.stdout, "无 sources 时不应渲染 Sources 区块"
        # KG 匹配实体数提示仍应出现（验证 typer.secho fg=BLUE 路径）
        assert "KG 匹配实体数" in result.stdout

    def test_qa_kg_entities_zero_does_not_crash(self, runner):
        """KG 匹配实体数为 0 时也不崩（边界路径覆盖 typer.secho）。"""
        mock_instance = _make_mock_rag(sources=[], kg_entities_matched=0)
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "无关问题", "-m", "mix"])

        assert result.exit_code == 0, f"CLI 崩溃: {result.exception}"
        assert "KG 匹配实体数: 0" in result.stdout

    def test_qa_stream_does_not_crash(self, runner):
        """stream 模式不崩（覆盖 answer_stream 路径）。"""
        mock_instance = _make_mock_rag()
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(
                app, ["qa", "稻瘟病怎么防治？", "-m", "mix", "--stream"]
            )

        assert result.exit_code == 0, f"CLI 崩溃: {result.exception}"
        assert "稻瘟病可用三环唑防治" in result.stdout


class TestCliRuntimeState:
    def test_qa_prints_dataset_index_and_mode(self, runner):
        mock_instance = _make_mock_rag()
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ), patch("PocketGraphRAG.cli.os.path.exists", return_value=True):
            result = runner.invoke(app, ["qa", "稻瘟病怎么防治？", "-m", "mix"])

        assert result.exit_code == 0
        assert "当前数据路径" in result.stdout
        assert "索引状态: 已构建" in result.stdout
        assert "检索模式: mix" in result.stdout

    def test_qa_shows_fallback_reason_when_response_mode_is_retrieval_fallback(
        self, runner
    ):
        mock_instance = _make_mock_rag(
            answer_text="## 检索到的结构化结果\n- 命中实体: 稻瘟病"
        )
        mock_instance.answer.return_value["pipeline_info"]["response_mode"] = (
            "retrieval_fallback"
        )
        mock_instance.answer.return_value["pipeline_info"]["fallback_reason"] = (
            "llm_empty_or_generic"
        )
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "稻瘟病怎么防治？", "-m", "mix"])

        assert result.exit_code == 0
        assert "回答模式: 检索兜底" in result.stdout


# ==========================
# Citation sources 列表展示
# ==========================


class TestCliSourcesDisplay:
    """P4-B：Sources 列表正确展示，编号 [1][2] 对应答案中的标注。"""

    def test_sources_list_rendered_with_indices(self, runner):
        """Sources 列表应按顺序编号 [1] [2] [3]，对应答案里的 [1][2] 标注。"""
        mock_instance = _make_mock_rag()
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "稻瘟病怎么防治？", "-m", "mix"])

        assert result.exit_code == 0
        assert "📚 来源" in result.stdout
        # 三个 source 都应编号出现
        assert "[1] 稻瘟病" in result.stdout
        assert "[2] 三环唑" in result.stdout
        assert "[3] 稻瘟灵" in result.stdout

    def test_sources_count_matches(self, runner):
        """Sources 区块条目数 = result['sources'] 长度。"""
        sources = [
            {"entity": "实体A", "score": 0.5, "text": "..."},
            {"entity": "实体B", "score": 0.3, "text": "..."},
        ]
        mock_instance = _make_mock_rag(sources=sources)
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "问题", "-m", "mix"])

        assert result.exit_code == 0
        assert "[1] 实体A" in result.stdout
        assert "[2] 实体B" in result.stdout
        assert "[3]" not in result.stdout, "不应出现第 3 条"

    def test_sources_render_source_type_labels_when_provided(self, runner):
        """来源元数据带 source_type 时，CLI 应显示 KG/向量/社区摘要标签。"""
        sources = [
            {"entity": "实体A", "score": 0.5, "text": "...", "source_type": "kg"},
            {"entity": "实体B", "score": 0.3, "text": "...", "source_type": "vector"},
            {
                "entity": "社区#1",
                "score": 0.2,
                "text": "...",
                "source_type": "community_summary",
            },
        ]
        mock_instance = _make_mock_rag(sources=sources)
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "问题", "-m", "mix"])

        assert result.exit_code == 0
        assert "[1] 实体A（相关度: 100%） [KG]" in result.stdout
        assert "[2] 实体B（相关度: 60%） [向量]" in result.stdout
        assert "[3] 社区#1（相关度: 40%） [社区摘要]" in result.stdout


# ==========================
# Score 归一化百分比展示
# ==========================


class TestCliScoreNormalization:
    """P4-B：RRF 量级分数（0.01）归一化为 0-100% 百分比展示。

    旧痛点：用户看到"相关度: 0.0162"以为系统坏了。
    修复：top 结果 100%，其他按比例。公式：pct = score / max_score * 100。
    """

    def test_top_source_is_100_percent(self, runner):
        """最高分 source 应显示 100%。"""
        sources = [
            {"entity": "稻瘟病", "score": 0.0162, "text": "..."},
            {"entity": "三环唑", "score": 0.0108, "text": "..."},
            {"entity": "稻瘟灵", "score": 0.0080, "text": "..."},
        ]
        mock_instance = _make_mock_rag(sources=sources)
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "问题", "-m", "mix"])

        assert result.exit_code == 0
        # top (0.0162) → 100%
        assert "[1] 稻瘟病（相关度: 100%）" in result.stdout

    def test_relative_percentages_correct(self, runner):
        """非 top source 的百分比 = score / max_score * 100（取整）。"""
        sources = [
            {"entity": "A", "score": 0.50, "text": "..."},  # 100%
            {"entity": "B", "score": 0.25, "text": "..."},  # 50%
            {"entity": "C", "score": 0.10, "text": "..."},  # 20%
        ]
        mock_instance = _make_mock_rag(sources=sources)
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "问题", "-m", "mix"])

        assert result.exit_code == 0
        assert "[1] A（相关度: 100%）" in result.stdout
        assert "[2] B（相关度: 50%）" in result.stdout
        assert "[3] C（相关度: 20%）" in result.stdout

    def test_all_zero_scores_no_division_error(self, runner):
        """所有 score=0 时不除零，全部显示 0%。"""
        sources = [
            {"entity": "A", "score": 0.0, "text": "..."},
            {"entity": "B", "score": 0.0, "text": "..."},
        ]
        mock_instance = _make_mock_rag(sources=sources)
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "问题", "-m", "mix"])

        assert result.exit_code == 0, f"除零崩溃: {result.exception}"
        assert "[1] A（相关度: 0%）" in result.stdout
        assert "[2] B（相关度: 0%）" in result.stdout

    def test_single_source_is_100_percent(self, runner):
        """只有一个 source 时也显示 100%。"""
        sources = [{"entity": "唯一", "score": 0.0001, "text": "..."}]
        mock_instance = _make_mock_rag(sources=sources)
        with patch(
            "PocketGraphRAG.rag_system.PocketGraphRAG",
            return_value=mock_instance,
        ):
            result = runner.invoke(app, ["qa", "问题", "-m", "mix"])

        assert result.exit_code == 0
        assert "[1] 唯一（相关度: 100%）" in result.stdout


# ==========================
# init 命令冒烟
# ==========================


class TestCliInitSmoke:
    """init 命令基础冒烟：不崩、输出关键配置项。"""

    def test_init_runs_without_crash(self, runner):
        """init 命令应正常退出，输出环境检查信息。"""
        result = runner.invoke(app, ["init"])
        # init 不应崩（即便没装 LLM / 没数据）
        assert result.exit_code == 0, f"init 崩溃: {result.exception}"
        assert "PocketGraphRAG 环境检查" in result.stdout
        # 关键配置项应出现
        assert "数据路径" in result.stdout
        assert "默认检索模式" in result.stdout

    def test_init_setup_does_not_crash(self, runner, tmp_path, monkeypatch):
        """--setup 不崩，且不污染真实仓库（mock 掉 shutil.copy 与 Path.exists）。

        init --setup 内部用局部 `import shutil` + `from pathlib import Path`，
        所以直接 patch 模块对象的属性即可拦截（局部 import 取的是同一模块对象）。
        """
        import shutil

        # 拦截 copy，避免向真实仓库写 .env
        monkeypatch.setattr(shutil, "copy", lambda *a, **k: None)

        # 拦截 Path.exists，让 .env / .env.example 都"不存在" → 走"未找到"分支
        from pathlib import Path

        monkeypatch.setattr(Path, "exists", lambda self: False)

        result = runner.invoke(app, ["init", "--setup"])
        assert result.exit_code == 0, f"init --setup 崩溃: {result.exception}"
        # "未找到 .env.example" 分支应被触发
        assert "未找到" in result.stdout
