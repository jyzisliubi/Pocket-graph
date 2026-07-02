import inspect
from unittest.mock import patch

from PocketGraphRAG import app as cli_app
from PocketGraphRAG import webapp


def test_get_recommended_questions_for_example_dataset():
    questions = webapp.get_recommended_questions("example")
    assert questions[0] == "这部电影讲了什么？"
    assert "这部电影的主角是谁？" in questions


def test_load_recommended_questions_markdown_for_user_dataset():
    markdown = webapp.load_recommended_questions_markdown("user")
    assert "推荐问题" in markdown
    assert "这个数据集里最重要的实体有哪些？" in markdown


def test_load_recommended_questions_markdown_uses_current_dataset(monkeypatch):
    monkeypatch.setattr(webapp, "current_dataset", "user")
    monkeypatch.setattr(
        webapp,
        "get_recommended_questions",
        lambda dataset_key: ["先看答案有没有来源", "再看图谱有没有承接"],
    )

    md = webapp.load_recommended_questions_markdown()

    assert "### 推荐问题" in md
    assert "- 先看答案有没有来源" in md
    assert "- 再看图谱有没有承接" in md


def test_load_runtime_status_shows_example_dataset_and_index_ready(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(webapp, "current_dataset", "example")
    monkeypatch.setattr(webapp, "DATA_PATH", str(tmp_path / "rice.txt"))
    monkeypatch.setattr(webapp, "INDEX_DIR", str(tmp_path / "index"))
    (tmp_path / "rice.txt").write_text("稻瘟病|防治药剂|三环唑\n", encoding="utf-8")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "faiss.index").write_text("ok", encoding="utf-8")

    with patch("PocketGraphRAG.webapp.detect_active_provider", return_value="ollama"):
        html = webapp.load_runtime_status()

    assert "示例数据（电影知识图谱）" in html
    assert "索引已就绪" in html
    assert "ollama" in html.lower()


def test_load_runtime_status_prefers_dataset_helper_and_provider_label(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(webapp, "current_dataset", "example")
    monkeypatch.setattr(webapp, "DATA_PATH", str(tmp_path / "rice.txt"))
    monkeypatch.setattr(webapp, "INDEX_DIR", str(tmp_path / "index"))
    (tmp_path / "rice.txt").write_text("稻瘟病|防治药剂|三环唑\n", encoding="utf-8")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "faiss.index").write_text("ok", encoding="utf-8")

    with patch("PocketGraphRAG.webapp.detect_active_provider", return_value="ollama"):
        html = webapp.load_runtime_status()

    assert "当前数据集" in html
    assert "示例数据（电影知识图谱）" in html
    assert "Ollama（本地）" in html
    assert "索引已就绪" in html


def test_load_runtime_status_shows_index_not_ready_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "current_dataset", "user")
    monkeypatch.setattr(webapp, "USER_TRIPLES_PATH", str(tmp_path / "user_triples.txt"))
    monkeypatch.setattr(webapp, "INDEX_DIR", str(tmp_path / "index"))
    (tmp_path / "user_triples.txt").write_text("苹果|属于|水果\n", encoding="utf-8")
    (tmp_path / "index").mkdir()

    with patch("PocketGraphRAG.webapp.detect_active_provider", return_value="ollama"):
        html = webapp.load_runtime_status()

    assert "用户数据" in html
    assert "索引未就绪" in html


def test_format_pipeline_info_surfaces_fallback_and_failure_bucket():
    html = webapp.format_pipeline_info(
        {
            "search_mode": "mix",
            "response_mode": "retrieval_fallback",
            "fallback_reason": "llm_empty_or_generic",
            "failure_bucket": "insufficient_context",
            "kg_path": {"search_type": "mix", "seed_entities": ["稻瘟病"]},
        }
    )
    assert "检索兜底" in html
    assert "证据不够完整" in html


def test_format_pipeline_info_marks_generation_step_when_answer_uses_fallback():
    output = webapp.format_pipeline_info(
        {
            "search_mode": "mix",
            "response_mode": "retrieval_fallback",
            "question_type": "list",
        }
    )

    assert "🤖 LLM 生成答案" in output
    assert "检索兜底" in output


def test_format_pipeline_info_shows_question_type():
    html = webapp.format_pipeline_info(
        {
            "search_mode": "mix",
            "response_mode": "llm_standardized",
            "question_type": "method",
            "kg_path": {"search_type": "mix", "seed_entities": ["稻瘟病"]},
        }
    )
    assert "问题类型" in html
    assert "method" in html


def test_format_pipeline_info_compacts_noisy_seed_entities():
    html = webapp.format_pipeline_info(
        {
            "search_mode": "mix",
            "response_mode": "llm_standardized",
            "question_type": "symptom",
            "kg_path": {
                "search_type": "mix",
                "seed_entities": ["稻瘟病", "感稻瘟病", "稻瘟病高发区或历史发病区域", "Ⅱ优633"],
            },
        }
    )
    assert "稻瘟病" in html
    assert "感稻瘟病" not in html
    assert "高发区或历史发病区域" not in html
    assert "Ⅱ优633" not in html


def test_format_pipeline_info_empty_state_guides_next_action():
    html = webapp.format_pipeline_info({})
    assert "等待提问" in html


def test_format_sources_shows_source_type_badges():
    html = webapp.format_sources(
        [
            {"entity": "稻瘟病", "score": 0.9, "text": "KG 证据", "source_type": "kg"},
            {
                "entity": "社区#1",
                "score": 0.7,
                "text": "社区摘要证据",
                "source_type": "community_summary",
            },
        ]
    )
    assert "KG" in html
    assert "社区摘要" in html
    assert "KG命中" in html
    assert "社区命中" in html


def test_format_sources_uses_relative_relevance_for_vector_scores():
    html = webapp.format_sources(
        [
            {"entity": "稻瘟病", "score": 0.02, "text": "核心证据", "source_type": "vector"},
            {"entity": "水稻穗颈瘟病", "score": 0.01, "text": "次级证据", "source_type": "vector"},
        ]
    )
    assert "相对相关度 100%" in html
    assert "相对相关度 50%" in html
    assert "0.0%" not in html


def test_format_sources_groups_low_relevance_candidates_but_keeps_indices():
    html = webapp.format_sources(
        [
            {
                "entity": "稻瘟病",
                "score": 0.91,
                "text": "命中症状证据",
                "source_type": "vector",
            },
            {
                "entity": "水稻穗颈瘟病",
                "score": 0.88,
                "text": "命中穗颈症状",
                "source_type": "vector",
            },
            {
                "entity": "异稻缘蝽",
                "score": 0.86,
                "text": "明显偏题的候选来源",
                "source_type": "vector",
            },
        ],
        query="稻瘟病有什么症状？",
    )
    assert "优先展示和当前问题更贴近的核心证据" in html
    assert "查看其余候选来源（1 条）" in html
    assert "[3] 异稻缘蝽" in html


def test_default_graph_hint_guides_first_action():
    assert "显示图谱概览" in webapp.DEFAULT_GRAPH_HINT


class DummyDemo:
    def __init__(self):
        self.launch_kwargs = None
        self.launch_count = 0
        self.fail_first = False

    def launch(self, **kwargs):
        self.launch_count += 1
        if self.fail_first and self.launch_count == 1:
            raise OSError("Cannot find empty port")
        self.launch_kwargs = kwargs


def test_launch_with_fallback_success():
    demo = DummyDemo()
    webapp._launch_with_fallback(demo, {"server_port": 7860})
    assert demo.launch_count == 1
    assert demo.launch_kwargs == {"server_port": 7860}


def test_launch_with_fallback_recovers_from_occupied_port():
    demo = DummyDemo()
    demo.fail_first = True
    webapp._launch_with_fallback(demo, {"server_port": 7860})
    assert demo.launch_count == 2
    assert "server_port" not in demo.launch_kwargs


def test_launch_with_fallback_prints_port_hint(capsys):
    demo = DummyDemo()
    demo.fail_first = True

    webapp._launch_with_fallback(demo, {"server_port": 7860})

    captured = capsys.readouterr()
    assert "默认端口 7860 被占用" in captured.out


def test_build_ui_wires_self_check_toggle_into_chat_inputs():
    source = inspect.getsource(webapp.build_ui)
    chat_inputs_block = source.split("chat_inputs = [", 1)[1].split("]", 1)[0]

    assert "use_query_router_cb" in chat_inputs_block
    assert "use_self_check_cb" in chat_inputs_block


def test_launch_with_fallback_raises_other_os_errors():
    import pytest

    demo = DummyDemo()

    def fail_launch(**kwargs):
        raise OSError("Permission denied")

    demo.launch = fail_launch

    with pytest.raises(OSError, match="Permission denied"):
        webapp._launch_with_fallback(demo, {"server_port": 7860})


def test_print_banner_uses_productized_copy(capsys):
    cli_app.print_banner()

    captured = capsys.readouterr()
    assert "PocketGraphRAG 问答系统" in captured.out
    assert "上传资料 -> 本地知识图谱 -> 带证据问答" in captured.out
    assert "垂直领域图 RAG 框架" not in captured.out
