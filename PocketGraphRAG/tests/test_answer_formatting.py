from PocketGraphRAG.rag_system import PocketGraphRAG


def _make_minimal_rag():
    rag = PocketGraphRAG.__new__(PocketGraphRAG)
    return rag


def test_classify_question_type_definition():
    rag = _make_minimal_rag()
    assert rag._classify_question_type("大型语言模型的定义是什么？") == "definition"


def test_classify_question_type_list():
    rag = _make_minimal_rag()
    assert rag._classify_question_type("常见的编程语言有哪些？") == "list"


def test_classify_question_type_method():
    rag = _make_minimal_rag()
    assert rag._classify_question_type("如何安装 Python？") == "method"


def test_classify_question_type_parameter():
    rag = _make_minimal_rag()
    assert rag._classify_question_type("地球的半径是多少？") == "parameter"


def test_classify_question_type_comparison():
    rag = _make_minimal_rag()
    assert rag._classify_question_type("Python 和 Java 有什么区别？") == "comparison"


def test_classify_question_type_generic():
    rag = _make_minimal_rag()
    assert rag._classify_question_type("帮我总结一下这个知识库") == "generic"


def test_build_answer_prompt_adds_symptom_style_hint():
    rag = _make_minimal_rag()
    prompt = rag._build_answer_prompt(
        question="稻瘟病有什么症状？",
        context="[1] 来源: 稻瘟病\n叶片出现病斑",
        question_type="symptom",
    )
    assert "不要用" in prompt
    assert "不要混入防治方法" in prompt


def test_build_answer_prompt_adds_parameter_boundary_hint():
    rag = _make_minimal_rag()
    prompt = rag._build_answer_prompt(
        question="稻瘟病用三环唑的用量是多少？",
        context="[1] 来源: 三环唑\n推荐用量 40-75克/亩",
        question_type="parameter",
    )
    assert "现有信息没有直接给出该数值" in prompt
    assert "不要把弱关联写成确定答案" in prompt


def test_render_standardized_answer_removes_section_headers_but_keeps_content():
    rag = _make_minimal_rag()
    answer = rag._render_standardized_answer(
        question_type="method",
        answer_text=(
            "结论\n三环唑可用于防治稻瘟病[1][2]。\n\n"
            "关键事实\n三环唑可用于防治稻瘟病[1][2]。\n"
            "分蘖末期至孕穗初期可施药[2]。"
        )
    )
    assert "结论" not in answer
    assert "关键事实" not in answer
    assert "三环唑可用于防治稻瘟病[1][2]。" in answer
    assert "分蘖末期至孕穗初期可施药[2]。" in answer


def test_render_standardized_answer_dedupes_repeated_sentences():
    rag = _make_minimal_rag()
    answer = rag._render_standardized_answer(
        question_type="generic",
        answer_text=(
            "稻瘟病是一种真菌性病害[1]。\n"
            "稻瘟病是一种真菌性病害[1]。\n"
            "严重时会导致减产[2]。"
        )
    )
    assert answer.count("稻瘟病是一种真菌性病害[1]。") == 1
    assert "严重时会导致减产[2]。" in answer


def test_render_standardized_answer_keeps_natural_professional_tone():
    rag = _make_minimal_rag()
    answer = rag._render_standardized_answer(
        question_type="parameter",
        answer_text="三环唑用于防治稻瘟病时，推荐用量一般为40-75克/亩[1]。",
    )
    assert answer == "三环唑用于防治稻瘟病时，推荐用量一般为40-75克/亩[1]。"


def test_cleanup_answer_text_strips_template_headers_and_blank_lines():
    rag = _make_minimal_rag()
    cleaned = rag._cleanup_answer_text(
        "结论\n\n稻瘟病是一种真菌性病害[1]。\n\n关键事实\n\n严重时会导致减产[2]。"
    )
    assert cleaned == "稻瘟病是一种真菌性病害[1]。\n严重时会导致减产[2]。"


def test_cleanup_answer_text_removes_duplicate_lines_preserving_order():
    rag = _make_minimal_rag()
    cleaned = rag._cleanup_answer_text(
        "稻瘟病是一种真菌性病害[1]。\n稻瘟病是一种真菌性病害[1]。\n严重时会导致减产[2]。"
    )
    assert cleaned == "稻瘟病是一种真菌性病害[1]。\n严重时会导致减产[2]。"


def test_cleanup_answer_text_softens_document_style_prefix():
    rag = _make_minimal_rag()
    cleaned = rag._cleanup_answer_text(
        "根据提供的信息，三环唑用于防治稻瘟病时，推荐用量为40-75克/亩[1]。"
    )
    assert cleaned == "三环唑用于防治稻瘟病时，推荐用量为40-75克/亩[1]。"


def test_cleanup_answer_text_removes_inline_document_style_prefix():
    rag = _make_minimal_rag()
    cleaned = rag._cleanup_answer_text(
        "知识库中未找到直接证据。不过，根据提供的信息，三环唑可用于防治稻瘟病[1]。"
    )
    assert cleaned == "知识库中未找到直接证据。不过，三环唑可用于防治稻瘟病[1]。"


def test_cleanup_answer_text_drops_uncited_advice_tail():
    rag = _make_minimal_rag()
    cleaned = rag._cleanup_answer_text(
        "三环唑可用于防治稻瘟病[1]。\n请参考具体农药说明书或咨询农业技术人员。"
    )
    assert cleaned == "三环唑可用于防治稻瘟病[1]。"


def test_cleanup_answer_text_softens_symptom_document_phrase():
    rag = _make_minimal_rag()
    cleaned = rag._cleanup_answer_text(
        "稻瘟病的症状包括叶片出现褐色病斑[1]。"
    )
    assert cleaned == "稻瘟病常见表现为叶片出现褐色病斑[1]。"


def test_standardize_answer_prefers_cleaned_raw_answer():
    rag = _make_minimal_rag()
    question_type, answer = rag._standardize_answer(
        question="稻瘟病用三环唑的用量是多少？",
        answer_text="根据上述信息，三环唑用于防治稻瘟病时，推荐用量为40-75克/亩[1]。",
        results=[],
    )
    assert question_type == "parameter"
    assert answer == "三环唑用于防治稻瘟病时，推荐用量为40-75克/亩[1]。"


def test_format_retrieval_fallback_uses_natural_summary():
    rag = _make_minimal_rag()
    answer = rag._format_retrieval_fallback(
        [("推荐用量 40-75克/亩", 0.9, {"entity": "三环唑"})],
        {"seed_entities": ["三环唑"]},
    )
    assert "我目前只能根据命中的知识确认这些信息" in answer
    assert "三环唑：推荐用量 40-75克/亩[1]" in answer
    assert "## 检索到的结构化结果" not in answer


