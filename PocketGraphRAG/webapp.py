"""
Gradio Web 界面

使用方式：
    python -m PocketGraphRAG.webapp

提供一个可视化的问答界面，展示 RAG 的完整流程：
- 用户输入问题
- 系统检索最相关的知识块
- LLM 基于检索结果生成回答
- 界面同时展示回答、参考来源和 Pipeline 信息

数据管理 Tab：
- 上传文档 → 自动抽取三元组 → 构建知识图谱 → 重建索引
"""

import json
import os

# 修复本地访问问题：确保 localhost/127.0.0.1 不走代理
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"


def _apply_gradio_compat_patches():
    """Apply compatibility patches for known gradio_client version mismatches.

    Some gradio_client versions (e.g. 1.3.x used with Gradio 4.44+) pass non-dict
    schema values to get_type() / _json_schema_to_python_type(), causing TypeError.
    We patch defensively: if the functions don't exist or signatures differ, the
    patch is a no-op rather than crashing startup.
    """
    try:
        import gradio_client.utils as _gcu

        if hasattr(_gcu, "get_type"):
            _orig_get_type = _gcu.get_type

            def _safe_get_type(schema):
                if not isinstance(schema, dict):
                    return "any"
                try:
                    return _orig_get_type(schema)
                except Exception:
                    return "any"

            _gcu.get_type = _safe_get_type

        if hasattr(_gcu, "_json_schema_to_python_type"):
            _orig_jstpt = _gcu._json_schema_to_python_type

            def _safe_jstpt(schema, defs=None):
                if not isinstance(schema, dict):
                    return "any"
                try:
                    return _orig_jstpt(schema, defs)
                except Exception:
                    return "any"

            _gcu._json_schema_to_python_type = _safe_jstpt
    except Exception:
        pass


_apply_gradio_compat_patches()

import gradio as gr

from .config import (
    DATA_PATH,
    EMBEDDING_MODEL,
    INDEX_DIR,
    RELATION_TEMPLATES,
    REVERSE_LINK_RELATIONS,
)
from .kg_extractor import extract_knowledge_graph_stream
from .llm import get_active_provider, has_llm
from .logging_config import get_logger
from .rag_system import PocketGraphRAG
from .settings_manager import (
    PROVIDERS as LLM_PROVIDERS,
)
from .settings_manager import (
    detect_active_provider,
    is_ollama_running,
    list_ollama_models,
    load_llm_settings,
    pull_ollama_model,
    save_llm_settings,
    test_llm_connection,
)

logger = get_logger(__name__)

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 用户数据目录
USER_DATA_DIR = os.path.join(_PROJECT_ROOT, "user_data")
USER_TRIPLES_PATH = os.path.join(USER_DATA_DIR, "triples.txt")
# 临时图谱 HTML 目录
GRAPH_TMP_DIR = os.path.join(USER_DATA_DIR, "graph_tmp")
os.makedirs(GRAPH_TMP_DIR, exist_ok=True)
# 已上传文档归档目录（E: 文档管理）
USER_DOCS_DIR = os.path.join(USER_DATA_DIR, "docs")
DOCS_MANIFEST_PATH = os.path.join(USER_DOCS_DIR, "manifest.json")
os.makedirs(USER_DOCS_DIR, exist_ok=True)


# 全局 RAG 实例（懒加载）
rag: PocketGraphRAG = None
# 当前使用的数据集: "example" 或 "user"
current_dataset: str = "example"
# 已抽取的三元组缓存
extracted_triples: list = []
# embedding 模型缓存（增量索引用，避免每次上传都重新加载模型）
_embedding_model = None

DEFAULT_SOURCES_HINT = (
    "### 检索到的知识来源\n\n"
    "先提一个推荐问题。回答完成后，这里会显示命中的知识块、来源类型和相关证据。"
)

DEFAULT_PIPELINE_HINT = """
<div style="padding:12px; background:#f8fafc; border-radius:8px; text-align:center; color:#64748b; font-size:13px; line-height:1.8;">
  <div style="font-size:18px; margin-bottom:6px;">🧭 当前路径</div>
  <div style="font-weight:600;">⏳ 等待提问</div>
  <div>提交问题后，这里会展示当前数据集、检索模式和答案生成路径。</div>
</div>
"""

DEFAULT_GRAPH_HINT = (
    "默认建议先点“显示图谱概览 (Top 30)”。"
    "如果你刚完成构图，请优先查看概览，确认你的资料已进入图谱。"
)


def _get_embedding_model():
    """懒加载并缓存 SentenceTransformer 模型（增量索引用）"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _clean_triple_part(value) -> str:
    """清理三元组字段，避免破坏磁盘上的 `h | r | t` 格式。"""
    return str(value or "").replace("|", "").replace("\n", " ").strip()


def _normalize_triples(triples_data) -> list[tuple[str, str, str]]:
    """把 Gradio state 中的三元组统一成稳定的 `(head, relation, tail)` 元组列表。"""
    normalized = []
    for item in triples_data or []:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        head, relation, tail = (_clean_triple_part(part) for part in item)
        if head and relation and tail:
            normalized.append((head, relation, tail))
    return normalized


def _write_triples(path: str, triples: list[tuple[str, str, str]], append: bool) -> None:
    """把三元组写入用户数据文件。"""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for head, relation, tail in triples:
            f.write(f"{head} | {relation} | {tail}\n")


def _record_doc_triple_map(doc_id: str, triples: list[tuple[str, str, str]]) -> None:
    """为全量构建路径补一份 doc_id -> triple_keys 映射。"""
    if not doc_id or not triples:
        return

    from .incremental_index import load_doc_map, save_doc_map

    doc_map = load_doc_map(INDEX_DIR)
    existing = doc_map.get(doc_id, [])
    seen = set(existing)
    for head, relation, tail in triples:
        triple_key = f"{head}|{relation}|{tail}"
        if triple_key not in seen:
            existing.append(triple_key)
            seen.add(triple_key)
    doc_map[doc_id] = existing
    save_doc_map(INDEX_DIR, doc_map)


def get_rag(
    use_multihop=False,
    search_mode="vector",
    data_path=None,
) -> PocketGraphRAG:
    """获取或创建 RAG 实例（双重检查锁，线程安全）"""
    global rag
    if rag is None:
        with _rag_lock:
            if rag is None:  # double-check：进入锁后再确认一次，避免重复创建
                rag = PocketGraphRAG(
                    use_multihop=use_multihop,
                    search_mode=search_mode,
                    use_conversation=True,
                    data_path=data_path,
                )
    return rag


def reload_rag(use_multihop=False, search_mode="vector", data_path=None):
    """强制重新创建 RAG 实例（切换数据集或重建索引后调用，加锁避免并发竞争）"""
    global rag
    with _rag_lock:
        rag = PocketGraphRAG(
            use_multihop=use_multihop,
            search_mode=search_mode,
            use_conversation=True,
            data_path=data_path,
        )
    return rag


def _score_to_color(score: float) -> str:
    """相似度分数转颜色徽章"""
    if score >= 0.8:
        return "🟢"
    elif score >= 0.6:
        return "🟡"
    else:
        return "🔴"


def _source_type_badge(source_type: str) -> str:
    """来源类型标签，帮助用户判断证据来自 KG、向量还是社区摘要。"""
    badge_map = {
        "kg": (
            "KG",
            "#dbeafe",
            "#1e40af",
        ),
        "vector": (
            "向量",
            "#f3e8ff",
            "#7c3aed",
        ),
        "community_summary": (
            "社区摘要",
            "#dcfce7",
            "#166534",
        ),
    }
    label, bg, fg = badge_map.get(source_type or "vector", badge_map["vector"])
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 6px;"
        f"border-radius:4px;font-size:11px;'>{label}</span>"
    )


def _format_source_score_label(
    score: float, source_type: str, max_score: float | None = None
) -> str:
    """把内部排序分转成更适合用户理解的展示标签。"""
    if source_type == "kg":
        return "KG命中"
    if source_type == "community_summary":
        return "社区命中"

    if max_score and max_score > 0:
        pct = max(1.0, (float(score or 0.0) / max_score) * 100)
        return f"相对相关度 {pct:.0f}%"

    return "已命中"


def _source_matches_query(query: str, source: dict) -> bool:
    """判断来源是否与当前问题明显相关，用于展示层去噪。"""
    import re

    normalized_query = (query or "").strip().lower()
    if not normalized_query:
        return True

    haystacks = [
        str(source.get("entity", "")).lower(),
        str(source.get("text", "")).lower(),
    ]
    for haystack in haystacks:
        if not haystack:
            continue
        if haystack in normalized_query or normalized_query in haystack:
            return True

    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", normalized_query)
    for token in tokens:
        if any(token in haystack for haystack in haystacks):
            return True
    return False


def _partition_sources_for_display(sources: list, query: str = "") -> tuple[list, list]:
    """把来源分成核心证据与候选证据，降低首屏噪声但保留原始 citation 顺序。"""
    if not sources:
        return [], []

    max_score = max((float(src.get("score", 0) or 0.0) for src in sources), default=0.0)
    core_sources = []
    extra_sources = []

    for index, src in enumerate(sources, 1):
        src_copy = dict(src)
        src_copy["_display_index"] = index
        score = float(src.get("score", 0) or 0.0)
        source_type = src.get("source_type", "vector")
        is_core = (
            index <= 2
            or source_type != "vector"
            or score >= max_score * 0.97
            or _source_matches_query(query, src)
        )
        if is_core:
            core_sources.append(src_copy)
        else:
            extra_sources.append(src_copy)

    if not core_sources and extra_sources:
        core_sources.append(extra_sources.pop(0))
    return core_sources, extra_sources


def _render_source_cards(sources: list, max_score: float | None = None) -> str:
    """把来源列表渲染成可折叠卡片。"""
    output = ""
    for src in sources:
        display_index = src.get("_display_index", "?")
        score = src.get("score", 0)
        score_label = _format_source_score_label(
            score,
            src.get("source_type", ""),
            max_score=max_score,
        )
        color = _score_to_color(score)
        entity = src.get("entity", "未知实体")
        text = src.get("text", "")
        source_type = src.get("source_type", "")
        type_label = _source_type_badge(source_type)
        output += f"""
<details style="margin-bottom:10px; border:1px solid #e2e8f0; border-radius:8px; overflow:hidden;">
  <summary style="padding:10px 14px; cursor:pointer; background:#f8fafc; font-weight:500; font-size:13px;">
    {color} **[{display_index}] {entity}** {type_label} `<span style="color:#64748b;font-weight:normal;">{score_label}</span>`
  </summary>
  <div style="padding:12px 14px; font-size:13px; line-height:1.7; color:#334155; border-top:1px solid #e2e8f0;">
    {text}
  </div>
</details>
"""
    return output


def format_sources(sources: list, query: str = "") -> str:
    """格式化参考知识来源为美观的 Markdown"""
    if not sources:
        return DEFAULT_SOURCES_HINT

    output = "### 📚 检索到的知识来源\n\n"
    core_sources, extra_sources = _partition_sources_for_display(sources, query)
    max_score = max((float(src.get("score", 0) or 0.0) for src in sources), default=0.0)
    if query and extra_sources:
        output += (
            "<div style='margin-bottom:10px; padding:10px 12px; background:#f8fafc; "
            "border:1px solid #e2e8f0; border-radius:8px; font-size:12px; color:#475569;'>"
            "优先展示和当前问题更贴近的核心证据；其余候选来源保留在下方，便于核对 citation。"
            "</div>\n"
        )
    output += _render_source_cards(core_sources, max_score=max_score)
    if extra_sources:
        output += (
            "<details style='margin-top:8px;'>"
            "<summary style='cursor:pointer; color:#64748b; font-size:12px;'>"
            f"查看其余候选来源（{len(extra_sources)} 条）"
            "</summary>"
            "<div style='margin-top:10px;'>"
            f"{_render_source_cards(extra_sources, max_score=max_score)}"
            "</div>"
            "</details>"
        )
    return output


def get_recommended_questions(dataset_key: str) -> list[str]:
    """为当前数据集返回高成功率的推荐问题。"""
    if dataset_key == "example":
        return [
            "这部电影讲了什么？",
            "这部电影的主角是谁？",
            "这部电影是哪年上映的？",
            "这部电影的导演是谁？",
        ]

    return [
        "这个数据集里最重要的实体有哪些？",
        "请先概括这批知识里最关键的风险点。",
        "针对这个实体，已知的关系和结论有哪些？",
    ]


def load_recommended_questions_markdown(dataset_key: str | None = None) -> str:
    """将当前数据集的推荐问题渲染成 Markdown。"""
    dataset_key = dataset_key or current_dataset
    questions = get_recommended_questions(dataset_key)
    lines = ["### 推荐问题", ""]
    for q in questions:
        lines.append(f"- {q}")
    return "\n".join(lines)


def _dataset_label(dataset_key: str | None = None) -> str:
    """返回当前数据集的人类可读标签。"""
    dataset_key = dataset_key or current_dataset
    return "示例数据（电影知识图谱）" if dataset_key == "example" else "用户数据"


def _index_exists() -> bool:
    """判断当前索引是否已准备好。"""
    return os.path.exists(os.path.join(INDEX_DIR, "faiss.index"))


def _index_status_label(index_ready: bool | None = None) -> str:
    """返回首页状态卡使用的索引文案。"""
    if index_ready is None:
        index_ready = _index_exists()
    return "索引已就绪" if index_ready else "索引未就绪"


def load_runtime_status() -> str:
    """展示当前数据集、索引和模型状态。"""
    dataset_label = _dataset_label()
    data_path = _get_data_path()
    provider = detect_active_provider() or "未配置"
    provider_label = _PROVIDER_LABELS.get(provider, provider)
    index_label = _index_status_label()

    return f"""
<div style="padding:14px; border:1px solid #dbeafe; border-radius:10px; background:#f8fbff; font-size:13px; line-height:1.8;">
  <div><strong>当前数据集</strong>: {dataset_label}</div>
  <div><strong>数据文件</strong>: <code>{data_path}</code></div>
  <div><strong>LLM 后端</strong>: {provider_label}</div>
  <div><strong>索引状态</strong>: {index_label}</div>
</div>
"""


def _compact_kg_display_items(items: list[str]) -> list[str]:
    """过滤展示层里明显像中间特征/参数噪声的 KG 命中项。"""
    import re

    noise_keywords = (
        "程度",
        "区域",
        "指数",
        "评价",
        "适宜",
        "时期",
        "初期",
        "高发区",
        "表现",
        "用量",
        "药剂",
        "温度",
        "湿度",
        "亩",
        "毫升",
        "千克",
        "倍液",
    )
    compact = []
    for item in items or []:
        text = str(item or "").strip()
        if not text:
            continue
        if any(keyword in text for keyword in noise_keywords):
            continue
        if text.startswith(("感", "抗")) and "病" in text:
            continue
        if re.search(r"\d", text) and len(text) <= 12:
            continue
        compact.append(text)

    return list(dict.fromkeys(compact))


def format_pipeline_info(info: dict) -> str:
    """格式化 Pipeline 信息为美观的流程展示"""
    if not info:
        return DEFAULT_PIPELINE_HINT

    steps = []
    steps.append("👤 用户提问")

    if info.get("query_rewritten"):
        steps.append("✏️ 对话查询改写")

    search_mode = info.get("search_mode", "vector")
    search_mode_steps = {
        "vector": "📊 向量检索",
        "local": "🎯 KG 邻域检索",
        "global": "🌐 KG 关系检索",
        "mix": "🔀 混合检索",
        "kg_only": "🧭 纯 KG 检索",
        "global_summary": "🧠 社区摘要检索",
        "kg_entity": "🎯 KG 实体检索",
        "kg_ppr": "🌟 KG Pagerank 检索",
        "hybrid": "🔀 混合检索",
    }
    steps.append(search_mode_steps.get(search_mode, "📊 向量检索"))

    if info.get("multihop_used"):
        steps.append("🔗 Multi-hop 推理")

    # KG 检索详情
    kg_path = info.get("kg_path", {})
    details = []
    if kg_path and kg_path.get("search_type") != "vector":
        seed_entities = _compact_kg_display_items(kg_path.get("seed_entities", []))
        expanded_entities = kg_path.get("expanded_entities", [])
        matched_relations = kg_path.get("matched_relations", [])

        if seed_entities:
            entities_preview = "、".join([f"`{e}`" for e in seed_entities[:3]])
            if len(seed_entities) > 3:
                entities_preview += f" 等{len(seed_entities)}个"
            details.append(f"**匹配实体**: {entities_preview}")
        if expanded_entities:
            details.append(f"**邻域扩展**: {len(expanded_entities)} 个实体")
        if matched_relations:
            rels_preview = "、".join([f"`{r}`" for r in matched_relations[:3]])
            if len(matched_relations) > 3:
                rels_preview += f" 等{len(matched_relations)}个"
            details.append(f"**匹配关系**: {rels_preview}")

    # 检索参数信息
    if info.get("reranker_used"):
        details.append("**重排序**: 已启用 Reranker")
    if info.get("hyde_used"):
        details.append("**HyDE**: 已启用假设性文档改写")
    if info.get("query_routed"):
        details.append("**查询路由**: LLM 自动选检索模式")
    if info.get("self_check_used"):
        level = info.get("self_check_level", "skipped")
        level_label = {
            "full": "通过✅",
            "partial": "部分警告⚠️",
            "none": "完全编造🔴",
        }.get(level, level)
        details.append(f"**答案自检**: {level_label}")
    if "vector_weight" in info:
        details.append(f"**向量权重**: {info['vector_weight']:.1f}")
    top_k_used = info.get("top_k")
    if top_k_used:
        details.append(f"**Top-K**: {top_k_used}")

    response_mode = info.get("response_mode")
    if response_mode == "retrieval_fallback":
        details.append("**回答模式**: 检索兜底（LLM 未组织出稳定答案）")
    elif response_mode == "retrieval_only":
        details.append("**回答模式**: 纯检索输出")
    elif response_mode == "llm_standardized":
        details.append("**回答模式**: 结构化标准答案")

    question_type = info.get("question_type")
    if question_type:
        details.append(f"**问题类型**: {question_type}")

    failure_bucket_labels = {
        "empty_retrieval": "没有检索到任何结果",
        "no_entity_or_relation_hit": "没有命中实体或关系",
        "insufficient_relation_context": "关系命中了，但证据不足",
        "low_vector_similarity": "向量相似度太低",
        "insufficient_context": "证据不够完整",
    }
    failure_bucket = info.get("failure_bucket")
    if failure_bucket:
        details.append(
            f"**失败解释**: {failure_bucket_labels.get(failure_bucket, failure_bucket)}"
        )

    steps.append("🤖 LLM 生成答案")

    # 流程箭头
    flow = " → ".join(
        [
            f"<span style='background:#667eea;color:white;padding:3px 10px;border-radius:12px;font-size:12px;white-space:nowrap;'>{s}</span>"
            for s in steps
        ]
    )

    output = f"""
<div style="padding:12px; background:linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%); border-radius:8px; font-size:12px;">
  <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:10px;">
    {flow}
  </div>
"""
    if details:
        output += '<div style="margin-top:8px; padding-top:8px; border-top:1px dashed #cbd5e1; font-size:12px; color:#475569; line-height:1.8;">'
        output += "<br>".join(details)
        output += "</div>"
    output += "</div>"
    return output


def _get_data_path():
    """获取当前数据集对应的三元组文件路径"""
    if current_dataset == "user" and os.path.exists(USER_TRIPLES_PATH):
        return USER_TRIPLES_PATH
    return DATA_PATH


def chat(
    question: str,
    history: list,
    use_multihop,
    search_mode,
    top_k,
    use_reranker,
    vector_weight,
    use_hyde=False,
    use_query_router=False,
    use_self_check=False,
):
    """处理用户提问 (流式)"""
    if not question.strip():
        yield "", history, "", ""
        return

    system = get_rag(use_multihop, search_mode, data_path=_get_data_path())

    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})

    sources_md = "### 检索到的知识来源\n\n正在检索..."
    pipeline_md = "正在处理..."

    yield "", history, sources_md, pipeline_md

    for step in system.answer_stream(
        question,
        top_k=top_k,
        use_reranker=use_reranker,
        vector_weight=vector_weight,
        use_hyde=use_hyde,
        use_query_router=use_query_router,
        use_self_check=use_self_check,
    ):
        if "status" in step and "chunk" not in step and "self_check" not in step:
            # 更新状态
            if "sources" in step:
                effective_query = step.get("effective_query") or question
                sources_md = format_sources(step["sources"], query=effective_query)
                pipeline_md = format_pipeline_info(step.get("pipeline_info", {}))
                if step.get("effective_query") and step["effective_query"] != question:
                    pipeline_md += f"\n\n**改写后查询**: {step['effective_query']}"
            else:
                pipeline_md = step["status"]
            yield "", history, sources_md, pipeline_md

        elif "self_check" in step:
            # 答案自检结果
            verification = step["self_check"]
            level = verification.get("support_level", "full")
            unsupported = verification.get("unsupported_sentences", [])
            pipeline_md = format_pipeline_info(step.get("pipeline_info", {}))
            if level == "full":
                pipeline_md += "\n\n✅ **答案自检通过**：所有事实陈述均被知识库支持"
            elif level == "partial":
                pipeline_md += f"\n\n⚠️ **答案自检警告**：部分内容未被知识库支持（{len(unsupported)} 句）"
            else:  # none
                pipeline_md += (
                    "\n\n🔴 **答案自检警告**：回答未能从知识库找到充分依据，请谨慎参考"
                )
            yield "", history, sources_md, pipeline_md

        elif "chunk" in step:
            # 更新回答内容
            history[-1]["content"] = step["full_answer"]
            if step.get("pipeline_info"):
                pipeline_md = format_pipeline_info(step["pipeline_info"])
            yield "", history, sources_md, pipeline_md


def clear_conversation():
    """清空对话历史"""
    global rag
    if rag:
        rag.reset_conversation()
    return [], format_sources([]), format_pipeline_info({})


# ========================
# LLM 状态指示灯（问答 Tab 顶部）
# ========================

_PROVIDER_LABELS = {
    "ollama": "Ollama（本地）",
    "siliconflow": "SiliconFlow",
    "deepseek": "DeepSeek",
    "dashscope": "DashScope 通义千问",
    "openai": "OpenAI 兼容",
}

_PROVIDER_MODEL_FIELD = {
    "ollama": "OLLAMA_MODEL",
    "siliconflow": "SILICONFLOW_MODEL",
    "deepseek": "DEEPSEEK_MODEL",
    "dashscope": "DASHSCOPE_MODEL",
    "openai": "OPENAI_MODEL",
}


def load_llm_status() -> str:
    """生成问答 Tab 顶部的 LLM 状态指示灯 HTML。

    显示当前 provider、模型名、是否可用；用户无需切到设置 Tab 即可看到后端状态。
    """
    provider = detect_active_provider()
    if not provider:
        return (
            '<div style="padding:10px 14px; background:#fef3c7; border:1px solid #fbbf24; '
            'border-radius:8px; font-size:13px; color:#92400e;">'
            "🔴 <strong>LLM 未配置</strong> — 当前为纯检索模式，无法生成回答。"
            "请到「⚙️ LLM 设置」Tab 配置 Ollama / SiliconFlow / DeepSeek 等 provider。"
            "</div>"
        )

    settings = load_llm_settings()
    model_field = _PROVIDER_MODEL_FIELD.get(provider, "")
    model = settings.get(model_field, "") or "（未设置模型名）"
    provider_label = _PROVIDER_LABELS.get(provider, provider)

    # Ollama 额外检测服务是否在跑
    extra = ""
    if provider == "ollama":
        base = settings.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
        if is_ollama_running(base):
            extra = '<span style="color:#16a34a;"> ●服务在线</span>'
        else:
            extra = (
                '<span style="color:#dc2626;"> ●服务未启动（请打开 Ollama 应用）</span>'
            )

    return (
        f'<div style="padding:10px 14px; background:#dcfce7; border:1px solid #22c55e; '
        f'border-radius:8px; font-size:13px; color:#166534;">'
        f"🟢 <strong>LLM 已就绪</strong> — Provider: <strong>{provider_label}</strong>"
        f' ｜ 模型: <code style="background:#bbf7d0;padding:2px 6px;border-radius:4px;">{model}</code>'
        f"{extra}"
        f"</div>"
    )


# ========================
# 数据管理功能
# ========================


def handle_upload(file, image_mode="OCR 文字提取"):
    """处理文件上传，读取内容（支持 TXT/MD/PDF/图片）"""
    if file is None:
        return "请上传文件", "", ""
    try:
        from PocketGraphRAG.data_importer import DataImporter

        filepath = file if isinstance(file, str) else file.name

        # 图片模式转换
        img_mode = "ocr" if image_mode == "OCR 文字提取" else "kg"

        importer = DataImporter()
        doc = importer.import_file(filepath, image_mode=img_mode)

        if doc is None:
            return "不支持的文件类型或读取失败", "", ""

        if not doc.content.strip():
            return "文件内容为空（可能是扫描件或图片，需配置 VLM）", "", ""

        extra = ""
        if doc.source_type == "pdf":
            pages = doc.metadata.get("num_pages", "?")
            ocr_used = doc.metadata.get("ocr_used", False)
            extra = f"，共 {pages} 页"
            if ocr_used:
                extra += "（扫描版 OCR 识别）"
        elif doc.source_type == "image":
            mode = doc.metadata.get("mode", "ocr")
            mode_label = "OCR 文字提取" if mode == "ocr" else "知识抽取模式"
            extra = f"（{mode_label}）"

        # 归档到文档管理清单（E 模块：查看/删除）
        doc_id = ""
        try:
            doc_id = _archive_doc(
                name=os.path.basename(filepath),
                source_type=doc.source_type,
                source=doc.source,
                content=doc.content,
            )
        except Exception as e:
            logger.warning("文档归档失败（不影响主流程）: %s", e)

        return (
            f"已读取 **{doc.source_type.upper()}** 文件: **{doc.source}**"
            f"（{len(doc.content)} 字符{extra}）",
            doc.content,
            doc_id,
        )
    except Exception as e:
        return f"读取文件失败: {e}", "", ""


def handle_url_import(url, use_playwright=True):
    """从 URL 导入网页内容，支持 Playwright 动态渲染"""
    if not url or not url.strip():
        return "请输入网页 URL", "", ""
    try:
        from PocketGraphRAG.data_importer import DataImporter

        importer = DataImporter()
        doc = importer._import_url(url.strip(), use_playwright=use_playwright)

        if doc is None:
            return "网页导入失败，请检查 URL 是否正确", "", ""

        if not doc.content.strip():
            return "未能提取到有效内容", "", ""

        rendered_with = doc.metadata.get("rendered_with", "unknown")
        render_label = (
            "Playwright 动态渲染" if rendered_with == "playwright" else "静态抓取"
        )

        # 归档到文档管理清单（E 模块）
        doc_id = ""
        try:
            doc_id = _archive_doc(
                name=doc.title or "网页导入",
                source_type="webpage",
                source=url,
                content=doc.content,
            )
        except Exception as e:
            logger.warning("网页归档失败（不影响主流程）: %s", e)

        return (
            f"已导入网页: **{doc.title}**\n\n来源: {url}\n\n"
            f"提取内容: {len(doc.content)} 字符\n\n"
            f"渲染方式: {render_label}",
            doc.content,
            doc_id,
        )
    except Exception as e:
        return f"网页导入失败: {e}", "", ""


# ========================
# 已上传文档管理（E）
# ========================


def _load_docs_manifest() -> list:
    """读取已上传文档清单。每项: {id, name, type, chars, imported_at, source, content_file}"""
    if not os.path.exists(DOCS_MANIFEST_PATH):
        return []
    try:
        with open(DOCS_MANIFEST_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("docs", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _save_docs_manifest(docs: list) -> None:
    """写入文档清单"""
    with open(DOCS_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"docs": docs}, f, ensure_ascii=False, indent=2)


def _archive_doc(name: str, source_type: str, source: str, content: str) -> str:
    """归档一份已上传文档（原文 + manifest 登记）。

    用于 E 模块的查看/删除。同一文件名重复上传会另存一条记录。
    """
    import datetime
    import uuid

    doc_id = (
        "doc_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex[:6]
    )
    content_file = f"{doc_id}.txt"
    content_path = os.path.join(USER_DOCS_DIR, content_file)
    with open(content_path, "w", encoding="utf-8") as f:
        f.write(content)

    docs = _load_docs_manifest()
    docs.append(
        {
            "id": doc_id,
            "name": name,
            "type": source_type,
            "source": source,
            "chars": len(content),
            "imported_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content_file": content_file,
        }
    )
    _save_docs_manifest(docs)
    return doc_id


def handle_list_docs():
    """列出已上传文档，返回 DataFrame 数据 + 数量文案"""
    docs = _load_docs_manifest()
    if not docs:
        return (
            [],
            "📂 暂无已归档文档。上传文件或导入网页后，会自动记录到此处。",
        )
    rows = [
        [d["imported_at"], d["name"], d["type"], d["chars"]]
        for d in reversed(docs)  # 最新的排前面
    ]
    return rows, f"📂 共 **{len(docs)}** 份已归档文档（选中一行可查看/删除）"


def handle_view_doc(evt: gr.SelectData):
    """DataFrame 选中行时，显示对应文档内容。

    由于 reversed 显示，evt.index[0] 对应 reversed 列表的位置。
    """
    docs = _load_docs_manifest()
    if not docs:
        return "暂无文档可查看", gr.update()

    # reversed 后的列表
    rev_docs = list(reversed(docs))
    idx = evt.index[0]
    if idx < 0 or idx >= len(rev_docs):
        return "请选择有效的文档行", gr.update()

    doc = rev_docs[idx]
    content_path = os.path.join(USER_DOCS_DIR, doc["content_file"])
    try:
        with open(content_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"读取文档失败: {e}", gr.update()

    info = (
        f"### 📄 {doc['name']}\n\n"
        f"- 类型: {doc['type']}\n"
        f"- 来源: {doc.get('source', '-')}\n"
        f"- 字符数: {doc['chars']}\n"
        f"- 导入时间: {doc['imported_at']}\n"
        f"- 文档ID: `{doc['id']}`\n\n"
        f"---\n\n```\n{content}\n```"
    )
    return info, doc["id"]


def _delete_doc_graph_hint(selected_doc_id: str) -> str:
    """删除归档后，尽量同步删除用户图谱中的对应内容。"""
    if not os.path.exists(USER_TRIPLES_PATH) or not os.path.exists(
        os.path.join(INDEX_DIR, "faiss.index")
    ):
        return (
            "\n\n💡 提示：当前还没有可同步的用户知识图谱。"
            "如果后续基于该文档构建过图谱，请重新上传并构建。"
        )

    from .incremental_index import load_doc_map, remove_document_incremental

    doc_map = load_doc_map(INDEX_DIR)
    if selected_doc_id not in doc_map:
        return (
            "\n\n💡 提示：删除归档不会自动更新已构建的知识图谱。"
            "这份文档可能是在旧版本下构建的，或尚未绑定到图谱映射。"
        )

    try:
        global rag

        stats = remove_document_incremental(
            doc_id=selected_doc_id,
            model=_get_embedding_model(),
            index_dir=INDEX_DIR,
            data_path=USER_TRIPLES_PATH,
            reverse_link_relations=REVERSE_LINK_RELATIONS,
            relation_templates=RELATION_TEMPLATES,
        )
        rag = None
        return (
            "\n\n🧹 已同步更新知识图谱："
            f"删除 **{stats['removed_triples']}** 条三元组，"
            f"重建 **{stats['affected_entities']}** 个受影响实体，"
            f"清理 **{stats['orphan_entities_removed']}** 个孤儿实体。"
        )
    except Exception as e:
        logger.exception("同步删除文档对应知识图谱失败")
        return (
            "\n\n⚠️ 已删除归档，但同步更新知识图谱失败："
            f"{e}。当前图谱中可能仍保留该文档内容。"
        )


def handle_delete_doc(selected_doc_id):
    """删除选中的文档，并尽量同步更新用户知识图谱。"""
    if not selected_doc_id:
        return "请先在上方表格中选中一个文档", "", gr.update()

    docs = _load_docs_manifest()
    target = next((d for d in docs if d["id"] == selected_doc_id), None)
    if target is None:
        return "未找到该文档（可能已被删除）", "", gr.update()

    # 删除内容文件
    content_path = os.path.join(USER_DOCS_DIR, target["content_file"])
    try:
        if os.path.exists(content_path):
            os.remove(content_path)
    except Exception as e:
        return f"删除文件失败: {e}", "", gr.update()

    # 从 manifest 移除
    docs = [d for d in docs if d["id"] != selected_doc_id]
    _save_docs_manifest(docs)

    # 刷新列表
    if docs:
        rows = [
            [d["imported_at"], d["name"], d["type"], d["chars"]] for d in reversed(docs)
        ]
        summary = f"📂 共 **{len(docs)}** 份已归档文档\n\n✅ 已删除: {target['name']}"
    else:
        rows = []
        summary = f"✅ 已删除: {target['name']}（清单已空）"

    hint = _delete_doc_graph_hint(selected_doc_id)
    return summary + hint, "", rows


def _format_extract_preview(result) -> str:
    """格式化抽取结果为带质量统计的预览 Markdown"""
    total = len(result.triples)
    if total == 0:
        return "未能抽取到任何有效三元组，请检查文档内容或降低置信度阈值。"

    high_qual = result.high_quality_count
    avg_conf = result.avg_confidence

    preview = f"""### ✅ 抽取完成！

| 指标 | 数值 |
|------|------|
| 三元组总数 | **{total}** 条 |
| 高质量 (≥0.8) | {high_qual} 条 ({high_qual / total * 100:.1f}%) |
| 平均置信度 | {avg_conf:.3f} |
| 去重移除 | {result.removed_duplicates} 条 |
| 低质量过滤 | {result.removed_low_quality} 条 |
| 原始抽取 | {result.raw_triples_count} 条 |

---

### 三元组预览（前 30 条，按置信度排序）

| 置信度 | 头实体 | 关系 | 尾实体 |
|--------|--------|------|--------|
"""

    # 按置信度排序，取前 30 条
    sorted_triples = sorted(result.triples, key=lambda x: x.confidence, reverse=True)
    for t in sorted_triples[:30]:
        conf_bar = (
            "🟢" if t.confidence >= 0.9 else "🟡" if t.confidence >= 0.75 else "🟠"
        )
        preview += (
            f"| {conf_bar} {t.confidence:.2f} | {t.head} | {t.relation} | {t.tail} |\n"
        )

    if total > 30:
        preview += f"\n... 还有 {total - 30} 条未显示"

    return preview


def handle_extract(content):
    """从上传的文本内容中抽取三元组（流式，实时返回抽取进度）

    改为生成器后，UI 不再像卡死：每个步骤 yield 一次进度文案，
    最终 yield 完整预览与三元组列表。
    """
    global extracted_triples

    if not content or not content.strip():
        yield "请先上传文件", []
        return

    if not has_llm():
        yield (
            (
                "未配置 LLM API Key，无法抽取三元组。请到「⚙️ LLM 设置」Tab "
                "配置 Ollama / SiliconFlow / DeepSeek 等 provider。"
            ),
            [],
        )
        return

    # 流式抽取，逐步报进度
    for step in extract_knowledge_graph_stream(content):
        if step.get("done"):
            result = step.get("result")
            if result is None or not result.triples:
                # 空 / 失败
                extracted_triples = []
                yield step.get("message") or "未能抽取到任何有效三元组。", []
                return
            triples_list = [t.to_tuple() for t in result.triples]
            extracted_triples = triples_list
            yield _format_extract_preview(result), triples_list
            return
        else:
            # 进度更新：文案 + 当前累积的三元组（部分）
            yield step.get("message", "抽取中…"), step.get("triples", [])


def handle_build_index(triples_data, doc_id=""):
    """基于抽取的三元组构建/追加索引。

    增量优先策略（对标 LightRAG，让"上传→KG"在生产可用）：
      - 已在用户数据集 且 用户索引已存在 → 增量追加（只编码新/受影响实体，不重建全局）
      - 否则（首次构建 / 从示例切到用户）→ 全量构建用户索引

    旧实现每次上传都覆写 USER_TRIPLES_PATH + 全量重建，第二次上传会丢失第一次
    的数据且浪费算力。新实现 append 到主文件 + 增量更新三层 FAISS 索引。
    """
    global rag, current_dataset, extracted_triples

    triples = _normalize_triples(triples_data)
    extracted_triples = triples

    if not triples:
        dataset_label = "用户数据" if current_dataset == "user" else "示例数据（电影知识图谱）"
        return (
            "请先抽取三元组",
            current_dataset,
            dataset_label,
            f"当前: {dataset_label}",
            load_runtime_status(),
            load_recommended_questions_markdown(current_dataset),
        )

    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)

        on_user_dataset = current_dataset == "user"
        user_index_exists = os.path.exists(os.path.join(INDEX_DIR, "faiss.index"))
        user_triples_exists = os.path.exists(USER_TRIPLES_PATH)

        if on_user_dataset and user_index_exists:
            # ✅ 增量路径：已在用户数据集，索引已存在 → 只追加新三元组
            from .incremental_index import add_triples_incremental

            model = _get_embedding_model()
            stats = add_triples_incremental(
                triples,
                model,
                index_dir=INDEX_DIR,
                data_path=USER_TRIPLES_PATH,
                reverse_link_relations=REVERSE_LINK_RELATIONS,
                relation_templates=RELATION_TEMPLATES,
                doc_id=doc_id or None,
            )
            msg = (
                f"✅ **增量追加完成**（未重建全局索引，秒级完成）\n"
                f"- 新增三元组: **{stats['new_triples']}**\n"
                f"- 跳过重复: {stats['skipped_duplicates']}\n"
                f"- 新增实体: {stats['new_entities']}\n"
                f"- 受影响重建实体: {stats['affected_entities']}\n"
                f"- 新增关系: {stats['new_relations']}\n"
                f"- 当前总 chunk: {stats['total_chunks']}"
            )
        else:
            # 全量路径：首次构建 或 从示例数据切到用户数据
            # 把新三元组并入 USER_TRIPLES_PATH（已有则 append，无则新建）
            if user_triples_exists:
                _write_triples(USER_TRIPLES_PATH, triples, append=True)
            else:
                _write_triples(USER_TRIPLES_PATH, triples, append=False)

            from .build_index import build_index_with_data

            build_index_with_data(USER_TRIPLES_PATH)
            _record_doc_triple_map(doc_id, triples)
            msg = (
                f"✅ 索引构建完成！本次新增 {len(triples)} 条三元组，"
                f"已切换到用户数据集。\n\n"
                f"💡 后续在用户数据集下再次上传文档，将自动走**增量追加**，无需全量重建。"
            )

        current_dataset = "user"
        rag = None  # 强制重新加载

        return (
            msg,
            "user",
            "用户数据",
            "当前: 用户数据",
            load_runtime_status(),
            load_recommended_questions_markdown("user"),
        )
    except Exception as e:
        logger.exception("构建索引失败")
        dataset_label = "用户数据" if current_dataset == "user" else "示例数据（电影知识图谱）"
        return (
            f"构建索引失败: {e}",
            current_dataset,
            dataset_label,
            f"当前: {dataset_label}",
            load_runtime_status(),
            load_recommended_questions_markdown(current_dataset),
        )


def handle_reset_user_dataset():
    """清空用户数据集并切回示例数据（兜底/重置场景）。

    删除 USER_TRIPLES_PATH，重建示例数据索引，切回 example 数据集。
    用于：用户想从头开始、或索引异常需要重置。
    """
    global rag, current_dataset

    try:
        # 清空用户三元组文件
        if os.path.exists(USER_TRIPLES_PATH):
            os.remove(USER_TRIPLES_PATH)

        # 重建示例数据索引
        from .build_index import build_index_with_data

        build_index_with_data(DATA_PATH)

        current_dataset = "example"
        rag = None  # 强制重新加载

        return (
            "✅ 用户数据集已清空，已切回示例数据（电影知识图谱）。\n\n"
            "现在上传新文档将从零开始构建用户数据集。",
            "示例数据（电影知识图谱）",
            "当前: 示例数据（电影知识图谱）",
            load_runtime_status(),
            load_recommended_questions_markdown("example"),
        )
    except Exception as e:
        logger.exception("重置用户数据集失败")
        return (
            f"重置失败: {e}",
            "用户数据",
            "当前: 用户数据",
            load_runtime_status(),
            load_recommended_questions_markdown(current_dataset),
        )


def handle_switch_dataset(dataset_choice):
    """切换数据集"""
    global current_dataset, rag

    if dataset_choice == "示例数据（电影知识图谱）":
        current_dataset = "example"
    else:
        current_dataset = "user"

    rag = None  # 强制重新加载

    if current_dataset == "example":
        # 重建示例数据索引
        try:
            from .build_index import build_index_with_data

            build_index_with_data(DATA_PATH)
            return (
                "已切换到示例数据集（电影知识图谱），索引已重建",
                load_runtime_status(),
                load_recommended_questions_markdown("example"),
            )
        except Exception as e:
            return (
                f"切换到示例数据集，但索引重建失败: {e}",
                load_runtime_status(),
                load_recommended_questions_markdown("example"),
            )
    else:
        if os.path.exists(USER_TRIPLES_PATH):
            try:
                from .build_index import build_index_with_data

                build_index_with_data(USER_TRIPLES_PATH)
                return (
                    "已切换到用户数据集，索引已重建",
                    load_runtime_status(),
                    load_recommended_questions_markdown("user"),
                )
            except Exception as e:
                return (
                    f"切换到用户数据集，但索引重建失败: {e}",
                    load_runtime_status(),
                    load_recommended_questions_markdown("user"),
                )
        else:
            return (
                "用户数据集为空，请先上传文档并构建索引",
                load_runtime_status(),
                load_recommended_questions_markdown("user"),
            )


def handle_load_example():
    """一键加载示例数据（电影知识图谱），构建索引并切换。

    新用户首次进入时的零配置快速体验入口。
    """
    global current_dataset, rag
    try:
        from .build_index import build_index_with_data

        build_index_with_data(DATA_PATH)
        current_dataset = "example"
        rag = None  # 强制重新加载
        return (
            "✅ 示例数据已加载！共电影知识三元组，索引已构建。\n\n"
            "👉 现在去「问答」Tab 直接提问即可，例如：\n"
            "- 这部电影讲了什么？\n"
            "- 这部电影的主角是谁？\n",
            "示例数据（电影知识图谱）",
            "当前: 示例数据（电影知识图谱）",
            load_runtime_status(),
            load_recommended_questions_markdown("example"),
        )
    except Exception as e:
        return (
            f"❌ 加载示例数据失败: {e}",
            "示例数据（电影知识图谱）",
            "当前: 示例数据（电影知识图谱）",
            load_runtime_status(),
            load_recommended_questions_markdown("example"),
        )


# ========================
# 知识图谱可视化
# ========================


def get_kg_retriever():
    """获取 KG 检索器（确保 RAG 系统已加载，且 KG 检索器已初始化）"""
    system = get_rag(search_mode="mix", data_path=_get_data_path())
    if system.kg_retriever is None:
        system._load_kg_retriever()
    return system.kg_retriever


def _enrich_subgraph(subgraph: dict, seed_entities: list = None) -> dict:
    """为子图添加社区检测、Pagerank、置信度等元数据"""
    try:
        retriever = get_kg_retriever()

        # 1. 社区检测着色
        communities = retriever.detect_communities()
        node_ids = {n["id"] for n in subgraph["nodes"]}
        subgraph_communities = {n: communities.get(n, 0) for n in node_ids}
        unique_communities = sorted(set(subgraph_communities.values()))
        community_colors = [
            "#ff7f50",
            "#5470c6",
            "#91cc75",
            "#fac858",
            "#ee6666",
            "#73c0de",
            "#3ba272",
            "#fc8452",
            "#9a60b4",
            "#ea7ccc",
            "#48b8d0",
            "#ff9f7f",
            "#67e0e3",
            "#8378ea",
            "#a9d86e",
        ]
        community_color_map = {
            cid: community_colors[i % len(community_colors)]
            for i, cid in enumerate(unique_communities)
        }

        # 2. Pagerank 节点重要性
        pagerank = retriever.compute_pagerank()
        pr_values = [pagerank.get(n["id"], 0) for n in subgraph["nodes"]]
        pr_min = min(pr_values) if pr_values else 0
        pr_max = max(pr_values) if pr_values else 1
        pr_range = pr_max - pr_min if pr_max > pr_min else 1

        # 3. 关系类型到颜色的映射
        all_relations = sorted(
            set(l.get("relation", "") for l in subgraph["links"] if l.get("relation"))
        )
        relation_colors = [
            "#5470c6",
            "#91cc75",
            "#fac858",
            "#ee6666",
            "#73c0de",
            "#3ba272",
            "#fc8452",
            "#9a60b4",
            "#ea7ccc",
            "#48b8d0",
        ]
        relation_color_map = {
            rel: relation_colors[i % len(relation_colors)]
            for i, rel in enumerate(all_relations)
        }

        # 4. 增强节点数据
        seed_set = set(seed_entities) if seed_entities else set()
        for node in subgraph["nodes"]:
            cid = subgraph_communities.get(node["id"], 0)
            node["community_id"] = cid
            node["color"] = community_color_map[cid]
            pr = pagerank.get(node["id"], 0)
            norm_pr = (pr - pr_min) / pr_range
            node["pagerank"] = pr
            node["importance"] = norm_pr
            if node["id"] in seed_set:
                node["is_seed"] = True
                node["size"] = 35 + norm_pr * 25
            else:
                node["is_seed"] = False
                node["size"] = 20 + norm_pr * 20
            node["value"] = node.get("degree", 1)

        # 5. 增强边数据（带置信度信息）
        for link in subgraph["links"]:
            rel = link.get("relation", "")
            link["color"] = relation_color_map.get(rel, "#94a3b8")
            link["confidence"] = link.get("confidence", 0.9)

        subgraph["communities"] = [
            {"id": cid, "color": community_color_map[cid], "name": f"社区 {cid}"}
            for cid in unique_communities
        ]
        subgraph["relation_colors"] = [
            {"relation": rel, "color": relation_color_map[rel]} for rel in all_relations
        ]

    except Exception:
        # 如果增强失败（空图谱等），返回原始数据
        for node in subgraph.get("nodes", []):
            node.setdefault("community_id", 0)
            node.setdefault("color", "#5470c6")
            node.setdefault("pagerank", 0)
            node.setdefault("importance", 0.5)
            node.setdefault("size", 25)
            node.setdefault("is_seed", False)
            node.setdefault("value", 1)
        for link in subgraph.get("links", []):
            link.setdefault("color", "#94a3b8")
            link.setdefault("confidence", 0.9)
        subgraph.setdefault(
            "communities", [{"id": 0, "color": "#5470c6", "name": "社区 0"}]
        )
        subgraph.setdefault("relation_colors", [])

    return subgraph


def get_graph_overview():
    """获取图谱概览数据（Top 实体子图，带社区着色）"""
    try:
        retriever = get_kg_retriever()
        stats = retriever.get_graph_stats()
        top_entities = retriever.get_top_entities(30)
        subgraph = retriever.get_subgraph(top_entities, max_hops=0)
        subgraph = _enrich_subgraph(subgraph, seed_entities=top_entities[:10])

        stats_text = f"""
### 图谱统计

| 指标 | 数值 |
|------|------|
| 实体总数 | {stats["total_entities"]} |
| 关系类型 | {stats["total_relations"]} |
| 三元组总数 | {stats["total_edges"]} |
| 平均度数 | {stats["avg_degree"]} |
| 社区数量 | {len(subgraph.get("communities", []))} |

**当前展示**: Top 30 高频实体，按社区着色，节点大小 = Pagerank 重要性
"""
        return stats_text, json.dumps(subgraph, ensure_ascii=False)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"加载图谱失败: {e}", json.dumps(
            {"nodes": [], "links": [], "communities": [], "relation_colors": []},
            ensure_ascii=False,
        )


def search_graph_entity(entity_name: str):
    """搜索实体并展示其邻域子图（带社区着色）"""
    if not entity_name or not entity_name.strip():
        return "请输入实体名称", json.dumps(
            {"nodes": [], "links": [], "communities": [], "relation_colors": []},
            ensure_ascii=False,
        )

    try:
        retriever = get_kg_retriever()
        matched = retriever.match_entities(entity_name.strip(), top_k=5, threshold=0.3)

        if not matched:
            return f"未找到匹配的实体: {entity_name}", json.dumps(
                {"nodes": [], "links": [], "communities": [], "relation_colors": []},
                ensure_ascii=False,
            )

        seed_entity = matched[0]
        subgraph = retriever.get_subgraph([seed_entity], max_hops=1)

        if len(subgraph["nodes"]) > 100:
            subgraph["nodes"].sort(key=lambda x: x["degree"], reverse=True)
            keep_nodes = {n["id"] for n in subgraph["nodes"][:100]}
            subgraph["nodes"] = [n for n in subgraph["nodes"] if n["id"] in keep_nodes]
            subgraph["links"] = [
                l
                for l in subgraph["links"]
                if l["source"] in keep_nodes and l["target"] in keep_nodes
            ]

        subgraph = _enrich_subgraph(subgraph, seed_entities=[seed_entity])

        result_text = f"""
### 搜索结果

**最匹配实体**: {seed_entity}
**匹配到的实体**: {", ".join(matched[:5])}
**子图节点数**: {len(subgraph["nodes"])}
**子图边数**: {len(subgraph["links"])}
**社区数量**: {len(subgraph.get("communities", []))}

🎯 橙色/高亮 = 中心实体 | 🎨 不同颜色 = 不同社区 | 📏 节点大小 = Pagerank 重要性
"""
        return result_text, json.dumps(subgraph, ensure_ascii=False)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"搜索失败: {e}", json.dumps(
            {"nodes": [], "links": [], "communities": [], "relation_colors": []},
            ensure_ascii=False,
        )


def build_graph_html(graph_data_json: str) -> str:
    """构建增强版 ECharts 知识图谱可视化 - 实体详情面板/社区着色/Pagerank 大小"""
    try:
        data = json.loads(graph_data_json)
    except Exception as e:
        logger.warning("图谱数据 JSON 解析失败，使用空数据: %s", e)
        data = {"nodes": [], "links": [], "communities": [], "relation_colors": []}

    graph_json = json.dumps(data, ensure_ascii=False)
    communities_json = json.dumps(data.get("communities", []), ensure_ascii=False)
    relation_colors_json = json.dumps(
        data.get("relation_colors", []), ensure_ascii=False
    )

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            overflow: hidden;
        }}
        .app-container {{
            display: flex;
            height: 100vh;
            padding: 10px;
            gap: 10px;
        }}
        .main-area {{
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }}
        .control-panel {{
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 10px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .control-group label {{
            font-size: 11px;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .control-group input[type="text"],
        .control-group select {{
            padding: 7px 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 13px;
            outline: none;
            transition: all 0.2s;
            background: white;
        }}
        .control-group input[type="text"]:focus,
        .control-group select:focus {{
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        .control-group input[type="range"] {{
            width: 100px;
            accent-color: #667eea;
        }}
        .control-group .value-display {{
            font-size: 11px;
            color: #94a3b8;
            text-align: right;
        }}
        .btn {{
            padding: 7px 16px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }}
        .btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        .btn.secondary {{
            background: white;
            color: #475569;
            border: 2px solid #e2e8f0;
            box-shadow: none;
        }}
        .btn.secondary:hover {{
            background: #f8fafc;
            border-color: #cbd5e1;
        }}
        .stats-bar {{
            display: flex;
            gap: 16px;
            padding: 10px 16px;
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            font-size: 12px;
            color: #475569;
            margin-bottom: 10px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        }}
        .stat-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .stat-value {{
            font-weight: 700;
            color: #667eea;
            font-size: 14px;
        }}
        #graph {{
            flex: 1;
            background: rgba(255,255,255,0.98);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            min-height: 400px;
        }}
        .detail-panel {{
            width: 320px;
            background: rgba(255,255,255,0.98);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transition: all 0.3s;
        }}
        .detail-panel.collapsed {{
            width: 48px;
        }}
        .detail-header {{
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .detail-header h3 {{
            margin: 0;
            font-size: 14px;
            font-weight: 600;
        }}
        .detail-toggle {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .detail-body {{
            flex: 1;
            overflow-y: auto;
            padding: 16px;
        }}
        .detail-panel.collapsed .detail-body {{
            display: none;
        }}
        .detail-empty {{
            text-align: center;
            color: #94a3b8;
            padding: 40px 20px;
            font-size: 13px;
        }}
        .detail-empty .hint {{
            font-size: 12px;
            margin-top: 8px;
            opacity: 0.7;
        }}
        .entity-name {{
            font-size: 18px;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 12px;
            word-break: break-all;
        }}
        .entity-meta {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 16px;
        }}
        .meta-card {{
            background: #f8fafc;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }}
        .meta-label {{
            font-size: 10px;
            color: #94a3b8;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .meta-value {{
            font-size: 16px;
            font-weight: 700;
            color: #667eea;
        }}
        .triples-section {{
            margin-top: 16px;
        }}
        .section-title {{
            font-size: 12px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .triple-item {{
            background: #f8fafc;
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-left: 3px solid #667eea;
            transition: all 0.2s;
            cursor: pointer;
        }}
        .triple-item:hover {{
            background: #f1f5f9;
            transform: translateX(2px);
        }}
        .triple-relation {{
            font-size: 11px;
            color: #667eea;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .triple-target {{
            font-size: 13px;
            color: #334155;
        }}
        .confidence-bar {{
            height: 3px;
            background: #e2e8f0;
            border-radius: 2px;
            margin-top: 6px;
            overflow: hidden;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981, #34d399);
            border-radius: 2px;
        }}
        .legend-section {{
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #e2e8f0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 0;
            font-size: 12px;
            color: #475569;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        .tooltip {{
            max-width: 320px;
            white-space: normal !important;
            word-break: break-all;
        }}
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}
    </style>
</head>
<body>
    <div class="app-container">
        <div class="main-area">
            <div class="control-panel">
                <div class="control-group" style="flex:1; min-width: 160px;">
                    <label>🔍 搜索实体</label>
                    <input type="text" id="searchInput" placeholder="输入实体名..." oninput="searchNode()">
                </div>
                <div class="control-group">
                    <label>📍 布局</label>
                    <select id="layoutSelect" onchange="changeLayout()">
                        <option value="force">力导向</option>
                        <option value="circular">圆形</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>📐 缩放</label>
                    <input type="range" id="sizeSlider" min="15" max="50" value="28">
                    <div class="value-display"><span id="sizeValue">28</span></div>
                </div>
                <button class="btn secondary" onclick="resetView()">🔄 重置</button>
            </div>
            <div class="stats-bar">
                <div class="stat-item">
                    <span>📊 节点</span>
                    <span class="stat-value" id="nodeCount">0</span>
                </div>
                <div class="stat-item">
                    <span>🔗 关系</span>
                    <span class="stat-value" id="linkCount">0</span>
                </div>
                <div class="stat-item">
                    <span>🏘️ 社区</span>
                    <span class="stat-value" id="communityCount">0</span>
                </div>
                <div class="stat-item" id="searchInfo" style="display:none;">
                    <span>🎯 匹配</span>
                    <span class="stat-value" id="searchResultCount">0</span>
                </div>
            </div>
            <div id="graph"></div>
        </div>
        <div class="detail-panel" id="detailPanel">
            <div class="detail-header">
                <h3>📋 实体详情</h3>
                <button class="detail-toggle" onclick="togglePanel()">◀</button>
            </div>
            <div class="detail-body" id="detailBody">
                <div class="detail-empty">
                    <div style="font-size:48px; margin-bottom:12px;">👆</div>
                    <div>点击任意节点查看详情</div>
                    <div class="hint">显示关联三元组、度数、Pagerank 重要性等</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        var chart = echarts.init(document.getElementById('graph'));
        var allNodes = [];
        var allLinks = [];
        var currentLayout = 'force';
        var baseNodeSize = 28;
        var selectedNodeId = null;
        var panelCollapsed = false;

        var graphData = {graph_json};
        var communities = {communities_json};
        var relationColors = {relation_colors_json};

        // 构建节点索引
        var nodeMap = {{}};
        var linkMap = {{}};

        // 初始化节点数据
        allNodes = graphData.nodes.map(function(node) {{
            var n = {{
                id: node.id,
                name: node.name,
                symbolSize: node.size || 28,
                degree: node.degree || 1,
                community_id: node.community_id || 0,
                color: node.color || '#5470c6',
                pagerank: node.pagerank || 0,
                importance: node.importance || 0.5,
                is_seed: node.is_seed || false,
                itemStyle: {{
                    color: node.color || '#5470c6',
                    borderColor: node.is_seed ? '#fff' : 'rgba(255,255,255,0.6)',
                    borderWidth: node.is_seed ? 3 : 2,
                    shadowBlur: node.is_seed ? 20 : 10,
                    shadowColor: node.color || '#5470c6',
                    shadowOffsetY: 4
                }},
                outgoing: [],
                incoming: []
            }};
            nodeMap[node.id] = n;
            return n;
        }});

        // 初始化边数据，并收集每个节点的出入边
        allLinks = graphData.links.map(function(link, idx) {{
            var l = {{
                id: 'link_' + idx,
                source: link.source,
                target: link.target,
                relation: link.relation || '',
                confidence: link.confidence || 0.9,
                color: link.color || '#94a3b8',
                lineStyle: {{
                    color: link.color || '#94a3b8',
                    curveness: 0.15,
                    width: 1.5,
                    opacity: 0.5
                }}
            }};
            linkMap[l.id] = l;
            if (nodeMap[link.source]) {{
                nodeMap[link.source].outgoing.push(l);
            }}
            if (nodeMap[link.target]) {{
                nodeMap[link.target].incoming.push(l);
            }}
            return l;
        }});

        // 更新统计
        function updateStats() {{
            document.getElementById('nodeCount').textContent = allNodes.length;
            document.getElementById('linkCount').textContent = allLinks.length;
            document.getElementById('communityCount').textContent = communities.length;
        }}

        // 显示实体详情
        function showEntityDetail(nodeId) {{
            selectedNodeId = nodeId;
            var node = nodeMap[nodeId];
            if (!node) return;

            var body = document.getElementById('detailBody');
            var outgoing = node.outgoing || [];
            var incoming = node.incoming || [];

            var html = '<div class="entity-name">' + node.name + '</div>';
            html += '<div class="entity-meta">';
            html += '<div class="meta-card"><div class="meta-label">度数</div><div class="meta-value">' + node.degree + '</div></div>';
            html += '<div class="meta-card"><div class="meta-label">Pagerank</div><div class="meta-value">' + (node.pagerank * 100).toFixed(1) + '%</div></div>';
            html += '<div class="meta-card"><div class="meta-label">社区</div><div class="meta-value" style="color:' + node.color + '">' + node.community_id + '</div></div>';
            html += '<div class="meta-card"><div class="meta-label">关联</div><div class="meta-value">' + (outgoing.length + incoming.length) + '</div></div>';
            html += '</div>';

            if (outgoing.length > 0) {{
                html += '<div class="triples-section"><div class="section-title">🔗 出边关系 (' + outgoing.length + ')</div>';
                outgoing.forEach(function(link) {{
                    var target = nodeMap[link.target];
                    var targetName = target ? target.name : link.target;
                    var confPct = Math.round(link.confidence * 100);
                    html += '<div class="triple-item" onclick="focusNode(\\'' + link.target.replace(/'/g, "\\\\'") + '\\')">';
                    html += '<div class="triple-relation">→ ' + link.relation + '</div>';
                    html += '<div class="triple-target">' + targetName + '</div>';
                    html += '<div class="confidence-bar"><div class="confidence-fill" style="width:' + confPct + '%;opacity:' + (0.5 + link.confidence * 0.5) + '"></div></div>';
                    html += '</div>';
                }});
                html += '</div>';
            }}

            if (incoming.length > 0) {{
                html += '<div class="triples-section"><div class="section-title">⬅️ 入边关系 (' + incoming.length + ')</div>';
                incoming.forEach(function(link) {{
                    var source = nodeMap[link.source];
                    var sourceName = source ? source.name : link.source;
                    var confPct = Math.round(link.confidence * 100);
                    html += '<div class="triple-item" onclick="focusNode(\\'' + link.source.replace(/'/g, "\\\\'") + '\\')" style="border-left-color:#94a3b8">';
                    html += '<div class="triple-relation" style="color:#64748b">← ' + link.relation + '</div>';
                    html += '<div class="triple-target">' + sourceName + '</div>';
                    html += '<div class="confidence-bar"><div class="confidence-fill" style="width:' + confPct + '%;opacity:' + (0.5 + link.confidence * 0.5) + '"></div></div>';
                    html += '</div>';
                }});
                html += '</div>';
            }}

            if (communities.length > 0) {{
                html += '<div class="legend-section"><div class="section-title">🏘️ 社区图例</div>';
                communities.forEach(function(c) {{
                    html += '<div class="legend-item"><div class="legend-dot" style="background:' + c.color + '"></div>' + c.name + '</div>';
                }});
                html += '</div>';
            }}

            body.innerHTML = html;

            // 更新节点选中状态
            allNodes.forEach(function(n) {{
                n.itemStyle.borderWidth = n.is_seed ? 3 : 2;
                n.itemStyle.shadowBlur = n.is_seed ? 20 : 10;
            }});
            node.itemStyle.borderWidth = 4;
            node.itemStyle.shadowBlur = 30;
            node.itemStyle.borderColor = '#1e293b';
            render();
        }}

        function focusNode(nodeId) {{
            chart.dispatchAction({{
                type: 'focusNodeAdjacency',
                seriesIndex: 0,
                dataIndex: allNodes.findIndex(function(n) {{ return n.id === nodeId; }})
            }});
            showEntityDetail(nodeId);
        }}

        function togglePanel() {{
            panelCollapsed = !panelCollapsed;
            var panel = document.getElementById('detailPanel');
            var btn = panel.querySelector('.detail-toggle');
            if (panelCollapsed) {{
                panel.classList.add('collapsed');
                btn.textContent = '▶';
            }} else {{
                panel.classList.remove('collapsed');
                btn.textContent = '◀';
            }}
            setTimeout(function() {{ chart.resize(); }}, 300);
        }}

        function getOption(layout) {{
            return {{
                backgroundColor: 'transparent',
                tooltip: {{
                    backgroundColor: 'rgba(30, 41, 59, 0.95)',
                    borderColor: 'transparent',
                    textStyle: {{ color: '#fff' }},
                    formatter: function(params) {{
                        if (params.dataType === 'node') {{
                            var n = nodeMap[params.data.id];
                            var pr = (n.pagerank * 100).toFixed(1);
                            return '<div class="tooltip"><b style="font-size:14px;">' + params.name + '</b><br/>' +
                                   '<span style="opacity:0.8">度数: ' + params.data.degree + ' | Pagerank: ' + pr + '% | 社区: ' + n.community_id + '</span><br/>' +
                                   '<span style="opacity:0.6;font-size:11px">点击查看详情 →</span></div>';
                        }} else if (params.dataType === 'edge') {{
                            var conf = Math.round(params.data.confidence * 100);
                            return '<div class="tooltip"><b>' + params.data.relation + '</b><br/>' +
                                   params.data.source + ' → ' + params.data.target + '<br/>' +
                                   '<span style="opacity:0.7">置信度: ' + conf + '%</span></div>';
                        }}
                        return '';
                    }}
                }},
                series: [{{
                    type: 'graph',
                    layout: layout,
                    animation: true,
                    animationDuration: 1000,
                    animationEasing: 'elasticOut',
                    label: {{
                        show: true,
                        position: 'right',
                        formatter: '{{b}}',
                        fontSize: 11,
                        color: '#334155',
                        fontWeight: 500,
                        backgroundColor: 'rgba(255,255,255,0.8)',
                        padding: [2, 6],
                        borderRadius: 4
                    }},
                    labelLayout: {{
                        hideOverlap: true
                    }},
                    draggable: true,
                    roam: true,
                    focusNodeAdjacency: true,
                    data: allNodes,
                    force: layout === 'force' ? {{
                        repulsion: 500,
                        gravity: 0.05,
                        edgeLength: [100, 250],
                        layoutAnimation: true,
                        friction: 0.6
                    }} : undefined,
                    circular: layout === 'circular' ? {{
                        rotateLabel: true
                    }} : undefined,
                    lineStyle: {{
                        curveness: 0.15,
                        width: 1.5,
                        opacity: 0.5
                    }},
                    emphasis: {{
                        scale: true,
                        focus: 'adjacency',
                        lineStyle: {{
                            width: 3,
                            opacity: 0.9
                        }},
                        label: {{
                            fontSize: 13,
                            fontWeight: 'bold',
                            backgroundColor: '#fff'
                        }}
                    }},
                    edges: allLinks
                }}]
            }};
        }}

        function render() {{
            chart.setOption(getOption(currentLayout), true);
        }}

        // 搜索节点
        function searchNode() {{
            var query = document.getElementById('searchInput').value.trim().toLowerCase();
            var searchInfo = document.getElementById('searchInfo');

            allNodes.forEach(function(node) {{
                node.itemStyle.opacity = 1;
                node.itemStyle.borderWidth = node.is_seed ? 3 : 2;
            }});
            allLinks.forEach(function(link) {{
                link.lineStyle.opacity = 0.5;
                link.lineStyle.width = 1.5;
            }});

            if (query) {{
                var matchedNodes = allNodes.filter(function(node) {{
                    return node.name.toLowerCase().indexOf(query) !== -1;
                }});
                var matchedIds = new Set(matchedNodes.map(function(n) {{ return n.id; }}));

                allNodes.forEach(function(node) {{
                    if (!matchedIds.has(node.id)) {{
                        node.itemStyle.opacity = 0.2;
                    }} else {{
                        node.itemStyle.opacity = 1;
                        node.itemStyle.borderWidth = 4;
                        node.itemStyle.borderColor = '#f59e0b';
                        node.itemStyle.shadowBlur = 25;
                        node.itemStyle.shadowColor = '#f59e0b';
                    }}
                }});

                allLinks.forEach(function(link) {{
                    if (matchedIds.has(link.source) || matchedIds.has(link.target)) {{
                        link.lineStyle.opacity = 1;
                        link.lineStyle.width = 2.5;
                    }} else {{
                        link.lineStyle.opacity = 0.1;
                    }}
                }});

                searchInfo.style.display = 'flex';
                document.getElementById('searchResultCount').textContent = matchedNodes.length;
            }} else {{
                searchInfo.style.display = 'none';
                // 恢复选中状态
                if (selectedNodeId && nodeMap[selectedNodeId]) {{
                    nodeMap[selectedNodeId].itemStyle.borderWidth = 4;
                    nodeMap[selectedNodeId].itemStyle.borderColor = '#1e293b';
                }}
            }}

            render();
        }}

        function changeLayout() {{
            currentLayout = document.getElementById('layoutSelect').value;
            render();
        }}

        function resetView() {{
            document.getElementById('searchInput').value = '';
            document.getElementById('layoutSelect').value = 'force';
            currentLayout = 'force';
            selectedNodeId = null;
            document.getElementById('searchInfo').style.display = 'none';
            document.getElementById('detailBody').innerHTML = '<div class="detail-empty"><div style="font-size:48px;margin-bottom:12px;">👆</div><div>点击任意节点查看详情</div><div class="hint">显示关联三元组、度数、Pagerank 重要性等</div></div>';

            allNodes.forEach(function(node) {{
                node.itemStyle.opacity = 1;
                node.itemStyle.borderColor = node.is_seed ? '#fff' : 'rgba(255,255,255,0.6)';
                node.itemStyle.borderWidth = node.is_seed ? 3 : 2;
                node.itemStyle.shadowBlur = node.is_seed ? 20 : 10;
                node.itemStyle.shadowColor = node.color;
            }});
            allLinks.forEach(function(link) {{
                link.lineStyle.opacity = 0.5;
                link.lineStyle.width = 1.5;
            }});

            render();
        }}

        // 点击节点事件
        chart.on('click', function(params) {{
            if (params.dataType === 'node') {{
                showEntityDetail(params.data.id);
            }}
        }});

        // 节点大小缩放
        document.getElementById('sizeSlider').addEventListener('input', function(e) {{
            var scale = e.target.value / 28;
            document.getElementById('sizeValue').textContent = e.target.value;
            allNodes.forEach(function(node) {{
                node.symbolSize = (node.is_seed ? 35 : 25) * scale;
            }});
            render();
        }});

        // 初始化
        updateStats();
        render();

        // 响应式
        window.addEventListener('resize', function() {{ chart.resize(); }});
    </script>
</body>
</html>
"""
    import base64

    html_bytes = html_content.encode("utf-8")
    html_b64 = base64.b64encode(html_bytes).decode("ascii")
    iframe_src = f"data:text/html;charset=utf-8;base64,{html_b64}"
    return f'<iframe src="{iframe_src}" width="100%" height="780px" frameborder="0" style="border-radius:12px;border:none;"></iframe>'


# ==========================
# LLM 设置 Tab 回调函数
# ==========================


def _load_settings_ui():
    """加载当前 LLM 配置，返回各控件初值"""
    s = load_llm_settings()
    provider = s.get("provider", "") or ""
    return (
        provider,
        s.get("OLLAMA_MODEL", ""),
        s.get("OLLAMA_API_BASE", "http://localhost:11434/v1"),
        s.get("SILICONFLOW_API_KEY", ""),
        s.get("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        s.get("DEEPSEEK_API_KEY", ""),
        s.get("DEEPSEEK_MODEL", "deepseek-chat"),
        s.get("DASHSCOPE_API_KEY", ""),
        s.get("DASHSCOPE_MODEL", "qwen-plus"),
        s.get("OPENAI_API_KEY", ""),
        s.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        s.get("OPENAI_MODEL", "gpt-3.5-turbo"),
        f"**当前生效后端**: {get_active_provider()}",
    )


def _save_settings_ui(
    provider,
    ollama_model,
    ollama_base,
    sf_key,
    sf_model,
    ds_key,
    ds_model,
    dq_key,
    dq_model,
    oa_key,
    oa_base,
    oa_model,
):
    """保存 LLM 配置（同时刷新问答 Tab 顶部的状态灯）"""
    if not provider:
        return (
            "⚠️ 请先选择一个 LLM 后端",
            f"**当前生效后端**: {get_active_provider()}",
            load_llm_status(),
        )

    values = {
        "provider": provider,
        "OLLAMA_MODEL": ollama_model or "",
        "OLLAMA_API_BASE": ollama_base or "http://localhost:11434/v1",
        "SILICONFLOW_API_KEY": sf_key or "",
        "SILICONFLOW_MODEL": sf_model or "",
        "DEEPSEEK_API_KEY": ds_key or "",
        "DEEPSEEK_MODEL": ds_model or "",
        "DASHSCOPE_API_KEY": dq_key or "",
        "DASHSCOPE_MODEL": dq_model or "",
        "OPENAI_API_KEY": oa_key or "",
        "OPENAI_API_BASE": oa_base or "https://api.openai.com/v1",
        "OPENAI_MODEL": oa_model or "",
    }
    try:
        save_llm_settings(values)
        active = get_active_provider()
        return (
            f"✅ 配置已保存到 .env 并已热生效！\n当前后端: {active}\n下次问答将使用新配置。",
            f"**当前生效后端**: {active}",
            load_llm_status(),
        )
    except Exception as e:
        return (
            f"❌ 保存失败: {e}",
            f"**当前生效后端**: {get_active_provider()}",
            load_llm_status(),
        )


def _test_connection():
    """测试当前 LLM 连接"""
    ok, msg = test_llm_connection()
    return msg


def _list_ollama(ollama_base):
    """列出本地 Ollama 模型"""
    ok, models, msg = list_ollama_models(ollama_base)
    if not ok:
        return msg, gr.update()
    if not models:
        return (
            "Ollama 已连接，但尚未下载任何模型。请运行: ollama pull qwen2.5:7b",
            gr.update(),
        )
    model_list = "\n".join(f"• {m}" for m in models)
    return f"已下载模型:\n{model_list}", gr.update(choices=models)


def build_ui():
    """构建 Gradio 界面"""
    with gr.Blocks(title="PocketGraphRAG 问答系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # PocketGraphRAG 问答系统

            先跑通内置电影知识图谱示例数据，再导入你自己的文档。
            这样最容易判断当前问题是数据问题、检索问题，还是模型问题。
            """
        )

        with gr.Tabs():
            # ========================
            # Tab 1: 问答
            # ========================
            with gr.TabItem("问答"):
                gr.Markdown(
                    """
                    **先跑通内置示例，再导入你自己的资料。**
                    默认路径会先证明系统能答、能给来源、能承接到图谱，再进入自由探索。
                    """
                )
                # LLM 状态指示灯（顶部，无需切到设置 Tab 即可看到后端状态）
                with gr.Row():
                    runtime_status_md = gr.Markdown(
                        value=load_runtime_status(), show_label=False
                    )
                    llm_status_md = gr.Markdown(
                        value=load_llm_status(), show_label=False
                    )
                    refresh_status_btn = gr.Button("🔄 刷新", size="sm", scale=0)

                # 高级特性开关
                with gr.Row():
                    use_multihop = gr.Checkbox(
                        label="Multi-hop 多跳",
                        value=False,
                        info="分解复杂问题为多个子查询",
                    )
                    search_mode = gr.Dropdown(
                        label="检索模式",
                        choices=[
                            "vector",
                            "local",
                            "global",
                            "mix",
                            "kg_only",
                            "global_summary",
                        ],
                        value="mix",
                        info="默认推荐 mix：优先展示图谱 + 向量结合能力。vector=纯向量 | local=实体邻域 | global=关系匹配 | mix=融合 | kg_only=纯KG | global_summary=社区摘要(归纳问答)",
                    )

                # 检索参数可调
                with gr.Accordion("🎛️ 检索参数（高级）", open=False):
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            label="Top-K 检索数量",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=5,
                            info="召回多少条知识块送给 LLM。越大越全但越慢",
                        )
                        use_reranker_cb = gr.Checkbox(
                            label="启用 Reranker 重排序",
                            value=False,
                            info="用 CrossEncoder 精排结果（首次启用需联网下载模型，失败则自动回退关键词重排）",
                        )
                        vector_weight_slider = gr.Slider(
                            label="向量权重（混合检索时）",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.5,
                            info="0.5=均衡 | >0.5 偏向语义相似 | <0.5 偏向知识图谱精确匹配。仅 mix/local/global 模式生效",
                        )
                        use_hyde_cb = gr.Checkbox(
                            label="启用 HyDE 假设性文档改写",
                            value=False,
                            info="用 LLM 先生成假设性答案再做向量检索，提升短问题召回（需 LLM；仅 vector/mix 生效，与多跳互斥）",
                        )
                        use_query_router_cb = gr.Checkbox(
                            label="启用查询路由（自动选检索模式）",
                            value=False,
                            info="LLM 自动判断问题类型选 vector/local/global/mix/global_summary，覆盖上方手动选择（需 LLM）",
                        )
                        use_self_check_cb = gr.Checkbox(
                            label="启用答案自检（降幻觉）",
                            value=False,
                            info="LLM 生成后校验答案是否被知识库支持，发现编造内容加⚠️警告，完全编造时强化约束重试1次（需 LLM）",
                        )

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown(
                            """
                            **建议动作**
                            1. 先点一个推荐问题
                            2. 看答案和来源是否同时出现
                            3. 再去「知识图谱」确认命中实体和图结构
                            """
                        )
                        recommended_questions_md = gr.Markdown(
                            value=load_recommended_questions_markdown(),
                            show_label=False,
                        )
                        chatbot = gr.Chatbot(
                            label="对话",
                            height=500,
                            type="messages",
                        )
                        with gr.Row():
                            question = gr.Textbox(
                                label="输入问题",
                                placeholder="例如：这部电影讲了什么？主角是谁？",
                                scale=4,
                            )
                            submit_btn = gr.Button("提交", variant="primary", scale=1)
                            clear_btn = gr.Button("清空对话", scale=1)

                        gr.Examples(
                            examples=get_recommended_questions("example"),
                            inputs=question,
                            label="示例问题",
                        )

                    with gr.Column(scale=2):
                        pipeline_display = gr.Markdown(
                            value=DEFAULT_PIPELINE_HINT,
                            label="Pipeline 信息",
                        )
                        sources_display = gr.Markdown(
                            value=DEFAULT_SOURCES_HINT,
                            label="参考来源",
                        )

                # 绑定事件
                chat_inputs = [
                    question,
                    chatbot,
                    use_multihop,
                    search_mode,
                    top_k_slider,
                    use_reranker_cb,
                    vector_weight_slider,
                    use_hyde_cb,
                    use_query_router_cb,
                    use_self_check_cb,
                ]
                chat_outputs = [question, chatbot, sources_display, pipeline_display]

                submit_btn.click(chat, inputs=chat_inputs, outputs=chat_outputs)
                question.submit(chat, inputs=chat_inputs, outputs=chat_outputs)
                clear_btn.click(
                    clear_conversation,
                    outputs=[chatbot, sources_display, pipeline_display],
                )

            # ========================
            # Tab 2: 数据管理
            # ========================
            with gr.TabItem("数据管理"):
                gr.Markdown(
                    """
                            ### 把资料变成你的本地知识图谱

                            按这条顺序走最稳：
                            1. 导入文件或网页
                            2. 抽取三元组
                            3. 构建并切换到用户数据
                            4. 回到「问答」验证
                            5. 到「知识图谱」查看结果
                    """
                )

                # 🚀 一键加载示例数据（新用户零配置快速体验）
                with gr.Accordion("🚀 快速开始（零配置）", open=True):
                    gr.Markdown(
                        """
                        首次使用？无需配置任何 API，**一键加载内置电影知识图谱示例数据**，
                        索引自动构建完成，立即可在「问答」Tab 提问。

                        > 💡 示例数据已内置，离线可用。如需回答自己的文档，请用下方「导入数据」。
                        """
                    )
                    load_example_btn = gr.Button(
                        "🚀 一键加载示例数据", variant="primary", size="lg"
                    )
                    load_example_status = gr.Markdown("")

                with gr.Row():
                    with gr.Column(scale=1):
                        # Step 1: 导入数据
                        gr.Markdown("#### Step 1: 导入数据")

                        with gr.Tabs():
                            with gr.TabItem("📄 文件上传"):
                                file_upload = gr.File(
                                    label="支持 .txt / .md / .pdf / 图片 (.jpg/.png/.webp 等)",
                                    file_types=[
                                        ".txt",
                                        ".md",
                                        ".markdown",
                                        ".pdf",
                                        "image",
                                    ],
                                    type="filepath",
                                )
                                with gr.Row():
                                    image_mode = gr.Radio(
                                        choices=["OCR 文字提取", "直接抽取知识"],
                                        value="OCR 文字提取",
                                        label="图片处理模式",
                                        interactive=True,
                                    )
                                upload_status = gr.Markdown("")
                                file_content = gr.State("")
                                doc_id_state = gr.State("")

                                file_upload.change(
                                    handle_upload,
                                    inputs=[file_upload, image_mode],
                                    outputs=[upload_status, file_content, doc_id_state],
                                )

                            with gr.TabItem("🌐 网页 URL"):
                                url_input = gr.Textbox(
                                    label="输入网页链接",
                                    placeholder="https://example.com/article",
                                )
                                with gr.Row():
                                    use_playwright = gr.Checkbox(
                                        label="启用 Playwright 动态渲染（支持 JS 动态内容）",
                                        value=True,
                                        interactive=True,
                                    )
                                url_import_btn = gr.Button(
                                    "导入网页", variant="secondary"
                                )
                                url_status = gr.Markdown("")

                                url_import_btn.click(
                                    handle_url_import,
                                    inputs=[url_input, use_playwright],
                                    outputs=[url_status, file_content, doc_id_state],
                                )

                        # Step 2: 抽取三元组
                        gr.Markdown("#### Step 2: 抽取三元组")
                        extract_btn = gr.Button("开始抽取", variant="secondary")
                        extract_status = gr.Markdown("")
                        triples_state = gr.State([])

                        extract_btn.click(
                            handle_extract,
                            inputs=[file_content],
                            outputs=[extract_status, triples_state],
                        )

                        # Step 3: 构建索引
                        gr.Markdown(
                            "#### Step 3: 构建索引\n"
                            "构建完成后会自动切换到用户数据。随后请回到「问答」验证答案，再去「知识图谱」查看图结构。"
                        )
                        build_btn = gr.Button("构建索引并切换", variant="primary")
                        build_status = gr.Markdown("")
                        dataset_state = gr.State("example")

                        # 重置用户数据集（清空用户三元组，切回示例数据）
                        with gr.Accordion("⚠️ 重置用户数据集", open=False):
                            gr.Markdown(
                                "清空已导入的用户三元组并切回示例数据。此操作不可撤销。"
                            )
                            reset_user_btn = gr.Button(
                                "确认清空用户数据并重置", variant="stop"
                            )
                            reset_user_status = gr.Markdown("")

                    with gr.Column(scale=1):
                        # 数据集切换
                        gr.Markdown("#### 数据集切换")
                        dataset_radio = gr.Radio(
                            choices=["示例数据（电影知识图谱）", "用户数据"],
                            value="示例数据（电影知识图谱）",
                            label="当前使用的数据集",
                        )
                        switch_status = gr.Markdown("当前: 示例数据（电影知识图谱）")

                        refresh_status_btn.click(
                            lambda: (
                                load_runtime_status(),
                                load_llm_status(),
                                load_recommended_questions_markdown(),
                            ),
                            outputs=[
                                runtime_status_md,
                                llm_status_md,
                                recommended_questions_md,
                            ],
                        )

                        build_btn.click(
                            handle_build_index,
                            inputs=[triples_state, doc_id_state],
                            outputs=[
                                build_status,
                                dataset_state,
                                dataset_radio,
                                switch_status,
                                runtime_status_md,
                                recommended_questions_md,
                            ],
                        )

                        # 重置按钮绑定（需在 dataset_radio 定义之后）
                        reset_user_btn.click(
                            handle_reset_user_dataset,
                            inputs=[],
                            outputs=[
                                reset_user_status,
                                dataset_radio,
                                switch_status,
                                runtime_status_md,
                                recommended_questions_md,
                            ],
                        )

                        dataset_radio.change(
                            handle_switch_dataset,
                            inputs=[dataset_radio],
                            outputs=[
                                switch_status,
                                runtime_status_md,
                                recommended_questions_md,
                            ],
                        )

                        # 一键加载示例数据：同步 dataset_radio 与 switch_status
                        load_example_btn.click(
                            handle_load_example,
                            inputs=[],
                            outputs=[
                                load_example_status,
                                dataset_radio,
                                switch_status,
                                runtime_status_md,
                                recommended_questions_md,
                            ],
                        )

                        # --- E: 已导入文档管理 ---
                        gr.Markdown("#### 📂 已导入文档管理")
                        docs_summary_md = gr.Markdown("📂 暂无已归档文档")
                        docs_df = gr.Dataframe(
                            headers=["导入时间", "文档名", "类型", "字符数"],
                            datatype=["str", "str", "str", "number"],
                            value=[],
                            interactive=False,
                            wrap=True,
                            label="点击表格行查看内容，再点「删除」可移除",
                        )
                        selected_doc_id = gr.State("")
                        with gr.Row():
                            delete_doc_btn = gr.Button(
                                "🗑️ 删除选中", variant="stop", size="sm"
                            )
                            refresh_docs_btn = gr.Button("🔄 刷新列表", size="sm")
                        doc_viewer_md = gr.Markdown(
                            "👆 点击上方表格中的一行，即可查看文档内容。"
                        )

                        docs_df.select(
                            handle_view_doc,
                            inputs=[],
                            outputs=[doc_viewer_md, selected_doc_id],
                        )
                        delete_doc_btn.click(
                            handle_delete_doc,
                            inputs=[selected_doc_id],
                            outputs=[docs_summary_md, doc_viewer_md, docs_df],
                        )
                        refresh_docs_btn.click(
                            handle_list_docs,
                            inputs=[],
                            outputs=[docs_df, docs_summary_md],
                        )

                        # 数据集信息
                        gr.Markdown(
                            """
                            ---
                            **说明**：
                            - 📚 示例数据：内置电影知识图谱
                            - 👤 用户数据：您上传的私有文档，自动抽取并构建
                            - 切换数据集后，下次提问将使用新数据集
                            - 支持 TXT、Markdown、PDF、网页 URL 多种格式
                            """
                        )

            # ========================
            # Tab 3: 知识图谱
            # ========================
            with gr.TabItem("知识图谱"):
                gr.Markdown(
                    """
                    ### 知识图谱可视化

                    交互式探索知识图谱中的实体和关系。

                    **内置交互功能：
                    - 🔍 **动态搜索** - 输入实体名实时高亮
                    - 📍 **多种布局** - 力导向 / 圆形 / 树状布局切换
                    - 📐 **节点大小** - 滑块调整节点尺寸
                    - 📊 **度数筛选** - 只显示连接多的核心节点
                    - 🔗 **关系筛选** - 按关系类型过滤
                    - 🔄 **重置视图** - 一键恢复默认
                    - 鼠标悬停查看详情，拖动节点可调整位置，滚轮缩放
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 实体搜索")
                        search_input = gr.Textbox(
                            label="输入实体名称",
                            placeholder="例如：电影、主角、导演",
                        )
                        search_btn = gr.Button("搜索实体", variant="primary")
                        overview_btn = gr.Button(
                            "显示图谱概览 (Top 30)", variant="secondary"
                        )

                        graph_info = gr.Markdown(DEFAULT_GRAPH_HINT)

                    with gr.Column(scale=3):
                        # 用 State 存储图谱数据，HTML + IFrame 方式渲染（ECharts JS 需要独立执行环境）
                        graph_data_state = gr.State(
                            json.dumps({"nodes": [], "links": []}, ensure_ascii=False)
                        )
                        graph_html = gr.HTML(
                            value=build_graph_html(
                                json.dumps(
                                    {"nodes": [], "links": []}, ensure_ascii=False
                                )
                            ),
                            label="图谱可视化",
                        )

                # 绑定事件
                search_btn.click(
                    search_graph_entity,
                    inputs=[search_input],
                    outputs=[graph_info, graph_data_state],
                ).then(
                    build_graph_html,
                    inputs=[graph_data_state],
                    outputs=[graph_html],
                )

                overview_btn.click(
                    get_graph_overview,
                    inputs=[],
                    outputs=[graph_info, graph_data_state],
                ).then(
                    build_graph_html,
                    inputs=[graph_data_state],
                    outputs=[graph_html],
                )

            # ========================
            # Tab 4: LLM 设置
            # ========================
            with gr.TabItem("⚙️ LLM 设置"):
                gr.Markdown(
                    """
                    ### LLM 后端配置

                    在网页里直接配置 LLM，无需手动编辑 `.env` 文件。保存后**立即热生效**，
                    下次问答自动使用新配置。

                    > 💡 **Ollama 用户**：模型文件需先下载，点下方「⬇️ 拉取/下载模型」即可，
                    > 无需手动在终端运行 `ollama pull`。
                    """
                )

                settings_status_md = gr.Markdown(
                    f"**当前生效后端**: {get_active_provider()}",
                    elem_id="settings_status",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        provider_dd = gr.Dropdown(
                            label="选择 LLM 后端",
                            choices=list(LLM_PROVIDERS.keys()),
                            value=detect_active_provider(),
                            info="选择一个 provider，下方会显示对应配置",
                        )

                        # --- Ollama 配置 ---
                        gr.Markdown("#### 🦙 Ollama（本地离线，推荐）")
                        with gr.Row():
                            ollama_model_tb = gr.Textbox(
                                label="Ollama 模型名",
                                placeholder="qwen2.5:7b",
                                info="可直接点下方按钮下载，无需手动跑 ollama pull",
                            )
                            ollama_base_tb = gr.Textbox(
                                label="Ollama API 地址",
                                value="http://localhost:11434/v1",
                                info="默认本机 Ollama 服务",
                            )
                        with gr.Row():
                            ollama_list_btn = gr.Button(
                                "📋 列出已下载模型", variant="secondary"
                            )
                            ollama_pull_btn = gr.Button(
                                "⬇️ 拉取/下载模型", variant="primary"
                            )
                        ollama_models_md = gr.Markdown("")
                        ollama_pull_md = gr.Markdown(
                            "点「拉取/下载模型」会从 Ollama 下载模型文件到本机，"
                            "首次可能几分钟到几十分钟。进度会实时显示。"
                        )

                        # --- SiliconFlow 配置 ---
                        gr.Markdown("#### 🟢 SiliconFlow（国内访问快，有免费额度）")
                        sf_key_tb = gr.Textbox(
                            label="API Key",
                            placeholder="sk-...",
                            type="password",
                        )
                        sf_model_tb = gr.Textbox(
                            label="模型名",
                            value="Qwen/Qwen2.5-7B-Instruct",
                        )

                        # --- DeepSeek 配置 ---
                        gr.Markdown("#### 🔵 DeepSeek（推理能力强）")
                        ds_key_tb = gr.Textbox(
                            label="API Key",
                            placeholder="sk-...",
                            type="password",
                        )
                        ds_model_tb = gr.Textbox(label="模型名", value="deepseek-chat")

                        # --- DashScope 配置 ---
                        gr.Markdown("#### 🟣 阿里云 DashScope（支持图片 VLM）")
                        dq_key_tb = gr.Textbox(
                            label="API Key",
                            placeholder="sk-...",
                            type="password",
                        )
                        dq_model_tb = gr.Textbox(label="模型名", value="qwen-plus")

                        # --- OpenAI 配置 ---
                        gr.Markdown("#### ⚪ OpenAI 或兼容接口")
                        oa_key_tb = gr.Textbox(
                            label="API Key",
                            placeholder="sk-...",
                            type="password",
                        )
                        oa_base_tb = gr.Textbox(
                            label="API Base",
                            value="https://api.openai.com/v1",
                        )
                        oa_model_tb = gr.Textbox(label="模型名", value="gpt-3.5-turbo")

                    with gr.Column(scale=1):
                        save_result_md = gr.Markdown(
                            "配置保存后会写入 `.env` 并热同步到运行时。"
                        )
                        save_btn = gr.Button(
                            "💾 保存配置并热生效", variant="primary", size="lg"
                        )
                        test_btn = gr.Button("🔌 测试连接", variant="secondary")
                        test_result_md = gr.Markdown(
                            "点测试连接会发一个最小 ping 请求验证配置。"
                        )

                        gr.Markdown(
                            """
                            ---
                            ### 📖 Provider 说明

                            | Provider | 需 API Key | 申请地址 |
                            |----------|-----------|----------|
                            | **Ollama** | 否 | https://ollama.com/ |
                            | **SiliconFlow** | 是 | https://siliconflow.cn/ |
                            | **DeepSeek** | 是 | https://platform.deepseek.com/ |
                            | **DashScope** | 是 | https://dashscope.console.aliyun.com/ |
                            | **OpenAI** | 是 | https://platform.openai.com/ |

                            ### 🔒 安全说明
                            - API Key 保存在本机 `.env` 文件，不会上传任何服务器
                            - 密码框输入不会明文显示
                            - 切换 provider 时会自动清空其他 provider 的 Key，避免多个后端同时生效
                            """
                        )

                _settings_outputs = [
                    provider_dd,
                    ollama_model_tb,
                    ollama_base_tb,
                    sf_key_tb,
                    sf_model_tb,
                    ds_key_tb,
                    ds_model_tb,
                    dq_key_tb,
                    dq_model_tb,
                    oa_key_tb,
                    oa_base_tb,
                    oa_model_tb,
                    settings_status_md,
                ]

                demo.load(
                    fn=_load_settings_ui,
                    inputs=[],
                    outputs=_settings_outputs,
                )

                # 页面加载时也刷新问答 Tab 的 LLM 状态灯
                demo.load(
                    fn=load_llm_status,
                    inputs=[],
                    outputs=[llm_status_md],
                )

                # 页面加载时刷新已导入文档列表（E 模块）
                demo.load(
                    fn=handle_list_docs,
                    inputs=[],
                    outputs=[docs_df, docs_summary_md],
                )

                save_btn.click(
                    fn=_save_settings_ui,
                    inputs=[
                        provider_dd,
                        ollama_model_tb,
                        ollama_base_tb,
                        sf_key_tb,
                        sf_model_tb,
                        ds_key_tb,
                        ds_model_tb,
                        dq_key_tb,
                        dq_model_tb,
                        oa_key_tb,
                        oa_base_tb,
                        oa_model_tb,
                    ],
                    outputs=[save_result_md, settings_status_md, llm_status_md],
                )

                test_btn.click(
                    fn=_test_connection,
                    inputs=[],
                    outputs=[test_result_md],
                )

                ollama_list_btn.click(
                    fn=_list_ollama,
                    inputs=[ollama_base_tb],
                    outputs=[ollama_models_md, ollama_model_tb],
                )

                ollama_pull_btn.click(
                    fn=pull_ollama_model,
                    inputs=[ollama_model_tb, ollama_base_tb],
                    outputs=[ollama_pull_md],
                )

    return demo


def _launch_with_fallback(demo, launch_kwargs: dict):
    """启动 Gradio；默认端口被占用时回退到可用端口。"""
    try:
        demo.launch(**launch_kwargs)
        return
    except OSError as e:
        if "Cannot find empty port" not in str(e):
            raise

    fallback_kwargs = dict(launch_kwargs)
    fallback_kwargs.pop("server_port", None)
    requested_port = launch_kwargs.get("server_port", 7860)
    print(f"⚠️ 默认端口 {requested_port} 被占用，正在回退到可用端口...")
    demo.launch(**fallback_kwargs)


def main():
    import sys

    print("正在初始化 RAG 系统...")
    get_rag()  # 预加载
    print("正在启动 Gradio 界面...")
    demo = build_ui()

    # 检查参数
    use_share = "--share" in sys.argv
    server_name = "127.0.0.1"
    if "--listen" in sys.argv:
        server_name = "0.0.0.0"

    port = int(os.environ.get("POCKET_WEBUI_PORT", "7860"))

    # 启动参数：show_api=False 避免 gradio_client 版本兼容性问题
    launch_kwargs = {
        "server_name": server_name,
        "server_port": port,
        "share": use_share,
        "show_error": True,
        "show_api": False,
        "quiet": False,
    }

    try:
        _launch_with_fallback(demo, launch_kwargs)
    except ValueError as e:
        if "localhost is not accessible" in str(e) and not use_share:
            print("[WARN] 本地访问受限，自动启用 Gradio Share 模式...")
            launch_kwargs["share"] = True
            _launch_with_fallback(demo, launch_kwargs)
        else:
            raise


if __name__ == "__main__":
    main()
