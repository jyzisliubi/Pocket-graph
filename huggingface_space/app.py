"""PocketGraphRAG HuggingFace Space Demo

轻量级在线 demo，用户无需本地部署即可体验 PocketGraphRAG。
- 上传文档（TXT/MD/PDF）→ 自动建索引
- 输入问题 → GraphRAG 检索 + LLM 生成答案
- 展示知识图谱路径 + 引用来源

部署：将本目录推送到 HuggingFace Space（SDK: Gradio）
"""

import os
import tempfile
from typing import Optional

import gradio as gr

from PocketGraphRAG.data_importer import DataImporter
from PocketGraphRAG.kg_extractor import extract_knowledge_graph
from PocketGraphRAG.rag_system import PocketGraphRAG

# 全局 RAG 实例
_rag: Optional[PocketGraphRAG] = None
_tempdir = tempfile.mkdtemp(prefix="pocketgraphrag_")


def init_rag():
    """初始化 RAG 实例（首次调用时懒加载）"""
    global _rag
    if _rag is None:
        print("Initializing PocketGraphRAG...")
        _rag = PocketGraphRAG()
        print(f"Loaded {_rag.graph_store.entity_count()} entities")
    return _rag


def process_upload(file):
    """处理上传的文档，建索引"""
    if file is None:
        return "请上传文档"
    try:
        importer = DataImporter()
        # 复制到 tempdir
        filename = os.path.basename(file.name)
        dest = os.path.join(_tempdir, filename)
        import shutil
        shutil.copy(file.name, dest)

        # 导入文档
        doc = importer.import_file(dest)
        if not doc.content.strip():
            return f"❌ 文档 {filename} 内容为空"

        # 抽取 KG
        result = extract_knowledge_graph(doc.content, verbose=False)
        added = _rag.add_triples(result.triples)
        return f"✅ {filename} 处理完成\n抽取 {len(result.triples)} 三元组，新增 {added} 条\n当前共 {_rag.graph_store.entity_count()} 实体"
    except Exception as e:
        return f"❌ 处理失败: {e}"


def answer_question(query, search_mode, top_k, use_reranker):
    """回答问题"""
    if not query.strip():
        return "请输入问题", "", ""
    rag = init_rag()
    if rag.graph_store.entity_count() == 0:
        return "请先上传文档建索引", "", ""
    try:
        result = rag.answer(
            query=query,
            search_mode=search_mode,
            top_k=int(top_k),
            use_reranker=use_reranker,
        )
        answer = result["answer"]
        # 格式化 sources
        sources_md = "## 📚 引用来源\n\n"
        for i, src in enumerate(result["sources"], 1):
            cid = src.get("citation_id", i)
            score = src.get("score", 0)
            entity = src.get("entity", "")
            text = src.get("text", "")[:200]
            sources_md += f"**[{cid}] {entity}** (score: {score:.3f})\n> {text}...\n\n"
        # 格式化 pipeline info
        pipe = result.get("pipeline_info", {})
        pipe_md = "## 🔧 流水线\n\n"
        pipe_md += f"- 搜索模式: {pipe.get('search_mode', search_mode)}\n"
        pipe_md += f"- KG 实体匹配: {pipe.get('kg_entities_matched', 0)}\n"
        pipe_md += f"- 重排器: {'✓' if pipe.get('reranker_used') else '✗'}\n"
        pipe_md += f"- 拒答: {'是' if pipe.get('refused') else '否'}\n"
        return answer, sources_md, pipe_md
    except Exception as e:
        return f"❌ 查询失败: {e}", "", ""


# ==========================
# Gradio UI
# ==========================

with gr.Blocks(title="PocketGraphRAG Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎯 PocketGraphRAG Demo
        Local-first GraphRAG with Multi-Model KG Fusion, Zero-LLM Query, and 2.7x LightRAG MRR on HotpotQA.

        ## 使用方式
        1. 上传文档（TXT/MD/PDF/Word）→ 自动抽取知识图谱
        2. 输入问题 → GraphRAG 检索 + LLM 生成答案
        3. 查看引用来源 [1][2] 和流水线信息
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 文档上传")
            file_upload = gr.File(
                label="上传文档",
                file_types=[".txt", ".md", ".pdf", ".docx", ".doc"],
            )
            upload_btn = gr.Button("🔄 建索引", variant="primary")
            upload_status = gr.Textbox(label="状态", lines=4)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 问答")
            query_input = gr.Textbox(label="问题", placeholder="例如：水稻稻瘟病的防治方法？", lines=2)
            with gr.Row():
                search_mode = gr.Dropdown(
                    choices=["mix", "local", "global", "vector", "kg_only"],
                    value="mix", label="搜索模式",
                )
                top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
                use_reranker = gr.Checkbox(value=True, label="KG-aware Reranker")
            ask_btn = gr.Button("🚀 提问", variant="primary")
            answer_output = gr.Markdown(label="回答")

            with gr.Accordion("📚 引用来源 & 流水线", open=False):
                sources_output = gr.Markdown()
                pipeline_output = gr.Markdown()

    upload_btn.click(process_upload, inputs=[file_upload], outputs=[upload_status])
    ask_btn.click(
        answer_question,
        inputs=[query_input, search_mode, top_k, use_reranker],
        outputs=[answer_output, sources_output, pipeline_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
