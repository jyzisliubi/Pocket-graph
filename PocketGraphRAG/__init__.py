"""
PocketGraphRAG - 轻量级、本地优先的垂直领域 GraphRAG 框架

核心 API:
    from PocketGraphRAG import PocketGraphRAG

    rag = PocketGraphRAG(search_mode="mix")
    result = rag.answer("这部电影讲了什么？")
    print(result["answer"])

模块结构:
    PocketGraphRAG.rag_system    - RAG 核心引擎
    PocketGraphRAG.kg_reasoning  - KG 双层检索器
    PocketGraphRAG.llm           - 统一 LLM 调用层
    PocketGraphRAG.data_processor - 知识图谱数据处理器
    PocketGraphRAG.build_index   - 索引构建
    PocketGraphRAG.kg_extractor  - 自动图谱抽取
    PocketGraphRAG.multihop      - 多跳查询分解
    PocketGraphRAG.conversation  - 对话记忆
    PocketGraphRAG.config        - 配置管理
    PocketGraphRAG.webapp        - Gradio Web 界面
    PocketGraphRAG.cli           - Typer CLI 入口
    PocketGraphRAG.evaluate      - 消融实验评测
"""

import os
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", message=".*Polars.*binary.*missing.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*tokenizers.*parallelism.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

__version__ = "0.3.2"
__author__ = "PocketGraphRAG Team"
__license__ = "MIT"

from .build_index import FAISSIndex, build_index, build_index_with_data
from .conversation import ConversationMemory
from .data_importer import DataImporter, ExtractedDocument
from .data_processor import KGProcessor
from .eval_harness import (
    DEFAULT_BENCHMARK_PATH,
    load_benchmark,
    run_evaluation,
)
from .incremental_index import (
    add_triples_incremental,
    build_manifest_from_data,
    load_doc_map,
    load_manifest,
    remove_document_incremental,
    reset_index,
    save_doc_map,
    save_manifest,
)
from .kg_extractor import (
    ExtractionResult,
    Triple,
    align_entities,
    deduplicate_triples,
    extract_knowledge_graph,
    extract_triples_from_text,
    filter_low_quality,
    semantic_chunk_text,
)
from .kg_reasoning import KGDualRetriever
from .llm import acall_llm, acall_llm_stream, call_llm, get_active_provider, has_llm
from .multihop import decompose_query, multi_hop_retrieve
from .rag_system import PocketGraphRAG
from .vlm_extractor import (
    call_vlm,
    encode_image_to_base64,
    extract_kg_from_image,
    get_vlm_provider,
    has_vlm,
    is_image_file,
    ocr_image,
    ocr_scanned_pdf,
)

# 存储抽象层（可插拔后端）
from .core.storages import (
    FAISSVectorStore,
    GraphStore,
    InMemoryGraphStore,
    JsonKVStorage,
    KVStore,
    VectorStore,
    get_graph_store,
    get_kv_store,
    get_vector_store,
)

__all__ = [
    "PocketGraphRAG",
    "KGDualRetriever",
    "KGProcessor",
    "FAISSIndex",
    "build_index",
    "build_index_with_data",
    "call_llm",
    "has_llm",
    "get_active_provider",
    "acall_llm",
    "acall_llm_stream",
    "call_vlm",
    "ocr_image",
    "extract_kg_from_image",
    "ocr_scanned_pdf",
    "has_vlm",
    "get_vlm_provider",
    "encode_image_to_base64",
    "is_image_file",
    "extract_knowledge_graph",
    "extract_triples_from_text",
    "semantic_chunk_text",
    "align_entities",
    "deduplicate_triples",
    "filter_low_quality",
    "Triple",
    "ExtractionResult",
    "multi_hop_retrieve",
    "decompose_query",
    "ConversationMemory",
    "DataImporter",
    "ExtractedDocument",
    "load_benchmark",
    "run_evaluation",
    "DEFAULT_BENCHMARK_PATH",
    "add_triples_incremental",
    "remove_document_incremental",
    "reset_index",
    "load_manifest",
    "save_manifest",
    "load_doc_map",
    "save_doc_map",
    "build_manifest_from_data",
    # 存储抽象层
    "VectorStore",
    "GraphStore",
    "KVStore",
    "FAISSVectorStore",
    "InMemoryGraphStore",
    "JsonKVStorage",
    "get_vector_store",
    "get_graph_store",
    "get_kv_store",
    "__version__",
]
