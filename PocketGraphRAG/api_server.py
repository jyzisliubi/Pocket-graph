"""
PocketGraphRAG REST API Server

基于 FastAPI 的 HTTP API 服务，支持：
- 问答接口（流式 + 非流式）
- 知识图谱查询接口
- 健康检查

使用方式：
    python -m PocketGraphRAG.api_server
    # 或
    uvicorn PocketGraphRAG.api_server:app --host 0.0.0.0 --port 8000
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    API_AUTH_ENABLED,
    API_KEYS,
    API_PROTECTED_PREFIX,
    API_PUBLIC_PATHS,
    DATA_PATH,
    EXTRACT_LLM_CONFIG,
    INDEX_DIR,
    KEYWORDS_LLM_CONFIG,
    LANGFUSE_ENABLED,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    QUERY_LLM_CONFIG,
    SEARCH_MODE,
    USER_DOCS_DIR,
    USER_TRIPLES_PATH,
    VLM_LLM_CONFIG,
)
from .kg_reasoning import KGDualRetriever
from .llm import get_active_provider, has_llm
from .logging_config import get_logger
from .rag_system import PocketGraphRAG
from .settings_manager import (
    detect_active_provider,
    is_ollama_running,
    list_ollama_models,
    load_llm_settings,
    save_llm_settings,
)

logger = get_logger(__name__)


_rag: PocketGraphRAG = None
_kg_retriever: KGDualRetriever = None

# 向后兼容：旧的单 key 环境变量 POCKET_API_KEY 仍生效
_LEGACY_API_KEY = os.environ.get("POCKET_API_KEY", "")
_ALL_API_KEYS = set(API_KEYS)
if _LEGACY_API_KEY:
    _ALL_API_KEYS.add(_LEGACY_API_KEY)
_AUTH_ENABLED = API_AUTH_ENABLED or bool(_LEGACY_API_KEY)

_CORS_ORIGINS = os.environ.get("POCKET_CORS_ORIGINS", "*")
_CORS_ALLOW_CREDENTIALS = os.environ.get("POCKET_CORS_CREDENTIALS", "").lower() in ("1", "true", "yes")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_header = APIKeyHeader(name="Authorization", auto_error=False)


async def _verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Depends(api_key_header),
    authorization: Optional[str] = Depends(bearer_header),
):
    """Verify API key if POCKET_API_KEYS / POCKET_API_KEY is configured.

    支持两种认证头（任一即可）：
      - X-API-Key: <key>
      - Authorization: Bearer <key>

    - 未配置任何 key 时：所有请求放行（本地开发模式）
    - 已配置：受保护路径必须带有效 key
    - 公开路径（健康检查、docs、llm/status）始终放行

    多 key 支持：POCKET_API_KEYS=k1,k2,k3 逗号分隔，便于团队/轮换场景。
    """
    if not _AUTH_ENABLED:
        return None
    path = request.url.path
    # 公开路径白名单
    if path in API_PUBLIC_PATHS:
        return None
    # 非受保护前缀放行（静态资源等）
    if not path.startswith(API_PROTECTED_PREFIX):
        return None

    # 提取候选 key（X-API-Key 优先，其次 Bearer）
    candidate = x_api_key
    if not candidate and authorization:
        # 支持 "Bearer xxx" 和裸 token 两种格式
        if authorization.lower().startswith("bearer "):
            candidate = authorization[7:].strip()
        else:
            candidate = authorization.strip()

    if not candidate or candidate not in _ALL_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Use X-API-Key or Authorization: Bearer header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return candidate


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup, clean up on shutdown."""
    global _rag, _kg_retriever

    logger.info("Initializing RAG system...")
    _rag = PocketGraphRAG(
        search_mode=SEARCH_MODE,
        use_multihop=False,
        use_conversation=False,
    )
    logger.info("RAG system ready.")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="PocketGraphRAG API",
    description="Lightweight GraphRAG API for vertical domains",
    version="0.3.3",
    lifespan=lifespan,
    # 全局鉴权：所有路由自动经过 _verify_api_key。
    # 若 POCKET_API_KEY 未设置则直接放行（本地开发模式）。
    dependencies=[Depends(_verify_api_key)],
)

_cors_origins_list = [o.strip() for o in _CORS_ORIGINS.split(",") if o.strip()] if _CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins_list,
    allow_credentials=_CORS_ALLOW_CREDENTIALS and _CORS_ORIGINS != "*",
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ==========================
# Pydantic Models
# ==========================


class QARequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="用户问题")
    search_mode: Optional[str] = Field(
        default=None,
        description="检索模式: vector / local / global / mix / kg_only",
    )
    use_multihop: Optional[bool] = Field(
        default=False, description="是否启用多跳查询分解"
    )
    top_k: Optional[int] = Field(default=5, ge=1, le=100, description="检索返回数量")
    use_reranker: Optional[bool] = Field(
        default=False,
        description="是否启用 CrossEncoder 重排序（首次启用需联网下载模型，失败回退关键词重排）",
    )
    vector_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="混合检索时向量结果权重 (0.0~1.0)，None 用默认值。仅 mix/local/global 生效",
    )
    use_hyde: Optional[bool] = Field(
        default=False,
        description="是否启用 HyDE（用 LLM 生成假设性文档做向量检索，提升短问题召回。需 LLM；与多跳互斥）",
    )
    use_query_router: Optional[bool] = Field(
        default=False,
        description="是否启用查询路由（LLM 自动选检索模式 vector/local/global/mix/global_summary，覆盖 search_mode。需 LLM）",
    )
    use_self_check: Optional[bool] = Field(
        default=False,
        description="是否启用答案自检（LLM 校验答案是否被上下文支持，发现幻觉加警告并重试。需 LLM；失败默认通过）",
    )


class Source(BaseModel):
    entity: str
    text: str
    score: float
    citation_id: Optional[int] = None  # 引用编号 [1] [2] ...，与 answer 中的标注对应


class KGPathInfo(BaseModel):
    search_type: str = ""
    seed_entities: List[str] = []
    expanded_entities: List[str] = []
    matched_relations: List[str] = []


class PipelineInfo(BaseModel):
    search_mode: str = "vector"
    query_rewritten: bool = False
    multihop_used: bool = False
    multihop_auto_triggered: Optional[bool] = None
    kg_path: KGPathInfo = KGPathInfo()
    kg_entities_matched: Optional[int] = None
    top_k: Optional[int] = None
    response_mode: Optional[str] = None
    failure_bucket: Optional[str] = None
    fallback_reason: Optional[str] = None
    question_type: Optional[str] = None
    reranker_used: Optional[bool] = None
    vector_weight: Optional[float] = None
    hyde_used: Optional[bool] = None
    query_routed: Optional[bool] = None
    self_check_used: Optional[bool] = None
    refused: Optional[bool] = None
    refuse_reason: Optional[str] = None
    llm_error: Optional[str] = None


class QAResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    pipeline_info: PipelineInfo = PipelineInfo()
    effective_query: str = ""


class RetrieveResponse(BaseModel):
    """只检索不生成的响应（BUG #3：补 /api/retrieve 端点）"""

    sources: List[Source] = []
    kg_path: KGPathInfo = KGPathInfo()
    query: str = ""


class GraphStats(BaseModel):
    total_entities: int
    total_relations: int
    total_edges: int
    avg_degree: float


class GraphNode(BaseModel):
    id: str
    name: str
    degree: int
    category: int
    symbolSize: float


class GraphLink(BaseModel):
    source: str
    target: str
    relation: str


class SubgraphResponse(BaseModel):
    nodes: List[GraphNode]
    links: List[GraphLink]


class EntitySearchResult(BaseModel):
    entity: str
    degree: int


class ExtractRequest(BaseModel):
    filename: str = Field(..., description="已上传的文档文件名")


class MultiModelExtractRequest(BaseModel):
    filename: str = Field(..., description="已上传的文档文件名")
    models: List[str] = Field(
        ...,
        description="LLM 模型名列表（至少 2 个），如 ['qwen-flash', 'qwen-max']",
    )
    strategy: str = Field(
        default="union",
        description="融合策略: union (并集去重) / intersect (交集)",
    )
    min_confidence: float = Field(default=0.6, description="最低置信度阈值")


class SettingsRequest(BaseModel):
    provider: str = Field(
        ...,
        description="LLM provider: ollama / siliconflow / deepseek / dashscope / openai / freellm-cn",
    )
    api_key: Optional[str] = Field(default=None, description="API Key（ollama 不需要）")
    model: Optional[str] = Field(default=None, description="模型名")
    api_base: Optional[str] = Field(
        default=None, description="API base URL（ollama / openai 可选）"
    )


class DocumentInfo(BaseModel):
    filename: str
    size: int
    uploaded_at: str


class UploadResponse(BaseModel):
    filename: str
    path: str
    size: int
    message: str


class BuildIndexStats(BaseModel):
    entities: int
    relations: int


class BuildIndexResponse(BaseModel):
    message: str
    stats: BuildIndexStats


# ==========================
# Helper
# ==========================


def _get_rag() -> PocketGraphRAG:
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")
    return _rag


def _safe_doc_path(filename: str) -> str:
    """拼接 USER_DOCS_DIR 与 filename，防止路径穿越。

    只保留 basename，并校验归一化后的绝对路径仍位于 USER_DOCS_DIR 内。
    """
    base = os.path.abspath(USER_DOCS_DIR)
    safe_name = os.path.basename(filename or "")
    if not safe_name or safe_name in (".", ".."):
        raise HTTPException(status_code=400, detail="非法文件名")
    full = os.path.abspath(os.path.join(base, safe_name))
    if full != base and not full.startswith(base + os.sep):
        raise HTTPException(status_code=400, detail="非法文件路径")
    return full


def _write_triples(path: str, triples: List, append: bool = True) -> int:
    """把三元组列表写入文件（head | relation | tail 格式），返回写入条数。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with open(path, mode, encoding="utf-8") as f:
        for t in triples:
            head = str(t[0]).replace("|", "").replace("\n", " ").strip()
            rel = str(t[1]).replace("|", "").replace("\n", " ").strip()
            tail = str(t[2]).replace("|", "").replace("\n", " ").strip()
            if head and rel and tail:
                f.write(f"{head} | {rel} | {tail}\n")
                count += 1
    return count


# ==========================
# Health Check
# ==========================


@app.get("/api/health", summary="健康检查")
async def health_check():
    """检查 API 服务、RAG 系统、LLM 后端三者的就绪状态。

    返回字段：
    - status: overall 服务状态 (ok / initializing)
    - rag_ready: RAG 系统是否初始化完成
    - llm: LLM 后端状态摘要 (provider / model / has_llm / ollama_running)
    - version / search_mode
    """
    provider = detect_active_provider()
    settings = load_llm_settings()
    model_field_map = {
        "ollama": "OLLAMA_MODEL",
        "siliconflow": "SILICONFLOW_MODEL",
        "deepseek": "DEEPSEEK_MODEL",
        "dashscope": "DASHSCOPE_MODEL",
        "openai": "OPENAI_MODEL",
    }
    model = settings.get(model_field_map.get(provider, ""), "") if provider else ""

    ollama_running = None
    if provider == "ollama":
        base = settings.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
        ollama_running = is_ollama_running(base)

    return {
        "status": "ok" if _rag is not None else "initializing",
        "version": "0.3.3",
        "search_mode": SEARCH_MODE,
        "rag_ready": _rag is not None,
        "llm": {
            "provider": provider,
            "provider_label": get_active_provider(),
            "model": model,
            "has_llm": has_llm(),
            "ollama_running": ollama_running,
        },
    }


@app.get("/api/llm/status", summary="LLM 后端详细状态")
async def llm_status():
    """返回当前 LLM 配置的详细信息（与 Web UI 状态灯一致）。

    敏感字段（API Key）会被脱敏，仅返回是否已配置。
    """
    settings = load_llm_settings()
    provider = settings.get("provider", "") or detect_active_provider()

    def _mask(key: str) -> dict:
        val = settings.get(key, "")
        return {"configured": bool(val), "masked": (val[:4] + "***") if val else ""}

    ollama_base = settings.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
    return {
        "provider": provider,
        "provider_label": get_active_provider(),
        "has_llm": has_llm(),
        "ollama": {
            "model": settings.get("OLLAMA_MODEL", ""),
            "api_base": ollama_base,
            "running": is_ollama_running(ollama_base) if provider == "ollama" else None,
        },
        "siliconflow": {
            "model": settings.get("SILICONFLOW_MODEL", ""),
            "api_key": _mask("SILICONFLOW_API_KEY"),
        },
        "deepseek": {
            "model": settings.get("DEEPSEEK_MODEL", ""),
            "api_key": _mask("DEEPSEEK_API_KEY"),
        },
        "dashscope": {
            "model": settings.get("DASHSCOPE_MODEL", ""),
            "api_key": _mask("DASHSCOPE_API_KEY"),
        },
        "openai": {
            "model": settings.get("OPENAI_MODEL", ""),
            "api_base": settings.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "api_key": _mask("OPENAI_API_KEY"),
        },
        "langfuse": {
            "enabled": LANGFUSE_ENABLED,
            "host": LANGFUSE_HOST,
            "public_key_configured": bool(LANGFUSE_PUBLIC_KEY),
            "secret_key_configured": bool(LANGFUSE_SECRET_KEY),
        },
        "api_auth": {
            "enabled": _AUTH_ENABLED,
            "key_count": len(_ALL_API_KEYS),
            # 不返回 key 本身，仅返回是否配置
        },
        "role_llm": {
            "extract": bool(EXTRACT_LLM_CONFIG),
            "query": bool(QUERY_LLM_CONFIG),
            "keywords": bool(KEYWORDS_LLM_CONFIG),
            "vlm": bool(VLM_LLM_CONFIG),
        },
    }


# ==========================
# Q&A Endpoints
# ==========================


@app.post("/api/qa", response_model=QAResponse, summary="问答（非流式）")
async def qa(request: QARequest):
    """Answer a question using GraphRAG (non-streaming)."""
    rag = _get_rag()

    # 校验 search_mode：避免无效值导致 500（应返回 422 + 友好错误）
    _valid_modes = {"vector", "local", "global", "mix", "kg_only", "global_summary", "drift"}
    if request.search_mode and request.search_mode not in _valid_modes:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_search_mode",
                "message": f"无效的 search_mode={request.search_mode!r}",
                "valid_modes": sorted(_valid_modes),
            },
        )

    # Langfuse Tracing：记录完整问答链路
    from .tracing import start_trace, is_tracing_enabled
    _trace_ctx = start_trace(
        "qa_request",
        input={"query": request.query, "search_mode": request.search_mode},
        metadata={"top_k": request.top_k, "use_reranker": request.use_reranker},
    )
    _trace = _trace_ctx.__enter__()

    # H4 修复：不再修改实例属性（原 try/finally 改属性会被并发请求互相污染），
    # 改为把请求参数透传给 answer()，由其用局部变量处理。
    # BUG 修复：drift 模式首次会构建社区摘要（同步 LLM 调用），可能阻塞 5+ 分钟。
    # 加 60s 超时保护，超时后降级到 mix 模式，避免整个服务卡死。
    import asyncio
    try:
        with _trace.span("retrieval_and_generation",
                          metadata={"search_mode": request.search_mode}) as _span:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    rag.answer,
                    request.query,
                    top_k=request.top_k,
                    use_reranker=bool(request.use_reranker),
                    vector_weight=request.vector_weight,
                    search_mode=request.search_mode,
                    use_multihop=request.use_multihop,
                    use_hyde=request.use_hyde,
                    use_query_router=request.use_query_router,
                ),
                timeout=120.0,
            )
            _span.set_output({
                "answer_length": len(result.get("answer", "")),
                "sources_count": len(result.get("sources") or []),
            })
    except asyncio.TimeoutError:
        _trace_ctx.__exit__(None, None, None)
        from fastapi import HTTPException
        raise HTTPException(
            status_code=504,
            detail={
                "error": "query_timeout",
                "message": "查询超时（120s），可能正在构建社区摘要或 LLM 响应慢。请稍后重试或换用更轻量的 search_mode（如 mix/local）。",
                "search_mode": request.search_mode,
            },
        )
    except ValueError as e:
        _trace_ctx.__exit__(None, None, None)
        # 兜底：search_mode 无效等 ValueError 转成 422
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        _trace_ctx.__exit__(None, None, None)
        raise

    _trace_ctx.__exit__(None, None, None)

    pipeline_info = result.get("pipeline_info", {})
    kg_path = pipeline_info.get("kg_path", {})

    return QAResponse(
        answer=result.get("answer", ""),
        sources=[
            Source(
                entity=s.get("entity", ""),
                text=s.get("text", ""),
                score=float(s.get("score", 0.0)),
            )
            for s in result.get("sources", [])
        ],
        pipeline_info=PipelineInfo(
            search_mode=pipeline_info.get("search_mode") or "vector",
            query_rewritten=pipeline_info.get("query_rewritten", False),
            multihop_used=pipeline_info.get("multihop_used", False),
            multihop_auto_triggered=pipeline_info.get("multihop_auto_triggered"),
            kg_path=KGPathInfo(
                search_type=kg_path.get("search_type", ""),
                seed_entities=kg_path.get("seed_entities", []),
                expanded_entities=kg_path.get("expanded_entities", []),
                matched_relations=kg_path.get("matched_relations", []),
            ),
            kg_entities_matched=pipeline_info.get("kg_entities_matched"),
            top_k=pipeline_info.get("top_k"),
            response_mode=pipeline_info.get("response_mode"),
            failure_bucket=pipeline_info.get("failure_bucket"),
            fallback_reason=pipeline_info.get("fallback_reason"),
            question_type=pipeline_info.get("question_type"),
            reranker_used=pipeline_info.get("reranker_used"),
            vector_weight=pipeline_info.get("vector_weight"),
            hyde_used=pipeline_info.get("hyde_used"),
            query_routed=pipeline_info.get("query_routed"),
            self_check_used=pipeline_info.get("self_check_used"),
            refused=pipeline_info.get("refused"),
            refuse_reason=pipeline_info.get("refuse_reason"),
            llm_error=pipeline_info.get("llm_error"),
        ),
        effective_query=result.get("effective_query", request.query),
    )


@app.post(
    "/api/retrieve",
    response_model=RetrieveResponse,
    summary="只检索不生成（不调用 LLM）",
)
async def retrieve(request: QARequest):
    """只跑检索层，返回 sources + kg_path，不调用 LLM 生成回答。

    用于评估检索质量、调试检索策略、节约 LLM 配额。
    """
    rag = _get_rag()
    # 空/空白 query 直接返回空结果（与 answer() 的拒答逻辑保持一致）
    if not request.query or not request.query.strip():
        return RetrieveResponse(sources=[], kg_path=KGPathInfo(), query=request.query)
    # search_mode 透传：retrieve() 接受 search_mode 参数
    results, kg_path = rag.retrieve(
        request.query,
        top_k=request.top_k,
        use_reranker=bool(request.use_reranker),
        vector_weight=request.vector_weight,
        search_mode=request.search_mode,
    )
    return RetrieveResponse(
        sources=[
            Source(
                entity=(m or {}).get("entity", ""),
                text=text,
                score=float(score),
            )
            for text, score, m in results
        ],
        kg_path=KGPathInfo(
            search_type=kg_path.get("search_type", ""),
            seed_entities=kg_path.get("seed_entities", []),
            expanded_entities=kg_path.get("expanded_entities", []),
            matched_relations=kg_path.get("matched_relations", []),
        ),
        query=request.query,
    )


@app.post("/api/qa/stream", summary="问答（流式，SSE）")
async def qa_stream(request: QARequest):
    """Answer a question using GraphRAG with Server-Sent Events streaming."""
    rag = _get_rag()

    # H4 修复：透传参数，不修改实例属性
    async def event_generator():
        try:
            for step in rag.answer_stream(
                request.query,
                top_k=request.top_k,
                use_reranker=bool(request.use_reranker),
                vector_weight=request.vector_weight,
                search_mode=request.search_mode,
                use_multihop=request.use_multihop,
                use_hyde=request.use_hyde,
                use_query_router=request.use_query_router,
                use_self_check=request.use_self_check,
            ):
                if "chunk" in step:
                    data = {
                        "type": "token",
                        "chunk": step["chunk"],
                        "full_answer": step.get("full_answer", ""),
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                elif "status" in step:
                    if "sources" in step:
                        sources = [
                            {
                                "entity": s.get("entity", ""),
                                "text": s.get("text", ""),
                                "score": float(s.get("score", 0)),
                            }
                            for s in step.get("sources", [])
                        ]
                        pipeline_info = step.get("pipeline_info", {})
                        data = {
                            "type": "sources",
                            "sources": sources,
                            "pipeline_info": pipeline_info,
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    else:
                        data = {"type": "status", "status": step["status"]}
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                elif step.get("done"):
                    data = {"type": "done", "answer": step.get("full_answer", "")}
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as e:
            # M28：SSE 流中途异常时发 error 事件，避免客户端只看到"流断开"
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            # 异常分支也补发 done 事件，让客户端明确知道流已结束，无需死等
            done = {"type": "done"}
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==========================
# Knowledge Graph Endpoints
# ==========================


@app.get("/api/graph/stats", response_model=GraphStats, summary="图谱统计信息")
async def graph_stats():
    """Get knowledge graph statistics."""
    rag = _get_rag()
    stats = rag.kg_retriever.get_graph_stats()
    return GraphStats(**stats)


@app.get(
    "/api/graph/entities",
    response_model=List[EntitySearchResult],
    summary="Top N 实体列表",
)
async def top_entities(
    limit: int = Query(default=50, ge=1, le=500, description="返回数量"),
):
    """Get top N entities by degree."""
    rag = _get_rag()
    top = rag.kg_retriever.get_top_entities(top_k=limit)
    return [
        EntitySearchResult(entity=e, degree=rag.kg_retriever.get_entity_degree(e))
        for e in top
    ]


@app.get(
    "/api/graph/search", response_model=List[EntitySearchResult], summary="搜索实体"
)
async def search_entities(
    q: str = Query(..., description="搜索关键词"),
    limit: int = Query(default=10, ge=1, le=50, description="返回数量"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="相似度阈值"),
):
    """Search entities by embedding similarity."""
    rag = _get_rag()
    entities = rag.kg_retriever.match_entities(q, top_k=limit, threshold=threshold)
    return [
        EntitySearchResult(entity=e, degree=rag.kg_retriever.get_entity_degree(e))
        for e in entities
    ]


@app.get(
    "/api/graph/entity/{name}/subgraph",
    response_model=SubgraphResponse,
    summary="获取实体邻域子图",
)
async def entity_subgraph(
    name: str,
    hops: int = Query(default=1, ge=1, le=3, description="邻域扩展跳数"),
):
    """Get the subgraph around a specific entity."""
    rag = _get_rag()
    if rag.kg_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="KG retriever not available in vector-only mode. "
            "Switch to mix/kg_only/local/global search mode to use KG features",
        )
    subgraph = rag.kg_retriever.get_subgraph([name], max_hops=hops)
    return SubgraphResponse(
        nodes=[GraphNode(**n) for n in subgraph["nodes"]],
        links=[GraphLink(**l) for l in subgraph["links"]],
    )


@app.post(
    "/api/graph/subgraph", response_model=SubgraphResponse, summary="获取多实体子图"
)
async def multi_entity_subgraph(
    entities: List[str],
    hops: int = Query(default=1, ge=1, le=3, description="邻域扩展跳数"),
):
    """Get the subgraph around multiple seed entities."""
    rag = _get_rag()
    subgraph = rag.kg_retriever.get_subgraph(entities, max_hops=hops)
    return SubgraphResponse(
        nodes=[GraphNode(**n) for n in subgraph["nodes"]],
        links=[GraphLink(**l) for l in subgraph["links"]],
    )


# ==========================
# Advanced Graph Algorithm Endpoints
# ==========================


class PagerankResponse(BaseModel):
    entity: str
    score: float


class CommunityResponse(BaseModel):
    entity: str
    community_id: int


class ShortestPathResponse(BaseModel):
    path: List[str]
    length: int


@app.get(
    "/api/graph/pagerank",
    response_model=List[PagerankResponse],
    summary="Pagerank 实体重要性排序",
)
async def graph_pagerank(
    top_n: int = Query(default=50, ge=1, le=500, description="返回前 N 个实体"),
):
    """Get entities ranked by Pagerank importance score."""
    rag = _get_rag()
    if hasattr(rag, "_pagerank_scores") and rag._pagerank_scores:
        pr = rag._pagerank_scores
    else:
        pr = rag.kg_retriever.compute_pagerank()

    sorted_entities = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    top_entities = sorted_entities[:top_n]
    return [PagerankResponse(entity=e, score=float(s)) for e, s in top_entities]


@app.get(
    "/api/graph/communities", response_model=List[CommunityResponse], summary="社区发现"
)
async def graph_communities():
    """Detect communities in the knowledge graph using label propagation."""
    rag = _get_rag()
    if rag.kg_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="KG retriever not available in vector-only mode. "
            "Switch to mix/kg_only/local/global search mode to use KG features",
        )
    communities = rag.kg_retriever.detect_communities()
    return [
        CommunityResponse(entity=e, community_id=int(c)) for e, c in communities.items()
    ]


@app.get("/api/graph/path", response_model=ShortestPathResponse, summary="最短路径")
async def graph_shortest_path(
    start: str = Query(..., description="起始实体"),
    end: str = Query(..., description="目标实体"),
    max_hops: int = Query(default=5, ge=1, le=10, description="最大搜索跳数"),
):
    """Find the shortest path between two entities."""
    rag = _get_rag()
    path = rag.kg_retriever.shortest_path(start, end, max_hops=max_hops)
    return ShortestPathResponse(path=path, length=len(path))


# ==========================
# Document Management Endpoints
# ==========================

_ALLOWED_UPLOAD_EXTS = {".txt", ".md", ".markdown", ".pdf", ".docx"}


@app.post(
    "/api/documents/upload",
    response_model=UploadResponse,
    summary="上传文档（multipart/form-data）",
)
async def upload_document(file: UploadFile = File(...)):
    """接收前端上传的文档，保存到 user_docs 目录。

    支持格式：.txt / .md / .pdf / .docx
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件")
    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，仅支持 {', '.join(sorted(_ALLOWED_UPLOAD_EXTS))}",
        )
    os.makedirs(USER_DOCS_DIR, exist_ok=True)
    dest = _safe_doc_path(filename)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    try:
        with open(dest, "wb") as f:
            f.write(content)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"保存文件失败: {e}")
    logger.info("文档已上传: %s (%s bytes)", dest, len(content))
    return UploadResponse(
        filename=filename,
        path=dest,
        size=len(content),
        message="上传成功",
    )


@app.get(
    "/api/documents",
    response_model=List[DocumentInfo],
    summary="列出已上传文档",
)
async def list_documents():
    """列出 user_docs 目录下已上传的文档。"""
    if not os.path.isdir(USER_DOCS_DIR):
        return []
    results = []
    for name in sorted(os.listdir(USER_DOCS_DIR)):
        full = os.path.join(USER_DOCS_DIR, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in _ALLOWED_UPLOAD_EXTS:
            continue
        try:
            st = os.stat(full)
            uploaded_at = datetime.fromtimestamp(st.st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            size = st.st_size
        except OSError:
            uploaded_at = ""
            size = 0
        results.append(
            DocumentInfo(filename=name, size=size, uploaded_at=uploaded_at)
        )
    return results


@app.delete("/api/documents/{filename}", summary="删除指定文档")
async def delete_document(filename: str):
    """删除 user_docs 目录下的指定文档。"""
    target = _safe_doc_path(filename)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    try:
        os.remove(target)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
    logger.info("文档已删除: %s", target)
    return {"message": "删除成功"}


@app.post("/api/documents/extract", summary="三元组抽取（SSE 流式进度）")
async def extract_document(req: ExtractRequest):
    """对指定文档执行三元组抽取，返回 SSE 流式进度。

    每个进度事件：``data: {"phase": "extracting", "message": "...", "triples_count": N}``
    结束事件：``data: {"phase": "done", "total_triples": N}``

    抽取完成后，三元组会追加保存到 ``user_docs/triples.txt``，供 ``/api/documents/build-index`` 使用。
    """
    from .data_importer import DataImporter
    from .kg_extractor import extract_knowledge_graph_stream
    from .llm import has_llm

    doc_path = _safe_doc_path(req.filename)
    if not os.path.exists(doc_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {req.filename}")

    importer = DataImporter()
    doc = importer.import_file(doc_path)
    if doc is None or not (doc.content or "").strip():
        raise HTTPException(
            status_code=400, detail=f"文档解析失败或内容为空: {req.filename}"
        )

    if not has_llm():
        raise HTTPException(
            status_code=503,
            detail="未配置 LLM 后端，无法执行三元组抽取。请先通过 /api/settings 配置。",
        )

    content = doc.content

    # 用同步生成器：Starlette 会自动放到线程池执行，避免阻塞事件循环
    def event_generator():
        total = 0
        try:
            for step in extract_knowledge_graph_stream(content):
                message = step.get("message", "")
                triples = step.get("triples", []) or []
                if step.get("done"):
                    result = step.get("result")
                    if result is not None and result.triples:
                        total = len(result.triples)
                        try:
                            _write_triples(
                                USER_TRIPLES_PATH,
                                [t.to_tuple() for t in result.triples],
                                append=True,
                            )
                        except OSError as e:
                            logger.warning("三元组持久化失败: %s", e)
                        data = {"phase": "done", "total_triples": total}
                    else:
                        data = {
                            "phase": "empty",
                            "message": message or "未能抽取到任何有效三元组",
                            "total_triples": 0,
                        }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    return
                # 进度事件：抽取流程各阶段统一标记为 extracting
                data = {
                    "phase": "extracting",
                    "message": message,
                    "triples_count": len(triples),
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"phase": "error", "message": f"抽取失败: {e}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/api/documents/extract-multi",
    summary="多模型 KG 融合抽取（PocketGraphRAG 独有）",
)
async def extract_multi_document(req: MultiModelExtractRequest):
    """用多个 LLM 抽取同一份文档并融合，覆盖每个模型的盲点。

    PocketGraphRAG 独有技术，实测在 HotpotQA 上 Hit Rate 0.80 → 0.86（+6%）。
    相当于集成学习，成本低收益高。

    抽取完成后，三元组会追加保存到 ``user_docs/triples.txt``。
    """
    from .data_importer import DataImporter
    from .kg_extractor import extract_triples_multi_model
    from .llm import has_llm

    doc_path = _safe_doc_path(req.filename)
    if not os.path.exists(doc_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {req.filename}")

    if not has_llm():
        raise HTTPException(
            status_code=503,
            detail="未配置 LLM 后端，无法执行三元组抽取。请先通过 /api/settings 配置。",
        )

    if not req.models or len(req.models) < 2:
        raise HTTPException(
            status_code=422,
            detail="多模型融合至少需要 2 个模型",
        )

    if req.strategy not in ("union", "intersect"):
        raise HTTPException(
            status_code=422,
            detail=f"无效的 strategy={req.strategy!r}，可选: union / intersect",
        )

    importer = DataImporter()
    doc = importer.import_file(doc_path)
    if doc is None or not (doc.content or "").strip():
        raise HTTPException(
            status_code=400, detail=f"文档解析失败或内容为空: {req.filename}"
        )

    # 在线程池中执行同步的多模型抽取，避免阻塞事件循环
    import asyncio

    try:
        triples, model_stats = await asyncio.to_thread(
            extract_triples_multi_model,
            text=doc.content,
            models=req.models,
            strategy=req.strategy,
        )
    except Exception as e:
        logger.exception("多模型融合抽取失败")
        raise HTTPException(status_code=500, detail=f"抽取失败: {e}")

    # 过滤低置信度
    filtered = [t for t in triples if t.confidence >= req.min_confidence]

    # 持久化到 user_docs/triples.txt
    try:
        _write_triples(
            USER_TRIPLES_PATH,
            [t.to_tuple() for t in filtered],
            append=True,
        )
    except OSError as e:
        logger.warning("三元组持久化失败: %s", e)

    return {
        "filename": req.filename,
        "strategy": req.strategy,
        "models": req.models,
        "model_stats": model_stats,
        "total_triples": len(filtered),
        "triples": [
            {
                "head": t.head,
                "relation": t.relation,
                "tail": t.tail,
                "confidence": t.confidence,
                "evidence": t.evidence,
            }
            for t in filtered
        ],
    }


@app.post(
    "/api/documents/build-index",
    response_model=BuildIndexResponse,
    summary="构建向量索引",
)
async def build_index_endpoint():
    """触发索引构建。

    合并主知识图谱数据（``DATA_PATH``）与用户抽取的三元组
    （``user_docs/triples.txt``）后构建向量索引，确保检索能覆盖
    完整知识库。若用户三元组不存在则仅使用主数据；若主数据不存在
    则仅使用用户三元组。
    """
    import tempfile

    from .config import RELATION_TEMPLATES, REVERSE_LINK_RELATIONS
    from .data_processor import KGProcessor

    from .build_index import _load_triples_file, build_index_with_data

    # 收集所有可用的三元组数据源（主 KG 优先，用户数据补充）
    sources: list[str] = []
    if os.path.exists(DATA_PATH):
        sources.append(DATA_PATH)
    if os.path.exists(USER_TRIPLES_PATH) and USER_TRIPLES_PATH not in sources:
        sources.append(USER_TRIPLES_PATH)
    if not sources:
        raise HTTPException(
            status_code=404, detail="未找到任何三元组数据文件"
        )

    # 合并所有数据源的三元组（去重），同时保留内存副本供统计使用
    all_triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for src in sources:
        for h, r, t in _load_triples_file(src):
            key = (h, r, t)
            if key not in seen:
                seen.add(key)
                all_triples.append(key)

    # 单一数据源直接使用；多源合并去重到临时文件
    merged_tmp: str | None = None
    if len(sources) == 1:
        data_path = sources[0]
    else:
        tmp_fd, data_path = tempfile.mkstemp(
            suffix=".txt", prefix="merged_triples_"
        )
        merged_tmp = data_path
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for h, r, t in all_triples:
                f.write(f"{h}|{r}|{t}\n")

    try:
        build_index_with_data(data_path, INDEX_DIR, run_tests=False)
    except Exception as e:
        logger.exception("索引构建失败")
        raise HTTPException(status_code=500, detail=f"索引构建失败: {e}")
    finally:
        # 清理临时合并文件（构建已完成或失败，不再需要）
        if merged_tmp and os.path.exists(merged_tmp):
            try:
                os.unlink(merged_tmp)
            except OSError:
                pass

    # 统计实体数（头尾并集）与关系类型数——直接用内存中的三元组，避免依赖临时文件
    entities_set = {t[0] for t in all_triples} | {t[2] for t in all_triples}
    relations_set = {t[1] for t in all_triples}
    entities = len(entities_set)
    relations = len(relations_set)

    return BuildIndexResponse(
        message="索引构建完成",
        stats=BuildIndexStats(entities=entities, relations=relations),
    )


# ==========================
# Settings Management Endpoints
# ==========================


@app.get("/api/settings", summary="获取当前 LLM 配置")
async def get_settings():
    """返回当前 LLM 配置（API Key 脱敏，仅返回是否已配置 + 掩码前缀）。"""
    settings = load_llm_settings()
    provider = settings.get("provider", "") or detect_active_provider()

    model_field_map = {
        "ollama": "OLLAMA_MODEL",
        "freellm-cn": "FREELM_CN_MODEL",
        "siliconflow": "SILICONFLOW_MODEL",
        "deepseek": "DEEPSEEK_MODEL",
        "dashscope": "DASHSCOPE_MODEL",
        "openai": "OPENAI_MODEL",
    }
    model_field = model_field_map.get(provider, "")
    model = settings.get(model_field, "") if model_field else ""

    def _key_configured(key: str) -> bool:
        return bool(settings.get(key, ""))

    return {
        "provider": provider,
        "provider_label": get_active_provider(),
        "model": model,
        "has_llm": has_llm(),
        "ollama": {
            "model": settings.get("OLLAMA_MODEL", ""),
            "api_base": settings.get(
                "OLLAMA_API_BASE", "http://localhost:11434/v1"
            ),
            "api_key_configured": False,
        },
        "freellm-cn": {
            "model": settings.get("FREELM_CN_MODEL", ""),
            "api_base": settings.get(
                "FREELM_CN_API_BASE", "http://localhost:8000/v1"
            ),
            "api_key_configured": _key_configured("FREELM_CN_API_KEY"),
        },
        "siliconflow": {
            "model": settings.get("SILICONFLOW_MODEL", ""),
            "api_key_configured": _key_configured("SILICONFLOW_API_KEY"),
        },
        "deepseek": {
            "model": settings.get("DEEPSEEK_MODEL", ""),
            "api_key_configured": _key_configured("DEEPSEEK_API_KEY"),
        },
        "dashscope": {
            "model": settings.get("DASHSCOPE_MODEL", ""),
            "api_key_configured": _key_configured("DASHSCOPE_API_KEY"),
        },
        "openai": {
            "model": settings.get("OPENAI_MODEL", ""),
            "api_base": settings.get(
                "OPENAI_API_BASE", "https://api.openai.com/v1"
            ),
            "api_key_configured": _key_configured("OPENAI_API_KEY"),
        },
    }


@app.post("/api/settings", summary="保存 LLM 配置")
async def save_settings(req: SettingsRequest):
    """保存 LLM 配置到 .env 并热同步到 config 模块。

    请求体示例：``{"provider": "siliconflow", "api_key": "sk-xxx", "model": "Qwen/..."}``
    仅覆盖当前 provider 相关字段，其他 provider 配置保留。
    """
    provider = (req.provider or "").strip().lower()
    valid_providers = {
        "ollama",
        "freellm-cn",
        "siliconflow",
        "deepseek",
        "dashscope",
        "openai",
    }
    if provider not in valid_providers:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 provider: {provider}，可选: {', '.join(sorted(valid_providers))}",
        )

    # 以现有配置为基底，仅覆盖当前 provider 相关字段
    values = dict(load_llm_settings())
    values["provider"] = provider

    if provider == "ollama":
        if req.model is not None:
            values["OLLAMA_MODEL"] = req.model.strip()
        if req.api_base is not None:
            values["OLLAMA_API_BASE"] = req.api_base.strip()
    elif provider == "freellm-cn":
        if req.api_key is not None:
            values["FREELM_CN_API_KEY"] = req.api_key.strip()
        if req.model is not None:
            values["FREELM_CN_MODEL"] = req.model.strip()
        if req.api_base is not None:
            values["FREELM_CN_API_BASE"] = req.api_base.strip()
    elif provider == "siliconflow":
        if req.api_key is not None:
            values["SILICONFLOW_API_KEY"] = req.api_key.strip()
        if req.model is not None:
            values["SILICONFLOW_MODEL"] = req.model.strip()
    elif provider == "deepseek":
        if req.api_key is not None:
            values["DEEPSEEK_API_KEY"] = req.api_key.strip()
        if req.model is not None:
            values["DEEPSEEK_MODEL"] = req.model.strip()
    elif provider == "dashscope":
        if req.api_key is not None:
            values["DASHSCOPE_API_KEY"] = req.api_key.strip()
        if req.model is not None:
            values["DASHSCOPE_MODEL"] = req.model.strip()
    elif provider == "openai":
        if req.api_key is not None:
            values["OPENAI_API_KEY"] = req.api_key.strip()
        if req.model is not None:
            values["OPENAI_MODEL"] = req.model.strip()
        if req.api_base is not None:
            values["OPENAI_API_BASE"] = req.api_base.strip()

    try:
        save_llm_settings(values)
    except Exception as e:
        logger.exception("保存 LLM 配置失败")
        raise HTTPException(status_code=500, detail=f"保存失败: {e}")

    return {"message": "保存成功", "provider": provider}


@app.get(
    "/api/settings/ollama-models",
    summary="列出 Ollama 已下载的模型",
)
async def list_ollama_models_endpoint():
    """列出本地 Ollama 已下载的模型。"""
    settings = load_llm_settings()
    base = settings.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
    ok, models, msg = list_ollama_models(base)
    if not ok:
        raise HTTPException(status_code=503, detail=msg)
    return {"models": models}


# ==========================
# Static Files & SPA Fallback（前端托管）
# ==========================
# 必须放在所有 API 路由之后：FastAPI 按注册顺序匹配，先匹配 /api/* 等具体路由，
# 兜底路由 /{full_path:path} 才不会拦截 API 请求。

FRONTEND_DIST = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "frontend", "dist"
)

if os.path.exists(FRONTEND_DIST):
    # 挂载 Vite 构建产物中的静态资源（JS/CSS/图片等哈希文件）
    _assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False, summary="SPA 前端兜底")
    async def spa_fallback(full_path: str):
        """所有非 /api 路径的兜底处理：返回静态文件或 index.html（SPA 路由）。"""
        # 不拦截 /api 开头的路径：交由 FastAPI 默认 404 处理
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="Not Found")
        # 命中 dist 下的真实静态文件（如 favicon.svg、vite.svg 等）直接返回
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        # 其余路径（含 "/" 和 /chat 等 SPA 路由）统一返回 index.html
        index_html = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(index_html):
            return FileResponse(index_html)
        raise HTTPException(status_code=404, detail="index.html not found")
else:
    @app.get("/", include_in_schema=False, summary="前端未构建提示")
    async def frontend_not_built():
        return {"message": "前端未构建。请执行：cd frontend && npm run build"}


def main():
    """Start the API server."""
    import uvicorn

    host = os.environ.get("POCKET_API_HOST", "0.0.0.0")
    port = int(os.environ.get("POCKET_API_PORT", "8000"))

    logger.info("Starting PocketGraphRAG API server on %s:%s...", host, port)
    logger.info("API docs: http://%s:%s/docs", host, port)

    uvicorn.run(
        "PocketGraphRAG.api_server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
