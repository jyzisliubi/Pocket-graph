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
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .config import SEARCH_MODE
from .kg_reasoning import KGDualRetriever
from .llm import get_active_provider, has_llm
from .logging_config import get_logger
from .rag_system import PocketGraphRAG
from .settings_manager import (
    detect_active_provider,
    is_ollama_running,
    load_llm_settings,
)

logger = get_logger(__name__)


_rag: PocketGraphRAG = None
_kg_retriever: KGDualRetriever = None

API_KEY = os.environ.get("POCKET_API_KEY", "")
_CORS_ORIGINS = os.environ.get("POCKET_CORS_ORIGINS", "*")
_CORS_ALLOW_CREDENTIALS = os.environ.get("POCKET_CORS_CREDENTIALS", "").lower() in ("1", "true", "yes")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_api_key(request: Request, key: Optional[str] = Depends(api_key_header)):
    """Verify API key if POCKET_API_KEY is configured.

    - If POCKET_API_KEY is not set: all requests allowed (local dev mode)
    - If set: requests must provide X-API-Key header matching the key
    - Health endpoint (/api/health, /) is always accessible
    """
    if not API_KEY:
        return None
    if request.url.path in ("/", "/api/health", "/docs", "/redoc", "/openapi.json"):
        return None
    if key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Use X-API-Key header.",
        )
    return key


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
    version="0.3.0",
    lifespan=lifespan,
)

_cors_origins_list = [o.strip() for o in _CORS_ORIGINS.split(",") if o.strip()] if _CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins_list,
    allow_credentials=_CORS_ALLOW_CREDENTIALS and _CORS_ORIGINS != "*",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ==========================
# Pydantic Models
# ==========================


class QARequest(BaseModel):
    query: str = Field(..., description="用户问题")
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


class KGPathInfo(BaseModel):
    search_type: str = ""
    seed_entities: List[str] = []
    expanded_entities: List[str] = []
    matched_relations: List[str] = []


class PipelineInfo(BaseModel):
    search_mode: str = "vector"
    query_rewritten: bool = False
    multihop_used: bool = False
    kg_path: KGPathInfo = KGPathInfo()


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
    symbolSize: int


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


# ==========================
# Helper
# ==========================


def _get_rag() -> PocketGraphRAG:
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")
    return _rag


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
        "version": "0.3.0",
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
    }


# ==========================
# Q&A Endpoints
# ==========================


@app.post("/api/qa", response_model=QAResponse, summary="问答（非流式）")
async def qa(request: QARequest):
    """Answer a question using GraphRAG (non-streaming)."""
    rag = _get_rag()

    # H4 修复：不再修改实例属性（原 try/finally 改属性会被并发请求互相污染），
    # 改为把请求参数透传给 answer()，由其用局部变量处理。
    result = rag.answer(
        request.query,
        top_k=request.top_k,
        use_reranker=bool(request.use_reranker),
        vector_weight=request.vector_weight,
        search_mode=request.search_mode,
        use_multihop=request.use_multihop,
        use_hyde=request.use_hyde,
        use_query_router=request.use_query_router,
    )

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
            search_mode=pipeline_info.get("search_mode", "vector"),
            query_rewritten=pipeline_info.get("query_rewritten", False),
            multihop_used=pipeline_info.get("multihop_used", False),
            kg_path=KGPathInfo(
                search_type=kg_path.get("search_type", ""),
                seed_entities=kg_path.get("seed_entities", []),
                expanded_entities=kg_path.get("expanded_entities", []),
                matched_relations=kg_path.get("matched_relations", []),
            ),
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
# Root
# ==========================


@app.get("/", summary="根路径")
async def root():
    return {
        "name": "PocketGraphRAG API",
        "version": "0.3.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "llm_status": "/api/llm/status",
            "qa": "/api/qa",
            "qa_stream": "/api/qa/stream",
            "graph_stats": "/api/graph/stats",
            "graph_entities": "/api/graph/entities",
            "graph_search": "/api/graph/search",
            "graph_entity_subgraph": "/api/graph/entity/{name}/subgraph",
        },
    }


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
