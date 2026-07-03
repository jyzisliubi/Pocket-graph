"""
RAG 问答系统核心引擎

完整 Pipeline：
  用户问题
    ↓
  [对话改写] ConversationMemory → 多轮上下文补全
    ↓
  [多跳分解] Multi-hop → 复杂问题拆分为子查询
    ↓
  [双层检索] search_mode 驱动：
    vector  → 纯向量检索
    local   → KG 实体匹配 + BFS 邻域
    global  → KG 关系关键词匹配
    mix     → 向量 + KG 融合
    kg_only → 纯 KG 检索
    ↓
  [Prompt 构建] 知识上下文 + 用户问题
    ↓
  [LLM 生成] 基于知识的精准回答

所有高级特性通过参数开关控制，默认关闭，按需启用。
"""

from typing import Any, Optional

from .answer_verifier import build_warning_prefix, verify_answer
from .build_index import FAISSIndex
from .community_summarizer import (
    build_community_summaries,
    build_hierarchical_community_summaries,
    search_communities,
)
from .config import (
    COMMUNITY_ALGORITHM,
    COMMUNITY_HIERARCHICAL,
    COMMUNITY_MAX_PER_LEVEL,
    COMMUNITY_MAX_TRIPLES,
    COMMUNITY_MIN_SIZE,
    COMMUNITY_RESOLUTION,
    COMMUNITY_RESOLUTIONS,
    COMMUNITY_ROLLUP_THRESHOLD,
    DATA_PATH,
    EMBEDDING_MODEL,
    ENTITY_SIMILARITY_THRESHOLD,
    FUSION_STRATEGY,
    HYDE_ENABLED,
    INDEX_DIR,
    KG_SEARCH_HOPS,
    QUERY_ROUTER_ENABLED,
    REFUSE_ANSWER_TEXT,
    REFUSE_ON_EMPTY_RETRIEVAL,
    RELATION_TEMPLATES,
    REVERSE_LINK_RELATIONS,
    RRF_K,
    SCHEMA_ENABLED,
    SCHEMA_PATH,
    SEARCH_MODE,
    SELF_CHECK_ENABLED,
    SYSTEM_PROMPT,
    TOP_K,
    USER_PROMPT_TEMPLATE,
    VECTOR_REFUSE_THRESHOLD,
    VECTOR_WEIGHT,
)
from .conversation import ConversationMemory
from .data_processor import KGProcessor
from .hyde import generate_hypothetical_document
from .kg_reasoning import KGDualRetriever
from .llm import call_llm, get_active_provider, has_llm
from .logging_config import get_logger
from .multihop import multi_hop_retrieve
from .query_router import QueryRouter
from .schema import RelationSchema

logger = get_logger(__name__)


def _tokenize_zh(text: str) -> list:
    """简易中文分词：按字符切分（适合关键词重合度计算）

    比加载 jieba 轻量，对 Jaccard 重排已足够。
    """
    # 去标点和空白，按字符切
    import re

    clean = re.sub(r"[\s\W_]+", "", text)
    return list(clean)


def _extract_citation_ids(text: str) -> list[str]:
    """提取回答中的引用编号 [1][2]。"""
    import re

    return re.findall(r"\[(\d+)\]", text or "")


class PocketGraphRAG:
    """PocketGraphRAG 问答系统"""

    def __init__(
        self,
        top_k: int = TOP_K,
        use_multihop: bool = False,
        use_conversation: bool = True,
        search_mode: str = SEARCH_MODE,
        data_path: Optional[str] = None,
        use_pagerank: bool = True,
        pagerank_weight: float = 0.3,
        fusion_strategy: Optional[str] = None,
        rrf_k: Optional[int] = None,
        use_hyde: bool = HYDE_ENABLED,
        use_query_router: bool = QUERY_ROUTER_ENABLED,
        use_self_check: bool = SELF_CHECK_ENABLED,
        use_schema: bool = SCHEMA_ENABLED,
    ):
        """
        Args:
            top_k: 最终返回的检索结果数量
            use_multihop: 是否启用多跳查询分解
            use_conversation: 是否启用多轮对话记忆
            search_mode: 检索模式，可选 "vector", "local", "global", "mix", "kg_only"
            data_path: 三元组数据文件路径，None 则使用默认配置
            use_pagerank: 是否启用 Pagerank 重要性加权排序
            pagerank_weight: Pagerank 分数在最终排序中的权重 (0.0-1.0)
            fusion_strategy: 融合策略，"weighted" 加权融合 或 "rrf" 倒数排位融合
            rrf_k: RRF 融合的 k 参数，值越小越看重高排名结果
            use_hyde: 是否启用 HyDE（假设性文档嵌入）查询改写。
                用 LLM 生成假设性答案文档做向量检索，提升短查询召回。
                仅影响 vector/mix 模式的向量检索部分，KG 检索仍用原 query。
                需 LLM；无 LLM 自动回退。可通过 POCKET_HYDE=1 开启
            use_query_router: 是否启用查询路由器。LLM 自动判断问题类型选择
                最优检索模式（vector/local/global/mix/global_summary），
                覆盖 search_mode。需 LLM；无 LLM 回退到 search_mode。
                可通过 POCKET_QUERY_ROUTER=1 开启
        """
        self.top_k = top_k
        self.use_multihop = use_multihop
        self.use_conversation = use_conversation
        # search_mode 校验：无效值直接报错，避免静默回退到 vector（BUG #7）
        _valid_modes = {"vector", "local", "global", "mix", "kg_only", "global_summary"}
        if search_mode not in _valid_modes:
            raise ValueError(
                f"无效的 search_mode={search_mode!r}，可选值: {sorted(_valid_modes)}"
            )
        self.search_mode = search_mode
        self.data_path = data_path or DATA_PATH
        self.use_pagerank = use_pagerank
        self.pagerank_weight = pagerank_weight
        self.fusion_strategy = (fusion_strategy or FUSION_STRATEGY).lower()
        self.rrf_k = rrf_k or RRF_K
        # HyDE：是否用 LLM 生成假设性文档做向量检索
        self.use_hyde = use_hyde
        # 查询路由器：LLM 自动选择检索模式
        self.use_query_router = use_query_router
        self._query_router: Optional[QueryRouter] = None
        # 答案自检：LLM 生成后校验是否被上下文支持
        self.use_self_check = use_self_check
        # Schema 驱动：关系名归一化，提升 match_relations 召回率
        self.use_schema = use_schema
        self._schema: Optional[RelationSchema] = None
        # 混合检索时向量结果的权重 (0.0~1.0)，KG 结果权重 = 1 - vector_weight
        # 默认 VECTOR_WEIGHT(0.4) 偏向 KG：评测显示纯 KG MRR=0.55 >> 纯向量 0.21
        # 同时影响 weighted 与 rrf 融合策略
        self.vector_weight = VECTOR_WEIGHT
        # reranker 模型缓存（惰性加载）
        self._reranker_model = None

        self.kg_retriever: Optional[KGDualRetriever] = None
        self._pagerank_scores: Optional[dict] = None
        # 社区摘要缓存（global_summary 模式用，惰性加载）
        self._community_data: Optional[dict] = None
        self.conversation = ConversationMemory() if use_conversation else None

        self._load_index()

    def _get_query_router(self) -> QueryRouter:
        """懒加载查询路由器实例"""
        if self._query_router is None:
            self._query_router = QueryRouter(default_mode=self.search_mode)
        return self._query_router

    def _get_schema(self) -> Optional[RelationSchema]:
        """懒加载 RelationSchema 实例"""
        if not self.use_schema:
            return None
        if self._schema is None:
            self._schema = RelationSchema(schema_path=SCHEMA_PATH or None)
        return self._schema

    def _load_index(self):
        """加载 FAISS 索引、Embedding 模型和 KG 检索器"""
        from sentence_transformers import SentenceTransformer

        logger.info("正在加载 Embedding 模型...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("正在加载 FAISS 索引...")
        self.index = FAISSIndex.load(INDEX_DIR, self.model)

        # 非 vector 模式需要加载 KG 检索器
        if self.search_mode != "vector":
            logger.info("正在加载知识图谱检索器...")
            self._load_kg_retriever()

        logger.info("RAG 系统初始化完成")
        self._print_mode()

    def _load_kg_retriever(self):
        """加载 KG 双层检索器

        通过 get_graph_store() 工厂创建图存储抽象层，支持 POCKET_GRAPH_BACKEND
        环境变量切换后端（memory / 未来 neo4j）。当前默认 memory 后端包装
        KGProcessor 的 dict 输出，行为与旧实现完全一致。
        """
        if self.kg_retriever is not None:
            return
        from .core.storages import get_graph_store

        processor = KGProcessor(
            self.data_path,
            reverse_link_relations=REVERSE_LINK_RELATIONS,
            relation_templates=RELATION_TEMPLATES,
            schema=self._get_schema(),
        )
        processor.load_triples()

        # 通过工厂创建 GraphStore，支持 POCKET_GRAPH_BACKEND 环境变量切换后端
        graph_store = get_graph_store(
            entity_relations=dict(processor.entity_relations),
            reverse_relations=dict(processor.reverse_relations),
        )

        self.kg_retriever = KGDualRetriever(
            entity_relations=dict(processor.entity_relations),
            reverse_relations=dict(processor.reverse_relations),
            model=self.model,
            index_dir=INDEX_DIR,
            threshold=ENTITY_SIMILARITY_THRESHOLD,
            n_hops=KG_SEARCH_HOPS,
            graph_store=graph_store,
        )
        logger.info("知识图谱加载完成: %s 条三元组", len(processor.triples))

        if self.use_pagerank:
            logger.info("正在计算 Pagerank...")
            self._pagerank_scores = self.kg_retriever.compute_pagerank()
            logger.info("Pagerank 计算完成: %s 个实体", len(self._pagerank_scores))

    def _load_community_summaries(self, force: bool = False) -> dict:
        """惰性加载社区摘要（global_summary 模式用）。

        根据 COMMUNITY_HIERARCHICAL 配置选择单层或层次模式：
        - 层次模式（默认）：构建多层级社区树，检索时支持 roll-up
        - 单层模式：构建单一分辨率的社区摘要（向后兼容）

        首次调用时会调用 LLM 生成摘要并缓存到 index_dir。
        无 LLM 时回退为实体列表摘要，仍可做实体级召回。
        """
        if self._community_data is not None and not force:
            return self._community_data
        if self.kg_retriever is None:
            self._load_kg_retriever()
        logger.info("正在构建社区摘要（首次可能较慢，会调用 LLM）...")
        if COMMUNITY_HIERARCHICAL:
            logger.info(
                "层次模式: algorithm=%s, resolutions=%s",
                COMMUNITY_ALGORITHM,
                COMMUNITY_RESOLUTIONS,
            )
            self._community_data = build_hierarchical_community_summaries(
                self.kg_retriever,
                self.model,
                INDEX_DIR,
                force=force,
                resolutions=COMMUNITY_RESOLUTIONS,
                max_triples_per_community=COMMUNITY_MAX_TRIPLES,
                algorithm=COMMUNITY_ALGORITHM,
                min_community_size=COMMUNITY_MIN_SIZE,
                max_communities_per_level=COMMUNITY_MAX_PER_LEVEL,
            )
            n_levels = len(self._community_data.get("levels", []))
            n_comm = len(self._community_data.get("communities", []))
            logger.info(
                "层次社区摘要构建完成: %s 层, 最细层 %s 个社区",
                n_levels,
                n_comm,
            )
        else:
            self._community_data = build_community_summaries(
                self.kg_retriever,
                self.model,
                INDEX_DIR,
                force=force,
                resolution=COMMUNITY_RESOLUTION,
                max_triples_per_community=COMMUNITY_MAX_TRIPLES,
                algorithm=COMMUNITY_ALGORITHM,
            )
            n = len(self._community_data.get("communities", []))
            logger.info("单层社区摘要构建完成: %s 个社区", n)
        return self._community_data

    def _get_pagerank(self, entity: str) -> float:
        """获取实体的 Pagerank 分数，归一化到 0-1 区间"""
        # M6 修复：原 `if not self._pagerank_scores` 会把空 dict 也判为"没缓存"
        # 改为 `is None`，区分"未计算"与"计算结果为空"
        if self._pagerank_scores is None:
            return 0.0
        return self._pagerank_scores.get(entity, 0.0)

    def _print_mode(self):
        """打印当前配置"""
        features = []
        if self.use_multihop:
            features.append("Multi-hop")
        if self.search_mode != "vector":
            features.append(f"KG-{self.search_mode}")
        if self.use_conversation:
            features.append("对话记忆")

        provider = get_active_provider()
        feature_str = " + ".join(features) if features else "基础模式"
        logger.info("LLM: %s", provider)
        logger.info("特性: %s", feature_str)

    # ======================
    # 知识检索（search_mode 驱动）
    # ======================

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        vector_weight: Optional[float] = None,
        search_mode: Optional[str] = None,
        use_multihop: Optional[bool] = None,
        use_hyde: Optional[bool] = None,
        use_query_router: Optional[bool] = None,
    ) -> tuple:
        """
        基于 search_mode 的检索入口。

        Args:
            query: 查询文本
            top_k: 最终返回数量
            use_reranker: 是否重排序。None=按模式自动（mix/local/global/global_summary
                默认开启 CrossEncoder 精排，对标 LightRAG）；True/False 强制开/关
            vector_weight: 混合检索时向量结果权重 (0.0~1.0)，None 用实例默认值
            search_mode: 本次检索覆盖 search_mode，None 用实例默认值（H4：避免并发污染）。
                若 use_query_router 为 True，则 search_mode 仅作路由失败时的回退
            use_multihop: 本次检索覆盖 use_multihop，None 用实例默认值
            use_hyde: 本次检索覆盖 use_hyde，None 用实例默认值。
                HyDE 用 LLM 生成假设性文档做向量检索；multihop 模式下忽略（已有查询分解）
            use_query_router: 本次检索覆盖 use_query_router，None 用实例默认值。
                LLM 自动选择检索模式，覆盖 search_mode。需 LLM；无 LLM 回退

        Returns:
            (结果列表, kg_path_info)
        """
        # H4 修复：用局部变量，绝不修改实例属性，避免并发请求互相污染
        effective_multihop = (
            self.use_multihop if use_multihop is None else bool(use_multihop)
        )
        effective_hyde = self.use_hyde if use_hyde is None else bool(use_hyde)
        effective_router = (
            self.use_query_router
            if use_query_router is None
            else bool(use_query_router)
        )

        # 查询路由：开启时 LLM 自动选模式，覆盖 search_mode
        if effective_router:
            effective_search_mode = self._get_query_router().route(query)
        else:
            effective_search_mode = search_mode or self.search_mode

        # Reranker 默认关闭：评测显示 CrossEncoder 会打乱 KG 的优质排序
        # （MRR 从 0.48 降到 0.35），且推理慢 5x。KG 实体级排序已足够精确。
        # 纯向量模式可考虑手动开启 use_reranker=True 补偿向量粗排的不足。
        if use_reranker is None:
            use_reranker = False

        top_k = self.top_k if top_k is None else top_k
        # H4 修复（延续）：vector_weight 也用局部变量，绝不修改实例属性，避免并发污染。
        # 通过参数链 _basic_retrieve → _retrieve_by_entities / _merge_results 透传本次权重。
        if vector_weight is not None:
            effective_vw = max(0.0, min(1.0, float(vector_weight)))
        else:
            effective_vw = self.vector_weight

        # HyDE：仅非 multihop 时生成假设性文档（multihop 已用 LLM 分解查询，避免叠加）
        vector_query = query
        if effective_hyde and not effective_multihop:
            hyde_doc = generate_hypothetical_document(query)
            if hyde_doc:
                vector_query = hyde_doc

        # 多跳查询分解
        if effective_multihop and has_llm():
            # Multi-hop 模式下，收集所有子查询的 KG 路径
            all_kg_paths = []

            def _multihop_retrieve_fn(q, k):
                results, kg_path = self._basic_retrieve(
                    q, k, search_mode=effective_search_mode, vector_weight=effective_vw
                )
                all_kg_paths.append(kg_path)
                return results

            results = multi_hop_retrieve(
                _multihop_retrieve_fn,
                query,
                top_k=top_k,
            )
            # 合并所有子查询的 KG 路径
            merged_path: dict[str, Any] = {
                "seed_entities": [],
                "expanded_entities": [],
                "matched_relations": [],
                "search_type": "multihop",
            }
            for p in all_kg_paths:
                merged_path["seed_entities"].extend(p.get("seed_entities", []))
                merged_path["expanded_entities"].extend(p.get("expanded_entities", []))
                merged_path["matched_relations"].extend(p.get("matched_relations", []))
            # 去重
            merged_path["seed_entities"] = list(set(merged_path["seed_entities"]))
            merged_path["expanded_entities"] = list(
                set(merged_path["expanded_entities"])
            )
            merged_path["matched_relations"] = list(
                set(merged_path["matched_relations"])
            )
            if use_reranker:
                results = self._rerank(query, results, top_k)
            return results, merged_path

        results, kg_path = self._basic_retrieve(
            query,
            top_k,
            search_mode=effective_search_mode,
            vector_query=vector_query,
            vector_weight=effective_vw,
        )
        if use_reranker:
            results = self._rerank(query, results, top_k)
        return results, kg_path

    def _basic_retrieve(
        self,
        query: str,
        top_k: int,
        search_mode: Optional[str] = None,
        vector_query: Optional[str] = None,
        vector_weight: Optional[float] = None,
    ) -> tuple:
        """根据 search_mode 执行检索，返回 (结果列表, kg_path_info)

        Args:
            vector_query: 向量检索用的查询文本（HyDE 时为假设性文档）；
                None 则用原 query。KG 实体/关系匹配始终用原 query
            vector_weight: 本次检索的向量权重，None 用 self.vector_weight
                （H4：参数透传，避免修改实例属性造成并发污染）
        """
        # H4：search_mode 由参数传入，不再读 self.search_mode
        mode = search_mode or self.search_mode
        # 校验 per-call search_mode，避免无效值静默回退到 vector（BUG #7）
        _valid_modes = {"vector", "local", "global", "mix", "kg_only", "global_summary"}
        if mode not in _valid_modes:
            raise ValueError(
                f"无效的 search_mode={mode!r}，可选值: {sorted(_valid_modes)}"
            )
        vector_query = vector_query or query
        vw = self.vector_weight if vector_weight is None else vector_weight
        # H3 修复：vector 模式实例化后切换到 KG 模式时，kg_retriever 可能为 None
        # 此处懒加载，避免 AttributeError 崩溃
        if mode != "vector" and self.kg_retriever is None:
            self._load_kg_retriever()

        kg_path: dict[str, Any] = {
            "seed_entities": [],
            "expanded_entities": [],
            "matched_relations": [],
            "search_type": "",
        }

        if mode == "vector":
            kg_path["search_type"] = "vector"
            return self._tag_result_source_type(
                self.index.search(vector_query, top_k), "vector"
            ), kg_path

        # vector 模式已 early return；以下所有 KG 模式都依赖 kg_retriever
        assert self.kg_retriever is not None

        if mode == "local":
            seed_with_scores = self.kg_retriever.match_entities(
                query, return_scores=True
            )
            seed_entities = [e for e, _ in seed_with_scores]
            max_seed_score = max((s for _, s in seed_with_scores), default=0.0)
            all_entities = self.kg_retriever.local_search(query)
            expanded = [e for e in all_entities if e not in seed_entities]
            kg_path["search_type"] = "local"
            kg_path["seed_entities"] = seed_entities
            kg_path["expanded_entities"] = expanded
            kg_path["max_seed_score"] = max_seed_score
            results = self._retrieve_by_entities(
                query, all_entities, top_k, vector_weight=vw
            )
            return results, kg_path

        elif mode == "global":
            matched_relations = self.kg_retriever.match_relations(query)
            all_entities = self.kg_retriever.global_search(query)
            kg_path["search_type"] = "global"
            kg_path["matched_relations"] = matched_relations
            results = self._retrieve_by_entities(
                query, all_entities, top_k, vector_weight=vw
            )
            return results, kg_path

        elif mode == "mix":
            seed_with_scores = self.kg_retriever.match_entities(
                query, return_scores=True
            )
            seed_entities = [e for e, _ in seed_with_scores]
            max_seed_score = max((s for _, s in seed_with_scores), default=0.0)
            # 关系值反查实体：聚合类问题（如"发病初期用什么药"）的目标实体，
            # 不并入 seed（避免种子过多稀释精确匹配），单独传给 _retrieve_pure_kg 给中间分
            rv_entities = self.kg_retriever.match_entities_by_relation_value(query)
            matched_relations = self.kg_retriever.match_relations(query)
            local_entities = self.kg_retriever.local_search(query)
            global_entities = self.kg_retriever.global_search(query)
            all_kg_entities = list(set(local_entities) | set(global_entities))
            expanded = [e for e in all_kg_entities if e not in seed_entities]

            # PPR 加权：从 seed 出发的个性化 PageRank，让高相关性的邻域实体优先
            # 替代固定 BFS 的无序扩展（expanded 实体排序不再随意）
            ppr_scores = self.kg_retriever.personalized_pagerank(seed_entities)

            kg_path["search_type"] = "mix"
            kg_path["seed_entities"] = seed_entities
            kg_path["expanded_entities"] = expanded
            kg_path["matched_relations"] = matched_relations
            kg_path["relation_value_entities"] = rv_entities
            kg_path["max_seed_score"] = max_seed_score

            vector_results = self._tag_result_source_type(
                self.index.search(vector_query, top_k), "vector"
            )
            focused_entities = list(dict.fromkeys(seed_entities + rv_entities))
            vector_results = self._filter_vector_results(
                query,
                vector_results,
                allowed_entities=focused_entities or all_kg_entities,
            )
            # 用纯 KG 检索（不调向量），避免 vector 成分在两层融合中被重复计算
            # 旧实现调 _retrieve_by_entities 会内部再跑一次向量搜索并融合，
            # 导致 _merge_results 第二层融合时 vector 权重被放大、稀释 KG 命中
            kg_results = self._retrieve_pure_kg(
                all_kg_entities, seed_entities, top_k,
                rv_entities=rv_entities, ppr_scores=ppr_scores,
            )
            return self._merge_results(
                vector_results, kg_results, top_k, vector_weight=vw
            ), kg_path

        elif mode == "kg_only":
            seed_with_scores = self.kg_retriever.match_entities(
                query, return_scores=True
            )
            seed_entities = [e for e, _ in seed_with_scores]
            max_seed_score = max((s for _, s in seed_with_scores), default=0.0)
            rv_entities = self.kg_retriever.match_entities_by_relation_value(query)
            matched_relations = self.kg_retriever.match_relations(query)
            all_entities = self.kg_retriever.mix_search(query)
            expanded = [e for e in all_entities if e not in seed_entities]

            # PPR 加权（同 mix 分支）：邻域扩展实体按 PPR 排序而非 BFS 顺序
            ppr_scores = self.kg_retriever.personalized_pagerank(seed_entities)

            kg_path["search_type"] = "kg_only"
            kg_path["seed_entities"] = seed_entities
            kg_path["expanded_entities"] = expanded
            kg_path["matched_relations"] = matched_relations
            kg_path["relation_value_entities"] = rv_entities
            kg_path["max_seed_score"] = max_seed_score

            results = self._retrieve_pure_kg(
                all_entities, seed_entities, top_k,
                rv_entities=rv_entities, ppr_scores=ppr_scores,
            )
            return results, kg_path

        elif mode == "global_summary":
            # 社区摘要检索：用 query 召回最相关社区的摘要作为上下文
            # 这是 GraphRAG 回答"全局归纳类"问题的核心能力（对标 MS GraphRAG Global Search）
            # 层次模式（默认）下：search_communities 自动走层次检索 + roll-up
            kg_path["search_type"] = "global_summary"
            comm_data = self._load_community_summaries()
            top_communities = search_communities(
                query, self.model, comm_data, top_k=top_k
            )
            kg_path["seed_entities"] = []
            kg_path["expanded_entities"] = []
            for c in top_communities:
                kg_path["seed_entities"].extend(c.get("entities", [])[:5])
            kg_path["matched_relations"] = []
            # 记录层次检索命中的层级分布（便于调试 / observability）
            kg_path["community_levels_hit"] = sorted(
                {c.get("level", 0) for c in top_communities}
            )
            results = [
                (
                    c["summary"],
                    float(c["score"]),
                    {
                        "entity": f"社区#L{c.get('level', 0)}-{c['id']}",
                        "source_type": "community_summary",
                        "level": c.get("level", 0),
                    },
                )
                for c in top_communities
            ]
            return results, kg_path

        else:
            kg_path["search_type"] = "vector"
            return self._tag_result_source_type(
                self.index.search(query, top_k), "vector"
            ), kg_path

    def _retrieve_pure_kg(
        self, entity_names: list, seed_entities: list, top_k: int,
        rv_entities: list = None, ppr_scores: dict = None,
    ) -> list:
        """纯 KG 检索：直接按实体名从索引中查找文本块，不使用向量搜索

        三级分数排序 + PPR 加权：
          - seed_entities（精确匹配 query 的实体）: 2.0
          - rv_entities（关系值反查到的实体，如聚合类问题的目标）: 1.5
          - 其他邻域扩展实体: 1.0 + ppr_score * pagerank_weight
            （PPR 让高相关性的邻域实体优先，替代固定 BFS 的无序扩展）
        走 FAISSIndex.get_chunks_by_entities() 倒排表查表 O(K)，替代旧 O(N) 扫描。
        """
        if not entity_names:
            return []

        seed_set = set(seed_entities)
        rv_set = set(rv_entities or [])

        # PPR top-K 加成：原始 PPR 是 sum=1 的概率分布，单实体 PPR ~1/N 量级，
        # 直接乘 pagerank_weight 后差异 ~1e-4，无法拉开排序。
        # 改用"top-K PPR 实体离散加成"：top-10 得 1.3 分（明显优于普通 expanded 1.0），
        # 让 PPR 真正起到过滤作用。排除种子实体（PPR 最大通常是种子本身）。
        ppr_top = set()
        if ppr_scores:
            ppr_sorted = sorted(
                ppr_scores.items(), key=lambda x: -x[1]
            )
            for e, _ in ppr_sorted:
                if e in seed_set or e in rv_set:
                    continue
                ppr_top.add(e)
                if len(ppr_top) >= 10:
                    break

        # O(K) 查表：K = 匹配到的 chunk 数（通常远小于 N）
        chunks = self.index.get_chunks_by_entities(entity_names)

        # 分两轮收集，避免低相关 expanded 实体（如品种详情）挤进 top-K：
        # 1. 优先收集高置信实体（seed/rv/ppr_top）的 chunk
        # 2. 若不足 top_k，再补充普通 expanded 实体的 chunk
        # 修复痛点：mix 模式 BFS 会扩展出 100+ 实体，品种详情（D优6511 等）
        # 作为普通 expanded 会用 cid tie-breaker 挤进 top-5，稀释防治类答案。
        high_conf = []  # (cid, text, score, meta)
        low_conf = []   # (cid, text, score, meta)
        for cid, text, meta in chunks:
            entity = (meta or {}).get("entity", "")
            if entity in seed_set:
                high_conf.append((cid, text, 2.0, meta))
            elif entity in rv_set:
                high_conf.append((cid, text, 1.5, meta))
            elif entity in ppr_top:
                high_conf.append((cid, text, 1.0 + self.pagerank_weight, meta))
            else:
                low_conf.append((cid, text, 1.0, meta))

        # 排序：score 降序为主，cid 升序为 tie-breaker（确定性 + 保留索引顺序）
        high_conf.sort(key=lambda x: (-x[2], x[0]))
        low_conf.sort(key=lambda x: (-x[2], x[0]))

        # 高置信不够 top_k 才补低置信（避免低相关 expanded 稀释结果）
        if len(high_conf) >= top_k:
            tmp = high_conf[:top_k]
        else:
            tmp = (high_conf + low_conf)[:top_k]

        return self._tag_result_source_type(
            [(text, score, meta) for _cid, text, score, meta in tmp],
            "kg",
        )

    def _retrieve_by_entities(
        self,
        query: str,
        entity_names: list,
        top_k: int,
        vector_weight: Optional[float] = None,
    ) -> list:
        """基于实体名列表从索引中检索文本块

        策略：
        1. 直接从索引中提取匹配实体的文本块（KG 精确匹配）
        2. 再用向量搜索补充
        3. 合并去重，按融合策略排序（RRF 或加权）

        Args:
            vector_weight: 本次检索的向量权重，None 用 self.vector_weight（H4：并发安全）
        """
        if not entity_names:
            return self._tag_result_source_type(
                self.index.search(query, top_k), "vector"
            )

        # 1. 直接提取匹配实体的文本块（按实体在 KG 中的重要性排序）
        # O(K) 查表替代 O(N) metadatas 扫描
        kg_direct = []
        for _cid, text, meta in self.index.get_chunks_by_entities(entity_names):
            ent = (meta or {}).get("entity", "")
            pr_score = self._get_pagerank(ent)
            # KG 精确匹配的基础分，结合 Pagerank
            score = 1.5 + pr_score * self.pagerank_weight
            kg_direct.append((text, score, meta))

        # 按 Pagerank 排序（KG 直接结果不依赖向量相似度）
        kg_direct.sort(key=lambda x: x[1], reverse=True)

        # 2. 向量搜索补充
        vector_results = self._tag_result_source_type(
            self.index.search(query, top_k), "vector"
        )
        vector_results = self._filter_vector_results(
            query, vector_results, allowed_entities=entity_names
        )
        if self.use_pagerank and self._pagerank_scores:
            vector_results = [
                (
                    text,
                    score
                    + self.pagerank_weight
                    * self._get_pagerank(meta.get("entity", ""))
                    * 0.5,
                    meta,
                )
                for text, score, meta in vector_results
            ]

        # 3. 融合两个结果列表（RRF 策略下也用 vector_weight 加权，避免 KG 被向量稀释）
        vw = self.vector_weight if vector_weight is None else vector_weight
        kw = 1.0 - vw
        if self.fusion_strategy == "rrf":
            return self._rrf_fusion(
                [kg_direct, vector_results], top_k, weights=[kw, vw]
            )
        else:
            return self._weighted_fusion(
                [kg_direct, vector_results], top_k, weights=[kw, vw]
            )

    def _merge_results(
        self,
        vector_results: list,
        kg_results: list,
        top_k: int,
        vector_weight: Optional[float] = None,
    ) -> list:
        """合并向量检索和 KG 检索结果，按配置的融合策略排序

        weighted / rrf 策略均按 vector_weight / (1-vector_weight) 对各自来源加权；
        rrf 原先忽略权重（等权融合），现统一生效，避免向量的不相关 top-k 稀释 KG 命中。

        Args:
            vector_weight: 本次检索的向量权重，None 用 self.vector_weight（H4：并发安全）
        """
        vw = self.vector_weight if vector_weight is None else vector_weight
        kw = 1.0 - vw
        if self.fusion_strategy == "rrf":
            return self._rrf_fusion(
                [vector_results, kg_results], top_k, weights=[vw, kw]
            )
        else:
            return self._weighted_fusion(
                [vector_results, kg_results], top_k, weights=[vw, kw]
            )

    def _entity_matches_query(self, query: str, entity: str) -> bool:
        """判断实体名与查询是否明显相关，用于过滤偏题向量结果。

        匹配策略（按优先级）：
        1. 精确子串包含（双向）
        2. focus token 在实体名中
        3. 实体名 token 在 query 中
        4. 中文字符级 n-gram 重叠（≥2字共享核心语素，如"盗梦空间"↔"Inception"共享"盗梦"）
        """
        import re

        normalized_query = (query or "").strip().lower()
        normalized_entity = (entity or "").strip().lower()
        if not normalized_query or not normalized_entity:
            return False
        if normalized_entity in normalized_query or normalized_query in normalized_entity:
            return True

        query_tokens = self._extract_query_focus_terms(normalized_query)
        entity_tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", normalized_entity)
        if any(token in normalized_entity for token in query_tokens):
            return True
        if any(token in normalized_query for token in entity_tokens):
            return True

        if self._chinese_overlap(normalized_query, normalized_entity, min_len=2):
            return True
        return False

    @staticmethod
    def _chinese_overlap(a: str, b: str, min_len: int = 2) -> bool:
        """检查两个中文字符串是否共享长度≥min_len的连续子串。"""
        if len(a) < min_len or len(b) < min_len:
            return False
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        for i in range(len(shorter) - min_len + 1):
            gram = shorter[i : i + min_len]
            if gram in longer:
                return True
        return False

    def _extract_query_focus_terms(self, query: str) -> list[str]:
        """抽取查询里的核心对象词，避免把整句问题当成一个 token。"""
        import re

        normalized = (query or "").strip().lower()
        if not normalized:
            return []

        generic_phrases = (
            "有什么症状",
            "有啥症状",
            "症状是什么",
            "是什么",
            "什么是",
            "什么意思",
            "怎么防治",
            "如何防治",
            "怎么处理",
            "如何处理",
            "怎么",
            "如何",
            "有哪些",
            "包括哪些",
            "适用于哪些",
            "可以防治哪些病害",
            "可以防治哪些",
            "用量是多少",
            "剂量是多少",
            "多少",
            "哪些",
            "症状",
            "表现",
            "特征",
            "特点",
            "用量",
            "剂量",
            "参数",
            "条件",
            "方法",
            "方式",
            "防治",
            "治疗",
            "预防",
            "定义",
            "概念",
            "含义",
            "吗",
            "呢",
            "请问",
        )
        for phrase in generic_phrases:
            normalized = normalized.replace(phrase, " ")

        parts = [
            part.strip()
            for part in re.split(r"[\s,，。！？?、：:；;（）()]+", normalized)
            if len(part.strip()) >= 2
        ]
        if parts:
            return list(dict.fromkeys(parts))

        return re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", (query or "").strip().lower())

    def _matches_focus_terms(self, text: str, focus_terms: list[str]) -> bool:
        normalized_text = (text or "").strip().lower()
        if not normalized_text or not focus_terms:
            return False
        return any(term and term in normalized_text for term in focus_terms)

    def _filter_vector_results(
        self,
        query: str,
        vector_results: list,
        allowed_entities: Optional[list] = None,
    ) -> list:
        """KG 已命中时，尽量过滤与当前实体集合明显无关的向量噪声。"""
        if not vector_results or not allowed_entities:
            return vector_results

        allowed_set = {str(entity or "").strip().lower() for entity in allowed_entities if entity}
        filtered = []

        for text, score, meta in vector_results:
            entity = str((meta or {}).get("entity", "")).strip()
            normalized_entity = entity.lower()
            keep = (
                normalized_entity in allowed_set
                or self._entity_matches_query(query, entity)
                or (not entity and self._entity_matches_query(query, text))
            )
            if keep:
                filtered.append((text, score, meta))

        return filtered or [vector_results[0]]

    def _filter_results_for_question(
        self,
        query: str,
        question_type: str,
        results: list,
    ) -> list:
        """按题型做轻量过滤，降低明显偏题的候选噪声。

        设计原则：
        - 不硬编码任何领域特定关键词，保持框架通用性
        - 核心逻辑依赖 query 焦点词与 result 实体/文本的匹配度
        - 三级分数排序(seed/rv/ppr)已处理大部分排序，这里只做保守过滤
        - 宁可保留少量噪声，也不误杀正确结果（filter为空时返回原results）
        - entity精确匹配的主实体即使含数字也保留（定义/症状描述中温度/用量是正常属性）
        """
        strict_types = ("symptom", "definition", "feature", "method", "parameter", "list")
        if question_type not in strict_types:
            return results
        if not results:
            return results

        focus_terms = self._extract_query_focus_terms(query)
        import re

        kept_items = []
        for text, score, meta in results:
            entity = str((meta or {}).get("entity", "")).strip()
            text_str = str(text or "")
            text_and_entity = f"{entity}\n{text_str}"

            entity_matches = self._entity_matches_query(query, entity)
            focus_hit = self._matches_focus_terms(text_and_entity, focus_terms)
            exact_focus_hit = any(
                term and term == entity.lower() for term in focus_terms
            )

            has_number = bool(
                re.search(r"\d+(?:\.\d+)?\s*(?:克|毫升|mg|g|ml|%|倍|个|天|次|小时|分钟|米|cm|mm)", text_str)
            )

            if question_type == "parameter":
                keep = (entity_matches or focus_hit) and has_number
            elif question_type == "definition":
                if entity_matches:
                    keep = True
                else:
                    keep = focus_hit and not has_number
            elif question_type in ("symptom", "feature"):
                if entity:
                    keep = entity_matches
                else:
                    keep = focus_hit
            elif question_type == "method":
                if entity_matches or exact_focus_hit:
                    keep = True
                elif focus_hit:
                    keep = not has_number
                else:
                    keep = False
            elif question_type == "list":
                if exact_focus_hit and has_number:
                    keep = False
                else:
                    keep = entity_matches or focus_hit
            else:
                keep = entity_matches or focus_hit

            if keep:
                priority = 0
                if question_type == "list":
                    if focus_hit and not exact_focus_hit and not entity_matches:
                        priority += 4
                    if exact_focus_hit:
                        priority -= 2
                if exact_focus_hit:
                    priority += 3
                if entity_matches:
                    priority += 2
                if focus_hit:
                    priority += 1
                if question_type == "parameter" and has_number:
                    priority += 2
                if question_type in ("symptom", "feature"):
                    if not has_number:
                        priority += 3
                    else:
                        priority -= 2
                elif question_type == "definition":
                    if not has_number:
                        priority += 2
                    elif not entity_matches:
                        priority -= 2
                if question_type == "method":
                    if focus_hit and not exact_focus_hit and not has_number:
                        priority += 2
                    if has_number and not entity_matches and not exact_focus_hit:
                        priority -= 2
                kept_items.append((priority, text, score, meta))

        kept_items.sort(
            key=lambda item: (item[0], float(item[2] or 0.0)),
            reverse=True,
        )
        filtered = [(text, score, meta) for _, text, score, meta in kept_items]
        return filtered or results

    def _weighted_fusion(
        self, result_lists: list, top_k: int, weights: Optional[list] = None
    ) -> list:
        """加权融合：对每个来源的分数乘以权重后合并去重排序

        Args:
            result_lists: 多个结果列表，每个列表是 (text, score, meta) 元组
            top_k: 返回 top_k 个结果
            weights: 每个来源的权重列表，None 表示等权

        Returns:
            融合后的结果列表
        """
        if weights is None:
            weights = [1.0] * len(result_lists)
        seen_texts: dict = {}
        for results, w in zip(result_lists, weights):
            for text, score, meta in results:
                adj = score * w
                if text in seen_texts:
                    # 取较高分
                    if adj > seen_texts[text][1]:
                        seen_texts[text] = (text, adj, meta)
                else:
                    seen_texts[text] = (text, adj, meta)
        merged = list(seen_texts.values())
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:top_k]

    def _rrf_fusion(
        self, result_lists: list, top_k: int, weights: Optional[list] = None
    ) -> list:
        """Reciprocal Rank Fusion (RRF) 倒数排位融合

        公式: score(d) = Σ w_i / (k + rank_i(d))

        优势：
        - 不需要归一化不同来源的分数
        - 对排名靠前的结果加权更大
        - 业界公认效果稳定
        - 支持 weights 给不同来源加权（如 KG 权重 > 向量）

        Args:
            result_lists: 多个结果列表，每个列表是 (text, score, meta) 元组
            top_k: 返回 top_k 个结果
            weights: 每个来源的权重，None 表示等权（向后兼容）

        Returns:
            融合后的结果列表
        """
        rrf_scores: dict[str, float] = {}
        text_meta: dict[str, tuple] = {}

        if weights is None:
            weights = [1.0] * len(result_lists)

        for results, w in zip(result_lists, weights):
            for rank, (text, _score, meta) in enumerate(results, 1):
                if text not in rrf_scores:
                    rrf_scores[text] = 0.0
                    text_meta[text] = (text, meta)
                rrf_scores[text] += w / (self.rrf_k + rank)

        # 按 RRF 分数排序
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建结果列表
        merged = []
        for text, score in sorted_items[:top_k]:
            _, meta = text_meta[text]
            merged.append((text, score, meta))

        return merged

    def _rerank(self, query: str, results: list, top_k: int) -> list:
        """对检索结果做重排序，提升最相关结果的排名

        统一委托给 core.reranker.Reranker 单例（全项目共用一个 CrossEncoder 实例）；
        若 CrossEncoder 加载失败则回退到关键词重合度（Jaccard）重排序。

        Args:
            query: 原始查询
            results: 检索结果列表 [(text, score, meta), ...]
            top_k: 重排后返回的数量

        Returns:
            重排后的结果列表
        """
        if not results:
            return results

        # 委托 core.reranker 单例，避免全项目重复加载 CrossEncoder
        from .core.reranker import get_reranker

        reranker = get_reranker()
        model = reranker._load()  # 失败时返回 None，且不会重复重试

        if model is not None:
            try:
                pairs = [[query, text] for text, _, _ in results]
                scores = model.predict(pairs)
                reranked = [
                    (text, float(rscore), meta)
                    for (text, _, meta), rscore in zip(results, scores)
                ]
                reranked.sort(key=lambda x: x[1], reverse=True)
                return reranked[:top_k]
            except Exception as e:
                logger.warning("Reranker 重排出错，保留原排序: %s", e)

        # CrossEncoder 不可用：直接返回原结果。
        # 旧实现回退到 Jaccard 字符重合度重排，但实测会打乱 KG/RRF 的优质排序
        # （评测：MRR 从 0.40 跌到 0.27），弊大于利，故移除。
        return results[:top_k]

    # ======================
    # 硬约束防幻觉：检索为空/低分时强制拒答
    # ======================

    def _should_refuse(
        self, results: list, kg_path: dict, search_mode: str
    ) -> tuple:
        """判断是否应该强制拒答（检索为空或低质量）。

        策略：
        1. 结果为空 → 拒答
        2. KG 模式（local/mix/kg_only）：
           a. seed_entities 和 matched_relations 全空 → 查询未命中 KG 任何实体/关系 → 拒答
           b. seed_entities 不为空但全部来自 embedding 匹配（无精确子串匹配），
              且最高 embedding 相似度 < KG_SEED_REFUSE_THRESHOLD → 查询与 KG 无关 → 拒答
        3. global 模式：matched_relations 和 expanded_entities 全空 → 拒答
        4. 纯向量模式：top1 余弦相似度 < VECTOR_REFUSE_THRESHOLD → 拒答

        Returns:
            (should_refuse, reason) — reason 为空字符串表示不拒答
        """
        if not REFUSE_ON_EMPTY_RETRIEVAL:
            return False, ""

        # 1. 结果为空
        if not results:
            return True, "检索结果为空"

        st = kg_path.get("search_type", "") or search_mode
        has_seed = bool(kg_path.get("seed_entities"))
        has_matched_rel = bool(kg_path.get("matched_relations"))
        has_expanded = bool(kg_path.get("expanded_entities"))

        # 2. KG 模式：实体/关系都没命中 → 查询与 KG 无关
        if st in ("local", "mix", "kg_only"):
            if not has_seed and not has_matched_rel:
                return True, f"{st} 模式未匹配到任何 KG 实体或关系"
            # seed 实体存在但最高匹配分数过低（全部来自低分 embedding 匹配，无精确子串匹配）
            # 阈值 0.65：正常领域查询的 embedding 匹配通常 >= 0.7，完全不相关的查询
            # 噪声匹配通常在 0.5~0.6 之间
            max_seed_score = kg_path.get("max_seed_score", 1.0)
            if has_seed and max_seed_score < 0.65:
                return True, (
                    f"{st} 模式 seed 实体最高匹配分数 {max_seed_score:.3f} 过低，"
                    f"查询可能与知识库无关"
                )
        # 3. global 模式：关系和扩展实体都没命中
        elif st == "global":
            if not has_matched_rel and not has_expanded:
                return True, "global 模式未匹配到任何 KG 关系"
        # 4. 纯向量模式：top1 相似度过低
        elif st == "vector":
            top_score = float(results[0][1]) if results else 0.0
            if top_score < VECTOR_REFUSE_THRESHOLD:
                return True, (
                    f"vector 模式 top1 相似度 {top_score:.3f} "
                    f"< 阈值 {VECTOR_REFUSE_THRESHOLD}"
                )

        return False, ""

    def _classify_failure_bucket(self, results: list, kg_path: dict, search_mode: str) -> str:
        """将失败场景归类为可解释的 failure bucket。"""
        if not results:
            return "empty_retrieval"

        st = kg_path.get("search_type", "") or search_mode
        has_seed = bool(kg_path.get("seed_entities"))
        has_matched_rel = bool(kg_path.get("matched_relations"))
        has_expanded = bool(kg_path.get("expanded_entities"))

        if st in ("local", "mix", "kg_only") and not has_seed and not has_matched_rel:
            return "no_entity_or_relation_hit"
        if st == "global" and not has_matched_rel and not has_expanded:
            return "insufficient_relation_context"
        if st == "vector":
            return "low_vector_similarity"
        return "insufficient_context"

    def _classify_question_type(self, question: str) -> str:
        """按问题意图做粗粒度分类，完全解耦具体领域。"""
        q = (question or "").strip()
        if any(k in q for k in ("区别", "对比", "不同", "哪个好")):
            return "comparison"
        if any(k in q for k in ("症状", "病症", "病状", "危害症状", "发病症状")):
            return "symptom"
        if any(k in q for k in ("特征", "特点", "表现", "现象")):
            return "feature"
        if any(k in q for k in ("有哪些", "包括哪些", "分类", "种类", "哪些")):
            return "list"
        if any(k in q for k in ("多少", "参数", "条件", "指标", "数值", "用量", "剂量", "浓度")):
            return "parameter"
        if any(k in q for k in ("怎么", "如何", "步骤", "方法", "方式", "处理", "防治", "治疗", "预防")):
            return "method"
        if any(k in q for k in ("是什么", "定义", "概念", "含义", "什么是")):
            return "definition"
        return "generic"

    def _build_answer_prompt(
        self,
        question: str,
        context: str,
        question_type: str,
    ) -> str:
        """为不同问法补一层轻量表达提示，减少资料整理腔。"""
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

        style_hint = ""
        if question_type == "symptom":
            style_hint = (
                "\n\n补充要求：优先直接描述表现，像正常专家回答一样自然组织句子。"
                '不要用"症状包括""根据提供的信息"这类资料整理式开头，'
                "也不要混入防治方法、用药建议或无关补充。"
            )
        elif question_type == "parameter":
            style_hint = (
                "\n\n补充要求：如果知识没有直接给出参数或用量数值，先明确说明"
                "现有信息没有直接给出该数值；如果只有相关线索，再单独补一句"
                "当前能确认的关联信息。不要把弱关联写成确定答案，也不要给出"
                '“请查说明书”“咨询专家”这类空泛建议。'
            )
        elif question_type == "definition":
            style_hint = (
                "\n\n补充要求：先用一句话直接解释对象是什么，再补充必要事实。"
                "保持简洁，不要机械分点。"
            )

        return user_prompt + style_hint

    def _render_standardized_answer(
        self,
        question_type: str,
        answer_text: str,
    ) -> str:
        """渲染更自然的答案：优先保留原答，只做最小清洗。"""
        cleaned_answer = self._cleanup_answer_text(answer_text)
        return cleaned_answer if cleaned_answer else answer_text

    def _cleanup_answer_text(self, answer_text: str) -> str:
        """对带引用的原答做最小清洗，保留自然表达。"""
        import re

        if not isinstance(answer_text, str):
            return answer_text

        header_lines = {"结论", "关键事实", "条件 / 限制", "条件/限制"}
        document_prefixes = (
            "根据提供的信息",
            "根据上述信息",
            "从提供的信息来看",
            "从上述信息来看",
        )
        inline_replacements = (
            ("不过，根据提供的信息，", "不过，"),
            ("但根据提供的信息，", "但"),
            ("根据提供的信息，", ""),
            ("根据上述信息，", ""),
            ("从提供的信息来看，", ""),
            ("从上述信息来看，", ""),
        )
        kept_lines: list[str] = []
        seen_lines: set[str] = set()

        for raw_line in answer_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            normalized_header = line.lstrip("#").strip()
            normalized_header = normalized_header.lstrip("-*").strip()
            normalized_header = normalized_header.rstrip("：:")
            if normalized_header in header_lines:
                continue

            for prefix in document_prefixes:
                if line.startswith(prefix):
                    trimmed = line[len(prefix) :].lstrip("，,:： ")
                    if trimmed:
                        line = trimmed
                    break

            for old, new in inline_replacements:
                line = line.replace(old, new)
            line = re.sub(r"^(.{1,40}?)的症状包括", r"\1常见表现为", line)
            line = line.replace("症状包括", "常见表现为")
            line = re.sub(r"([，。；！？])\1+", r"\1", line)

            if not _extract_citation_ids(line) and any(
                phrase in line
                for phrase in ("请参考", "咨询", "建议咨询", "说明书", "获得准确数据")
            ):
                continue

            if line in seen_lines:
                continue
            seen_lines.add(line)
            kept_lines.append(line)

        return "\n".join(kept_lines).strip()

    def _standardize_answer(
        self,
        question: str,
        answer_text: str,
        results: list,
    ) -> tuple[str, str]:
        """把带有效引用的回答整理成统一结构。"""
        question_type = self._classify_question_type(question)
        rendered = self._render_standardized_answer(
            question_type, answer_text
        )
        return question_type, rendered

    def _looks_like_empty_llm_answer(self, answer_text: Any) -> bool:
        """识别空回答或泛化拒答，触发结构化检索兜底。"""
        if answer_text is None:
            return True
        if not isinstance(answer_text, str):
            return False

        normalized = answer_text.strip()
        generic_refusals = {
            "",
            "知识库中未找到相关信息。",
            "未找到相关信息。",
            "抱歉，我无法从知识库中找到相关信息。",
        }
        return normalized in generic_refusals

    def _answer_has_valid_citations(self, answer_text: Any, results: list) -> bool:
        """检查回答是否至少包含一个有效引用编号。"""
        if not isinstance(answer_text, str) or not answer_text.strip():
            return False

        citation_ids = _extract_citation_ids(answer_text)
        if not citation_ids:
            return False

        max_allowed = len(results)
        return any(1 <= int(cid) <= max_allowed for cid in citation_ids)

    def _format_retrieval_fallback(self, results: list, kg_path: dict) -> str:
        """当 LLM 输出过弱时，用自然短答展示当前可确认的信息。"""
        if not results:
            return "知识库中未找到相关信息。"

        summary_lines = []
        for idx, (text, score, meta) in enumerate(results[:3], 1):
            entity = str((meta or {}).get("entity", "")).strip() or f"来源{idx}"
            snippet = " ".join(str(text or "").split())
            if not snippet:
                continue
            summary_lines.append(f"{entity}：{snippet}[{idx}]")

        if not summary_lines:
            return "知识库中未找到相关信息。"

        return (
            "我目前只能根据命中的知识确认这些信息：\n"
            + "\n".join(summary_lines)
            + "\n如果你想要更完整的回答，可以把问题再问得更具体一点。"
        )

    def _tag_result_source_type(self, results: list, source_type: str) -> list:
        """为检索结果补充来源类型，避免 UI/CLI 无法解释来源归属。"""
        tagged = []
        for text, score, meta in results:
            meta_copy = dict(meta or {})
            meta_copy.setdefault("source_type", source_type)
            tagged.append((text, score, meta_copy))
        return tagged

    # ======================
    # 上下文构建
    # ======================

    def _build_context(self, results: list) -> str:
        """将检索结果格式化为 Prompt 上下文。

        每条知识用 [1] [2] ... 编号，与 sources 列表顺序一致；
        LLM 据此在回答中用 [1][2] 标注来源，实现 citation 回溯。
        """
        context_parts = []
        for i, (text, score, meta) in enumerate(results, 1):
            context_parts.append(
                f"[{i}] 来源: {meta['entity']}（相关度: {score:.2f}）\n{text}"
            )
        return "\n\n".join(context_parts)

    def _promote_relation_target_results(
        self,
        query: str,
        question_type: str,
        results: list,
        kg_path: dict,
        top_k: int,
    ) -> list:
        """对“哪些病害”这类问题优先抬升关系值反查出来的目标实体。"""
        if question_type != "list":
            return results
        if not any(k in (query or "") for k in ("哪些病", "哪些病害", "防治哪些")):
            return results
        if any("病" in str((meta or {}).get("entity", "")) for _, _, meta in results):
            return results

        rv_entities = list(dict.fromkeys(kg_path.get("relation_value_entities") or []))
        if not rv_entities:
            return results
        effective_top_k = self.top_k if top_k is None else top_k

        promoted = self._retrieve_pure_kg(
            rv_entities,
            seed_entities=rv_entities,
            top_k=min(effective_top_k, len(rv_entities)),
        )
        if not promoted:
            return results

        merged = []
        seen_texts = set()
        for item in promoted + results:
            text = item[0]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            merged.append(item)
            if len(merged) >= effective_top_k:
                break
        return merged

    # ======================
    # 核心问答
    # ======================

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        vector_weight: Optional[float] = None,
        search_mode: Optional[str] = None,
        use_multihop: Optional[bool] = None,
        use_hyde: Optional[bool] = None,
        use_query_router: Optional[bool] = None,
        use_self_check: Optional[bool] = None,
    ) -> dict:
        """
        RAG 问答主入口。

        完整流程：
        1. 对话改写：基于历史上下文补全追问
        2. 检索：search_mode 驱动的双层检索
        3. Prompt 构建：拼接检索结果
        4. LLM 生成：调用大模型生成回答
        5. 答案自检：校验是否被上下文支持，发现幻觉加警告并重试（可选）
        6. 更新对话记忆

        Args:
            question: 用户问题
            top_k: 检索数量，默认使用配置值
            search_mode: 覆盖检索模式（H4：避免并发污染实例属性）
            use_multihop: 覆盖多跳开关
            use_hyde: 覆盖 HyDE 开关（用 LLM 生成假设性文档做向量检索）
            use_query_router: 覆盖查询路由开关（LLM 自动选检索模式）
            use_self_check: 覆盖答案自检开关（LLM 校验答案是否被上下文支持）

        Returns:
            dict: {answer, sources, question, pipeline_info}
        """
        # 空/空白 query 直接拒答，避免触发无意义 LLM 调用（BUG #5）
        if not question or not question.strip():
            return {
                "answer": "请提供有效的问题。",
                "sources": [],
                "question": question or "",
                "pipeline_info": {
                    "multihop_used": False,
                    "search_mode": None,
                    "kg_entities_matched": 0,
                    "query_rewritten": False,
                    "hyde_used": False,
                    "query_routed": False,
                    "self_check_used": False,
                    "self_check_level": "skipped",
                    "top_k": self.top_k if top_k is None else top_k,
                    "response_mode": "refused",
                    "failure_bucket": "empty_query",
                    "fallback_reason": "查询为空",
                    "question_type": None,
                },
            }
        effective_multihop = (
            self.use_multihop if use_multihop is None else bool(use_multihop)
        )
        effective_hyde = self.use_hyde if use_hyde is None else bool(use_hyde)
        effective_router = (
            self.use_query_router
            if use_query_router is None
            else bool(use_query_router)
        )
        effective_self_check = (
            self.use_self_check if use_self_check is None else bool(use_self_check)
        )
        # H4：vector_weight 用局部变量，绝不修改 self.vector_weight
        effective_vw = (
            self.vector_weight
            if vector_weight is None
            else max(0.0, min(1.0, float(vector_weight)))
        )

        pipeline_info: dict[str, Any] = {
            "multihop_used": False,
            "search_mode": None,  # 路由后回填
            "kg_entities_matched": 0,
            "query_rewritten": False,
            "hyde_used": False,
            "query_routed": False,
            "self_check_used": False,
            "self_check_level": "skipped",
            "top_k": self.top_k if top_k is None else top_k,
                    "response_mode": "llm",
            "failure_bucket": None,
            "fallback_reason": None,
            "question_type": None,
        }

        # Step 1: 对话改写（多轮上下文补全）
        effective_query = question
        if self.conversation and self.conversation.history and has_llm():
            rewritten = self.conversation.rewrite_query(question, call_llm)
            if rewritten != question:
                effective_query = rewritten
                pipeline_info["query_rewritten"] = True
        question_type_hint = self._classify_question_type(effective_query)
        pipeline_info["question_type"] = question_type_hint

        # Step 2: 检索（H4：透传，不修改实例属性）
        results, kg_path = self.retrieve(
            effective_query,
            top_k,
            use_reranker=use_reranker,
            vector_weight=vector_weight,
            search_mode=search_mode,
            use_multihop=effective_multihop,
            use_hyde=effective_hyde,
            use_query_router=effective_router,
        )
        results = self._filter_results_for_question(
            effective_query, question_type_hint, results
        )
        results = self._promote_relation_target_results(
            effective_query, question_type_hint, results, kg_path, top_k
        )

        pipeline_info["kg_path"] = kg_path
        pipeline_info["kg_entities_matched"] = len(kg_path.get("seed_entities", []))
        pipeline_info["reranker_used"] = use_reranker
        pipeline_info["vector_weight"] = effective_vw
        pipeline_info["hyde_used"] = (
            effective_hyde and not effective_multihop and has_llm()
        )
        # 路由后的实际检索模式 + 是否走了路由
        pipeline_info["search_mode"] = kg_path.get("search_type") or self.search_mode
        pipeline_info["query_routed"] = effective_router and has_llm()

        # 硬约束防幻觉：检索为空/低分时强制拒答，跳过 LLM
        refused, refuse_reason = self._should_refuse(
            results, kg_path, pipeline_info["search_mode"]
        )
        if refused:
            logger.info("强制拒答：%s", refuse_reason)
            pipeline_info["refused"] = True
            pipeline_info["refuse_reason"] = refuse_reason
            pipeline_info["failure_bucket"] = self._classify_failure_bucket(
                results, kg_path, pipeline_info["search_mode"]
            )
            if self.conversation:
                self.conversation.add("user", question)
                self.conversation.add("assistant", REFUSE_ANSWER_TEXT)
            return {
                "answer": REFUSE_ANSWER_TEXT,
                "sources": [],
                "question": question,
                "effective_query": effective_query,
                "pipeline_info": pipeline_info,
            }

        # Step 3: 构建 Prompt
        context = self._build_context(results)

        system_prompt = SYSTEM_PROMPT
        user_prompt = self._build_answer_prompt(
            effective_query, context, question_type_hint
        )

        # Step 4: LLM 生成
        answer_text = None
        if has_llm():
            try:
                answer_text = call_llm(system_prompt, user_prompt)
            except Exception as e:
                logger.warning("LLM 调用失败，回退到检索模式: %s", e)
                answer_text = None
                pipeline_info["llm_error"] = str(e)[:200]

        # 无 LLM 时返回纯检索结果
        if answer_text is None:
            answer_text = self._retrieval_only_answer(results)
            pipeline_info["response_mode"] = "retrieval_only"
        elif self._looks_like_empty_llm_answer(answer_text):
            answer_text = self._format_retrieval_fallback(results, kg_path)
            pipeline_info["response_mode"] = "retrieval_fallback"
            pipeline_info["fallback_reason"] = "llm_empty_or_generic"
        elif not self._answer_has_valid_citations(answer_text, results):
            answer_text = self._format_retrieval_fallback(results, kg_path)
            pipeline_info["response_mode"] = "retrieval_fallback"
            pipeline_info["fallback_reason"] = "missing_citations"
        else:
            question_type, answer_text = self._standardize_answer(
                effective_query, answer_text, results
            )
            pipeline_info["question_type"] = question_type
            pipeline_info["response_mode"] = "llm_standardized"

        # Step 5: 答案自检（LLM 校验是否被上下文支持，发现幻觉加警告 + 重试 1 次）
        warning_prefix = ""
        if effective_self_check and has_llm() and answer_text:
            verification = verify_answer(answer_text, context)
            pipeline_info["self_check_used"] = True
            pipeline_info["self_check_level"] = verification.get(
                "support_level", "full"
            )
            pipeline_info["self_check_unsupported"] = verification.get(
                "unsupported_sentences", []
            )

            level = verification.get("support_level", "full")
            if level in ("partial", "none"):
                # 幻觉：加警告前缀；none 时尝试重试 1 次（强化 prompt 约束）
                warning_prefix = build_warning_prefix(verification)
                if level == "none":
                    logger.info("答案自检判定 none(完全编造)，强化约束重试 1 次")
                    retry_prompt = user_prompt + (
                        "\n\n注意：请严格基于上述知识信息回答，"
                        "不要编造知识库中没有的内容。如确实无相关信息，请直接说明。"
                    )
                    retry_answer = call_llm(system_prompt, retry_prompt)
                    if retry_answer and retry_answer.strip():
                        answer_text = retry_answer
                        # 复检一次（用重试后的答案）
                        verification2 = verify_answer(answer_text, context)
                        pipeline_info["self_check_level"] = verification2.get(
                            "support_level", level
                        )
                        pipeline_info["self_check_unsupported"] = verification2.get(
                            "unsupported_sentences", []
                        )
                        warning_prefix = build_warning_prefix(verification2)

        if warning_prefix:
            answer_text = warning_prefix + answer_text

        # Step 6: 更新对话记忆
        if self.conversation:
            self.conversation.add("user", question)
            # 记忆里存原文（不带警告前缀），避免前缀污染后续对话改写
            self.conversation.add(
                "assistant",
                answer_text.split("\n\n", 1)[-1] if warning_prefix else answer_text,
            )

        sources = [
            {
                "entity": meta["entity"],
                "score": float(score),
                "text": text,
                "source_type": meta.get("source_type", "vector"),
            }
            for text, score, meta in results
        ]

        return {
            "answer": answer_text,
            "sources": sources,
            "question": question,
            "effective_query": effective_query,
            "pipeline_info": pipeline_info,
        }

    # ask 作为 answer 的别名（BUG #1：文档/用户期望 ask() 可用）
    ask = answer

    def answer_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        vector_weight: Optional[float] = None,
        search_mode: Optional[str] = None,
        use_multihop: Optional[bool] = None,
        use_hyde: Optional[bool] = None,
        use_query_router: Optional[bool] = None,
        use_self_check: Optional[bool] = None,
    ):
        """
        流式 RAG 问答主入口。
        返回一个生成器，每次 yield 一个字典，包含当前状态或生成的文本片段。

        注意：自检在流式结束后做（需要完整答案），自检结果通过 self_check 字段 yield。
        """
        effective_multihop = (
            self.use_multihop if use_multihop is None else bool(use_multihop)
        )
        effective_hyde = self.use_hyde if use_hyde is None else bool(use_hyde)
        effective_router = (
            self.use_query_router
            if use_query_router is None
            else bool(use_query_router)
        )
        effective_self_check = (
            self.use_self_check if use_self_check is None else bool(use_self_check)
        )
        # H4：vector_weight 用局部变量，绝不修改 self.vector_weight
        effective_vw = (
            self.vector_weight
            if vector_weight is None
            else max(0.0, min(1.0, float(vector_weight)))
        )

        pipeline_info: dict[str, Any] = {
            "multihop_used": False,
            "search_mode": None,  # 路由后回填
            "kg_entities_matched": 0,
            "query_rewritten": False,
            "hyde_used": False,
            "query_routed": False,
            "self_check_used": False,
            "self_check_level": "skipped",
            "top_k": self.top_k if top_k is None else top_k,
                    "response_mode": "llm",
            "failure_bucket": None,
            "fallback_reason": None,
            "question_type": None,
        }

        yield {"status": "正在理解问题..."}

        # Step 1: 对话改写
        effective_query = question
        if self.conversation and self.conversation.history and has_llm():
            rewritten = self.conversation.rewrite_query(question, call_llm)
            if rewritten != question:
                effective_query = rewritten
                pipeline_info["query_rewritten"] = True
        question_type_hint = self._classify_question_type(effective_query)
        pipeline_info["question_type"] = question_type_hint

        yield {"status": "正在检索知识库..."}

        # Step 2: 检索（H4：透传，不修改实例属性）
        results, kg_path = self.retrieve(
            effective_query,
            top_k,
            use_reranker=use_reranker,
            vector_weight=vector_weight,
            search_mode=search_mode,
            use_multihop=effective_multihop,
            use_hyde=effective_hyde,
            use_query_router=effective_router,
        )
        results = self._filter_results_for_question(
            effective_query, question_type_hint, results
        )
        results = self._promote_relation_target_results(
            effective_query, question_type_hint, results, kg_path, top_k
        )

        pipeline_info["kg_path"] = kg_path
        pipeline_info["kg_entities_matched"] = len(kg_path.get("seed_entities", []))
        pipeline_info["reranker_used"] = use_reranker
        pipeline_info["vector_weight"] = effective_vw
        pipeline_info["hyde_used"] = (
            effective_hyde and not effective_multihop and has_llm()
        )
        pipeline_info["search_mode"] = kg_path.get("search_type") or self.search_mode
        pipeline_info["query_routed"] = effective_router and has_llm()

        # 硬约束防幻觉：检索为空/低分时强制拒答，跳过 LLM
        refused, refuse_reason = self._should_refuse(
            results, kg_path, pipeline_info["search_mode"]
        )
        if refused:
            logger.info("强制拒答：%s", refuse_reason)
            pipeline_info["refused"] = True
            pipeline_info["refuse_reason"] = refuse_reason
            pipeline_info["failure_bucket"] = self._classify_failure_bucket(
                results, kg_path, pipeline_info["search_mode"]
            )
            refused_sources = [
                {
                    "entity": meta.get("entity", ""),
                    "score": float(score),
                    "text": text,
                    "source_type": meta.get("source_type", "vector"),
                }
                for text, score, meta in results
            ]
            yield {
                "chunk": REFUSE_ANSWER_TEXT,
                "full_answer": REFUSE_ANSWER_TEXT,
                "sources": refused_sources,
                "pipeline_info": pipeline_info,
                "effective_query": effective_query,
                "refused": True,
            }
            if self.conversation:
                self.conversation.add("user", question)
                self.conversation.add("assistant", REFUSE_ANSWER_TEXT)
            yield {
                "done": True,
                "answer": REFUSE_ANSWER_TEXT,
                "sources": refused_sources,
                "question": question,
                "pipeline_info": pipeline_info,
                "effective_query": effective_query,
                "refused": True,
            }
            return

        # Step 3: 构建 Prompt
        context = self._build_context(results)

        system_prompt = SYSTEM_PROMPT
        user_prompt = self._build_answer_prompt(
            effective_query, context, question_type_hint
        )

        sources = [
            {
                "entity": meta["entity"],
                "score": float(score),
                "text": text,
                "source_type": meta.get("source_type", "vector"),
            }
            for text, score, meta in results
        ]

        yield {
            "status": "正在生成回答...",
            "sources": sources,
            "pipeline_info": pipeline_info,
            "effective_query": effective_query,
        }

        # Step 4: LLM 生成 (流式)
        full_answer = ""
        is_retrieval_only = False
        llm_streamed_chunks = False
        if has_llm():
            try:
                stream_gen = call_llm(system_prompt, user_prompt, stream=True)
                if stream_gen:
                    for chunk in stream_gen:
                        full_answer += chunk
                        llm_streamed_chunks = True
                        yield {"chunk": chunk, "full_answer": full_answer}
                else:
                    is_retrieval_only = True
                    pipeline_info["llm_error"] = "LLM 返回空流"
            except Exception as e:
                logger.warning("LLM 调用失败，回退到检索模式: %s", e)
                is_retrieval_only = True
                pipeline_info["llm_error"] = str(e)[:200]
        else:
            is_retrieval_only = True

        if is_retrieval_only:
            full_answer = self._retrieval_only_answer(results)
            pipeline_info["response_mode"] = "retrieval_only"
            yield {
                "chunk": full_answer,
                "full_answer": full_answer,
                "pipeline_info": pipeline_info,
            }
        elif self._looks_like_empty_llm_answer(full_answer):
            full_answer = self._format_retrieval_fallback(results, kg_path)
            pipeline_info["response_mode"] = "retrieval_fallback"
            pipeline_info["fallback_reason"] = "llm_empty_or_generic"
            yield {
                "chunk": full_answer,
                "full_answer": full_answer,
                "pipeline_info": pipeline_info,
            }
        elif not self._answer_has_valid_citations(full_answer, results):
            full_answer = self._format_retrieval_fallback(results, kg_path)
            pipeline_info["response_mode"] = "retrieval_fallback"
            pipeline_info["fallback_reason"] = "missing_citations"
            yield {
                "chunk": full_answer,
                "full_answer": full_answer,
                "pipeline_info": pipeline_info,
            }
        else:
            question_type, full_answer = self._standardize_answer(
                effective_query, full_answer, results
            )
            pipeline_info["question_type"] = question_type
            pipeline_info["response_mode"] = "llm_standardized"
            if not llm_streamed_chunks:
                yield {
                    "chunk": full_answer,
                    "full_answer": full_answer,
                    "pipeline_info": pipeline_info,
                }
            else:
                yield {
                    "full_answer": full_answer,
                    "pipeline_info": pipeline_info,
                }

        # Step 5: 答案自检（流式结束后做，需要完整答案）
        if effective_self_check and has_llm() and full_answer:
            yield {"status": "正在自检答案可靠性..."}
            verification = verify_answer(full_answer, context)
            pipeline_info["self_check_used"] = True
            pipeline_info["self_check_level"] = verification.get(
                "support_level", "full"
            )
            pipeline_info["self_check_unsupported"] = verification.get(
                "unsupported_sentences", []
            )
            yield {"self_check": verification, "pipeline_info": pipeline_info}

        # Step 6: 更新对话记忆
        if self.conversation:
            self.conversation.add("user", question)
            self.conversation.add("assistant", full_answer)

        # Step 7: yield 最终 done 事件，包含完整结果（与非流式 answer() 返回格式对齐）
        yield {
            "done": True,
            "answer": full_answer,
            "sources": sources,
            "question": question,
            "pipeline_info": pipeline_info,
            "effective_query": effective_query,
        }

    def _retrieval_only_answer(self, results: list) -> str:
        """无 LLM 时的纯检索结果展示"""
        # Score 归一化展示：RRF 融合后分数常是 0.01 量级，用户看"相似度: 0.0162"
        # 会以为系统坏了。改为相对百分比：top 结果 100%，其他按比例。
        # 仅影响展示，不影响实际排序。
        max_score = max((s for _, s, _ in results), default=0.0)
        answer = "## 检索到的相关知识\n\n"
        for i, (text, score, meta) in enumerate(results, 1):
            pct = (score / max_score * 100) if max_score > 0 else 0.0
            answer += f"### [{i}] {meta['entity']}（相关度: {pct:.0f}%）\n"
            answer += f"{text}\n\n"
        answer += "---\n"
        answer += "提示：当前为纯检索模式。设置 SILICONFLOW_API_KEY、DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量可启用大模型生成回答。"
        return answer

    def reset_conversation(self):
        """重置对话历史"""
        if self.conversation:
            self.conversation.clear()
