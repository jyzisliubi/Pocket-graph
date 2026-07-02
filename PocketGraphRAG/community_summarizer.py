"""社区摘要生成（GraphRAG 全局归纳问答核心能力）

参考 MS GraphRAG 的核心思路：用社区发现算法分社区 → LLM 为每个社区生成摘要 →
问答时用 query 与社区摘要做向量匹配，召回最相关的社区摘要作为上下文。

这让系统能回答"全局归纳类"问题，例如"水稻最常见的三类病害是什么"——
这类问题无法靠单实体检索命中，需要社区级聚合知识。

支持两种模式：
- 单层模式（默认，向后兼容）：Louvain 分一层社区，每个社区生成摘要
- 层次模式（P3 新增）：多分辨率 Leiden/Louvain 聚类，构建父子社区树，
  检索时可跨层级 drill-down / roll-up，对标 MS GraphRAG 的 Hierarchical Summaries

设计要点：
- 惰性生成：首次进入 global_summary 模式且 has_llm 时才构建
- 持久化缓存：摘要 + embedding 存到 index_dir/community_summaries.json，避免重复调用 LLM
- 轻量召回：社区数通常 < 100，用 numpy 余弦相似度全量计算即可，无需额外 FAISS 索引
- 优雅降级：无 LLM 时回退为"社区实体列表"作为摘要，仍可做实体级召回
- 算法降级：leidenalg 未安装时自动回退到 networkx 内置 Louvain
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

from .llm import call_llm, has_llm
from .logging_config import get_logger

logger = get_logger(__name__)

COMMUNITY_SUMMARY_FILE = "community_summaries.json"
HIERARCHICAL_COMMUNITY_FILE = "community_summaries_hierarchical.json"

COMMUNITY_SUMMARY_PROMPT = """你是知识图谱分析专家。下面是知识图谱中一个社区（紧密关联的实体集合）的相关三元组。

请基于这些三元组，生成一段 200-400 字的中文摘要，概括这个社区讨论的核心主题、关键实体、主要关系和重要结论。

社区实体（共 {n_entities} 个）:
{entities}

相关三元组（共 {n_triples} 条）:
{triples}

要求:
1. 只基于给定三元组，不要编造未出现的信息
2. 首句点明该社区的核心主题（如"本社区聚焦稻瘟病的症状识别与化学防治"）
3. 列出关键实体和它们之间的关系
4. 语言简洁、信息密度高，200-400 字
摘要:"""

PARENT_COMMUNITY_SUMMARY_PROMPT = """你是知识图谱分析专家。下面是一个父社区及其下属子社区的摘要。

请基于子社区的摘要，生成一段 200-400 字的中文摘要，概括父社区的整体主题、覆盖的子主题、关键实体和它们之间的关系。

父社区层级: Level {level} (共 {n_entities} 个实体, {n_children} 个子社区)

子社区摘要:
{child_summaries}

要求:
1. 综合子社区摘要，不要遗漏重要信息
2. 首句点明父社区的整体主题
3. 列出涵盖的子主题和关键实体
4. 语言简洁、信息密度高，200-400 字
父社区摘要:"""


# ==========================
# Leiden 算法支持（可选，未安装则回退 Louvain）
# ==========================

def _is_leiden_available() -> bool:
    """检查 leidenalg + igraph 是否可用"""
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_communities_leiden(
    kg_retriever,
    resolution: float = 1.0,
    seed: int = 42,
) -> tuple:
    """用 Leiden 算法做社区发现。

    Leiden 相比 Louvain 的优势：
    - 保证社区连通性（Louvain 可能产生 disconnected communities）
    - 层次化 refinement，质量更稳定
    - 是 MS GraphRAG / nano-graphrag 的默认算法

    Args:
        kg_retriever: KGDualRetriever 实例
        resolution: 分辨率参数，越大社区越小越多
        seed: 随机种子

    Returns:
        (communities, community_map):
          communities = [[entity, ...], ...]
          community_map = {entity: community_id}
    """
    import igraph as ig
    import leidenalg

    if not kg_retriever.all_entities:
        return [], {}

    entities = list(kg_retriever.all_entities)
    ent_idx = {e: i for i, e in enumerate(entities)}

    # 构建 igraph 无向图
    edges = set()
    for head, rels in kg_retriever.entity_relations.items():
        if head not in ent_idx:
            continue
        h_i = ent_idx[head]
        for _, tail in rels:
            if tail in ent_idx and tail != head:
                t_i = ent_idx[tail]
                if h_i < t_i:
                    edges.add((h_i, t_i))
                else:
                    edges.add((t_i, h_i))

    if not edges:
        return [[e] for e in entities], {e: i for i, e in enumerate(entities)}

    g = ig.Graph(n=len(entities), edges=list(edges), directed=False)

    # RBConfigurationVertexPartition：支持 resolution 参数（Traag et al. 2013）
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )

    communities = []
    for member_indices in partition:
        members = sorted([entities[i] for i in member_indices])
        communities.append(members)

    community_map = {}
    for cid, members in enumerate(communities):
        for ent in members:
            community_map[ent] = cid

    logger.info(
        "Leiden 发现 %s 个社区 (resolution=%.3f)，最大社区 %s 实体，质量=%.3f",
        len(communities),
        resolution,
        max(len(c) for c in communities) if communities else 0,
        partition.quality(),
    )
    return communities, community_map


def _detect_communities_at_resolution(
    kg_retriever,
    resolution: float,
    algorithm: str = "auto",
    seed: int = 42,
) -> tuple:
    """在指定分辨率下做社区发现，按 algorithm 选择算法。

    Args:
        algorithm: "leiden" 强制 Leiden, "louvain" 强制 Louvain, "auto" 优先 Leiden

    Returns:
        (communities, community_map)
    """
    if algorithm == "leiden" and not _is_leiden_available():
        logger.warning(
            "leidenalg 未安装，回退到 Louvain。pip install leidenalg igraph 启用 Leiden"
        )
        algorithm = "louvain"

    if algorithm == "leiden":
        try:
            return _detect_communities_leiden(kg_retriever, resolution, seed)
        except Exception as e:
            logger.warning("Leiden 失败，回退 Louvain: %s", e)

    # Louvain fallback
    return kg_retriever.detect_communities_louvain(resolution=resolution, seed=seed)


def _collect_community_triples(
    kg_retriever, community_entities: List[str]
) -> List[str]:
    """收集社区内所有三元组的文本表示"""
    ent_set = set(community_entities)
    triples = []
    seen = set()
    for head, rels in kg_retriever.entity_relations.items():
        if head not in ent_set:
            continue
        for rel, tail in rels:
            if tail not in ent_set:
                continue
            key = (head, rel, tail)
            if key in seen:
                continue
            seen.add(key)
            triples.append(f"{head} --[{rel}]--> {tail}")
    return triples


def _sample_triples_for_prompt(
    triples: List[str], max_triples: int
) -> List[str]:
    """采样三元组喂给 LLM，避免大社区 prompt 过长导致 LLM 慢。

    策略：取前 max_triples 条（已按收集顺序，通常覆盖核心实体）。
    若需要更智能采样，可按实体度数排序，但简单截断已足够。
    """
    if len(triples) <= max_triples:
        return triples
    return triples[:max_triples]


def _generate_summary_for_community(
    members: List[str],
    triples: List[str],
    max_triples_per_community: int,
    llm_available: bool,
) -> str:
    """为单个社区生成摘要（共享逻辑）"""
    sampled = _sample_triples_for_prompt(triples, max_triples_per_community)
    triples_text = "\n".join(sampled)

    if llm_available and triples:
        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            n_entities=len(members),
            entities="、".join(members[:30]),
            n_triples=len(triples),
            triples=triples_text,
        )
        summary = call_llm(
            "你是知识图谱分析专家。",
            prompt,
            temperature=0.2,
            max_tokens=600,
            role="extract",  # 摘要生成走抽取角色（通常用大模型）
        )
        if not summary:
            summary = (
                f"本社区包含 {len(members)} 个实体：{('、'.join(members[:15]))}等，"
                f"核心三元组 {len(triples)} 条。"
            )
    else:
        summary = (
            f"本社区包含 {len(members)} 个实体：{('、'.join(members[:15]))}等，"
            f"核心三元组 {len(triples)} 条。"
        )
    return summary.strip()


# ==========================
# 单层模式（向后兼容）
# ==========================

def build_community_summaries(
    kg_retriever,
    model,
    index_dir: str,
    force: bool = False,
    resolution: float = 1.0,
    max_triples_per_community: int = 80,
    algorithm: str = "auto",
) -> dict:
    """生成并缓存社区摘要（单层模式，向后兼容）。

    Args:
        kg_retriever: KGDualRetriever 实例（需有 entity_relations / all_entities）
        model: SentenceTransformer，用于摘要 embedding
        index_dir: 索引目录，摘要缓存于此
        force: 是否强制重建（忽略缓存）
        resolution: 社区发现分辨率
        max_triples_per_community: 单社区喂给 LLM 的最大三元组数（超长截断）
        algorithm: "leiden" / "louvain" / "auto"

    Returns:
        {
            "communities": [{"id", "entities", "summary", "n_triples"}, ...],
            "embeddings": np.ndarray,  # [n_communities, dim]
            "resolution": float,
            "algorithm": str,
        }
    """
    cache_path = os.path.join(index_dir, COMMUNITY_SUMMARY_FILE)

    # 1. 读缓存
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            if (
                data.get("resolution") == resolution
                and data.get("algorithm", "louvain") == algorithm
                and data.get("communities")
            ):
                embeddings = np.array(data["embeddings"], dtype="float32")
                logger.info(
                    "社区摘要从缓存加载 %s 个社区摘要", len(data["communities"])
                )
                return {
                    "communities": data["communities"],
                    "embeddings": embeddings,
                    "resolution": resolution,
                    "algorithm": algorithm,
                }
        except Exception as e:
            logger.warning("社区摘要缓存读取失败，将重建: %s", e)

    # 2. 分社区
    communities, _ = _detect_communities_at_resolution(
        kg_retriever, resolution, algorithm=algorithm
    )
    if not communities:
        return {
            "communities": [],
            "embeddings": np.array([], dtype="float32"),
            "resolution": resolution,
            "algorithm": algorithm,
        }

    logger.info("社区摘要共 %s 个社区，开始生成摘要...", len(communities))

    # 3. 逐社区生成摘要
    llm_available = has_llm()
    if not llm_available:
        logger.warning("未配置 LLM，社区摘要回退为实体列表（召回能力受限）")

    community_records = []
    for cid, members in enumerate(communities):
        triples = _collect_community_triples(kg_retriever, members)
        summary = _generate_summary_for_community(
            members, triples, max_triples_per_community, llm_available
        )
        community_records.append(
            {
                "id": cid,
                "entities": members,
                "summary": summary,
                "n_triples": len(triples),
            }
        )
        logger.info(
            "  [%s/%s] %s 实体, %s 三元组 -> 摘要 %s 字",
            cid + 1,
            len(communities),
            len(members),
            len(triples),
            len(summary),
        )

    # 4. 摘要 embedding
    summaries = [c["summary"] for c in community_records]
    embeddings = model.encode(
        summaries, normalize_embeddings=True, show_progress_bar=False
    )
    embeddings = np.array(embeddings, dtype="float32")

    # 5. 缓存
    try:
        os.makedirs(index_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "resolution": resolution,
                    "algorithm": algorithm,
                    "communities": community_records,
                    "embeddings": embeddings.tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("社区摘要已缓存到 %s", cache_path)
    except Exception as e:
        logger.warning("社区摘要缓存写入失败: %s", e)

    return {
        "communities": community_records,
        "embeddings": embeddings,
        "resolution": resolution,
        "algorithm": algorithm,
    }


def search_communities(
    query: str,
    model,
    community_data: dict,
    top_k: int = 3,
) -> List[dict]:
    """用 query 向量召回最相关的社区摘要。

    自动检测单层 / 层次数据结构，层次模式下调到 `search_communities_hierarchical`。

    Returns:
        [{"id", "entities", "summary", "score", ...}, ...] 按相似度降序
    """
    # 层次数据结构 → 走层次检索
    if "levels" in community_data:
        return search_communities_hierarchical(
            query, model, community_data, top_k=top_k
        )

    communities = community_data.get("communities", [])
    embeddings = community_data.get("embeddings")
    if not communities or embeddings is None or len(embeddings) == 0:
        return []

    q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")[0]

    scores = embeddings @ q_vec  # 余弦相似度（已归一化）

    k = min(top_k, len(communities))
    # 取 top-k 索引
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_idx:
        c = communities[int(idx)]
        results.append(
            {
                "id": c["id"],
                "entities": c["entities"],
                "summary": c["summary"],
                "score": float(scores[idx]),
                "level": 0,
            }
        )
    return results


# ==========================
# 层次模式（P3 新增）
# ==========================

DEFAULT_RESOLUTIONS = [0.5, 1.0, 2.0]  # 由粗到细


def _build_hierarchy(
    level_communities: List[List[List[str]]],
) -> List[List[Optional[int]]]:
    """构建层级间的父子关系。

    通过集合包含判定：level N+1 的社区 C_child，若被 level N 的某社区 C_parent 包含
    （或重叠率 > 0.5），则 C_parent 是 C_child 的父节点。

    Args:
        level_communities: 每层的社区列表，levels[0] 是最粗层

    Returns:
        parent_ids[i][j] = level i 层社区 j 的父社区在 level i-1 层的 id
        （最粗层 parent_ids[0] 全为 None）
    """
    n_levels = len(level_communities)
    parent_ids: List[List[Optional[int]]] = []

    for i in range(n_levels):
        if i == 0:
            parent_ids.append([None] * len(level_communities[i]))
            continue

        prev_level = level_communities[i - 1]  # 父层（更粗）
        cur_level = level_communities[i]  # 子层（更细）
        parents: List[Optional[int]] = []

        for child in cur_level:
            child_set = set(child)
            best_parent = None
            best_overlap = 0.0
            for p_idx, parent in enumerate(prev_level):
                parent_set = set(parent)
                if not parent_set:
                    continue
                # Jaccard 重叠率
                overlap = len(child_set & parent_set) / len(child_set | parent_set)
                # 也可以用包含率：|child ∩ parent| / |child|
                containment = len(child_set & parent_set) / len(child_set)
                # 优先 containment 高的（child 几乎都在 parent 内）
                score = max(overlap, containment)
                if score > best_overlap:
                    best_overlap = score
                    best_parent = p_idx
            # 阈值：至少 30% 重叠才认父
            if best_overlap < 0.3:
                parents.append(None)
            else:
                parents.append(best_parent)
        parent_ids.append(parents)

    return parent_ids


def _generate_parent_summary(
    members: List[str],
    child_summaries: List[str],
    level: int,
    llm_available: bool,
) -> str:
    """父社区摘要：基于子社区摘要生成（MS GraphRAG 标准做法）。

    相比直接看三元组，优势：
    - token 更少（子摘要 200-400 字 vs 父三元组可能上千条）
    - 信息已提炼，父摘要质量更高
    - 递归构建，层次越高摘要越抽象
    """
    if llm_available and child_summaries:
        # 拼接子摘要（限制长度避免超 token）
        joined = "\n\n---\n\n".join(child_summaries[:20])
        prompt = PARENT_COMMUNITY_SUMMARY_PROMPT.format(
            level=level,
            n_entities=len(members),
            n_children=len(child_summaries),
            child_summaries=joined,
        )
        summary = call_llm(
            "你是知识图谱分析专家。",
            prompt,
            temperature=0.2,
            max_tokens=600,
            role="extract",
        )
        if not summary:
            summary = (
                f"本父社区包含 {len(members)} 个实体、{len(child_summaries)} 个子社区，"
                f"核心实体：{('、'.join(members[:15]))}等。"
            )
    else:
        summary = (
            f"本父社区包含 {len(members)} 个实体、{len(child_summaries)} 个子社区，"
            f"核心实体：{('、'.join(members[:15]))}等。"
        )
    return summary.strip()


def build_hierarchical_community_summaries(
    kg_retriever,
    model,
    index_dir: str,
    force: bool = False,
    resolutions: List[float] = None,
    max_triples_per_community: int = 80,
    algorithm: str = "auto",
    min_community_size: int = 5,
    max_communities_per_level: int = 50,
) -> dict:
    """构建多层级社区摘要（对标 MS GraphRAG Hierarchical Summaries）。

    优化策略（对标 MS GraphRAG）：
    1. 最细层用三元组生成摘要（leaf-level）
    2. 父层用子社区摘要聚合生成（recursive summarization）
    3. 小社区（< min_community_size）跳过 LLM，用实体列表兜底
    4. 每层最多 max_communities_per_level 个社区调 LLM（按大小排序取 top-N）

    Args:
        kg_retriever: KGDualRetriever 实例
        model: SentenceTransformer
        index_dir: 索引目录
        force: 强制重建
        resolutions: 分辨率列表，升序（由粗到细）。默认 [0.5, 1.0, 2.0]
        max_triples_per_community: 单社区喂给 LLM 的最大三元组数
        algorithm: "leiden" / "louvain" / "auto"
        min_community_size: 小于此规模的社区跳过 LLM 摘要（用实体列表）
        max_communities_per_level: 每层最多对 N 个大社区生成 LLM 摘要

    Returns:
        {
            "levels": [...],
            "algorithm": str,
            "resolutions": [float, ...],
            # 兼容字段
            "communities": [...],
            "embeddings": np.ndarray,
            "resolution": float,
        }
    """
    if resolutions is None:
        resolutions = list(DEFAULT_RESOLUTIONS)
    # 升序排列（粗 → 细）
    resolutions = sorted(set(float(r) for r in resolutions))
    if len(resolutions) < 1:
        resolutions = list(DEFAULT_RESOLUTIONS)

    cache_path = os.path.join(index_dir, HIERARCHICAL_COMMUNITY_FILE)

    # 1. 读缓存
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            if (
                data.get("resolutions") == resolutions
                and data.get("algorithm", "louvain") == algorithm
                and data.get("levels")
            ):
                # 重建 numpy embeddings
                levels = []
                dim_mismatch = False
                expected_dim = None
                for lv in data["levels"]:
                    emb = np.array(lv["embeddings"], dtype="float32")
                    # 维度校验：缓存里 embeddings 列数必须等于 model 输出维度，
                    # 否则视为脏缓存（早期测试用 mock model 写入的 8 维向量等），
                    # 触发重建避免运行时 matmul 崩溃。
                    if emb.size > 0 and emb.ndim == 2:
                        if expected_dim is None:
                            try:
                                expected_dim = model.get_sentence_embedding_dimension()
                            except Exception:
                                expected_dim = None
                        if expected_dim and emb.shape[1] != expected_dim:
                            logger.warning(
                                "层次社区摘要缓存维度不匹配（level=%s 缓存=%s 期望=%s），将重建",
                                lv.get("level"), emb.shape[1], expected_dim,
                            )
                            dim_mismatch = True
                            break
                    levels.append(
                        {
                            "level": lv["level"],
                            "resolution": lv["resolution"],
                            "communities": lv["communities"],
                            "embeddings": emb,
                        }
                    )
                if not dim_mismatch:
                    logger.info(
                        "层次社区摘要从缓存加载 %s 层", len(levels)
                    )
                    result = {
                        "levels": levels,
                        "algorithm": algorithm,
                        "resolutions": resolutions,
                    }
                    # 兼容字段：指向最细层
                    finest = levels[-1]
                    result["communities"] = finest["communities"]
                    result["embeddings"] = finest["embeddings"]
                    result["resolution"] = finest["resolution"]
                    return result
        except Exception as e:
            logger.warning("层次社区摘要缓存读取失败，将重建: %s", e)

    # 2. 每层分社区
    logger.info(
        "构建层次社区摘要: %s 层, resolutions=%s, algorithm=%s",
        len(resolutions),
        resolutions,
        algorithm,
    )
    level_communities: List[List[List[str]]] = []
    for r in resolutions:
        comms, _ = _detect_communities_at_resolution(
            kg_retriever, r, algorithm=algorithm
        )
        level_communities.append(comms)
        logger.info("  level %s (r=%.3f): %s 个社区", len(level_communities) - 1, r, len(comms))

    # 3. 构建父子关系
    parent_ids = _build_hierarchy(level_communities)
    # 反向：child_ids[level][parent_id] = [child_id, ...]
    child_ids_per_level: List[List[List[int]]] = []
    for i in range(len(level_communities)):
        child_ids_per_level.append([[] for _ in level_communities[i]])
    for i in range(1, len(parent_ids)):
        for child_idx, p in enumerate(parent_ids[i]):
            if p is not None:
                child_ids_per_level[i - 1][p].append(child_idx)

    # 4. 逐层生成摘要（自底向上：最细层用三元组，父层用子摘要）
    llm_available = has_llm()
    if not llm_available:
        logger.warning("未配置 LLM，社区摘要回退为实体列表（召回能力受限）")

    levels_data = []
    # child_summaries_by_level[i][cid] = [summary_str, ...] 用于父层聚合
    # 在自底向上遍历时填充
    child_summaries_by_level: List[dict] = [{} for _ in range(len(level_communities))]

    # 反向遍历：从最细层开始，父层依赖子层摘要
    for i in range(len(level_communities) - 1, -1, -1):
        r = resolutions[i]
        comms = level_communities[i]
        # 按社区大小降序排序（大的优先得到 LLM 摘要）
        comm_with_idx = sorted(
            enumerate(comms), key=lambda x: -len(x[1])
        )

        # 限制 LLM 调用数：只对 top-N 大社区生成 LLM 摘要
        llm_eligible_ids = set()
        for idx, members in comm_with_idx[:max_communities_per_level]:
            if len(members) >= min_community_size:
                llm_eligible_ids.add(idx)

        community_records = [None] * len(comms)
        is_leaf = (i == len(level_communities) - 1)

        for cid, members in enumerate(comms):
            triples = _collect_community_triples(kg_retriever, members)

            if cid in llm_eligible_ids and llm_available:
                if is_leaf:
                    # 最细层：基于三元组生成
                    summary = _generate_summary_for_community(
                        members, triples, max_triples_per_community, True
                    )
                else:
                    # 父层：基于子社区摘要生成（已在上一步填充）
                    child_sums = child_summaries_by_level[i].get(cid, [])
                    if child_sums:
                        summary = _generate_parent_summary(
                            members, child_sums, i, True
                        )
                    else:
                        # 没有子摘要可聚合，退化为三元组摘要
                        summary = _generate_summary_for_community(
                            members, triples, max_triples_per_community, True
                        )
            else:
                # 小社区或无 LLM：实体列表兜底
                summary = (
                    f"本社区包含 {len(members)} 个实体：{('、'.join(members[:15]))}等，"
                    f"核心三元组 {len(triples)} 条。"
                )

            summary = summary.strip() if isinstance(summary, str) else summary
            community_records[cid] = {
                "id": cid,
                "entities": members,
                "summary": summary,
                "n_triples": len(triples),
                "parent_id": parent_ids[i][cid],
                "child_ids": child_ids_per_level[i][cid],
            }
            if cid in llm_eligible_ids:
                logger.info(
                    "  L%s [%s/%s] %s 实体, %s 三元组 -> LLM 摘要 %s 字",
                    i, cid + 1, len(comms), len(members), len(triples), len(summary),
                )

            # 把自己的摘要填到父社区的 child_summaries 列表里
            parent_id = parent_ids[i][cid]
            if parent_id is not None:
                child_summaries_by_level[i - 1].setdefault(parent_id, []).append(summary)

        # embedding
        summaries = [c["summary"] for c in community_records if c]
        if summaries:
            emb = model.encode(
                summaries, normalize_embeddings=True, show_progress_bar=False
            )
            emb = np.array(emb, dtype="float32")
        else:
            emb = np.array([], dtype="float32")

        levels_data.insert(0, {
            "level": i,
            "resolution": r,
            "communities": community_records,
            "embeddings": emb,
        })

    # 5. 缓存
    try:
        os.makedirs(index_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "resolutions": resolutions,
                    "algorithm": algorithm,
                    "levels": [
                        {
                            "level": lv["level"],
                            "resolution": lv["resolution"],
                            "communities": lv["communities"],
                            "embeddings": lv["embeddings"].tolist(),
                        }
                        for lv in levels_data
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("层次社区摘要已缓存到 %s", cache_path)
    except Exception as e:
        logger.warning("层次社区摘要缓存写入失败: %s", e)

    result = {
        "levels": levels_data,
        "algorithm": algorithm,
        "resolutions": resolutions,
    }
    # 兼容字段：指向最细层
    finest = levels_data[-1]
    result["communities"] = finest["communities"]
    result["embeddings"] = finest["embeddings"]
    result["resolution"] = finest["resolution"]
    return result


def search_communities_hierarchical(
    query: str,
    model,
    community_data: dict,
    top_k: int = 3,
    level: int = -1,
    roll_up: bool = True,
    roll_up_threshold: float = 0.7,
) -> List[dict]:
    """多层级社区摘要检索。

    策略：
    1. 在指定 level 的所有社区摘要上做向量召回
    2. （可选）Roll-up: 若某父社区的所有子社区得分都高于阈值，替换为父社区
       （父社区摘要提供更广的全局视角，适合归纳类问题）
    3. 去重（roll-up 后可能产生重复）

    Args:
        query: 用户查询
        model: SentenceTransformer
        community_data: build_hierarchical_community_summaries 的返回值
        top_k: 返回前 K 个社区
        level: 从哪一层检索。-1=最细层，0=最粗层
        roll_up: 是否启用 roll-up 优化
        roll_up_threshold: 子社区得分高于此值时考虑 roll-up

    Returns:
        [{"level", "id", "entities", "summary", "score", "parent_id"}, ...]
    """
    levels = community_data.get("levels", [])
    if not levels:
        return []

    # 解析 level 索引
    if level < 0:
        lv_idx = len(levels) + level  # -1 → 最后（最细）
    else:
        lv_idx = min(level, len(levels) - 1)
    lv_idx = max(0, lv_idx)

    target_level = levels[lv_idx]
    communities = target_level["communities"]
    embeddings = target_level["embeddings"]
    if not communities or embeddings is None or len(embeddings) == 0:
        return []

    q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")[0]
    scores = embeddings @ q_vec

    # 取 top-k*2 候选（roll-up 可能合并，多取一些避免数量不足）
    k = min(top_k * 2, len(communities))
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    seen_ids = set()
    for idx in top_idx:
        idx = int(idx)
        c = communities[idx]
        score = float(scores[idx])

        # Roll-up: 若有父社区，且所有兄弟（包括自己）得分都 ≥ threshold，则用父社区
        if (
            roll_up
            and c.get("parent_id") is not None
            and lv_idx > 0
        ):
            parent_id = c["parent_id"]
            parent_level = levels[lv_idx - 1]
            siblings = [
                sib for sib in communities if sib["parent_id"] == parent_id
            ]
            if siblings and parent_id not in seen_ids:
                sibling_scores = [
                    float(scores[sib["id"]]) for sib in siblings
                ]
                if all(s >= roll_up_threshold for s in sibling_scores):
                    # 替换为父社区
                    parent = parent_level["communities"][parent_id]
                    parent_emb = parent_level["embeddings"][parent_id]
                    parent_score = float(parent_emb @ q_vec)
                    results.append(
                        {
                            "level": lv_idx - 1,
                            "id": parent_id,
                            "entities": parent["entities"],
                            "summary": parent["summary"],
                            "score": parent_score,
                            "parent_id": parent.get("parent_id"),
                        }
                    )
                    seen_ids.add(parent_id)
                    continue

        if c["id"] in seen_ids:
            continue
        results.append(
            {
                "level": lv_idx,
                "id": c["id"],
                "entities": c["entities"],
                "summary": c["summary"],
                "score": score,
                "parent_id": c.get("parent_id"),
            }
        )
        seen_ids.add(c["id"])

        if len(results) >= top_k:
            break

    # 按 score 降序
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]
