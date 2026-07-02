"""
Knowledge Graph Dual-Layer Retriever (LightRAG-style)

支持两种 KG 检索模式：
- Local Search: 基于实体匹配 + BFS 邻域扩展（从实体出发）
- Global Search: 基于关系关键词匹配（从关系出发）

以及混合模式 Mix = Local + Global，和纯 KG 模式。

实体匹配使用嵌入向量相似度（而非子串匹配），通过独立的实体 FAISS 索引实现。
"""

import json
import os
from collections import deque
from typing import Optional

import faiss
import numpy as np

from .core.storages import GraphStore, InMemoryGraphStore
from .logging_config import get_logger

logger = get_logger(__name__)


class KGDualRetriever:
    """LightRAG 风格的知识图谱双层检索器"""

    def __init__(
        self,
        entity_relations: dict,
        reverse_relations: dict,
        model,
        index_dir: str,
        threshold: float = 0.5,
        n_hops: int = 2,
        relation_threshold: float = 0.3,
        graph_store: Optional[GraphStore] = None,
    ):
        """
        Args:
            entity_relations: {head_entity: [(relation, tail_entity), ...]}
                当 graph_store 提供时，此参数仅用于向后兼容，实际数据从 store 读取
            reverse_relations: {tail_entity: [(head_entity, relation), ...]}
                同上
            model: SentenceTransformer 模型实例，用于编码查询
            index_dir: 实体嵌入索引所在目录
            threshold: 实体匹配相似度阈值
            n_hops: 局部检索 BFS 跳数
            relation_threshold: 关系匹配相似度阈值（Global Search）
            graph_store: GraphStore 抽象层实例。提供时作为图数据的唯一数据源，
                支持未来切换 Neo4j 等后端。None 时自动用 InMemoryGraphStore
                包装传入的 dict（保持向后兼容）
        """
        # 图存储抽象层：优先用外部传入的 store，否则包装 dict
        if graph_store is not None:
            self.graph_store: GraphStore = graph_store
        else:
            self.graph_store = InMemoryGraphStore(
                entity_relations=entity_relations,
                reverse_relations=reverse_relations,
            )

        # 向后兼容：保留对底层 dict 的直接引用
        # InMemoryGraphStore 暴露 entity_relations / reverse_relations 公有属性
        # 未来切换 Neo4j 后端时，这些 dict 引用将不可用，需迁移到 store API
        if isinstance(self.graph_store, InMemoryGraphStore):
            self.entity_relations = self.graph_store.entity_relations
            self.reverse_relations = self.graph_store.reverse_relations
        else:
            # 非 memory 后端：dict 访问不可用，标记为 None 以便及早发现
            self.entity_relations = {}
            self.reverse_relations = {}

        self.model = model
        self.index_dir = index_dir
        self.threshold = threshold
        self.n_hops = n_hops
        self.relation_threshold = relation_threshold

        # 收集所有唯一实体
        self.all_entities = sorted(
            set(entity_relations.keys()) | set(reverse_relations.keys())
        )

        # 实体索引映射（用于图算法）
        self.entity_idx = {e: i for i, e in enumerate(self.all_entities)}
        self.idx_entity = {i: e for i, e in enumerate(self.all_entities)}

        # 邻接表缓存（懒加载）
        self._adj_cache = None

        # 收集所有唯一关系类型
        all_relations = set()
        for rels in entity_relations.values():
            for rel, _ in rels:
                all_relations.add(rel)
        self.all_relations = sorted(all_relations)

        # 加载或构建实体嵌入索引
        self._entity_index = None
        self._entity_names = None
        self._load_entity_index()

        # 加载或构建关系嵌入索引
        self._relation_index = None
        self._relation_names = None
        self._load_relation_index()

    # ==========================
    # 实体嵌入索引
    # ==========================

    def _load_entity_index(self):
        """从磁盘加载实体嵌入索引，若不存在则构建"""
        index_path = os.path.join(self.index_dir, "entity_faiss.index")
        names_path = os.path.join(self.index_dir, "entity_names.json")

        if os.path.exists(index_path) and os.path.exists(names_path):
            self._entity_index = faiss.read_index(index_path)
            with open(names_path, encoding="utf-8") as f:
                self._entity_names = json.load(f)
            logger.info("实体嵌入索引加载完成: %s 个实体", len(self._entity_names))
        else:
            self._build_entity_index()

    def _build_entity_index(self):
        """构建实体嵌入索引并保存"""
        logger.info("正在构建实体嵌入索引...")
        embeddings = self.model.encode(
            self.all_entities,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self._entity_index = faiss.IndexFlatIP(dim)
        self._entity_index.add(embeddings)
        self._entity_names = list(self.all_entities)

        # 保存到磁盘
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(
            self._entity_index,
            os.path.join(self.index_dir, "entity_faiss.index"),
        )
        with open(
            os.path.join(self.index_dir, "entity_names.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self._entity_names, f, ensure_ascii=False, indent=2)

        logger.info("实体嵌入索引构建完成: %s 个实体", len(self._entity_names))

    def match_entities(
        self, query: str, top_k: int = 5, threshold: float = None,
        return_scores: bool = False,
    ) -> list:
        """匹配查询中的实体：精确子串匹配 boost + 嵌入相似度补充

        精确子串匹配优先（query 直接包含 KG 实体名 → 高置信种子，不受 threshold 限制），
        再用嵌入相似度补充到 top_k。解决长实体名（如"东方毛眼水蝇"）在整句 embedding
        中被近似实体挤出 top_k 的问题。

        Args:
            query: 查询文本
            top_k: 最多返回的实体数
            threshold: 嵌入相似度阈值（None 则使用实例默认值），仅作用于 embedding 分支
            return_scores: 若 True，返回 [(entity, score), ...] 而非 [entity, ...]
                精确子串匹配的 score 记为 1.0，embedding 分支记录实际相似度分数。
                用于拒答判断：若最高 seed score 过低，说明查询与 KG 无关。

        Returns:
            匹配到的实体名称列表（精确匹配在前，embedding 在后）。
            若 return_scores=True，返回 [(entity, score), ...]。
        """
        threshold = threshold if threshold is not None else self.threshold
        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")

        scores, indices = self._entity_index.search(
            query_vec, min(top_k, len(self._entity_names))
        )

        matched = []
        matched_scores = []
        seen = set()
        # 1. 精确子串匹配 boost：query 直接包含实体名 → 高置信种子
        #    长实体名（>=2 字）在整句 embedding 里信号被稀释，但子串匹配是确定性的。
        for name in self._entity_names:
            if len(name) >= 2 and name in query and name not in seen:
                matched.append(name)
                matched_scores.append(1.0)
                seen.add(name)
                if len(matched) >= top_k:
                    if return_scores:
                        return list(zip(matched, matched_scores))
                    return matched
        # 2. embedding 相似度补充到 top_k
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score >= threshold:
                name = self._entity_names[idx]
                if name not in seen:
                    matched.append(name)
                    matched_scores.append(float(score))
                    seen.add(name)
            if len(matched) >= top_k:
                break
        if return_scores:
            return list(zip(matched, matched_scores))
        return matched

    def match_entities_by_relation_value(
        self, query: str, top_k: int = 10
    ) -> list:
        """按关系值（tail）反查头实体：query 里出现的 tail 值 → 收集对应 head 实体

        解决聚合类问题：
            query="哪些农药可以在发病初期使用？"
            → "发病初期" 是 (药剂, 施药时期, 发病初期) 的 tail 值
            → 反查到所有施药时期=发病初期的药剂实体

        与 match_entities（正向实体匹配）互补：后者只找 query 里出现的实体名，
        无法处理"按关系值查头实体"的聚合问题。

        Args:
            query: 查询文本
            top_k: 最多返回的实体数

        Returns:
            头实体名称列表
        """
        matched = []
        seen = set()
        for tail, rels in self.reverse_relations.items():
            # tail 是关系值（如"发病初期""分蘖末期"），需在 query 里出现
            if len(tail) < 2 or tail not in query:
                continue
            for head, _rel in rels:
                if head not in seen:
                    matched.append(head)
                    seen.add(head)
                if len(matched) >= top_k:
                    return matched
        return matched

    # ==========================
    # 关系嵌入索引
    # ==========================

    def _load_relation_index(self):
        """从磁盘加载关系嵌入索引，若不存在则构建"""
        index_path = os.path.join(self.index_dir, "relation_faiss.index")
        names_path = os.path.join(self.index_dir, "relation_names.json")

        if os.path.exists(index_path) and os.path.exists(names_path):
            self._relation_index = faiss.read_index(index_path)
            with open(names_path, encoding="utf-8") as f:
                self._relation_names = json.load(f)
            logger.info("关系嵌入索引加载完成: %s 种关系", len(self._relation_names))
        else:
            self._build_relation_index()

    def _build_relation_index(self):
        """构建关系嵌入索引并保存"""
        logger.info("正在构建关系嵌入索引...")
        if not self.all_relations:
            self._relation_index = faiss.IndexFlatIP(1)
            self._relation_names = []
            return

        embeddings = self.model.encode(
            self.all_relations,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self._relation_index = faiss.IndexFlatIP(dim)
        self._relation_index.add(embeddings)
        self._relation_names = list(self.all_relations)

        # 保存到磁盘
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(
            self._relation_index,
            os.path.join(self.index_dir, "relation_faiss.index"),
        )
        with open(
            os.path.join(self.index_dir, "relation_names.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self._relation_names, f, ensure_ascii=False, indent=2)

        logger.info("关系嵌入索引构建完成: %s 种关系", len(self._relation_names))

    def match_relations(
        self, query: str, top_k: int = 5, threshold: float = None
    ) -> list:
        """使用嵌入相似度匹配查询中的关系

        Args:
            query: 查询文本
            top_k: 最多返回的关系数
            threshold: 相似度阈值（None 则使用实例默认值）

        Returns:
            匹配到的关系名称列表
        """
        if not self._relation_names:
            return []

        threshold = threshold if threshold is not None else self.relation_threshold
        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
        scores, indices = self._relation_index.search(
            query_vec, min(top_k, len(self._relation_names))
        )
        matched = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score >= threshold:
                matched.append(self._relation_names[idx])
            if len(matched) >= top_k:
                break
        return matched

    # ==========================
    # Local Search（从实体出发）
    # ==========================

    def local_search(self, query: str, n_hops: int = None) -> list:
        """局部检索：匹配查询中的实体，BFS 扩展邻域

        Args:
            query: 查询文本
            n_hops: BFS 跳数（None 则使用实例默认值）

        Returns:
            需要检索其文本块的实体名称列表
        """
        n_hops = n_hops if n_hops is not None else self.n_hops
        seed_entities = self.match_entities(query)
        # 关系值反查：聚合类问题（如"发病初期可以用什么药"）需要按 tail 值反查 head
        seed_entities += self.match_entities_by_relation_value(query)
        if not seed_entities:
            return []
        # 去重保序（精确匹配 + 关系值反查 + embedding 匹配合并后去重）
        seed_entities = list(dict.fromkeys(seed_entities))

        # BFS 扩展
        visited = set(seed_entities)
        queue = deque([(e, 0) for e in seed_entities])

        while queue:
            entity, depth = queue.popleft()
            if depth >= n_hops:
                continue
            # 正向关系
            for _, tail in self.entity_relations.get(entity, []):
                if tail not in visited:
                    visited.add(tail)
                    queue.append((tail, depth + 1))
            # 反向关系
            for _, rel_src in self.reverse_relations.get(entity, []):
                if rel_src not in visited:
                    visited.add(rel_src)
                    queue.append((rel_src, depth + 1))

        return list(visited)

    # ==========================
    # Global Search（从关系出发）
    # ==========================

    def global_search(self, query: str) -> list:
        """全局检索：匹配查询中的关系，收集拥有这些关系的实体

        使用嵌入相似度匹配关系，而非子串匹配。

        Args:
            query: 查询文本

        Returns:
            需要检索其文本块的实体名称列表
        """
        matched_relations = self.match_relations(query)

        if not matched_relations:
            return []

        entities = set()
        for head, rels in self.entity_relations.items():
            for rel, _ in rels:
                if rel in matched_relations:
                    entities.add(head)

        # 也从反向关系中收集
        for tail, rels in self.reverse_relations.items():
            for _, rel in rels:
                if rel in matched_relations:
                    entities.add(tail)

        return list(entities)

    # ==========================
    # Mix Search
    # ==========================

    def mix_search(self, query: str, n_hops: int = None) -> list:
        """混合检索：Local + Global 去重合并

        Args:
            query: 查询文本
            n_hops: BFS 跳数

        Returns:
            去重后的实体名称列表
        """
        local_entities = set(self.local_search(query, n_hops))
        global_entities = set(self.global_search(query))
        return list(local_entities | global_entities)

    # ==========================
    # 图谱数据导出（用于可视化）
    # ==========================

    def get_entity_degree(self, entity: str) -> int:
        """获取实体的度数（出度+入度）"""
        out_degree = len(self.entity_relations.get(entity, []))
        in_degree = len(self.reverse_relations.get(entity, []))
        return out_degree + in_degree

    def get_top_entities(self, top_k: int = 50) -> list:
        """获取度数最高的 Top K 个实体"""
        entities_with_degree = [
            (e, self.get_entity_degree(e)) for e in self.all_entities
        ]
        entities_with_degree.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in entities_with_degree[:top_k]]

    def get_subgraph(self, entities: list, max_hops: int = 1) -> dict:
        """获取指定实体的子图（节点和边）

        Args:
            entities: 中心实体列表
            max_hops: 邻域扩展跳数

        Returns:
            {"nodes": [{"id": 名称, "name": 名称, "degree": 度数, "category": 类别}, ...],
             "links": [{"source": 头实体, "target": 尾实体, "relation": 关系}, ...]}
        """
        # BFS 收集所有涉及的实体
        visited = set(entities)
        queue = [
            (e, 0)
            for e in entities
            if e in self.entity_relations or e in self.reverse_relations
        ]

        while queue:
            entity, depth = queue.pop(0)
            if depth >= max_hops:
                continue

            # 正向关系
            for rel, tail in self.entity_relations.get(entity, []):
                if tail not in visited:
                    visited.add(tail)
                    queue.append((tail, depth + 1))

            # 反向关系
            for head, rel in self.reverse_relations.get(entity, []):
                if head not in visited:
                    visited.add(head)
                    queue.append((head, depth + 1))

        # 构建节点和边
        nodes = []
        entity_set = set()
        for e in visited:
            degree = self.get_entity_degree(e)
            category = 0 if e in entities else 1  # 0=中心实体, 1=邻域实体
            nodes.append(
                {
                    "id": e,
                    "name": e,
                    "degree": degree,
                    "category": category,
                    "symbolSize": min(max(degree * 0.5 + 10, 12), 50),
                }
            )
            entity_set.add(e)

        links = []
        # 遍历所有节点的正向关系，只保留两端都在子图中的边
        for head in entity_set:
            for rel, tail in self.entity_relations.get(head, []):
                if tail in entity_set:
                    links.append(
                        {
                            "source": head,
                            "target": tail,
                            "relation": rel,
                        }
                    )

        return {"nodes": nodes, "links": links}

    def get_graph_stats(self) -> dict:
        """获取图谱统计信息"""
        total_edges = sum(len(rels) for rels in self.entity_relations.values())
        avg_degree = (
            total_edges * 2 / len(self.all_entities) if self.all_entities else 0
        )
        return {
            "total_entities": len(self.all_entities),
            "total_relations": len(self.all_relations),
            "total_edges": total_edges,
            "avg_degree": round(avg_degree, 2),
        }

    # ==========================
    # 图算法: Pagerank
    # ==========================

    def compute_pagerank(
        self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> dict:
        """计算所有实体的 Pagerank 分数（NumPy 向量化优化版）

        Pagerank 算法衡量实体在知识图谱中的重要性。
        入度多、且来自重要实体的入度多的实体，Pagerank 分数越高。

        Args:
            damping: 阻尼系数（默认 0.85，标准值）
            max_iter: 最大迭代次数
            tol: 收敛阈值

        Returns:
            {entity_name: pagerank_score} 字典
        """
        n = len(self.all_entities)
        if n == 0:
            return {}

        adj = self._build_adjacency_list()

        # 构建 COO 格式边列表（无向图，每条边存两个方向）
        src_list = []
        dst_list = []
        for u in range(n):
            for v in adj[u]:
                src_list.append(u)
                dst_list.append(v)

        if not src_list:
            # 没有边，返回均匀分布
            return {self.idx_entity[i]: 1.0 / n for i in range(n)}

        src = np.array(src_list, dtype=np.int64)
        dst = np.array(dst_list, dtype=np.int64)

        # 计算每个节点的出度
        out_degree = np.array([len(adj[i]) for i in range(n)], dtype=np.float64)
        # 避免除零（孤立节点出度设为 1，贡献为 0，因为没有边）
        out_degree_safe = np.where(out_degree > 0, out_degree, 1.0)

        # 初始化 Pagerank
        pr = np.ones(n, dtype=np.float64) / n
        teleport = (1 - damping) / n

        for _ in range(max_iter):
            # 每条边的贡献 = pr[src] / out_degree[src]
            edge_contrib = pr[src] / out_degree_safe[src]

            # 将贡献累加到目标节点（scatter-add）
            new_pr = np.full(n, teleport, dtype=np.float64)
            np.add.at(new_pr, dst, damping * edge_contrib)

            # 处理悬挂节点（没有出边的节点）- 将它们的 PR 均匀分给所有节点
            # （无向图中其实不存在纯悬挂节点，但以防万一）
            dangling = pr[out_degree == 0].sum()
            if dangling > 0:
                new_pr += damping * dangling / n

            # 归一化
            new_pr = new_pr / new_pr.sum()

            # 检查收敛
            if np.abs(new_pr - pr).sum() < tol:
                pr = new_pr
                break

            pr = new_pr

        return {self.idx_entity[i]: float(pr[i]) for i in range(n)}

    # ==========================
    # 图算法: 社区发现 (标签传播)
    # ==========================

    def detect_communities(self, max_iter: int = 50) -> dict:
        """使用标签传播算法（Label Propagation）进行社区发现

        将知识图谱中的实体划分为若干社区，同一社区内的实体连接更紧密。

        Args:
            max_iter: 最大迭代次数

        Returns:
            {entity_name: community_id} 字典
        """
        n = len(self.all_entities)
        if n == 0:
            return {}

        entity_idx = {e: i for i, e in enumerate(self.all_entities)}
        idx_entity = {i: e for i, e in enumerate(self.all_entities)}

        # 构建无向邻接表
        adj = [set() for _ in range(n)]
        for head, rels in self.entity_relations.items():
            head_idx = entity_idx.get(head)
            if head_idx is None:
                continue
            for _, tail in rels:
                tail_idx = entity_idx.get(tail)
                if tail_idx is not None and tail_idx != head_idx:
                    adj[head_idx].add(tail_idx)
                    adj[tail_idx].add(head_idx)

        # 初始化标签：每个节点一个独立社区
        labels = np.arange(n, dtype=np.int32)

        for iteration in range(max_iter):
            changed = False
            # 随机顺序更新
            order = np.random.permutation(n)

            for i in order:
                neighbors = adj[i]
                if not neighbors:
                    continue

                # 统计邻居标签出现次数
                neighbor_labels = labels[list(neighbors)]
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)

                # 选择出现次数最多的标签（如果有多个，选最小的）
                max_count = counts.max()
                best_labels = unique_labels[counts == max_count]
                new_label = best_labels.min()

                if new_label != labels[i]:
                    labels[i] = new_label
                    changed = True

            if not changed:
                break

        # 重新编号社区（0, 1, 2, ...）
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}

        return {idx_entity[i]: int(label_map[labels[i]]) for i in range(n)}

    def detect_communities_louvain(
        self, resolution: float = 1.0, seed: int = 42
    ) -> tuple:
        """用 Louvain 层次聚类做社区发现（替代标签传播）。

        Louvain 通过模块度（modularity）最大化，相比标签传播：
        - 结果更稳定（标签传播随机性大）
        - 支持层次聚类（resolution 参数控制社区粒度，越大社区越小）
        - 是 MS GraphRAG / LightRAG 社区摘要的核心算法

        依赖：networkx >= 3.0 内置 `louvain_communities`，无需额外安装。

        Args:
            resolution: 分辨率参数。>1 得到更小更多社区，<1 得到更大更少社区。
            seed: 随机种子，保证可复现

        Returns:
            (communities, community_map):
              communities = [[entity, ...], ...] 社区列表
              community_map = {entity: community_id} 每个实体所属社区
        """
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        if not self.all_entities:
            return [], {}

        # 构建 networkx 无向图
        G = nx.Graph()
        G.add_nodes_from(self.all_entities)
        for head, rels in self.entity_relations.items():
            for _, tail in rels:
                if tail != head:
                    G.add_edge(head, tail)

        if G.number_of_edges() == 0:
            # 无边图：每个实体自成社区
            return [[e] for e in self.all_entities], {
                e: i for i, e in enumerate(self.all_entities)
            }

        try:
            communities_sets = louvain_communities(G, resolution=resolution, seed=seed)
        except Exception as e:
            logger.warning("Louvain 失败，回退标签传播: %s", e)
            cm = self.detect_communities()
            # 把 map 转回 communities 列表
            inv: dict = {}
            for ent, cid in cm.items():
                inv.setdefault(cid, []).append(ent)
            return list(inv.values()), cm

        # 编号
        communities = [sorted(list(c)) for c in communities_sets]
        community_map = {}
        for cid, members in enumerate(communities):
            for ent in members:
                community_map[ent] = cid

        logger.info(
            "Louvain 发现 %s 个社区，最大社区 %s 实体，模块度 %.3f",
            len(communities),
            max(len(c) for c in communities),
            nx.algorithms.community.modularity(G, communities_sets),
        )
        return communities, community_map

    # ==========================
    # 图算法: 最短路径 (BFS)
    # ==========================

    def shortest_path(self, start: str, end: str, max_hops: int = 5) -> list:
        """查找两个实体之间的最短路径（BFS）

        Args:
            start: 起始实体
            end: 目标实体
            max_hops: 最大搜索跳数

        Returns:
            路径实体列表 [start, ..., end]，如果不存在则返回空列表
        """
        if start == end:
            return [start]

        if start not in self.entity_relations and start not in self.reverse_relations:
            return []
        if end not in self.entity_relations and end not in self.reverse_relations:
            return []

        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            if len(path) > max_hops:
                continue

            # 正向关系
            for _, tail in self.entity_relations.get(current, []):
                if tail == end:
                    return path + [tail]
                if tail not in visited:
                    visited.add(tail)
                    queue.append((tail, path + [tail]))

            # 反向关系
            for head, _ in self.reverse_relations.get(current, []):
                if head == end:
                    return path + [head]
                if head not in visited:
                    visited.add(head)
                    queue.append((head, path + [head]))

        return []

    def path_between_entities(self, entities: list, max_hops: int = 5) -> list:
        """查找多个实体之间的关联路径（两两最短路径合并）

        Args:
            entities: 实体列表
            max_hops: 最大搜索跳数

        Returns:
            路径实体列表（去重后），如果没有关联则返回空列表
        """
        if len(entities) < 2:
            return entities

        all_path_entities = set(entities)

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                path = self.shortest_path(entities[i], entities[j], max_hops=max_hops)
                if path:
                    all_path_entities.update(path)

        return list(all_path_entities)

    # ========================
    # 个性化 Pagerank
    # ========================

    def personalized_pagerank(
        self,
        seed_entities: list,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict:
        """个性化 Pagerank (Personalized PageRank, PPR) - NumPy 向量化优化版

        从种子实体出发，计算其他实体与种子实体的相关性分数。
        与普通 Pagerank 的区别：随机跳转只跳转到种子实体，而非所有实体。

        性能说明：本实现使用 KGDualRetriever.__init__ 时预计算的 entity_idx /
        all_entities / _adj_cache（懒加载），单次调用 ~50-200ms（6k 实体）。
        GraphStore 后端的 personalized_pagerank 实现目前未做缓存，每次重算
        all_entities + COO 边列表，性能差 5-10x，故暂不切换。
        未来若后端做了等价缓存优化，可改为优先走 store。

        Args:
            seed_entities: 种子实体列表
            damping: 阻尼系数，默认 0.85
            max_iter: 最大迭代次数
            tol: 收敛阈值

        Returns:
            {实体名: PPR 分数} 字典
        """
        n = len(self.all_entities)
        if n == 0:
            return {}

        # 过滤有效种子实体
        valid_seeds = [e for e in seed_entities if e in self.entity_idx]
        if not valid_seeds:
            return dict.fromkeys(self.all_entities, 0.0)

        adj = self._build_adjacency_list()

        # 构建 COO 边列表
        src_list = []
        dst_list = []
        for u in range(n):
            for v in adj[u]:
                src_list.append(u)
                dst_list.append(v)

        src = np.array(src_list, dtype=np.int64)
        dst = np.array(dst_list, dtype=np.int64)

        # 出度
        out_degree = np.array([len(adj[i]) for i in range(n)], dtype=np.float64)
        out_degree_safe = np.where(out_degree > 0, out_degree, 1.0)

        # 种子节点索引
        seed_indices = np.array(
            [self.entity_idx[e] for e in valid_seeds], dtype=np.int64
        )
        n_seeds = len(seed_indices)

        # 初始化：种子实体均匀分布
        pr = np.zeros(n, dtype=np.float64)
        pr[seed_indices] = 1.0 / n_seeds

        # 跳转向量：只跳转到种子
        teleport = np.zeros(n, dtype=np.float64)
        teleport[seed_indices] = (1 - damping) / n_seeds

        for _ in range(max_iter):
            edge_contrib = pr[src] / out_degree_safe[src]
            new_pr = teleport.copy()
            np.add.at(new_pr, dst, damping * edge_contrib)

            # 悬挂节点处理
            dangling = pr[out_degree == 0].sum()
            if dangling > 0:
                new_pr[seed_indices] += damping * dangling / n_seeds

            # 归一化
            new_pr = new_pr / new_pr.sum()

            diff = np.sum(np.abs(new_pr - pr))
            pr = new_pr
            if diff < tol:
                break

        return {self.idx_entity[i]: float(pr[i]) for i in range(n)}

    # ========================
    # 节点中心性
    # ========================

    def degree_centrality(self) -> dict:
        """度中心性 (Degree Centrality)

        节点的度数 / (n-1)，反映节点的直接连接数量。

        Returns:
            {实体名: 度中心性分数} 字典
        """
        n = len(self.all_entities)
        if n <= 1:
            return dict.fromkeys(self.all_entities, 0.0)

        adj = self._build_adjacency_list()
        return {self.idx_entity[i]: len(adj[i]) / (n - 1) for i in range(n)}

    def closeness_centrality(self, max_hops: int = 5) -> dict:
        """接近中心性 (Closeness Centrality)

        节点到其他所有节点的平均距离的倒数，反映节点在图中的"中心"程度。
        为了计算效率，限制最大搜索跳数。

        Args:
            max_hops: BFS 最大搜索跳数

        Returns:
            {实体名: 接近中心性分数} 字典
        """
        n = len(self.all_entities)
        if n <= 1:
            return dict.fromkeys(self.all_entities, 0.0)

        adj = self._build_adjacency_list()
        centrality = {}

        for i in range(n):
            # BFS 计算从 i 到所有可达节点的距离
            distances = {i: 0}
            queue = [i]
            head = 0

            while head < len(queue):
                curr = queue[head]
                head += 1
                if distances[curr] >= max_hops:
                    continue
                for neighbor in adj[curr]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[curr] + 1
                        queue.append(neighbor)

            # 接近中心性 = (可达节点数) / (总距离)
            # 归一化: (可达节点数 - 1) / (n - 1) * (可达节点数 - 1) / 总距离
            reachable = len(distances) - 1  # 去掉自己
            if reachable > 0:
                total_dist = sum(distances.values()) - distances[i]  # 去掉自己的 0
                # Wasserman-Faust 归一化
                centrality[self.idx_entity[i]] = (reachable / (n - 1)) * (
                    reachable / total_dist
                )
            else:
                centrality[self.idx_entity[i]] = 0.0

        return centrality

    def betweenness_centrality_approx(self, k: int = None) -> dict:
        """介数中心性近似 (Betweenness Centrality - Approximate)

        使用 Brandes 算法的采样版本，随机选 k 个节点作为源点计算。
        介数中心性衡量节点作为其他节点之间最短路径"桥梁"的次数。

        Args:
            k: 采样源点数，None 则使用全部节点（精确计算，大图较慢）

        Returns:
            {实体名: 介数中心性分数} 字典
        """
        n = len(self.all_entities)
        if n <= 2:
            return dict.fromkeys(self.all_entities, 0.0)

        adj = self._build_adjacency_list()

        # 确定采样节点
        if k is None or k >= n:
            sources = list(range(n))
        else:
            # 随机采样 k 个节点
            rng = np.random.RandomState(42)
            sources = rng.choice(n, size=k, replace=False).tolist()

        betweenness = np.zeros(n, dtype=np.float64)

        for s in sources:
            # BFS 从 s 出发
            stack = []
            predecessors = [[] for _ in range(n)]
            sigma = np.zeros(n, dtype=np.float64)
            sigma[s] = 1.0
            dist = np.full(n, -1, dtype=np.int32)
            dist[s] = 0
            queue = [s]
            q_head = 0

            while q_head < len(queue):
                v = queue[q_head]
                q_head += 1
                stack.append(v)

                for w in adj[v]:
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

            # 反向累积
            delta = np.zeros(n, dtype=np.float64)
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        # 归一化（有向图除 (n-1)(n-2)，无向图除 2）
        norm = (n - 1) * (n - 2)
        if norm > 0:
            # 采样缩放
            scale = n / len(sources) if sources else 1.0
            betweenness = betweenness * scale / norm / 2  # 无向图除 2

        return {self.idx_entity[i]: float(betweenness[i]) for i in range(n)}

    # ========================
    # 连接组件 & 聚类系数
    # ========================

    def connected_components(self) -> list:
        """连接组件检测 (Connected Components)

        使用 BFS 找出图中所有连通分量。

        Returns:
            列表，每个元素是一个连通分量的实体列表，按大小降序排列
        """
        n = len(self.all_entities)
        if n == 0:
            return []

        adj = self._build_adjacency_list()
        visited = [False] * n
        components = []

        for i in range(n):
            if not visited[i]:
                # BFS 找连通分量
                component = []
                queue = [i]
                visited[i] = True
                head = 0

                while head < len(queue):
                    curr = queue[head]
                    head += 1
                    component.append(self.idx_entity[curr])

                    for neighbor in adj[curr]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)

                components.append(component)

        # 按大小降序排列
        components.sort(key=len, reverse=True)
        return components

    def clustering_coefficient(self) -> dict:
        """局部聚类系数 (Local Clustering Coefficient)

        节点的邻居之间实际存在的边数 / 可能存在的最大边数。
        反映节点的邻居之间有多"紧密"。

        Returns:
            {实体名: 聚类系数} 字典
        """
        n = len(self.all_entities)
        if n == 0:
            return {}

        adj = self._build_adjacency_list()
        adj_sets = [set(neighbors) for neighbors in adj]

        result = {}
        for i in range(n):
            neighbors = adj_sets[i]
            k = len(neighbors)
            if k < 2:
                result[self.idx_entity[i]] = 0.0
                continue

            # 计算邻居之间的边数
            links = 0
            neighbor_list = list(neighbors)
            for a in range(len(neighbor_list)):
                for b in range(a + 1, len(neighbor_list)):
                    if neighbor_list[b] in adj_sets[neighbor_list[a]]:
                        links += 1

            # 可能的最大边数 = k * (k-1) / 2
            max_links = k * (k - 1) / 2
            result[self.idx_entity[i]] = links / max_links if max_links > 0 else 0.0

        return result

    def global_clustering_coefficient(self) -> float:
        """全局聚类系数 (Global Clustering Coefficient)

        整个图的三元组闭合比例 = 3 * 三角形数 / 三元组总数。
        也叫传递性 (Transitivity)。

        Returns:
            全局聚类系数 (0.0 - 1.0)
        """
        n = len(self.all_entities)
        if n < 3:
            return 0.0

        adj = self._build_adjacency_list()
        adj_sets = [set(neighbors) for neighbors in adj]

        triangles = 0  # 每个三角形会被计数 3 次（每个顶点各一次）
        triples = 0  # 长度为 2 的路径数（中间节点视角）

        for i in range(n):
            neighbors = list(adj_sets[i])
            k = len(neighbors)
            if k < 2:
                continue

            # 以 i 为中心的三元组数量
            triples += k * (k - 1) / 2

            # 邻居之间的边数（即三角形数）
            for a in range(len(neighbors)):
                for b in range(a + 1, len(neighbors)):
                    if neighbors[b] in adj_sets[neighbors[a]]:
                        triangles += 1

        # 全局聚类系数 = 3 * 三角形数 / 三元组总数
        # triangles 已经是每个三角形被数 3 次（每个顶点一次），
        # 但因为我们是从中间节点数的三元组，而三角形每个角都是一个中间节点，
        # 所以 triangles 的计数就是 3 * 实际三角形数
        if triples > 0:
            return triangles / triples
        return 0.0

    # ========================
    # 辅助方法
    # ========================

    def _build_adjacency_list(self) -> list:
        """构建无向图邻接表（用于图算法），带缓存

        Returns:
            list[set[int]]: 每个节点的邻居索引集合
        """
        if self._adj_cache is not None:
            return self._adj_cache

        n = len(self.all_entities)
        adj = [set() for _ in range(n)]

        for entity, relations in self.entity_relations.items():
            if entity not in self.entity_idx:
                continue
            u = self.entity_idx[entity]
            for _, target in relations:
                if target in self.entity_idx:
                    v = self.entity_idx[target]
                    adj[u].add(v)
                    adj[v].add(u)

        self._adj_cache = adj
        return adj
