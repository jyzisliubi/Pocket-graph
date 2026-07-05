"""内存图存储后端（默认实现）

包装 entity_relations / reverse_relations dict，满足 GraphStore 抽象接口。
适合小到中等规模 KG（万级三元组），无需外部数据库。

对于大规模 KG（百万级三元组），请实现 Neo4jGraphStore 或 PgVectorGraphStore。
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional, Tuple

from .base import GraphStore


class InMemoryGraphStore(GraphStore):
    """内存三元组图存储。

    内部维护两个 dict：
      entity_relations:   {head: [(relation, tail), ...]}
      reverse_relations:  {tail: [(head, relation), ...]}

    与 PocketGraphRAG.data_processor.KGProcessor 的输出格式完全兼容，
    可直接用 KGProcessor.entity_relations 初始化。
    """

    def __init__(
        self,
        entity_relations: dict = None,
        reverse_relations: dict = None,
    ):
        """
        Args:
            entity_relations: {head: [(relation, tail), ...]}，None 则空图
            reverse_relations: {tail: [(head, relation), ...]}，None 则从 entity_relations 反推
        """
        self.entity_relations: dict = dict(entity_relations or {})
        if reverse_relations is not None:
            self.reverse_relations: dict = dict(reverse_relations)
        else:
            self.reverse_relations = {}
            for head, rels in self.entity_relations.items():
                for rel, tail in rels:
                    self.reverse_relations.setdefault(tail, []).append((head, rel))

        # 去重集合：保证 add_triple 幂等
        self._triple_set = set()
        for head, rels in self.entity_relations.items():
            for rel, tail in rels:
                self._triple_set.add((head, rel, tail))

    # ==========================
    # GraphStore 抽象接口实现
    # ==========================

    def add_triple(self, head: str, relation: str, tail: str) -> bool:
        key = (head, relation, tail)
        if key in self._triple_set:
            return False
        self._triple_set.add(key)
        self.entity_relations.setdefault(head, []).append((relation, tail))
        self.reverse_relations.setdefault(tail, []).append((head, relation))
        return True

    def add_triples(self, triples: Iterable[Tuple[str, str, str]]) -> int:
        added = 0
        for h, r, t in triples:
            if self.add_triple(h, r, t):
                added += 1
        return added

    def neighbors(self, entity: str, hops: int = 2) -> List[str]:
        """BFS 扩展：返回 entity 在 hops 跳内的所有可达实体（含自身）。

        若 entity 不在图中，返回空列表。
        """
        # entity 不在图中：返回空
        if entity not in self.entity_relations and entity not in self.reverse_relations:
            return []
        if hops <= 0:
            return [entity]

        visited = {entity}
        queue = deque([(entity, 0)])
        while queue:
            cur, depth = queue.popleft()
            if depth >= hops:
                continue
            # 出边
            for rel, tail in self.entity_relations.get(cur, []):
                if tail not in visited:
                    visited.add(tail)
                    queue.append((tail, depth + 1))
            # 入边
            for head, rel in self.reverse_relations.get(cur, []):
                if head not in visited:
                    visited.add(head)
                    queue.append((head, depth + 1))
        return list(visited)

    def relations_of(self, entity: str) -> List[Tuple[str, str]]:
        return list(self.entity_relations.get(entity, []))

    def reverse_relations_of(self, entity: str) -> List[Tuple[str, str]]:
        return list(self.reverse_relations.get(entity, []))

    def all_entities(self) -> List[str]:
        return sorted(
            set(self.entity_relations.keys()) | set(self.reverse_relations.keys())
        )

    def all_relations(self) -> List[str]:
        rels = set()
        for rels_list in self.entity_relations.values():
            for r, _ in rels_list:
                rels.add(r)
        return sorted(rels)

    def __len__(self) -> int:
        return len(self._triple_set)

    def cleanup_orphan_entities(self) -> int:
        """清理孤儿实体（v0.3.7：对标 graphrag v3.0.9）

        删除图中没有任何边的实体（entity_relations 和 reverse_relations 中
        对应的列表均为空的实体）。这些实体在三元组删除后遗留，是 phantom entities。

        Returns:
            清理的孤儿实体数量
        """
        orphans = []
        all_entities = set(self.entity_relations.keys()) | set(
            self.reverse_relations.keys()
        )
        for e in all_entities:
            has_outgoing = bool(self.entity_relations.get(e))
            has_incoming = bool(self.reverse_relations.get(e))
            if not has_outgoing and not has_incoming:
                orphans.append(e)
        for e in orphans:
            self.entity_relations.pop(e, None)
            self.reverse_relations.pop(e, None)
        return len(orphans)

    # ==========================
    # 图算法（基础实现，大规模图建议用 NetworkX 或 Neo4j）
    # ==========================

    def pagerank(
        self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> dict:
        """幂迭代 PageRank。

        对于 >10k 实体的图，建议改用 NetworkX 的 pagerank（C 实现，更快）。
        """
        import numpy as np

        entities = self.all_entities()
        n = len(entities)
        if n == 0:
            return {}
        if n == 1:
            return {entities[0]: 1.0}

        idx = {e: i for i, e in enumerate(entities)}
        # 构建出度 + 转移矩阵
        out_degree = np.zeros(n)
        # 用稀疏 COO 风格记录边
        rows, cols = [], []
        for head, rels in self.entity_relations.items():
            if head not in idx:
                continue
            h_i = idx[head]
            for rel, tail in rels:
                if tail not in idx:
                    continue
                t_i = idx[tail]
                rows.append(h_i)
                cols.append(t_i)
                out_degree[h_i] += 1

        if not rows:
            # 无边图：均匀分布
            return dict.fromkeys(entities, 1.0 / n)

        # 转移概率：M[i][j] = 从 j 转移到 i 的概率 = 1/out_degree[j]（如果 j→i 有边）
        # PageRank 迭代：v_new = damping * M @ v + (1-damping) / n
        scores = np.ones(n) / n
        teleport = np.ones(n) / n

        for _ in range(max_iter):
            new_scores = (1 - damping) * teleport.copy()
            # 累加每个出边贡献
            for h_i, t_i in zip(rows, cols):
                if out_degree[h_i] > 0:
                    new_scores[t_i] += damping * scores[h_i] / out_degree[h_i]
            # 处理 dangling nodes（无出度的节点）
            dangling_sum = sum(scores[i] for i in range(n) if out_degree[i] == 0)
            new_scores += damping * dangling_sum / n * teleport

            if np.abs(new_scores - scores).sum() < tol:
                break
            scores = new_scores

        # 归一化
        total = scores.sum()
        if total > 0:
            scores = scores / total
        return {entities[i]: float(scores[i]) for i in range(n)}

    def communities(self, max_iter: int = 50) -> List[List[str]]:
        """标签传播社区发现（简单实现）"""
        import numpy as np

        entities = self.all_entities()
        n = len(entities)
        if n == 0:
            return []

        idx = {e: i for i, e in enumerate(entities)}
        # 构建邻接表（无向）
        adj = [set() for _ in range(n)]
        for head, rels in self.entity_relations.items():
            if head not in idx:
                continue
            h_i = idx[head]
            for rel, tail in rels:
                if tail not in idx:
                    continue
                t_i = idx[tail]
                adj[h_i].add(t_i)
                adj[t_i].add(h_i)

        # 初始化每个节点一个独立标签
        labels = list(range(n))
        rng = np.random.RandomState(42)

        for _ in range(max_iter):
            changed = False
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                if not adj[i]:
                    continue
                # 选邻居中最多的标签
                neighbor_labels = [labels[j] for j in adj[i]]
                if not neighbor_labels:
                    continue
                counts = {}
                for lbl in neighbor_labels:
                    counts[lbl] = counts.get(lbl, 0) + 1
                max_count = max(counts.values())
                candidates = [lbl for lbl, c in counts.items() if c == max_count]
                new_label = rng.choice(candidates)
                if new_label != labels[i]:
                    labels[i] = new_label
                    changed = True
            if not changed:
                break

        # 按标签分组
        groups = {}
        for i, lbl in enumerate(labels):
            groups.setdefault(lbl, []).append(entities[i])
        return list(groups.values())

    def shortest_path(
        self, start: str, end: str, max_hops: int = 5
    ) -> Optional[List[str]]:
        """BFS 最短路径"""
        if start == end:
            return (
                [start]
                if start in self.entity_relations or start in self.reverse_relations
                else None
            )

        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            cur, path = queue.popleft()
            if len(path) - 1 >= max_hops:
                continue
            neighbors = set()
            for rel, tail in self.entity_relations.get(cur, []):
                neighbors.add(tail)
            for head, rel in self.reverse_relations.get(cur, []):
                neighbors.add(head)
            for nxt in neighbors:
                if nxt == end:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

    def personalized_pagerank(
        self,
        seed_entities: List[str],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict:
        """个性化 PageRank (PPR) - NumPy 向量化优化版

        从种子实体出发，计算其他实体与种子实体的相关性分数。
        与普通 PageRank 的区别：随机跳转只跳转到种子实体，而非所有实体。

        Args:
            seed_entities: 种子实体列表
            damping: 阻尼系数，默认 0.85
            max_iter: 最大迭代次数
            tol: 收敛阈值

        Returns:
            {实体名: PPR 分数} 字典
        """
        import numpy as np

        entities = self.all_entities()
        n = len(entities)
        if n == 0:
            return {}

        idx = {e: i for i, e in enumerate(entities)}

        # 过滤有效种子
        valid_seeds = [e for e in seed_entities if e in idx]
        if not valid_seeds:
            return dict.fromkeys(entities, 0.0)

        # 构建 COO 边列表（无向：出边+入边都计入邻接）
        src_list: list = []
        dst_list: list = []
        for head, rels in self.entity_relations.items():
            if head not in idx:
                continue
            h_i = idx[head]
            for rel, tail in rels:
                if tail not in idx:
                    continue
                t_i = idx[tail]
                src_list.append(h_i)
                dst_list.append(t_i)

        # 出度（含入边贡献，模拟无向图）
        out_degree = np.zeros(n, dtype=np.float64)
        for h_i, t_i in zip(src_list, dst_list):
            out_degree[h_i] += 1.0
            out_degree[t_i] += 1.0  # 反向边也计入
            src_list.append(t_i)
            dst_list.append(h_i)

        out_degree_safe = np.where(out_degree > 0, out_degree, 1.0)
        src = np.array(src_list, dtype=np.int64)
        dst = np.array(dst_list, dtype=np.int64)

        # 种子节点索引
        seed_indices = np.array(
            [idx[e] for e in valid_seeds], dtype=np.int64
        )
        n_seeds = len(seed_indices)

        # 初始分布：种子均匀
        pr = np.zeros(n, dtype=np.float64)
        pr[seed_indices] = 1.0 / n_seeds

        # 跳转向量：只跳种子
        teleport = np.zeros(n, dtype=np.float64)
        teleport[seed_indices] = (1 - damping) / n_seeds

        for _ in range(max_iter):
            edge_contrib = pr[src] / out_degree_safe[src]
            new_pr = teleport.copy()
            np.add.at(new_pr, dst, damping * edge_contrib)

            # 悬挂节点处理：把质量均分给种子
            dangling = pr[out_degree == 0].sum()
            if dangling > 0:
                new_pr[seed_indices] += damping * dangling / n_seeds

            # 归一化
            total = new_pr.sum()
            if total > 0:
                new_pr = new_pr / total

            diff = np.sum(np.abs(new_pr - pr))
            pr = new_pr
            if diff < tol:
                break

        return {entities[i]: float(pr[i]) for i in range(n)}
