"""Neo4j 图存储后端（可选）

需安装：pip install neo4j

特性：
  - 原生图数据库，Cypher 查询语言
  - 适合大规模 KG（百万级三元组）
  - 内置图算法库（GDS）：PageRank / 社区发现 / 最短路径
  - 支持 BFS 邻域扩展、最短路径等图查询

对标 microsoft/graphrag 的 Neo4j 集成和 LightRAG 的 NanoVectorDB。
对于百万级三元组的大规模 KG，Neo4j 比 InMemoryGraphStore 性能更好。

配置：
  POCKET_NEO4J_URI=bolt://localhost:7687
  POCKET_NEO4J_USER=neo4j
  POCKET_NEO4J_PASSWORD=password

使用方式：
  # 通过环境变量切换后端
  export POCKET_GRAPH_BACKEND=neo4j
  # 或通过工厂创建
  from PocketGraphRAG.core.storages import get_graph_store
  gs = get_graph_store(backend="neo4j", uri="bolt://localhost:7687", ...)
"""

from __future__ import annotations

import os
from collections import deque
from typing import Iterable, List, Optional, Tuple

from .base import GraphStore


class Neo4jGraphStore(GraphStore):
    """Neo4j 图存储。

    数据模型：
      - 节点：(:Entity {name: "实体名"})
      - 关系：(:Entity)-[:RELATION {name: "关系名"}]->(:Entity)

    Cypher 查询示例：
      - 邻域：MATCH (e:Entity {name: $entity})-[*1..2]-(n) RETURN DISTINCT n
      - PageRank：CALL gds.pageRank.stream('entityGraph')
      - 社区：CALL gds.louvain.stream('entityGraph')

    注意：图算法（pagerank/communities）需要 Neo4j GDS 库。
    未安装 GDS 时回退到纯 Python 实现（小图可用）。
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        entity_relations: Optional[dict] = None,
        reverse_relations: Optional[dict] = None,
    ):
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Neo4jGraphStore 需要 neo4j 驱动。"
                "请安装：pip install neo4j"
            ) from e

        self.uri = uri or os.environ.get(
            "POCKET_NEO4J_URI", "bolt://localhost:7687"
        )
        self.user = user or os.environ.get("POCKET_NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("POCKET_NEO4J_PASSWORD", "")
        self.database = database
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        # 验证连接
        try:
            self._driver.verify_connectivity()
        except Exception as e:
            self._driver.close()
            raise ConnectionError(f"Neo4j 连接失败: {e}") from e

        # 初始化约束 + 索引
        self._init_schema()

        # 如果传入了初始数据，批量导入
        if entity_relations:
            self._bulk_import(entity_relations, reverse_relations)

    def _init_schema(self):
        """创建约束和索引（幂等）"""
        with self._driver.session(database=self.database) as session:
            # Entity 节点唯一约束
            session.run(
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def _bulk_import(
        self,
        entity_relations: dict,
        reverse_relations: Optional[dict] = None,
    ):
        """批量导入三元组数据。

        用 UNWIND 批量插入，比逐条 add_triple 快 10-100x。
        """
        triples = []
        for head, rels in entity_relations.items():
            for rel, tail in rels:
                triples.append({"head": head, "relation": rel, "tail": tail})

        if not triples:
            return

        with self._driver.session(database=self.database) as session:
            # 批量创建节点 + 关系
            session.run(
                """
                UNWIND $triples AS t
                MERGE (h:Entity {name: t.head})
                MERGE (tail:Entity {name: t.tail})
                MERGE (h)-[r:RELATION {name: t.relation}]->(tail)
                """,
                triples=triples,
            )

    # ==========================
    # GraphStore 抽象接口实现
    # ==========================

    def add_triple(self, head: str, relation: str, tail: str) -> bool:
        """增加一条三元组（幂等）。"""
        with self._driver.session(database=self.database) as session:
            result = session.run(
                """
                MERGE (h:Entity {name: $head})
                MERGE (t:Entity {name: $tail})
                MERGE (h)-[r:RELATION {name: $relation}]->(t)
                RETURN count(r) AS created
                """,
                head=head,
                relation=relation,
                tail=tail,
            )
            # MERGE 不返回是否新建，用 properties 跟踪
            # 简化：始终返回 True（幂等语义）
            return True

    def add_triples(self, triples: Iterable[Tuple[str, str, str]]) -> int:
        """批量增加三元组。"""
        triple_list = [{"head": h, "relation": r, "tail": t} for h, r, t in triples]
        if not triple_list:
            return 0
        with self._driver.session(database=self.database) as session:
            session.run(
                """
                UNWIND $triples AS t
                MERGE (h:Entity {name: t.head})
                MERGE (tail:Entity {name: t.tail})
                MERGE (h)-[r:RELATION {name: t.relation}]->(tail)
                """,
                triples=triple_list,
            )
        return len(triple_list)

    def neighbors(self, entity: str, hops: int = 2) -> List[str]:
        """BFS 邻域扩展：返回 entity 在 hops 跳内的所有可达实体（含自身）。"""
        with self._driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $entity})-[*0..$hops]-(n:Entity)
                RETURN DISTINCT n.name AS name
                """,
                entity=entity,
                hops=hops,
            )
            return [record["name"] for record in result if record["name"]]

    def relations_of(self, entity: str) -> List[Tuple[str, str]]:
        """返回 [(relation, tail), ...]"""
        with self._driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $entity})-[r:RELATION]->(t:Entity)
                RETURN r.name AS relation, t.name AS tail
                """,
                entity=entity,
            )
            return [(record["relation"], record["tail"]) for record in result]

    def reverse_relations_of(self, entity: str) -> List[Tuple[str, str]]:
        """返回 [(head, relation), ...]"""
        with self._driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (h:Entity)-[r:RELATION]->(e:Entity {name: $entity})
                RETURN h.name AS head, r.name AS relation
                """,
                entity=entity,
            )
            return [(record["head"], record["relation"]) for record in result]

    def all_entities(self) -> List[str]:
        with self._driver.session(database=self.database) as session:
            result = session.run("MATCH (e:Entity) RETURN e.name AS name ORDER BY name")
            return [record["name"] for record in result]

    def all_relations(self) -> List[str]:
        with self._driver.session(database=self.database) as session:
            result = session.run(
                "MATCH ()-[r:RELATION]->() RETURN DISTINCT r.name AS name ORDER BY name"
            )
            return [record["name"] for record in result]

    def __len__(self) -> int:
        with self._driver.session(database=self.database) as session:
            result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS cnt")
            return result.single()["cnt"]

    # ==========================
    # 图算法（优先用 GDS，未安装时回退纯 Python）
    # ==========================

    def pagerank(
        self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> dict:
        """PageRank。优先用 Neo4j GDS，未安装时回退纯 Python。"""
        # 尝试 GDS
        try:
            with self._driver.session(database=self.database) as session:
                # 投影图（如果不存在）
                session.run(
                    "CALL gds.graph.project('entityGraph', 'Entity', 'RELATION')",
                )
                result = session.run(
                    "CALL gds.pageRank.stream('entityGraph', "
                    "{dampingFactor: $damping, maxIterations: $max_iter}) "
                    "YIELD nodeId, score "
                    "RETURN gds.util.asNode(nodeId).name AS name, score "
                    "ORDER BY score DESC",
                    damping=damping,
                    max_iter=max_iter,
                )
                return {record["name"]: record["score"] for record in result}
        except Exception:
            # GDS 未安装，回退纯 Python
            return self._pagerank_python(damping, max_iter, tol)

    def _pagerank_python(
        self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> dict:
        """纯 Python PageRank 回退实现（小图可用）。"""
        import numpy as np

        entities = self.all_entities()
        n = len(entities)
        if n == 0:
            return {}
        if n == 1:
            return {entities[0]: 1.0}

        idx = {e: i for i, e in enumerate(entities)}
        out_degree = np.zeros(n)
        rows, cols = [], []
        for entity in entities:
            rels = self.relations_of(entity)
            h_i = idx[entity]
            for _, tail in rels:
                if tail in idx:
                    t_i = idx[tail]
                    rows.append(h_i)
                    cols.append(t_i)
                    out_degree[h_i] += 1

        if not rows:
            return dict.fromkeys(entities, 1.0 / n)

        scores = np.ones(n) / n
        teleport = np.ones(n) / n
        for _ in range(max_iter):
            new_scores = (1 - damping) * teleport.copy()
            for h_i, t_i in zip(rows, cols):
                if out_degree[h_i] > 0:
                    new_scores[t_i] += damping * scores[h_i] / out_degree[h_i]
            dangling_sum = sum(
                scores[i] for i in range(n) if out_degree[i] == 0
            )
            new_scores += damping * dangling_sum / n * teleport
            if np.abs(new_scores - scores).sum() < tol:
                break
            scores = new_scores
        return {entities[i]: float(scores[i]) for i in range(n)}

    def personalized_pagerank(
        self,
        seed_entities: List[str],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict:
        """个性化 PageRank。

        优先用 Neo4j GDS 的 pageRank with sourceNodes。
        """
        if not seed_entities:
            return {}
        try:
            with self._driver.session(database=self.database) as session:
                # 用 seed 节点做 PPR
                result = session.run(
                    """
                    CALL gds.pageRank.stream('entityGraph', {
                        dampingFactor: $damping,
                        maxIterations: $max_iter,
                        sourceNodes: [n in $seeds | MATCH (e:Entity {name: n}) RETURN e]
                    })
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).name AS name, score
                    """,
                    damping=damping,
                    max_iter=max_iter,
                    seeds=seed_entities,
                )
                return {record["name"]: record["score"] for record in result}
        except Exception:
            # GDS 未安装，回退：用 InMemoryGraphStore 的 PPR
            return self._ppr_python(seed_entities, damping, max_iter, tol)

    def _ppr_python(
        self,
        seed_entities: List[str],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict:
        """纯 Python PPR 回退。"""
        import numpy as np

        entities = self.all_entities()
        n = len(entities)
        if n == 0 or not seed_entities:
            return {}
        idx = {e: i for i, e in enumerate(entities)}
        # 种子向量：均匀分布在 seed 上
        seed_vec = np.zeros(n)
        valid_seeds = [idx[s] for s in seed_entities if s in idx]
        if not valid_seeds:
            return {}
        seed_vec[valid_seeds] = 1.0 / len(valid_seeds)

        out_degree = np.zeros(n)
        rows, cols = [], []
        for entity in entities:
            rels = self.relations_of(entity)
            h_i = idx[entity]
            for _, tail in rels:
                if tail in idx:
                    t_i = idx[tail]
                    rows.append(h_i)
                    cols.append(t_i)
                    out_degree[h_i] += 1

        if not rows:
            return {entities[i]: float(seed_vec[i]) for i in range(n)}

        scores = seed_vec.copy()
        for _ in range(max_iter):
            new_scores = (1 - damping) * seed_vec
            for h_i, t_i in zip(rows, cols):
                if out_degree[h_i] > 0:
                    new_scores[t_i] += damping * scores[h_i] / out_degree[h_i]
            if np.abs(new_scores - scores).sum() < tol:
                break
            scores = new_scores
        return {entities[i]: float(scores[i]) for i in range(n)}

    def communities(self, max_iter: int = 50) -> List[List[str]]:
        """社区发现。优先用 Neo4j GDS Louvain。"""
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    "CALL gds.louvain.stream('entityGraph', {maxIterations: $max_iter}) "
                    "YIELD nodeId, communityId "
                    "RETURN collect(gds.util.asNode(nodeId).name) AS members, communityId "
                    "ORDER BY communityId",
                    max_iter=max_iter,
                )
                return [record["members"] for record in result]
        except Exception:
            raise NotImplementedError(
                "社区发现需要 Neo4j GDS 库。未安装时请用 InMemoryGraphStore。"
            )

    def shortest_path(
        self, start: str, end: str, max_hops: int = 5
    ) -> Optional[List[str]]:
        """最短路径。用 Cypher 的 shortestPath 函数。"""
        with self._driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (s:Entity {name: $start}), (e:Entity {name: $end})
                CALL apoc.algo.shortestPath(s, e, 'RELATION', $max_hops)
                YIELD path
                RETURN [n IN nodes(path) | n.name] AS path_nodes
                LIMIT 1
                """,
                start=start,
                end=end,
                max_hops=max_hops,
            )
            record = result.single()
            if record is None:
                return None
            return record["path_nodes"]

    def close(self):
        """关闭驱动连接"""
        if self._driver:
            self._driver.close()
