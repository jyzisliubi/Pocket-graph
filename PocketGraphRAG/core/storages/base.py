"""VectorStore / GraphStore 抽象基类

设计原则：
  - 抽象接口只规定"做什么"，不规定"怎么做"
  - 类型签名用最小公约数（list/tuple/dict），便于多后端实现
  - 所有方法都允许抛 NotImplementedError，让骨架后端可渐进实现

接口契约：
  VectorStore:
    - add(items, embeddings): 批量入库
    - search(query_vec, top_k): 向量检索
    - remove_by_entity(entity): 按实体删除（增量重建用）
    - save(path) / load(path): 持久化
    - __len__: 当前条数

  GraphStore:
    - add_triple(head, rel, tail): 增加三元组
    - neighbors(entity, hops): 邻域扩展（BFS）
    - relations_of(entity): 实体的关系列表
    - all_entities / all_relations: 唯一集合
    - pagerank / communities / shortest_path: 图算法（可选）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple


class VectorStore(ABC):
    """向量存储抽象基类。

    所有后端必须实现 add/search/remove_by_entity/save/load/size。
    子类可选择实现 query_encoding（默认用外部传入的 embedding）。
    """

    @abstractmethod
    def add(self, items: List[Tuple[str, dict]], embeddings) -> int:
        """批量添加文本块。

        Args:
            items: [(text, metadata), ...]，metadata 至少包含 "entity" 字段
            embeddings: 对应的向量矩阵 (n, dim)，与 items 等长

        Returns:
            实际新增条数
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vec, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """检索与 query_vec 最相似的 top_k 个文本块。

        Returns:
            [(text, score, metadata), ...] 按相似度降序
        """
        raise NotImplementedError

    @abstractmethod
    def remove_by_entity(self, entity: str) -> int:
        """删除某实体对应的所有条目（增量重建场景）。

        Returns:
            实际删除条数
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """持久化到 path（目录或文件，由实现决定）"""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """从 path 加载（与 save 配对）"""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """当前条目数"""
        raise NotImplementedError

    # 便利属性（子类可覆盖）
    @property
    def texts(self) -> List[str]:
        """所有文本（与索引顺序对齐），子类应覆盖"""
        raise NotImplementedError(
            f"{type(self).__name__}.texts 未实现；如需访问请实现该属性或使用 items()"
        )

    @property
    def metadatas(self) -> List[dict]:
        """所有元数据（与索引顺序对齐），子类应覆盖"""
        raise NotImplementedError(
            f"{type(self).__name__}.metadatas 未实现；如需访问请实现该属性"
        )


class GraphStore(ABC):
    """图存储抽象基类。

    所有后端必须实现三元组 CRUD 和邻域查询。
    图算法（pagerank/communities/path）为可选，未实现时抛 NotImplementedError。
    """

    @abstractmethod
    def add_triple(self, head: str, relation: str, tail: str) -> bool:
        """增加一条三元组。返回是否新增（重复返回 False）"""
        raise NotImplementedError

    @abstractmethod
    def add_triples(self, triples: Iterable[Tuple[str, str, str]]) -> int:
        """批量增加三元组。返回实际新增数"""
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, entity: str, hops: int = 2) -> List[str]:
        """BFS 扩展：从 entity 出发，返回 hops 跳内的所有实体（含自身）"""
        raise NotImplementedError

    @abstractmethod
    def relations_of(self, entity: str) -> List[Tuple[str, str]]:
        """返回 [(relation, tail), ...]，即 entity 作为 head 的所有出边"""
        raise NotImplementedError

    @abstractmethod
    def reverse_relations_of(self, entity: str) -> List[Tuple[str, str]]:
        """返回 [(head, relation), ...]，即 entity 作为 tail 的所有入边"""
        raise NotImplementedError

    @abstractmethod
    def all_entities(self) -> List[str]:
        """所有唯一实体"""
        raise NotImplementedError

    @abstractmethod
    def all_relations(self) -> List[str]:
        """所有唯一关系名"""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """三元组总数"""
        raise NotImplementedError

    # 图算法（可选实现）
    def pagerank(self, damping: float = 0.85, max_iter: int = 100) -> dict:
        """PageRank 中心性。未实现的后端应抛 NotImplementedError"""
        raise NotImplementedError(f"{type(self).__name__}.pagerank 未实现")

    def personalized_pagerank(
        self,
        seed_entities: List[str],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict:
        """个性化 PageRank (PPR)：从种子实体出发的相关性分数。

        与普通 PageRank 的区别：随机跳转只跳转到种子实体，而非所有实体。
        用于检索阶段为种子实体的邻域节点加权（高 PPR 的邻域实体优先召回）。

        未实现的后端应抛 NotImplementedError，调用方应回退到本地实现。
        """
        raise NotImplementedError(
            f"{type(self).__name__}.personalized_pagerank 未实现"
        )

    def communities(self, max_iter: int = 50) -> List[List[str]]:
        """社区发现。未实现的后端应抛 NotImplementedError"""
        raise NotImplementedError(f"{type(self).__name__}.communities 未实现")

    def shortest_path(
        self, start: str, end: str, max_hops: int = 5
    ) -> Optional[List[str]]:
        """最短路径。未实现的后端应抛 NotImplementedError"""
        raise NotImplementedError(f"{type(self).__name__}.shortest_path 未实现")

    def cleanup_orphan_entities(self) -> int:
        """清理孤儿实体（v0.3.7：对标 graphrag v3.0.9 phantom entities 清理）

        删除图中没有任何边（既非 head 也非 tail）的实体。
        这些实体通常在三元组删除后遗留，占用存储且不影响检索质量。

        Returns:
            清理的孤儿实体数量。未实现的后端返回 0。
        """
        return 0


class KVStore(ABC):
    """键值存储抽象基类。

    用于存储文档原文、chunk→doc_id 映射、抽取缓存等结构化 KV 数据。
    对标 LightRAG 的 BaseKVStorage，但默认实现零外部依赖（JSON 文件）。

    与 VectorStore 的区别：
      - VectorStore 存向量 + 文本块，按相似度检索
      - KVStore 存结构化键值对，按 key 精确查找

    接口契约：
      - get(key) / upsert(key, value) / delete(key)：单条 CRUD
      - get_by_ids(keys)：批量获取
      - keys() / __len__：枚举与计数
      - save(path) / load(path)：持久化
    """

    @abstractmethod
    def get(self, key: str) -> Optional[dict]:
        """按 key 获取一条记录，不存在返回 None"""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, key: str, value: dict) -> bool:
        """插入或更新。返回是否新增（已存在则更新并返回 False）"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除。返回是否删除成功（不存在返回 False）"""
        raise NotImplementedError

    @abstractmethod
    def get_by_ids(self, keys: List[str]) -> List[Optional[dict]]:
        """批量获取，返回与 keys 等长的列表，缺失项为 None"""
        raise NotImplementedError

    @abstractmethod
    def keys(self) -> List[str]:
        """所有 key（顺序由实现决定）"""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """持久化到 path"""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """从 path 加载（与 save 配对）"""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """当前条目数"""
        raise NotImplementedError
