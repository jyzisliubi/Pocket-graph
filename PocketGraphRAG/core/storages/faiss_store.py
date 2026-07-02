"""FAISS 向量存储后端（默认实现）

包装 PocketGraphRAG.build_index.FAISSIndex，使其满足 VectorStore 抽象接口。
保留向后兼容：原有 FAISSIndex 的所有方法（build/add_chunks/search/save/load）继续可用。

新代码推荐：
  from PocketGraphRAG.core.storages import FAISSVectorStore
  store = FAISSVectorStore(model=model, dimension=512)
  store.add([("text", {"entity": "E1"})], embeddings)
  results = store.search(query_vec, top_k=5)

旧代码兼容：
  from PocketGraphRAG.build_index import FAISSIndex  # 仍可用
  index = FAISSIndex(); index.build(chunks, model)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import VectorStore


class FAISSVectorStore(VectorStore):
    """FAISS + numpy embeddings 缓存的向量存储。

    内部复用 PocketGraphRAG.build_index.FAISSIndex 的实现（包括增量 add/remove_by_entity
    的 embeddings 缓存机制），通过适配器模式暴露 VectorStore 接口。
    """

    def __init__(self, model=None, dimension: int = 512, _delegate=None):
        """
        Args:
            model: SentenceTransformer 模型实例，用于 search(query_str) 时编码查询文本；
                   若调用方只传 query_vec（已编码向量），可不传 model
            dimension: 向量维度，仅在 _delegate 未提供时用作初始 FAISS 索引维度
            _delegate: 已存在的 FAISSIndex 实例（用于包装存量索引），不传则创建新的
        """
        if _delegate is not None:
            self._idx = _delegate
        else:
            # 懒加载，避免 import 时硬依赖 faiss
            from ...build_index import FAISSIndex

            self._idx = FAISSIndex(dimension=dimension)
        self.model = model or self._idx.model

    # ==========================
    # VectorStore 抽象接口实现
    # ==========================

    def add(self, items: List[Tuple[str, dict]], embeddings) -> int:
        """批量入库。embeddings 必须是 (n, dim) 的 ndarray 或可转 ndarray 的对象"""
        if not items:
            return 0
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings 必须是 2D 矩阵，收到 shape={embeddings.shape}"
            )

        # 构造 FAISSIndex.add_chunks 所需的 chunks 结构
        chunks = [{"text": text, "metadata": meta} for text, meta in items]
        # 跳过 model.encode，直接用外部传入的 embeddings
        return self._add_chunks_with_embeddings(chunks, embeddings)

    def _add_chunks_with_embeddings(self, chunks, embeddings) -> int:
        """内部方法：用已编码的 embeddings 直接 append 到 FAISSIndex"""

        if self._idx.index is None:
            raise ValueError("索引未初始化，请先 build() 或 load()")

        # 维度校验
        if embeddings.shape[1] != self._idx.dimension:
            raise ValueError(
                f"embeddings 维度 {embeddings.shape[1]} 与索引维度 {self._idx.dimension} 不一致"
            )

        self._idx.index.add(embeddings)
        self._idx.texts.extend([c["text"] for c in chunks])
        self._idx.metadatas.extend([c["metadata"] for c in chunks])
        if self._idx._embeddings is None:
            self._idx._embeddings = embeddings
        else:
            self._idx._embeddings = np.vstack([self._idx._embeddings, embeddings])
        return len(chunks)

    def search(self, query_vec=None, top_k: int = 5, query: str = None):
        """向量检索。

        Args:
            query_vec: 查询向量 (dim,) 或 (1, dim)，优先使用
            top_k: 返回数量
            query: 查询文本（需 model），当 query_vec 为 None 时使用

        Returns:
            [(text, score, metadata), ...] 按相似度降序
        """
        if query_vec is None and query is None:
            raise ValueError("search 需要 query_vec 或 query 至少一个参数")

        if query_vec is None:
            if self.model is None:
                raise ValueError("query_vec=None 且未配置 model，无法编码 query 文本")
            query_vec = self.model.encode([query], normalize_embeddings=True)
            query_vec = np.asarray(query_vec, dtype="float32")

        query_vec = np.asarray(query_vec, dtype="float32")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        scores, indices = self._idx.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                (self._idx.texts[idx], float(score), self._idx.metadatas[idx])
            )
        return results

    def remove_by_entity(self, entity: str) -> int:
        return self._idx.remove_by_entity(entity)

    def save(self, path: str) -> None:
        self._idx.save(path)

    def load(self, path: str) -> None:
        """从目录加载（需要先在构造时提供 model）"""
        from ...build_index import FAISSIndex

        if self.model is None:
            raise ValueError("load 需要 model，请在构造时传入 model=...")
        self._idx = FAISSIndex.load(path, self.model)

    def __len__(self) -> int:
        return len(self._idx.texts)

    # ==========================
    # 向后兼容属性
    # ==========================

    @property
    def texts(self) -> List[str]:
        return self._idx.texts

    @property
    def metadatas(self) -> List[dict]:
        return self._idx.metadatas

    @property
    def dimension(self) -> int:
        return self._idx.dimension

    @property
    def delegate(self):
        """暴露底层 FAISSIndex 实例（向后兼容旧代码直接访问 .index / ._embeddings）"""
        return self._idx

    # 旧 API 兼容方法
    def build(self, chunks: list, model) -> None:
        """旧 API：用 chunks + model 全量构建"""
        self._idx.build(chunks, model)
        self.model = model

    def add_chunks(self, chunks: list, model=None) -> int:
        """旧 API：增量追加 chunks（内部会调 model.encode）"""
        return self._idx.add_chunks(chunks, model)
