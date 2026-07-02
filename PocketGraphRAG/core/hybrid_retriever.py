"""混合检索：BM25 关键词召回 + 向量语义召回，RRF 融合 + 可选 Rerank 精排。

作为 GraphRAG 的 fallback 检索方案，当知识图谱数据不足时启用。
适配 PocketGraphRAG 的 FAISSIndex 和 SentenceTransformer 接口。
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

# Fallback RAG 配置（从环境变量读取，默认值与 PocketBrain 保持一致）
USE_HYBRID = os.getenv("USE_HYBRID", "true").lower() == "true"
USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"
TOP_K = int(os.getenv("TOP_K", "5"))
RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", "15"))
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.0"))


def _rrf_fuse(rank_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion 融合多个排序列表。

    每个文档的 RRF 分数 = sum( 1 / (k + rank_i) )，rank 从 1 开始。
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for ranked in rank_lists:
        for rank, c in enumerate(ranked, start=1):
            key = c.get("text", "")[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = c

    fused = []
    for key, score in scores.items():
        c = dict(doc_map[key])
        c["fusion_score"] = score
        fused.append(c)
    fused.sort(key=lambda x: x["fusion_score"], reverse=True)
    return fused


class HybridRetriever:
    """混合检索器：BM25 + 向量 + RRF 融合。

    适配 PocketGraphRAG 的 FAISSIndex 接口。
    """

    def __init__(self, index, model):
        """
        Args:
            index: PocketGraphRAG.build_index.FAISSIndex 实例
            model: SentenceTransformer 实例
        """
        self._index = index
        self._model = model
        self._bm25 = None
        self._bm25_docs: List[str] = []

    def _ensure_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25_docs = [self._tokenize(text) for text in self._index.texts]
            self._bm25 = BM25Okapi(self._bm25_docs)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文友好的轻量分词：保留英文词，中文按字。"""
        import re

        tokens = []
        for word in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text):
            tokens.append(word)
        return tokens

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_bm25: bool = True,
        use_vector: bool = True,
        use_rerank: bool = True,
    ) -> List[Dict]:
        """混合检索。

        Args:
            query: 用户问题
            top_k: 最终返回数量
            use_bm25 / use_vector / use_rerank: 各阶段开关
        """
        top_k = top_k or TOP_K
        recall_k = max(top_k * 3, RECALL_TOP_K)

        rank_lists = []

        if use_vector:
            # FAISSIndex.search 接受文本，内部编码
            vec_results_raw = self._index.search(query, recall_k)
            vec_results = []
            for text, score, meta in vec_results_raw:
                vec_results.append(
                    {
                        "text": text,
                        "source": meta.get("entity", "未知"),
                        "score": float(score),
                        "meta": meta,
                    }
                )
            if vec_results:
                rank_lists.append(vec_results)

        if use_bm25:
            self._ensure_bm25()
            tokens = self._tokenize(query)
            if tokens and self._index.texts:
                bm25_scores = self._bm25.get_scores(tokens)
                idx_sorted = bm25_scores.argsort()[::-1][:recall_k]
                bm25_results = []
                for idx in idx_sorted:
                    if bm25_scores[idx] <= 0:
                        continue
                    meta = (
                        self._index.metadatas[idx]
                        if idx < len(self._index.metadatas)
                        else {}
                    )
                    bm25_results.append(
                        {
                            "text": self._index.texts[idx],
                            "source": meta.get("entity", "未知"),
                            "bm25_score": float(bm25_scores[idx]),
                            "meta": meta,
                        }
                    )
                if bm25_results:
                    rank_lists.append(bm25_results)

        if len(rank_lists) == 1:
            fused = rank_lists[0]
        elif len(rank_lists) > 1:
            fused = _rrf_fuse(rank_lists)
        else:
            return []

        # 精排
        if use_rerank and USE_RERANK and len(fused) > 0:
            from PocketGraphRAG.core.reranker import get_reranker

            threshold = RERANK_SCORE_THRESHOLD if USE_RERANK else 0.0
            return get_reranker().rerank(
                query, fused, top_k=top_k, score_threshold=threshold
            )

        return fused[:top_k]
