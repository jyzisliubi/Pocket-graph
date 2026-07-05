"""Reranker：基于 CrossEncoder (bge-reranker) 的二次精排。

提升检索质量的关键组件。向量召回是"粗排"，CrossEncoder 对 (query, passage)
做交叉注意力打分，是"精排"——这是工业级 RAG 的标准两阶段检索范式。

中国网络环境：优先从本地路径加载（modelscope snapshot_download 预下载），
未配置本地路径时自动从 HuggingFace 下载（建议预先设置 HF_ENDPOINT=https://hf-mirror.com）。

v0.3.6 新增 KG-aware 模式：把 chunk 关联的 KG 实体/关系注入 passage，
让 CrossEncoder 能利用图谱结构信息做精排。LightRAG 默认开启 reranker 但不利用
KG 上下文，PocketGraphRAG 的 KG-aware 是独有增强。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

# Reranker 模型配置（通过环境变量覆盖）
# 默认 bge-reranker-v2-m3（多语言、效果优于 base）；与 config.RERANKER_MODEL 保持一致
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_LOCAL_PATH = os.environ.get("RERANKER_LOCAL_PATH", "")

# KG-aware 模式：把 chunk 关联的实体注入 passage 增强精排
# 默认开启（已通过 HotpotQA 验证有效）；设为 0 关闭
KG_AWARE_ENABLED = os.environ.get("POCKET_KG_AWARE_RERANKER", "1").lower() in (
    "1", "true", "yes", "on"
)
# 注入实体的最大数量（过多会稀释 query-passage 匹配信号）
KG_AWARE_MAX_ENTITIES = int(os.environ.get("POCKET_KG_AWARE_MAX_ENTITIES", "5"))


class Reranker:
    """CrossEncoder 二次精排。单例，懒加载模型。

    全项目共用一个 CrossEncoder 实例（避免 rag_system._rerank 与 core 各加载一份）。
    加载失败后缓存失败状态，不再重复重试，避免每次检索都尝试下载模型。
    """

    _instance = None
    _model = None
    _load_failed = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_available(self) -> bool:
        """CrossEncoder 是否已成功加载（或可加载）"""
        return self._model is not None

    def _load(self):
        """加载 CrossEncoder 模型。失败后置 _load_failed=True，不再重试。"""
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder

            model_name = RERANKER_MODEL
            local_path = RERANKER_LOCAL_PATH

            # 优先用本地路径（modelscope 预下载或手动指定）
            if local_path and os.path.isdir(local_path):
                load_from = local_path
            elif os.path.isdir(model_name):
                load_from = model_name
            else:
                # 用 snapshot_download 解析 HF 缓存路径
                # 修复 Windows 下符号链接断裂导致 CrossEncoder 加载失败的问题
                from huggingface_hub import snapshot_download

                try:
                    load_from = snapshot_download(model_name, local_files_only=True)
                except Exception:
                    # 本地缓存未命中，走镜像下载
                    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                    load_from = snapshot_download(model_name)
            self._model = CrossEncoder(load_from, max_length=512)
            logger.info(f"[Reranker] 已加载: {load_from}")
            return self._model
        except Exception as e:
            logger.warning(f"[Reranker] CrossEncoder 加载失败，后续将跳过精排: {e}")
            self._load_failed = True
            return None

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict]:
        """对召回的 chunks 做二次精排。

        Args:
            query: 用户问题
            chunks: 粗排召回的候选 [{"text", "source", ...}]
            top_k: 精排后保留数量
            score_threshold: rerank score 低于此值则丢弃（0 表示不过滤）

        Returns:
            精排后的 chunks，新增 rerank_score 字段，按分数降序
        """
        if not chunks:
            return []

        model = self._load()
        if model is None:
            return chunks[:top_k]

        # KG-aware：把 chunk 关联的实体注入 passage
        passages = [self._build_kg_aware_passage(c) for c in chunks]
        scores = model.predict([(query, p) for p in passages])

        scored = []
        for c, s in zip(chunks, scores):
            c = dict(c)
            c["rerank_score"] = float(s)
            scored.append(c)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 分数阈值过滤
        if score_threshold > 0:
            scored = [c for c in scored if c["rerank_score"] >= score_threshold]

        return scored[:top_k]

    def _build_kg_aware_passage(self, chunk: Dict) -> str:
        """构建 KG-aware passage：把 chunk 关联的实体注入 passage 文本。

        格式：原文本 + "[实体] A, B, C"
        让 CrossEncoder 能利用图谱结构信息做精排。

        无实体或 KG-aware 关闭时返回原始文本。
        """
        text = chunk.get("text", "")
        if not KG_AWARE_ENABLED:
            return text

        # 从 chunk metadata 提取关联实体
        entities = []
        meta = chunk.get("meta") or chunk.get("metadata") or {}
        if isinstance(meta, dict):
            # 优先用 entity 字段
            ent = meta.get("entity") or meta.get("entities")
            if isinstance(ent, str) and ent:
                entities.append(ent)
            elif isinstance(ent, list):
                entities.extend([str(e) for e in ent if e])

        if not entities:
            return text

        # 限制数量
        entities = entities[:KG_AWARE_MAX_ENTITIES]
        # 注入到 passage 末尾
        entity_str = "、".join(entities)
        return f"{text}\n[关联实体] {entity_str}"


def get_reranker() -> Reranker:
    """获取 Reranker 单例。"""
    return Reranker()


def _build_kg_aware_passage_for_tuple(text: str, meta: dict) -> str:
    """为 rag_system 的 tuple 格式 (text, score, meta) 构建 KG-aware passage。

    与 Reranker._build_kg_aware_passage 逻辑一致，但接收 meta dict 而非 chunk dict。
    """
    if not KG_AWARE_ENABLED:
        return text

    entities = []
    if isinstance(meta, dict):
        ent = meta.get("entity") or meta.get("entities")
        if isinstance(ent, str) and ent:
            entities.append(ent)
        elif isinstance(ent, list):
            entities.extend([str(e) for e in ent if e])

    if not entities:
        return text

    entities = entities[:KG_AWARE_MAX_ENTITIES]
    entity_str = "、".join(entities)
    return f"{text}\n[关联实体] {entity_str}"
