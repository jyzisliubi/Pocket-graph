"""增量索引模块 —— 文档级增量更新，对标 LightRAG 的核心护城河

设计目标：上传新文档后，只对新/受影响实体重新编码，不重建全局索引。
这使得"多格式文件上传 → KG"这一核心卖点在生产环境真正可用（旧实现
每上传一个文件就全量重建三个 FAISS 索引，不可用）。

三层索引的增量策略：
  1. 主向量索引 (FAISSIndex)    : 新实体→add_chunks；受影响实体→remove_by_entity + add_chunks
  2. 实体嵌入索引 (entity_faiss) : 纯 append（实体名不变，只需加新实体）
  3. 关系嵌入索引 (relation_faiss): 纯 append（关系名不变，只需加新关系）

去重：通过 manifest（已索引三元组 hash 集合）在三元组级别去重；
同一 (head, relation, tail) 不会重复进入索引。

向后兼容：旧索引（无 embeddings.npy / 无 manifest）首次增量时会自动迁移：
  - FAISSIndex.load 自动 reconstruct 出 embeddings 缓存
  - manifest 缺失时从 data_path 全量重建

用法::

    from PocketGraphRAG.incremental_index import add_triples_incremental
    from sentence_transformers import SentenceTransformer
    from PocketGraphRAG.config import INDEX_DIR, DATA_PATH

    model = SentenceTransformer(EMBEDDING_MODEL)
    new_triples = [("新实体", "关系", "新尾实体"), ...]
    stats = add_triples_incremental(
        new_triples, model, index_dir=INDEX_DIR, data_path=DATA_PATH,
    )
    print(stats)
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .logging_config import get_logger

logger = get_logger(__name__)

MANIFEST_FILENAME = "triples_manifest.json"
DOCS_MANIFEST_FILENAME = "docs_manifest.json"


# ==========================
# Manifest 管理
# ==========================


def _triple_key(head: str, relation: str, tail: str) -> str:
    """三元组稳定键（与磁盘格式一致，便于去重）"""
    return f"{head.strip()}|{relation.strip()}|{tail.strip()}"


def load_doc_map(index_dir: str) -> dict:
    """加载文档→三元组反向映射 {doc_id: [triple_key, ...]}。

    用于按文档删除：remove_document_incremental 根据 doc_id 找到要删的三元组。
    旧索引无此文件时返回空 dict（向后兼容，仅不支持按文档删除）。
    """
    path = os.path.join(index_dir, DOCS_MANIFEST_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_doc_map(index_dir: str, doc_map: dict):
    """持久化文档→三元组反向映射"""
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, DOCS_MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc_map, f, ensure_ascii=False, indent=2)


def load_manifest(index_dir: str) -> set:
    """加载已索引三元组的 hash 集合。

    若 manifest 文件不存在但 data_path 里有三元组，从 data_path 重建 manifest
    （首次增量迁移场景）。
    """
    manifest_path = os.path.join(index_dir, MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("triples", []))
    return set()


def save_manifest(index_dir: str, triples_set: set):
    """持久化 manifest（三元组 hash 集合）"""
    os.makedirs(index_dir, exist_ok=True)
    manifest_path = os.path.join(index_dir, MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"triples": sorted(triples_set)}, f, ensure_ascii=False, indent=2)


def build_manifest_from_data(data_path: str) -> set:
    """从三元组文件重建 manifest 集合（旧索引迁移用）"""
    manifest = set()
    if not os.path.exists(data_path):
        return manifest
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            h, r, t = [p.strip() for p in parts]
            if h and r and t:
                manifest.add(_triple_key(h, r, t))
    return manifest


# ==========================
# 实体/关系嵌入索引的增量 append
# ==========================


def _append_embedding_index(
    index_path: str,
    names_path: str,
    new_names: List[str],
    model: SentenceTransformer,
) -> int:
    """向已有的实体/关系嵌入索引追加新条目（纯 append）。

    若索引文件不存在，则新建。返回实际新增条目数（去重后）。

    Args:
        index_path: FAISS 索引文件路径（entity_faiss.index / relation_faiss.index）
        names_path: 名称列表 JSON 路径
        new_names: 待追加的名称列表
        model: 编码模型

    Returns:
        实际新增条目数
    """
    if not new_names:
        return 0

    # 加载已有名称
    existing_names: List[str] = []
    existing_set = set()
    if os.path.exists(names_path):
        with open(names_path, encoding="utf-8") as f:
            existing_names = json.load(f)
            existing_set = set(existing_names)

    # 去重：只编码真正新增的
    fresh = []
    seen = set(existing_set)
    for name in new_names:
        if name and name not in seen:
            seen.add(name)
            fresh.append(name)

    if not fresh:
        return 0

    embs = model.encode(
        fresh,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    embs = np.array(embs, dtype="float32")
    dim = embs.shape[1]

    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        # 维度一致性校验
        if index.d != dim:
            raise ValueError(
                f"现有索引维度 {index.d} 与新编码维度 {dim} 不一致，无法增量追加"
            )
        index.add(embs)
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

    faiss.write_index(index, index_path)
    updated_names = existing_names + fresh
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(updated_names, f, ensure_ascii=False, indent=2)

    logger.info(
        "嵌入索引 %s 追加 %s 条，共 %s 条",
        os.path.basename(index_path),
        len(fresh),
        len(updated_names),
    )
    return len(fresh)


# ==========================
# 主增量 API
# ==========================


def add_triples_incremental(
    new_triples: List[Tuple[str, str, str]],
    model: SentenceTransformer,
    index_dir: str,
    data_path: str,
    reverse_link_relations=None,
    relation_templates: dict = None,
    schema=None,
    doc_id: str = None,
) -> dict:
    """增量添加三元组到现有索引（不重建全局）。

    完整流程：
      1. 加载 manifest，对 new_triples 做三元组级去重
      2. 全部重复 → 直接返回，零计算
      3. 加载现有 FAISSIndex（自动迁移旧格式）
      4. 加载现有 KGProcessor（从 data_path 读全量三元组）
      5. KGProcessor.add_triples 合并新三元组到内存
      6. 分类处理：
         - 新实体 → 用 process_entities 生成新 chunk → FAISSIndex.add_chunks
         - 受影响实体（已存在但获得新三元组）→ remove_by_entity + add_chunks 重建
         - 新关系 → _append_embedding_index 追加关系索引
         - 所有新实体（head + tail 中不在现有集合的）→ 追加实体索引
      7. 持久化：append 三元组到 data_path、保存 FAISSIndex、保存 manifest
      8. 若提供 doc_id，记录 doc_id→triple_keys 反向映射（支持后续按文档删除）

    Args:
        new_triples: 待添加的三元组 [(head, relation, tail), ...]
        model: SentenceTransformer 编码模型
        index_dir: 索引目录
        data_path: 主三元组文件路径（会 append）
        reverse_link_relations: 反向链接关系集合，None 用配置/自动推断
        relation_templates: 关系模板
        schema: RelationSchema 实例
        doc_id: 文档标识符。提供则记录到 doc_map，支持 remove_document_incremental 删除

    Returns:
        统计字典：
          {
            "new_triples": 实际新增三元组数,
            "skipped_duplicates": 重复跳过数,
            "new_entities": 新增实体数,
            "affected_entities": 受影响重建实体数,
            "new_relations": 新增关系数,
            "total_chunks": 操作后总 chunk 数,
          }
    """
    from .build_index import FAISSIndex
    from .data_processor import KGProcessor

    stats = {
        "new_triples": 0,
        "skipped_duplicates": 0,
        "new_entities": 0,
        "affected_entities": 0,
        "new_relations": 0,
        "total_chunks": 0,
    }

    # ---------- 1. manifest 去重 ----------
    manifest = load_manifest(index_dir)
    if not manifest and os.path.exists(data_path):
        # 旧索引迁移：从 data_path 重建 manifest
        manifest = build_manifest_from_data(data_path)
        logger.info(
            "manifest 缺失，从 %s 重建 %s 条已索引记录", data_path, len(manifest)
        )

    unique_new: List[Tuple[str, str, str]] = []
    for h, r, t in new_triples:
        h = (h or "").strip()
        r = (r or "").strip()
        t = (t or "").strip()
        if not h or not r or not t:
            continue
        key = _triple_key(h, r, t)
        if key in manifest:
            stats["skipped_duplicates"] += 1
            continue
        manifest.add(key)
        unique_new.append((h, r, t))

    if not unique_new:
        logger.info("增量添加：全部 %s 条三元组已存在，跳过", len(new_triples))
        return stats

    stats["new_triples"] = len(unique_new)

    # ---------- 2. 加载现有索引 ----------
    faiss_index_path = os.path.join(index_dir, "faiss.index")
    if os.path.exists(faiss_index_path):
        faiss_index = FAISSIndex.load(index_dir, model)
    else:
        # 首次增量：空目录，新建空 FAISSIndex（dimension 用 EMBEDDING_DIM 默认值，
        # 第一次 add_chunks 时会校验维度一致性）
        faiss_index = FAISSIndex()
        faiss_index.model = model
        import numpy as _np
        # EMBEDDING_DIM 为 None 时从模型推断维度，并创建对应维度的空索引
        if faiss_index.dimension is None:
            faiss_index.dimension = int(model.get_sentence_embedding_dimension())
            faiss_index.index = faiss.IndexFlatIP(faiss_index.dimension)
        faiss_index._embeddings = _np.zeros(
            (0, faiss_index.dimension), dtype="float32"
        )
        logger.info("索引目录为空，新建空 FAISSIndex（首次增量）")
    stats["total_chunks"] = len(faiss_index.texts)

    # ---------- 3. 加载现有 KGProcessor（全量三元组）----------
    # 首次增量若 data_path 不存在，KGProcessor.load_triples 会失败，
    # 此时视为空索引，直接用 add_triples 建立内存状态。
    processor = KGProcessor(
        data_path,
        reverse_link_relations=reverse_link_relations,
        relation_templates=relation_templates,
        schema=schema,
    )
    existing_entities_before: set = set()
    existing_relations_before: set = set()
    if os.path.exists(data_path):
        try:
            processor.load_triples()
            existing_entities_before = set(processor.entity_relations.keys()) | set(
                processor.reverse_relations.keys()
            )
            existing_relations_before = {r for _, r, _ in processor.triples}
        except Exception as e:
            logger.warning("加载现有三元组失败，按空索引处理: %s", e)

    # ---------- 4. 合并新三元组到内存 ----------
    added = processor.add_triples(unique_new)

    # ---------- 5. 分类处理实体 ----------
    # 全量合并后的实体集合
    all_entities_after = set(processor.entity_relations.keys()) | set(
        processor.reverse_relations.keys()
    )
    new_entities = all_entities_after - existing_entities_before
    # 受影响实体：已存在，但因为新增三元组其 chunk 文本变了，需要重建
    affected_entities: set = set()
    for h, _r, t in added:
        if h in existing_entities_before:
            affected_entities.add(h)
        if t in existing_entities_before:
            affected_entities.add(t)
    # 受影响实体中已被算作"新实体"的不重复算
    affected_entities = affected_entities - new_entities

    # 5a. 先 remove 受影响实体的旧 chunk
    for entity in affected_entities:
        faiss_index.remove_by_entity(entity)

    # 5b. 为新实体 + 受影响实体生成 chunk 并 add
    entities_to_add = list(new_entities) + list(affected_entities)
    if entities_to_add:
        chunks = processor.process_entities(entities_to_add)
        if chunks:
            faiss_index.add_chunks(chunks, model)

    stats["new_entities"] = len(new_entities)
    stats["affected_entities"] = len(affected_entities)
    stats["total_chunks"] = len(faiss_index.texts)

    # ---------- 6. 关系索引增量 append ----------
    all_relations_after = {r for _, r, _ in processor.triples}
    new_relations = list(all_relations_after - existing_relations_before)
    if new_relations:
        relation_index_path = os.path.join(index_dir, "relation_faiss.index")
        relation_names_path = os.path.join(index_dir, "relation_names.json")
        stats["new_relations"] = _append_embedding_index(
            relation_index_path, relation_names_path, new_relations, model
        )

    # ---------- 7. 实体索引增量 append ----------
    if new_entities:
        entity_index_path = os.path.join(index_dir, "entity_faiss.index")
        entity_names_path = os.path.join(index_dir, "entity_names.json")
        _append_embedding_index(
            entity_index_path, entity_names_path, list(new_entities), model
        )

    # ---------- 8. 持久化 ----------
    # 8a. append 三元组到 data_path（不存在则创建）
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    # 确保文件末尾有换行，避免 append 时拼到原最后一行末尾
    if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
        with open(data_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last_byte = f.read(1)
        needs_newline = last_byte != b"\n"
    else:
        needs_newline = False
    with open(data_path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        for h, _r, t in added:
            # 清理分隔符，避免破坏格式
            h_clean = h.replace("|", "").replace("\n", " ")
            r_clean = _r.replace("|", "").replace("\n", " ")
            t_clean = t.replace("|", "").replace("\n", " ")
            f.write(f"{h_clean} | {r_clean} | {t_clean}\n")

    # 8b. 保存主索引 + manifest
    faiss_index.save(index_dir)
    save_manifest(index_dir, manifest)

    # 8c. 若提供 doc_id，记录 doc_id → triple_keys 反向映射（支持按文档删除）
    if doc_id is not None:
        doc_map = load_doc_map(index_dir)
        added_keys = [_triple_key(h, r, t) for h, r, t in added]
        existing = doc_map.get(doc_id, [])
        # 去重保序合并
        seen = set(existing)
        for k in added_keys:
            if k not in seen:
                existing.append(k)
                seen.add(k)
        doc_map[doc_id] = existing
        save_doc_map(index_dir, doc_map)
        logger.info("文档 %s 记录 %s 条三元组映射", doc_id, len(added_keys))

    logger.info(
        "增量索引完成: 新增三元组 %s, 跳过重复 %s, 新实体 %s, 受影响实体 %s, 新关系 %s, 总 chunk %s",
        stats["new_triples"],
        stats["skipped_duplicates"],
        stats["new_entities"],
        stats["affected_entities"],
        stats["new_relations"],
        stats["total_chunks"],
    )
    return stats


def _rebuild_entity_index_excluding(
    index_dir: str, exclude_entities: set, model: SentenceTransformer
) -> int:
    """重建实体嵌入索引，排除指定孤儿实体。

    FAISS IndexFlatIP 不支持 remove，删除孤儿实体需重建。
    关系嵌入索引通常不需重建（关系名不会因删文档而消失，孤儿关系影响小）。

    Returns:
        重建后剩余实体数
    """
    entity_index_path = os.path.join(index_dir, "entity_faiss.index")
    entity_names_path = os.path.join(index_dir, "entity_names.json")
    if not os.path.exists(entity_names_path):
        return 0

    with open(entity_names_path, encoding="utf-8") as f:
        all_names = json.load(f)
    kept_names = [n for n in all_names if n not in exclude_entities]
    removed = len(all_names) - len(kept_names)

    if not kept_names:
        # 全删空了
        if os.path.exists(entity_index_path):
            os.remove(entity_index_path)
        with open(entity_names_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)
        logger.info("实体嵌入索引清空（所有实体均成孤儿）")
        return 0

    embs = model.encode(
        kept_names,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    embs = np.array(embs, dtype="float32")
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, entity_index_path)
    with open(entity_names_path, "w", encoding="utf-8") as f:
        json.dump(kept_names, f, ensure_ascii=False, indent=2)

    logger.info(
        "实体嵌入索引重建：排除 %s 个孤儿实体，剩余 %s 个",
        removed,
        len(kept_names),
    )
    return len(kept_names)


def remove_document_incremental(
    doc_id: str,
    model: SentenceTransformer,
    index_dir: str,
    data_path: str,
    reverse_link_relations=None,
    relation_templates: dict = None,
    schema=None,
) -> dict:
    """按文档 ID 增量删除其所有三元组（不重建全局索引）。

    级联清理流程：
      1. 从 doc_map 查 doc_id → triple_keys
      2. 加载 FAISSIndex + KGProcessor（全量三元组）
      3. 将三元组分为 to_remove / remaining
      4. 用 remaining 重建 KGProcessor 内存状态
      5. 受影响实体分类：
         - 孤儿实体（remaining 中无）→ remove_by_entity 删 chunk + 从实体嵌入索引删除
         - 存活实体（remaining 中有）→ remove_by_entity + process_entities 重建 chunk
      6. 重写 data_path（remaining）
      7. 更新 manifest（triples set）+ doc_map（删 doc_id 条目）
      8. 保存主索引 + 实体嵌入索引（若有孤儿）

    Args:
        doc_id: 要删除的文档标识符
        model: SentenceTransformer 编码模型
        index_dir: 索引目录
        data_path: 主三元组文件路径（会被重写）
        reverse_link_relations, relation_templates, schema: 同 add_triples_incremental

    Returns:
        {
            "removed_triples": 删除的三元组数,
            "affected_entities": 存活但需重建 chunk 的实体数,
            "orphan_entities_removed": 变成孤儿被清除的实体数,
            "total_chunks": 操作后总 chunk 数,
        }
    """
    from .build_index import FAISSIndex
    from .data_processor import KGProcessor

    stats = {
        "removed_triples": 0,
        "affected_entities": 0,
        "orphan_entities_removed": 0,
        "total_chunks": 0,
    }

    # 1. 查 doc_map
    doc_map = load_doc_map(index_dir)
    triple_keys_to_remove = set(doc_map.pop(doc_id, []))
    if not triple_keys_to_remove:
        logger.warning("文档 %s 不在 doc_map 中，无操作（可能未用 doc_id 增量添加）", doc_id)
        return stats

    # 2. 加载现有索引
    faiss_index = FAISSIndex.load(index_dir, model)
    stats["total_chunks"] = len(faiss_index.texts)

    # 3. 加载全量三元组，分 to_remove / remaining
    processor = KGProcessor(
        data_path,
        reverse_link_relations=reverse_link_relations,
        relation_templates=relation_templates,
        schema=schema,
    )
    if os.path.exists(data_path):
        try:
            processor.load_triples()
        except Exception as e:
            logger.warning("加载现有三元组失败: %s", e)

    triples_to_remove = []
    remaining_triples = []
    for h, r, t in processor.triples:
        key = _triple_key(h, r, t)
        if key in triple_keys_to_remove:
            triples_to_remove.append((h, r, t))
        else:
            remaining_triples.append((h, r, t))

    stats["removed_triples"] = len(triples_to_remove)
    if not triples_to_remove:
        logger.warning("doc_map 记录了 %s 条但 data_path 未匹配到，可能已手动清理", doc_id)
        # 仍要更新 doc_map（清除悬空记录）
        save_doc_map(index_dir, doc_map)
        return stats

    # 4. 用 remaining 重建 KGProcessor 内存状态
    processor = KGProcessor(
        data_path,
        reverse_link_relations=reverse_link_relations,
        relation_templates=relation_templates,
        schema=schema,
    )
    processor.add_triples(remaining_triples)

    # 5. 受影响实体分类
    affected_entities = set()
    for h, _r, t in triples_to_remove:
        affected_entities.add(h)
        affected_entities.add(t)
    remaining_entities = set(processor.entity_relations.keys()) | set(
        processor.reverse_relations.keys()
    )
    orphan_entities = affected_entities - remaining_entities
    affected_alive = affected_entities & remaining_entities

    # 6. 删孤儿实体的 chunk（实体嵌入索引稍后批量重建）
    for entity in orphan_entities:
        faiss_index.remove_by_entity(entity)

    # 7. 存活但受影响实体：remove 旧 chunk + 用 remaining 三元组重建
    for entity in affected_alive:
        faiss_index.remove_by_entity(entity)
    if affected_alive:
        chunks = processor.process_entities(list(affected_alive))
        if chunks:
            faiss_index.add_chunks(chunks, model)

    stats["affected_entities"] = len(affected_alive)
    stats["orphan_entities_removed"] = len(orphan_entities)
    stats["total_chunks"] = len(faiss_index.texts)

    # 8. 重写 data_path（remaining）
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        for h, r, t in remaining_triples:
            h_clean = h.replace("|", "").replace("\n", " ")
            r_clean = r.replace("|", "").replace("\n", " ")
            t_clean = t.replace("|", "").replace("\n", " ")
            f.write(f"{h_clean} | {r_clean} | {t_clean}\n")

    # 9. 更新 manifest + doc_map
    manifest = load_manifest(index_dir)
    manifest -= triple_keys_to_remove
    save_manifest(index_dir, manifest)
    save_doc_map(index_dir, doc_map)

    # 10. 保存主索引
    faiss_index.save(index_dir)

    # 11. 实体嵌入索引孤儿清理（重建排除孤儿）
    if orphan_entities:
        _rebuild_entity_index_excluding(index_dir, orphan_entities, model)

    logger.info(
        "文档 %s 删除完成: 删三元组 %s, 存活重建实体 %s, 孤儿清除实体 %s, 总 chunk %s",
        doc_id,
        stats["removed_triples"],
        stats["affected_entities"],
        stats["orphan_entities_removed"],
        stats["total_chunks"],
    )
    return stats


def reset_index(
    model: SentenceTransformer,
    index_dir: str,
    data_path: str,
    reverse_link_relations=None,
    relation_templates: dict = None,
    schema=None,
) -> dict:
    """清空并全量重建索引（兜底/重置场景）。

    删除 index_dir 下所有索引文件 + manifest，然后用 data_path 当前内容
    全量重建。用于：用户主动"重置数据集"、或索引损坏需要从磁盘三元组重建。

    Args:
        model: 编码模型
        index_dir: 索引目录
        data_path: 三元组文件（不会被清空，作为重建源）
        reverse_link_relations, relation_templates, schema: 同 build_index_with_data

    Returns:
        {"total_triples": n, "total_chunks": n, "total_entities": n, "total_relations": n}
    """
    from .build_index import build_index_with_data

    # 清理旧索引文件
    if os.path.isdir(index_dir):
        for fname in os.listdir(index_dir):
            if fname.endswith((".index", ".json", ".npy")):
                try:
                    os.remove(os.path.join(index_dir, fname))
                except OSError:
                    pass

    # 全量重建
    index = build_index_with_data(
        data_path,
        index_dir=index_dir,
        run_tests=False,
    )

    # 重建 manifest
    manifest = build_manifest_from_data(data_path)
    save_manifest(index_dir, manifest)

    # 统计
    from .data_processor import KGProcessor

    processor = KGProcessor(
        data_path,
        reverse_link_relations=reverse_link_relations,
        relation_templates=relation_templates,
        schema=schema,
    )
    processor.load_triples()
    all_entities = set(processor.entity_relations.keys()) | set(
        processor.reverse_relations.keys()
    )
    all_relations = {r for _, r, _ in processor.triples}

    return {
        "total_triples": len(processor.triples),
        "total_chunks": len(index.texts),
        "total_entities": len(all_entities),
        "total_relations": len(all_relations),
    }
