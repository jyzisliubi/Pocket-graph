"""
FAISS 向量索引构建器

功能：
1. 使用 BGE 中文 embedding 模型将文本块向量化
2. 构建 FAISS 索引，支持高效的相似度检索
3. 构建实体嵌入索引（用于 KG Local Search）
4. 构建关系嵌入索引（用于 KG Global Search）
5. 保存索引和元数据到磁盘，支持离线加载

使用方式：
    python -m PocketGraphRAG.build_index
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    DATA_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INDEX_DIR,
    RELATION_TEMPLATES,
    REVERSE_LINK_RELATIONS,
)
from .data_processor import KGProcessor
from .logging_config import get_logger

logger = get_logger(__name__)


class FAISSIndex:
    """FAISS 向量索引管理器（支持增量 add/remove）

    内部维护一个 embeddings 缓存 (self._embeddings)，使得 remove_by_entity
    可以用 numpy 切片 + IndexFlatIP 重建实现，O(n) 内存操作、零模型调用。
    增量 add (add_chunks) 只编码新 chunk，append 到缓存与索引。

    注意：本类是历史遗留的具体实现，新代码推荐用
    `PocketGraphRAG.core.storages.FAISSVectorStore`（实现了 VectorStore 抽象接口）。
    本类继续维护以保证向后兼容。
    """

    def __init__(self, dimension: int = EMBEDDING_DIM):
        # dimension 仅作为兜底默认值；实际 build 时会从 embeddings 形状覆盖
        # EMBEDDING_DIM 为 None（动态化默认）时延迟创建索引，
        # 待 build() 从模型推断维度或调用方显式设置后再创建。
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) if dimension is not None else None
        self.texts: list = []
        self.metadatas: list = []
        self.model = None
        # embeddings 缓存：与 texts/metadatas 行对齐，shape (n, dim)
        # 增量 add/remove 的关键，避免 remove 时重新调用模型编码
        self._embeddings: np.ndarray | None = None
        # entity -> chunk_ids 倒排表：O(1) 查表替代 O(N) metadatas 扫描
        # 由 _rebuild_entity_index 从 metadatas 派生，build/add/remove/load 时同步维护
        self._entity_to_chunks: dict[str, list[int]] = {}

    def build(self, chunks: list, model: SentenceTransformer):
        """从文本块构建 FAISS 索引。

        H6 修复：不信任传入的 dimension，而是从实际 embeddings 形状读取维度，
        这样切换 EMBEDDING_MODEL（如 bge-small→bge-large）时不会因维度不匹配崩溃。
        """
        self.model = model
        texts = [chunk["text"] for chunk in chunks]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )
        embeddings = np.array(embeddings, dtype="float32")

        # 从实际 embeddings 形状读取维度，覆盖默认值
        actual_dim = int(embeddings.shape[1])
        if actual_dim != self.dimension:
            logger.info(
                "实际 embedding 维度 %s 与默认 %s 不一致，按实际维度构建索引",
                actual_dim,
                self.dimension,
            )
            self.dimension = actual_dim

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        self.texts = texts
        self.metadatas = [chunk["metadata"] for chunk in chunks]
        self._embeddings = embeddings
        self._rebuild_entity_index()

        logger.info("索引构建完成: %s 个文档, %s 维向量", len(texts), self.dimension)

    def add_chunks(self, chunks: list, model: SentenceTransformer = None) -> int:
        """增量追加文本块到现有索引（不重建全局）。

        只对新 chunk 调用模型编码，append 到 embeddings 缓存与 FAISS 索引。
        用于增量索引场景：上传新文档后只编码新实体的 chunk。

        Args:
            chunks: 新文本块列表，每个含 text/metadata
            model: 编码模型，None 用已加载的 model

        Returns:
            实际新增的 chunk 数
        """
        if not chunks:
            return 0
        model = model or self.model
        if model is None:
            raise ValueError("add_chunks 需要模型，请先 build() 或传入 model")
        if self.index is None:
            raise ValueError("索引未初始化，请先 build() 或 load()")

        texts = [c["text"] for c in chunks]
        embs = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        embs = np.array(embs, dtype="float32")

        # 维度校验：新 chunk 必须与现有索引维度一致
        if embs.shape[1] != self.dimension:
            raise ValueError(
                f"新 chunk embedding 维度 {embs.shape[1]} 与索引维度 {self.dimension} 不一致"
            )

        self.index.add(embs)
        # 增量追加前 texts 的长度，即新 chunk 的起始 id
        base_id = len(self.texts)
        self.texts.extend(texts)
        self.metadatas.extend([c["metadata"] for c in chunks])
        # append 到缓存
        if self._embeddings is None:
            self._embeddings = embs
        else:
            self._embeddings = np.vstack([self._embeddings, embs])
        # 追加新 chunk 到倒排表（不重建，O(k) k=新增 chunk 数）
        for offset, c in enumerate(chunks):
            ent = c.get("metadata", {}).get("entity")
            if ent:
                self._entity_to_chunks.setdefault(ent, []).append(base_id + offset)

        logger.info("增量追加 %s 个 chunk，当前共 %s 个", len(texts), len(self.texts))
        return len(texts)

    def remove_by_entity(self, entity: str) -> int:
        """删除某实体对应的所有 chunk（增量重建场景：实体获得新三元组后需替换旧 chunk）

        利用 embeddings 缓存做 numpy 切片 + IndexFlatIP 重建，
        不重新调用模型编码。O(n) 内存操作，n 为当前 chunk 总数。

        Args:
            entity: 要删除的实体名

        Returns:
            实际删除的 chunk 数
        """
        if not self.texts:
            return 0

        # 找出该实体对应的位置
        positions_to_remove = [
            i for i, m in enumerate(self.metadatas) if m.get("entity") == entity
        ]
        if not positions_to_remove:
            return 0

        remove_set = set(positions_to_remove)
        keep_mask = [i not in remove_set for i in range(len(self.texts))]

        # 用缓存重建索引
        if self._embeddings is not None:
            kept_embs = self._embeddings[keep_mask]
            self.dimension = (
                int(kept_embs.shape[1]) if kept_embs.size else self.dimension
            )
            self.index = faiss.IndexFlatIP(self.dimension)
            if kept_embs.shape[0] > 0:
                self.index.add(kept_embs)
            self._embeddings = kept_embs
        else:
            # 缓存缺失（不应发生，但兜底）：从现有索引 reconstruct
            self._rebuild_from_index_reconstruct(keep_mask)

        self.texts = [t for t, k in zip(self.texts, keep_mask) if k]
        self.metadatas = [m for m, k in zip(self.metadatas, keep_mask) if k]
        # 删除后 chunk id 全部重排，重建倒排表保证 id 一致
        self._rebuild_entity_index()

        logger.info(
            "删除实体 '%s' 的 %s 个 chunk，剩余 %s 个",
            entity,
            len(positions_to_remove),
            len(self.texts),
        )
        return len(positions_to_remove)

    def _rebuild_from_index_reconstruct(self, keep_mask: list):
        """兜底：embeddings 缓存缺失时，从 FAISS 索引 reconstruct 出向量重建。

        IndexFlatIP 支持 reconstruct_n，可逐个还原向量。
        """
        n = self.index.ntotal
        if n == 0:
            self.index = faiss.IndexFlatIP(self.dimension)
            self._embeddings = np.zeros((0, self.dimension), dtype="float32")
            return
        all_embs = self.index.reconstruct_n(0, n)
        kept_embs = all_embs[keep_mask]
        self.dimension = int(kept_embs.shape[1]) if kept_embs.size else self.dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        if kept_embs.shape[0] > 0:
            self.index.add(kept_embs)
        self._embeddings = kept_embs

    def _rebuild_entity_index(self) -> None:
        """从 self.metadatas 重建 entity -> chunk_ids 倒排表。

        O(N) 全量扫描，仅在 build / load / remove_by_entity 后调用一次。
        之后的检索走 get_chunks_by_entity() 为 O(1) 查表。
        """
        self._entity_to_chunks = {}
        for i, meta in enumerate(self.metadatas):
            ent = (meta or {}).get("entity")
            if ent:
                self._entity_to_chunks.setdefault(ent, []).append(i)

    def get_chunks_by_entity(self, entity: str) -> list[int]:
        """返回某实体对应的所有 chunk_id（在 self.texts / self.metadatas 中的索引）。

        O(1) 查表，替代旧的 O(N) metadatas 线性扫描。
        未命中返回空列表。
        """
        return self._entity_to_chunks.get(entity, [])

    def get_chunks_by_entities(self, entities: list[str]) -> list[tuple[int, str, dict]]:
        """批量查表：返回 (chunk_id, text, metadata) 列表，去重。

        Args:
            entities: 实体名列表

        Returns:
            [(chunk_id, text, metadata), ...]
            同一 chunk 关联多实体时只返回一次。
            调用方自行用 metadata['entity'] 判断是否种子实体（用于排序加成）。
        """
        seen = set()
        out = []
        for ent in entities:
            for cid in self._entity_to_chunks.get(ent, []):
                if cid in seen:
                    continue
                seen.add(cid)
                out.append((cid, self.texts[cid], self.metadatas[cid]))
        return out

    def search(self, query: str, top_k: int = 5) -> list:
        """检索与查询最相似的 top_k 个文本块。

        Returns:
            [(文本, 相似度分数, 元数据), ...] 按相似度降序排列
        """
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_vec = np.array(query_vec, dtype="float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.texts[idx], float(score), self.metadatas[idx]))
        return results

    def save(self, index_dir: str):
        """保存索引、文本、元数据、embeddings 缓存到指定目录"""
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "texts.json"), "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)
        with open(
            os.path.join(index_dir, "metadatas.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)
        # 保存 embeddings 缓存，使增量操作在 reload 后仍可用
        if self._embeddings is not None:
            np.save(os.path.join(index_dir, "embeddings.npy"), self._embeddings)
        # 写入 embedding 模型指纹，load 时校验防维度错配
        try:
            from .config import EMBEDDING_MODEL

            fingerprint = {
                "model": EMBEDDING_MODEL,
                "dimension": int(self.dimension) if self.dimension else None,
            }
            with open(
                os.path.join(index_dir, "embedding_model.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(fingerprint, f, ensure_ascii=False, indent=2)
        except Exception as e:  # 指纹写入失败不阻塞保存
            logger.warning("写入 embedding 模型指纹失败（不影响索引）: %s", e)
        logger.info("索引已保存到: %s", index_dir)

    @classmethod
    def load(cls, index_dir: str, model: SentenceTransformer) -> "FAISSIndex":
        """从磁盘加载已构建好的索引。

        向后兼容：若 embeddings.npy 不存在（旧索引），用 reconstruct_n
        从 FAISS 索引还原缓存，使后续增量操作可用。

        模型指纹校验：若索引目录含 embedding_model.json，校验当前 EMBEDDING_MODEL
        与索引构建时使用的模型一致；不一致则抛错，提示用户重建索引。
        维度不一致（如 bge-small 512 维 vs bge-m3 1024 维）必然导致检索质量崩塌，
        不能静默放行。
        """
        # 友好校验：索引文件缺失时给出可操作提示，而非原始 faiss C++ 报错（BUG #9）
        index_path = os.path.join(index_dir, "faiss.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"索引文件不存在: {index_path}\n"
                f"请先构建索引: pocketgraphrag build\n"
                f"或: python -m PocketGraphRAG.build_index"
            )
        # 模型指纹校验（防维度错配）
        fingerprint_path = os.path.join(index_dir, "embedding_model.json")
        if os.path.exists(fingerprint_path):
            try:
                with open(fingerprint_path, encoding="utf-8") as f:
                    fingerprint = json.load(f)
                from .config import EMBEDDING_MODEL

                indexed_model = fingerprint.get("model", "")
                indexed_dim = fingerprint.get("dimension")
                # 维度优先校验（最可靠的硬性指标）
                if indexed_dim is not None:
                    try:
                        cur_dim = int(model.get_sentence_embedding_dimension())
                    except Exception:
                        cur_dim = None
                    if cur_dim is not None and cur_dim != int(indexed_dim):
                        raise RuntimeError(
                            f"索引维度与当前 embedding 模型不匹配："
                            f"索引构建时为 {indexed_dim} 维，"
                            f"当前模型 {EMBEDDING_MODEL} 为 {cur_dim} 维。\n"
                            f"请删除 {index_dir} 后重建索引：\n"
                            f"  pocketgraphrag build\n"
                            f"或切换回原模型：\n"
                            f"  POCKET_EMBEDDING_MODEL={indexed_model}"
                        )
                # 模型名不一致但维度一致时仅警告（同维度不同模型仍可用）
                if indexed_model and indexed_model != EMBEDDING_MODEL:
                    logger.warning(
                        "Embedding 模型切换：索引构建用 %s，当前为 %s。"
                        "维度一致可继续，但语义空间不同，建议重建索引以获得最佳效果。",
                        indexed_model,
                        EMBEDDING_MODEL,
                    )
            except RuntimeError:
                raise
            except Exception as e:  # 指纹读取失败不阻塞加载
                logger.warning("读取 embedding 模型指纹失败（跳过校验）: %s", e)
        instance = cls()
        instance.model = model
        instance.index = faiss.read_index(index_path)
        # H6 修复：从加载的索引同步 dimension，避免与实际维度不一致
        instance.dimension = int(instance.index.d)
        with open(os.path.join(index_dir, "texts.json"), encoding="utf-8") as f:
            instance.texts = json.load(f)
        with open(os.path.join(index_dir, "metadatas.json"), encoding="utf-8") as f:
            instance.metadatas = json.load(f)

        emb_path = os.path.join(index_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            instance._embeddings = np.load(emb_path)
        elif instance.index.ntotal > 0:
            # 旧索引迁移：reconstruct 出缓存
            logger.info("未发现 embeddings.npy，从 FAISS 索引还原缓存（一次性迁移）")
            instance._embeddings = instance.index.reconstruct_n(
                0, instance.index.ntotal
            )
        else:
            instance._embeddings = np.zeros((0, instance.dimension), dtype="float32")

        # 重建 entity -> chunk_ids 倒排表（O(N)，仅 load 时一次）
        instance._rebuild_entity_index()

        logger.info(
            "索引加载完成: %s 个文档, %s 维", len(instance.texts), instance.dimension
        )
        return instance


def build_entity_embedding_index(
    data_path: str, model: SentenceTransformer, index_dir: str
):
    """构建实体嵌入索引并保存到索引目录

    从三元组数据中提取所有唯一实体名，用 embedding 模型编码，
    保存为 FAISS 索引和实体名列表 JSON 文件。
    """
    processor = KGProcessor(
        data_path,
        reverse_link_relations=REVERSE_LINK_RELATIONS,
        relation_templates=RELATION_TEMPLATES,
    )
    processor.load_triples()

    all_entities = sorted(
        set(processor.entity_relations.keys()) | set(processor.reverse_relations.keys())
    )
    logger.info("唯一实体数: %s", len(all_entities))

    embeddings = model.encode(
        all_entities,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    entity_index = faiss.IndexFlatIP(dim)
    entity_index.add(embeddings)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(entity_index, os.path.join(index_dir, "entity_faiss.index"))
    with open(os.path.join(index_dir, "entity_names.json"), "w", encoding="utf-8") as f:
        json.dump(all_entities, f, ensure_ascii=False, indent=2)

    logger.info("实体嵌入索引已保存: %s 个实体 -> %s", len(all_entities), index_dir)


def build_relation_embedding_index(
    data_path: str, model: SentenceTransformer, index_dir: str
):
    """构建关系嵌入索引并保存到索引目录

    从三元组数据中提取所有唯一关系名，用 embedding 模型编码，
    保存为 FAISS 索引和关系名列表 JSON 文件。
    """
    processor = KGProcessor(
        data_path,
        reverse_link_relations=REVERSE_LINK_RELATIONS,
        relation_templates=RELATION_TEMPLATES,
    )
    processor.load_triples()

    all_relations = set()
    for rels in processor.entity_relations.values():
        for rel, _ in rels:
            all_relations.add(rel)
    all_relations = sorted(all_relations)
    logger.info("唯一关系数: %s", len(all_relations))

    if not all_relations:
        return

    embeddings = model.encode(
        all_relations,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    relation_index = faiss.IndexFlatIP(dim)
    relation_index.add(embeddings)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(relation_index, os.path.join(index_dir, "relation_faiss.index"))
    with open(
        os.path.join(index_dir, "relation_names.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_relations, f, ensure_ascii=False, indent=2)

    logger.info("关系嵌入索引已保存: %s 种关系 -> %s", len(all_relations), index_dir)


def build_index_with_data(
    data_path: str, index_dir: str = INDEX_DIR, run_tests: bool = False
):
    """使用指定数据路径构建完整索引

    Args:
        data_path: 三元组数据文件路径
        index_dir: 索引保存目录
        run_tests: 是否运行快速检索测试
    """
    print("=" * 50)
    print("  PocketGraphRAG - 索引构建")
    print("=" * 50)

    # 1. 加载并处理数据
    print("\n[1/5] 处理知识图谱数据...")
    processor = KGProcessor(
        data_path,
        reverse_link_relations=REVERSE_LINK_RELATIONS,
        relation_templates=RELATION_TEMPLATES,
    )
    processor.load_triples()
    chunks = processor.process()

    stats = processor.get_statistics()
    print(f"  三元组总数: {stats['总三元组数']}")
    print(f"  实体总数:   {stats['唯一头实体数']}")
    print(f"  关系类型:   {stats['关系类型数']}")
    print(f"  文本块数:   {len(chunks)}")

    chunks_path = os.path.join(index_dir, "chunks.json")
    processor.save_chunks(chunks, chunks_path)
    print(f"  文本块已保存: {chunks_path}")

    # 2. 加载 embedding 模型
    print(f"\n[2/5] 加载 Embedding 模型: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  模型加载完成")

    # 3. 构建 FAISS 向量索引
    print("\n[3/5] 构建 FAISS 向量索引...")
    index = FAISSIndex()
    index.build(chunks, model)
    index.save(index_dir)

    # 4. 构建实体嵌入索引
    print("\n[4/5] 构建实体嵌入索引...")
    build_entity_embedding_index(data_path, model, index_dir)

    # 5. 构建关系嵌入索引
    print("\n[5/5] 构建关系嵌入索引...")
    build_relation_embedding_index(data_path, model, index_dir)

    # 6. 写 manifest（保证全量构建后也能正确增量）
    from .incremental_index import build_manifest_from_data, save_manifest

    manifest = build_manifest_from_data(data_path)
    save_manifest(index_dir, manifest)
    print(f"  manifest 已写入: {len(manifest)} 条已索引记录")

    print("\n" + "=" * 50)
    print("  索引构建完成！")
    print(f"  索引目录: {index_dir}")
    print("=" * 50)

    # 快速测试
    if run_tests:
        print("\n快速检索测试：")
        test_queries = [
            "肖申克的救赎的导演是谁？",
            "诺兰导演了哪些电影？",
            "评分最高的科幻电影有哪些？",
        ]
        for q in test_queries:
            results = index.search(q, top_k=2)
            print(f"\n  Q: {q}")
            for text, score, meta in results:
                print(f"    [{score:.4f}] {meta['entity']}")

    return index


def build_index():
    """主流程：使用默认数据路径构建索引"""
    build_index_with_data(DATA_PATH, INDEX_DIR, run_tests=True)


def _parse_cli_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="PocketGraphRAG 索引构建（支持全量构建 / 增量追加 / 重置）"
    )
    sub = parser.add_subparsers(dest="command")

    # 默认（无子命令）：全量构建
    parser.add_argument("--data", default=DATA_PATH, help="三元组文件路径")
    parser.add_argument("--index-dir", default=INDEX_DIR, help="索引目录")

    # 增量追加
    p_add = sub.add_parser("add", help="增量追加三元组文件到现有索引")
    p_add.add_argument(
        "--input", required=True, help="待追加的三元组文件（head|relation|tail 格式）"
    )
    p_add.add_argument(
        "--data", default=DATA_PATH, help="主三元组文件路径（会 append）"
    )
    p_add.add_argument("--index-dir", default=INDEX_DIR, help="索引目录")

    # 重置
    p_reset = sub.add_parser("reset", help="清空并全量重建索引")
    p_reset.add_argument("--data", default=DATA_PATH, help="三元组文件路径（重建源）")
    p_reset.add_argument("--index-dir", default=INDEX_DIR, help="索引目录")

    # 清洗：剔除句子片段/用法描述/LLM 占位符实体，备份原文件后重写 + 重建索引
    p_clean = sub.add_parser(
        "clean",
        help="清洗三元组文件（剔除句子型/用法描述/占位符实体）并重建索引",
    )
    p_clean.add_argument("--data", default=DATA_PATH, help="待清洗的三元组文件")
    p_clean.add_argument("--index-dir", default=INDEX_DIR, help="索引目录")
    p_clean.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份原文件（默认备份为 .bak）",
    )

    return parser.parse_args()


def _load_triples_file(path: str) -> list:
    """从文件加载三元组列表 [(head, relation, tail), ...]"""
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            h, r, t = [p.strip() for p in parts]
            if h and r and t:
                triples.append((h, r, t))
    return triples


def clean_triples_file(path: str, backup: bool = True) -> dict:
    """清洗三元组文件：剔除头/尾为低质量实体的三元组。

    低质量判定见 ``kg_extractor.is_low_quality_entity``：句子片段、
    用法描述（含方法动词+数字）、LLM 占位符、列举性"…等"短语。

    Args:
        path: 三元组文件路径，清洗后原地写回
        backup: 是否备份原文件为 ``<path>.bak``

    Returns:
        统计字典 {original, kept, removed, backup_path}
    """
    from .kg_extractor import is_low_quality_entity

    triples = _load_triples_file(path)
    kept = [
        (h, r, t)
        for h, r, t in triples
        if not is_low_quality_entity(h) and not is_low_quality_entity(t)
    ]
    removed = len(triples) - len(kept)

    backup_path = None
    if backup and removed > 0:
        import shutil

        backup_path = path + ".bak"
        shutil.copy2(path, backup_path)
        logger.info("原文件已备份至: %s", backup_path)

    if removed > 0:
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in kept:
                f.write(f"{h} | {r} | {t}\n")

    return {
        "original": len(triples),
        "kept": len(kept),
        "removed": removed,
        "backup_path": backup_path,
    }


def main():
    """CLI 入口：支持全量构建 / 增量追加 / 重置"""
    args = _parse_cli_args()

    if args.command == "add":
        # 增量追加
        from .incremental_index import add_triples_incremental

        print("=" * 50)
        print("  PocketGraphRAG - 增量追加索引")
        print("=" * 50)
        new_triples = _load_triples_file(args.input)
        print(f"  待追加三元组: {len(new_triples)} 条")
        print(f"\n加载 Embedding 模型: {EMBEDDING_MODEL} ...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        stats = add_triples_incremental(
            new_triples,
            model,
            index_dir=args.index_dir,
            data_path=args.data,
            reverse_link_relations=REVERSE_LINK_RELATIONS,
            relation_templates=RELATION_TEMPLATES,
        )
        print("\n" + "=" * 50)
        print("  增量追加完成！")
        print(f"  新增三元组:   {stats['new_triples']}")
        print(f"  跳过重复:     {stats['skipped_duplicates']}")
        print(f"  新增实体:     {stats['new_entities']}")
        print(f"  受影响重建:   {stats['affected_entities']}")
        print(f"  新增关系:     {stats['new_relations']}")
        print(f"  总 chunk 数:  {stats['total_chunks']}")
        print("=" * 50)

    elif args.command == "reset":
        # 重置
        from .incremental_index import reset_index

        print("=" * 50)
        print("  PocketGraphRAG - 重置索引（全量重建）")
        print("=" * 50)
        print(f"\n加载 Embedding 模型: {EMBEDDING_MODEL} ...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        stats = reset_index(
            model,
            index_dir=args.index_dir,
            data_path=args.data,
            reverse_link_relations=REVERSE_LINK_RELATIONS,
            relation_templates=RELATION_TEMPLATES,
        )
        print("\n" + "=" * 50)
        print("  重置完成！")
        print(f"  三元组总数: {stats['total_triples']}")
        print(f"  chunk 总数: {stats['total_chunks']}")
        print(f"  实体总数:   {stats['total_entities']}")
        print(f"  关系总数:   {stats['total_relations']}")
        print("=" * 50)

    elif args.command == "clean":
        # 清洗三元组文件 + 重建索引
        print("=" * 50)
        print("  PocketGraphRAG - 清洗三元组 + 重建索引")
        print("=" * 50)
        stats = clean_triples_file(args.data, backup=not args.no_backup)
        print(f"\n  原始三元组:   {stats['original']}")
        print(f"  剔除低质量:   {stats['removed']}")
        print(f"  保留三元组:   {stats['kept']}")
        if stats["backup_path"]:
            print(f"  原文件备份:   {stats['backup_path']}")
        if stats["removed"] == 0:
            print("\n  数据已干净，无需重建索引。")
            print("=" * 50)
        else:
            print("\n  即将重建索引...")
            build_index_with_data(args.data, args.index_dir, run_tests=False)
            print("=" * 50)
            print("  清洗 + 重建完成！")
            print("=" * 50)

    else:
        # 默认：全量构建
        build_index_with_data(args.data, args.index_dir, run_tests=True)


if __name__ == "__main__":
    main()
