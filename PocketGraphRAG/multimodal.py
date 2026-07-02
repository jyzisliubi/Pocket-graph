"""
PocketGraphRAG 多模态支持模块

提供图文混合检索基础框架：
- 实体-图片关联管理
- 图片 Embedding 索引（基于 CLIP / SentenceTransformer）
- 文本搜图、图搜文、图文融合检索

设计原则：
- 可选启用，不影响纯文本模式
- 兼容 SentenceTransformer 中支持 CLIP 的模型
- 失败自动降级为纯文本模式
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np


class MultimodalIndex:
    """多模态索引管理器

    支持：
    - 实体关联多张图片
    - 图片 Embedding 索引（FAISS）
    - 文本搜图、图搜文
    """

    def __init__(
        self,
        index_dir: str,
        model=None,
        image_dir: str = None,
    ):
        """
        Args:
            index_dir: 索引目录（保存图片 embedding 和映射关系）
            model: SentenceTransformer / CLIP 模型实例
            image_dir: 图片存储目录
        """
        self.index_dir = index_dir
        self.model = model
        self.image_dir = image_dir or os.path.join(index_dir, "images")

        self.image_index = None  # FAISS 索引
        self.image_paths = []  # 图片路径列表，与索引一一对应
        self.image_entities = []  # 每张图片关联的实体名

        # 实体 -> 图片索引列表 的映射
        self.entity_image_map: Dict[str, List[int]] = {}

        self._available = False
        self._try_load()

    def _try_load(self):
        """尝试加载已有索引"""
        index_path = os.path.join(self.index_dir, "image_faiss.index")
        mapping_path = os.path.join(self.index_dir, "image_mapping.json")

        if os.path.exists(index_path) and os.path.exists(mapping_path):
            try:
                import faiss

                self.image_index = faiss.read_index(index_path)
                with open(mapping_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.image_paths = data.get("image_paths", [])
                self.image_entities = data.get("image_entities", [])
                self.entity_image_map = {
                    k: v for k, v in data.get("entity_image_map", {}).items()
                }
                self._available = True
                print(f"[OK] 多模态索引加载完成: {len(self.image_paths)} 张图片")
            except Exception as e:
                print(f"[WARN] 多模态索引加载失败: {e}")
                self._available = False

    @property
    def available(self) -> bool:
        """多模态功能是否可用"""
        return self._available and self.model is not None

    def add_images(
        self,
        entity: str,
        image_paths: List[str],
    ):
        """为实体添加图片并构建索引

        Args:
            entity: 实体名称
            image_paths: 图片路径列表
        """
        if self.model is None:
            raise RuntimeError("多模态模型未初始化，无法添加图片")

        # 检查模型是否支持 encode_image
        if not hasattr(self.model, "encode") and not hasattr(
            self.model, "encode_image"
        ):
            # 尝试用 text 编码模拟（对于不支持图片的模型）
            pass

        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"[WARN] 图片不存在: {img_path}")
                continue

            try:
                # 尝试用 CLIP 模式编码图片
                if hasattr(self.model, "encode_image"):
                    from PIL import Image

                    img = Image.open(img_path).convert("RGB")
                    emb = self.model.encode_image([img])
                else:
                    # 降级：用文件名作为文本描述来编码
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    emb = self.model.encode([img_name], normalize_embeddings=True)

                emb = np.array(emb, dtype="float32")

                if self.image_index is None:
                    import faiss

                    dim = emb.shape[1]
                    self.image_index = faiss.IndexFlatIP(dim)

                self.image_index.add(emb)
                self.image_paths.append(img_path)
                self.image_entities.append(entity)

                # 更新实体映射
                idx = len(self.image_paths) - 1
                if entity not in self.entity_image_map:
                    self.entity_image_map[entity] = []
                self.entity_image_map[entity].append(idx)

            except Exception as e:
                print(f"[WARN] 图片编码失败 {img_path}: {e}")

        self._available = self.image_index is not None and self.image_index.ntotal > 0

    def save(self):
        """保存多模态索引到磁盘"""
        if self.image_index is None:
            return

        os.makedirs(self.index_dir, exist_ok=True)

        import faiss

        faiss.write_index(
            self.image_index,
            os.path.join(self.index_dir, "image_faiss.index"),
        )

        mapping_data = {
            "image_paths": self.image_paths,
            "image_entities": self.image_entities,
            "entity_image_map": self.entity_image_map,
        }
        with open(
            os.path.join(self.index_dir, "image_mapping.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)

        print(f"[OK] 多模态索引已保存: {len(self.image_paths)} 张图片")

    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[Tuple[str, str, float]]:
        """用文本搜索相关图片

        Args:
            query: 查询文本
            top_k: 返回 top_k 张图片
            threshold: 相似度阈值

        Returns:
            [(实体名, 图片路径, 相似度分数), ...]
        """
        if not self.available or self.image_index is None:
            return []

        try:
            # CLIP 模型用 encode_text，普通模型用 encode
            if hasattr(self.model, "encode_text"):
                query_vec = self.model.encode_text([query])
            else:
                query_vec = self.model.encode([query], normalize_embeddings=True)

            query_vec = np.array(query_vec, dtype="float32")
            scores, indices = self.image_index.search(
                query_vec, min(top_k, self.image_index.ntotal)
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or score < threshold:
                    continue
                results.append(
                    (self.image_entities[idx], self.image_paths[idx], float(score))
                )
            return results

        except Exception as e:
            print(f"[WARN] 文本搜图失败: {e}")
            return []

    def search_by_image(
        self,
        image_path: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[Tuple[str, str, float]]:
        """用图片搜索相关图片/实体

        Args:
            image_path: 查询图片路径
            top_k: 返回 top_k
            threshold: 相似度阈值

        Returns:
            [(实体名, 图片路径, 相似度分数), ...]
        """
        if not self.available or self.image_index is None:
            return []

        try:
            if hasattr(self.model, "encode_image"):
                from PIL import Image

                img = Image.open(image_path).convert("RGB")
                query_vec = self.model.encode_image([img])
            else:
                # 降级：用文件名作为文本
                img_name = os.path.splitext(os.path.basename(image_path))[0]
                query_vec = self.model.encode([img_name], normalize_embeddings=True)

            query_vec = np.array(query_vec, dtype="float32")
            scores, indices = self.image_index.search(
                query_vec, min(top_k, self.image_index.ntotal)
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or score < threshold:
                    continue
                # 跳过自身（相似度为 1.0 的第一张）
                if os.path.abspath(self.image_paths[idx]) == os.path.abspath(
                    image_path
                ):
                    continue
                results.append(
                    (self.image_entities[idx], self.image_paths[idx], float(score))
                )
            return results

        except Exception as e:
            print(f"[WARN] 以图搜图失败: {e}")
            return []

    def get_entity_images(self, entity: str) -> List[str]:
        """获取某个实体的所有图片路径

        Args:
            entity: 实体名

        Returns:
            图片路径列表
        """
        indices = self.entity_image_map.get(entity, [])
        return [self.image_paths[i] for i in indices]

    def stats(self) -> dict:
        """获取多模态索引统计信息"""
        return {
            "available": self.available,
            "total_images": len(self.image_paths),
            "total_entities_with_images": len(self.entity_image_map),
            "image_dir": self.image_dir,
        }


def is_multimodal_model(model_name: str) -> bool:
    """判断模型是否为多模态（CLIP）模型

    Args:
        model_name: 模型名称

    Returns:
        True 如果模型名包含 CLIP 相关关键词
    """
    clip_keywords = ["clip", "vit", "multimodal", "img", "image"]
    model_lower = model_name.lower()
    return any(kw in model_lower for kw in clip_keywords)
