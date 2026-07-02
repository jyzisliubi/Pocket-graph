"""存储抽象层 —— 可插拔的向量库 / 图存储后端

设计目标：
  - 默认实现保持零外部依赖（FAISS + NetworkX 风格的内存图）
  - 业务代码（rag_system / kg_reasoning）只依赖抽象接口，不感知具体后端
  - 新增后端（pgvector / Chroma / Milvus / Neo4j）只需实现接口，无需改动业务代码

模块组织：
  base.py              — VectorStore / GraphStore / KVStore 抽象基类
  faiss_store.py       — FAISSVectorStore（默认，包装现有 FAISSIndex）
  in_memory_graph.py   — InMemoryGraphStore（默认，包装 entity_relations dict）
  json_kv_store.py     — JsonKVStorage（默认，JSON 文件持久化）
  chroma_store.py      — ChromaVectorStore（可选，需 pip install chromadb）
  pgvector_store.py    — PgVectorStore（可选，需 pip install psycopg[binary]）
  factory.py           — get_vector_store / get_graph_store / get_kv_store 工厂函数

使用方式：
  # 推荐：通过工厂创建
  from PocketGraphRAG.core.storages import get_vector_store, get_graph_store, get_kv_store
  vs = get_vector_store(backend="faiss", model=model, dimension=512)
  gs = get_graph_store(backend="memory", entity_relations=er)
  kv = get_kv_store(backend="json", path="data/doc_store.json")

  # 或直接导入具体实现
  from PocketGraphRAG.core.storages import FAISSVectorStore, InMemoryGraphStore, JsonKVStorage
"""

from .base import GraphStore, KVStore, VectorStore
from .factory import get_graph_store, get_kv_store, get_vector_store
from .faiss_store import FAISSVectorStore
from .in_memory_graph import InMemoryGraphStore
from .json_kv_store import JsonKVStorage

__all__ = [
    "VectorStore",
    "GraphStore",
    "KVStore",
    "FAISSVectorStore",
    "InMemoryGraphStore",
    "JsonKVStorage",
    "get_vector_store",
    "get_graph_store",
    "get_kv_store",
]
