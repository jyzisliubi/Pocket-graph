# Architecture

PocketGraphRAG is a **local-first GraphRAG** framework. Unlike heavy enterprise
solutions, it trades global summarization (which requires many LLM calls at
query time) for **KG dual-layer retrieval + PageRank-enhanced ranking**, so the
query path can run with **zero LLM calls** in pure-retrieval modes.

## High-Level Pipeline

```
User question
    │
    ▼
[Conversation Rewrite] ConversationMemory → multi-turn context completion
    │
    ▼
[Multi-hop Decomposition] Complex questions → sub-queries
    │
    ▼
[Dual-Layer Retrieval] search_mode switches the path:
    vector  → pure vector similarity
    local   → KG entity match + BFS neighborhood
    global  → KG relation embedding match
    mix     → vector + KG fusion (default)
    kg_only → pure KG (local + global)
    drift   → DRIFT search (predict → retrieve → expand)
    hyde    → HyDE hypothetical document embedding
    │
    ▼
[Prompt Construction] Knowledge context + user question
    │
    ▼
[LLM Generation] Knowledge-grounded answer with [1][2] citations
```

All advanced features are **off by default** and enabled on demand, so the
default path stays cheap and predictable.

## Module Map

| Module | Responsibility |
|--------|----------------|
| `rag_system.py` | Main engine: orchestrates rewrite → retrieve → generate |
| `kg_extractor.py` | KG triple extraction (v2 5-stage pipeline) |
| `kg_reasoning.py` | KG dual-layer retrieval + DRIFT search entry |
| `drift_search.py` | Three-stage dynamic reasoning traversal |
| `hyde.py` | Hypothetical document embedding for short queries |
| `build_index.py` | FAISS vector index build/load |
| `incremental_index.py` | Document-level incremental add/remove with locks |
| `community_summarizer.py` | Community detection + hierarchical summaries |
| `multihop.py` | Multi-hop query decomposition |
| `query_router.py` | Auto-selects search mode per query |
| `answer_verifier.py` | Self-check / refuse-on-empty logic |
| `llm.py` | Multi-LLM backend (Ollama / DashScope / SiliconFlow / OpenAI) |
| `llm_cache.py` | LLM response cache (InMemory LRU+TTL / Redis) |
| `concurrency.py` | Entity-keyed locks for concurrent indexing |
| `api_server.py` | FastAPI REST API + SSE streaming |
| `cli.py` | Typer CLI |
| `data_importer.py` | Multi-modal document import |
| `eval_harness.py` | RAGAS evaluation harness |
| `core/storages/` | Storage backend abstractions |

## Storage Layer

### GraphStore (ABC)

`core/storages/base.py` defines the `GraphStore` ABC. The reference
implementation is `InMemoryGraphStore`:

- **Personalized PageRank** via NumPy vectorized iteration (no NetworkX in the
  hot path), with **`cid` (chunk_id ascending) tie-breaker** for deterministic
  ordering — fixes the PYTHONHASHSEED non-determinism that previously caused
  flaky MRR.
- **Orphan entity cleanup** (`cleanup_orphan_entities`): removes entities that
  have no edges after a document is deleted.

Pluggable backends: `Neo4jGraphStore`, `PostgresAGEGraphStore` (PG + Apache AGE).

### VectorStore (ABC)

`FAISSVectorStore` is the reference implementation:

- **Entity → chunk_ids inverted index** for O(1) retrieval given a seed entity.
- **Dynamic dimension**: no hardcoded `512` — the dimension is inferred from
  the embedding model, so switching `bge-small-zh` → `bge-m3` "just works".
- **Index versioning**: `save()` writes `embedding_model.json` with the model
  fingerprint; `load()` validates dimension compatibility and raises
  `RuntimeError` on mismatch (prevents silent corruption when the model changes).

Pluggable backends: `pgvector`, `Chroma`.

## Retrieval Scoring

The ranking combines three signals with discrete weights (continuous PPR
multiplication was tried and abandoned — see [Lessons](#lessons-learned)):

| Signal | Weight | Meaning |
|--------|:------:|---------|
| Seed entity (query match) | **2.0** | Directly matched the query embedding |
| Relation-value entity | **1.5** | Appeared as the object of a matched relation |
| Expanded entity (BFS neighbor) | **1.0** | Reached via neighborhood expansion |
| Top-10 PPR entity bonus | **+1.3** | Discrete boost (more effective than weighted multiply) |

Fusion strategy: `rrf` (default, `k=60`) or `weighted`. Configurable via
`POCKET_FUSION_STRATEGY` / `POCKET_RRF_K`.

## v0.3.7 Concurrency & Isolation

### WORKSPACE data isolation

`POCKET_WORKSPACE` environment variable physically separates index, docs and
data directories per workspace — enables multi-tenant / multi-dataset deployments
from a single binary. `default` keeps backward compatibility (no path change).

### LLM Cache

Two backends, switchable via `POCKET_LLM_CACHE_BACKEND`:

- `InMemoryCache` — LRU + TTL, in-process, zero dependencies.
- `RedisCache` — shared across replicas (production multi-replica K8s).

`should_cache_role()` whitelists which LLM roles are cacheable (extraction and
summarization are; one-shot QA is not, to keep answers fresh).

### Entity-keyed Locks

`concurrency.EntityLockManager` provides **reentrant `RLock` per entity name**.
`lock_multiple()` acquires locks in sorted order to prevent deadlocks during
concurrent incremental indexing. This is what lets two documents be indexed in
parallel without corrupting the shared KG.

### Phantom Cleanup

When a document is removed, `remove_document_incremental` uses the
`doc_id → triple_keys` reverse mapping to delete exactly that document's
triples, then sweeps orphan entities that lost all edges.

## Multi-Model KG Fusion (signature feature)

Different LLMs have different extraction blind spots. `multi-extract` runs the
same document through multiple LLMs (e.g. `qwen-flash` + `qwen-max`) and
**unions** their triple sets — an ensemble-learning effect.

| Strategy | Triples | Hit Rate | MRR |
|----------|:-------:|:--------:|:---:|
| single qwen-flash | 185 | 0.80 | 0.5200 |
| single qwen-max | 152 | 0.78 | 0.5050 |
| **fused union** | **247** | **0.86** | **0.5633** |

> Note: stronger models (qwen-max) extract **more conservatively** — fewer
> triples, not more. The union is what recovers coverage.

## Cross-Lingual Retrieval

`bge-small-zh` is a Chinese-only model. `_cross_lingual_hint` detects non-Chinese
queries and logs a suggestion to switch to `bge-m3` (multilingual). Setting
`POCKET_EMBEDDING_MODEL=bge-m3` enables true cross-lingual retrieval.

Short model aliases are supported: `bge-m3` auto-expands to `BAAI/bge-m3`, with
the local cache path prioritized over HF hub download.

## Citation Traceability

Answers carry inline `[1][2]` markers. Each marker maps back to a `Source`
with `entity`, `text`, `score`, and `citation_id`. The `/api/retrieve` and
`/api/qa` endpoints return the same source shape so the UI can render
clickable citations consistently.

## Lessons Learned

- **Set iteration broke MRR determinism**: PYTHONHASHSEED randomizes set order,
  causing flaky test results. Fix: `cid` (chunk_id ascending) tie-breaker.
- **Continuous PPR weights were too weak**: `pagerank_weight=0.3` multiplied
  into scores made the PPR contribution invisible. Fix: discrete top-10 boost.
- **Stronger model ≠ more triples**: qwen-max extracts more conservatively than
  qwen-flash. Multi-model union is the cheap, high-recall strategy.
- **Don't trust "matched_relations" for refusal**: generic relations like
  "类型" can be vector noise. Refusal logic must key off `seed_entities == []`.
