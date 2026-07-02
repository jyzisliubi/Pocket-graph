# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2026-07-01

### Added
- **存储抽象对齐**: 新增 `KVStore` ABC + `JsonKVStorage` 实现 + `get_kv_store` 工厂函数，对标 LightRAG `BaseKVStorage`。用于文档原文、chunk→doc_id 映射、抽取缓存等结构化 KV 数据。支持纯内存模式和 JSON 文件持久化（原子写入）。24 个单元测试。
- **P1-1 Gleaning 多轮抽取**: 参考 microsoft/graphrag 的 gleaning 设计，`extract_triples_from_text` 新增 `gleaning_steps` 参数。首轮抽取后追问 LLM "是否遗漏"，循环 N 轮合并去重，显著提升召回率。配置项 `GLEANING_STEPS`（默认 0，向后兼容）。17 个单元测试。
- **P1-2 实体类型约束**: `DEFAULT_ENTITY_TYPES` 25 个水稻领域实体类型，`RelationSchema.build_prompt_constraint()` 加入实体类型软约束提示，引导 LLM 抽取规范实体名，减少"句子片段当实体"问题。5 个单元测试。
- **P1-3 异步并发抽取**: `extract_triples_from_text_async` + `extract_knowledge_graph_async`，`asyncio.Semaphore` 控制并发度（默认 4），大文档抽取显著加速。单个 chunk 失败不影响其他 chunk。8 个单元测试。
- **公开 benchmark 适配器**: `benchmark_adapters.py` 支持 HotpotQA / MuSiQue 数据集转换为项目 benchmark 格式，`build_corpus_from_benchmark` 辅助构建 KG 文档源。15 个单元测试。
- **examples/ 目录扩充**: 4 个可运行示例脚本（quickstart、compare_search_modes、extract_custom_data、call_rest_api）。

### Changed
- `extract_triples_from_text` 重构：抽取 `_parse_triples_result` 独立解析函数，JSON/delimiter 解析逻辑复用，gleaning 循环复用同一解析器。
- `extract_knowledge_graph` 透传 `gleaning_steps` 参数。
- 顶层包 `PocketGraphRAG.__init__` 导出存储抽象层（`VectorStore`, `GraphStore`, `KVStore`, `FAISSVectorStore`, `InMemoryGraphStore`, `JsonKVStorage`, `get_vector_store`, `get_graph_store`, `get_kv_store`）。

### Tests
- 测试基线从 442 → 476 passed（+34 新测试：gleaning 17 + entity_types 5 + async 8 + KVStore 24 - 重叠 20）。

## [0.3.1] - 2026-06-30

### Added
- **Schema 驱动关系归一化**: 1275 种碎片化关系名归一化为 292 种标准关系（减少 77%）。三层归一化：同义词字典 → 白名单 → 正则模式。
- **eval_harness 双向归一化匹配**: Relation Coverage 从 0.18 提升到 0.67（+267%）。
- **P0-3 delimiter 格式 fallback**: JSON 解析失败时改用 `<|#|>...<|#|>` 格式解析，7 个单元测试。
- **cli.py build_cmd bug 修复**: `build_index(data_path=...)` 调用修复。

## [0.3.0] - 2026-06-29

### Added
- **Incremental indexing** (`incremental_index.py`): document-level add/remove without full rebuild of the three FAISS indexes. New entities → `add_chunks`; affected entities → `remove_by_entity` + `add_chunks` (rebuilt from in-memory embeddings cache, zero model calls for the remove step). Triple-level dedup via persisted `triples_manifest.json` hash set. Legacy indexes auto-migrate on first incremental call (missing `embeddings.npy` → `reconstruct_n`; missing manifest → rebuilt from `data_path`).
- `FAISSIndex.add_chunks()` / `FAISSIndex.remove_by_entity()` / embeddings cache (`embeddings.npy`).
- `KGProcessor.process_entities()` / `KGProcessor.add_triples()` for selective re-chunking and in-memory merge.
- CLI subcommands `python -m PocketGraphRAG.build_index add --triples ...` and `reset`.
- Web UI "数据管理" tab now picks the incremental path automatically when a user index already exists; new "重置用户数据集" accordion.
- 26 unit tests covering KGProcessor new methods, FAISSIndex incremental ops, manifest, end-to-end `add_triples_incremental`, `reset_index`, and legacy backward-compat migration.

### Changed
- `requirements.txt` now installs the package via `-e .[dev]` to keep a single source of truth for dependencies.
- PyPI release workflow switched to Trusted Publishing (OIDC).
- Default first-run retrieval experience now points to `mix` instead of a weakened vector-only path, aligning runtime defaults, docs, and the Web UI.
- Web UI question submission now forwards the self-check toggle correctly, reducing Gradio endpoint mismatch noise during interactive QA.
- Windows startup no longer blocks on missing LLM credentials; it warns and continues so users can validate retrieval, sources, and graph state first.

### Documentation
- README / README-zh: corrected inaccurate "LightRAG English only" claim to "English-first prompts (tunable to Chinese)"; softened "100% high quality" overclaims to honest counts + a disclaimer that confidence is LLM-self-reported; replaced "Production-Ready" with "Tested (alpha)"; added Incremental Indexing section and updated Roadmap / project structure.
- README / README-zh / project metadata: tightened launch messaging around source-first installation, graph-first defaults, and the current release status.

## [0.2.0] - 2026-01-01

### Added
- KG dual-layer retrieval (LightRAG-style): `local`, `global`, `mix`, `kg_only` search modes.
- KG extraction v2 pipeline: semantic chunking → LLM extraction → entity alignment → deduplication → quality filter.
- Multi-source data import: TXT, Markdown, PDF (text + scanned), Word, images (OCR/VLM), web pages (Playwright).
- VLM multimodal support for direct knowledge extraction from images.
- PageRank-enhanced ranking and graph algorithms suite (community detection, shortest path).
- REST API server (FastAPI) with streaming SSE support.
- Desktop pet module (PyQt5) with multiple skins.
- 190+ unit tests, GitHub Actions CI/CD, Ruff linting.

## [0.1.0] - 2025-09-01

### Added
- Initial release: lightweight vertical-domain GraphRAG with FAISS + entity-level chunking.
- Gradio Web UI with Q&A, Data Management, and Knowledge Graph visualization tabs.
- Native Ollama support for fully offline local deployment.

[Unreleased]: https://github.com/JayZ/PocketGraphRAG/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/JayZ/PocketGraphRAG/releases/tag/v0.3.0
[0.2.0]: https://github.com/JayZ/PocketGraphRAG/releases/tag/v0.2.0
[0.1.0]: https://github.com/JayZ/PocketGraphRAG/releases/tag/v0.1.0
