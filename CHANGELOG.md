# Changelog

All notable changes to PocketGraphRAG are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.7] - 2026-07-06

### Added
- **WORKSPACE 数据隔离**（对标 LightRAG v1.4.0）：`POCKET_WORKSPACE` 环境变量实现多租户/多数据集物理隔离，含路径穿越防护
- **LLM Cache**（对标 LightRAG v1.4.3 Redis cache）：`InMemoryCache`（LRU+TTL）+ `RedisCache` 双后端，`POCKET_LLM_CACHE=1` 启用，Redis 不可用自动降级内存
- **Entity-keyed Locks 防竞态**（对标 LightRAG v1.4.3）：`EntityLockManager`（RLock 可重入 + `lock_multiple` 防死锁），`incremental_index.py` 全局锁包装 `add/remove` 函数防止 FAISS lost-update
- **Phantom entities 清理**（对标 graphrag v3.0.9）：`GraphStore.cleanup_orphan_entities()` + CLI `pocketgraphrag cleanup --dry-run`
- **Docker 部署**：多阶段 `Dockerfile` + `docker-compose.yml`（含可选 Redis/Neo4j profiles）
- **GitHub Actions CI**：4 版本 Python 矩阵测试 + Docker 构建验证
- **SECURITY.md** 安全策略
- **K8s + Helm Chart**：`deploy/k8s/` manifests（Deployment/Service/Ingress/PVC/probes）+ `deploy/helm/` 模板化部署
- **多架构 Docker 构建**：linux/amd64 + linux/arm64，`docker-build-push.sh` 一键构建推送
- **Cosign 镜像签名 + SBOM**（对标 LightRAG 供应链安全）：`docker-publish.yml` GitHub OIDC keyless 签名 + SBOM 生成
- **CodeQL + pip-audit 安全扫描 CI**（对标 microsoft/graphrag）：`.github/workflows/security.yml` Python SAST + 依赖漏洞扫描
- **API 限流**（对标 LightRAG/RAGFlow）：slowapi 软依赖，`POCKET_RATE_LIMIT="100/minute"` 启用，未安装自动跳过
- **Prometheus 指标**：`prometheus-fastapi-instrumentator` 软依赖，`POCKET_METRICS=1` 暴露 `/metrics` 端点
- **中文 README**：`README-zh.md` 双语文档
- **CONTRIBUTING.md + CODEOWNERS + .editorconfig**：开源社区标配
- **mkdocs 文档完善**：architecture / installation / deployment 实际内容 + nav 重构
- **前端复制答案按钮**：Chat 消息操作 UX（对标 ChatGPT/Claude/LightRAG WebUI）
- **Issue 模板**：.yml 格式（bug / feature / question）
- **API OpenAPI tags 分组**：Swagger UI `/docs` 按 5 分组（QA / Graph / Documents / Settings / System）展示路由，对标 LightRAG/microsoft graphrag 文档专业度
- **文档预览**：`GET /api/documents/{filename}/raw` 端点 + 前端 Modal，支持 txt/md/pdf/docx 多格式纯文本预览（对标 ChatGPT/Claude 文件预览），超大文件自动截断到 50,000 字符
- **Sitemap / RSS 数据导入**：`POST /api/documents/import-url` 端点 + 前端 Dialog，支持 sitemap.xml（含 sitemapindex 递归）/ RSS 2.0 / Atom 1.0 批量导入（对标 RAGFlow 多数据源），feed 自带 `content:encoded` 优先避免逐页抓取

### Fixed
- `/api/retrieve` 端点 `citation_id` 为 null → 现在正确返回 1/2/3
- `/api/graph/subgraph` POST 端点 422 错误（前端用 JSON body，后端期望 query param）→ 改用 `Body(...)` 接受 JSON
- `/api/qa/stream` 的 `done` 事件 `answer` 字段为空（rag_system yield `"answer"` 键，api_server 读 `"full_answer"`）→ 键名对齐
- 版本号不一致 bug：`__init__.py` 仍是 `0.3.2`，与 `pyproject.toml`/`api_server.py` 的 `0.3.7` 不一致 → 单源版本号（`pyproject.toml` 用 `dynamic = ["version"]` 从 `__init__.py` 读取，`api_server.py` 用 `from . import __version__`）

## [0.3.6] - 2026-07-05

### Added
- **bge-m3 跨语言检索**：模型别名表（`POCKET_EMBEDDING_MODEL=bge-m3` 自动展开），索引版本化（`embedding_model.json` 指纹校验），动态维度（移除硬编码 512）
- **跨语言提示**：`_cross_lingual_hint` 检测非中文查询并建议切换 bge-m3
- **Neo4j + pgvector 后端**：`Neo4jGraphStore`（GDS fallback）+ `PgVectorStore`（HNSW/IVFFlat + O(1) 实体删除）
- **Langfuse Tracing**：`tracing.py` 可选依赖 + NoOp fallback
- **KG-aware Reranker**：Cross-Encoder 重排序 + 单元测试
- **Multi-Model KG Fusion**：CLI `multi-extract` + API `/api/documents/extract-multi` + 前端 Dialog，Hit Rate 0.80→0.86

## [0.3.5] - 2026-07-05

### Added
- **多分块策略**：fixed / recursive / paragraph / semantic 四种策略，统一入口 `chunk_with_strategy()`
- **Citation 引用回溯**：`Source.citation_id` 字段 + 答案中 `[1][2]` 标注回溯
- **mkdocs 文档站**：Material 主题 + GitHub Actions 自动部署到 GitHub Pages
- **多模态 PDF 解析**：文本层 + 表格层（pdfplumber → Markdown）+ 图片层（PyMuPDF）+ OCR 兜底
- **Setup Wizard**：交互式 `.env` 配置，6 providers + 3 embeddings + 4 storage backends
- **PyPI 发布 workflow**：OIDC Trusted Publishing
- **HuggingFace Space** demo：Gradio UI 在线体验
- **API Key 认证**：多 key + Bearer 认证 + 4 角色 LLM 配置

## [0.3.4] - 2026-07-04

### Added
- DRIFT Search 三阶段（predict → retrieve → expand）
- HyDE 假设性文档嵌入
- Query Router 查询路由
- Self-Check 答案自检
- 拒答逻辑（基于 `seed_entities=[]` 判断，`matched_relations` 噪声不可信）

## [0.3.0] - 2026-07-03

### Added
- 首个公开版本
- GraphRAG 核心引擎：KG 抽取 + FAISS 向量检索 + Personalized PageRank + Louvain 社区检测
- HotpotQA MRR 达 LightRAG 的 2.7×
- Zero-LLM Query 模式（纯检索不调用 LLM）
- React + TypeScript Web UI
- FastAPI REST API + SSE 流式
- 601 单元测试

[0.3.7]: https://github.com/jyzisliubi/Pocket-graph/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/jyzisliubi/Pocket-graph/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/jyzisliubi/Pocket-graph/compare/v0.3.0...v0.3.5
[0.3.0]: https://github.com/jyzisliubi/Pocket-graph/releases/tag/v0.3.0
