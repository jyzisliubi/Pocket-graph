<div align="center">

# PocketGraphRAG

**本地优先的 GraphRAG —— 在公开 HotpotQA 上超越 LightRAG，查询时零 LLM 调用**<br>
上传文档 → 抽取三元组 → 构建私有图谱 → 带引用问答。无需 Neo4j，无需云服务。

[![CI](https://github.com/jyzisliubi/Pocket-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/jyzisliubi/Pocket-graph/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Docker](https://img.shields.io/badge/Docker-支持-blue.svg)](./Dockerfile)
[![中文](https://img.shields.io/badge/语言-中文-red.svg)](#)
[![English](https://img.shields.io/badge/Lang-English-lightgrey.svg)](./README.md)

[English](./README.md) | [更新日志](./CHANGELOG.md) | [贡献指南](./CONTRIBUTING.md) | [部署指南](./deploy/README.md)

</div>

---

### ⚡ 30 秒安装运行

```bash
# 安装
pip install -e .                         # 源码安装（推荐）
# pip install "pocketgraphrag[cli]"      # 添加 pocketgraphrag CLI

# 运行
pocketgraphrag webui                     # React Web UI http://localhost:8000
pocketgraphrag ask "你的问题"             # 单次问答，带引用来源
```

### 🐳 Docker 部署

```bash
# 基础部署
docker-compose up -d

# 启用 Redis 缓存 + Neo4j 图存储
docker-compose --profile cache --profile graph up -d
```

<a name="hotpotqa-vs-lightrag"></a>
### 🏆 HotpotQA vs. LightRAG（公开 benchmark，N=50，top_k=25）

| 框架 | Hit Rate ↑ | MRR ↑ | 查询时 LLM |
|------|:----------:|:-----:|:----------:|
| **PocketGraphRAG** | **0.86** | **0.5633** | **零调用**（纯 embedding + 图遍历） |
| LightRAG v1.5.4 | 0.82 | 0.2093 | 需要（关键词抽取） |
| **提升** | +0.04 | **2.7×** | 查询时零 LLM 调用 |

> 复现：`python bench_data/eval_merged.py`，数据集：`bench_data/hotpotqa_50.json`

---

## 核心特性

- **🆕 多模型 KG 融合（独创）** — 融合多个 LLM 抽取的 KG，覆盖每个模型盲点；HotpotQA Hit Rate 0.80 → 0.86
- **KG 双层检索** — `local`（实体邻域）+ `global`（关系嵌入）+ `mix` 模式
- **Zero-LLM Query** — 纯检索模式，查询时不调用 LLM，零成本
- **DRIFT Search** — 三阶段动态推理灵活遍历（predict → retrieve → expand）
- **HyDE 假设性文档嵌入** — 用 LLM 生成假设文档提升短问题召回
- **KG-aware Reranker** — Cross-Encoder 重排序（bge-reranker-v2-m3）
- **Citation 引用回溯** — 答案中 `[1][2]` 标注可回溯到来源
- **4 种分块策略** — fixed / recursive / paragraph / semantic
- **多模态 PDF 解析** — 文本层 + 表格层 + 图片层 + OCR 兜底
- **WORKSPACE 数据隔离** — 多租户/多数据集物理隔离（v0.3.7）
- **LLM Cache** — InMemoryCache(LRU+TTL) + RedisCache 双后端（v0.3.7）
- **Entity-keyed Locks** — 防并发增量索引竞态（v0.3.7）
- **多后端存储** — FAISS / Neo4j / pgvector / Chroma
- **跨语言检索** — bge-m3 多语言 embedding 支持
- **Langfuse Tracing** — LLM 可观测性
- **RAGAS 评测** — 内置 RAGAS 0.1.x + 0.4+ 双 API 兼容

## 快速开始

### 1. 环境准备

```bash
# Python 3.9+
git clone https://github.com/jyzisliubi/Pocket-graph.git
cd Pocket-graph
pip install -e ".[web,docs]"

# （可选）启动 Ollama 作为本地 LLM
ollama pull qwen2.5:7b
```

### 2. 构建索引

```bash
# 使用内置电影 KG 数据集
pocketgraphrag build

# 或从文档抽取三元组
pocketgraphrag extract --input your_doc.txt
pocketgraphrag build
```

### 3. 开始问答

```bash
# CLI 问答
pocketgraphrag ask "霸王别姬的导演是谁"

# 启动 Web UI
pocketgraphrag webui

# 交互式 REPL
pocketgraphrag shell
```

### 4. 多模型 KG 融合（独创特性）

```bash
# 用多个 LLM 抽取同一份文档并融合
pocketgraphrag multi-extract --input doc.txt --models qwen-flash,qwen-max

# 或通过 API
curl -X POST http://localhost:8000/api/documents/extract-multi \
  -H "Content-Type: application/json" \
  -d '{"filename":"doc.txt","models":["qwen-flash","qwen-max"]}'
```

## 部署方式

| 方式 | 命令 | 适用场景 |
|------|------|----------|
| **pip 安装** | `pip install pocketgraphrag` | 本地开发 |
| **Docker** | `docker-compose up -d` | 快速部署 |
| **K8s** | `kubectl apply -f deploy/k8s/` | 生产集群 |
| **Helm** | `helm install my-release deploy/helm/pocketgraphrag/` | K8s 模板化 |
| **HuggingFace Space** | 见 `huggingface_space/` 目录 | 在线体验 |

详细部署说明见 [部署指南](./deploy/README.md)。

## 检索模式

| 模式 | 说明 | LLM 需要 |
|------|------|----------|
| `vector` | 纯向量检索 | 否 |
| `local` | KG 实体邻域检索 | 否 |
| `global` | KG 关系嵌入检索 | 否 |
| `mix` | 向量 + KG 融合（默认） | 否 |
| `kg_only` | 纯 KG 检索 | 否 |
| `drift` | DRIFT Search 三阶段 | 是 |
| `hyde` | HyDE 假设性文档 | 是 |
| `global_summary` | 社区摘要检索 | 是 |

## 配置

通过环境变量或 `.env` 文件配置：

```bash
# 检索配置
POCKET_SEARCH_MODE=mix
POCKET_TOP_K=5

# LLM 配置
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_API_BASE=http://localhost:11434/v1

# Embedding 配置
POCKET_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
# 跨语言：POCKET_EMBEDDING_MODEL=bge-m3

# v0.3.7 新特性
POCKET_WORKSPACE=default                    # 工作区隔离
POCKET_LLM_CACHE=1                          # LLM 缓存
POCKET_LLM_CACHE_BACKEND=redis              # memory / redis
POCKET_LLM_CACHE_REDIS_URL=redis://localhost:6379/0

# API 认证
POCKET_API_KEYS=key1,key2,key3              # 多 key 轮换
POCKET_API_AUTH_ENABLED=1                   # 启用认证
```

使用 `python -m PocketGraphRAG.setup_wizard` 启动交互式配置向导。

## 项目结构

```
PocketGraphRAG/
├── PocketGraphRAG/           # 核心 Python 包
│   ├── rag_system.py         # RAG 主引擎
│   ├── kg_extractor.py       # KG 三元组抽取
│   ├── kg_reasoning.py       # KG 双层检索 + DRIFT
│   ├── llm.py                # 多 LLM 后端集成
│   ├── llm_cache.py          # LLM 缓存（v0.3.7）
│   ├── concurrency.py        # 实体级锁（v0.3.7）
│   ├── api_server.py         # FastAPI REST API
│   ├── cli.py                # Typer CLI
│   ├── data_importer.py      # 多模态文档导入
│   ├── eval_harness.py       # RAGAS 评测
│   └── core/storages/        # 存储后端抽象
├── frontend/                 # React + TypeScript Web UI
├── deploy/                   # K8s + Helm 部署
├── docs/                     # mkdocs 文档
├── Dockerfile                # Docker 镜像
├── docker-compose.yml        # Docker Compose
└── pyproject.toml            # Python 项目配置
```

## 性能对比

### HotpotQA Benchmark（N=50，top_k=25）

| 框架 | Hit Rate | MRR | 查询时 LLM | 首次索引 LLM |
|------|:--------:|:---:|:----------:|:------------:|
| **PocketGraphRAG** | **0.86** | **0.5633** | 零 | 可选 |
| LightRAG v1.5.4 | 0.82 | 0.2093 | 需要 | 需要 |
| Naive RAG | 0.68 | 0.1820 | 需要 | 不需要 |

### 多模型 KG 融合效果

| 策略 | 三元组数 | Hit Rate | MRR |
|------|:--------:|:--------:|:---:|
| 单模型 qwen-flash | 185 | 0.80 | 0.5200 |
| 单模型 qwen-max | 152 | 0.78 | 0.5050 |
| **融合 union** | **247** | **0.86** | **0.5633** |

## 贡献

欢迎提交 Issue 和 PR！请先阅读 [贡献指南](./CONTRIBUTING.md)。

## 许可证

MIT License - 见 [LICENSE](./LICENSE)
