# Awesome-List 提交材料

本文件提供向各 awesome-list 提交 PocketGraphRAG 的现成文案，直接复制使用。

---

## 1. 提交目标列表

按优先级排序：

| 仓库 | star | 适合分类 | 难度 |
|---|---|---|---|
| [ggerganov/awesome-llm](https://github.com/ggerganov/awesome-llm) | 20k+ | RAG / Chinese | 中 |
| [hymie001/RAG-Survey](https://github.com/hymie001/RAG-Survey) | 2k+ | Graph RAG | 低 |
| [xing5/awesome-local-llm](https://github.com/xing5/awesome-local-llm) | 1k+ | Local-first | 中 |
| [NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) | 6k+ | RAG | 中 |
| [shubham-uspehra/awesome-rag](https://github.com/shubham-uspehra/awesome-rag) | 1k+ | RAG | 低 |
| [ZhongTr0n/iOS_Chinese_Reading](https://github.com/) | - | 中文 AI | - |

---

## 2. 标准项目描述（英文，用于 PR body）

```markdown
## PocketGraphRAG

A lightweight, local-first GraphRAG framework for vertical domains. Pure FAISS (no Neo4j), native Ollama support for fully offline runs, and Chinese-optimized with BGE-zh embeddings + Chinese prompts.

**Highlights:**
- KG dual-layer retrieval (local/global/mix/kg_only) — LightRAG-style
- KG extraction v2 pipeline: semantic chunking → alignment → dedup → quality filter
- Multi-source import: TXT / Markdown / PDF / Word / Images (OCR/VLM) / Web (Playwright)
- PageRank-enhanced ranking + community detection + shortest path
- Built-in Gradio Web UI (Q&A + graph visualization + data management)
- FastAPI REST API with streaming SSE
- 190+ unit tests, CI on 3 OS × 4 Python versions

**Differentiators vs Microsoft GraphRAG / LightRAG:**
- Chinese-native (BGE-zh + Chinese prompts; competitors are English-first)
- Vertical-domain ready (built-in rice-disease / movie / cat-encyclopedia examples)
- Low indexing cost (single-pass extraction vs GraphRAG's multi-round community reports)
- All-in-one Web UI (Q&A + graph + data management, no separate frontend)

**Links:**
- Repo: https://github.com/JayZ/PocketGraphRAG
- Docs: https://JayZ.github.io/PocketGraphRAG/
- License: MIT
```

---

## 3. 中文版描述（用于中文社区/小红书/公众号）

```markdown
## PocketGraphRAG

轻量级、本地优先的垂直领域 GraphRAG 框架。纯 FAISS（无需 Neo4j），原生 Ollama 离线支持，中文优化（BGE-zh + 中文 Prompt）。

**核心特性：**
- KG 双层检索（local/global/mix/kg_only）— LightRAG 风格
- KG 抽取 v2：语义切分 → 实体对齐 → 去重 → 质量过滤
- 多源导入：TXT / Markdown / PDF / Word / 图片(OCR/VLM) / 网页(Playwright)
- PageRank 排序 + 社区发现 + 最短路径
- 内置 Gradio Web UI（问答 + 图谱可视化 + 数据管理）
- FastAPI REST API + 流式 SSE
- 190+ 单测，3 OS × 4 Python 版本 CI

**vs 微软 GraphRAG / LightRAG 的差异：**
- 中文原生（BGE-zh + 中文 prompt；竞品英文优先）
- 垂直领域开箱即用（内置电影/猫科示例 + 中文 prompt）
- 索引成本低（单趟抽取 vs GraphRAG 多轮社区报告）
- Web UI 三合一（问答+图谱+数据管理，无需另起前端）

**链接：**
- 仓库：https://github.com/JayZ/PocketGraphRAG
- 文档：https://JayZ.github.io/PocketGraphRAG/
- 许可证：MIT
```

---

## 4. awesome-llm 提交 PR 模板

提交到 `ggerganov/awesome-llm` 的 RAG 章节时，PR 标题和 body 模板：

**PR 标题：**
```
Add PocketGraphRAG to RAG section
```

**PR body：**
```markdown
Added PocketGraphRAG to the RAG section.

- [x] Project is open source (MIT)
- [x] Has clear documentation and README
- [x] Has tests and CI
- [x] Actively maintained (latest release 2026-01)
- [x] Provides unique value: Chinese-optimized local-first GraphRAG with Web UI

One-line description for the list:
> PocketGraphRAG - Lightweight, local-first GraphRAG for vertical domains. Pure FAISS, native Ollama, Chinese-optimized. ([GitHub](https://github.com/JayZ/PocketGraphRAG))
```

---

## 5. 提交前检查

- [ ] 确认 GitHub 仓库设为 Public
- [ ] 确认 README 顶部 demo.gif 正常渲染
- [ ] 确认 CI 徽章是绿的（push 后等 Actions 跑完）
- [ ] 确认有至少 1 个 GitHub Release（打 v0.2.0 tag）
- [ ] 给仓库加 topics：`graphrag` `rag` `knowledge-graph` `llm` `faiss` `ollama` `chinese` `local-first`

---

## 6. 仓库 Topics（GitHub 仓库页面 → About → 齿轮 → Topics）

建议添加以下 topics（提升被搜索到的概率）：
```
graphrag  rag  knowledge-graph  llm  retrieval-augmented-generation
faiss  ollama  chinese-nlp  local-first  gradio
fastapi  knowledge-graph-visualization  entity-extraction
```
