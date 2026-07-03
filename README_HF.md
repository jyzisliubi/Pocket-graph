---
title: PocketGraphRAG
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - rag
  - knowledge-graph
  - graphrag
  - lightrag
  - faiss
  - react
  - fastapi
short_description: Local-first GraphRAG that beats LightRAG on HotpotQA
---

# PocketGraphRAG on HuggingFace Space

A local-first GraphRAG that beats LightRAG on public HotpotQA — zero LLM calls at query time.

Upload docs → extract triples → build a private graph → ask with citations. No Neo4j, no cloud required.

## Features

- **Multi-Model KG Fusion** — Merge KGs from different LLMs to cover blind spots
- **Zero-LLM Query** — Pure embedding + graph traversal at query time
- **Deterministic Retrieval** — `cid` tie-breaker guarantees reproducible MRR
- **React Professional UI** — Vite + TypeScript + Tailwind + shadcn/ui, Dark/Light dual theme
- **d3-force Knowledge Graph** — Interactive force-directed visualization
- **SSE Streaming** — Real-time answer generation with source citations

## Usage

The Space starts with a built-in movie knowledge graph demo. Just type a question in the Chat tab:

- "无间道是什么类型的电影？"
- "盗梦空间的导演是谁？"
- "克里斯托弗·诺兰导演了哪些电影？"

Switch to the **Knowledge Graph** tab to explore the graph visually, or **Analytics** for statistics.

## Full Source

[github.com/jyzisliubi/Pocket-graph](https://github.com/jyzisliubi/Pocket-graph)
