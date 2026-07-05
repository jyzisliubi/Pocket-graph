---
title: PocketGraphRAG Demo
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
tags:
  - graphrag
  - rag
  - knowledge-graph
  - llm
  - chinese-embedding
---

# PocketGraphRAG HuggingFace Space

Local-first GraphRAG with Multi-Model KG Fusion, Zero-LLM Query, and 2.7x LightRAG MRR on HotpotQA.

## Features

- 🎯 **Zero-LLM Query**: retrieve without LLM keyword extraction
- 🔀 **Multi-Model KG Fusion**: ensemble extraction from multiple LLMs
- 🌐 **Cross-lingual**: bge-m3 alias for multilingual retrieval
- 📊 **KG-aware Reranker**: cross-encoder + entity boost
- 📑 **Citation traceability**: [1][2] annotations in answers

## Usage

1. Upload a document (TXT/MD/PDF/Word)
2. Wait for KG extraction
3. Ask questions in any supported language

## Links

- [GitHub](https://github.com/jyzisliubi/Pocket-graph)
- [Documentation](https://jyzisliubi.github.io/Pocket-graph/)
- [PyPI](https://pypi.org/project/pocketgraphrag/)
