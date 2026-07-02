# PocketGraphRAG

**A Lightweight, Local-First GraphRAG Framework for Vertical Domains.**

PocketGraphRAG is a lightweight GraphRAG (Retrieval-Augmented Generation based on
Knowledge Graphs) framework designed for vertical domains. Unlike heavy
enterprise solutions, it focuses on **simplicity, speed, and local deployment**.

It uses **Entity-level Chunking** instead of traditional token-based chunking,
ensuring that the LLM receives complete and structured context.

## Highlights

- **KG dual-layer retrieval** (LightRAG-style): `local` + `global` + `mix` + `kg_only`.
- **KG extraction v2** pipeline: semantic chunking → alignment → dedup → quality filter.
- **Multi-source import**: TXT / Markdown / PDF / Word / Images (OCR/VLM) / Web pages.
- **PageRank-enhanced ranking** + community detection + shortest path.
- **REST API** (FastAPI) with streaming SSE.
- **Native Ollama** support — run fully offline.
- **Async API**: `acall_llm` / `acall_llm_stream`.
- **190+ unit tests**, CI/CD, Ruff, type hints.

## Quick Install

```bash
# PyPI 包即将发布；目前请使用源码安装
git clone https://github.com/JayZ/PocketGraphRAG.git
cd PocketGraphRAG
pip install -r requirements.txt
# 贡献者可装全部 extras：pip install -e ".[all]"
```

Then configure an LLM provider and launch:

```bash
python -m PocketGraphRAG.webapp   # Gradio Web UI on :7860
```

## Next Steps

- [Installation & Setup](getting-started.md)
- [Quick Start](quickstart.md)
- [Architecture](architecture.md)
- [Python API](python-api.md)
- [REST API](rest-api.md)
- [Evaluation Harness](evaluation.md)
