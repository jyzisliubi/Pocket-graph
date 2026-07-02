# Installation

## Requirements

- Python 3.9+ (3.8 works but CI tests 3.9–3.12)
- ~1 GB disk for the BGE embedding model (auto-downloaded on first run)

## Install

**From source (recommended):**

```bash
git clone https://github.com/JayZ/PocketGraphRAG.git
cd PocketGraphRAG
pip install -r requirements.txt      # 普通用户
# 或 pip install -e ".[dev]"          # 贡献者（含 pytest/ruff/pre-commit）
pre-commit install                    # 仅贡献者需要
```

**From PyPI (coming soon):**

```bash
pip install pocketgraphrag           # 即将发布 / coming soon
```

**Optional extras:**

| Extra | Installs | Use |
|-------|----------|-----|
| `[web]` | gradio, fastapi, uvicorn, pydantic | Gradio Web UI + REST API |
| `[docs]` | python-docx, pdfplumber, PyPDF2, beautifulsoup4, lxml, Pillow | Multi-format document import |
| `[playwright]` | playwright | Dynamic web page scraping |
| `[cli]` | typer, uvicorn | Modern subcommand CLI (`pocketgraphrag-cli`) |
| `[eval]` | ragas | RAGAS-based evaluation |
| `[all]` | all of the above + dev | Full development environment |

```bash
pip install -e ".[all]"
```

**Docker:**

```bash
docker-compose up -d
# or
docker build -t pocketgraphrag .
docker run -p 7860:7860 -v $(pwd)/index:/app/index -v $(pwd)/models:/app/models pocketgraphrag
```

## Configure an LLM

Copy `.env.example` to `.env` and fill in one provider (or set env vars):

| Provider | Env var | Notes |
|----------|---------|-------|
| **Ollama (local)** | `OLLAMA_MODEL` | Run fully offline. `ollama pull qwen2:7b` |
| SiliconFlow | `SILICONFLOW_API_KEY` | Free Qwen models |
| DashScope | `DASHSCOPE_API_KEY` | Free tier, supports VLM |
| DeepSeek | `DEEPSEEK_API_KEY` | Strong reasoning |
| OpenAI-compatible | `OPENAI_API_KEY` + `OPENAI_API_BASE` | Any compatible endpoint |

The unified LLM layer tries providers in priority order and auto-falls-back.

## Build the Index

```bash
python -m PocketGraphRAG.build_index
```

This downloads the BGE embedding model and builds the FAISS + entity + relation indexes.
The repo ships with a rice-disease demo dataset so you can skip data preparation.

## Launch

```bash
python -m PocketGraphRAG.webapp          # Gradio Web UI on :7860
python -m PocketGraphRAG.app             # CLI interactive mode
python -m PocketGraphRAG.api_server      # FastAPI REST API on :8000
pocketgraphrag-cli serve web            # (with [cli] extra)
```
