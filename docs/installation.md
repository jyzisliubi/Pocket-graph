# Installation

PocketGraphRAG runs on Python 3.9+ and works fully offline with Ollama.

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9 – 3.12 | 3.13 not yet tested in CI |
| Disk | ~1.5 GB | Embedding model (~1 GB) + index + deps |
| RAM | 2 GB minimum | 4 GB+ recommended for large KGs |
| OS | Linux / macOS / Windows | Windows tested on PowerShell |

No GPU required — embedding runs on CPU by default. GPU acceleration is
optional (set `POCKET_DEVICE=cuda` if you have a CUDA build of PyTorch).

## 1. Install PocketGraphRAG

### Option A — From source (recommended, latest)

```bash
git clone https://github.com/jyzisliubi/Pocket-graph.git
cd Pocket-graph
pip install -e ".[web,docs]"     # Web UI + docs site
# Contributors:
pip install -e ".[web,docs,dev]"
pre-commit install
```

### Option B — From PyPI

```bash
pip install "pocketgraphrag[web]"          # Web UI + REST API
pip install "pocketgraphrag[web,cli]"      # + pocketgraphrag CLI
pip install "pocketgraphrag[all]"          # everything (incl. Neo4j, Redis)
```

### Option C — Docker (no Python needed)

```bash
docker-compose up -d                      # http://localhost:8000
docker-compose --profile cache up -d      # + Redis LLM Cache
```

See [Deployment](deployment.md) for full Docker / K8s instructions.

## 2. Optional extras

| Extra | Installs | When to use |
|-------|----------|-------------|
| `[web]` | fastapi, uvicorn, pydantic | REST API + Web UI |
| `[cli]` | typer, rich | `pocketgraphrag` command-line |
| `[docs]` | mkdocs, mkdocs-material | Build this docs site |
| `[dev]` | pytest, ruff, mypy, pre-commit | Contributors |
| `[neo4j]` | neo4j-driver | Neo4j graph backend |
| `[redis]` | redis | Shared LLM cache across replicas |
| `[all]` | everything above | One-shot install |

## 3. Configure an LLM provider

Pick **one** provider. All are optional for pure-retrieval modes (`vector`,
`local`, `global`, `mix`, `kg_only`) — those need an LLM only at index time.

### Ollama (fully offline, recommended)

```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:7b
```

```bash
# .env
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_API_BASE=http://localhost:11434/v1
```

### DashScope (Qwen API, free tier available)

```bash
# .env
DASHSCOPE_API_KEY=sk-xxx
DASHSCOPE_MODEL=qwen-flash          # free; qwen-max for higher quality
```

### SiliconFlow

```bash
# .env
SILICONFLOW_API_KEY=sk-xxx
SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### OpenAI-compatible

Any OpenAI-compatible endpoint works:

```bash
# .env
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1   # or your proxy
OPENAI_MODEL=gpt-4o-mini
```

## 4. Configure the embedding model

| Alias | Full name | Languages | Size |
|-------|-----------|-----------|------|
| `bge-small-zh` (default) | `BAAI/bge-small-zh-v1.5` | Chinese | ~100 MB |
| `bge-m3` | `BAAI/bge-m3` | Multilingual | ~560 MB |
| `bge-large-zh` | `BAAI/bge-large-zh-v1.5` | Chinese | ~1.3 GB |

```bash
# .env
POCKET_EMBEDDING_MODEL=bge-small-zh      # alias or full HF id
# For English / multilingual:
# POCKET_EMBEDDING_MODEL=bge-m3
```

Short aliases are expanded automatically. The local HF cache path is
preferred over a fresh download when available.

## 5. Verify the install

```bash
# Build the demo movie KG index
pocketgraphrag build

# Ask a question (zero-LLM retrieval path)
pocketgraphrag ask "霸王别姬的导演是谁"

# Launch the Web UI
pocketgraphrag webui                      # http://localhost:8000
```

## 6. (Optional) Configure v0.3.7 features

```bash
# .env
POCKET_WORKSPACE=default                  # multi-tenant isolation
POCKET_LLM_CACHE=1                        # enable LLM cache
POCKET_LLM_CACHE_BACKEND=memory           # or redis
POCKET_LLM_CACHE_REDIS_URL=redis://localhost:6379/0
POCKET_API_KEYS=key1,key2                 # API key auth
POCKET_API_AUTH_ENABLED=1
```

## Troubleshooting

### `ModuleNotFoundError: faiss`

Install the CPU build: `pip install faiss-cpu`. For GPU:
`pip install faiss-gpu` (Linux only).

### Embedding model download is slow

Set `HF_ENDPOINT=https://hf-mirror.com` to use the China mirror, or pre-download
the model and point `TRANSFORMERS_CACHE` at it.

### `RuntimeError: embedding dimension mismatch`

You switched embedding models against an existing index. Either delete
`index/` and rebuild, or use a different `POCKET_WORKSPACE` for the new model.

### Windows: long path errors

Enable long paths:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## Next steps

- [Quick Start](quickstart.md) — build your first KG and ask a question
- [Architecture](architecture.md) — how dual-layer retrieval works
- [Search Modes](search-modes.md) — pick the right mode per query
- [Deployment](deployment.md) — production Docker / K8s
