# Quick Start

## 1. Install

```bash
# PyPI 包即将发布；目前请使用源码安装
git clone https://github.com/JayZ/PocketGraphRAG.git
cd PocketGraphRAG
pip install -r requirements.txt
```

## 2. Configure

```bash
copy .env.example .env       # Windows
# cp .env.example .env      # Linux/Mac
# Edit .env and fill ONE provider key
```

Easiest local path (no API key):

```bash
ollama pull qwen2:7b
set OLLAMA_MODEL=qwen2:7b
set OLLAMA_API_BASE=http://localhost:11434/v1
```

## 3. Build & Launch

The repo includes a rice-disease demo dataset, so no data prep is needed.

```bash
python -m PocketGraphRAG.build_index
python -m PocketGraphRAG.webapp
```

Open <http://localhost:7860>.

## 4. Ask a Question

**Via Python API:**

```python
from PocketGraphRAG import PocketGraphRAG

rag = PocketGraphRAG(search_mode="mix", use_multihop=True, use_pagerank=True)
result = rag.answer("盗梦空间讲了什么？")
print(result["answer"])
print(f"Sources: {len(result['sources'])}")
print(f"KG entities matched: {result['pipeline_info']['kg_entities_matched']}")
```

**Streaming:**

```python
for chunk in rag.answer_stream("三环唑的用量是多少？"):
    if "chunk" in chunk:
        print(chunk["chunk"], end="", flush=True)
```

**Async (FastAPI / anyio contexts):**

```python
import asyncio
from PocketGraphRAG import acall_llm

async def main():
    answer = await acall_llm("你是知识问答助手", "盗梦空间讲了什么？")
    print(answer)

asyncio.run(main())
```

**Via CLI (one-shot):**

```bash
pocketgraphrag-cli qa "盗梦空间讲了什么？" --search-mode mix --stream
```

## 5. Bring Your Own Data

Use the Web UI's "Data Management" tab (upload → extract triples → build index),
or the CLI:

```bash
pocketgraphrag-cli extract -i your_document.txt -o my_triples.txt
set POCKET_DATA_PATH="my_triples.txt"
pocketgraphrag-cli build
pocketgraphrag-cli serve web
```
