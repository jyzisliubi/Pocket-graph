# Python API

## Initialization

```python
from PocketGraphRAG import PocketGraphRAG

rag = PocketGraphRAG(
    search_mode="mix",        # vector | local | global | mix | kg_only
    use_multihop=True,        # multi-hop query decomposition
    use_conversation=True,    # multi-turn memory + query rewriting
    use_pagerank=True,        # PageRank-weighted ranking
    pagerank_weight=0.3,
    fusion_strategy="rrf",    # or "weighted"
    rrf_k=60,
    top_k=5,
)
```

## One-shot Q&A

```python
result = rag.answer("盗梦空间讲了什么？")
print(result["answer"])
print(result["sources"])            # [{"entity", "score", "text"}, ...]
print(result["pipeline_info"])      # search_mode, kg_path, kg_entities_matched, ...
```

## Streaming

```python
for chunk in rag.answer_stream("三环唑的用量是多少？"):
    if "chunk" in chunk:
        print(chunk["chunk"], end="", flush=True)
```

The generator also yields `status`, `sources`, and `pipeline_info` chunks.

## Async API

For FastAPI / anyio contexts. Mirrors the sync API 1:1:

```python
import asyncio
from PocketGraphRAG import acall_llm, acall_llm_stream

async def ask():
    # non-streaming
    answer = await acall_llm("你是知识问答助手", "盗梦空间讲了什么？")
    print(answer)

    # streaming
    async for chunk in acall_llm_stream("你是知识问答助手", "三环唑用量？"):
        print(chunk, end="", flush=True)

asyncio.run(ask())
```

## Direct KG Extraction

```python
from PocketGraphRAG import extract_knowledge_graph

result = extract_knowledge_graph(text, min_confidence=0.6)
print(f"{len(result.triples)} triples, avg conf {result.avg_confidence:.3f}")
for t in result.triples:
    print(f"[{t.confidence:.2f}] {t.head} --[{t.relation}]--> {t.tail}")
```

## Multi-source Import

```python
from PocketGraphRAG import DataImporter

importer = DataImporter()
doc = importer.import_file("document.pdf")
# doc = importer.import_file("image.png", image_mode="ocr")
# doc = importer.import_url("https://example.com/article")
```

## Reset Conversation

```python
rag.reset_conversation()
```
