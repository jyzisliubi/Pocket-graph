# FAQ — Frequently Asked Questions

> Moved out of the main README to keep the first screen focused. Core FAQ stays
> in [README.md](../README.md#faq); this file holds the longer answers.

## Setup & Requirements

### Do I need a GPU to run this?

No. The BGE embedding model (`BAAI/bge-small-zh-v1.5`) runs fine on CPU. For LLM
generation, Ollama with a small model (like `qwen2:7b`) works on most modern
CPUs. If you do have a GPU it speeds up the first index build, but it is not
required for day-to-day querying.

### What Python versions are supported?

Python 3.9+ (3.8 mostly works, but CI tests 3.9–3.13). See
[getting-started.md](getting-started.md) for install details.

### How do I add my own data?

Two paths:

- **Web UI**: open the "Data Management" tab, upload a `.txt` / `.md` / `.pdf` /
  `.docx` / image file, click extract, then build index. Subsequent uploads go
  through incremental indexing (only affected entities are re-encoded).
- **CLI**:
  ```bash
  python -m PocketGraphRAG.kg_extractor --input your_document.txt --output my_triples.txt
  set POCKET_DATA_PATH="my_triples.txt"
  python -m PocketGraphRAG.build_index
  python -m PocketGraphRAG.webapp
  ```

## Comparison with Other Frameworks

### How is this different from LightRAG?

PocketGraphRAG differs on four points that matter for vertical-domain use:

1. **Entity-level chunking** — knowledge is grouped by entity, not by token
   count, so the LLM gets coherent context.
2. **Interactive graph visualization** — ECharts force-directed graph, search,
   and 1-hop neighborhood exploration in the Web UI.
3. **Web-based data management** — upload → extract → build → switch dataset,
   all from the browser; LightRAG is CLI-first.
4. **Chinese optimization** — BGE-zh embedding + Chinese prompts by default.
5. **Zero-LLM query** — at query time PocketGraphRAG uses pure embedding +
   graph traversal (no LLM call); LightRAG needs an LLM for keyword extraction.
6. **Deterministic retrieval** — a `cid` tie-breaker guarantees reproducible
   MRR; LightRAG / nano-graphrag do not.

See the HotpotQA head-to-head in the
[README](../README.md#-hotpotqa-vs-lightrag).

### How is this different from Microsoft GraphRAG?

Microsoft GraphRAG requires Neo4j and builds expensive community reports
(high indexing token cost). PocketGraphRAG ships with FAISS only (no external
DB), single-pass indexing, and a built-in Gradio Web UI for the whole loop.
Microsoft GraphRAG is CLI-only and English-first by default.

### Can I use this for production?

It is currently **alpha**. The core pipeline (extraction → indexing → retrieval
→ generation) is stable and covered by 216+ unit tests, but we recommend
thorough testing on your own data before production use. Known rough edges:
the reranker can hurt domain KG-RAG (disabled by default), and very large
corpora (>100k triples) have not been stress-tested — the in-memory graph dict
+ JSON dump will want a pluggable backend (Neo4j adapter is on the roadmap).

## Retrieval & Quality

### Which search mode should I use?

| Mode | Best for |
|------|----------|
| `mix` (default) | Best overall quality — vector + local + global |
| `kg_only` | Domain KG-RAG baseline; highest MRR on entity-centric queries |
| `local` | Entity-neighborhood queries |
| `global` | Relation-centric queries |
| `vector` | Pure vector baseline / general queries |

For vertical-domain KGs, `kg_only` often gives the highest MRR because
entity-level KG ranking is already precise and vector noise can drag the rank.

### Why is the reranker disabled by default?

The general-purpose `bge-reranker-v2-m3` disrupts the KG's entity-level ranking
in domain tests — MRR dropped from 0.48 → 0.35 with 5× latency. Entity-level KG
ranking is already precise enough. You can re-enable it, but it is off by
default.

### What is Multi-Model KG Fusion and why does it help?

Different LLMs have different extraction blind spots. A smaller/faster model
(e.g. `qwen-flash`) extracts more triples but with more noise; a larger model
(e.g. `qwen-max`) extracts fewer but more conservative triples. Their **union**
recovers entities that either model alone misses — like an ensemble of
extractors. On HotpotQA this lifted Hit Rate 0.80 → 0.86. This is unique to
PocketGraphRAG; LightRAG and nano-graphrag extract KG with a single model only.

### How does incremental indexing work?

See [README → Incremental Indexing](../README.md#incremental-indexing). In
short: only entities that gained new triples (and brand-new entities) get
re-encoded; identical triples are skipped via a persisted manifest hash set.
Cost is proportional to the delta, not the corpus size. Legacy indexes are
auto-migrated on first incremental call.

## Data Import

### Which file formats are supported?

TXT, Markdown, PDF (text + scanned), Word (`.docx`), images (OCR/VLM), static
web pages, and dynamic web pages (Playwright). See
[data-import.md](data-import.md) for the full format/quality matrix and
per-source extraction examples.

### Can I run fully offline?

Yes. Install Ollama, pull a model (`ollama pull qwen2:7b`), and set
`OLLAMA_MODEL` / `OLLAMA_API_BASE`. The entire pipeline — embedding, KG
retrieval, and generation — runs without any external API. The BGE embedding
model is cached locally after the first run.

## Troubleshooting

### The Web UI starts but answers are empty / no sources.

This usually means the LLM provider is not configured. The Web UI still starts
so you can verify retrieval, sources, and graph state before wiring up
generation. Check `.env` for a valid `DASHSCOPE_API_KEY` /
`SILICONFLOW_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`, or run Ollama.

### `pocketgraphrag` command not found.

The modern Typer CLI requires the `[cli]` extra:

```bash
pip install "pocketgraphrag[cli]"
```

Or use the always-available legacy entry: `python -m PocketGraphRAG.app`.

### First index build is slow / downloads a large model.

The first run downloads the BGE embedding model (~1 GB). Subsequent builds reuse
the cached model. If you are behind a slow connection, pre-download the model to
`models/BAAI/bge-small-zh-v1.5/`.
