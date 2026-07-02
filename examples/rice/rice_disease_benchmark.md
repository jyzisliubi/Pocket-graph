# Domain Benchmark: rice_disease_benchmark_v1

> Moved out of the main README. For the public HotpotQA head-to-head vs
> LightRAG, see the [README first screen](../../README.md#-hotpotqa-vs-lightrag).
> For the eval harness, see [docs/evaluation.md](../../docs/evaluation.md).

## Setup

Real numbers from the built-in `eval_harness` on the bundled 20-question domain
benchmark (`PocketGraphRAG/benchmark/rice_disease_v1.json`, v1.1).

- **20 questions** across 6 types: `single_fact`, `reverse_link`, `comparison`,
  `multi_hop_kg`, `list_type`, `general_knowledge`.
- LLM = Ollama `qwen2.5:7b`
- Reranker = `BAAI/bge-reranker-v2-m3` (local cache)
- `top_k=5`

Reproduce with:

```bash
python -m PocketGraphRAG.eval_harness --search-mode mix --top-k 5 --no-generation
```

## Results (N=20, top_k=5)

| Search Mode | Hit Rate ↑ | MRR ↑ | Entity Coverage ↑ | Relation Coverage ↑ |
|-------------|:----------:|:------:|:------------------:|:--------------------:|
| `vector` (baseline) | 0.60 | 0.36 | 0.00 | 0.00 |
| `kg_only` (pure KG) | **0.95** | **0.83** | **1.00** | **0.67** |
| `mix` (vector + KG, weighted RRF) | **0.95** | 0.76 | **1.00** | **0.67** |

## Key Findings

- **KG retrieval adds massive value.** Pure KG (`kg_only`) lifts Hit Rate from
  0.60 → 0.95 (+58%) and MRR from 0.36 → 0.83 (+130%), reaching 100% entity
  coverage that pure vector retrieval cannot reach (0.00 → 1.00), because
  vector chunks are entity-scoped and the expected entities often don't appear
  as chunk labels.
- **Three-level scoring + PPR top-10 boost** (v0.3.0) drove Hit Rate from
  0.70 → 0.95 and MRR from 0.56 → 0.83 on `kg_only` mode. Seed entities get
  2.0× score, relation-value entities 1.5×, expanded entities 1.0×; top-10
  PersonalizedPageRank entities get a 1.3× boost.
- **Schema normalization massively improved Relation Coverage** (v0.3.1):
  normalized 1275 fragmented relation names into 292 standard relations (77%
  reduction), combined with bidirectional normalization matching in
  `eval_harness`, lifted Relation Coverage from 0.18 → 0.67 (+272%).
- **Weighted RRF fusion** with `VECTOR_WEIGHT=0.3` (KG-biased) keeps `mix` Hit
  Rate at 0.95 while MRR drops slightly to 0.76 (vector noise drags rank), so
  `kg_only` is the recommended mode for domain KG-RAG.
- **CrossEncoder reranker hurts domain KG-RAG.** The general-purpose reranker
  (bge-reranker-v2-m3) disrupts KG's entity-level ranking — MRR dropped from
  0.48 to 0.35, with 5× latency. Entity-level KG ranking is already precise
  enough; reranker is disabled by default.

> These are self-reported numbers on a small in-domain benchmark, not a
> third-party evaluation.

## Chinese-Scenario Comparison (rice-disease domain, qualitative)

> Qualitative observations on the built-in `benchmark/rice_disease_v1.json`
> (20 questions), not a strict third-party eval.

| Dimension | PocketGraphRAG | Microsoft GraphRAG | LightRAG |
|-----------|---------------|-------------------|----------|
| **Chinese entity matching** | BGE-zh vectors + rule alignment | English prompts, needs tuning | English prompts, needs tuning |
| **Indexing token cost** | Low (single-pass) | High (community reports) | Medium |
| **Vertical-domain ready** | Built-in rice/movie/cat examples + Chinese prompts | Generic, self-tune | Generic, self-tune |
| **Offline run** | Native Ollama | Needs setup | Supported |
| **Web UI** | Built-in Gradio (Q&A + graph + data mgmt) | CLI only | Separate webui |
