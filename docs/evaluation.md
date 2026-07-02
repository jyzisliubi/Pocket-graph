# Evaluation Harness

PocketGraphRAG ships a built-in benchmark and standardized evaluation metrics,
inspired by the MultiHop-RAG / LightRAG `reproduce/` evaluation approach.

## Built-in Benchmark

Located at `PocketGraphRAG/benchmark/rice_disease_v1.json` (v1.1):

- **20 questions** across 6 types: `single_fact`, `reverse_link`, `comparison`,
  `multi_hop_kg`, `list_type`, `general_knowledge`.
- Each question annotates `expected_entities`, `expected_relations`,
  `expected_answer_keywords`, and `ground_truth` (reference answer for RAGAS
  `context_recall`).
- Difficulty levels: `easy` / `medium` / `hard`.

## Metrics

### Retrieval layer (no LLM required)

| Metric | Definition |
|--------|------------|
| `hit_rate` | Fraction of questions where at least one expected entity appears in retrieved sources. |
| `mrr` | Mean Reciprocal Rank of the first expected entity in retrieved sources. |
| `entity_coverage` | Fraction of expected entities hit by the KG seed match. |
| `relation_coverage` | Fraction of expected relations hit by the KG relation match. |

### Generation layer (LLM required)

| Metric | Definition |
|--------|------------|
| `answer_keyword_hit` | Fraction of `expected_answer_keywords` present in the generated answer. |

### RAGAS (optional, 4 standard metrics)

When `ragas` is installed and an LLM is configured, the harness computes the
four canonical RAGAS metrics:

| Metric | Definition |
|--------|------------|
| `faithfulness` | Answer is faithful to the retrieved context (no hallucination). |
| `answer_relevancy` | Answer addresses the question (reverse-generated question similarity). |
| `context_precision` | Top-k retrieved context is relevant to the answer. |
| `context_recall` | Answer covers the key info in `ground_truth` (requires reference answer). |

```bash
pip install -e ".[eval]"
python -m PocketGraphRAG.eval_harness --ragas --search-mode mix
```

### Using local Ollama as RAGAS evaluator

RAGAS needs an LLM to grade answers. PocketGraphRAG reuses its own `call_llm`
layer (Ollama has highest priority), so you can use a local Ollama model as the
RAGAS evaluator — no OpenAI key required:

```bash
# Make sure Ollama is running and the model is pulled
ollama pull qwen2.5:7b
ollama serve   # if not already running

# Run RAGAS evaluation with the local model
python -m PocketGraphRAG.eval_harness --ragas \
    --ollama-model qwen2.5:7b \
    --search-mode mix --top-k 5
```

The `--ollama-model` flag patches both the environment variable and the
in-memory `PocketGraphRAG.llm.OLLAMA_MODEL` constant so that the LangChain
LLM wrapper used by RAGAS routes to your local Ollama.

## CLI Usage

```bash
# Retrieval + generation metrics on default dataset
python -m PocketGraphRAG.eval_harness --search-mode mix --top-k 5

# Retrieval only (no LLM calls)
python -m PocketGraphRAG.eval_harness --no-generation

# With RAGAS (4 metrics, requires ground_truth in benchmark)
python -m PocketGraphRAG.eval_harness --ragas

# With RAGAS + explicit Ollama model
python -m PocketGraphRAG.eval_harness --ragas --ollama-model qwen2.5:7b
```

## Python API

```python
from PocketGraphRAG import PocketGraphRAG, load_benchmark, run_evaluation

rag = PocketGraphRAG(search_mode="mix", use_pagerank=True)
dataset = load_benchmark()
report = run_evaluation(rag, dataset, top_k=5, run_generation=True, run_ragas=True)
print(report["summary"])
# RAGAS results (None if ragas not installed / LLM unavailable)
print(report["ragas"])
for q in report["per_question"]:
    print(q.id, q.first_expected_rank, q.answer_keyword_hit)
```

## Reproduced Numbers

Reference run on the built-in `rice_disease_v1` benchmark,
`search_mode=mix`, `top_k=5`, evaluator LLM = Ollama `qwen2.5:7b`:

| Metric | Value | Notes |
|--------|-------|-------|
| `hit_rate` | 0.5500 | Retrieval (20 questions) |
| `mrr` | 0.4792 | Retrieval (20 questions) |
| `entity_coverage` | 0.5714 | Retrieval (20 questions) |
| `relation_coverage` | 0.1818 | Retrieval (20 questions) |
| `answer_keyword_hit` | 0.6210 | Generation (20 questions) |
| `faithfulness` | 0.6131 | RAGAS (5-question subset) |
| `answer_relevancy` | _n/a_ | RAGAS (skipped: Ollama n=3 too slow) |
| `context_precision` | 0.4667 | RAGAS (5-question subset) |
| `context_recall` | 0.4667 | RAGAS (5-question subset) |

> **Note on RAGAS subset**: Retrieval/generation metrics are computed over all
> 20 questions. RAGAS scores are from the first 5 questions because the local
> `qwen2.5:7b` evaluator is slow (~100s/question for 3 metrics).
> `answer_relevancy` is skipped locally (it needs `n=3` reverse question
> generation, which Ollama handles poorly); set `RAGAS_SKIP_ANSWER_RELEVANCY=0`
> and use a faster/OpenAI LLM to enable it. Per-question scores are saved to
> `eval_result.json`.

Reproduce with:

```bash
# Retrieval + generation (20 questions, no LLM-dependent RAGAS)
python -m PocketGraphRAG.eval_harness --search-mode mix --top-k 5

# RAGAS (uses Ollama; subset script writes eval_result.json)
python -m PocketGraphRAG.eval_harness --ragas --ollama-model qwen2.5:7b \
    --search-mode mix --top-k 5
```

## Ablation Study

For a different purpose — measuring the contribution of each feature — run:

```bash
python -m PocketGraphRAG.evaluate
```

This compares 5 baselines: Basic RAG → Pure KG → KG Local → KG Mix → Full Mode
(+ Multi-hop + Conversation memory).
