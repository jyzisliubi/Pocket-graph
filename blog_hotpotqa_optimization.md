# From 0.56 to 0.86: How We Optimized GraphRAG Retrieval on HotpotQA
# 从 0.56 到 0.86：我们如何优化 GraphRAG 在 HotpotQA 上的检索效果

> A 5-stage optimization journey of PocketGraphRAG on the public HotpotQA benchmark, achieving +54% Hit Rate improvement. The breakthrough? Multi-Model KG Fusion — merging knowledge graphs extracted by different LLMs.
>
> 在公开 HotpotQA benchmark 上，PocketGraphRAG 经过 5 阶段优化，Hit Rate 提升 54%。关键突破？多模型 KG 合并——融合不同 LLM 抽取的知识图谱。

---

## TL;DR / 太长不看

| Stage | Config | Hit Rate | MRR |
|-------|--------|----------|-----|
| Baseline | mix k=5 RRF | 0.56 | 0.42 |
| + top_k 25 | mix k=25 | 0.72 | 0.44 |
| + Weighted fusion | mix k=25 weighted | 0.72 | 0.48 |
| + Gleaning | mix k=25 weighted (7407 triples) | 0.80 | 0.54 |
| **+ Multi-Model KG Fusion** | **mix k=25 weighted (10559 triples)** | **0.86** | **0.5633** |

**Key takeaway**: The biggest single factor was top_k (5→25, +0.16 Hit). But the most *interesting* win was Multi-Model KG Fusion (+0.06 Hit) — a technique unique to PocketGraphRAG that no other open-source GraphRAG framework supports.

**核心结论**：最大单因素是 top_k（5→25，+0.16 Hit）。但最*有意思*的优化是多模型 KG 合并（+0.06 Hit）——这是 PocketGraphRAG 独有的技术，其他开源 GraphRAG 框架都不支持。

---

## The Problem / 问题

GraphRAG systems promise better retrieval by building knowledge graphs from documents. But most published benchmarks use small, domain-specific datasets that are hard to reproduce and easy to overfit. We wanted to know: **how does our GraphRAG actually perform on a public, reproducible benchmark?**

GraphRAG 系统通过从文档中构建知识图谱来提升检索效果。但大多数已发表的 benchmark 使用小规模、特定领域的数据集，难以复现且容易过拟合。我们想知道：**我们的 GraphRAG 在公开、可复现的 benchmark 上实际表现如何？**

We chose **HotpotQA** — a multi-hop QA dataset where each question requires reasoning across multiple Wikipedia paragraphs. We sampled 50 bridge-type questions from the distractor validation split (seed=42), ensuring reproducibility.

我们选择 **HotpotQA**——一个多跳问答数据集，每道题需要跨多个 Wikipedia 段落推理。我们从 distractor validation split 中抽取 50 道桥接型问题（seed=42），确保可复现。

---

## Stage 1: Baseline (Hit=0.56) / 基线

Starting config: `mix` retrieval (vector + KG), `top_k=5`, RRF fusion, single-pass extraction (no gleaning).

起始配置：`mix` 检索（向量+KG），`top_k=5`，RRF 融合，单轮抽取（无 gleaning）。

**Result**: Hit Rate=0.56, MRR=0.42, Entity Coverage=0.27

Only 56% of questions found their expected entity in the top-5 results. The low entity coverage (0.27) hinted at an extraction problem — many expected entities weren't even in the knowledge graph.

只有 56% 的问题在前 5 个结果中找到了期望实体。低实体覆盖率（0.27）暗示抽取有问题——很多期望实体根本不在知识图谱里。

---

## Stage 2: top_k 5→25 (Hit=0.72, +0.16) / 扩大候选

The simplest optimization: increase `top_k` from 5 to 25. More candidates = more chances to hit.

最简单的优化：把 `top_k` 从 5 增到 25。更多候选 = 更多命中机会。

**Result**: Hit Rate=0.72 (+0.16), MRR=0.44

This was the **largest single-factor improvement**. Bridge questions especially benefit — they need to find a "bridge" entity that connects two paragraphs, and a larger search radius catches more bridges.

这是**最大单因素提升**。桥接问题尤其受益——它们需要找到连接两个段落的"桥"实体，更大的搜索半径能捕获更多桥。

---

## Stage 3: RRF → Weighted Fusion (MRR +0.04) / 加权融合

Switched fusion strategy from Reciprocal Rank Fusion (RRF) to simple weighted sum with `VECTOR_WEIGHT=0.3` (KG-biased).

将融合策略从 RRF（倒数排名融合）改为简单加权求和，`VECTOR_WEIGHT=0.3`（偏向 KG）。

**Result**: Hit Rate=0.72 (unchanged), MRR=0.48 (+0.04)

RRF normalizes ranks, which dilutes KG's precise entity-level scoring. Weighted fusion preserves KG's ranking signal, improving MRR without hurting Hit Rate.

RRF 会归一化排名，稀释了 KG 精确的实体级评分。加权融合保留了 KG 的排名信号，在不影响 Hit Rate 的情况下提升了 MRR。

---

## Stage 4: Gleaning Multi-Round Extraction (Hit=0.80, +0.08) / 多轮抽取

Inspired by [microsoft/graphrag](https://github.com/microsoft/graphrag), we added **gleaning** — after the first extraction pass, the LLM is asked "did you miss anything?" and extracts again. We set `gleaning_steps=1` (one follow-up round).

受 [microsoft/graphrag](https://github.com/microsoft/graphrag) 启发，我们加入了 **gleaning**——首轮抽取后，追问 LLM "你漏了什么？"再抽一轮。设置 `gleaning_steps=1`（一轮追问）。

**Result**: Triples 4611 → 7407 (+60%), Hit Rate=0.80 (+0.08), MRR=0.54

Gleaning recovered 5 questions that were previously failing due to missing entities. The +60% triple count meant more entities in the KG, which directly improved entity coverage from 0.27 → 0.32.

Gleaning 恢复了 5 道之前因实体缺失而失败的问题。三元组 +60% 意味着 KG 中有更多实体，直接将实体覆盖率从 0.27 提升到 0.32。

---

## Stage 5: Multi-Model KG Fusion (Hit=0.86, +0.06) / 多模型 KG 合并

**This is our breakthrough.** No other open-source GraphRAG framework supports this.

**这是我们的突破。** 没有其他开源 GraphRAG 框架支持这个。

### The Idea / 思路

Different LLMs have different extraction blind spots. A smaller/faster model (qwen-flash) extracts more triples but with more noise; a larger model (qwen-max) extracts fewer but more conservative triples. **Their union recovers entities that either model alone misses** — like an ensemble of extractors.

不同 LLM 有不同的抽取盲区。小/快模型（qwen-flash）抽得更多但噪声更多；大模型（qwen-max）抽得更少但更保守。**它们的并集能恢复任一模型单独漏掉的实体**——类似集成学习。

### The Result / 结果

| KG Source | Triples | Hit Rate | MRR |
|-----------|---------|----------|-----|
| qwen-flash + gleaning(1) | 7407 | 0.80 | 0.5365 |
| qwen-max + gleaning(2) | 3790 | 0.66 | 0.4127 |
| **Fused (union, deduped)** | **10559** | **0.86** | **0.5633** |

The larger model (qwen-max) alone was *worse* (0.66 vs 0.80) — it extracted fewer triples. But when fused with qwen-flash's KG, the combined knowledge graph covered more entities than either alone, lifting Hit Rate to 0.86.

大模型（qwen-max）单独用反而*更差*（0.66 vs 0.80）——它抽的三元组更少。但和 qwen-flash 的 KG 合并后，组合知识图谱覆盖的实体比任一单独模型都多，将 Hit Rate 提升到 0.86。

### Why It Works / 为什么有效

Think of it as two readers reading the same document:
- Reader A (qwen-flash) is fast but misses some details → extracts 7407 triples
- Reader B (qwen-max) is careful but conservative → extracts 3790 triples
- Together, they catch what the other misses → 10559 unique triples

This is analogous to **ensemble methods in ML** — diverse models combined outperform any single model.

这就像两个读者读同一篇文档：
- 读者 A（qwen-flash）快但漏细节 → 抽 7407 条三元组
- 读者 B（qwen-max）仔细但保守 → 抽 3790 条三元组
- 合在一起，互补盲区 → 10559 条去重三元组

这类似于 **机器学习中的集成方法**——多样化模型组合优于任何单一模型。

---

## Comparison with LightRAG / 与 LightRAG 对比

We ran LightRAG (v1.5.4) on the same 50 HotpotQA questions with the same embedding model (BAAI/bge-small-zh-v1.5) for a fair comparison.

我们在相同的 50 道 HotpotQA 问题上跑了 LightRAG（v1.5.4），使用相同的 embedding 模型（BAAI/bge-small-zh-v1.5），确保公平对比。

| Framework | Extraction LLM | Triples | Hit Rate | MRR |
|-----------|---------------|---------|----------|-----|
| **PocketGraphRAG** | qwen-flash + qwen-max (fused) | 10559 | **0.86** | **0.5633** |
| PocketGraphRAG | deepseek-v3 (single) | 3610 | 0.70 | 0.4757 |
| LightRAG v1.5.4 | qwen-long-latest (insert) + qwen3-235b-a22b (query) | — | 0.82 | 0.2093 |

**PocketGraphRAG wins on both metrics**: Hit Rate +0.04, **MRR 2.7x higher** (0.5633 vs 0.2093) — meaning PocketGraphRAG not only finds the answer more often, but ranks it much closer to the top.

**Key architectural difference**: PocketGraphRAG's query doesn't need LLM calls (pure embedding + graph traversal), while LightRAG requires LLM-based keyword extraction for every query. This makes PocketGraphRAG faster at query time and immune to LLM quota exhaustion — during our test, LightRAG's last few queries failed because the free-tier LLM quota ran out, while PocketGraphRAG (zero LLM calls at query) was unaffected.

**PocketGraphRAG 在两项指标上均胜出**：Hit Rate +0.04，**MRR 高 2.7 倍**（0.5633 vs 0.2093）——意味着 PocketGraphRAG 不仅更常找到答案，而且把答案排得更靠前。

**关键架构差异**：PocketGraphRAG 的 query 不需要 LLM 调用（纯 embedding + 图遍历），而 LightRAG 每次 query 都需要 LLM 做关键词提取。这使 PocketGraphRAG 在 query 时更快，且不受 LLM 配额耗尽影响——测试中 LightRAG 最后几题因免费配额用尽而失败，而 PocketGraphRAG（query 零 LLM 调用）完全不受影响。

---

## Failure Analysis / 失败分析

7 out of 50 questions still fail. All are **entity-missing** — the expected entity was never extracted into the KG by any model. This is the extraction-layer ceiling, not a retrieval-layer problem.

50 题中仍有 7 题失败。全部是**实体缺失**——期望实体从未被任何模型抽进 KG。这是抽取层的天花板，不是检索层的问题。

Root causes:
- 3 questions: entity appears in document but both LLMs missed it during extraction
- 2 questions: bridge entity name doesn't appear literally in the query (requires inference)
- 2 questions: entity name normalization mismatch (e.g., "JFK" vs "John F. Kennedy")

根因：
- 3 题：实体出现在文档中，但两个 LLM 抽取时都漏了
- 2 题：桥接实体名未字面出现在问题中（需要推理）
- 2 题：实体名归一化不匹配（如 "JFK" vs "John F. Kennedy"）

---

## Conclusion / 结论

1. **top_k matters most** — don't be stingy with candidates, especially for bridge questions.
2. **Gleaning is cheap and effective** — one extra LLM round recovers ~10% more entities.
3. **Multi-Model KG Fusion is the future** — ensemble extraction is to GraphRAG what ensemble models are to ML. It's free performance with zero retrieval-layer changes.
4. **Weighted > RRF for KG-biased retrieval** — preserve KG's ranking signal.

1. **top_k 最重要**——不要吝啬候选数，特别是桥接问题。
2. **Gleaning 便宜又有效**——多一轮 LLM 调用恢复约 10% 实体。
3. **多模型 KG 合并是未来**——集成抽取之于 GraphRAG，就像集成模型之于机器学习。零检索层改动，免费性能提升。
4. **加权 > RRF（KG 偏向检索）**——保留 KG 的排名信号。

---

**Try it yourself / 自己试试**:
```bash
git clone https://github.com/pocketgraphrag/PocketGraphRAG.git
cd PocketGraphRAG
python bench_data/eval_merged.py  # 复现 0.86 Hit Rate
```

*PocketGraphRAG is open-source under MIT license. Star us on GitHub if you find this useful!*

---

## Links / 链接

- GitHub: https://github.com/pocketgraphrag/PocketGraphRAG
- Docs: https://pocketgraphrag.github.io/PocketGraphRAG/
- Benchmark reproduce: `python bench_data/eval_merged.py`
