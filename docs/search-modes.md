# Search Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `vector` | Pure vector similarity search | General queries |
| `local` | Entity embedding match + BFS neighborhood expansion | Entity-centric queries |
| `global` | Relation embedding match + entity collection | Relation-centric queries |
| `mix` | Vector + Local + Global combined | Best overall quality |
| `kg_only` | Pure KG (Local + Global, no vector) | KG baseline evaluation |

## Configuration

| Env var | Description | Default |
|---------|-------------|---------|
| `POCKET_SEARCH_MODE` | Default search mode | `vector` |
| `ENTITY_SIMILARITY_THRESHOLD` | Entity match threshold | `0.5` |
| `KG_SEARCH_HOPS` | BFS expansion depth | `2` |
| `TOP_K` | Number of retrieved results | `5` |
| `POCKET_FUSION_STRATEGY` | `rrf` or `weighted` | `rrf` |
| `POCKET_RRF_K` | RRF k parameter | `60` |
| `POCKET_PAGERANK_WEIGHT` | PageRank weight in ranking | `0.3` |
| `POCKET_REVERSE_LINK_RELATIONS` | Reverse-link relations (comma-separated) | auto-detected |
| `POCKET_AUTO_REVERSE_LINK` | Auto-detect reverse-link relations | `true` |

## How Each Mode Works

### `local`

1. Match the query against entity embeddings → seed entities.
2. BFS-expand `KG_SEARCH_HOPS` hops from seeds → neighborhood.
3. Retrieve chunks whose entity is in the seed ∪ neighborhood set.
4. PageRank-weighted ranking + fusion with vector results.

### `global`

1. Match the query against **relation** embeddings → matched relations.
2. Collect all entities connected by those relations.
3. Retrieve their chunks.

### `mix`

Combines vector + local + global, deduplicates, and applies the configured
fusion strategy (RRF by default).

### `kg_only`

Same as `mix` but skips vector search entirely — a pure KG baseline useful for
ablation studies.
