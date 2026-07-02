# REST API

PocketGraphRAG ships a FastAPI-based REST API server for programmatic access.

## Start

```bash
python -m PocketGraphRAG.api_server --host 0.0.0.0 --port 8000
# or with the [cli] extra:
pocketgraphrag-cli serve api --port 8000
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/qa` | Non-streaming Q&A |
| POST | `/api/qa/stream` | Streaming Q&A (SSE) |
| GET | `/api/graph/stats` | Graph statistics (entities, relations, triples) |
| GET | `/api/graph/entities` | List all entities |
| GET | `/api/graph/relations` | List all relations |
| GET | `/api/graph/entities/search?q=...` | Search entities by name |
| GET | `/api/graph/entity/{name}/detail` | Entity detail with neighbors |
| GET | `/api/graph/entity/{name}/subgraph` | Entity neighborhood subgraph |
| POST | `/api/graph/subgraph` | Subgraph for multiple seed entities |
| GET | `/api/graph/pagerank` | Entities ranked by PageRank importance |
| GET | `/api/graph/communities` | Community detection results |
| GET | `/api/graph/path?start=&end=` | Shortest path between two entities |
| GET | `/health` | Health check |

## Example

```bash
curl -X POST http://localhost:8000/api/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "盗梦空间讲了什么？", "search_mode": "mix"}'
```

Streaming (SSE):

```bash
curl -N -X POST http://localhost:8000/api/qa/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "三环唑用量是多少？"}'
```
