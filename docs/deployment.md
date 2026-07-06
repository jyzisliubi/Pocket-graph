# Deployment

PocketGraphRAG ships with first-class support for Docker, Kubernetes and Helm.
This guide covers production-grade deployments. For the raw manifests, see
[`deploy/`](https://github.com/jyzisliubi/Pocket-graph/tree/main/deploy).

## Choose a deployment path

| Path | Best for | State | Effort |
|------|----------|-------|--------|
| **Docker Compose** | Single host, dev/staging | Local volumes | Low |
| **Docker single container** | Quick trial | Local volumes | Lowest |
| **Kubernetes (YAML)** | Existing cluster | PVC | Medium |
| **Helm Chart** | Templated K8s, multi-env | PVC | Medium |
| **HuggingFace Space** | Public demo | Ephemeral | Low |

## 1. Docker Compose (recommended for single host)

```bash
# Base: Web UI + API only
docker-compose up -d

# With Redis LLM cache + Neo4j graph backend
docker-compose --profile cache --profile graph up -d
```

Ports exposed:

| Port | Service | Profile |
|------|---------|---------|
| 8000 | PocketGraphRAG Web UI + API | always |
| 6379 | Redis | `cache` |
| 7474, 7687 | Neo4j (HTTP, Bolt) | `graph` |

Persistent volumes (bind-mount under the project dir):

| Host path | Container path | Purpose |
|-----------|----------------|---------|
| `./index` | `/app/index` | FAISS vector index |
| `./user_docs` | `/app/user_docs` | Uploaded documents |
| `./data` | `/app/data` | Triple data |
| `./models` | `/app/models` | Embedding model cache |

### Connect to a host LLM

Point the container at host Ollama:

```yaml
environment:
  OLLAMA_API_BASE: http://host.docker.internal:11434/v1
  OLLAMA_MODEL: qwen2.5:7b
```

## 2. Docker single container

```bash
docker build -t pocketgraphrag .

docker run -d --name pocketgraphrag \
  -p 8000:8000 \
  -v "$PWD/index:/app/index" \
  -v "$PWD/user_docs:/app/user_docs" \
  -e OLLAMA_API_BASE=http://host.docker.internal:11434/v1 \
  -e OLLAMA_MODEL=qwen2.5:7b \
  pocketgraphrag
```

### Multi-arch build & push

```bash
REGISTRY=ghcr.io/jyzisliubi ./docker-build-push.sh --push
```

Produces `linux/amd64` + `linux/arm64` images. Releases are automatically
signed with [Cosign keyless](https://github.com/sigstore/cosign) and an SBOM
is attached — see `.github/workflows/docker-publish.yml`.

## 3. Kubernetes

### 3.1 Plain YAML

```bash
kubectl apply -f deploy/k8s/
kubectl port-forward svc/pocketgraphrag 8080:80
```

The manifests include:

- `Deployment` (1 replica by default, configurable)
- `Service` (ClusterIP, port 80 → 8000)
- `Ingress` (nginx-ingress, 100 MB body, 300 s timeout)
- 3 `PersistentVolumeClaim`s: `index` (5 Gi), `docs` (1 Gi), `models` (2 Gi)
- `livenessProbe` + `readinessProbe` against `/api/health`

### 3.2 Helm

```bash
helm install my-release deploy/helm/pocketgraphrag \
  --set env.OLLAMA_MODEL=qwen2.5:7b \
  --set redis.enabled=true
```

Key `values.yaml` knobs:

```yaml
replicaCount: 1
image:
  repository: ghcr.io/jyzisliubi/pocket-graph
  tag: latest
resources:
  requests: { memory: "2Gi", cpu: "1000m" }
  limits:   { memory: "4Gi", cpu: "2000m" }
persistence:
  index:    { size: 5Gi }
  docs:     { size: 1Gi }
  models:   { size: 2Gi }
redis:
  enabled: false          # set true for shared LLM cache
neo4j:
  enabled: false          # set true for Neo4j graph backend
ingress:
  enabled: true
  hostname: pocketgraphrag.example.com
  tls: true
```

## 4. Production checklist

### Statefulness

- **Single replica**: local PVC is fine.
- **Multi-replica** (`replicaCount > 1`): you MUST switch to:
  - Shared storage (NFS / EFS) or object storage (S3) for `index/` and `user_docs/`
  - `POCKET_LLM_CACHE_BACKEND=redis` with a shared Redis
  - `Neo4j` or `PostgresAGE` graph backend (the in-memory graph is not shared)
  - Make sure the API is stateless — do not rely on per-Pod memory.

### Resource sizing

| Workload | CPU req | Mem req | CPU lim | Mem lim |
|----------|:-------:|:-------:|:-------:|:-------:|
| Dev / small KG (< 10k triples) | 500m | 1Gi | 1000m | 2Gi |
| Production (< 100k triples) | 1000m | 2Gi | 2000m | 4Gi |
| Large KG (> 100k triples) | 2000m | 4Gi | 4000m | 8Gi |

Embedding model load is the main memory driver. `bge-m3` needs ~1.5 GB resident.

### Autoscaling

```bash
kubectl autoscale deployment pocketgraphrag \
  --min=2 --max=10 --cpu-percent=70
```

Only effective with shared storage + Redis cache (otherwise each Pod re-indexes).

### API authentication

```bash
POCKET_API_KEYS=key1,key2,key3
POCKET_API_AUTH_ENABLED=1
```

Clients send `Authorization: Bearer key1`. Keys rotate round-robin, so
rotating one does not drop in-flight requests.

### TLS

Terminate TLS at the Ingress:

```yaml
ingress:
  tls: true
  hostname: pocketgraphrag.example.com
```

Behind a load balancer, also set `POCKET_BEHIND_PROXY=1` so the API trusts
`X-Forwarded-*` headers.

## 5. Observability

### Health checks

| Probe | Endpoint | Meaning |
|-------|----------|---------|
| Liveness | `GET /api/health` | Process is alive |
| Readiness | `GET /api/health` | RAG system loaded |

Both are wired into Docker `HEALTHCHECK` and K8s probes.

### Langfuse tracing

```bash
POCKET_LANGFUSE=1
POCKET_LANGFUSE_PUBLIC_KEY=pk-lf-xxx
POCKET_LANGFUSE_SECRET_KEY=sk-lf-xxx
POCKET_LANGFUSE_HOST=https://cloud.langfuse.com
```

Every LLM call (extraction, summarization, QA) is then traced end-to-end in
the Langfuse UI, with retrieval sources attached as metadata.

### Logs

```bash
# Docker Compose
docker-compose logs -f pocketgraphrag

# Kubernetes
kubectl logs -f deployment/pocketgraphrag
```

Logs are JSON-structured when `POCKET_LOG_JSON=1` is set, which makes them
easy to ship to Loki / Elasticsearch / Datadog.

## 6. Backup & restore

### What to back up

| Path | Content | Frequency |
|------|---------|-----------|
| `index/` | FAISS index + `embedding_model.json` | After each rebuild |
| `user_docs/` | Uploaded documents | On change |
| `data/` | Triple data (JSON) | After each incremental add |
| `models/` | Embedding model cache | Once (re-downloadable) |

### Backup

```bash
# Snapshot the three data volumes
tar -czf pocketgraphrag-backup-$(date +%F).tar.gz index/ user_docs/ data/

# K8s: use Velero for volume snapshots
velero backup create pocketgraphrag-$(date +%F) \
  --include-resources persistentvolumeclaims,persistentvolumes \
  --selector app=pocketgraphrag
```

### Restore

```bash
# Docker Compose
docker-compose down
tar -xzf pocketgraphrag-backup-YYYY-MM-DD.tar.gz
docker-compose up -d

# K8s: restore the PVC from Velero snapshot, then:
kubectl rollout restart deployment/pocketgraphrag
```

## 7. Upgrading

### Docker Compose

```bash
git pull
docker-compose build
docker-compose up -d    # picks up the new image
```

### Helm

```bash
helm repo update
helm upgrade my-release pocketgraphrag/pocketgraphrag \
  --set image.tag=0.3.7
```

### Index compatibility

The `embedding_model.json` fingerprint guards against silent corruption. If
you switch embedding models against an existing index, the loader raises
`RuntimeError: embedding dimension mismatch`. To migrate:

1. Back up the old `index/`.
2. Set `POCKET_WORKSPACE=migration` (isolates the new index).
3. Rebuild: `pocketgraphrag build`.
4. Verify, then swap workspaces.

## 8. Performance tuning

| Knob | Default | Tune when |
|------|---------|-----------|
| `POCKET_TOP_K` | 5 | Increase for recall, decrease for latency |
| `POCKET_SEARCH_MODE` | mix | `kg_only` is fastest for KG-heavy queries |
| `POCKET_FUSION_STRATEGY` | rrf | `weighted` if you tuned the weights |
| `KG_SEARCH_HOPS` | 2 | 1 = faster, lower recall; 3 = slower, higher recall |
| `POCKET_LLM_CACHE` | off | Turn on for repeated extraction workloads |
| `POCKET_WORKERS` | 1 | Increase for parallel document ingestion |

For large KGs (> 100k triples), prefer `Neo4j` over the in-memory graph —
the NumPy PPR iteration gets memory-heavy.

## Next steps

- [Installation](installation.md) — first-time setup
- [Architecture](architecture.md) — how retrieval is scored
- [REST API](rest-api.md) — programmatic access
- [`deploy/`](https://github.com/jyzisliubi/Pocket-graph/tree/main/deploy) — raw manifests