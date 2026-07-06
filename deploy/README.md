# PocketGraphRAG Deployment Guide

PocketGraphRAG 支持多种部署方式，从本地开发到生产级 K8s 集群。

## 1. Docker Compose（推荐快速启动）

```bash
# 基础部署（Web UI + API）
docker-compose up -d

# 启用 Redis LLM Cache + Neo4j 图存储
docker-compose --profile cache --profile graph up -d

# 查看日志
docker-compose logs -f pocketgraphrag

# 停止
docker-compose down
```

**端口映射**：
- 8000: PocketGraphRAG Web UI + API
- 6379: Redis（仅 --profile cache）
- 7474/7687: Neo4j（仅 --profile graph）

**数据持久化**：
- `./index`: FAISS 向量索引
- `./user_docs`: 用户上传文档
- `./data`: 三元组数据
- `./models`: Embedding 模型缓存

## 2. Docker 单容器

```bash
# 构建镜像
docker build -t pocketgraphrag .

# 运行（连接宿主机 Ollama）
docker run -d \
  --name pocketgraphrag \
  -p 8000:8000 \
  -v ./index:/app/index \
  -v ./user_docs:/app/user_docs \
  -e OLLAMA_API_BASE=http://host.docker.internal:11434/v1 \
  -e OLLAMA_MODEL=qwen2.5:7b \
  pocketgraphrag

# 多架构构建并推送
REGISTRY=ghcr.io/jyzisliubi ./docker-build-push.sh --push
```

## 3. Kubernetes

### 3.1 直接使用 YAML

```bash
# 部署
kubectl apply -f deploy/k8s/

# 端口转发访问
kubectl port-forward svc/pocketgraphrag 8080:80

# 查看 Pod 状态
kubectl get pods -l app=pocketgraphrag
```

### 3.2 使用 Helm Chart

```bash
# 添加仓库（如果已发布）
helm repo add pocketgraphrag https://jyzisliubi.github.io/Pocket-graph/helm/
helm install my-release pocketgraphrag/pocketgraphrag

# 或从源码安装
cd deploy/helm
helm install my-release ./pocketgraphrag \
  --set env.OLLAMA_MODEL=qwen2.5:7b \
  --set redis.enabled=true

# 自定义 values
helm install my-release ./pocketgraphrag -f my-values.yaml

# 升级
helm upgrade my-release ./pocketgraphrag --set image.tag=0.3.7

# 卸载
helm uninstall my-release
```

### 3.3 生产部署建议

1. **多副本**：设置 `replicaCount > 1`，需配合：
   - 共享存储（NFS/EFS）或对象存储（S3）存索引
   - Redis LLM Cache 共享缓存
   - 无状态 API（不要在 Pod 本地存数据）

2. **资源限制**：
   ```yaml
   resources:
     requests:
       memory: "4Gi"
       cpu: "2000m"
     limits:
       memory: "8Gi"
       cpu: "4000m"
   ```

3. **自动扩缩**：
   ```bash
   kubectl autoscale deployment pocketgraphrag --min=2 --max=10 --cpu-percent=70
   ```

4. **Ingress 配置**：
   编辑 `deploy/k8s/ingress.yaml` 设置域名和 TLS。

## 4. 配置说明

### 4.1 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `POCKET_SEARCH_MODE` | mix | 检索模式：vector/local/global/mix/kg_only |
| `POCKET_WORKSPACE` | default | 工作区隔离（v0.3.7） |
| `POCKET_EMBEDDING_MODEL` | BAAI/bge-small-zh-v1.5 | Embedding 模型 |
| `OLLAMA_API_BASE` | http://localhost:11434/v1 | Ollama API 地址 |
| `OLLAMA_MODEL` | qwen2.5:7b | Ollama 模型名 |
| `POCKET_LLM_CACHE` | (空) | 设为 1 启用 LLM Cache（v0.3.7） |
| `POCKET_LLM_CACHE_BACKEND` | memory | memory / redis |
| `POCKET_API_KEYS` | (空) | API Key 认证（逗号分隔多 key） |
| `POCKET_API_AUTH_ENABLED` | (空) | 设为 1 启用 API 认证 |

### 4.2 健康检查

- **Liveness**: `GET /api/health` — 服务是否存活
- **Readiness**: `GET /api/health` — RAG 系统是否就绪
- **Docker**: `HEALTHCHECK` 内置
- **K8s**: `livenessProbe` + `readinessProbe` 内置

## 5. 监控与日志

### 5.1 Langfuse Tracing

```bash
# 启用 Langfuse
export POCKET_LANGFUSE=1
export POCKET_LANGFUSE_PUBLIC_KEY=pk-lf-xxx
export POCKET_LANGFUSE_SECRET_KEY=sk-lf-xxx
export POCKET_LANGFUSE_HOST=https://cloud.langfuse.com
```

### 5.2 日志

```bash
# Docker
docker-compose logs -f pocketgraphrag

# K8s
kubectl logs -f deployment/pocketgraphrag
```
