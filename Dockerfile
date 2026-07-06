# PocketGraphRAG Dockerfile
# 多阶段构建：builder 阶段安装依赖 + 构建前端，runtime 阶段仅复制产物
# 对标 LightRAG 的 Docker 实践，支持 clone-and-run

# ==========================
# Stage 1: Builder
# ==========================
FROM python:3.10-slim AS builder

# 系统依赖（编译 faiss/torch 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层
COPY pyproject.toml README.md ./
COPY PocketGraphRAG/ ./PocketGraphRAG/

# 安装核心 + web + docs 依赖（不装 dev/eval 减小体积）
RUN pip install --no-cache-dir -e ".[web,docs]"

# ==========================
# Stage 2: Runtime
# ==========================
FROM python:3.10-slim AS runtime

# 仅安装运行时必要的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目代码
COPY --from=builder /app /app

# 复制前端预构建产物
COPY frontend/dist /app/frontend/dist

# 创建数据目录
RUN mkdir -p /app/index /app/user_docs /app/data

# 环境变量默认值
ENV POCKET_DATA_PATH=/app/data/triples.txt \
    POCKET_INDEX_DIR=/app/index \
    POCKET_USER_DOCS_DIR=/app/user_docs \
    POCKET_SEARCH_MODE=mix \
    POCKET_HOST=0.0.0.0 \
    POCKET_PORT=8000

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health', timeout=5)" || exit 1

# 启动命令
CMD ["python", "-m", "PocketGraphRAG.api_server", "--host", "0.0.0.0", "--port", "8000"]
