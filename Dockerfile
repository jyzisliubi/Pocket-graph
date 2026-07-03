# PocketGraphRAG Dockerfile
# 轻量级 GraphRAG 框架 - 支持 Web UI 和 REST API
# 多阶段构建：先构建前端，再打包 Python 后端

# ==========================
# 阶段1：构建前端（React + Vite）
# ==========================
FROM node:18-slim AS frontend-builder
WORKDIR /app/frontend
# 先复制 package 元信息以利用 Docker 层缓存
COPY frontend/package*.json ./
RUN npm ci --registry=https://registry.npmmirror.com
# 再复制源码并构建
COPY frontend/ .
RUN npm run build

# ==========================
# 阶段2：Python 后端
# ==========================
FROM python:3.11-slim

LABEL maintainer="PocketGraphRAG Team"
LABEL description="PocketGraphRAG - Lightweight GraphRAG Framework for Vertical Domains"

# 设置工作目录
WORKDIR /app

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（sentence-transformers 和 faiss 需要的编译/运行依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制前端构建产物
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# 复制项目代码（含 pyproject.toml / requirements.txt）
COPY . .

# 安装 Python 依赖：核心 + web(fastapi/uvicorn) + docs(pdf/docx 解析)
RUN pip install --no-cache-dir -e ".[web,docs]"

# 暴露端口
# - 8000: FastAPI REST API + 前端静态托管
EXPOSE 8000

# 数据卷：挂载自定义数据和索引
VOLUME ["/app/data", "/app/index"]

# 启动 API 服务（同时托管前端）
CMD ["uvicorn", "PocketGraphRAG.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
