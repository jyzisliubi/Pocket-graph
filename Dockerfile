# PocketGraphRAG Dockerfile
# 轻量级 GraphRAG 框架 - 支持 Web UI 和 REST API

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

# 安装系统依赖（sentence-transformers 和 faiss 需要的编译依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements 并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY PocketGraphRAG/ /app/PocketGraphRAG/
COPY examples/ /app/examples/

# 暴露端口
# - 7860: Gradio Web UI
# - 8000: FastAPI REST API
EXPOSE 7860 8000

# 数据卷：挂载自定义数据和索引
VOLUME ["/app/data", "/app/index"]

# 默认启动 Web UI
CMD ["python", "-m", "PocketGraphRAG.webapp", "--host", "0.0.0.0", "--port", "7860"]
