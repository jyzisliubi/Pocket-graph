#!/bin/bash
set -e

echo "========================================"
echo "  PocketGraphRAG - Lightweight GraphRAG"
echo "========================================"
echo ""

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "[INFO] 激活虚拟环境..."
    source .venv/bin/activate
fi

# 加载 .env 配置
if [ -f ".env" ]; then
    echo "[INFO] 加载 .env 配置..."
    set -a
    source .env
    set +a
fi

# 检查 LLM API Key
if [ -z "$DASHSCOPE_API_KEY" ] && [ -z "$DEEPSEEK_API_KEY" ] && [ -z "$SILICONFLOW_API_KEY" ] && [ -z "$OPENAI_API_KEY" ] && [ -z "$OLLAMA_MODEL" ]; then
    echo ""
    echo "[WARNING] 未检测到 LLM API Key 配置！"
    echo "[INFO] 请复制 .env.example 为 .env 并填入你的 API Key，或设置环境变量。"
    echo ""
    echo "[INFO] 推荐使用免费 API:"
    echo "       - 阿里云 DashScope: https://dashscope.console.aliyun.com/"
    echo "       - 硅基流动 SiliconFlow: https://siliconflow.cn/"
    echo "       - DeepSeek: https://platform.deepseek.com/"
    echo ""
    exit 1
fi

# 检查依赖
echo "[INFO] 检查依赖..."
if ! python -c "import gradio, faiss, sentence_transformers" 2>/dev/null; then
    echo "[INFO] 安装依赖中..."
    pip install -r requirements.txt
fi

# 安装 Playwright 浏览器
if ! python -c "from playwright.sync_api import sync_playwright" 2>/dev/null; then
    echo "[INFO] 安装 Playwright 浏览器..."
    playwright install chromium
fi

echo ""
echo "[INFO] 启动 Web UI..."
echo "[INFO] 访问地址: http://localhost:7860"
echo "[INFO] 按 Ctrl+C 停止服务"
echo ""

python -m PocketGraphRAG.webapp
