#!/bin/bash

echo "========================================"
echo "  freellm-cn 服务启动脚本"
echo "========================================"
echo ""

# 配置 freellm-cn 项目路径（根据实际情况修改）
FREELM_CN_DIR="$(cd "$(dirname "$0")" && pwd)/freellm-cn"

# 配置服务端口
FREELM_CN_PORT=8000

# 配置 API Key（可修改）
FREELM_CN_API_KEY="sk-freellm-cn-default-key"

# 检查目录是否存在
if [ ! -d "$FREELM_CN_DIR" ]; then
    echo "[错误] 未找到 freellm-cn 目录: $FREELM_CN_DIR"
    echo ""
    echo "请先将 freellm-cn 项目克隆到此目录下:"
    echo "  cd $(dirname "$0")"
    echo "  git clone https://github.com/your-org/freellm-cn.git"
    echo ""
    echo "或修改本脚本中的 FREELM_CN_DIR 变量指向正确的路径。"
    echo ""
    exit 1
fi

echo "[信息] freellm-cn 目录: $FREELM_CN_DIR"
echo "[信息] 服务端口: $FREELM_CN_PORT"
echo "[信息] API Key: $FREELM_CN_API_KEY"
echo ""

cd "$FREELM_CN_DIR"

# 检查是否有虚拟环境
if [ -f ".venv/bin/activate" ]; then
    echo "[信息] 激活虚拟环境..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "[信息] 激活虚拟环境..."
    source venv/bin/activate
fi

echo ""
echo "[启动] 正在启动 freellm-cn 服务..."
echo "[提示] 服务启动后，访问 http://localhost:$FREELM_CN_PORT/v1"
echo "[提示] 按 Ctrl+C 停止服务"
echo ""

# 启动服务（根据实际项目调整启动命令）
# 示例 1: 使用 uvicorn 启动 FastAPI 服务
# uvicorn main:app --host 0.0.0.0 --port $FREELM_CN_PORT

# 示例 2: 使用 python 直接启动
# python -m freellm_cn --port $FREELM_CN_PORT --api-key $FREELM_CN_API_KEY

# 示例 3: 如果有 start 脚本
# ./start.sh

# 默认尝试常见的启动方式（请根据实际项目修改）
if [ -f "main.py" ]; then
    python main.py --port $FREELM_CN_PORT --api-key $FREELM_CN_API_KEY
elif [ -f "app.py" ]; then
    python app.py --port $FREELM_CN_PORT --api-key $FREELM_CN_API_KEY
else
    echo "[错误] 未找到启动入口文件（main.py 或 app.py）"
    echo "请修改本脚本中的启动命令以适配你的 freellm-cn 项目。"
    exit 1
fi
