@echo off
chcp 65001 >nul
echo === PocketGraphRAG Web UI ===
echo.

cd /d d:\rice_rag

REM 检查前端是否已构建
if not exist "frontend\dist\index.html" (
    echo [INFO] 前端未构建，正在构建...
    cd frontend
    call npm.cmd run build
    cd ..
    echo [INFO] 前端构建完成
)

REM 启动后端服务
echo [INFO] 启动服务...
echo [INFO] 访问 http://localhost:8000
python -m uvicorn PocketGraphRAG.api_server:app --host 0.0.0.0 --port 8000
