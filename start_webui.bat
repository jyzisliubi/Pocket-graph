@echo off
chcp 65001 >nul
echo ========================================
echo   PocketGraphRAG - Lightweight GraphRAG
echo ========================================
echo.

REM 检查虚拟环境
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] 激活虚拟环境...
    call .venv\Scripts\activate.bat
)

REM 加载 .env 配置（如果存在）
if exist ".env" (
    echo [INFO] 加载 .env 配置...
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
)

REM 检查 LLM API Key
if "%DASHSCOPE_API_KEY%"=="" if "%DEEPSEEK_API_KEY%"=="" if "%SILICONFLOW_API_KEY%"=="" if "%OPENAI_API_KEY%"=="" if "%OLLAMA_MODEL%"=="" (
    echo.
    echo [WARNING] 未检测到 LLM API Key 配置！
    echo [INFO] 仍会继续启动 Web UI，你可以先体验示例数据、检索结果和知识图谱。
    echo [INFO] 如需完整问答生成，请复制 .env.example 为 .env 并填入你的 API Key，或设置环境变量。
    echo [INFO] 支持: DASHSCOPE_API_KEY / DEEPSEEK_API_KEY / SILICONFLOW_API_KEY / OPENAI_API_KEY / OLLAMA_MODEL
    echo [INFO] 推荐免费/低门槛方案:
    echo        - 本地离线: 安装 Ollama 并设置 OLLAMA_MODEL
    echo        - 阿里云 DashScope (有免费额度): https://dashscope.console.aliyun.com/
    echo        - 硅基流动 SiliconFlow (免费模型): https://siliconflow.cn/
    echo        - DeepSeek (新用户有免费额度): https://platform.deepseek.com/
    echo.
)

REM 检查依赖
echo [INFO] 检查依赖...
python -c "import gradio, faiss, sentence_transformers" 2>nul
if errorlevel 1 (
    echo [INFO] 安装依赖中...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] 依赖安装失败！
        pause
        exit /b 1
    )
)

REM 安装 Playwright 浏览器（首次运行）
python -c "from playwright.sync_api import sync_playwright" 2>nul
if errorlevel 1 (
    echo [INFO] 安装 Playwright 浏览器...
    playwright install chromium
)

echo.
echo [INFO] 启动 Web UI...
echo [INFO] 访问地址: http://localhost:7860
echo [INFO] 按 Ctrl+C 停止服务
echo.

python -m PocketGraphRAG.webapp

pause
