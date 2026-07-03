@echo off
chcp 65001 >nul
echo ========================================
echo   freellm-cn 服务启动脚本
echo ========================================
echo.

REM 配置 freellm-cn 项目路径（根据实际情况修改）
set FREELM_CN_DIR=%~dp0freellm-cn

REM 配置服务端口
set FREELM_CN_PORT=8000

REM 配置 API Key（可修改）
set FREELM_CN_API_KEY=sk-freellm-cn-default-key

REM 检查目录是否存在
if not exist "%FREELM_CN_DIR%" (
    echo [错误] 未找到 freellm-cn 目录: %FREELM_CN_DIR%
    echo.
    echo 请先将 freellm-cn 项目克隆到此目录下:
    echo   cd /d %~dp0
    echo   git clone https://github.com/your-org/freellm-cn.git
    echo.
    echo 或修改本脚本中的 FREELM_CN_DIR 变量指向正确的路径。
    echo.
    pause
    exit /b 1
)

echo [信息] freellm-cn 目录: %FREELM_CN_DIR%
echo [信息] 服务端口: %FREELM_CN_PORT%
echo [信息] API Key: %FREELM_CN_API_KEY%
echo.

cd /d "%FREELM_CN_DIR%"

REM 检查是否有虚拟环境
if exist ".venv\Scripts\activate.bat" (
    echo [信息] 激活虚拟环境...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [信息] 激活虚拟环境...
    call venv\Scripts\activate.bat
)

echo.
echo [启动] 正在启动 freellm-cn 服务...
echo [提示] 服务启动后，访问 http://localhost:%FREELM_CN_PORT%/v1
echo [提示] 按 Ctrl+C 停止服务
echo.

REM 启动服务（根据实际项目调整启动命令）
REM 示例 1: 使用 uvicorn 启动 FastAPI 服务
REM uvicorn main:app --host 0.0.0.0 --port %FREELM_CN_PORT%

REM 示例 2: 使用 python 直接启动
REM python -m freellm_cn --port %FREELM_CN_PORT% --api-key %FREELM_CN_API_KEY%

REM 示例 3: 如果有 start 脚本
REM start.bat

REM 默认尝试常见的启动方式（请根据实际项目修改）
if exist "main.py" (
    python main.py --port %FREELM_CN_PORT% --api-key %FREELM_CN_API_KEY%
) else if exist "app.py" (
    python app.py --port %FREELM_CN_PORT% --api-key %FREELM_CN_API_KEY%
) else (
    echo [错误] 未找到启动入口文件（main.py 或 app.py）
    echo 请修改本脚本中的启动命令以适配你的 freellm-cn 项目。
    pause
    exit /b 1
)

pause
