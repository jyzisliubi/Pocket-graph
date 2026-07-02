"""允许通过 python -m PocketGraphRAG 直接启动 CLI。

优先使用 Typer CLI（cli.py），提供完整子命令；若 Typer 未安装，
回退到轻量 argparse 入口（app.py）。
"""

try:
    from .cli import main
except ImportError:
    from .app import main

main()
