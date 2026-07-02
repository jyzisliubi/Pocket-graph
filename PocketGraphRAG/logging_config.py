"""统一日志配置

全项目通过 get_logger("pocketgraphrag.xxx") 获取 logger，
保证日志级别、格式、handler 统一，替代散落的 print()。

- 默认级别 INFO，可通过环境变量 POCKET_LOG_LEVEL 覆盖（DEBUG/INFO/WARNING/ERROR）
- 格式：[时间] [级别] [模块] 消息
- 避免重复初始化 handler（多次调用 get_logger 只配置一次）

使用方式：
    from .logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("正在加载索引...")
    logger.warning("降级到关键词重排: %s", e)
"""

from __future__ import annotations

import logging
import os
import sys

_LOGGER_NAME_PREFIX = "pocketgraphrag"
_CONFIGURED = False


def _configure_root_logger() -> None:
    """配置 pocketgraphrag 根 logger 的 handler 与格式（只执行一次）"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("POCKET_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger(_LOGGER_NAME_PREFIX)
    root.setLevel(level)
    # 避免向 root logger 重复传播导致双重输出
    root.propagate = False

    # 已有 handler 则不重复添加（防止多次 import 重复输出）
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str = "pocketgraphrag") -> logging.Logger:
    """获取统一命名空间下的 logger。

    传入 __name__ 时会自动归一到 pocketgraphrag 命名空间下，
    例如 PocketGraphRAG.rag_system -> pocketgraphrag.rag_system
    """
    _configure_root_logger()
    # 统一前缀，避免各模块日志分散
    if name.startswith(_LOGGER_NAME_PREFIX):
        return logging.getLogger(name)
    # 去掉可能的 "PocketGraphRAG." 前缀，统一小写
    short = name.split(".")[-1] if "." in name else name
    return logging.getLogger(f"{_LOGGER_NAME_PREFIX}.{short.lower()}")
