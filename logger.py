# logger.py
"""
统一日志（loguru）
- 控制台 + 文件 + 错误文件
- 异步写入（enqueue=True），避免 IO 阻塞
- 日志目录自动创建
"""
import sys
from loguru import logger
from settings import get_settings


def setup_logger(name: str = "cca"):
    """初始化全局日志（幂等，可重复调用）"""
    cfg = get_settings()
    logger.remove()  # 清除默认 handler

    if cfg.log_enable_console:
        logger.add(
            sys.stderr,
            level=cfg.log_level,
            format=cfg.log_format,
            colorize=True,
            backtrace=True,
            diagnose=False,
        )

    if cfg.log_enable_file:
        logger.add(
            cfg.log_path / f"{name}.log",
            level=cfg.log_level,
            format=cfg.log_format,
            rotation=cfg.log_rotation,
            retention=cfg.log_retention,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )
        logger.add(
            cfg.log_path / f"{name}.error.log",
            level="ERROR",
            format=cfg.log_format,
            rotation=cfg.log_rotation,
            retention=cfg.log_retention,
            encoding="utf-8",
            enqueue=True,
        )

    return logger


def get_logger(module_name: str):
    """获取带模块名的 logger"""
    return logger.bind(module=module_name)


setup_logger()