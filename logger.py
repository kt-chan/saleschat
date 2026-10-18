import sys
from loguru import logger
from settings import get_settings


def setup_logger():
    cfg = get_settings()
    logger.remove()
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{function}</cyan> | <level>{message}</level>"
    logger.add(sys.stderr, level=cfg.log_level, format=fmt, colorize=True)
    logger.add(
        cfg.log_path / "cca.log",
        level=cfg.log_level,
        rotation="100 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
    )
    logger.add(
        cfg.log_path / "error.log",
        level="ERROR",
        rotation="100 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
    )


setup_logger()


def get_logger(name):
    return logger.bind(module=name)
