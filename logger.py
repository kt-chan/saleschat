import sys
from loguru import logger
from settings import get_settings

def setup_logger(name: str = "cca"):
    cfg = get_settings()
    logger.remove()
    
    # 动态读取控制台与文件配置
    if cfg.log_enable_console:
        logger.add(sys.stderr, level=cfg.log_level, format=cfg.log_format, colorize=True)
        
    if cfg.log_enable_file:
        logger.add(
            cfg.log_path / f"{name}.log", 
            level=cfg.log_level, 
            format=cfg.log_format,
            rotation=cfg.log_rotation, 
            retention=cfg.log_retention, 
            encoding="utf-8", 
            enqueue=True
        )
        logger.add(
            cfg.log_path / f"{name}.error.log", 
            level="ERROR", 
            format=cfg.log_format,
            rotation=cfg.log_rotation, 
            retention=cfg.log_retention, 
            encoding="utf-8", 
            enqueue=True
        )
    return logger

def get_logger(module_name: str):
    return logger.bind(module=module_name)

setup_logger()