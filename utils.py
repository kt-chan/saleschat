import json, time, hashlib, functools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional
from tqdm import tqdm
from logger import get_logger
from settings import get_settings

log = get_logger(__name__)

def retry(max_retries: int = None, backoff: float = 2.0):
    """重试装饰器，支持从环境变量读取最大重试次数"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = max_retries if max_retries is not None else get_settings().llm_max_retries
            for attempt in range(retries):
                try: return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1: raise e
                    time.sleep(backoff ** attempt)
        return wrapper
    return decorator

def parallel_map(func: Callable, items: List[Any], max_workers: int = None, desc: str = "Processing", timeout_per_task: int = 300) -> List[Any]:
    """并发映射执行器"""
    if not items: return []
    workers = max_workers if max_workers is not None else get_settings().concurrency_labeler
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(func, item): idx for idx, item in enumerate(items)}
        with tqdm(total=len(items), desc=desc) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try: results[idx] = future.result(timeout=timeout_per_task)
                except Exception as e: log.error(f"[{desc}] 任务 {idx} 失败: {e}")
                finally: pbar.update(1)
    return results

class FileCache:
    def __init__(self, cache_dir: Path, namespace: str = "default"):
        self.cache_dir = Path(cache_dir) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        return self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._key_to_path(key)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f: return json.load(f)
        return None

    def set(self, key: str, value: Any):
        with open(self._key_to_path(key), "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)