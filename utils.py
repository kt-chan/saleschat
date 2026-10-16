# utils.py
"""
工具集合：
- retry: 重试装饰器（指数退避）
- parallel_map: 并行执行（进度条、超时）
- ConcurrencyLimiter: 信号量限流
- FileCache: 文件级 KV 缓存
"""
import json
import time
import hashlib
import threading
import functools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional

from tqdm import tqdm
from logger import get_logger

log = get_logger(__name__)


# ============================================================
# 1. 重试
# ============================================================
def retry(max_retries: int = 3, backoff: float = 2.0, exceptions=(Exception,)):
    """指数退避重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    wait = backoff ** attempt
                    log.warning(
                        f"{func.__name__} 第 {attempt + 1} 次失败: {e}, "
                        f"{wait:.1f}s 后重试"
                    )
                    time.sleep(wait)
            log.error(f"{func.__name__} 重试 {max_retries} 次后仍失败")
            raise last_exc
        return wrapper
    return decorator


# ============================================================
# 2. 并行执行
# ============================================================
def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: int = 8,
    desc: str = "Processing",
    timeout: Optional[float] = None,
    return_exceptions: bool = True,
) -> List[Any]:
    """
    并行执行（带进度条）
    :param timeout: 单任务超时（秒），None 表示不超时
    :param return_exceptions: True 时失败元素返回异常对象
    """
    if not items:
        log.warning(f"[{desc}] 输入为空，直接返回")
        return []

    results: List[Any] = [None] * len(items)
    failed = 0

    log.info(f"[{desc}] 开始: 总数={len(items)}, 并发={max_workers}")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): idx
                   for idx, item in enumerate(items)}
        with tqdm(total=len(items), desc=desc) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    # 只在明确指定 timeout 时才传递
                    if timeout is not None:
                        results[idx] = future.result(timeout=timeout)
                    else:
                        results[idx] = future.result()
                except Exception as e:
                    log.error(f"[{desc}] 任务 {idx} 失败: {e}")
                    failed += 1
                    if return_exceptions:
                        results[idx] = e
                finally:
                    pbar.update(1)

    elapsed = time.time() - t0
    log.info(f"[{desc}] 完成: 成功={len(items) - failed}, 失败={failed}, "
             f"耗时={elapsed:.1f}s")
    return results


# ============================================================
# 3. 并发限流
# ============================================================
class ConcurrencyLimiter:
    """信号量限流器"""

    def __init__(self, max_concurrency: int):
        self.semaphore = threading.Semaphore(max_concurrency)
        self.max_concurrency = max_concurrency

    def __enter__(self):
        self.semaphore.acquire()
        return self

    def __exit__(self, *args):
        self.semaphore.release()


# ============================================================
# 4. 文件缓存
# ============================================================
class FileCache:
    """基于文件的 KV 缓存"""

    def __init__(self, cache_dir: Path, namespace: str = "default"):
        self.cache_dir = Path(cache_dir) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"缓存目录: {self.cache_dir}")

    def _key_to_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"缓存读取失败 {path}: {e}")
        return None

    def set(self, key: str, value: Any) -> None:
        path = self._key_to_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
        except Exception as e:
            log.warning(f"缓存写入失败 {path}: {e}")

    def has(self, key: str) -> bool:
        return self._key_to_path(key).exists()