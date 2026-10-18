import json, time, re, functools, os, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional, Dict
from tqdm import tqdm
import psycopg2
from psycopg2 import pool as pg_pool
from pgvector.psycopg2 import register_vector
import redis
import fakeredis

from logger import get_logger
from settings import get_settings

log = get_logger(__name__)
_db_pool = None
_embedded_pg_server = None


def safe_json_loads(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
    return {}


def get_db_pool():
    global _db_pool, _embedded_pg_server
    if _db_pool is None:
        cfg = get_settings()
        if cfg.use_embedded_db:
            try:
                import pgserver

                pg_data_dir = str(Path(cfg.cache_dir) / "local_pgdata")
                os.makedirs(pg_data_dir, exist_ok=True)
                _embedded_pg_server = pgserver.get_server(
                    pg_data_dir, cleanup_mode="stop"
                )
                dsn = _embedded_pg_server.get_uri()
                _db_pool = pg_pool.ThreadedConnectionPool(
                    cfg.db_pool_min, cfg.db_pool_max, dsn=dsn
                )
                log.info("本地嵌入式 PostgreSQL 启动成功")
            except Exception as e:
                log.error(f"本地数据库启动失败: {e}")
                raise e
        else:
            _db_pool = pg_pool.ThreadedConnectionPool(
                cfg.db_pool_min,
                cfg.db_pool_max,
                host=cfg.db_host,
                port=cfg.db_port,
                dbname=cfg.db_name,
                user=cfg.db_user,
                password=cfg.db_password,
            )
            log.info("远程 Pgvector ThreadedConnectionPool 初始化完成")
    return _db_pool


def get_db_conn():
    global _db_pool
    if _db_pool is None:
        get_db_pool()
    conn = _db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        try:
            register_vector(conn)
        except psycopg2.ProgrammingError as e:
            if "vector type not found" not in str(e):
                raise e
        return conn
    except Exception as e:
        log.warning(f"数据库连接心跳异常，正在尝试重连: {e}")
        _db_pool.putconn(conn, close=True)
        conn = _db_pool.getconn()
        try:
            register_vector(conn)
        except:
            pass
        return conn


def release_db_conn(conn):
    if _db_pool and conn:
        _db_pool.putconn(conn)


def clear_system_data():
    global _db_pool, _embedded_pg_server
    cfg = get_settings()
    log.warning("⚠️ CLEAR_CACHE_ON_START=True，正在清空本地缓存...")

    if cfg.use_embedded_db and _db_pool is not None:
        _db_pool.closeall()
        _db_pool = None
        _embedded_pg_server = None
        log.info("嵌入式数据库连接已释放")

    for p in [cfg.cache_path, cfg.output_path]:
        if p.exists():
            for item in p.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    log.info("本地目录 (cache/, output/) 已彻底清空")

    if not cfg.use_embedded_db:
        try:
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE qa_pairs RESTART IDENTITY;")
            conn.commit()
            release_db_conn(conn)
            log.info("外部 Pgvector 数据库表 (qa_pairs) 已清空")
        except Exception as e:
            log.warning(f"清空外部数据库失败: {e}")


def init_db():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id SERIAL PRIMARY KEY,
                    call_id VARCHAR(64),
                    stage VARCHAR(64) NOT NULL,
                    primary_intent VARCHAR(64) NOT NULL,
                    tags JSONB,
                    risk VARCHAR(16),
                    session_ptp BOOLEAN,
                    session_outcomes JSONB,
                    rewritten_question TEXT NOT NULL,
                    rewritten_answer TEXT NOT NULL,
                    embedding vector(768),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='qa_pairs' AND column_name='session_ptp');"
            )
            if not cur.fetchone()[0]:
                cur.execute("ALTER TABLE qa_pairs ADD COLUMN session_ptp BOOLEAN;")

            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_emb ON qa_pairs USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_stage_intent ON qa_pairs (stage, primary_intent);"
            )
        conn.commit()
        log.info("数据库平滑升级检查完成")
    finally:
        release_db_conn(conn)


class SessionStore:
    def __init__(self):
        cfg = get_settings()
        self.ttl = cfg.session_ttl
        self.memory = {}
        self.use_redis = False
        self.use_fake_redis = False

        if cfg.use_embedded_redis:
            try:
                self.r = fakeredis.FakeRedis(decode_responses=True)
                self.use_fake_redis = True
                log.info("已启用本地内存模拟 Redis (fakeredis) 存储会话")
            except Exception as e:
                log.error(f"fakeredis 初始化失败: {e}")
        elif cfg.redis_url:
            try:
                self.r = redis.from_url(cfg.redis_url, decode_responses=True)
                self.r.ping()
                self.use_redis = True
                log.info("远程 Redis 会话存储已连接")
            except Exception as e:
                log.warning(f"远程 Redis 连接失败，退化为本地字典存储: {e}")

    def get(self, sid: str) -> dict:
        if self.use_fake_redis or self.use_redis:
            data = self.r.get(f"session:{sid}")
            return json.loads(data) if data else {"stage": "", "history": []}
        return self.memory.get(sid, {"stage": "", "history": []})

    def set(self, sid: str, state: dict):
        if self.use_fake_redis or self.use_redis:
            self.r.setex(
                f"session:{sid}", self.ttl, json.dumps(state, ensure_ascii=False)
            )
        else:
            self.memory[sid] = state


def retry(max_retries: int = 3, backoff: float = 2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(backoff**attempt)

        return wrapper

    return decorator


def parallel_map(
    func: Callable, items: List[Any], max_workers: int = 8, desc: str = "Processing"
) -> List[Any]:
    if not items:
        return []
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): idx for idx, item in enumerate(items)}
        with tqdm(total=len(items), desc=desc) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=120)
                except Exception as e:
                    log.error(f"[{desc}] 任务 {idx} 失败: {e}")
                finally:
                    pbar.update(1)
    return results
