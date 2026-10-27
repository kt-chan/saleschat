"""
FastAPI 服务：催收话术推荐。

特性:
  - 离线文档: /docs 与 /redoc 的静态资源全部内嵌(fastapi-offline),
    无外网也能渲染 Swagger UI / ReDoc
  - 生命周期: lifespan 替代已废弃的 @app.on_event
  - 健康检查:
      /health        轻量存活检查 (liveness, 向后兼容)
      /health/ready  就绪检查 (探测 PG + TEI)
  - 优雅关闭: 关闭 DB 连接池
  - 异常兜底: 结构化 JSON 响应, 不暴露 traceback, 始终返回 fallback

启动方式(三选一):
  python -m app.server                        # 独立运行
  python main.py --mode online                # 由 main.py 统一启动
  uvicorn app.server:app --host 0.0.0.0       # 外部 uvicorn
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi_offline import FastAPIOffline
from pydantic import BaseModel, Field

from app.logger import get_logger
from app.online import get_retriever, suggest_script_for_conversation
from app.settings import get_settings
from app.utils import get_db_conn, release_db_conn

log = get_logger(__name__)
cfg = get_settings()


# ---------------- 数据模型 ----------------


class SuggestReq(BaseModel):
    conversation: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="整段对话文本，每行以『客户：』/『催收员：』开头",
    )
    stage_hint: Optional[str] = Field(
        None,
        max_length=64,
        description="可选：当前阶段提示；缺失时由状态机自动推断",
    )


class SuggestResp(BaseModel):
    success: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)


class HealthResp(BaseModel):
    status: str
    detail: Dict[str, str] = Field(default_factory=dict)


# ---------------- 生命周期 ----------------


@asynccontextmanager
async def lifespan(app: FastAPIOffline):
    # ===== startup =====
    t0 = time.time()
    log.info("FastAPI 启动中，预热 retriever ...")
    log.info("FastAPI Docs running on http://127.0.0.1:8000/docs")
    try:
        get_retriever()
        log.info(f"retriever 预热完成，耗时 {time.time() - t0:.2f}s")
    except Exception as e:
        # 预热失败不阻塞启动：端点首次调用会重试初始化
        log.warning(f"retriever 预热失败（{e}），改为首次请求时懒加载")

    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        release_db_conn(conn)
        log.info("PostgreSQL 连接正常")
    except Exception as e:
        log.warning(f"启动时 PostgreSQL 未就绪: {e}")

    yield

    # ===== shutdown =====
    log.info("FastAPI 关闭中 ...")
    try:
        import app.utils as utils

        if utils._db_pool is not None:
            utils._db_pool.close()
            utils._db_pool = None
            log.info("DB 连接池已关闭")
    except Exception as e:
        log.warning(f"关闭 DB 连接池异常: {e}")


# ---------------- App ----------------

app = FastAPIOffline(
    title="催收话术推荐",
    description=(
        "输入对话文本，返回推荐的下一句话术。\n\n"
        "**依赖服务**：PostgreSQL（状态机 + 历史话术）、"
        "TEI（向量嵌入）、LLM（话术生成）。"
    ),
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ---------------- 统一异常响应 ----------------


@app.exception_handler(HTTPException)
async def _http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )


@app.exception_handler(Exception)
async def _unhandled_exc_handler(request: Request, exc: Exception):
    log.exception(f"未处理异常: {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "internal server error"},
    )


# ---------------- 端点 ----------------


@app.get("/", include_in_schema=False)
def root():
    return {"service": "催收话术推荐", "docs": "/docs", "health": "/health"}


@app.post("/chat", response_model=SuggestResp)
def chat(req: SuggestReq):
    """
    根据对话历史推荐下一句话术。

    响应统一包一层 `{"success": bool, "data": {...}}`：
      - success=true  → data 为上游原始输出
      - success=false → data 含 error 与 fallback 话术（保证 HTTP 200，
                        前端始终能拿到可念给客户的兜底文案）
    """
    t0 = time.time()
    try:
        result = suggest_script_for_conversation(
            req.conversation,
            stage_hint=req.stage_hint,
        )
        if not isinstance(result, dict):
            result = {"result": result}
        log.info(
            f"/suggest 完成 耗时={time.time() - t0:.2f}s "
            f"stage_hint={req.stage_hint} chars={len(req.conversation)}"
        )
        return SuggestResp(success=True, data={"script": result["script"]})
    except Exception as e:
        log.exception(f"/suggest 失败: {e}")
        return SuggestResp(
            success=False,
            data={
                "error": str(e),
                "fallback": cfg.suggest_fallback_script,
            },
        )


@app.post("/suggest", response_model=SuggestResp)
def suggest(req: SuggestReq):
    """
    根据对话历史推荐下一句话术。

    响应统一包一层 `{"success": bool, "data": {...}}`：
      - success=true  → data 为上游原始输出
      - success=false → data 含 error 与 fallback 话术（保证 HTTP 200，
                        前端始终能拿到可念给客户的兜底文案）
    """
    t0 = time.time()
    try:
        result = suggest_script_for_conversation(
            req.conversation,
            stage_hint=req.stage_hint,
        )
        if not isinstance(result, dict):
            result = {"result": result}
        log.info(
            f"/suggest 完成 耗时={time.time() - t0:.2f}s "
            f"stage_hint={req.stage_hint} chars={len(req.conversation)}"
        )
        return SuggestResp(success=True, data=result)
    except Exception as e:
        log.exception(f"/suggest 失败: {e}")
        return SuggestResp(
            success=False,
            data={
                "error": str(e),
                "fallback": cfg.suggest_fallback_script,
            },
        )


@app.get("/health", response_model=HealthResp)
def health():
    """
    轻量存活检查（liveness）。

    只检查进程是否响应，不探测依赖——适合 docker healthcheck / k8s
    livenessProbe，避免依赖抖动导致容器被反复重启。
    """
    return HealthResp(status="ok")


@app.get("/health/ready", response_model=HealthResp)
def health_ready():
    """
    就绪检查（readiness）。

    探测 PG + TEI，任一不通返回 503，让 k8s 暂时摘流量等待恢复。
    不要用作 livenessProbe。
    """
    detail: Dict[str, str] = {}

    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        release_db_conn(conn)
        detail["postgres"] = "ok"
    except Exception as e:
        detail["postgres"] = f"error: {e}"

    try:
        r = requests.get(f"{cfg.embedding_tei_url}/info", timeout=3)
        r.raise_for_status()
        detail["tei"] = "ok"
    except Exception as e:
        detail["tei"] = f"error: {e}"

    if not all(v == "ok" for v in detail.values()):
        raise HTTPException(status_code=503, detail=f"not ready: {detail}")
    return HealthResp(status="ready", detail=detail)


# ---------------- 独立启动 ----------------

if __name__ == "__main__":
    log.info("FastAPI is starting   ...")
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # 有进程内单例（retriever / DB pool），保持 1
        log_level="info",
        timeout_keep_alive=30,  # LLM 调用较慢，容忍长连接
    )
