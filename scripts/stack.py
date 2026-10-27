"""
Compose stack 管理辅助脚本（postgres + tei）。

用法:
  python scripts/stack.py up          # 启动全部服务并等待就绪
  python scripts/stack.py down        # 停止并删除容器（保留数据卷）
  python scripts/stack.py restart     # 重启全部
  python scripts/stack.py status      # 显示状态
  python scripts/stack.py logs [svc]  # 跟踪日志（可选服务名 postgres / tei）

环境变量（bash 直接可覆盖 .env）:
  TEI_HEALTH_TIMEOUT   等待 TEI 就绪的超时秒数（默认 180）
  PG_HEALTH_TIMEOUT    等待 PG 就绪的超时秒数（默认 30）
  EMBEDDING_TEI_URL    覆盖 .env 中的 TEI endpoint
  EMBEDDING_TEI_STYLE  覆盖 .env 中的风格（auto / native / openai）

TEI 探测策略（按顺序尝试，任一成功即返回）:
  1) GET  {base}/health          —— TEI 原生标准健康端点
  2) GET  {base}/info            —— 老版本 TEI
  3) POST {endpoint} 小 probe    —— 兜底：任意 OpenAI 兼容服务

EMBEDDING_TEI_URL 支持的形式（与 app 端保持一致）:
  - base URL              http://127.0.0.1:8080
  - native 完整 endpoint  http://127.0.0.1:8080/embed
  - openai 完整 endpoint  http://xxx.xx.xxx.xx:8010/v1/embeddings
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

WORKSPACE = Path(__file__).resolve().parent.parent
COMPOSE_FILE = WORKSPACE / "docker" / "docker-compose.yml"
ENV_FILE = WORKSPACE / ".env"

TEI_HEALTH_TIMEOUT = int(os.environ.get("TEI_HEALTH_TIMEOUT", "180"))
PG_HEALTH_TIMEOUT = int(os.environ.get("PG_HEALTH_TIMEOUT", "30"))


# ============================================================
# .env 读取（仅处理 KEY=VALUE 行，不做类型转换）
# ============================================================


def _read_env_file() -> dict:
    """解析 WORKSPACE/.env，返回 {key: value}。忽略注释和空行。"""
    result = {}
    if not ENV_FILE.exists():
        return result
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                result[k] = v
    except Exception as e:
        print(f"WARN: 读取 {ENV_FILE} 失败: {e}", file=sys.stderr)
    return result


def _get_config(key: str, default: str = "") -> str:
    """优先级：进程环境变量 > .env 文件 > default。"""
    if key in os.environ and os.environ[key] != "":
        return os.environ[key]
    env = _read_env_file()
    return env.get(key, default)


# ============================================================
# compose 调用
# ============================================================


def _resolve_compose_file() -> Path:
    candidates = [
        COMPOSE_FILE,
        WORKSPACE / "docker-compose.yml",
        WORKSPACE / "docker-compose.yaml",
        WORKSPACE / "compose.yml",
        WORKSPACE / "compose.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    env_path = os.environ.get("COMPOSE_FILE")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    print(
        f"ERROR: 未找到 compose 文件\n"
        f"  已尝试：\n" + "\n".join(f"    - {p}" for p in candidates) + "\n"
        f"  可通过 COMPOSE_FILE 环境变量显式指定。",
        file=sys.stderr,
    )
    sys.exit(2)


def _compose(*args) -> int:
    compose_file = _resolve_compose_file()
    cmd = ["docker", "compose", "-f", str(compose_file), *args]
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=WORKSPACE, check=False).returncode


# ============================================================
# endpoint 解析（与 app/utils.py 的 _resolve_endpoint 对齐）
# ============================================================


def _strip_endpoint_suffix(url: str) -> str:
    """把 .../v1/embeddings 或 .../embed 后缀去掉，返回 base URL。"""
    u = url.rstrip("/")
    for suffix in ("/v1/embeddings", "/embeddings", "/embed", "/info", "/health"):
        if u.endswith(suffix):
            return u[: -len(suffix)]
    return u


def _resolve_tei_endpoint(raw_url: str, style: str) -> Tuple[str, str, str]:
    """
    返回 (base_url, endpoint_url, 实际风格)。
      base_url    用于 GET /health 和 /info
      endpoint    用于 POST 探测 / 真实推理
    """
    u = (raw_url or "").strip().rstrip("/")
    if not u:
        raise ValueError("EMBEDDING_TEI_URL 为空")

    style = (style or "auto").lower()

    if style == "openai":
        if u.endswith("/v1/embeddings") or u.endswith("/embeddings"):
            endpoint = u
        else:
            endpoint = f"{u}/v1/embeddings"
        return _strip_endpoint_suffix(endpoint), endpoint, "openai"

    if style == "native":
        if u.endswith("/embed"):
            endpoint = u
        else:
            endpoint = f"{u}/embed"
        return _strip_endpoint_suffix(endpoint), endpoint, "native"

    # auto
    if u.endswith("/v1/embeddings") or u.endswith("/embeddings"):
        return _strip_endpoint_suffix(u), u, "openai"
    if u.endswith("/embed"):
        return _strip_endpoint_suffix(u), u, "native"
    return u, f"{u}/embed", "native"


# ============================================================
# 就绪探测
# ============================================================


def _make_session():
    """禁用系统代理的 requests.Session（HTTP_PROXY 是 urllib 假死的常见原因）。"""
    import requests

    sess = requests.Session()
    sess.trust_env = False
    return sess


def _wait_pg() -> bool:
    print(f"等待 PostgreSQL 就绪（最长 {PG_HEALTH_TIMEOUT}s）...")
    compose_file = _resolve_compose_file()
    deadline = time.time() + PG_HEALTH_TIMEOUT
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        rc = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "exec",
                "-T",
                "postgres",
                "pg_isready",
                "-U",
                "cca_user",
                "-d",
                "cca_db",
            ],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
        ).returncode
        if rc == 0:
            print(f"PostgreSQL 已就绪（尝试 {attempt}）")
            return True
        if attempt % 5 == 0:
            print(f"  [{attempt}] PostgreSQL 尚未就绪")
        time.sleep(1)
    print(f"PostgreSQL 启动超时（{PG_HEALTH_TIMEOUT}s）", file=sys.stderr)
    return False


def _probe_tei_once(sess, base: str, endpoint: str, style: str, model: str) -> str:
    """
    单次探测；返回 "" 表示未就绪，否则返回成功来源的描述。

    三层策略：/health → /info → probe embed
    """
    import requests

    # ---- 1) /health ----
    try:
        r = sess.get(f"{base}/health", timeout=(2, 3))
        if r.status_code == 200:
            return f"/health 200"
    except requests.exceptions.ConnectTimeout:
        return "CONNECT_TIMEOUT"
    except requests.exceptions.ConnectionError:
        return "CONN_ERROR"
    except Exception:
        pass

    # ---- 2) /info ----
    try:
        r = sess.get(f"{base}/info", timeout=(2, 3))
        if r.status_code == 200:
            body = r.text[:120]
            return f"/info 200 {body}"
    except Exception:
        pass

    # ---- 3) probe embed ----
    try:
        if style == "openai":
            payload = {"input": ["ping"], "model": model}
        else:
            payload = {"inputs": ["ping"], "truncate": True}

        r = sess.post(endpoint, json=payload, timeout=(3, 10))
        if r.status_code == 200:
            try:
                body = r.json()
                if isinstance(body, dict) and "data" in body:
                    return "probe embed OK (openai)"
                if isinstance(body, list):
                    return "probe embed OK (native)"
            except Exception:
                return "probe embed OK（响应非 JSON，仍视为可用）"
    except Exception:
        pass

    return ""


def _wait_tei() -> bool:
    raw_url = _get_config("EMBEDDING_TEI_URL", "http://127.0.0.1:8080")
    style = _get_config("EMBEDDING_TEI_STYLE", "auto").lower()
    model = _get_config("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")

    try:
        base, endpoint, resolved_style = _resolve_tei_endpoint(raw_url, style)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False

    print(
        f"等待 TEI 就绪 | base={base} endpoint={endpoint} "
        f"style={resolved_style} model={model}（最长 {TEI_HEALTH_TIMEOUT}s）..."
    )

    sess = _make_session()
    deadline = time.time() + TEI_HEALTH_TIMEOUT
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        result = _probe_tei_once(sess, base, endpoint, resolved_style, model)

        if result == "CONNECT_TIMEOUT":
            if attempt % 5 == 1:
                print(f"  [{attempt}] 连接 {base} 超时（服务可能尚未监听）")
        elif result == "CONN_ERROR":
            if attempt % 5 == 1:
                print(f"  [{attempt}] 无法连接 {base}（端口未开？）")
        elif result:
            print(f"TEI 已就绪（尝试 {attempt}，{result}）")
            return True
        else:
            if attempt % 10 == 0:
                print(
                    f"  [{attempt}] TEI 尚未就绪（端口已开，/health /info / probe 均未成功）"
                )

        time.sleep(1)

    print(f"TEI 启动超时（{TEI_HEALTH_TIMEOUT}s）", file=sys.stderr)
    return False


# ============================================================
# 命令
# ============================================================


def cmd_up(args):
    rc = _compose("up", "-d")
    if rc != 0:
        sys.exit(rc)
    if not _wait_pg():
        _compose("logs", "--tail", "40", "postgres")
        sys.exit(1)
    if not _wait_tei():
        _compose("logs", "--tail", "80", "tei")
        sys.exit(1)


def cmd_down(args):
    _compose("down")


def cmd_restart(args):
    _compose("restart")


def cmd_status(args):
    _compose("ps")


def cmd_logs(args):
    if args.service:
        _compose("logs", "-f", args.service)
    else:
        _compose("logs", "-f")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up").set_defaults(func=cmd_up)
    sub.add_parser("down").set_defaults(func=cmd_down)
    sub.add_parser("restart").set_defaults(func=cmd_restart)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p_logs = sub.add_parser("logs")
    p_logs.add_argument("service", nargs="?")
    p_logs.set_defaults(func=cmd_logs)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
