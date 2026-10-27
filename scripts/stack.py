import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
COMPOSE_FILE = WORKSPACE / "docker" / "docker-compose.yml"

TEI_INFO_URL = "http://127.0.0.1:8080/info"
TEI_HEALTH_TIMEOUT = int(os.environ.get("TEI_HEALTH_TIMEOUT", "180"))
PG_HEALTH_TIMEOUT = int(os.environ.get("PG_HEALTH_TIMEOUT", "30"))


def _resolve_compose_file() -> Path:
    """按优先级查找 compose 文件；找不到时明确报错并退出。"""
    candidates = [
        COMPOSE_FILE,  # 优先 docker/docker-compose.yml
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


def _wait_http(url: str, timeout: int, label: str) -> bool:
    deadline = time.time() + timeout
    print(f"等待 {label} 就绪: {url}（最长 {timeout}s）...")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    body = r.read().decode("utf-8", errors="ignore")
                    print(f"{label} 已就绪: {body[:200]}")
                    return True
        except Exception:
            pass
        time.sleep(1)
    print(f"{label} 启动超时", file=sys.stderr)
    return False


def _wait_pg() -> bool:
    print(f"等待 PostgreSQL 就绪（最长 {PG_HEALTH_TIMEOUT}s）...")
    compose_file = _resolve_compose_file()
    deadline = time.time() + PG_HEALTH_TIMEOUT
    while time.time() < deadline:
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
            print("PostgreSQL 已就绪")
            return True
        time.sleep(1)
    print("PostgreSQL 启动超时", file=sys.stderr)
    return False


def cmd_up(args):
    rc = _compose("up", "-d")
    if rc != 0:
        sys.exit(rc)
    if not _wait_pg():
        _compose("logs", "--tail", "40", "postgres")
        sys.exit(1)
    if not _wait_http(TEI_INFO_URL, TEI_HEALTH_TIMEOUT, "TEI"):
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
