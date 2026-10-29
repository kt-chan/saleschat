#!/usr/bin/env python3
# healthcheck.py
import sys
import os
import socket
import urllib.request
import urllib.error
import json

def check_llm():
    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    if not base.endswith("/chat/completions"):
        base += "/chat/completions"
    
    model = os.environ.get("LLM_MODEL", "")
    key = os.environ.get("LLM_API_KEY", "")
    host = os.environ.get("LLM_HOST", "")

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.1
    }).encode("utf-8")

    req = urllib.request.Request(base, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {key}")
    if host:
        req.add_header("Host", host)

    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"✅ LLM OK (HTTP {r.status})")
            sys.exit(0)
    except Exception as e:
        print(f"❌ LLM FAILED: {e}")
        sys.exit(1)

def check_pg():
    host = os.environ.get("DB_HOST", "")
    port = int(os.environ.get("DB_PORT", 5432))
    try:
        s = socket.create_connection((host, port), timeout=5)
        s.close()
        print(f"✅ PG OK (TCP {host}:{port})")
        sys.exit(0)
    except Exception as e:
        print(f"❌ PG FAILED: {e}")
        sys.exit(1)

def check_tei():
    # 直接使用已经配置好的完整端点，例如 http://.../v1/embeddings
    url = os.environ.get("EMBEDDING_TEI_URL", "").rstrip("/")
    model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
    
    # 构造一个最小的 embedding 请求体
    payload = json.dumps({
        "input": ["ping"],
        "model": model
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            print(f"✅ TEI OK (HTTP {r.status})")
            sys.exit(0)
    except urllib.error.HTTPError as e:
        # 如果返回了 4xx/5xx，说明网络通了，但业务参数不对（比如模型名写错），打印详情
        err_body = e.read().decode('utf-8', errors='ignore')
        print(f"❌ TEI FAILED (HTTP {e.code}): {err_body[:300]}")
        sys.exit(1)
    except Exception as e:
        # 真正的网络错误：DNS失败、拒绝连接、超时
        print(f"❌ TEI FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: healthcheck.py [llm|pg|tei]")
        sys.exit(1)
    
    target = sys.argv[1].lower()
    if target == "llm":
        check_llm()
    elif target == "pg":
        check_pg()
    elif target == "tei":
        check_tei()
    else:
        print(f"Unknown target: {target}")
        sys.exit(1)