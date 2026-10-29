#!/usr/bin/env bash
#
# start.sh — CCA Kubernetes 部署入口 (Host 快检 + Pod 深检)
#
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$DIR"

for c in kubectl curl; do
    command -v "$c" >/dev/null 2>&1 || { echo "ERROR: 缺少命令 $c" >&2; exit 1; }
done

[ -f .env ] || { echo "ERROR: .env 不存在" >&2; exit 1; }
set -a; . ./.env; set +a

# 必改项校验
if [ -z "${LLM_API_KEY:-}" ] || [ "${LLM_API_KEY}" = "sk-REPLACE_ME" ]; then
    echo "ERROR: .env 的 LLM_API_KEY 未设置或仍为占位符" >&2; exit 1
fi
if [ -z "${APP_IMAGE:-}" ]; then
    echo "ERROR: .env 的 APP_IMAGE 未设置" >&2; exit 1
fi

# 默认值
: "${NS:=cca}" "${INFRA_PG_NS:=gv}" "${INFRA_TEI_NS:=model-service}"
: "${INFRA_PG_SVC:=gv-sql-service}" "${INFRA_PG_PORT:=14003}"
: "${INFRA_TEI_DEPLOY:=tei-svc}" "${INFRA_TEI_PORT:=8010}"
: "${PF_PORT:=8000}" "${PF_ADDRESS:=127.0.0.1}"
: "${MODEL_SHORT_NAME:=bge-base-zh-v1.5}"
: "${HF_CACHE_DIR:=$HOME/app/saleschat/models/huggingface}"
: "${DATA_DIR_LOCAL:=$HOME/app/saleschat/data/202610}"
: "${DATA_BATCH:=$(basename "$DATA_DIR_LOCAL")}"
: "${LLM_TIMEOUT:=240}"

TPL="$DIR/manifests/template"
OUT="$DIR/manifests/runtime"
PROBE_TPL="$DIR/manifests/probe-pod.yaml"
PF_PID="$DIR/.pf.pid"; PF_LOG="$DIR/.pf.log"

L() { echo "[$(date +%H:%M:%S)] $*"; }
W() { echo "[$(date +%H:%M:%S)] WARN: $*" >&2; }

# ============================================================
# 0-A. Host 层快速预检
# ============================================================
precheck_host_llm() {
    L "Running Host LLM pre-check..."
    local base="${LLM_BASE_URL%/}"
    base="${base%/chat/completions}"
    local url="${base}/chat/completions"

    L "  endpoint : $url"
    [ -n "${LLM_HOST:-}" ] && L "  host     : $LLM_HOST (via Header)"

    local host_arg=()
    if [ -n "${LLM_HOST:-}" ]; then
        host_arg=(-H "Host: $LLM_HOST")
    fi

    local tmp code
    tmp=$(mktemp)
    code=$(curl -sS -o "$tmp" -w '%{http_code}' \
        --connect-timeout 5 --max-time "$LLM_TIMEOUT" \
        "${host_arg[@]}" \
        -X POST "$url" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${LLM_API_KEY}" \
        -d "{\"model\":\"${LLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"temperature\":0.1}" \
        2>/dev/null || true)

    local body; body=$(cat "$tmp" 2>/dev/null || true); rm -f "$tmp"

    if [ "${code:-000}" = "200" ]; then
        L "  ✅ Host -> LLM 连通正常 (HTTP 200)"
    else
        W "  ⚠️ Host -> LLM 异常 (HTTP ${code:-000})"
        [ -n "$body" ] && W "  body: $(printf '%s' "$body" | head -c 300)"
        W "  (若目标是内网 IP，请检查本地 VPN 或路由)"
    fi
}

# ============================================================
# 0-B. Pod 层深度探针预检
# ============================================================
run_probe_pod() {
    local name=$1
    local target=$2
    
    kubectl -n "$NS" delete pod "$name" --ignore-not-found >/dev/null 2>&1
    
    export PROBE_POD_NAME="$name"
    export PROBE_TARGET="$target"
    
    local probe_tmp
    probe_tmp=$(mktemp)
    _render_file "$PROBE_TPL" "$probe_tmp"
    
    if ! kubectl apply -f "$probe_tmp" >/dev/null; then
        W "  无法创建探针 Pod ($name)，请检查前置资源 (ConfigMap/Secret) 是否就绪"
        rm -f "$probe_tmp"
        return 1
    fi
    rm -f "$probe_tmp"

    if kubectl -n "$NS" wait --for=jsonpath='{.status.phase}'=Succeeded pod/"$name" --timeout=30s >/dev/null 2>&1; then
        kubectl -n "$NS" logs "$name"
        kubectl -n "$NS" delete pod "$name" >/dev/null 2>&1
        return 0
    else
        kubectl -n "$NS" logs "$name" 2>/dev/null || true
        kubectl -n "$NS" delete pod "$name" >/dev/null 2>&1
        return 1
    fi
}

precheck_pod_all() {
    local skip_infra=${1:-false}
    
    L "Running In-Cluster deep pre-check..."
    
    # 无论如何，LLM 是外部服务，必须检查
    if run_probe_pod "cca-probe-llm" "llm"; then
        L "  ✅ Pod -> LLM 连通正常"
    else
        W "  ⚠️ Pod -> LLM 异常 (请查看上方探针日志)"
        [ "${PRECHECK_STRICT:-0}" = "1" ] && exit 1
    fi

    if [ "$skip_infra" = "true" ]; then
        L "  ⏭️  Dev mode: Skipping PG and TEI pre-checks (infrastructure will be recreated)"
    else
        if run_probe_pod "cca-probe-pg" "pg"; then
            L "  ✅ Pod -> PG 连通正常"
        else
            W "  ⚠️ Pod -> PG 异常 (请查看上方探针日志)"
            [ "${PRECHECK_STRICT:-0}" = "1" ] && exit 1
        fi

        if run_probe_pod "cca-probe-tei" "tei"; then
            L "  ✅ Pod -> TEI 连通正常"
        else
            W "  ⚠️ Pod -> TEI 异常 (请查看上方探针日志)"
            [ "${PRECHECK_STRICT:-0}" = "1" ] && exit 1
        fi
    fi
}

# ============================================================
# 0-C. 统一预检调度
# ============================================================
precheck_infra() {
    # 接收参数：true 表示跳过 PG/TEI 检查
    local skip_infra=${1:-false}
    
    # 1. 宿主机极速探路
    precheck_host_llm
    
    # 2. 检查必要文件是否已生成
    if [ ! -f "$OUT/03-configmap.yaml" ] || [ ! -f "$OUT/02-secret-app.yaml" ] || [ ! -f "$OUT/03c-probe-script.yaml" ] || [ ! -f "$PROBE_TPL" ]; then
        W "配置未渲染或缺少探针资源，跳过集群内深度预检。"
        return 0
    fi
    
    # 3. 提前创建 Namespace 和基础配置 (确保探针 Pod 有环境可跑)
    L "Ensuring namespaces and base configs exist for in-cluster pre-check..."
    
    kubectl apply -f "$OUT/00-namespace-app.yaml" >/dev/null 2>&1 || true
    if [ "${SKIP_INFRA:-0}" != "1" ] && [ -f "$OUT/00-namespace-infra.yaml" ]; then
        kubectl apply -f "$OUT/00-namespace-infra.yaml" >/dev/null 2>&1 || true
    fi
    
    sleep 1
    
    kubectl apply -f "$OUT/02-secret-app.yaml" >/dev/null 2>&1 || true
    kubectl apply -f "$OUT/03-configmap.yaml" >/dev/null 2>&1 || true
    kubectl apply -f "$OUT/03c-probe-script.yaml" >/dev/null 2>&1 || true

    echo "----------------------------------------------------"
    # 4. K8s 集群内深度网络排查
    precheck_pod_all "$skip_infra"
    echo "----------------------------------------------------"
}

# ============================================================
# 1. Render
# ============================================================
_render_file() {
    local f=$1 d=$2
    if command -v envsubst >/dev/null 2>&1; then
        envsubst < "$f" > "$d"
        return
    fi
    
    if command -v python3 >/dev/null 2>&1; then
        python3 -c '
import os, re, sys
with open(sys.argv[1]) as fp: content = fp.read()
content = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", lambda m: os.environ.get(m.group(1) or m.group(2), ""), content)
with open(sys.argv[2], "w") as fp: fp.write(content)
' "$f" "$d"
        return
    fi
    
    W "没有 envsubst 和 python3，使用基础 shell 替换..."
    local c p v; c=$(<"$f")
    for p in $(grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*' "$f" | sort -u); do
        v="${p#\$}"; v="${v#\{}"; v="${v%\}}"
        c="${c//$p/${!v-}}"
    done
    printf '%s\n' "$c" > "$d"
}

render() {
    L "Rendering manifests -> manifests/runtime/"
    [ -d "$TPL" ] || { echo "ERROR: 模板目录不存在: $TPL" >&2; exit 1; }
    [ -n "$OUT" ] && [ "$OUT" != "/" ] || { echo "ERROR: OUT 非法: '$OUT'" >&2; exit 1; }
    
    rm -rf "$OUT" && mkdir -p "$OUT/upload"
    
    for f in "$TPL"/*.yaml; do 
        [ -f "$f" ] || continue
        _render_file "$f" "$OUT/$(basename "$f")"
    done
    
    for f in "$TPL"/upload/*.yaml; do 
        [ -f "$f" ] || continue
        _render_file "$f" "$OUT/upload/$(basename "$f")"
    done

    if [ -f "$DIR/prompts.yaml" ]; then
        L "  Generating cca-prompts ConfigMap from root prompts.yaml"
        kubectl create configmap cca-prompts \
            --from-file=prompts.yaml="$DIR/prompts.yaml" \
            --namespace="$NS" \
            --dry-run=client -o yaml > "$OUT/03b-prompts.yaml"
    else
        W "  ⚠️ 根目录未找到 prompts.yaml，跳过 cca-prompts 生成！"
        echo "" > "$OUT/03b-prompts.yaml"
    fi

    if [ -f "$DIR/healthcheck.py" ]; then
        L "  Generating cca-probe-script ConfigMap from root healthcheck.py"
        kubectl create configmap cca-probe-script \
            --from-file=healthcheck.py="$DIR/healthcheck.py" \
            --namespace="$NS" \
            --dry-run=client -o yaml > "$OUT/03c-probe-script.yaml"
    else
        W "  ⚠️ 根目录未找到 healthcheck.py，跳过探针脚本生成！"
        echo "" > "$OUT/03c-probe-script.yaml"
    fi
    
    for f in "$OUT"/*.yaml; do 
        [ -s "$f" ] || [[ "$(basename "$f")" == "03b-prompts.yaml" || "$(basename "$f")" == "03c-probe-script.yaml" ]] || { echo "ERROR: 渲染结果为空: $f" >&2; exit 1; }
    done
    L "  ✅ 渲染完成"
}

# ============================================================
# 2. Upload
# ============================================================
upload() {
    L "Uploading assets to PVC..."
    kubectl apply -f "$OUT/00-namespace-app.yaml" >/dev/null
    kubectl apply -f "$OUT/01-pvc-app.yaml" >/dev/null
    if [ "${SKIP_INFRA:-0}" != "1" ] && [ "${SKIP_MODEL_UPLOAD:-0}" != "1" ]; then
        kubectl apply -f "$OUT/00-namespace-infra.yaml" >/dev/null
        kubectl apply -f "$OUT/01-pvc-infra.yaml" >/dev/null
    fi

    if [ "${SKIP_MODEL_UPLOAD:-0}" != "1" ] && [ -d "$HF_CACHE_DIR" ]; then
        L "  同步模型到 TEI PVC..."
        kubectl -n "$INFRA_TEI_NS" delete pod tei-upload --ignore-not-found --wait=true >/dev/null 2>&1
        kubectl apply -f "$OUT/upload/tei-upload.yaml" >/dev/null
        kubectl -n "$INFRA_TEI_NS" wait --for=condition=Ready pod/tei-upload --timeout=120s >/dev/null
        
        local snap_dir="$HF_CACHE_DIR/hub/models--BAAI--bge-base-zh-v1.5"
        if [ -d "$snap_dir" ]; then
            kubectl -n "$INFRA_TEI_NS" exec tei-upload -- mkdir -p "/data/models/$MODEL_SHORT_NAME"
            tar -chf - -C "$snap_dir" . | kubectl -n "$INFRA_TEI_NS" exec -i tei-upload -- tar -xf - -C "/data/models/$MODEL_SHORT_NAME/"
        fi
        kubectl -n "$INFRA_TEI_NS" delete pod tei-upload --wait=true >/dev/null 2>&1
    fi

    if [ -d "$DATA_DIR_LOCAL" ]; then
        L "  同步业务数据到 App PVC (批次 $DATA_BATCH)..."
        kubectl -n "$NS" delete pod app-upload --ignore-not-found --wait=true >/dev/null 2>&1
        kubectl apply -f "$OUT/upload/app-upload.yaml" >/dev/null
        kubectl -n "$NS" wait --for=condition=Ready pod/app-upload --timeout=120s >/dev/null

        kubectl -n "$NS" exec app-upload -- mkdir -p "/app/data/$DATA_BATCH"
        tar -cf - -C "$DATA_DIR_LOCAL" . | kubectl -n "$NS" exec -i app-upload -- tar -xf - -C "/app/data/$DATA_BATCH/"
        kubectl -n "$NS" delete pod app-upload --wait=true >/dev/null 2>&1
    fi
    L "  ✅ 上传完成"
}

# ============================================================
# 3. Up
# ============================================================
up() {
    L "Applying Kubernetes manifests..."
    kubectl apply -f "$OUT/00-namespace-app.yaml" >/dev/null
    kubectl apply -f "$OUT/01-pvc-app.yaml" >/dev/null
    kubectl apply -f "$OUT/02-secret-app.yaml" >/dev/null
    kubectl apply -f "$OUT/03-configmap.yaml" >/dev/null
    
    if [ -s "$OUT/03b-prompts.yaml" ]; then
        kubectl apply -f "$OUT/03b-prompts.yaml" >/dev/null
    fi

    if [ "${SKIP_INFRA:-0}" = "1" ]; then
        L "  SKIP_INFRA=1 → 跳过 PG/TEI 部署"
    else
        kubectl apply -f "$OUT/00-namespace-infra.yaml" >/dev/null
        kubectl apply -f "$OUT/01-pvc-infra.yaml" >/dev/null
        kubectl apply -f "$OUT/02-secret-infra.yaml" >/dev/null
        kubectl apply -f "$OUT/04-postgres.yaml" >/dev/null
        kubectl apply -f "$OUT/05-tei.yaml" >/dev/null
    fi

    kubectl apply -f "$OUT/06-app.yaml" >/dev/null

    if [ "${SKIP_INFRA:-0}" != "1" ]; then
        L "  等待 PostgreSQL 就绪..."
        kubectl -n "$INFRA_PG_NS" wait --for=condition=Ready pod/postgres-0 --timeout=180s >/dev/null
        L "  等待 TEI 就绪..."
        kubectl -n "$INFRA_TEI_NS" wait --for=condition=available deploy/"$INFRA_TEI_DEPLOY" --timeout=300s >/dev/null
    fi

    L "  等待 App 启动..."
    kubectl -n "$NS" rollout status deploy/cca-app --timeout=300s
    L "  ✅ 部署成功"
}

# ============================================================
# 4. Port-Forward
# ============================================================
pf_stop() {
    if [ -f "$PF_PID" ]; then
        kill "$(cat "$PF_PID" 2>/dev/null)" 2>/dev/null || true
        rm -f "$PF_PID"
    fi
    pkill -f "kubectl.*-n ${NS}.*port-forward.*svc/cca-app" 2>/dev/null || true
}

pf() {
    pf_stop
    L "Starting port-forward -> http://$PF_ADDRESS:$PF_PORT"
    : > "$PF_LOG"
    nohup kubectl -n "$NS" port-forward --address "$PF_ADDRESS" svc/cca-app "$PF_PORT:8000" >> "$PF_LOG" 2>&1 &
    echo $! > "$PF_PID"
    
    for _ in {1..15}; do
        if curl -sf -o /dev/null --max-time 2 "http://$PF_ADDRESS:$PF_PORT/health" 2>/dev/null; then
            L "  ✅ 端口转发已就绪"
            return 0
        fi
        sleep 1
    done
    W "端口转发启动较慢或健康检查未响应，请检查日志: $PF_LOG"
}

# ============================================================
# 5. Offline
# ============================================================
offline() {
    pf_stop
    L "Running offline pipeline job..."
    kubectl -n "$NS" delete job cca-offline --ignore-not-found --wait=true >/dev/null 2>&1
    kubectl apply -f "$OUT/07-offline-job.yaml" >/dev/null
    kubectl -n "$NS" logs -f job/cca-offline 2>/dev/null || true
    kubectl -n "$NS" wait --for=condition=complete job/cca-offline --timeout=1800s >/dev/null
    pf
    L "  ✅ 离线建库完成"
}

# ============================================================
# 6. Status
# ============================================================
status() {
    for n in "$INFRA_PG_NS" "$INFRA_TEI_NS" "$NS"; do
        echo ""; echo "=== Namespace: $n ==="
        kubectl -n "$n" get pods,svc,jobs 2>/dev/null || echo "  (无资源)"
    done
    echo ""
    if [ -f "$PF_PID" ] && kill -0 "$(cat "$PF_PID" 2>/dev/null)" 2>/dev/null; then
        echo "Port-Forward: Running → http://$PF_ADDRESS:$PF_PORT"
    else
        echo "Port-Forward: Not running"
    fi
}

# ============================================================
# 入口分发
# ============================================================
case "${1:-prod}" in
    prod)
        render
        precheck_infra false
        export SKIP_INFRA=1 SKIP_MODEL_UPLOAD=1
        upload; up; offline; pf
        ;;
    dev)
        render
        # 传入 true 表示跳过 PG 和 TEI 预检
        precheck_infra true
        upload; up; offline; pf
        ;;
    online)
        render
        precheck_infra false
        up; pf
        ;;
    render)   render ;;
    precheck) 
        render
        # 支持 ./start.sh precheck skip-infra
        if [ "${2:-}" = "skip-infra" ]; then
            precheck_infra true
        else
            precheck_infra false
        fi
        ;;
    up)       up ;;
    offline)  offline ;;
    pf)       pf ;;
    pf-stop)  pf_stop; L "Port-forward 已停止" ;;
    status)   status ;;
    *)
        echo "用法: $0 [prod|dev|online|render|precheck [skip-infra]|up|offline|pf|pf-stop|status]" >&2
        exit 1
        ;;
esac