#!/usr/bin/env bash
# scripts/k8s/lib/common.sh —— 公共函数

if [ -t 1 ]; then
    _C_RED=$'\033[31m'; _C_GRN=$'\033[32m'; _C_YEL=$'\033[33m'; _C_RST=$'\033[0m'
else
    _C_RED=""; _C_GRN=""; _C_YEL=""; _C_RST=""
fi

ok()   { echo "${_C_GRN}[ OK ]${_C_RST} $*"; }
info() { echo "[INFO] $*"; }
warn() { echo "${_C_YEL}[WARN]${_C_RST} $*" >&2; }
err()  { echo "${_C_RED}[FAIL]${_C_RST} $*" >&2; }
die()  { echo "${_C_RED}[FAIL]${_C_RST} $*" >&2; exit 1; }
step() { echo ""; echo "━━━ $* ━━━"; }
log()  { echo "[$(date +%H:%M:%S)] $*"; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1"
}

require_k8s() {
    kubectl cluster-info >/dev/null 2>&1 || die "kubectl 无法连接集群
  检查: kubectl config current-context"
}

check_infra() {
    local ns="$1" svc="$2" label="$3"
    if ! kubectl -n "$ns" get svc "$svc" >/dev/null 2>&1; then
        err "$label: service '$ns/$svc' 不存在"
        return 1
    fi
    local ep
    ep=$(kubectl -n "$ns" get endpoints "$svc" \
        -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null | wc -w | tr -d ' ')
    local port
    port=$(kubectl -n "$ns" get svc "$svc" \
        -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "?")
    ok "$label: $ns/$svc (ClusterIP, port=$port, ${ep:-0} endpoints)"
    return 0
}

is_deploy_ready() {
    local ns="$1" name="$2" min="${3:-1}"
    local ready
    ready=$(kubectl -n "$ns" get deploy "$name" \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    case "$ready" in ""|"null") ready=0 ;; esac
    [ "$ready" -ge "$min" ]
}

derive_model_dir_name() {
    local model_id="$1"
    echo "models--$(echo "$model_id" | sed 's|/|--|g')"
}

is_minikube() {
    command -v minikube >/dev/null 2>&1 && minikube status >/dev/null 2>&1
}

# ------------------------------------------------------------
# 同步 config.env 到 cca-config ConfigMap
# ------------------------------------------------------------
SYNC_APP_CONFIG_VARS=(
    # 基础
    LLM_BASE_URL
    LLM_MODEL
    DB_HOST
    DB_PORT
    DB_NAME
    DB_USER
    EMBEDDING_TEI_URL
    EMBEDDING_MODEL
    EMBEDDING_TEI_STYLE
    LOG_LEVEL
    # 运行时行为
    CLEAR_CACHE_ON_START
    STATE_MACHINE_MODE
    # Embedding
    EMB_CACHE_SIZE
    EMBEDDING_SORT_BY_LENGTH
    EMBEDDING_TEI_TIMEOUT
    EMBEDDING_TEI_BATCH_SIZE
    EMBEDDING_TEI_CONCURRENCY
    EMBEDDING_TEI_RETRY_MAX
    EMBEDDING_TEI_RETRY_BACKOFF
    EMBEDDING_TEI_TRUNCATE
    # 检索
    ONLINE_TOP_K
    ONLINE_SIM_THRESHOLD
    HYBRID_MAX_HOPS
    HYBRID_GOLDEN_WEIGHT
    HYBRID_PG_WEIGHT
    HYBRID_PG_TOP_K
    HYBRID_GOLDEN_TOP_K_PER_NODE
    HYBRID_HOP_DECAY
    HYBRID_VOTE_TOP_N
)

sync_app_config() {
    local ns="$1" cm="$2"
    local patch="" first=1

    if ! kubectl -n "$ns" get cm "$cm" >/dev/null 2>&1; then
        return 0
    fi

    for var in "${SYNC_APP_CONFIG_VARS[@]}"; do
        local val="${!var:-}"
        [ -z "$val" ] && continue
        val="${val//\\/\\\\}"
        val="${val//\"/\\\"}"
        [ $first -eq 0 ] && patch+=","
        patch+="\"$var\":\"$val\""
        first=0
    done

    [ -z "$patch" ] && return 0

    kubectl -n "$ns" patch configmap "$cm" --type merge \
        -p "{\"data\":{$patch}}" >/dev/null
    info "已从 config.env 同步到 $ns/$cm"
}

sync_app_secret() {
    local ns="$1" sec="$2" key="$3" val="$4"
    [ -z "$val" ] && return 0
    local b64
    b64=$(printf '%s' "$val" | base64 | tr -d '\n')
    if kubectl -n "$ns" get secret "$sec" >/dev/null 2>&1; then
        kubectl -n "$ns" patch secret "$sec" --type merge \
            -p "{\"data\":{\"$key\":\"$b64\"}}" >/dev/null
    else
        kubectl -n "$ns" create secret generic "$sec" \
            --from-literal="$key=$val" >/dev/null
    fi
    info "已同步 $key → $ns/secret/$sec"
}

# ------------------------------------------------------------
# Port-forward 统一管理（幂等：先停再起，等就绪）
# 用法: restart_port_forward [namespace] [service] [port] [address]
# ------------------------------------------------------------
restart_port_forward() {
    local ns="${1:-${NS:-cca}}"
    local svc="${2:-${APP_DEPLOY:-cca-app}}"
    local port="${3:-${PF_PORT:-8000}}"
    local addr="${4:-${PF_ADDRESS:-127.0.0.1}}"
    local pid_file="${PF_PID_FILE:-$HOME/.cca-port-forward.pid}"
    local log_file="${PF_LOG_FILE:-$HOME/.cca-port-forward.log}"

    if [ "${PF_ENABLE:-1}" != "1" ]; then
        info "PF_ENABLE=0，跳过 port-forward"
        return 0
    fi

    # 1. 停旧 PF
    if [ -f "$pid_file" ]; then
        local old_pid
        old_pid=$(cat "$pid_file" 2>/dev/null || true)
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            info "停止旧 port-forward (PID=$old_pid)"
            kill "$old_pid" 2>/dev/null || true
            sleep 1
            kill -0 "$old_pid" 2>/dev/null && kill -9 "$old_pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
    pkill -f "kubectl.*port-forward.*-n $ns.*$svc.*${port}:8000" 2>/dev/null || true
    sleep 0.5

    # 2. 端口占用检查
    if command -v ss >/dev/null 2>&1 && ss -tlnp 2>/dev/null | grep -qE ":${port}\b"; then
        warn "端口 $port 已被占用，port-forward 未启动"
        ss -tlnp 2>/dev/null | grep -E ":${port}\b" >&2 || true
        return 1
    fi

    # 3. 起新 PF
    : > "$log_file"
    nohup kubectl -n "$ns" port-forward --address "$addr" \
        svc/"$svc" "${port}:8000" >> "$log_file" 2>&1 &
    echo $! > "$pid_file"
    disown $! 2>/dev/null || true

    # 4. 等就绪
    local ready=0
    for _ in $(seq 1 30); do
        curl -sf -o /dev/null --max-time 2 "http://${addr}:${port}/health" && { ready=1; break; }
        sleep 1
    done

    if [ "$ready" = "1" ]; then
        ok "port-forward 就绪: http://${addr}:${port} (PID=$(cat "$pid_file"))"
        return 0
    else
        warn "port-forward 30s 未就绪，日志: $log_file"
        tail -10 "$log_file" 2>/dev/null | sed 's/^/    /' >&2 || true
        return 1
    fi
}

stop_port_forward() {
    local pid_file="${PF_PID_FILE:-$HOME/.cca-port-forward.pid}"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null || true)
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
        info "port-forward 已停止"
    fi
}