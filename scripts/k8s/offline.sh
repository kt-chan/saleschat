#!/usr/bin/env bash
#
# scripts/k8s/offline.sh
#
# 上传数据 + 跑离线建库 Job + **结束时确保 app 起来并重启 port-forward**
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

[ -f "$SCRIPT_DIR/config.env" ] && { set -a; source "$SCRIPT_DIR/config.env"; set +a; }
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

TEI_DEPLOY="tei-svc"
TEI_LABEL="tei"
APP_DEPLOY="cca-app"
APP_LABEL="cca-app"

NS="${NS:-cca}"
INFRA_PG_NS="${INFRA_PG_NS:-gv}"
INFRA_PG_SVC="${INFRA_PG_SVC:-gv-sql-service}"
INFRA_PG_PORT="${INFRA_PG_PORT:-14003}"
INFRA_TEI_NS="${INFRA_TEI_NS:-model-service}"
INFRA_TEI_SVC="${INFRA_TEI_SVC:-tei-svc}"
INFRA_TEI_PORT="${INFRA_TEI_PORT:-8010}"
DATA_DIR_LOCAL="${DATA_DIR_LOCAL:-$HOME/app/saleschat/data/202610}"
DB_HOST="${DB_HOST:-gv-sql-service.gv.svc.cluster.local}"
DB_PORT="${DB_PORT:-14003}"
DB_NAME="${DB_NAME:-cca_db}"
DB_USER="${DB_USER:-cca_user}"
IMAGE="${IMAGE:-ghcr.io/kt-chan/saleschat-amd64:latest}"
PF_PORT="${PF_PORT:-8000}"
PF_ADDRESS="${PF_ADDRESS:-127.0.0.1}"
PF_ENABLE="${PF_ENABLE:-1}"
PF_PID_FILE="$HOME/.cca-port-forward.pid"
PF_LOG_FILE="$HOME/.cca-port-forward.log"

BATCH_ARG=""
DATA_DIR_ARG=""
UPLOAD_DATA=1
FORCE_UPLOAD=0

while [ $# -gt 0 ]; do
    case "$1" in
        --no-upload)    UPLOAD_DATA=0; shift ;;
        --data-dir)     DATA_DIR_ARG="$2"; shift 2 ;;
        --force-upload) FORCE_UPLOAD=1; shift ;;
        -h|--help)      sed -n '2,15p' "$0"; exit 0 ;;
        -*)             die "未知参数: $1" ;;
        *)              BATCH_ARG="$1"; shift ;;
    esac
done

if [ -n "$BATCH_ARG" ] && [ -z "$DATA_DIR_ARG" ]; then
    BATCH_NAME=$(basename "$BATCH_ARG")
    if [ -d "$BATCH_ARG" ]; then
        DATA_DIR_ARG="$BATCH_ARG"
    elif [ -d "$PROJECT_ROOT/data/$BATCH_NAME" ]; then
        DATA_DIR_ARG="$PROJECT_ROOT/data/$BATCH_NAME"
    elif [ -d "$(dirname "$DATA_DIR_LOCAL")/$BATCH_NAME" ]; then
        DATA_DIR_ARG="$(dirname "$DATA_DIR_LOCAL")/$BATCH_NAME"
    fi
fi
[ -n "$DATA_DIR_ARG" ] && DATA_DIR_LOCAL="$DATA_DIR_ARG"

require_cmd kubectl
require_k8s

# ---- 恢复 trap：无论成功/失败，都保证 app 起来 + PF 重启 ----
APP_REPLICAS_ORIG=0
APP_SCALED=0
cleanup() {
    local rc=$?
    # 恢复 app（即使中途失败）
    if [ "${APP_SCALED:-0}" = "1" ] || [ "${APP_REPLICAS_ORIG:-0}" -lt 1 ]; then
        local target="${APP_REPLICAS_ORIG:-0}"
        [ "$target" -lt 1 ] && target=1
        local cur
        cur=$(kubectl -n "$NS" get deploy "$APP_DEPLOY" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
        case "${cur:-}" in ""|"null") cur=0 ;; esac
        if [ "$cur" -lt "$target" ]; then
            echo "" >&2
            echo "==> [cleanup] 恢复 $APP_DEPLOY replicas=$target" >&2
            kubectl -n "$NS" scale deploy/"$APP_DEPLOY" --replicas="$target" >/dev/null 2>&1 || true
        fi
    fi
    # 重启 PF（app Pod IP 变了，旧 PF 失效）
    if [ "${PF_ENABLE:-1}" = "1" ] && kubectl -n "$NS" get deploy "$APP_DEPLOY" >/dev/null 2>&1; then
        # 只在 app 有 Ready Pod 时才重启 PF
        if is_deploy_ready "$NS" "$APP_DEPLOY" 1; then
            echo "==> [cleanup] 重启 port-forward" >&2
            restart_port_forward "$NS" "$APP_DEPLOY" "$PF_PORT" "$PF_ADDRESS" 2>&1 | tail -3 >&2 || true
        fi
    fi
    exit $rc
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  离线建库"
echo "============================================================"
echo "  PG         : ${INFRA_PG_NS}/${INFRA_PG_SVC}:${INFRA_PG_PORT}"
echo "  TEI        : ${INFRA_TEI_NS}/${INFRA_TEI_SVC}:${INFRA_TEI_PORT}"
echo "  DB (app)   : ${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo "  app ns     : ${NS}"
echo "  UPLOAD     : $([ "$UPLOAD_DATA" = "1" ] && echo yes || echo no)"
echo "============================================================"

# 0. 从 config.env 同步配置
if kubectl -n "$NS" get cm cca-config >/dev/null 2>&1; then
    info "从 config.env 同步配置到 cca-config"
    sync_app_config "$NS" "cca-config"
fi

# 1/6 前置检查
step "1/6 前置检查"
check_infra "$INFRA_PG_NS"  "$INFRA_PG_SVC"  "PostgreSQL" || die "PG Service 缺失"
check_infra "$INFRA_TEI_NS" "$INFRA_TEI_SVC" "TEI"        || die "TEI Service 缺失"

PG_SVC_PORT=$(kubectl -n "$INFRA_PG_NS" get svc "$INFRA_PG_SVC" \
    -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "")
[ "$PG_SVC_PORT" = "$INFRA_PG_PORT" ] || die "PG Service 端口不一致: 实际=$PG_SVC_PORT 期望=$INFRA_PG_PORT"
TEI_SVC_PORT=$(kubectl -n "$INFRA_TEI_NS" get svc "$INFRA_TEI_SVC" \
    -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "")
[ "$TEI_SVC_PORT" = "$INFRA_TEI_PORT" ] || die "TEI Service 端口不一致: 实际=$TEI_SVC_PORT 期望=$INFRA_TEI_PORT"
ok "Service 端口正确 (PG:${PG_SVC_PORT} TEI:${TEI_SVC_PORT})"

if kubectl -n "$INFRA_PG_NS" get pod postgres-0 >/dev/null 2>&1; then
    kubectl -n "$INFRA_PG_NS" wait --for=condition=ready pod/postgres-0 --timeout=60s >/dev/null 2>&1 \
        || die "postgres-0 未 Ready"
    kubectl -n "$INFRA_PG_NS" exec postgres-0 -- \
        pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1 \
        || die "pg_isready 失败"
    ok "PG Pod Ready"
fi

if kubectl -n "$INFRA_TEI_NS" get deploy "$TEI_DEPLOY" >/dev/null 2>&1; then
    TEI_SPEC_REPLICAS=$(kubectl -n "$INFRA_TEI_NS" get deploy "$TEI_DEPLOY" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "?")
    [ "$TEI_SPEC_REPLICAS" = "0" ] && die "TEI deploy $TEI_DEPLOY spec.replicas=0（未恢复）
  修复: kubectl -n $INFRA_TEI_NS scale deploy/$TEI_DEPLOY --replicas=1"
    TEI_READY=$(kubectl -n "$INFRA_TEI_NS" get deploy "$TEI_DEPLOY" \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    case "${TEI_READY:-}" in ""|"null") TEI_READY=0 ;; esac
    if [ "$TEI_READY" -lt 1 ]; then
        EP_COUNT=$(kubectl -n "$INFRA_TEI_NS" get endpoints "$INFRA_TEI_SVC" \
            -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null | wc -w | tr -d ' ')
        [ "${EP_COUNT:-0}" -ge 1 ] || die "TEI 未就绪 (spec=$TEI_SPEC_REPLICAS ready=$TEI_READY endpoints=$EP_COUNT)"
    fi
    ok "TEI Ready (spec=$TEI_SPEC_REPLICAS ready=$TEI_READY)"
fi

CM_DB_HOST=$(kubectl -n "$NS" get cm cca-config -o jsonpath='{.data.DB_HOST}' 2>/dev/null || echo "")
CM_DB_PORT=$(kubectl -n "$NS" get cm cca-config -o jsonpath='{.data.DB_PORT}' 2>/dev/null || echo "")
[ -n "$CM_DB_HOST" ] || die "cca-config 不存在"
[ "$CM_DB_HOST" = "$DB_HOST" ] || die "cca-config.DB_HOST 不一致: $CM_DB_HOST vs $DB_HOST"
[ "$CM_DB_PORT" = "$DB_PORT" ] || die "cca-config.DB_PORT 不一致: $CM_DB_PORT vs $DB_PORT"
ok "cca-config 与 config.env 一致"

# 2/6 上传数据
if [ "$UPLOAD_DATA" = "1" ]; then
    step "2/6 上传数据"
    if [ ! -x "$SCRIPT_DIR/upload.sh" ]; then
        warn "upload.sh 不存在或不可执行，跳过上传"
    else
        UPLOAD_ARGS=(--data-only)
        [ -n "$DATA_DIR_ARG" ] && UPLOAD_ARGS+=(--data-dir "$DATA_DIR_ARG")
        [ "$FORCE_UPLOAD" = "1" ] && UPLOAD_ARGS+=(--force)
        info "调用: upload.sh ${UPLOAD_ARGS[*]}"
        bash "$SCRIPT_DIR/upload.sh" "${UPLOAD_ARGS[@]}" || die "数据上传失败"
        ok "数据上传完成"
    fi
else
    step "2/6 上传数据（--no-upload 跳过）"
    if [ -n "$BATCH_ARG" ] && [ -z "$DATA_DIR_ARG" ]; then
        BATCH_NAME=$(basename "$BATCH_ARG")
        TARGET_DIR="/app/data/$BATCH_NAME"
        info "切换 DATA_DIR -> $TARGET_DIR"
        kubectl -n "$NS" patch configmap cca-config --type merge \
            -p "{\"data\":{\"DATA_DIR\":\"$TARGET_DIR\"}}" >/dev/null
        ok "DATA_DIR=$TARGET_DIR"
    fi
fi

CUR_DATA_DIR=$(kubectl -n "$NS" get cm cca-config -o jsonpath='{.data.DATA_DIR}' 2>/dev/null || echo "/app/data")
info "当前 DATA_DIR=$CUR_DATA_DIR"

# 3/6 暂停 app
step "3/6 暂停 app（释放 app-data RWO）"
if kubectl -n "$NS" get deploy "$APP_DEPLOY" >/dev/null 2>&1; then
    APP_REPLICAS_ORIG=$(kubectl -n "$NS" get deploy "$APP_DEPLOY" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
    case "${APP_REPLICAS_ORIG:-}" in ""|"null") APP_REPLICAS_ORIG=0 ;; esac

    # 停下 PF（避免 app 消失后 PF 指向死 Pod）
    stop_port_forward 2>/dev/null || true

    if [ "${APP_REPLICAS_ORIG:-0}" -gt 0 ]; then
        kubectl -n "$NS" scale deploy/"$APP_DEPLOY" --replicas=0 >/dev/null
        APP_SCALED=1
        kubectl -n "$NS" wait --for=delete pod -l "app=$APP_LABEL" --timeout=120s >/dev/null 2>&1 || true
        ok "app 已暂停 (原 replicas=$APP_REPLICAS_ORIG)"
    else
        info "app 已是 replicas=0"
    fi
fi

# 4/6 运行 Job
step "4/6 运行 Job"
kubectl -n "$NS" delete job cca-offline --ignore-not-found --wait=true >/dev/null 2>&1 || true
kubectl apply -f "$SCRIPT_DIR/manifests/offline-job.yaml" >/dev/null

info "跟踪日志（Ctrl+C 退出跟踪）"
kubectl -n "$NS" logs -f job/cca-offline 2>/dev/null || true

echo ""
info "等待 Job 完成（最长 30 分钟）..."
JOB_OK=0
if kubectl -n "$NS" wait --for=condition=complete job/cca-offline --timeout=1800s >/dev/null 2>&1; then
    JOB_OK=1
    ok "Job 完成"
else
    err "Job 失败或超时"
    kubectl -n "$NS" get job cca-offline 2>/dev/null || true
    kubectl -n "$NS" logs job/cca-offline --tail=40 2>/dev/null || true
fi

# 5/6 验证
step "5/6 验证结果"
if [ "$JOB_OK" = "1" ]; then
    ROWS=$(kubectl -n "$NS" exec job/cca-offline -- python -c "
from app.utils import get_db_conn, release_db_conn
c = get_db_conn(); c.cursor().execute('SELECT count(*) FROM qa_pairs'); print(c.cursor().fetchone()[0]); release_db_conn(c)
" 2>/dev/null | tail -1 || echo "")
    [ -n "$ROWS" ] && ok "qa_pairs 行数: $ROWS" || warn "无法查询 qa_pairs"

    CHECK_POD="output-check-$$"
    kubectl -n "$NS" run "$CHECK_POD" --image="$IMAGE" --restart=Never \
        --image-pull-policy=Never \
        --overrides="{\"spec\":{\"containers\":[{\"name\":\"c\",\"image\":\"$IMAGE\",\"imagePullPolicy\":\"Never\",\"command\":[\"sh\",\"-c\",\"ls -la /out/ | head -20\"],\"volumeMounts\":[{\"name\":\"o\",\"mountPath\":\"/out\"}]}],\"volumes\":[{\"name\":\"o\",\"persistentVolumeClaim\":{\"claimName\":\"app-output\"}}]}}" >/dev/null 2>&1 || true
    kubectl -n "$NS" wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$CHECK_POD" --timeout=60s >/dev/null 2>&1 || true
    echo "    --- app-output 内容 ---"
    kubectl -n "$NS" logs "$CHECK_POD" 2>/dev/null | sed 's/^/    /' || true
    kubectl -n "$NS" delete pod "$CHECK_POD" --wait=false >/dev/null 2>&1 || true
fi

# 6/6 恢复 app + 重启 port-forward
step "6/6 恢复 app 与 port-forward"

# 目标副本数：原本的值（若曾暂停）；否则至少 1
if [ "${APP_SCALED:-0}" = "1" ] && [ "${APP_REPLICAS_ORIG:-0}" -ge 1 ]; then
    TARGET_REPLICAS="$APP_REPLICAS_ORIG"
else
    TARGET_REPLICAS=1
fi

CUR_REPLICAS=$(kubectl -n "$NS" get deploy "$APP_DEPLOY" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
case "${CUR_REPLICAS:-}" in ""|"null") CUR_REPLICAS=0 ;; esac

if [ "$CUR_REPLICAS" -ne "$TARGET_REPLICAS" ]; then
    kubectl -n "$NS" scale deploy/"$APP_DEPLOY" --replicas="$TARGET_REPLICAS" >/dev/null
    ok "app 副本: $CUR_REPLICAS → $TARGET_REPLICAS"
else
    info "app 副本已是 $TARGET_REPLICAS"
fi
APP_SCALED=0

info "等待 $APP_DEPLOY 就绪（最长 180s）..."
if kubectl -n "$NS" rollout status deploy/"$APP_DEPLOY" --timeout=180s >/dev/null 2>&1; then
    ok "$APP_DEPLOY Ready"
else
    warn "$APP_DEPLOY rollout 未在 180s 内完成，检查: kubectl -n $NS get pods -l app=$APP_LABEL"
fi

# 重启 port-forward（app 换了 Pod，旧的 PF 会指向死 IP）
restart_port_forward "$NS" "$APP_DEPLOY" "$PF_PORT" "$PF_ADDRESS" \
    || warn "port-forward 启动失败，手动: kubectl -n $NS port-forward svc/$APP_DEPLOY ${PF_PORT}:8000"

# 汇总
echo ""
echo "============================================================"
if [ "$JOB_OK" = "1" ]; then
    echo "  ✅ 建库完成"
    echo ""
    echo "  在线服务:"
    if [ "${PF_ENABLE:-1}" = "1" ] && [ -f "$PF_PID_FILE" ] && kill -0 "$(cat "$PF_PID_FILE" 2>/dev/null)" 2>/dev/null; then
        echo "    http://${PF_ADDRESS}:${PF_PORT}"
        echo "    curl http://${PF_ADDRESS}:${PF_PORT}/health/ready | jq ."
    else
        echo "    kubectl -n $NS port-forward svc/$APP_DEPLOY ${PF_PORT}:8000 &"
    fi
else
    echo "  ❌ 建库失败（app 与 port-forward 已恢复）"
    echo ""
    echo "  排查:"
    echo "    kubectl -n $NS logs job/cca-offline --tail=100"
fi
echo "============================================================"

[ "$JOB_OK" = "1" ] || exit 1