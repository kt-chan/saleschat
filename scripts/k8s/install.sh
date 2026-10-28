#!/usr/bin/env bash
#
# scripts/k8s/install.sh
#
set -euo pipefail

# 任何命令失败都显示行号，不再静默退出
trap 'echo "[FAIL] line $LINENO: $BASH_COMMAND (exit=$?)" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFESTS_DIR="$SCRIPT_DIR/manifests"

[ -f "$SCRIPT_DIR/config.env" ] && { set -a; source "$SCRIPT_DIR/config.env"; set +a; }
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

# TEI/APP Deployment 名 = Service 名
TEI_DEPLOY="${INFRA_TEI_SVC:-tei-svc}"
TEI_LABEL="${INFRA_TEI_LABEL:-tei}"
APP_DEPLOY="${APP_DEPLOY:-cca-app}"
APP_LABEL="${APP_LABEL:-$APP_DEPLOY}"

NS="${NS:-cca}"
INFRA_PG_NS="${INFRA_PG_NS:-gv}"
INFRA_PG_SVC="${INFRA_PG_SVC:-gv-sql-service}"
INFRA_PG_PORT="${INFRA_PG_PORT:-14003}"
INFRA_TEI_NS="${INFRA_TEI_NS:-model-service}"
INFRA_TEI_SVC="${INFRA_TEI_SVC:-tei-svc}"
INFRA_TEI_PORT="${INFRA_TEI_PORT:-8010}"
IMAGE="${IMAGE:-ghcr.io/kt-chan/saleschat-amd64:latest}"
PF_PORT="${PF_PORT:-8000}"
PF_ADDRESS="${PF_ADDRESS:-127.0.0.1}"
PF_PID_FILE="$HOME/.cca-port-forward.pid"
PF_LOG_FILE="$HOME/.cca-port-forward.log"

DB_HOST="${DB_HOST:-gv-sql-service.gv.svc.cluster.local}"
DB_PORT="${DB_PORT:-14003}"
EMBEDDING_TEI_URL="${EMBEDDING_TEI_URL:-http://tei-svc.model-service.svc.cluster.local:8010/v1/embeddings}"
DATA_DIR_LOCAL="${DATA_DIR_LOCAL:-$HOME/app/saleschat/data/202610}"

MODE="standard"; SKIP_INFRA=0; INFRA_ONLY=0; SKIP_BUILD=0
PF_ENABLE="${PF_ENABLE:-1}"; AUTO_UPLOAD=1; UPLOAD_DATA=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dev)          MODE="dev"; SKIP_INFRA=1; shift ;;
        --skip-infra)   SKIP_INFRA=1; shift ;;
        --infra-only)   INFRA_ONLY=1; shift ;;
        --no-pf)        PF_ENABLE=0; shift ;;
        --no-upload)    AUTO_UPLOAD=0; shift ;;
        --upload-data)  UPLOAD_DATA=1; shift ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        --pf-port)      PF_PORT="$2"; shift 2 ;;
        --pf-address)   PF_ADDRESS="$2"; shift 2 ;;
        -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
        *) die "未知参数: $1" ;;
    esac
done

log "=========================================="
log "  CCA 部署  (mode=$MODE)"
log "=========================================="
log "  NS=$NS  PG_NS=$INFRA_PG_NS  TEI_NS=$INFRA_TEI_NS"
log "  IMAGE=$IMAGE  PF_PORT=$PF_PORT"
log "  AUTO_UPLOAD=$AUTO_UPLOAD  UPLOAD_DATA=$UPLOAD_DATA"
log "=========================================="

# ============================================================
# 安全地处理 StatefulSet 不可变字段
# ⚠️ 关键：每条命令都有显式分支 + 结尾 return 0，避免 set -e 静默退出
# ============================================================
pre_apply_infra() {
    log "    [预检 1/3] PG StatefulSet serviceName"

    if kubectl -n "$INFRA_PG_NS" get sts postgres >/dev/null 2>&1; then
        local cur
        cur=$(kubectl -n "$INFRA_PG_NS" get sts postgres \
            -o jsonpath='{.spec.serviceName}' 2>/dev/null || echo "")
        log "      sts/postgres 当前 serviceName=$cur, 期望=$INFRA_PG_SVC"

        if [ -n "$cur" ] && [ "$cur" != "$INFRA_PG_SVC" ]; then
            log "      ⚠️  serviceName 不可变，需删除 sts 后重建"

            # 确保 PVC 保留
            local policy
            policy=$(kubectl -n "$INFRA_PG_NS" get sts postgres \
                -o jsonpath='{.spec.persistentVolumeClaimRetentionPolicy.whenDeleted}' 2>/dev/null || echo "")
            if [ "$policy" = "Delete" ]; then
                log "      PVC 策略=Delete，先改 Retain"
                kubectl -n "$INFRA_PG_NS" patch sts postgres --type=merge \
                    -p '{"spec":{"persistentVolumeClaimRetentionPolicy":{"whenDeleted":"Retain"}}}' \
                    2>/dev/null || true
            fi

            log "      删除 sts/postgres（PVC 保留）"
            kubectl -n "$INFRA_PG_NS" delete sts postgres --wait=false || true

            # 手动等 Pod 删除（最多 60s）
            log "      等待 postgres-0 Pod 删除（最长 60s）..."
            local i
            for i in $(seq 1 60); do
                if ! kubectl -n "$INFRA_PG_NS" get pod postgres-0 >/dev/null 2>&1; then
                    log "      ✅ Pod 已删除"
                    break
                fi
                sleep 1
            done

            if kubectl -n "$INFRA_PG_NS" get pod postgres-0 >/dev/null 2>&1; then
                warn "      Pod 60s 未删除，强制删除"
                kubectl -n "$INFRA_PG_NS" delete pod postgres-0 \
                    --grace-period=0 --force 2>/dev/null || true
                sleep 3
            fi
            log "      ✅ StatefulSet 已删除（PVC 保留）"
        else
            log "      ✅ serviceName 已是 $INFRA_PG_SVC，无需重建"
        fi
    else
        log "      sts/postgres 不存在，跳过"
    fi

    log "    [预检 2/3] 旧 PG Service"
    if kubectl -n "$INFRA_PG_NS" get svc postgres >/dev/null 2>&1; then
        log "      清理旧 Service $INFRA_PG_NS/postgres"
        kubectl -n "$INFRA_PG_NS" delete svc postgres --ignore-not-found || true
        log "      ✅ 已删除"
    else
        log "      旧 Service $INFRA_PG_NS/postgres 不存在，跳过"
    fi

    log "    [预检 3/3] 旧 TEI Service"
    if kubectl -n "$INFRA_TEI_NS" get svc tei >/dev/null 2>&1; then
        log "      清理旧 Service $INFRA_TEI_NS/tei"
        kubectl -n "$INFRA_TEI_NS" delete svc tei --ignore-not-found || true
        log "      ✅ 已删除"
    else
        log "      旧 Service $INFRA_TEI_NS/tei 不存在，跳过"
    fi

    log "    ✅ 预检完成"
    return 0
}

# ============================================================
# 0. 前置检查
# ============================================================
log "==> 0. 前置检查"
command -v kubectl >/dev/null 2>&1 || die "kubectl 未安装"
kubectl cluster-info >/dev/null 2>&1 || die "kubectl 无法连接集群"
for f in infra-gv.yaml infra-ms.yaml cca-secret.yaml cca.yaml; do
    [ -f "$MANIFESTS_DIR/$f" ] || die "缺少 manifest: $f"
done
IS_MINIKUBE=0
is_minikube && IS_MINIKUBE=1

# ============================================================
# 1. infra
# ============================================================
if [ "$INFRA_ONLY" = "1" ] || [ "$SKIP_INFRA" = "0" ]; then
    log ""
    log "==> 1. infra（$INFRA_PG_NS + $INFRA_TEI_NS）"

    if [ "$SKIP_INFRA" = "0" ]; then
        log "    预检 StatefulSet / Service 冲突"
        pre_apply_infra

        log "    apply infra-gv.yaml"
        kubectl apply -f "$MANIFESTS_DIR/infra-gv.yaml" || die "infra-gv.yaml apply 失败"
        log "    apply infra-ms.yaml"
        kubectl apply -f "$MANIFESTS_DIR/infra-ms.yaml" || die "infra-ms.yaml apply 失败"
    fi

    log "    等待 postgres-0（最长 180s）..."
    kubectl -n "$INFRA_PG_NS" wait --for=condition=ready pod/postgres-0 --timeout=180s \
        || die "postgres-0 未就绪
  排查: kubectl -n $INFRA_PG_NS describe pod postgres-0
        kubectl -n $INFRA_PG_NS logs postgres-0 --tail=50"
    log "    ✅ PG 就绪"

    TEI_SPEC_REPLICAS=$(kubectl -n "$INFRA_TEI_NS" get deploy "$TEI_DEPLOY" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "")
    if [ "$TEI_SPEC_REPLICAS" = "0" ]; then
        log "    ⚠️  $TEI_DEPLOY spec.replicas=0，恢复至 1"
        kubectl -n "$INFRA_TEI_NS" scale deploy/"$TEI_DEPLOY" --replicas=1
        sleep 2
    fi

    log "    等待 TEI（60s 快检）..."
    if kubectl -n "$INFRA_TEI_NS" wait --for=condition=available deploy/"$TEI_DEPLOY" --timeout=60s 2>/dev/null; then
        log "    ✅ TEI 已就绪"
    else
        log "    TEI 60s 未就绪（可能缺模型或首次加载）"
        if [ "$AUTO_UPLOAD" = "1" ]; then
            [ -x "$SCRIPT_DIR/upload.sh" ] || die "upload.sh 不存在或不可执行"
            log "    调用 upload.sh --model-only..."
            bash "$SCRIPT_DIR/upload.sh" --model-only \
                || die "模型上传失败
  排查: ./upload.sh --model-only --force"
        else
            log "    --no-upload，跳过模型上传"
        fi

        log "    等待 TEI 最终就绪（最长 900s）..."
        if ! kubectl -n "$INFRA_TEI_NS" wait --for=condition=available deploy/"$TEI_DEPLOY" --timeout=900s 2>/dev/null; then
            kubectl -n "$INFRA_TEI_NS" get pods -l "app=$TEI_LABEL" -o wide 2>/dev/null | sed 's/^/    /' || true
            kubectl -n "$INFRA_TEI_NS" logs deploy/"$TEI_DEPLOY" --tail=30 2>/dev/null | sed 's/^/    /' || true
            die "TEI 未就绪
  排查: kubectl -n $INFRA_TEI_NS logs deploy/$TEI_DEPLOY --tail=80"
        fi
        log "    ✅ TEI 就绪"
    fi

    PG_PORT=$(kubectl -n "$INFRA_PG_NS" get svc "$INFRA_PG_SVC" \
        -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "?")
    TEI_PORT=$(kubectl -n "$INFRA_TEI_NS" get svc "$INFRA_TEI_SVC" \
        -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "?")
    log "    $INFRA_PG_SVC:$PG_PORT  $INFRA_TEI_SVC:$TEI_PORT"

    if [ "$INFRA_ONLY" = "1" ]; then
        log ""
        log "=========================================="
        log "  ✅ infra 就绪（infra-only）"
        log "=========================================="
        kubectl -n "$INFRA_PG_NS" get pods,svc
        kubectl -n "$INFRA_TEI_NS" get pods,svc
        exit 0
    fi
else
    log ""
    log "==> 1. infra 跳过（--skip-infra / --dev）"
fi

# ============================================================
# 2. 镜像
# ============================================================
log ""
log "==> 2. 镜像"
NEED_IMAGE=1
if [ "$IS_MINIKUBE" = "1" ] && minikube image ls 2>/dev/null | grep -qF "$IMAGE"; then
    log "    minikube 已有: $IMAGE"
    NEED_IMAGE=0
fi

if [ "$NEED_IMAGE" = "1" ]; then
    if [ "$SKIP_BUILD" = "1" ]; then
        warn "镜像不在 minikube 且 --skip-build，可能 ImagePullBackOff"
    else
        DF="$PROJECT_ROOT/docker/dockerfile"
        [ -f "$DF" ] || DF="$PROJECT_ROOT/docker/Dockerfile"
        [ -f "$DF" ] || die "找不到 Dockerfile"
        log "    构建: $IMAGE"
        docker build -f "$DF" -t "$IMAGE" "$PROJECT_ROOT" || die "镜像构建失败"
        if [ "$IS_MINIKUBE" = "1" ]; then
            log "    加载到 minikube ..."
            minikube image load "$IMAGE" || die "minikube image load 失败"
        fi
        log "    ✅ 镜像就绪"
    fi
fi

# ============================================================
# 3. 配置 + app
# ============================================================
log ""
log "==> 3. 部署配置与应用"
kubectl get ns "$NS" >/dev/null 2>&1 || kubectl create namespace "$NS" >/dev/null

log "    apply cca-secret.yaml"
kubectl apply -f "$MANIFESTS_DIR/cca-secret.yaml" || die "cca-secret.yaml apply 失败"
kubectl -n "$NS" get secret cca-secret >/dev/null 2>&1 || die "cca-secret 未创建"

if [ -f "$PROJECT_ROOT/prompts.yaml" ]; then
    kubectl -n "$NS" create configmap cca-prompts \
        --from-file=prompts.yaml="$PROJECT_ROOT/prompts.yaml" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
fi

log "    apply cca.yaml"
kubectl apply -f "$MANIFESTS_DIR/cca.yaml" || die "cca.yaml apply 失败"

log "    从 config.env 同步关键配置"
sync_app_config "$NS" "cca-config"
if [ -n "${LLM_API_KEY:-}" ]; then
    sync_app_secret "$NS" "cca-secret" "LLM_API_KEY" "$LLM_API_KEY"
fi
if [ -n "${DB_PASSWORD:-}" ]; then
    sync_app_secret "$NS" "cca-secret" "DB_PASSWORD" "$DB_PASSWORD"
fi

log "    重启 deploy/$APP_DEPLOY 加载最新配置"
kubectl -n "$NS" rollout restart deploy/"$APP_DEPLOY" >/dev/null
if ! kubectl -n "$NS" rollout status deploy/"$APP_DEPLOY" --timeout=300s >/dev/null 2>&1; then
    kubectl -n "$NS" get pods -l "app=$APP_LABEL" -o wide 2>/dev/null | sed 's/^/    /' || true
    kubectl -n "$NS" describe pod -l "app=$APP_LABEL" 2>/dev/null | grep -A5 Events | tail -10 | sed 's/^/    /' || true
    die "app 启动失败"
fi
log "    ✅ $APP_DEPLOY 就绪"

if [ "$UPLOAD_DATA" = "1" ]; then
    log ""
    log "==> 3.6 上传数据（--upload-data）"
    if [ ! -d "$DATA_DIR_LOCAL" ]; then
        warn "    DATA_DIR_LOCAL=$DATA_DIR_LOCAL 不存在，跳过"
    elif [ ! -x "$SCRIPT_DIR/upload.sh" ]; then
        die "upload.sh 不存在或不可执行"
    else
        log "    调用 upload.sh --data-only（源: $DATA_DIR_LOCAL）..."
        bash "$SCRIPT_DIR/upload.sh" --data-only --data-dir "$DATA_DIR_LOCAL" \
            || die "数据上传失败"
        log "    ✅ 数据上传完成"
    fi
fi

# ============================================================
# 4. 验证
# ============================================================
log ""
log "==> 4. 验证配置"
POD=$(kubectl -n "$NS" get pod -l "app=$APP_LABEL" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
[ -n "$POD" ] || die "找不到 $APP_DEPLOY Pod"
log "    Pod: $POD"

log "    容器内 env："
ACTUAL=$(kubectl -n "$NS" exec "$POD" -- sh -c \
    'echo "$DB_HOST|$DB_PORT|$EMBEDDING_TEI_URL|$LLM_BASE_URL|$LLM_MODEL|$CLEAR_CACHE_ON_START"' 2>/dev/null || echo "")
echo "      DB_HOST              = $(echo "$ACTUAL" | cut -d'|' -f1)"
echo "      DB_PORT              = $(echo "$ACTUAL" | cut -d'|' -f2)"
echo "      EMBEDDING_TEI_URL    = $(echo "$ACTUAL" | cut -d'|' -f3)"
echo "      LLM_BASE_URL         = $(echo "$ACTUAL" | cut -d'|' -f4)"
echo "      LLM_MODEL            = $(echo "$ACTUAL" | cut -d'|' -f5)"
echo "      CLEAR_CACHE_ON_START = $(echo "$ACTUAL" | cut -d'|' -f6)"

EXPECTED="${DB_HOST}|${DB_PORT}|${EMBEDDING_TEI_URL}|${LLM_BASE_URL:-}|${LLM_MODEL:-}|${CLEAR_CACHE_ON_START:-false}"
if [ "$ACTUAL" != "$EXPECTED" ]; then
    warn "    ⚠️  env 与 config.env 期望值不符"
    warn "        实际: $ACTUAL"
    warn "        期望: $EXPECTED"
else
    log "    ✅ env 与 config.env 一致"
fi

log "    验证 DB 连接..."
DB_OK=$(kubectl -n "$NS" exec "$POD" -- python -c '
from app.utils import get_db_conn, release_db_conn
c = get_db_conn(); c.cursor().execute("SELECT 1"); release_db_conn(c); print("OK")
' 2>&1 | tail -1)
[ "$DB_OK" = "OK" ] && log "    ✅ DB 连接正常" || warn "    ❌ DB 连接失败: $DB_OK"

log "    验证 TEI 连接..."
TEI_OK=$(kubectl -n "$NS" exec "$POD" -- python -c '
from app.utils import get_embedder
v = get_embedder().encode("ping"); print(f"OK {len(v)}")
' 2>&1 | tail -1)
echo "$TEI_OK" | grep -qE "^OK [0-9]+" \
    && log "    ✅ TEI 连接正常（dims=${TEI_OK#OK }）" \
    || warn "    ❌ TEI 连接失败: $TEI_OK"

# ============================================================
# 5. port-forward
# ============================================================
log ""
log "==> 5. port-forward"
restart_port_forward "$NS" "$APP_DEPLOY" "$PF_PORT" "$PF_ADDRESS" \
    || warn "port-forward 启动失败"

# ============================================================
# 6. 汇总
# ============================================================
log ""
log "=========================================="
log "  ✅ 部署完成"
log "=========================================="
kubectl -n "$NS" get pods,svc
if [ "$PF_ENABLE" = "1" ] && [ -f "$PF_PID_FILE" ]; then
    log ""
    log "  http://${PF_ADDRESS}:${PF_PORT}"
    log "  Docs  : http://${PF_ADDRESS}:${PF_PORT}/docs"
    log "  Ready : curl http://${PF_ADDRESS}:${PF_PORT}/health/ready | jq ."
    log "  停止PF: kill \$(cat $PF_PID_FILE)"
fi
log ""
log "  后续步骤:"
log "    ./offline.sh                       # 上传数据 + 离线建库"
log "=========================================="