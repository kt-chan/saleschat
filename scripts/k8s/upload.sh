#!/usr/bin/env bash
#
# scripts/k8s/upload.sh
#
# 上传模型到 tei-data PVC / 上传数据到 app-data PVC
#
# 用法:
#   ./upload.sh                        # 模型 + 数据
#   ./upload.sh --model-only           # 只上传模型
#   ./upload.sh --data-only            # 只上传数据
#   ./upload.sh --data-dir data/202611 # 指定数据源目录
#   ./upload.sh --batch 202611-v2      # 自定义批次名
#   ./upload.sh --force                # 强制重传
#   ./upload.sh --dry-run              # 只打印
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

[ -f "$SCRIPT_DIR/config.env" ] && { set -a; source "$SCRIPT_DIR/config.env"; set +a; }

# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

# ---------- 硬编码的 Deployment 名 ----------
TEI_DEPLOY="tei-svc"
TEI_LABEL="tei"
APP_DEPLOY="cca-app"
APP_LABEL="cca-app"

# ---------- 默认值 ----------
NS="${NS:-cca}"
INFRA_TEI_NS="${INFRA_TEI_NS:-model-service}"
TEI_PVC="${TEI_PVC:-tei-data}"
TEI_HF_HOME="${TEI_HF_HOME:-/data}"
TEI_HUB_SUBDIR="${TEI_HUB_SUBDIR:-hub}"
MODEL_ID="${MODEL_ID:-BAAI/bge-base-zh-v1.5}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-$(derive_model_dir_name "$MODEL_ID")}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/app/saleschat/models/huggingface}"

APP_DATA_PVC="${APP_DATA_PVC:-app-data}"
DATA_DIR_LOCAL="${DATA_DIR_LOCAL:-$HOME/app/saleschat/data/202610}"
CALLS_FILE_PATTERN="${CALLS_FILE_PATTERN:-*.jsonl}"

IMAGE="${IMAGE:-ghcr.io/kt-chan/saleschat-amd64:latest}"

# ---------- 参数 ----------
MODE="all"
FORCE=0
DRY_RUN=0
DATA_DIR_ARG=""
BATCH_ARG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --model-only)  MODE="model"; shift ;;
        --data-only)   MODE="data"; shift ;;
        --force)       FORCE=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --data-dir)    DATA_DIR_ARG="$2"; shift 2 ;;
        --batch)       BATCH_ARG="$2"; shift 2 ;;
        --image)       IMAGE="$2"; shift 2 ;;
        -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
        *) die "未知参数: $1" ;;
    esac
done

[ -n "$DATA_DIR_ARG" ] && DATA_DIR_LOCAL="$DATA_DIR_ARG"

# ---------- 前置 ----------
command -v kubectl >/dev/null 2>&1 || die "kubectl 未安装"
kubectl cluster-info >/dev/null 2>&1 || die "kubectl 无法连接集群"

# ---------- 状态 + 清理 trap ----------
TEI_REPLICAS=""
APP_WAS_SCALED=0
APP_REPLICAS=0
CURRENT_POD=""
CURRENT_NS=""

cleanup() {
    local rc=$?
    if [ -n "$CURRENT_POD" ] && [ -n "$CURRENT_NS" ]; then
        kubectl -n "$CURRENT_NS" delete pod "$CURRENT_POD" --wait=false 2>/dev/null || true
    fi
    if [ -n "${TEI_REPLICAS:-}" ] && [ "${TEI_REPLICAS:-0}" -gt 0 ]; then
        echo "" >&2
        echo "==> [cleanup] 恢复 $TEI_DEPLOY replicas=$TEI_REPLICAS" >&2
        kubectl -n "$INFRA_TEI_NS" scale deploy/"$TEI_DEPLOY" --replicas="$TEI_REPLICAS" 2>/dev/null || true
    fi
    if [ "$APP_WAS_SCALED" = "1" ] && [ "${APP_REPLICAS:-0}" -gt 0 ]; then
        echo "==> [cleanup] 恢复 $APP_DEPLOY replicas=$APP_REPLICAS" >&2
        kubectl -n "$NS" scale deploy/"$APP_DEPLOY" --replicas="$APP_REPLICAS" 2>/dev/null || true
    fi
    exit $rc
}
trap cleanup EXIT INT TERM

# ---------- 通用函数 ----------
launch_upload_pod() {
    local pod="$1" ns="$2" pvc="$3" mount="$4"
    kubectl -n "$ns" delete pod "$pod" --ignore-not-found --wait=true >/dev/null 2>&1 || true
    kubectl -n "$ns" run "$pod" --image="$IMAGE" --restart=Never \
        --image-pull-policy=Never \
        --overrides="{\"spec\":{\"containers\":[{\"name\":\"u\",\"image\":\"$IMAGE\",\"imagePullPolicy\":\"Never\",\"command\":[\"sleep\",\"1800\"],\"volumeMounts\":[{\"name\":\"d\",\"mountPath\":\"$mount\"}]}],\"volumes\":[{\"name\":\"d\",\"persistentVolumeClaim\":{\"claimName\":\"$pvc\"}}]}}" >/dev/null
    kubectl -n "$ns" wait --for=condition=Ready "pod/$pod" --timeout=120s >/dev/null 2>&1 \
        || die "upload pod $ns/$pod 未就绪（PVC $pvc 可能被占用）"
    CURRENT_POD="$pod"
    CURRENT_NS="$ns"
}

cleanup_upload_pod() {
    if [ -n "$CURRENT_POD" ] && [ -n "$CURRENT_NS" ]; then
        kubectl -n "$CURRENT_NS" delete pod "$CURRENT_POD" --wait=true >/dev/null 2>&1 || true
        CURRENT_POD=""
        CURRENT_NS=""
    fi
}

# ⚠️ 关键修复：info 走 stderr，stdout 只输出数值
scale_down() {
    local ns="$1" name="$2" label="$3"
    local cur
    cur=$(kubectl -n "$ns" get deploy "$name" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
    case "${cur:-}" in ""|"null") cur=0 ;; esac
    if [ "${cur:-0}" -gt 0 ]; then
        echo "[INFO] 暂停 $name (replicas=$cur → 0)" >&2
        kubectl -n "$ns" scale deploy/"$name" --replicas=0 >/dev/null
        kubectl -n "$ns" wait --for=delete pod -l "app=$label" --timeout=120s >/dev/null 2>&1 || true
        echo "$cur"
    else
        echo "[INFO] $name 已是 replicas=0" >&2
        echo 0
    fi
}

scale_up() {
    local ns="$1" name="$2" replicas="$3"
    [ "${replicas:-0}" -gt 0 ] || return 0
    echo "[INFO] 恢复 $name replicas=$replicas" >&2
    kubectl -n "$ns" scale deploy/"$name" --replicas="$replicas" >/dev/null
    kubectl -n "$ns" rollout status deploy/"$name" --timeout=600s >/dev/null 2>&1 || \
        warn "$name rollout 未在 600s 内完成" >&2
}

# ============================================================
# 上传模型
# ============================================================
upload_model() {
    step "上传模型 → $TEI_PVC"

    local src="$HF_CACHE_DIR/hub/$MODEL_DIR_NAME"
    local remote_dir="$TEI_HF_HOME/$TEI_HUB_SUBDIR/$MODEL_DIR_NAME"

    [ -d "$src" ] || die "本地模型目录不存在: $src
  检查 config.env 的 HF_CACHE_DIR 和 MODEL_DIR_NAME"
    [ -d "$src/snapshots" ] || die "非 HF 缓存目录（缺 snapshots/）: $src"

    local size
    size=$(du -sh "$src" | cut -f1)
    info "源: $src ($size)"
    info "目标: $TEI_PVC:$remote_dir/"

    if [ "$DRY_RUN" = "1" ]; then
        info "[dry-run] 暂停 $TEI_DEPLOY → tar 上传 → 恢复 $TEI_DEPLOY"
        return 0
    fi

    if [ "$FORCE" = "0" ]; then
        if kubectl -n "$INFRA_TEI_NS" get pod -l "app=$TEI_LABEL" --no-headers 2>/dev/null | grep -q Running; then
            local has_config
            has_config=$(kubectl -n "$INFRA_TEI_NS" exec deploy/"$TEI_DEPLOY" -- \
                sh -c "ls $remote_dir/snapshots/*/config.json 2>/dev/null | head -1" 2>/dev/null || echo "")
            if [ -n "$has_config" ]; then
                ok "模型已存在（$has_config），跳过（--force 重传）"
                return 0
            fi
        fi
    fi

    TEI_REPLICAS=$(scale_down "$INFRA_TEI_NS" "$TEI_DEPLOY" "$TEI_LABEL")
    info "已暂停 $TEI_DEPLOY（原 replicas=$TEI_REPLICAS）"

    local pod="upload-model-$(date +%s)"
    launch_upload_pod "$pod" "$INFRA_TEI_NS" "$TEI_PVC" "$TEI_HF_HOME"

    info "清空远端目录: $remote_dir"
    kubectl -n "$INFRA_TEI_NS" exec "$pod" -- mkdir -p "$TEI_HF_HOME/$TEI_HUB_SUBDIR"
    kubectl -n "$INFRA_TEI_NS" exec "$pod" -- rm -rf "$remote_dir"

    info "tar 管道上传（$size）..."
    (cd "$HF_CACHE_DIR/hub" && tar -cf - "$MODEL_DIR_NAME") \
        | kubectl -n "$INFRA_TEI_NS" exec -i "$pod" -- tar -xf - -C "$TEI_HF_HOME/$TEI_HUB_SUBDIR/"

    info "验证模型完整性"
    local config_ok weights_ok rsize
    config_ok=$(kubectl -n "$INFRA_TEI_NS" exec "$pod" -- \
        sh -c "ls $remote_dir/snapshots/*/config.json 2>/dev/null | head -1" 2>/dev/null || echo "")
    weights_ok=$(kubectl -n "$INFRA_TEI_NS" exec "$pod" -- \
        sh -c "ls $remote_dir/snapshots/*/pytorch_model.bin $remote_dir/snapshots/*/model.safetensors 2>/dev/null | head -1" 2>/dev/null || echo "")
    rsize=$(kubectl -n "$INFRA_TEI_NS" exec "$pod" -- du -sh "$remote_dir" 2>/dev/null | cut -f1 || echo "?")

    echo "    本地: $size"
    echo "    远端: $rsize"
    echo "    config.json: ${config_ok:-缺失}"
    echo "    权重文件: ${weights_ok:-缺失}"

    if [ -z "$config_ok" ] || [ -z "$weights_ok" ]; then
        kubectl -n "$INFRA_TEI_NS" exec "$pod" -- find "$remote_dir" -maxdepth 3 -type f 2>/dev/null | head -20 | sed 's/^/    /' || true
        cleanup_upload_pod
        die "模型上传不完整"
    fi

    cleanup_upload_pod
    ok "模型上传完成"

    scale_up "$INFRA_TEI_NS" "$TEI_DEPLOY" "$TEI_REPLICAS"
    TEI_REPLICAS=""

    info "等待 $TEI_DEPLOY Ready（最长 600s）..."
    if ! kubectl -n "$INFRA_TEI_NS" wait --for=condition=available deploy/"$TEI_DEPLOY" --timeout=600s >/dev/null 2>&1; then
        kubectl -n "$INFRA_TEI_NS" get pods -l "app=$TEI_LABEL" -o wide 2>/dev/null | sed 's/^/    /' >&2 || true
        kubectl -n "$INFRA_TEI_NS" logs deploy/"$TEI_DEPLOY" --tail=30 2>/dev/null | sed 's/^/    /' >&2 || true
        die "TEI（$TEI_DEPLOY）恢复后未就绪
  排查: kubectl -n $INFRA_TEI_NS logs deploy/$TEI_DEPLOY --tail=80"
    fi
    ok "TEI（$TEI_DEPLOY）Ready"
}

# ============================================================
# 上传数据
# ============================================================
upload_data() {
    step "上传数据 → $APP_DATA_PVC"

    local src="$DATA_DIR_LOCAL"
    local batch="${BATCH_ARG:-$(basename "$src")}"
    local remote_dir="/app/data/$batch"

    [ -d "$src" ] || die "本地数据目录不存在: $src
  用 --data-dir 指定或检查 config.env 的 DATA_DIR_LOCAL"

    local count
    count=$(find "$src" -maxdepth 1 -name "$CALLS_FILE_PATTERN" | wc -l | tr -d ' ')
    [ "$count" -gt 0 ] || die "$src 下没有匹配 $CALLS_FILE_PATTERN 的文件"

    local size
    size=$(du -sh "$src" | cut -f1)
    info "源: $src ($count 个 $CALLS_FILE_PATTERN, $size)"
    info "目标: $APP_DATA_PVC:$remote_dir/"

    if [ "$DRY_RUN" = "1" ]; then
        info "[dry-run] 暂停 $APP_DEPLOY → tar 上传 → 恢复 → patch DATA_DIR"
        return 0
    fi

    if [ "$FORCE" = "0" ]; then
        if kubectl -n "$NS" get pod -l "app=$APP_LABEL" --no-headers 2>/dev/null | grep -q Running; then
            local rcount
            rcount=$(kubectl -n "$NS" exec deploy/"$APP_DEPLOY" -- \
                sh -c "find $remote_dir -maxdepth 1 -name '$CALLS_FILE_PATTERN' 2>/dev/null | wc -l" 2>/dev/null || echo 0)
            if [ "${rcount:-0}" = "$count" ]; then
                ok "数据已存在（$rcount/$count），跳过（--force 重传）"
                kubectl -n "$NS" patch configmap cca-config --type merge \
                    -p "{\"data\":{\"DATA_DIR\":\"$remote_dir\"}}" >/dev/null
                ok "DATA_DIR=$remote_dir"
                return 0
            fi
        fi
    fi

    APP_REPLICAS=$(scale_down "$NS" "$APP_DEPLOY" "$APP_LABEL")
    APP_WAS_SCALED=1

    local pod="upload-data-$(date +%s)"
    launch_upload_pod "$pod" "$NS" "$APP_DATA_PVC" "/app/data"

    info "清空远端目录: $remote_dir"
    kubectl -n "$NS" exec "$pod" -- mkdir -p "$remote_dir"
    kubectl -n "$NS" exec "$pod" -- \
        sh -c "find $remote_dir -maxdepth 1 -name '$CALLS_FILE_PATTERN' -delete 2>/dev/null || true"

    info "tar 管道上传（$size）..."
    (cd "$src" && tar -cf - .) \
        | kubectl -n "$NS" exec -i "$pod" -- tar -xf - -C "$remote_dir/"

    local rcount rsize
    rcount=$(kubectl -n "$NS" exec "$pod" -- \
        sh -c "find $remote_dir -maxdepth 1 -name '$CALLS_FILE_PATTERN' | wc -l" | tr -d ' ')
    rsize=$(kubectl -n "$NS" exec "$pod" -- du -sh "$remote_dir" 2>/dev/null | cut -f1 || echo "?")

    echo "    本地: $count 文件 / $size"
    echo "    远端: $rcount 文件 / $rsize"

    if [ "$rcount" -ne "$count" ]; then
        kubectl -n "$NS" exec "$pod" -- ls -la "$remote_dir" || true
        cleanup_upload_pod
        die "文件数不一致：本地 $count 远端 $rcount"
    fi

    local sample
    sample=$(kubectl -n "$NS" exec "$pod" -- \
        sh -c "head -c 150 \$(find $remote_dir -maxdepth 1 -name '$CALLS_FILE_PATTERN' | head -1)" 2>/dev/null | tr -d '\n' || echo "")
    [ -n "$sample" ] && echo "    样例: ${sample}..."
    ok "数据上传完成 ($rcount 个)"

    cleanup_upload_pod

    scale_up "$NS" "$APP_DEPLOY" "$APP_REPLICAS"
    APP_WAS_SCALED=0

    info "更新 ConfigMap DATA_DIR=$remote_dir"
    kubectl -n "$NS" patch configmap cca-config --type merge \
        -p "{\"data\":{\"DATA_DIR\":\"$remote_dir\"}}" >/dev/null
    ok "DATA_DIR=$remote_dir"

    # 顺带同步其他配置（sync_app_config 是幂等的）
    sync_app_config "$NS" "cca-config"
}

# ============================================================
# 主流程
# ============================================================
echo "============================================================"
echo "  CCA 上传"
echo "============================================================"
echo "  MODE       : $MODE"
echo "  NS         : $NS"
echo "  TEI_NS     : $INFRA_TEI_NS"
echo "  TEI_DEPLOY : $TEI_DEPLOY"
echo "  IMAGE      : $IMAGE"
echo "  FORCE      : $FORCE"
echo "  DRY_RUN    : $DRY_RUN"
echo "============================================================"

case "$MODE" in
    model) upload_model ;;
    data)  upload_data ;;
    all)   upload_model; upload_data ;;
esac

echo ""
echo "============================================================"
if [ "$DRY_RUN" = "1" ]; then
    echo "  ✅ dry-run 完成"
else
    echo "  ✅ 上传完成"
    echo ""
    echo "  下一步: ./offline.sh"
fi
echo "============================================================"