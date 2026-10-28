#!/usr/bin/env bash
#
# run.sh — CCA Kubernetes 部署
#
#   ./run.sh                # 完整部署：render + upload + up + offline + pf（首次必用）
#   ./run.sh online         # 只部署服务：render + up + pf（迭代时用，跳过 upload 和 offline）
#   ./run.sh render         # 只渲染 manifests/runtime/
#   ./run.sh upload         # 上传模型/数据到 PVC
#   ./run.sh up             # 部署服务
#   ./run.sh offline        # 跑离线建库 Job
#   ./run.sh pf             # 起 port-forward
#   ./run.sh pf-stop        # 停 port-forward
#   ./run.sh status         # 查看状态
#
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$DIR"

[ -f .env ] || { echo "ERROR: .env 不存在，先执行: cp .env.example .env && vim .env" >&2; exit 1; }
set -a; . ./.env; set +a

# ---------- 默认值 ----------
: "${NS:=cca}" "${INFRA_PG_NS:=gv}" "${INFRA_TEI_NS:=model-service}"
: "${INFRA_PG_SVC:=gv-sql-service}" "${INFRA_PG_PORT:=14003}"
: "${INFRA_TEI_SVC:=tei-svc}" "${INFRA_TEI_DEPLOY:=tei-svc}" "${INFRA_TEI_LABEL:=tei}" "${INFRA_TEI_PORT:=8010}"
: "${APP_IMAGE:=ghcr.io/kt-chan/saleschat-amd64:latest}"
: "${TEI_IMAGE:=ghcr.io/huggingface/text-embeddings-inference:89-1.9}"
: "${IMAGE_PULL_POLICY:=Never}" "${STORAGE_CLASS:=standard}"
: "${PF_PORT:=8000}" "${PF_ADDRESS:=127.0.0.1}"
: "${MODEL_ID:=BAAI/bge-base-zh-v1.5}"
: "${MODEL_DIR_NAME:=models--BAAI--bge-base-zh-v1.5}"
: "${MODEL_SHORT_NAME:=bge-base-zh-v1.5}"
: "${HF_CACHE_DIR:=$HOME/app/saleschat/models/huggingface}"
: "${DATA_DIR_LOCAL:=$HOME/app/saleschat/data/202610}"
: "${DATA_DIR_BATCH:=$(basename "$DATA_DIR_LOCAL")}"
: "${CALLS_FILE_PATTERN:=*.jsonl}"

TPL="$DIR/manifests/template"
OUT="$DIR/manifests/runtime"
PF_PID="$DIR/.pf.pid"; PF_LOG="$DIR/.pf.log"

L() { echo "[$(date +%H:%M:%S)] $*"; }
W() { echo "[$(date +%H:%M:%S)] WARN: $*" >&2; }

# ---------- render ----------
_render_file() {
    local f=$1 d=$2
    if command -v envsubst >/dev/null 2>&1; then
        envsubst < "$f" > "$d"; return
    fi
    local c p v
    c=$(<"$f")
    for p in $(grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*\}' "$f" | sort -u); do
        v="${p#\$\{}"; v="${v%\}}"
        c="${c//$p/${!v-}}"
    done
    printf '%s\n' "$c" > "$d"
}
render() {
    L "render → manifests/runtime/"
    [ -d "$TPL" ] || { echo "ERROR: 模板目录不存在: $TPL" >&2; exit 1; }
    rm -rf "$OUT"
    mkdir -p "$OUT/upload"
    local f
    for f in "$TPL"/*.yaml;        do if [ -f "$f" ]; then _render_file "$f" "$OUT/$(basename "$f")";        fi; done
    for f in "$TPL"/upload/*.yaml; do if [ -f "$f" ]; then _render_file "$f" "$OUT/upload/$(basename "$f")"; fi; done
    L "  ✅ $(find "$OUT" -name '*.yaml' | wc -l | tr -d ' ') 文件"
}

# ---------- upload ----------
upload() {
    L "upload"

    # ---------- 1. 探测现状 ----------
    local need_model=1 need_data=1
    local cnt_local=0 cnt_remote=0

    # 模型：exec TEI pod 检查扁平路径下关键文件
    if kubectl -n "$INFRA_TEI_NS" get deploy "$INFRA_TEI_DEPLOY" >/dev/null 2>&1; then
        local r
        r=$(kubectl -n "$INFRA_TEI_NS" get deploy "$INFRA_TEI_DEPLOY" \
            -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
        case "${r:-}" in ""|"null") r=0;; esac
        if [ "$r" -ge 1 ] && \
           kubectl -n "$INFRA_TEI_NS" exec deploy/"$INFRA_TEI_DEPLOY" -- \
               test -f "/data/models/$MODEL_SHORT_NAME/config.json" >/dev/null 2>&1; then
            need_model=0
            L "  模型已就绪 ($MODEL_SHORT_NAME) → 跳过"
        fi
    fi

    # 数据：exec app pod 比对文件数
    [ -d "$DATA_DIR_LOCAL" ] && \
        cnt_local=$(find "$DATA_DIR_LOCAL" -maxdepth 1 -name "$CALLS_FILE_PATTERN" | wc -l | tr -d ' ')
    if [ "$cnt_local" -gt 0 ] && kubectl -n "$NS" get deploy cca-app >/dev/null 2>&1; then
        local r
        r=$(kubectl -n "$NS" get deploy cca-app \
            -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
        case "${r:-}" in ""|"null") r=0;; esac
        if [ "$r" -ge 1 ]; then
            cnt_remote=$(kubectl -n "$NS" exec deploy/cca-app -- \
                sh -c "find /app/data/$DATA_DIR_BATCH -maxdepth 1 -name '$CALLS_FILE_PATTERN' 2>/dev/null | wc -l" \
                2>/dev/null | tr -d ' ' || echo 0)
            if [ "${cnt_remote:-0}" = "$cnt_local" ]; then
                need_data=0
                L "  数据已就绪 ($cnt_remote/$cnt_local) → 跳过"
            fi
        fi
    fi

    # 都就绪：不动任何服务
    if [ "$need_model" = "0" ] && [ "$need_data" = "0" ]; then
        L "  ✅ 模型和数据均已在 PVC，跳过 upload"
        return 0
    fi

    # ---------- 2. 只停需要的服务 ----------
    kubectl apply -f "$OUT/00-namespace.yaml" -f "$OUT/01-pvc.yaml" >/dev/null

    local orig_app=0 orig_tei=0

    # 上传数据 → 需要停 app 释放 app-data
    if [ "$need_data" = "1" ]; then
        if kubectl -n "$NS" get deploy cca-app >/dev/null 2>&1; then
            orig_app=$(kubectl -n "$NS" get deploy cca-app -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
        fi
        case "${orig_app:-}" in ""|"null") orig_app=0;; esac
        if [ "$orig_app" -gt 0 ]; then
            L "  暂停 app ($orig_app → 0，为上传数据)"
            kubectl -n "$NS" scale deploy/cca-app --replicas=0 >/dev/null
            kubectl -n "$NS" wait --for=delete pod -l app=cca-app --timeout=120s 2>/dev/null || true
        fi
    fi

    # 上传模型 → 需要停 TEI 释放 tei-data
    if [ "$need_model" = "1" ]; then
        if kubectl -n "$INFRA_TEI_NS" get deploy "$INFRA_TEI_DEPLOY" >/dev/null 2>&1; then
            orig_tei=$(kubectl -n "$INFRA_TEI_NS" get deploy "$INFRA_TEI_DEPLOY" \
                -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
        fi
        case "${orig_tei:-}" in ""|"null") orig_tei=0;; esac
        if [ "$orig_tei" -gt 0 ]; then
            L "  暂停 TEI ($orig_tei → 0，为上传模型)"
            kubectl -n "$INFRA_TEI_NS" scale deploy/"$INFRA_TEI_DEPLOY" --replicas=0 >/dev/null
            kubectl -n "$INFRA_TEI_NS" wait --for=delete pod -l "app=$INFRA_TEI_LABEL" --timeout=120s 2>/dev/null || true
        fi
    fi

    # ---------- 3. 只起需要的 upload pod ----------
    L "  起 upload pods"
    if [ "$need_model" = "1" ]; then
        kubectl -n "$INFRA_TEI_NS" delete pod tei-upload --ignore-not-found --wait=true >/dev/null 2>&1 || true
        kubectl apply -f "$OUT/upload/tei-upload.yaml" >/dev/null
    fi
    if [ "$need_data" = "1" ]; then
        kubectl -n "$NS" delete pod app-upload --ignore-not-found --wait=true >/dev/null 2>&1 || true
        kubectl apply -f "$OUT/upload/app-upload.yaml" >/dev/null
    fi
    if [ "$need_model" = "1" ]; then
        kubectl -n "$INFRA_TEI_NS" wait --for=condition=Ready pod/tei-upload --timeout=180s >/dev/null
    fi
    if [ "$need_data" = "1" ]; then
        kubectl -n "$NS" wait --for=condition=Ready pod/app-upload --timeout=180s >/dev/null
    fi

    # ---------- 4. 上传模型 ----------
    if [ "$need_model" = "1" ]; then
        if [ -d "$HF_CACHE_DIR/hub/$MODEL_DIR_NAME" ]; then
            local snap_dir="$HF_CACHE_DIR/hub/$MODEL_DIR_NAME"
            local snap_hash=""
            if [ -f "$snap_dir/refs/main" ]; then
                snap_hash=$(cat "$snap_dir/refs/main" | tr -d '\n')
            fi
            if [ -z "$snap_hash" ] || [ ! -d "$snap_dir/snapshots/$snap_hash" ]; then
                snap_hash=$(ls "$snap_dir/snapshots/" 2>/dev/null | head -1 || echo "")
            fi

            if [ -n "$snap_hash" ] && [ -d "$snap_dir/snapshots/$snap_hash" ]; then
                local src="$snap_dir/snapshots/$snap_hash"
                L "  模型: $MODEL_SHORT_NAME (snapshot $snap_hash, $(du -sh "$src" 2>/dev/null | cut -f1))"
                kubectl -n "$INFRA_TEI_NS" exec tei-upload -- mkdir -p "/data/models/$MODEL_SHORT_NAME"
                kubectl -n "$INFRA_TEI_NS" exec tei-upload -- \
                    sh -c "rm -rf /data/models/$MODEL_SHORT_NAME/* 2>/dev/null || true"
                # tar -h：解引用 HF 缓存里的软链
                tar -chf - -C "$src" . \
                    | kubectl -n "$INFRA_TEI_NS" exec -i tei-upload -- \
                        tar -xf - -C "/data/models/$MODEL_SHORT_NAME/"
                # 校验
                local has_cfg has_w
                has_cfg=$(kubectl -n "$INFRA_TEI_NS" exec tei-upload -- \
                    test -f "/data/models/$MODEL_SHORT_NAME/config.json" && echo y || echo n)
                has_w=$(kubectl -n "$INFRA_TEI_NS" exec tei-upload -- \
                    sh -c "test -f /data/models/$MODEL_SHORT_NAME/pytorch_model.bin -o -f /data/models/$MODEL_SHORT_NAME/model.safetensors && echo y || echo n")
                [ "$has_cfg" = "y" ] && [ "$has_w" = "y" ] \
                    || W "模型不完整: config=$has_cfg weights=$has_w"
                kubectl -n "$INFRA_TEI_NS" exec tei-upload -- \
                    du -sh "/data/models/$MODEL_SHORT_NAME"
            else
                W "找不到模型的 snapshot: $snap_dir"
            fi
        else
            W "本地模型不存在: $HF_CACHE_DIR/hub/$MODEL_DIR_NAME（跳过）"
        fi
    fi

    # ---------- 5. 上传数据 ----------
    if [ "$need_data" = "1" ]; then
        if [ -d "$DATA_DIR_LOCAL" ]; then
            local b="$DATA_DIR_BATCH" cnt="$cnt_local" rc
            if [ "$cnt" -gt 0 ]; then
                L "  数据: → /app/data/$b/ ($cnt 个)"
                kubectl -n "$NS" exec app-upload -- mkdir -p "/app/data/$b"
                kubectl -n "$NS" exec app-upload -- sh -c \
                    "find /app/data/$b -maxdepth 1 -name '$CALLS_FILE_PATTERN' -delete 2>/dev/null || true"
                tar -cf - -C "$DATA_DIR_LOCAL" . \
                    | kubectl -n "$NS" exec -i app-upload -- tar -xf - -C "/app/data/$b/"
                rc=$(kubectl -n "$NS" exec app-upload -- \
                    sh -c "find /app/data/$b -maxdepth 1 -name '$CALLS_FILE_PATTERN' | wc -l" | tr -d ' ')
                [ "$rc" = "$cnt" ] || W "文件数不一致: 本地 $cnt / 远端 $rc"
            else
                W "$DATA_DIR_LOCAL 下无 $CALLS_FILE_PATTERN"
            fi
        else
            W "本地数据目录不存在: $DATA_DIR_LOCAL（跳过）"
        fi
    fi

    # ---------- 6. 清理 + 恢复 ----------
    [ "$need_model" = "1" ] && kubectl -n "$INFRA_TEI_NS" delete pod tei-upload --wait=true >/dev/null 2>&1 || true
    [ "$need_data" = "1" ]  && kubectl -n "$NS" delete pod app-upload --wait=true >/dev/null 2>&1 || true

    if [ "$orig_tei" -gt 0 ]; then
        L "  恢复 TEI ($orig_tei)"
        kubectl -n "$INFRA_TEI_NS" scale deploy/"$INFRA_TEI_DEPLOY" --replicas="$orig_tei" >/dev/null
    fi
    if [ "$orig_app" -gt 0 ]; then
        L "  恢复 app ($orig_app)"
        kubectl -n "$NS" scale deploy/cca-app --replicas="$orig_app" >/dev/null
    fi
    L "  ✅"
}

# ---------- up ----------
up() {
    L "apply manifests"

    # 记录 apply 前的 ConfigMap/Secret hash
    local hash_before=""
    if kubectl -n "$NS" get cm cca-config >/dev/null 2>&1; then
        hash_before=$(kubectl -n "$NS" get cm cca-config -o yaml 2>/dev/null | sha256sum | cut -d' ' -f1)
        hash_before="$hash_before:$(kubectl -n "$NS" get secret cca-secret -o yaml 2>/dev/null | sha256sum | cut -d' ' -f1)"
    fi

    local f
    for f in 00-namespace 01-pvc 02-secret 03-configmap 04-postgres 05-tei 06-app; do
        kubectl apply -f "$OUT/$f.yaml" >/dev/null
    done
    if [ -f "$DIR/../../prompts.yaml" ]; then
        kubectl -n "$NS" create configmap cca-prompts \
            --from-file=prompts.yaml="$DIR/../../prompts.yaml" \
            --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    fi

    # 只在配置变化时 restart
    local hash_after=""
    if kubectl -n "$NS" get cm cca-config >/dev/null 2>&1; then
        hash_after=$(kubectl -n "$NS" get cm cca-config -o yaml 2>/dev/null | sha256sum | cut -d' ' -f1)
        hash_after="$hash_after:$(kubectl -n "$NS" get secret cca-secret -o yaml 2>/dev/null | sha256sum | cut -d' ' -f1)"
    fi

    if [ "$hash_before" != "$hash_after" ] || [ -z "$hash_before" ]; then
        L "  配置有变化 → 重启 app"
        kubectl -n "$NS" rollout restart deploy/cca-app >/dev/null
    else
        L "  配置未变 → 跳过 app 重启"
    fi

    # 等待就绪
    L "  等待 postgres"
    kubectl -n "$INFRA_PG_NS" wait --for=condition=ready pod/postgres-0 --timeout=180s >/dev/null
    L "  等待 TEI"
    kubectl -n "$INFRA_TEI_NS" wait --for=condition=available deploy/"$INFRA_TEI_DEPLOY" --timeout=600s >/dev/null
    L "  等待 app"
    kubectl -n "$NS" rollout status deploy/cca-app --timeout=300s
    L "  ✅"
}

# ---------- pf ----------
pf_stop() {
    local p
    if [ -f "$PF_PID" ]; then
        p=$(cat "$PF_PID" 2>/dev/null || true)
        [ -n "$p" ] && kill "$p" 2>/dev/null || true
        rm -f "$PF_PID"
    fi
    pkill -f "kubectl.*port-forward.*-n $NS.*cca-app" 2>/dev/null || true
}
pf() {
    pf_stop
    L "port-forward $PF_ADDRESS:$PF_PORT"
    : > "$PF_LOG"
    nohup kubectl -n "$NS" port-forward --address "$PF_ADDRESS" \
        svc/cca-app "$PF_PORT:8000" >> "$PF_LOG" 2>&1 &
    echo $! > "$PF_PID"
    local i
    for i in $(seq 1 30); do
        if curl -sf -o /dev/null --max-time 2 "http://$PF_ADDRESS:$PF_PORT/health" 2>/dev/null; then
            L "  ✅ http://$PF_ADDRESS:$PF_PORT"
            return
        fi
        sleep 1
    done
    W "pf 30s 未就绪，日志: $PF_LOG"
}

# ---------- offline ----------
offline() {
    pf_stop
    local orig=1
    if kubectl -n "$NS" get deploy cca-app >/dev/null 2>&1; then
        orig=$(kubectl -n "$NS" get deploy cca-app -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 1)
    fi
    case "${orig:-}" in ""|"null") orig=1;; esac

    L "  暂停 app ($orig → 0)"
    kubectl -n "$NS" scale deploy/cca-app --replicas=0 2>/dev/null || true
    kubectl -n "$NS" wait --for=delete pod -l app=cca-app --timeout=120s 2>/dev/null || true

    L "  运行 Job"
    kubectl -n "$NS" delete job cca-offline --ignore-not-found --wait=true >/dev/null 2>&1 || true
    kubectl apply -f "$OUT/07-offline-job.yaml" >/dev/null
    kubectl -n "$NS" logs -f job/cca-offline 2>/dev/null || true

    local ok=0
    kubectl -n "$NS" wait --for=condition=complete job/cca-offline --timeout=1800s >/dev/null 2>&1 \
        && ok=1 || W "Job 失败/超时"

    L "  恢复 app ($orig)"
    kubectl -n "$NS" scale deploy/cca-app --replicas="$orig" >/dev/null
    kubectl -n "$NS" rollout status deploy/cca-app --timeout=180s >/dev/null 2>&1 || true
    pf || true

    [ "$ok" = "1" ] && L "  ✅" || return 1
}

# ---------- status ----------
status() {
    local n
    for n in "$INFRA_PG_NS" "$INFRA_TEI_NS" "$NS"; do
        echo ""; echo "=== $n ==="
        kubectl -n "$n" get pods,svc 2>/dev/null || echo "  (无)"
    done
    echo ""
    if [ -f "$PF_PID" ] && kill -0 "$(cat "$PF_PID" 2>/dev/null || echo 0)" 2>/dev/null; then
        echo "PF: Running → http://$PF_ADDRESS:$PF_PORT"
    else
        echo "PF: Not running"
    fi
}

# ---------- 分发 ----------
case "${1:-all}" in
    all)      render; upload; up; offline; pf ;;   # 首次：完整部署 + 建库
    online)   render; up; pf ;;                    # 迭代：只部署服务
    render)   render ;;
    upload)   render; upload ;;
    up)       render; up ;;
    offline)  render; offline ;;
    pf)       pf ;;
    pf-stop)  pf_stop; L "pf 已停" ;;
    status)   status ;;
    help|-h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) echo "用法: $0 [all|online|render|upload|up|offline|pf|pf-stop|status]"; exit 1 ;;
esac