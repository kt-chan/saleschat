#!/usr/bin/env bash
#
# stop.sh — 停止 / 清理
#
#   ./stop.sh           # 删 workload（保留 PVC / config / secret）
#   ./stop.sh --purge   # 额外删 PVC（数据丢失）
#   ./stop.sh --all     # 删整个 namespace（最彻底）
#
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$DIR"

[ -f .env ] || { echo "ERROR: .env 不存在" >&2; exit 1; }
set -a; . ./.env; set +a

: "${NS:=cca}" "${INFRA_PG_NS:=gv}" "${INFRA_TEI_NS:=model-service}"

OUT="$DIR/manifests/runtime"
PF_PID="$DIR/.pf.pid"

L() { echo "[$(date +%H:%M:%S)] $*"; }

# 停 pf
if [ -f "$PF_PID" ]; then
    p=$(cat "$PF_PID" 2>/dev/null || true)
    [ -n "$p" ] && kill "$p" 2>/dev/null || true
    rm -f "$PF_PID"
fi
pkill -f "kubectl.*port-forward.*-n $NS.*cca-app" 2>/dev/null || true

_del() {
    local f
    for f in "$@"; do
        if [ -f "$OUT/$f.yaml" ]; then
            kubectl delete -f "$OUT/$f.yaml" --ignore-not-found >/dev/null 2>&1 || true
        fi
    done
}

case "${1:-default}" in
    --purge|purge)
        L "删除所有（含 PVC）"
        _del 07-offline-job 06-app 05-tei 04-postgres 03-configmap 02-secret 01-pvc
        rm -rf "$OUT"
        ;;
    --all|all)
        L "删除 namespace"
        for n in "$NS" "$INFRA_PG_NS" "$INFRA_TEI_NS"; do
            if kubectl get ns "$n" >/dev/null 2>&1; then
                kubectl delete ns "$n" --wait=true --timeout=180s
            fi
        done
        rm -rf "$OUT"
        ;;
    default|"")
        L "删除 workload（保留 PVC）"
        _del 06-app 05-tei 04-postgres 03-configmap 02-secret
        ;;
    *)
        echo "用法: $0 [--purge|--all]"; exit 1
        ;;
esac
L "✅"