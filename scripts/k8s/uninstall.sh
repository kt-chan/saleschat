#!/usr/bin/env bash
# scripts/k8s/uninstall.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/config.env" ] && { set -a; source "$SCRIPT_DIR/config.env"; set +a; }
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

NS="${NS:-cca}"
INFRA_PG_NS="${INFRA_PG_NS:-gv}"
INFRA_TEI_NS="${INFRA_TEI_NS:-model-service}"
PF_PID_FILE="$HOME/.cca-port-forward.pid"

PURGE=0; ALL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --purge) PURGE=1; shift ;;
        --all)   ALL=1; PURGE=1; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

echo "=========================================="
echo "  CCA 卸载（仅 cca namespace）"
echo "=========================================="
echo "  $NS            : 会被清理"
echo "  $INFRA_PG_NS            : 共享 PG，不触碰"
echo "  $INFRA_TEI_NS : 共享 TEI，不触碰"
if [ "$ALL" = 1 ]; then echo "  模式           : ALL";
elif [ "$PURGE" = 1 ]; then echo "  模式           : PURGE";
else echo "  模式           : DEFAULT"; fi
echo "=========================================="

# 停 PF
stop_port_forward 2>/dev/null || true
pkill -f "kubectl.*port-forward.*-n $NS.*cca-app" 2>/dev/null || true

if [ "$ALL" = 1 ]; then
    kubectl get ns "$NS" >/dev/null 2>&1 && kubectl delete ns "$NS" --wait=true --timeout=180s || true
    echo "✅ cca namespace 已删除"
    echo "   $INFRA_PG_NS / $INFRA_TEI_NS 保持运行"
    exit 0
fi

kubectl get ns "$NS" >/dev/null 2>&1 || { echo "namespace $NS 不存在，无需清理"; exit 0; }

echo "==> 停止 cca workload / svc"
kubectl -n "$NS" delete deploy,sts,ds,rs,job --all --wait=true --timeout=120s 2>/dev/null || true
kubectl -n "$NS" delete pod --all --wait=true --timeout=60s 2>/dev/null || true
kubectl -n "$NS" delete svc --all --wait=true --timeout=30s 2>/dev/null || true

if [ "$PURGE" = 1 ]; then
    echo "==> 删除 cca 的 PVC / config / secret"
    kubectl -n "$NS" delete pvc --all --wait=true --timeout=120s 2>/dev/null || true
    kubectl -n "$NS" delete configmap cca-config cca-prompts --ignore-not-found 2>/dev/null || true
    kubectl -n "$NS" delete secret cca-secret --ignore-not-found 2>/dev/null || true
else
    echo "==> 保留 PVC / config / secret（--purge 清除）"
fi

echo ""
echo "=========================================="
echo "  ✅ 清理完成"
echo "=========================================="
for n in "$NS" "$INFRA_PG_NS" "$INFRA_TEI_NS"; do
    if kubectl get ns "$n" >/dev/null 2>&1; then
        pods=$(kubectl -n "$n" get pods --no-headers 2>/dev/null | wc -l | tr -d ' ')
        pvcs=$(kubectl -n "$n" get pvc --no-headers 2>/dev/null | wc -l | tr -d ' ')
        note=""
        [ "$n" = "$INFRA_PG_NS" ]  && note=" ← 未触碰"
        [ "$n" = "$INFRA_TEI_NS" ] && note=" ← 未触碰"
        printf "    %-14s pods=%-2s pvc=%-2s%s\n" "$n" "$pods" "$pvcs" "$note"
    fi
done