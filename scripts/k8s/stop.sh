#!/usr/bin/env bash
#
# stop.sh — 停止 / 清理
#
# 安全约束：绝不删除 INFRA_PG_NS / INFRA_TEI_NS 的 namespace 或其中的资源。
#          如需清理本地测试用的 PG/TEI，请手动执行（见文末注释）。
#
#   ./stop.sh           # 停 cca-app workload（保留 PVC / config / secret）
#   ./stop.sh --purge   # 停 cca-app + 删 cca 的 PVC / config / secret（cca 内数据丢失）
#   ./stop.sh --all     # 删除 cca namespace 本身（cca 内所有资源都删）
#
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$DIR"

[ -f .env ] || { echo "ERROR: .env 不存在" >&2; exit 1; }
set -a; . ./.env; set +a

: "${NS:=cca}" "${INFRA_PG_NS:=gv}" "${INFRA_TEI_NS:=model-service}"

OUT="$DIR/manifests/runtime"
PF_PID="$DIR/.pf.pid"

L() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# 安全守护：确认 infra namespace 永不被本脚本触碰
# ============================================================
_guard_infra() {
    if [ "$NS" = "$INFRA_PG_NS" ] || [ "$NS" = "$INFRA_TEI_NS" ]; then
        echo "ERROR: NS ($NS) 不能与 INFRA_PG_NS ($INFRA_PG_NS) 或 INFRA_TEI_NS ($INFRA_TEI_NS) 相同" >&2
        echo "       本脚本永不操作 infra namespace，请修改 .env 的 NS" >&2
        exit 1
    fi
}
_guard_infra

# 停 pf
if [ -f "$PF_PID" ]; then
    p=$(cat "$PF_PID" 2>/dev/null || true)
    [ -n "$p" ] && kill "$p" 2>/dev/null || true
    rm -f "$PF_PID"
fi
pkill -f "kubectl.*port-forward.*-n $NS.*cca-app" 2>/dev/null || true

# 只在 cca namespace 内操作，杜绝误伤
_del_in_app() {
    local f
    for f in "$@"; do
        if [ -f "$OUT/$f.yaml" ]; then
            # 强制指定 namespace 为 $NS，即使 yaml 里写错也只影响 cca
            kubectl -n "$NS" delete -f "$OUT/$f.yaml" --ignore-not-found >/dev/null 2>&1 || true
        fi
    done
}

echo "=========================================="
echo "  CCA 停止 / 清理"
echo "=========================================="
echo "  操作范围   : $NS"
echo "  保护范围   : $INFRA_PG_NS（PG） / $INFRA_TEI_NS（TEI）— 永不触碰"
case "${1:-default}" in
    --purge|purge) echo "  模式       : PURGE（删 cca 内 workload + PVC + config）" ;;
    --all|all)     echo "  模式       : ALL（删 cca namespace）" ;;
    *)             echo "  模式       : DEFAULT（删 cca 内 workload）" ;;
esac
echo "=========================================="

case "${1:-default}" in
    --purge|purge)
        L "删除 cca 内所有资源（含 PVC）"
        _del_in_app 07-offline-job 06-app 03-configmap 02-secret-app 01-pvc-app
        kubectl -n "$NS" delete configmap cca-prompts --ignore-not-found >/dev/null 2>&1 || true
        L "  ✅ cca 内已清空（PVC 数据丢失）"
        L "  ⚠️  $INFRA_PG_NS / $INFRA_TEI_NS 未触碰"
        ;;
    --all|all)
        L "删除 cca namespace"
        if kubectl get ns "$NS" >/dev/null 2>&1; then
            kubectl delete ns "$NS" --wait=true --timeout=180s
            L "  ✅ namespace $NS 已删除"
        else
            L "  namespace $NS 不存在"
        fi
        L "  ⚠️  $INFRA_PG_NS / $INFRA_TEI_NS 未触碰"
        ;;
    default|"")
        L "删除 cca 内 workload（保留 PVC / config / secret）"
        _del_in_app 06-app
        L "  ✅ cca workload 已停（app-data / app-output / cca-config / cca-secret 保留）"
        L "  ⚠️  $INFRA_PG_NS / $INFRA_TEI_NS 未触碰"
        ;;
    *)
        echo "用法: $0 [--purge|--all]" >&2
        echo "" >&2
        echo "  ./stop.sh            删除 cca 内 workload（保留数据）" >&2
        echo "  ./stop.sh --purge    删除 cca 内所有资源（含 PVC）" >&2
        echo "  ./stop.sh --all      删除 cca namespace 本身" >&2
        echo "" >&2
        echo "  本脚本永不删除 $INFRA_PG_NS / $INFRA_TEI_NS。" >&2
        echo "  如需清理本地测试的 PG/TEI，请手动执行：" >&2
        echo "    kubectl delete ns $INFRA_PG_NS $INFRA_TEI_NS" >&2
        exit 1
        ;;
esac

L "✅"