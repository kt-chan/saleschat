#!/usr/bin/env bash
# scripts/k8s/status.sh —— 部署状态一览
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/config.env" ] && { set -a; source "$SCRIPT_DIR/config.env"; set +a; }

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
APP_DEPLOY="cca-app"
APP_LABEL="cca-app"
PF_PID_FILE="$HOME/.cca-port-forward.pid"

echo "=========================================="
echo "  CCA 状态总览"
echo "=========================================="

for n in "$INFRA_PG_NS" "$INFRA_TEI_NS" "$NS"; do
    echo ""
    echo "--- $n pods ---"
    kubectl -n "$n" get pods -o wide 2>/dev/null | sed 's/^/  /' || echo "  (namespace 不存在)"
done

echo ""
echo "--- infra services ---"
for svc_ns_pair in "$INFRA_PG_NS:$INFRA_PG_SVC:$INFRA_PG_PORT" "$INFRA_TEI_NS:$INFRA_TEI_SVC:$INFRA_TEI_PORT"; do
    IFS=':' read -r sn sc sp <<< "$svc_ns_pair"
    port=$(kubectl -n "$sn" get svc "$sc" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "?")
    ep=$(kubectl -n "$sn" get endpoints "$sc" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null | wc -w | tr -d ' ')
    printf "  %s/%s: port=%s endpoints=%s\n" "$sn" "$sc" "$port" "${ep:-0}"
done

echo ""
echo "--- app env ---"
POD=$(kubectl -n "$NS" get pod -l "app=$APP_LABEL" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -n "$POD" ]; then
    kubectl -n "$NS" exec "$POD" -- sh -c '
echo "  DB_HOST           = $DB_HOST"
echo "  DB_PORT           = $DB_PORT"
echo "  EMBEDDING_TEI_URL = $EMBEDDING_TEI_URL"
echo "  LLM_BASE_URL      = $LLM_BASE_URL"
echo "  LLM_MODEL         = $LLM_MODEL"
echo "  DATA_DIR          = $DATA_DIR"
' 2>/dev/null || echo "  (exec 失败)"
else
    echo "  ($APP_DEPLOY Pod 不存在)"
fi

echo ""
echo "--- 健康检查 ---"
if [ -n "$POD" ]; then
    DB_OK=$(kubectl -n "$NS" exec "$POD" -- python -c 'from app.utils import get_db_conn,release_db_conn;c=get_db_conn();c.cursor().execute("SELECT 1");release_db_conn(c);print("OK")' 2>/dev/null | tail -1 || echo "FAIL")
    echo "  DB: $DB_OK"

    TEI_OK=$(kubectl -n "$NS" exec "$POD" -- python -c 'from app.utils import get_embedder;v=get_embedder().encode("ping");print(f"OK dims={len(v)}")' 2>/dev/null | tail -1 || echo "FAIL")
    echo "  TEI: $TEI_OK"
fi

echo ""
echo "--- port-forward ---"
if [ -f "$PF_PID_FILE" ]; then
    PID=$(cat "$PF_PID_FILE" 2>/dev/null || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "  Running (PID=$PID)"
        echo "  Log: $HOME/.cca-port-forward.log"
    else
        echo "  Stopped (stale PID file)"
    fi
else
    echo "  Not started"
fi

echo ""
echo "=========================================="