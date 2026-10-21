#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-models.txt}"
CACHE_DIR="${HF_HOME:?HF_HOME must be set}"

# 构建阶段必须允许联网；父层若残留离线 ENV 会阻断下载
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_PROGRESS_BARS=1

# token 优先级：已有 $HF_TOKEN > Docker secret
if [[ -z "${HF_TOKEN:-}" && -f /run/secrets/HF_TOKEN ]]; then
    HF_TOKEN="$(< /run/secrets/HF_TOKEN)"
    export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] && echo "hf token  : loaded" || echo "hf token  : not set"

echo "cache dir : $CACHE_DIR"
mkdir -p "$CACHE_DIR"

mapfile -t REPOS < <(
    grep -vE '^\s*(#|$)' "$MANIFEST" \
    | sed -E 's/[[:space:]]*#.*$//' \
    | awk 'NF'
)
(( ${#REPOS[@]} )) || { echo "empty manifest: $MANIFEST" >&2; exit 2; }

echo "models    : ${#REPOS[@]}"
for repo in "${REPOS[@]}"; do
    echo ">>> $repo"
    if [[ "$repo" == *"@"* ]]; then
        hf download "${repo%@*}" --revision "${repo##*@}" --cache-dir "$CACHE_DIR"
    else
        hf download "$repo" --cache-dir "$CACHE_DIR"
    fi
done

echo "all models ready under $CACHE_DIR"