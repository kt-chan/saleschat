#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-models.txt}"

# HF_HOME 是根目录；hf CLI 的 --cache-dir 要的是 hub cache 本身
HF_HOME="${HF_HOME:?HF_HOME must be set}"
HUB_CACHE="${HF_HOME}/hub"

export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_PROGRESS_BARS=1

if [[ -z "${HF_TOKEN:-}" && -f /run/secrets/HF_TOKEN ]]; then
    HF_TOKEN="$(< /run/secrets/HF_TOKEN)"
    export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] && echo "hf token  : loaded" || echo "hf token  : not set (public repos only)"

echo "hf home   : $HF_HOME"
echo "hub cache : $HUB_CACHE"
mkdir -p "$HUB_CACHE"

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
        hf download "${repo%@*}" --revision "${repo##*@}" --cache-dir "$HUB_CACHE"
    else
        hf download "$repo" --cache-dir "$HUB_CACHE"
    fi
done

echo "all models ready under $HUB_CACHE"