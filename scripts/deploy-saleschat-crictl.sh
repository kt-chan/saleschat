#!/usr/bin/env bash
#
# deploy-saleschat-crictl.sh
#
# Import a pre-built Docker image archive into the CRI runtime
# (containerd or CRI-O) and deploy it via crictl using generated
# pod-config.yaml / container-config.yaml.
#
# ENVIRONMENT HANDLING
#   By default we do NOT inline .env. The app is expected to load
#   <project_root>/.env itself, e.g. at the top of app/main.py:
#
#       from pathlib import Path
#       PROJECT_ROOT = Path(__file__).resolve().parent.parent
#       _env = PROJECT_ROOT / ".env"
#       if _env.is_file():
#           from dotenv import load_dotenv
#           load_dotenv(_env)          # override=False → YAML envs: win
#
#   Only the four operational overrides that compose's `environment:`
#   block would inject (TZ, HF_HOME, HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE)
#   go into `envs:` in the container config. This mirrors compose's
#   precedence: `environment:` beats `env_file:`.
#
#   Use --inline-env to fall back to the legacy behaviour of baking the
#   whole .env into container-config.yaml. WARNING: that writes secrets
#   (LLM_API_KEY, DB_PASSWORD, ...) into a plaintext YAML on disk.
#
# --dry-run does NOT require crictl, ctr, podman, skopeo, unzip, or
# an existing image tar. It only writes the YAML files and prints them.
#
# Usage:
#   ./deploy-saleschat-crictl.sh --image-tar PATH [options]
#
#   --image-tar PATH          .tar / .tar.gz / .tgz / .zip
#                             (required unless --clean or --dry-run)
#   --image-ref REF           override detected image ref
#   --env-file PATH           default: <project>/.env (used only with
#                             --inline-env, and as existence hint)
#   --project-root PATH       default: parent of this script
#   --state-dir PATH          default: <project>/.crictl
#   --pod-name NAME           default: cca_saleschat-pod
#   --container-name NAME     default: cca_saleschat
#   --namespace NS            default: default
#   --runtime NAME            auto|containerd|crio   (default: auto)
#   --ctr-namespace NS        containerd namespace    (default: k8s.io)
#   --inline-env              bake .env into envs: (legacy; leaks secrets)
#   --dry-run                 only write YAML and print it (no runtime needed)
#   --clean                   tear down the existing pod/container and exit
#   --no-follow               do not tail logs after start
#   -h|--help

set -euo pipefail
trap 'echo "ERROR: line $LINENO: $BASH_COMMAND" >&2' ERR

# ------------------------------------------------------------------ defaults
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$PROJECT_ROOT/.crictl"
POD_NAMESPACE="default"
POD_NAME="cca_saleschat-pod"
CONTAINER_NAME="cca_saleschat"
CTR_NAMESPACE="k8s.io"
IMAGE_TAR=""
IMAGE_REF=""
ENV_FILE=""
RUNTIME="auto"
DRY_RUN=false
CLEAN=false
FOLLOW=true
INLINE_ENV=false

# ------------------------------------------------------------------ helpers
log()  { printf '\033[1;34m[*]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'; }

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

yaml_str() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\t'/\\t}"
  printf '"%s"' "$s"
}

# ------------------------------------------------------------------ argparse
while (($#)); do
  case "$1" in
    --image-tar)         IMAGE_TAR="${2:-}";       shift 2 ;;
    --image-ref)         IMAGE_REF="${2:-}";       shift 2 ;;
    --env-file)          ENV_FILE="${2:-}";        shift 2 ;;
    --project-root)      PROJECT_ROOT="${2:-}";    shift 2 ;;
    --state-dir)         STATE_DIR="${2:-}";       shift 2 ;;
    --pod-name)          POD_NAME="${2:-}";        shift 2 ;;
    --container-name)    CONTAINER_NAME="${2:-}";  shift 2 ;;
    --namespace)         POD_NAMESPACE="${2:-}";   shift 2 ;;
    --runtime)           RUNTIME="${2:-}";         shift 2 ;;
    --ctr-namespace)     CTR_NAMESPACE="${2:-}";   shift 2 ;;
    --inline-env)        INLINE_ENV=true;          shift ;;
    --dry-run)           DRY_RUN=true;             shift ;;
    --clean)             CLEAN=true;               shift ;;
    --no-follow)         FOLLOW=false;             shift ;;
    -h|--help)           usage; exit 0 ;;
    *) die "unknown argument: $1 (use --help)" ;;
  esac
done

[[ -z "$ENV_FILE" ]] && ENV_FILE="$PROJECT_ROOT/.env"
[[ -z "$IMAGE_TAR" && "$CLEAN" == "false" && "$DRY_RUN" == "false" ]] \
  && die "--image-tar is required"

# dry-run does not need crictl; everything else does
$DRY_RUN || command -v crictl >/dev/null || die "crictl not found in PATH"
mkdir -p "$STATE_DIR"

POD_CONFIG="$STATE_DIR/pod-config.yaml"
CONTAINER_CONFIG="$STATE_DIR/container-config.yaml"
LOG_DIR="$STATE_DIR/logs"
mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------ teardown
teardown() {
  log "tearing down existing '$CONTAINER_NAME' / '$POD_NAME' (if any)"
  local cid pid
  while IFS= read -r cid; do
    [[ -z "$cid" ]] && continue
    crictl stop "$cid"  >/dev/null 2>&1 || true
    crictl rm -f "$cid" >/dev/null 2>&1 || true
  done < <(crictl ps -a -q --name "^${CONTAINER_NAME}$" 2>/dev/null || true)

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    crictl stopp "$pid" >/dev/null 2>&1 || true
    crictl rmp  -f "$pid" >/dev/null 2>&1 || true
  done < <(crictl pods -q --name "^${POD_NAME}$" 2>/dev/null || true)
}

if $CLEAN; then
  teardown
  log "done."
  exit 0
fi

# ------------------------------------------------------------------ runtime
detect_runtime() {
  case "$RUNTIME" in
    containerd|crio) printf '%s' "$RUNTIME"; return ;;
    auto) ;;
    *) die "invalid --runtime: $RUNTIME" ;;
  esac
  local info
  info="$(crictl info 2>/dev/null || true)"
  if grep -qiE 'cri-o|crio' <<<"$info"; then printf 'crio'; return; fi
  if command -v ctr >/dev/null 2>&1; then printf 'containerd'; return; fi
  if command -v podman >/dev/null 2>&1 || command -v skopeo >/dev/null 2>&1; then
    printf 'crio'; return
  fi
  printf 'containerd'
}

if $DRY_RUN; then
  RT="${RUNTIME:-auto}"; [[ "$RT" == "auto" ]] && RT="containerd"
  log "dry-run: assuming CRI runtime: $RT (no probing)"
else
  RT="$(detect_runtime)"
  log "using CRI runtime: $RT"
fi

# ------------------------------------------------------------------ archive
prepare_archive() {
  local src="$1"
  [[ -f "$src" ]] || die "image archive not found: $src"
  case "$src" in
    *.tar|*.tar.gz|*.tgz)
      printf '%s\n' "$src" ;;
    *.zip)
      command -v unzip >/dev/null || die "unzip required for .zip archives"
      local tmp
      tmp="$(mktemp -d)"
      unzip -q "$src" -d "$tmp"
      local found
      found="$(find "$tmp" -maxdepth 3 -type f \
        \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) \
        | head -n1)"
      [[ -n "$found" ]] || die "no .tar/.tar.gz/.tgz found inside $src"
      printf '%s\n' "$found"
      ;;
    *)
      die "unsupported archive '$src' (want .tar/.tar.gz/.tgz/.zip)"
      ;;
  esac
}

list_refs() {
  crictl images -q 2>/dev/null | sort -u || true
}

parse_import_output() {
  local out="$1" ref=""
  ref="$(sed -nE 's/^unpacking ([^ ]+) \(sha256:.*$/\1/p' <<<"$out" | tail -n1)"
  [[ -n "$ref" ]] && { printf '%s' "$ref"; return; }
  ref="$(sed -nE 's/^Loaded image: (.+)$/\1/p' <<<"$out" | tail -n1)"
  [[ -n "$ref" ]] && { printf '%s' "$ref"; return; }
  ref="$(grep -oE '[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]+)*:[a-zA-Z0-9._-]+' <<<"$out" \
    | tail -n1 || true)"
  [[ -n "$ref" ]] && printf '%s' "$ref"
}

import_containerd() {
  local tar="$1"
  command -v ctr >/dev/null || die "ctr not found; cannot import into containerd"
  local before after out ref
  before="$(list_refs)"
  out="$(ctr -n "$CTR_NAMESPACE" images import "$tar" 2>&1)" || die "ctr import failed:
$out"
  after="$(list_refs)"
  ref="$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") \
    | sed '/^$/d' | head -n1 || true)"
  [[ -z "$ref" ]] && ref="$(parse_import_output "$out")"
  [[ -n "$ref" ]] && printf '%s' "$ref"
}

import_crio() {
  local tar="$1" out
  if command -v podman >/dev/null 2>&1; then
    out="$(podman load -i "$tar" 2>&1)" || die "podman load failed:
$out"
    printf '%s' "$(parse_import_output "$out")"
    return
  fi
  if command -v skopeo >/dev/null 2>&1; then
    local detected
    detected="$(skopeo inspect "docker-archive:$tar" --format '{{.Name}}:{{.Tag}}' 2>/dev/null || true)"
    [[ -z "$detected" ]] && die "skopeo could not inspect $tar; pass --image-ref"
    skopeo copy "docker-archive:$tar" "containers-storage:$detected" >/dev/null \
      || die "skopeo copy failed"
    printf '%s' "$detected"
    return
  fi
  die "need podman or skopeo to import images for CRI-O"
}

# --- import, or dry-run fallback for IMAGE_REF ----------------------
if $DRY_RUN; then
  if [[ -z "$IMAGE_REF" ]]; then
    if [[ -n "$IMAGE_TAR" ]]; then
      base="$(basename "$IMAGE_TAR")"
      base="${base%.tar.gz}"; base="${base%.tgz}"
      base="${base%.tar}";    base="${base%.zip}"
      IMAGE_REF="${base}:latest"
      warn "dry-run: no --image-ref, inferred from archive name: $IMAGE_REF"
      [[ -f "$IMAGE_TAR" ]] || warn "dry-run: archive does not exist: $IMAGE_TAR"
    else
      IMAGE_REF="cca_saleschat:latest"
      warn "dry-run: no --image-ref and no --image-tar, using placeholder: $IMAGE_REF"
    fi
  fi
  log "dry-run: skipping image import"
else
  ARCHIVE="$(prepare_archive "$IMAGE_TAR")"
  log "importing image archive: $ARCHIVE"

  DETECTED_REF=""
  if [[ "$RT" == "containerd" ]]; then
    DETECTED_REF="$(import_containerd "$ARCHIVE")"
  else
    DETECTED_REF="$(import_crio "$ARCHIVE")"
  fi

  if [[ -z "$IMAGE_REF" ]]; then
    if [[ -n "$DETECTED_REF" ]]; then
      IMAGE_REF="$DETECTED_REF"
      log "detected image ref: $IMAGE_REF"
    else
      die "could not detect image ref from import; pass --image-ref REF"
    fi
  fi

  if ! crictl inspecti "$IMAGE_REF" >/dev/null 2>&1; then
    warn "crictl inspecti '$IMAGE_REF' failed"
    warn "try --image-ref with the exact ref shown by 'crictl images'"
  fi
fi

# ------------------------------------------------------------------ env
# Default: do NOT inline .env. The app loads <PROJECT_ROOT>/.env itself;
# we only inject the 4 operational overrides that compose's
# `environment:` block would have set (these beat env_file in compose).
parse_env_file() {
  local file="$1"
  [[ -f "$file" ]] || { warn ".env file not found: $file"; return 0; }
  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    line="$(trim "$line")"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == export[[:space:]]* ]] && { line="${line#export }"; line="$(trim "$line")"; }
    [[ "$line" == *=* ]] || continue
    key="$(trim "${line%%=*}")"
    val="$(trim "${line#*=}")"
    [[ -z "$key" ]] && continue

    if [[ ${#val} -ge 2 && ${val:0:1} == '"' && ${val: -1} == '"' ]]; then
      val="${val:1:${#val}-2}"
      val="${val//\\\\/\\}"; val="${val//\\\"/\"}"
    elif [[ ${#val} -ge 2 && ${val:0:1} == "'" && ${val: -1} == "'" ]]; then
      val="${val:1:${#val}-2}"
    else
      if [[ "$val" =~ ^(.*[^[:space:]])[[:space:]]+#.*$ ]]; then
        val="$(trim "${BASH_REMATCH[1]}")"
      fi
    fi
    printf '%s\t%s\n' "$key" "$val"
  done < "$file"
}

declare -A ENVS=()

if $INLINE_ENV; then
  warn "--inline-env: baking $ENV_FILE into container-config.yaml (secrets included!)"
  while IFS=$'\t' read -r k v; do
    [[ -z "${k:-}" ]] && continue
    ENVS["$k"]="$v"
  done < <(parse_env_file "$ENV_FILE")
fi

ENVS["TZ"]="Asia/Hong_Kong"
ENVS["HF_HOME"]="/app/models/huggingface"
ENVS["HF_HUB_OFFLINE"]="1"
ENVS["TRANSFORMERS_OFFLINE"]="1"

if ! $INLINE_ENV; then
  if [[ ! -f "$ENV_FILE" ]]; then
    warn "no .env at $ENV_FILE and .env is not being inlined"
    warn "ensure the app loads .env itself, or use --inline-env"
  fi
fi

# ------------------------------------------------------------------ YAML gen
generate_pod_config() {
  local uid
  uid="$(cat /proc/sys/kernel/random/uuid 2>/dev/null || date +%s%N)-$$"
  cat > "$POD_CONFIG" <<EOF
metadata:
  name: $(yaml_str "$POD_NAME")
  namespace: $(yaml_str "$POD_NAMESPACE")
  uid: $(yaml_str "$uid")
  attempt: 1
log_directory: $(yaml_str "$LOG_DIR")
linux:
  security_context: {}
EOF
}

emit_mount() {
  local cpath="$1" hpath="$2" ro="${3:-false}"
  if [[ ! -e "$hpath" ]]; then
    warn "skip mount (source missing): $hpath"
    return 0
  fi
  cat >> "$CONTAINER_CONFIG" <<EOF
  - container_path: $(yaml_str "$cpath")
    host_path: $(yaml_str "$hpath")
    readonly: $ro
EOF
}

generate_container_config() {
  cat > "$CONTAINER_CONFIG" <<EOF
metadata:
  name: $(yaml_str "$CONTAINER_NAME")
  attempt: 1

image:
  image: $(yaml_str "$IMAGE_REF")

command:
  - "python"
  - "-m"
  - "app.main"
  - "--mode"
  - "offline"

envs:
EOF

  local k
  while IFS= read -r k; do
    printf '  - key: %s\n    value: %s\n' \
      "$(yaml_str "$k")" "$(yaml_str "${ENVS[$k]}")" >> "$CONTAINER_CONFIG"
  done < <(printf '%s\n' "${!ENVS[@]}" | LC_ALL=C sort)

  printf '\nmounts:\n' >> "$CONTAINER_CONFIG"

  [[ -f /etc/timezone ]]  && emit_mount "/etc/timezone"  "/etc/timezone"  true
  [[ -f /etc/localtime ]] && emit_mount "/etc/localtime" "/etc/localtime" true

  emit_mount "/app/.env"        "$PROJECT_ROOT/.env"       true
  emit_mount "/app/data"        "$PROJECT_ROOT/data"       false
  emit_mount "/app/output"      "$PROJECT_ROOT/output"     false
  emit_mount "/app/log"         "$PROJECT_ROOT/log"        false
  emit_mount "/app/models"      "$PROJECT_ROOT/models"     false

  printf '\nlog_path: %s\n' "$(yaml_str "$CONTAINER_NAME.log")" >> "$CONTAINER_CONFIG"
}

log "writing $POD_CONFIG"
generate_pod_config
log "writing $CONTAINER_CONFIG"
generate_container_config

if $DRY_RUN; then
  echo
  echo "===== $POD_CONFIG ====="
  cat "$POD_CONFIG"
  echo
  echo "===== $CONTAINER_CONFIG ====="
  cat "$CONTAINER_CONFIG"
  exit 0
fi

# ------------------------------------------------------------------ run
teardown

log "starting via crictl run"
CID="$(crictl run "$CONTAINER_CONFIG" "$POD_CONFIG" | tail -n1)"
[[ -n "$CID" ]] || die "crictl run returned no container id"
log "container id: $CID"

if $FOLLOW; then
  log "tailing logs (Ctrl+C to detach; container keeps running)"
  exec crictl logs -f "$CID"
fi

log "started. follow logs with:  crictl logs -f $CID"