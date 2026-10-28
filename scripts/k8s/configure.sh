#!/usr/bin/env bash
# 配置向导 —— 设置 config.env 里的变量
#
# 用法：
#   ./configure.sh                    显示当前配置（默认）
#   ./configure.sh --show             同无参数
#   ./configure.sh --wizard           交互式向导（推荐）
#   ./configure.sh --edit             用 $EDITOR 打开 config.env
#   ./configure.sh --reset            重置为模板默认值
#   ./configure.sh --set KEY=value    修改单项
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config.env"
EXAMPLE="$SCRIPT_DIR/config.env.example"

if [ -t 1 ]; then
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
    C_BLUE=$'\033[34m'; C_CYAN=$'\033[36m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
    C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""; C_CYAN=""; C_BOLD=""; C_RESET=""
fi

ensure_config() {
    if [ ! -f "$CONFIG" ]; then
        if [ -f "$EXAMPLE" ]; then
            cp "$EXAMPLE" "$CONFIG"
            chmod 600 "$CONFIG" 2>/dev/null || true
            echo -e "${C_YELLOW}[INFO]${C_RESET} 已从模板创建: $CONFIG"
        else
            echo "ERROR: 缺少 $EXAMPLE" >&2
            exit 1
        fi
    fi
}

get_value() {
    local key="$1"
    (
        set +eu
        unset "$key" 2>/dev/null || true
        # shellcheck disable=SC1090
        source "$CONFIG" 2>/dev/null || true
        printf '%s' "${!key:-}"
    )
}

set_value() {
    local key="$1" value="$2"
    ensure_config
    local tmp; tmp=$(mktemp)
    local found=0
    while IFS= read -r line; do
        if [[ "$line" =~ ^[[:space:]]*${key}= ]]; then
            printf '%s="%s"\n' "$key" "$value" >> "$tmp"
            found=1
        else
            printf '%s\n' "$line" >> "$tmp"
        fi
    done < "$CONFIG"
    if [ "$found" = "0" ]; then
        printf '\n%s="%s"\n' "$key" "$value" >> "$tmp"
    fi
    mv "$tmp" "$CONFIG"
    chmod 600 "$CONFIG" 2>/dev/null || true
}

# 危险配置项提示（CLEAR_CACHE_ON_START=true 会清空 cache/output/qa_pairs）
warn_if_destructive() {
    local key="$1" value="$2"
    case "$key" in
        CLEAR_CACHE_ON_START)
            if [ "$value" = "true" ]; then
                echo -e "  ${C_RED}⚠️  警告${C_RESET}: CLEAR_CACHE_ON_START=true 会在下次启动时清空"
                echo -e "      cache/、output/、qa_pairs 表！数据不可恢复。"
                echo -e "      如只想重跑一次，建议改为 offline.sh 前手动 ./uninstall.sh --purge"
            fi
            ;;
    esac
}

cmd_show() {
    if [ ! -f "$CONFIG" ]; then
        echo -e "${C_YELLOW}[WARN]${C_RESET} 配置不存在，将使用脚本默认值"
        echo "       运行 ./configure.sh --wizard 创建"
        echo ""
    fi

    echo "${C_BOLD}当前配置${C_RESET}  ($CONFIG)"
    echo "────────────────────────────────────────────────────────────"

    section() { echo ""; echo "${C_CYAN}  $1${C_RESET}"; }
    row() {
        local key="$1" desc="$2"
        local val
        val=$(get_value "$key")
        if [ -z "$val" ]; then
            printf "    %-22s = %s\n" "$key" "${C_YELLOW}<未设置>${C_RESET}"
        else
            # CLEAR_CACHE_ON_START=true 时用红色高亮显示值
            if [ "$key" = "CLEAR_CACHE_ON_START" ] && [ "$val" = "true" ]; then
                printf "    %-22s = %s    # %s\n" "$key" "${C_RED}${val}${C_RESET}" "$desc"
            else
                printf "    %-22s = %s    # %s\n" "$key" "$val" "$desc"
            fi
        fi
    }

    section "环境"
    row CCA_ENV             "dev / prod"

    section "App"
    row NS                  "app namespace"
    row IMAGE               "app 镜像"
    row PF_PORT             "port-forward 端口"
    row PF_ENABLE           "1=启用 0=禁用"

    section "PostgreSQL 集群位置"
    row INFRA_PG_NS         "PG namespace"
    row INFRA_PG_SVC        "PG service 名"
    row INFRA_PG_PORT       "PG 端口"

    section "TEI 集群位置"
    row INFRA_TEI_NS        "TEI namespace"
    row INFRA_TEI_SVC       "TEI service 名"
    row INFRA_TEI_PORT      "TEI 端口"
    row INFRA_TEI_LABEL     "TEI Pod 的 app label（与 Service 名可能不同）"

    section "App 连接的 PG"
    row DB_HOST             "DB 主机（默认 bridge service 名）"
    row DB_PORT             "DB 端口"
    row DB_NAME             "数据库名"
    row DB_USER             "数据库用户"
    echo -e "    ${C_YELLOW}<DB_PASSWORD>${C_RESET}    = ***    # 在 manifests/cca-secret.yaml"

    section "App 连接的 TEI / Embedding"
    row EMBEDDING_TEI_URL   "TEI 端点（默认 bridge service）"
    row EMBEDDING_MODEL     "Embedding 模型（完整 repo id）"
    row EMBEDDING_TEI_STYLE "auto / native / openai"

    section "LLM"
    row LLM_BASE_URL        "LLM API base url"
    row LLM_MODEL           "LLM 模型名"
    echo -e "    ${C_YELLOW}<LLM_API_KEY>${C_RESET}    = ***    # 在 manifests/cca-secret.yaml"

    section "运行时行为"
    row STATE_MACHINE_MODE  "off / guide / enforce"
    row CLEAR_CACHE_ON_START "true=启动清空 cache/output/qa_pairs（⚠️ 危险）"
    row LOG_LEVEL           "日志级别"

    section "Embedding 运行参数"
    row EMB_CACHE_SIZE             "在线 embedding LRU 缓存条数"
    row EMBEDDING_SORT_BY_LENGTH   "按长度排序分块（true/false）"
    row EMBEDDING_TEI_TIMEOUT      "TEI HTTP 超时（秒）"
    row EMBEDDING_TEI_BATCH_SIZE   "每次请求条数"
    row EMBEDDING_TEI_CONCURRENCY  "并发连接数"
    row EMBEDDING_TEI_RETRY_MAX    "失败重试次数"
    row EMBEDDING_TEI_TRUNCATE     "native 风格下由服务端截断"

    section "检索参数"
    row ONLINE_TOP_K               "在线 top-k"
    row ONLINE_SIM_THRESHOLD       "在线相似度阈值"
    row HYBRID_MAX_HOPS            "图 BFS 最大跳数"
    row HYBRID_PG_TOP_K            "向量检索深度"

    section "数据与本地路径"
    row CALLS_FILE_PATTERN  "offline 扫描的文件 glob"
    row HF_CACHE_DIR        "本地 HF 缓存"
    row DATA_DIR_LOCAL      "本地数据目录"
    row DEFAULT_BATCH       "默认批次（空=自动选最新）"
    row PROMPTS_FILE        "prompts.yaml 路径"

    echo ""
    echo "────────────────────────────────────────────────────────────"

    # 危险配置醒目提示
    if [ "$(get_value CLEAR_CACHE_ON_START)" = "true" ]; then
        echo -e "  ${C_RED}⚠️  CLEAR_CACHE_ON_START=true${C_RESET} — 下次 app 启动会清空数据！"
        echo "     修改：./configure.sh --set CLEAR_CACHE_ON_START=false"
        echo "────────────────────────────────────────────────────────────"
    fi

    echo "  修改：./configure.sh --wizard   （交互式）"
    echo "        ./configure.sh --set KEY=value"
    echo "        ./configure.sh --edit      （编辑器）"
    echo "────────────────────────────────────────────────────────────"
}

cmd_wizard() {
    ensure_config

    echo "${C_BOLD}CCA 配置向导${C_RESET}"
    echo "────────────────────────────────────────────────────────────"
    echo "  每项会显示当前值，直接回车保持，输入新值覆盖"
    echo "  Ctrl+C 可随时中断（已输入的不保存）"
    echo "────────────────────────────────────────────────────────────"

    ask() {
        local key="$1" desc="$2"
        local cur display new
        cur=$(get_value "$key")
        display="$cur"
        if [ -z "$display" ]; then
            display="(空)"
        fi
        echo ""
        echo "${C_CYAN}${desc}${C_RESET}  [当前: ${display}]"
        read -r -p "  ${key} > " new || new=""
        if [ -n "$new" ] && [ "$new" != "$cur" ]; then
            set_value "$key" "$new"
            echo -e "  ${C_GREEN}✓${C_RESET} 已更新"
            warn_if_destructive "$key" "$new"
        else
            echo -e "  ${C_BOLD}-${C_RESET} 保持原值"
            warn_if_destructive "$key" "$cur"
        fi
    }

    ask CCA_ENV          "运行环境 (dev/prod)"

    ask NS               "app 的 namespace"
    ask IMAGE            "app 镜像地址"
    ask PF_PORT          "port-forward 端口"
    ask PF_ENABLE        "是否启用 port-forward (1/0)"

    ask INFRA_PG_NS      "PostgreSQL 的 namespace"
    ask INFRA_PG_SVC     "PostgreSQL 的 Service 名"
    ask INFRA_PG_PORT    "PostgreSQL 端口"

    ask INFRA_TEI_NS     "TEI 的 namespace"
    ask INFRA_TEI_SVC    "TEI 的 Service 名"
    ask INFRA_TEI_PORT   "TEI 端口"
    ask INFRA_TEI_LABEL  "TEI Pod 的 app label（如 tei，与 Service 名可能不同）"

    ask DB_HOST          "app 连的 DB 主机（完整 DNS）"
    ask DB_PORT          "app 连的 DB 端口"
    ask DB_NAME          "数据库名"
    ask DB_USER          "数据库用户"

    ask EMBEDDING_TEI_URL "app 连的 TEI 端点"
    ask EMBEDDING_MODEL  "Embedding 模型（完整 repo id）"
    ask EMBEDDING_TEI_STYLE "TEI 风格 (auto/native/openai)"

    ask LLM_BASE_URL     "LLM base URL"
    ask LLM_MODEL        "LLM 模型名"

    echo ""
    echo "${C_CYAN}── 运行时行为 ──${C_RESET}"
    ask STATE_MACHINE_MODE    "检索模式 (off/guide/enforce)"
    ask CLEAR_CACHE_ON_START  "启动清空缓存？(true/false)（⚠️ true 会删数据）"
    ask LOG_LEVEL             "日志级别 (DEBUG/INFO/WARNING)"

    echo ""
    echo "${C_CYAN}── 数据与本地路径 ──${C_RESET}"
    ask CALLS_FILE_PATTERN "offline 扫描的文件 glob"
    ask HF_CACHE_DIR     "本地 HF 缓存目录"
    ask DATA_DIR_LOCAL   "本地数据目录"
    ask DEFAULT_BATCH    "默认批次（空=自动选最新）"
    ask PROMPTS_FILE     "prompts.yaml 路径"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo -e "${C_GREEN}✓ 配置已保存到${C_RESET} $CONFIG"

    # 危险配置最终确认
    if [ "$(get_value CLEAR_CACHE_ON_START)" = "true" ]; then
        echo ""
        echo -e "${C_RED}⚠️  CLEAR_CACHE_ON_START=true${C_RESET}"
        echo "   下次 app 启动会清空 cache/、output/、qa_pairs 表！"
        read -r -p "  确认？(yes/其他=改为 false) > " ans
        if [ "${ans:-}" != "yes" ]; then
            set_value CLEAR_CACHE_ON_START "false"
            echo -e "  ${C_GREEN}✓${C_RESET} 已改为 false"
        fi
    fi

    echo ""
    echo "下一步："
    echo "  ./configure.sh --show     # 检查配置"
    echo "  ./install.sh              # 部署"
    echo "────────────────────────────────────────────────────────────"
}

cmd_reset() {
    if [ -f "$CONFIG" ]; then
        local backup="$CONFIG.bak.$(date +%Y%m%d-%H%M%S)"
        cp "$CONFIG" "$backup"
        echo -e "${C_YELLOW}[INFO]${C_RESET} 已备份: $backup"
    fi
    cp "$EXAMPLE" "$CONFIG"
    chmod 600 "$CONFIG" 2>/dev/null || true
    echo -e "${C_GREEN}✓${C_RESET} 已重置为模板默认值: $CONFIG"
}

cmd_edit() {
    ensure_config
    local editor="${EDITOR:-vi}"
    if ! command -v "$editor" >/dev/null 2>&1; then
        echo "ERROR: 编辑器 '$editor' 不可用" >&2
        echo "       设置 EDITOR=vim ./configure.sh --edit" >&2
        exit 1
    fi
    "$editor" "$CONFIG"
    echo -e "${C_GREEN}✓${C_RESET} 已保存"
    echo "检查：./configure.sh --show"
}

cmd_set() {
    if [ $# -lt 1 ]; then
        echo "用法: ./configure.sh --set KEY=value [KEY2=value2 ...]" >&2
        exit 1
    fi
    ensure_config
    for pair in "$@"; do
        local key="${pair%%=*}"
        local value="${pair#*=}"
        if [ "$key" = "$pair" ]; then
            echo "ERROR: 格式应为 KEY=value，收到 '$pair'" >&2
            exit 1
        fi
        set_value "$key" "$value"
        echo -e "  ${C_GREEN}✓${C_RESET} $key = $value"
        warn_if_destructive "$key" "$value"
    done
    echo ""
    echo "已保存到 $CONFIG"
    echo "检查：./configure.sh --show"
}

case "${1:-show}" in
    --show|show|"")   cmd_show ;;
    --wizard|wizard)  cmd_wizard ;;
    --reset|reset)    cmd_reset ;;
    --edit|edit)      cmd_edit ;;
    --set)
        shift
        cmd_set "$@"
        ;;
    -h|--help)
        sed -n '2,15p' "$0"
        ;;
    *)
        echo "未知参数: $1" >&2
        echo "运行 ./configure.sh --help 查看用法" >&2
        exit 1
        ;;
esac