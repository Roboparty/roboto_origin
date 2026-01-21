#!/bin/bash

###############################################################################
# roboto_origin 自动同步脚本
# 功能：
#   1. 自动拉取四个主模块的最新代码（使用 subtree）
#   2. 读取各主模块的 .gitmodules，自动拉取其子模块（使用 subtree --squash）
#   3. 确保本地永远是四个主模块的快照
###############################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log_info "开始同步 roboto_origin 所有 subtree 模块"
echo "========================================"

###############################################################################
# 辅助函数
###############################################################################

# 函数：获取远程仓库的默认分支
get_default_branch() {
    local repo_url="$1"

    # 尝试使用 git ls-remote 获取 HEAD 指向的分支
    local default_branch=$(git ls-remote --symref "$repo_url" HEAD 2>/dev/null | \
        grep -o 'refs/heads/[^"]*' | \
        sed 's|refs/heads/||' | \
        head -n1)

    # 如果检测失败，尝试常见的分支名
    if [ -z "$default_branch" ]; then
        for branch in main master master dev development; do
            if git ls-remote --exit-code "$repo_url" "refs/heads/$branch" &>/dev/null; then
                default_branch="$branch"
                break
            fi
        done
    fi

    echo "$default_branch"
}

###############################################################################
# 第一部分：同步四个主模块
###############################################################################

# 定义主模块数组：格式 "模块名|仓库URL|分支名"
MAIN_MODULES=(
    "Atom01_hardware|https://github.com/Roboparty/Atom01_hardware.git|main"
    "atom01_deploy|https://github.com/Roboparty/atom01_deploy.git|main"
    "atom01_train|https://github.com/Roboparty/atom01_train.git|main"
    "atom01_description|https://github.com/Roboparty/atom01_description.git|main"
)

log_info "步骤 1/2: 同步四个主模块"
echo ""

for module_config in "${MAIN_MODULES[@]}"; do
    IFS='|' read -r module_name module_url module_branch <<< "$module_config"
    module_path="modules/$module_name"

    echo "----------------------------------------"
    log_info "处理主模块: $module_name"

    # 检查模块目录是否存在
    if [ ! -d "$module_path" ]; then
        log_warn "模块目录 $module_path 不存在，首次添加..."
        git subtree add --prefix="$module_path" "$module_url" "$module_branch" --squash
    else
        log_info "更新已存在的模块: $module_name"
        git subtree pull --prefix="$module_path" "$module_url" "$module_branch" --squash
    fi

    log_success "主模块 $module_name 同步完成"
    echo ""
done

###############################################################################
# 第二部分：自动读取并同步主模块的子模块
###############################################################################

echo "========================================"
log_info "步骤 2/2: 自动同步主模块的子模块"
echo ""

# 函数：解析 .gitmodules 文件并同步子模块
sync_submodules() {
    local main_module_path="$1"
    local gitmodules_file="$main_module_path/.gitmodules"

    # 检查 .gitmodules 是否存在
    if [ ! -f "$gitmodules_file" ]; then
        log_info "$main_module_path 没有子模块，跳过"
        echo ""
        return
    fi

    local main_module_name=$(basename "$main_module_path")
    log_info "发现 $main_module_name 包含子模块，正在解析..."

    # 解析 .gitmodules 文件
    # 格式：
    # [submodule "xxx"]
    #     path = xxx
    #     url = xxx

    local in_submodule=false
    local submodule_name=""
    local submodule_path=""
    local submodule_url=""

    while IFS= read -r line || [ -n "$line" ]; do
        # 去除首尾空白
        line=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

        # 跳过空行和注释
        if [ -z "$line" ] || [[ "$line" == \#* ]]; then
            continue
        fi

        # 检测 [submodule "xxx"]
        if [[ "$line" =~ ^\[submodule\ \"(.+)\"\]$ ]]; then
            # 如果之前有子模块信息，先处理它
            if [ -n "$submodule_name" ] && [ -n "$submodule_path" ] && [ -n "$submodule_url" ]; then
                sync_one_submodule "$main_module_name" "$submodule_name" "$submodule_path" "$submodule_url"
            fi

            # 开始新的子模块
            submodule_name="${BASH_REMATCH[1]}"
            submodule_path=""
            submodule_url=""
            in_submodule=true
            continue
        fi

        # 解析 path 和 url
        if [ "$in_submodule" = true ]; then
            if [[ "$line" =~ ^path\ =\ (.+)$ ]]; then
                submodule_path="${BASH_REMATCH[1]}"
            elif [[ "$line" =~ ^url\ =\ (.+)$ ]]; then
                submodule_url="${BASH_REMATCH[1]}"
            fi
        fi
    done < "$gitmodules_file"

    # 处理最后一个子模块
    if [ -n "$submodule_name" ] && [ -n "$submodule_path" ] && [ -n "$submodule_url" ]; then
        sync_one_submodule "$main_module_name" "$submodule_name" "$submodule_path" "$submodule_url"
    fi

    echo ""
}

# 函数：同步单个子模块
sync_one_submodule() {
    local main_module_name="$1"
    local submodule_name="$2"
    local submodule_rel_path="$3"  # 相对于主模块的路径
    local submodule_url="$4"

    local full_submodule_path="modules/$main_module_name/$submodule_rel_path"

    echo "  → 处理子模块: $submodule_name"
    log_info "    路径: $full_submodule_path"
    log_info "    仓库: $submodule_url"

    # 自动检测默认分支
    local submodule_branch=$(get_default_branch "$submodule_url")
    if [ -z "$submodule_branch" ]; then
        log_error "    无法检测默认分支，跳过 $submodule_name"
        return 1
    fi
    log_info "    检测到分支: $submodule_branch"

    # 检查子模块目录是否存在
    if [ ! -d "$full_submodule_path" ]; then
        log_warn "    子模块目录不存在，首次添加..."
        git subtree add --prefix="$full_submodule_path" "$submodule_url" "$submodule_branch" --squash
    else
        log_info "    更新已存在的子模块..."
        git subtree pull --prefix="$full_submodule_path" "$submodule_url" "$submodule_branch" --squash
    fi

    log_success "    子模块 $submodule_name 同步完成"
}

# 遍历所有主模块，查找并同步子模块
for module_config in "${MAIN_MODULES[@]}"; do
    IFS='|' read -r module_name module_url module_branch <<< "$module_config"
    module_path="modules/$module_name"

    echo "----------------------------------------"
    sync_submodules "$module_path"
done

###############################################################################
# 完成
###############################################################################

echo "========================================"
log_success "所有模块同步完成！"
echo ""
log_info "当前仓库状态："
git status --short
echo ""
log_info "最近的同步提交："
git log --oneline -5
