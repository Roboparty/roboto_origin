#!/bin/bash

# Subtree 同步脚本（包括动态处理 submodule 转为 subtree）
# 流程：
# 1. 同步主模块的 subtree
# 2. 从远程获取主模块的 .gitmodules
# 3. 解析 .gitmodules，将 submodule 也作为 subtree 拉取

echo "==================================="
echo "开始同步所有 subtree（含 submodule 转 subtree）..."
echo "==================================="

# 将 HTTPS URL 转换为 SSH URL
convert_to_ssh_url() {
    local url="$1"
    # 转换 https://github.com/user/repo.git -> git@github.com:user/repo.git
    if echo "$url" | grep -q "^https://github.com/"; then
        echo "$url" | sed 's|https://github.com/|git@github.com:|'
    else
        echo "$url"
    fi
}

# 确保 remote 存在的辅助函数
ensure_remote() {
    local remote_name="$1"
    local remote_url="$2"

    # 转换为 SSH URL
    remote_url=$(convert_to_ssh_url "$remote_url")

    if ! git remote | grep -q "^${remote_name}$"; then
        echo "  添加 remote: $remote_name ($remote_url)"
        git remote add "$remote_name" "$remote_url"
    else
        # 更新已存在的 remote URL
        local current_url=$(git remote get-url "$remote_name")
        if [ "$current_url" != "$remote_url" ]; then
            echo "  更新 remote URL: $remote_name -> $remote_url"
            git remote set-url "$remote_name" "$remote_url"
        fi
    fi
}

# Subtree 同步辅助函数
sync_subtree() {
    local prefix="$1"
    local remote_name="$2"
    local branch="${3:-main}"

    echo ""
    echo "同步 $prefix..."

    # 确保 remote 存在
    if ! git remote | grep -q "^${remote_name}$"; then
        echo "  ⚠ remote $remote_name 不存在，跳过"
        return 1
    fi

    # 检查目录是否为空或不存在
    local need_add=false
    if [ ! -d "$prefix" ] || ! ls -A "$prefix" >/dev/null 2>&1; then
        need_add=true
    fi

    # 拉取更新
    if git fetch "$remote_name" "$branch" >/dev/null 2>&1; then
        if [ "$need_add" = true ]; then
            # 首次添加
            if git subtree add --prefix="$prefix" "$remote_name" "$branch" >/dev/null 2>&1; then
                echo "  ✓ $prefix 首次添加成功"
                return 0
            else
                echo "  ⚠ $prefix 添加失败"
                return 1
            fi
        else
            # 更新已有的 subtree
            if git subtree pull --prefix="$prefix" "$remote_name" "$branch" -m "Merge subtree $prefix" >/dev/null 2>&1; then
                echo "  ✓ $prefix 同步成功"
                return 0
            else
                echo "  ⚠ $prefix 同步失败或已是最新"
                return 1
            fi
        fi
    else
        echo "  ⚠ 无法从 $remote_name 获取 $branch 分支"
        return 1
    fi
}

# 从远程获取 .gitmodules 并解析，将 submodule 作为 subtree 同步
sync_submodules_as_subtrees() {
    local remote_name="$1"
    local base_path="$2"
    local remote_branch="${3:-main}"

    echo ""
    echo "处理 $base_path 的 submodule（转为 subtree）..."

    # 从远程获取 .gitmodules 文件
    local tmp_gitmodules=$(mktemp)
    if git show "$remote_name/$remote_branch:.gitmodules" > "$tmp_gitmodules" 2>/dev/null; then
        echo "  ✓ 从远程获取 .gitmodules"

        # 解析 .gitmodules 文件
        local current_path=""
        local current_url=""

        while IFS='=' read -r key value; do
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)

            if [ "$key" = "path" ]; then
                current_path="$value"
            elif [ "$key" = "url" ]; then
                current_url="$value"

                if [ -n "$current_path" ] && [ -n "$current_url" ]; then
                    local full_path="$base_path/$current_path"
                    local submodule_name="submodule_$(echo "$current_path" | tr '/' '_')_$remote_name"

                    echo "    发现 submodule: $current_path"
                    echo "      URL: $current_url"

                    # 添加 remote
                    ensure_remote "$submodule_name" "$current_url"

                    # 尝试获取远程仓库的默认分支
                    local submodule_branch="main"
                    if git ls-remote --heads "$submodule_name" | grep -q "refs/heads/master"; then
                        submodule_branch="master"
                    fi

                    # 同步 subtree
                    sync_subtree "$full_path" "$submodule_name" "$submodule_branch"

                    current_path=""
                    current_url=""
                fi
            fi
        done < "$tmp_gitmodules"

        rm -f "$tmp_gitmodules"
    else
        echo "  ⚠ 远程仓库没有 .gitmodules 文件或获取失败"
        rm -f "$tmp_gitmodules"
    fi
}

# ========== 主流程 ==========

echo ""
echo "==================================="
echo "步骤 1: 同步主模块 subtree..."
echo "==================================="

# Atom01_hardware
sync_subtree "modules/Atom01_hardware" "Atom01_hardware" "main"

# atom01_description
sync_subtree "modules/atom01_description" "atom01_description" "main"

# atom01_train
sync_subtree "modules/atom01_train" "atom01_train" "main"

# atom01_deploy
sync_subtree "modules/atom01_deploy" "atom01_deploy" "main"

echo ""
echo "==================================="
echo "步骤 2: 将 submodule 转为 subtree（动态获取 .gitmodules）..."
echo "==================================="

# 处理 atom01_deploy 的 submodule
sync_submodules_as_subtrees "atom01_deploy" "modules/atom01_deploy" "main"

# 处理 atom01_train 的 submodule
sync_submodules_as_subtrees "atom01_train" "modules/atom01_train" "main"

echo ""
echo "==================================="
echo "步骤 3: 清理旧的 .gitmodules 文件..."
echo "==================================="

# 删除 .gitmodules 文件（因为不再使用 submodule）
for gitmodules in modules/atom01_deploy/.gitmodules modules/atom01_train/.gitmodules; do
    if [ -f "$gitmodules" ]; then
        echo "  删除 $gitmodules"
        rm -f "$gitmodules"
    fi
done

echo ""
echo "==================================="
echo "✓ 所有 subtree 同步完成！"
echo "==================================="
echo ""
echo "现在可以提交并推送更新："
echo "  git add ."
echo "  git commit -m 'chore: update all subtrees (submodules converted to subtree)'"
echo "  git push"
echo ""
