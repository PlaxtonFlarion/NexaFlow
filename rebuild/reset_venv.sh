#!/bin/bash

# ========== 配置区 ==========
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
REQ_FILE="$PROJECT_DIR/requirements.txt"
PYTHON="python3"
# ============================

echo "项目路径：$PROJECT_DIR"
echo "虚拟环境目录：$VENV_DIR"

# 用户确认是否删除 venv
read -r -p "确认要删除并重建 venv 吗？(y/N): " CONFIRM
if [[ "$CONFIRM" == "y" || "$CONFIRM" == "Y" ]]; then
    if [ -d "$VENV_DIR" ]; then
        echo "正在删除虚拟环境..."
        rm -rf "$VENV_DIR"
        echo "虚拟环境已删除。"
    else
        echo "无虚拟环境，跳过删除。"
    fi
else
    echo "保留虚拟环境，跳过删除。"
fi

# 清理 __pycache__
echo "正在清理 __pycache__ 缓存..."
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} +
echo "缓存清理完成。"

# 如不存在 venv 则创建
if [ ! -d "$VENV_DIR" ]; then
    echo "正在创建新的虚拟环境..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "虚拟环境已存在，跳过创建。"
fi

# 使用虚拟环境的 Python
VENV_PY="$VENV_DIR/bin/python3"

# 升级 pip
echo "升级 pip..."
$VENV_PY -m pip install --upgrade pip

# 安装依赖
if [ -f "$REQ_FILE" ]; then
    echo "安装 requirements.txt（使用清华镜像）..."
    $VENV_PY -m pip install -r "$REQ_FILE" -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "未找到 requirements.txt，跳过依赖安装。"
fi

echo "虚拟环境处理完成。"
