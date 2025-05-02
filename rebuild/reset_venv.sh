#!/bin/bash

# ========== 配置区 ==========
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
REQ_FILE="$PROJECT_DIR/requirements.txt"
PYTHON="python3"
# ============================

echo "项目路径：$PROJECT_DIR"
echo "虚拟环境目录：$VENV_DIR"

# 用户确认
read -r -p "确认要删除并重建 venv 吗？(y/N): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "操作已取消。"
    exit 0
fi

# 删除 venv
if [ -d "$VENV_DIR" ]; then
    echo "正在删除虚拟环境..."
    rm -rf "$VENV_DIR"
    echo "虚拟环境已删除。"
else
    echo "无虚拟环境，跳过删除。"
fi

# 删除 __pycache__
echo "正在清理 __pycache__ 缓存..."
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} +
echo "缓存清理完成。"

# 创建 venv
echo "正在创建新的虚拟环境..."
$PYTHON -m venv "$VENV_DIR"

# 使用 venv 的 python 执行 pip 安装
VENV_PY="$VENV_DIR/bin/python3"

# 升级 pip
echo "升级 pip..."
$VENV_PY -m pip install --upgrade pip

# 升级 setuptools wheel
echo "升级 setuptools 和 wheel..."
$VENV_PY -m pip install --upgrade setuptools wheel

# 安装 requirements.txt
if [ -f "$REQ_FILE" ]; then
    echo "安装 requirements.txt（使用清华镜像）..."
    $VENV_PY -m pip install -r "$REQ_FILE" -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "未找到 requirements.txt，跳过依赖安装。"
fi

echo "虚拟环境重建完成。"