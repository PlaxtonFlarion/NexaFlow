#!/bin/bash

# ========== 配置区 ==========
# 当前脚本在 rebuild/ 目录，要回到上一级
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# venv目录
VENV_DIR="$PROJECT_DIR/venv"

# requirements.txt路径
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

# Python可执行文件
PYTHON_BIN="python3"
# ============================

echo "⚙️ 准备清理虚拟环境: $VENV_DIR 和 __pycache__ ..."

read -r -p "⚠️  确认要删除 venv 并重建环境吗？(y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "❌ 操作取消，未进行任何更改。"
    exit 0
fi

# 删除 venv
if [ -d "$VENV_DIR" ]; then
    echo "🧹 正在删除虚拟环境目录..."
    rm -rf "$VENV_DIR"
    echo "✅ venv 删除完成。"
else
    echo "✅ venv 不存在，无需删除。"
fi

# 删除 __pycache__
echo "🧹 正在清理项目内 __pycache__ ..."
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} +
echo "✅ __pycache__ 清理完成。"

# 创建新的虚拟环境
echo "📦 创建新的虚拟环境..."
$PYTHON_BIN -m venv "$VENV_DIR"

# 激活 venv
echo "📢 激活虚拟环境..."
source "$VENV_DIR/bin/activate"

# 升级 pip setuptools wheel
echo "🚀 升级 pip setuptools wheel..."
pip install --upgrade pip setuptools wheel

# 安装 requirements.txt，使用清华源
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "📥 使用清华镜像安装 requirements.txt ..."
    pip install -r "$REQUIREMENTS_FILE" -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "⚠️ 未找到 requirements.txt，跳过依赖安装。"
fi

echo "🎯 虚拟环境初始化完成！"