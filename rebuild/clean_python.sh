#!/bin/bash

# 设置要卸载的Python版本（比如310、311）
VERSION=311

# 格式化成3.11这样的
MAJOR=${VERSION:0:1}
MINOR=${VERSION:1:2}
PYTHON_VERSION="$MAJOR.$MINOR"

echo "⚙️ 准备卸载 Python $PYTHON_VERSION ..."

# 增加确认提示
read -r -p "⚠️  确认要彻底删除 Python $PYTHON_VERSION 相关所有文件吗？(y/N): " confirm

# 判断用户输入
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "❌ 操作取消，未进行任何删除。"
    exit 0
fi

echo "✅ 开始执行卸载..."

# 删除 /Library/Frameworks 里的Python.framework
if [ -d "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION" ]; then
    echo "🧹 删除 Framework..."
    sudo rm -rf "/Library/Frameworks/Python.framework/Versions/$PYTHON_VERSION"
else
    echo "✅ Framework 已不存在。"
fi

# 删除 /usr/local/bin 软链接
echo "🧹 清理 /usr/local/bin 软链接..."
sudo find /usr/local/bin -lname "*/Python.framework/Versions/$PYTHON_VERSION/*" -delete

# 删除 /Applications 里的Launcher和IDLE
if [ -d "/Applications/Python $PYTHON_VERSION" ]; then
    echo "🧹 删除 /Applications/Python $PYTHON_VERSION..."
    sudo rm -rf "/Applications/Python $PYTHON_VERSION"
else
    echo "✅ /Applications 里无此版本启动器。"
fi

# 删除用户的 Library 缓存
if [ -d "$HOME/Library/Python/$PYTHON_VERSION" ]; then
    echo "🧹 清理用户Library缓存..."
    rm -rf "$HOME/Library/Python/$PYTHON_VERSION"
else
    echo "✅ 无用户缓存目录。"
fi

# 检查.zshrc 和 .bash_profile
echo "🔍 检查环境变量..."
if grep -q "$PYTHON_VERSION" ~/.zshrc 2>/dev/null; then
    echo "⚠️  检测到 ~/.zshrc 包含Python $PYTHON_VERSION的路径，请手动编辑清理。"
fi
if grep -q "$PYTHON_VERSION" ~/.bash_profile 2>/dev/null; then
    echo "⚠️  检测到 ~/.bash_profile 包含Python $PYTHON_VERSION的路径，请手动编辑清理。"
fi

echo "🎯 Python $PYTHON_VERSION 卸载完成！"