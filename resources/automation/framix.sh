#!/bin/bash

# 设置路径
EXE_PATH="$(dirname "$0")/framix"

# 检查是否存在
if [ ! -f "$EXE_PATH" ]; then
    echo "WARN: not found $EXE_PATH"
    read -p "请按任意键继续..."
    exit 1
fi

# 运行
"$EXE_PATH"

# 等待用户输入以继续
read -p "请按任意键继续..."