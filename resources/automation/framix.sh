#!/bin/bash

# 设置路径
EXE_PATH="$(dirname "$0")/framix"

# 检查是否存在
if [ ! -f "$EXE_PATH" ]; then
    # 显示警告
    osascript -e 'tell application "Terminal"
        do script "echo WARN: not found '$EXE_PATH'"
        activate
    end tell'
    exit 1
fi

# 运行
osascript -e 'tell application "Terminal"
    do script "'"$EXE_PATH"'"
    activate
end tell'