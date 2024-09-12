@echo off

:: 设置路径
set EXE_PATH="%~dp0framix.dist\framix.exe"

:: 检查是否存在
if not exist %EXE_PATH% (
    echo WARN: not found %EXE_PATH%
    pause
    exit /b
)

:: 运行
%EXE_PATH%

:: 等待用户输入以继续
pause