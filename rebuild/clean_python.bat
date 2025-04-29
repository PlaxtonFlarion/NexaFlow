@echo off
setlocal enabledelayedexpansion

:: ========= 配置区 =========
:: 只需要设置小版本号，比如310, 311, 312
set "VERSION=311"
:: ==========================

:: 格式化成 3.11
set "MAJOR=%VERSION:~0,1%"
set "MINOR=%VERSION:~1,2%"
set "PYTHON_VERSION=%MAJOR%.%MINOR%"

echo ⚙️ 准备卸载 Python %PYTHON_VERSION% ...

:: 确认提示
set /p CONFIRM=⚠️  确认要彻底删除 Python %PYTHON_VERSION% 相关所有文件吗？(y/N):
if /i not "%CONFIRM%"=="y" (
    echo ❌ 操作取消，未进行任何删除。
    exit /b 0
)

echo ✅ 开始执行卸载...

:: 删除 Python 安装目录
set "PYTHON_DIR=C:\Program Files\Python%VERSION%"
if exist "%PYTHON_DIR%" (
    echo 🧹 删除 Framework (Python目录)...
    rmdir /s /q "%PYTHON_DIR%"
) else (
    echo ✅ Framework 已不存在。
)

:: 清理 C:\Program Files (x86)\Python Launcher
set "LAUNCHER_DIR=C:\Program Files (x86)\Python Launcher"
if exist "%LAUNCHER_DIR%" (
    echo 🧹 删除 Python Launcher...
    rmdir /s /q "%LAUNCHER_DIR%"
) else (
    echo ✅ Launcher 已不存在。
)

:: 删除用户 Library 缓存
set "USER_CACHE=%USERPROFILE%\AppData\Local\Programs\Python\Python%VERSION%"
if exist "%USER_CACHE%" (
    echo 🧹 清理用户 Library 缓存...
    rmdir /s /q "%USER_CACHE%"
) else (
    echo ✅ 无用户缓存目录。
)

:: 检查环境变量（PATH）
echo 🔍 检查环境变量（请手动清理）...
echo ----------------------------------
echo 请检查系统环境变量 PATH 是否包含:
echo "C:\Program Files\Python%VERSION%\"
echo "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python%VERSION%\"
echo ----------------------------------

echo 🎯 Python %PYTHON_VERSION% 卸载完成！

endlocal
pause