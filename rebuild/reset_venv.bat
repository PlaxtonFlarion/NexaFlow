@echo off
setlocal enabledelayedexpansion

:: ========== 配置区（全局变量） ==========
:: 项目根目录（本脚本在 rebuild/ 目录，回到上一级）
set "PROJECT_DIR=%~dp0.."

:: 虚拟环境目录
set "VENV_DIR=%PROJECT_DIR%\venv"

:: requirements.txt文件
set "REQ_FILE=%PROJECT_DIR%\requirements.txt"

:: 指定 Python 可执行文件
set "PYTHON=python"
:: =========================================

echo ⚙️ 项目路径检测中：%PROJECT_DIR%
echo ⚙️ 虚拟环境目录：%VENV_DIR%

:: 确认操作
set /p CONFIRM=⚠️  确认要删除 venv 并重建环境吗？(y/N): 
if /i not "%CONFIRM%"=="y" (
    echo ❌ 操作取消。
    exit /b 0
)

:: 删除 venv
if exist "%VENV_DIR%" (
    echo 🧹 正在删除虚拟环境...
    rmdir /s /q "%VENV_DIR%"
    echo ✅ venv 已删除。
) else (
    echo ✅ venv 不存在，无需删除。
)

:: 删除 __pycache__ 缓存
echo 🧹 正在清理 __pycache__ ...
for /r "%PROJECT_DIR%" %%d in (.) do (
    if "%%~nxd"=="__pycache__" (
        rd /s /q "%%d"
    )
)
echo ✅ __pycache__ 清理完成。

:: 创建新的虚拟环境
echo 📦 创建新的虚拟环境...
%PYTHON% -m venv "%VENV_DIR%"

:: 激活虚拟环境
call "%VENV_DIR%\Scripts\activate.bat"

:: 升级 pip、setuptools、wheel
echo 🚀 升级 pip setuptools wheel...
pip install --upgrade pip setuptools wheel

:: 安装 requirements.txt，使用清华镜像
if exist "%REQ_FILE%" (
    echo 📥 使用清华镜像安装依赖 requirements.txt ...
    pip install -r "%REQ_FILE%" -i https://pypi.tuna.tsinghua.edu.cn/simple
) else (
    echo ⚠️ 没有找到 requirements.txt，跳过安装。
)

echo 🎯 虚拟环境重建完成！
endlocal
pause