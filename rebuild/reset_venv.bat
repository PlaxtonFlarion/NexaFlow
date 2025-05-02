@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ========== 配置区 ==========
:: 获取项目根目录（rebuild/ 上一级）
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"
set "VENV_DIR=%PROJECT_DIR%\venv"
set "REQ_FILE=%PROJECT_DIR%\requirements.txt"
set "PYTHON=python"
:: ============================

echo 项目路径：%PROJECT_DIR%
echo 虚拟环境目录：%VENV_DIR%

:: 用户确认
set /p CONFIRM=确认要删除并重建 venv 吗？(y/N):
if /i not "%CONFIRM%"=="y" (
    echo 操作已取消。
    exit /b 0
)

:: 删除旧虚拟环境
if exist "%VENV_DIR%" (
    echo 正在删除虚拟环境...
    rmdir /s /q "%VENV_DIR%"
    echo 虚拟环境已删除。
) else (
    echo 无虚拟环境，跳过删除。
)

:: 删除 __pycache__ 缓存
echo 正在清理 __pycache__ 缓存...
for /r "%PROJECT_DIR%" %%d in (.) do (
    if /i "%%~nxd"=="__pycache__" (
        rd /s /q "%%d"
    )
)
echo 缓存清理完成。

:: 创建新的虚拟环境
echo 正在创建新的虚拟环境...
%PYTHON% -m venv "%VENV_DIR%"

:: 使用 venv 的 Python 安装依赖（无需激活）
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

:: 升级 pip
echo 升级 pip...
"%VENV_PY%" -m pip install --upgrade pip

:: 升级 setuptools 和 wheel
echo 升级 setuptools 和 wheel...
"%VENV_PY%" -m pip install --upgrade setuptools wheel

:: 安装 requirements.txt
if exist "%REQ_FILE%" (
    echo 正在安装 requirements.txt 依赖（清华镜像）...
    "%VENV_PY%" -m pip install -r "%REQ_FILE%" -i https://pypi.tuna.tsinghua.edu.cn/simple
) else (
    echo 未找到 requirements.txt，跳过依赖安装。
)

echo 虚拟环境重建完成。
endlocal
pause