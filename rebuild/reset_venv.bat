@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ========== 配置区 ==========
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"
set "VENV_DIR=%PROJECT_DIR%\venv"
set "REQ_FILE=%PROJECT_DIR%\requirements.txt"
set "PYTHON=python"
:: ============================

echo 项目路径：%PROJECT_DIR%
echo 虚拟环境目录：%VENV_DIR%
echo.

:: 用户确认是否删除 venv
set /p CONFIRM=是否删除并重建 venv？(y/N):
if /i "%CONFIRM%"=="y" (
    if exist "%VENV_DIR%" (
        echo 正在删除虚拟环境...
        rmdir /s /q "%VENV_DIR%"
        echo 虚拟环境已删除。
    ) else (
        echo 无虚拟环境，跳过删除。
    )
) else (
    echo 保留虚拟环境，跳过删除。
)

:: 清理 __pycache__ 缓存
echo 正在清理 __pycache__ 缓存...
for /r "%PROJECT_DIR%" %%d in (.) do (
    if /i "%%~nxd"=="__pycache__" (
        rd /s /q "%%d"
    )
)
echo 缓存清理完成。

:: 如虚拟环境不存在则创建
if not exist "%VENV_DIR%" (
    echo 正在创建新的虚拟环境...
    %PYTHON% -m venv "%VENV_DIR%"
) else (
    echo 虚拟环境已存在，跳过创建。
)

:: 使用 venv 的 Python
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

:: 升级 pip
echo 升级 pip...
"%VENV_PY%" -m pip install --upgrade pip

:: 安装 requirements.txt
if exist "%REQ_FILE%" (
    echo 正在安装 requirements.txt 依赖（清华镜像）...
    "%VENV_PY%" -m pip install -r "%REQ_FILE%" -i https://pypi.tuna.tsinghua.edu.cn/simple
) else (
    echo 未找到 requirements.txt，跳过依赖安装。
)

echo 虚拟环境处理完成。
endlocal
pause
