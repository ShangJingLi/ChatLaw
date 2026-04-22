@echo off
chcp 65001 >nul
cd /d %~dp0
setlocal

set PY=python-3.11.9-embed-amd64\python.exe

echo =========================
echo ChatLaw 启动器
echo =========================

REM =========================
REM 0. 检查 Python
REM =========================
if not exist "%PY%" (
    echo [ERROR] 未找到内置 Python：%PY%
    pause
    exit /b 1
)

REM =========================
REM 1. 如果 bootstrap 存在，则初始化
REM =========================
if exist "bootstrap" (
    echo [INFO] 初始化环境...

    if not exist "bootstrap\get-pip.py" (
        echo [ERROR] bootstrap 存在但缺少 get-pip.py
        pause
        exit /b 1
    )

    REM 安装 pip
    %PY% bootstrap\get-pip.py
    if errorlevel 1 (
        echo [ERROR] pip 安装失败
        pause
        exit /b 1
    )

    REM 验证 pip
    %PY% -m pip --version >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] pip 安装后仍不可用，请检查 python311._pth
        pause
        exit /b 1
    )

    REM 升级 pip
    %PY% -m pip install --upgrade pip
    if errorlevel 1 (
        echo [ERROR] pip 升级失败
        pause
        exit /b 1
    )

    REM 安装依赖
    if not exist "requirements.txt" (
        echo [ERROR] 未找到 requirements.txt
        pause
        exit /b 1
    )

    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] requirements 安装失败
        pause
        exit /b 1
    )

    REM 初始化成功后才删除 bootstrap
    echo [INFO] 初始化完成，删除 bootstrap...
    rmdir /s /q bootstrap
    if errorlevel 1 (
        echo [WARN] bootstrap 删除失败，但不影响运行
    )
)

REM =========================
REM 2. 启动程序
REM =========================
echo [INFO] 启动程序...

REM 这里不再 pause，让成功启动时终端自动退出
%PY% launcher_gui.py
set EXITCODE=%ERRORLEVEL%

REM 如果 GUI 启动失败，则停住终端方便排查
if not "%EXITCODE%"=="0" (
    echo [ERROR] 程序运行失败，退出码：%EXITCODE%
    pause
    exit /b %EXITCODE%
)

exit /b 0