@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Derive image name from descriptor.json (strip namespace for local build)
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Content '%SCRIPT_DIR%\descriptor.json' | ConvertFrom-Json).'container-image'.image.Split('/')[-1]"`) do set "IMAGE_NAME=%%I"

REM Read version from version.txt
set /p VERSION=<"%SCRIPT_DIR%\version.txt"

pushd "%SCRIPT_DIR%" >nul
if errorlevel 1 (
    echo Failed to change directory to %SCRIPT_DIR%
    exit /b 1
)

docker build -t %IMAGE_NAME%:%VERSION% -t %IMAGE_NAME%:latest %* .
set "EXITCODE=%ERRORLEVEL%"

popd >nul
endlocal & exit /b %EXITCODE%
