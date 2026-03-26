@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Read version from version.txt
set /p VERSION=<"%SCRIPT_DIR%\version.txt"
if not defined VERSION (
    echo ERROR: version.txt is empty or missing
    exit /b 1
)

REM Allow override via command-line argument
if not "%~1"=="" set "VERSION=%~1"

REM Derive full image path from descriptor.json
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Content '%SCRIPT_DIR%\descriptor.json' | ConvertFrom-Json).'container-image'.image"`) do set "DESC_IMAGE=%%I"
REM Split into namespace and name
for /f "tokens=1,2 delims=/" %%A in ("%DESC_IMAGE%") do (
    set "DESC_NS=%%A"
    set "DESC_NAME=%%B"
)
if not defined IMAGE_NAME set "IMAGE_NAME=%DESC_NAME%"
if not defined IMAGE_NAMESPACE set "IMAGE_NAMESPACE=%DESC_NS%"
if "%IMAGE_NAMESPACE%"=="" (
    set "FULL_IMAGE=%IMAGE_NAME%"
) else (
    set "FULL_IMAGE=%IMAGE_NAMESPACE%/%IMAGE_NAME%"
)

REM Tag specific version
if not "%~1"=="--skip-build" if not "%~2"=="--skip-build" (
    docker build -t "%IMAGE_NAME%" .
    if errorlevel 1 (
        echo Build failed
        exit /b 1
    )
)

docker tag "%IMAGE_NAME%" "%FULL_IMAGE%:%VERSION%"
docker tag "%IMAGE_NAME%" "%FULL_IMAGE%:latest"

echo Pushing %FULL_IMAGE%:%VERSION% and %FULL_IMAGE%:latest
docker push "%FULL_IMAGE%:%VERSION%"
docker push "%FULL_IMAGE%:latest"

endlocal
