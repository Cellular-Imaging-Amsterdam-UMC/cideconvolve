@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

REM Local Docker image name used by config.yaml
set "IMAGE_NAME=w_cideconvolve"

REM Read version from version.txt
set /p VERSION=<"%REPO_ROOT%\version.txt"
if not defined VERSION (
    echo ERROR: version.txt is empty or missing
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to change directory to %REPO_ROOT%
    exit /b 1
)

echo Building CUDA 13.2 headless %IMAGE_NAME%:%VERSION% and %IMAGE_NAME%:latest
docker build ^
    %* ^
    --build-arg PYTORCH_VERSION=2.13.0 ^
    --build-arg PYTORCH_CUDA=cu132 ^
    -t %IMAGE_NAME%:%VERSION% ^
    -t %IMAGE_NAME%:latest ^
    .
if errorlevel 1 (
    popd >nul
    endlocal & exit /b 1
)

echo Building Jupyter %IMAGE_NAME%:%VERSION%-jupyter and %IMAGE_NAME%:latest-jupyter
docker build ^
    -f Dockerfile.jupyter ^
    --build-arg BASE_IMAGE=%IMAGE_NAME%:%VERSION% ^
    -t %IMAGE_NAME%:%VERSION%-jupyter ^
    -t %IMAGE_NAME%:latest-jupyter ^
    %* .
set "EXITCODE=%ERRORLEVEL%"

popd >nul
endlocal & exit /b %EXITCODE%
