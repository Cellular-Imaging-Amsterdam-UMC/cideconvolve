@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

REM Local Docker image name used by config.yaml
set "IMAGE_NAME=w_cideconvolve"

REM Read version from version.txt
set /p VERSION=<"%REPO_ROOT%\version.txt"

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to change directory to %REPO_ROOT%
    exit /b 1
)

echo Building CUDA 13.2 primary image
docker build ^
    %* ^
    --build-arg PYTORCH_VERSION=2.13.0 ^
    --build-arg PYTORCH_CUDA=cu132 ^
    --build-arg CUDA_TOOLKIT_IMAGE=nvidia/cuda:13.2.0-devel-ubuntu22.04 ^
    --build-arg CUDA_HOME_PATH=/usr/local/cuda-13.2 ^
    -t %IMAGE_NAME%:%VERSION% ^
    -t %IMAGE_NAME%:latest ^
    .
if errorlevel 1 (
    popd >nul
    endlocal & exit /b 1
)

echo Building CUDA 13.0 cluster fallback image
docker build ^
    %* ^
    --build-arg PYTORCH_VERSION=2.13.0 ^
    --build-arg PYTORCH_CUDA=cu130 ^
    --build-arg CUDA_TOOLKIT_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu22.04 ^
    --build-arg CUDA_HOME_PATH=/usr/local/cuda-13.0 ^
    -t %IMAGE_NAME%:%VERSION%-cu130 ^
    -t %IMAGE_NAME%:latest-cu130 ^
    .
set "EXITCODE=%ERRORLEVEL%"

popd >nul
endlocal & exit /b %EXITCODE%
