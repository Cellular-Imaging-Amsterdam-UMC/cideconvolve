@echo off
setlocal EnableExtensions

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

if not exist "%REPO_ROOT%\version.txt" (
    echo ERROR: version.txt is missing.
    exit /b 1
)
set /p VERSION=<"%REPO_ROOT%\version.txt"
if not defined VERSION (
    echo ERROR: version.txt is empty.
    exit /b 1
)
set "SKIP_BUILD=0"
set "DRY_RUN=0"
set "CONFIRMED=0"

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--skip-build" (
    set "SKIP_BUILD=1"
    shift
    goto :parse_args
)
if /I "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto :parse_args
)
if /I "%~1"=="--yes" (
    set "CONFIRMED=1"
    shift
    goto :parse_args
)
echo ERROR: Unknown argument: %~1
exit /b 1

:args_done
powershell -NoProfile -Command "$v=$env:VERSION; if ($v -notmatch '\Av?(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?\z') { exit 1 }"
if errorlevel 1 (
    echo ERROR: Version must be SemVer with an optional v prefix and no build metadata, for example v1.2.3 or v1.2.3-rc.1.
    exit /b 1
)
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$p=Join-Path $env:REPO_ROOT 'config.yaml'; $text=Get-Content $p -Raw; $org=[regex]::Match($text,'(?ms)^docker_image:\s*.*?^\s*org:\s*([^\r\n#]+)').Groups[1].Value.Trim(); $name=[regex]::Match($text,'(?ms)^docker_image:\s*.*?^\s*name:\s*([^\r\n#]+)').Groups[1].Value.Trim(); $image=$org+'/'+$name; if ($image -ceq 'cellularimagingcf/w_cideconvolve') { $image } else { exit 1 }"`) do set "FULL_IMAGE=%%I"
if not defined FULL_IMAGE (
    echo ERROR: Could not determine Docker image from config.yaml.
    exit /b 1
)
for /f "tokens=1,2 delims=/" %%A in ("%FULL_IMAGE%") do set "IMAGE_NAME=%%B"
if not defined IMAGE_NAME (
    echo ERROR: Invalid Docker image %FULL_IMAGE%.
    exit /b 1
)
if /I not "%FULL_IMAGE%"=="cellularimagingcf/w_cideconvolve" (
    echo ERROR: Refusing to publish to unexpected Docker repository: %FULL_IMAGE%
    echo Expected: cellularimagingcf/w_cideconvolve
    exit /b 1
)

echo Docker image: %FULL_IMAGE%
echo Version: %VERSION%
if "%DRY_RUN%"=="0" if "%CONFIRMED%"=="0" (
    echo ERROR: Publishing Docker images requires explicit confirmation.
    echo Re-run with --yes, or use --dry-run to inspect all commands safely.
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 exit /b 1

if "%SKIP_BUILD%"=="0" (
    if "%DRY_RUN%"=="1" (
        echo [dry-run] call builddocker.cmd
    ) else (
        call builddocker.cmd
        if errorlevel 1 goto :build_failed
    )
    call :run_or_echo docker build -f Dockerfile.gradio --build-arg BASE_IMAGE="%IMAGE_NAME%:%VERSION%" -t "%IMAGE_NAME%:%VERSION%-gradio" -t "%IMAGE_NAME%:latest-gradio" .
    if errorlevel 1 goto :build_failed
    call :run_or_echo docker build -f Dockerfile.jupyter --build-arg BASE_IMAGE="%IMAGE_NAME%:%VERSION%" -t "%IMAGE_NAME%:%VERSION%-jupyter" -t "%IMAGE_NAME%:latest-jupyter" .
    if errorlevel 1 goto :build_failed
)

call :tag_and_push "%IMAGE_NAME%:%VERSION%" "%FULL_IMAGE%:%VERSION%"
if errorlevel 1 goto :push_failed
call :tag_and_push "%IMAGE_NAME%:latest" "%FULL_IMAGE%:latest"
if errorlevel 1 goto :push_failed
call :tag_and_push "%IMAGE_NAME%:%VERSION%-cu130" "%FULL_IMAGE%:%VERSION%-cu130"
if errorlevel 1 goto :push_failed
call :tag_and_push "%IMAGE_NAME%:latest-cu130" "%FULL_IMAGE%:latest-cu130"
if errorlevel 1 goto :push_failed
call :tag_and_push "%IMAGE_NAME%:%VERSION%-gradio" "%FULL_IMAGE%:%VERSION%-gradio"
if errorlevel 1 goto :push_failed
call :tag_and_push "%IMAGE_NAME%:latest-gradio" "%FULL_IMAGE%:latest-gradio"
if errorlevel 1 goto :push_failed
call :tag_and_push "%IMAGE_NAME%:%VERSION%-jupyter" "%FULL_IMAGE%:%VERSION%-jupyter"
if errorlevel 1 goto :push_failed
call :tag_and_push "%IMAGE_NAME%:latest-jupyter" "%FULL_IMAGE%:latest-jupyter"
if errorlevel 1 goto :push_failed

popd >nul
if "%DRY_RUN%"=="1" (
    echo Dry run completed for all Docker variants of %FULL_IMAGE% version %VERSION%.
) else (
    echo Pushed all Docker variants for %FULL_IMAGE% version %VERSION%.
)
endlocal & exit /b 0

:run_or_echo
if "%DRY_RUN%"=="1" (
    echo [dry-run] %*
    exit /b 0
)
echo Running: %*
%*
exit /b %ERRORLEVEL%

:tag_and_push
set "LOCAL_TAG=%~1"
set "REMOTE_TAG=%~2"
call :run_or_echo docker tag "%LOCAL_TAG%" "%REMOTE_TAG%"
if errorlevel 1 exit /b 1
call :run_or_echo docker push "%REMOTE_TAG%"
exit /b %ERRORLEVEL%

:build_failed
echo ERROR: Docker build failed.
popd >nul
endlocal & exit /b 1

:push_failed
echo ERROR: Docker tag or push failed. Run docker login and verify access to:
echo   %FULL_IMAGE%
popd >nul
endlocal & exit /b 1