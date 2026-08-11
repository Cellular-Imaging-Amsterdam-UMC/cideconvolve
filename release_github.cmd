@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"
if not exist "%REPO_ROOT%\version.txt" (
    echo ERROR: version.txt is missing.
    exit /b 1
)
set /p TAG=<"%REPO_ROOT%\version.txt"
if not defined TAG (
    echo ERROR: version.txt is empty.
    exit /b 1
)
set "PYTHON_EXE=C:\Users\p000881\AppData\Local\miniconda3\envs\deconvolve\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: The deconvolve Python interpreter was not found: %PYTHON_EXE%
    exit /b 1
)
set "DRY_RUN=0"
set "CONFIRMED=0"

:parse_args
if "%~1"=="" goto :args_done
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
powershell -NoProfile -Command "$v=$env:TAG; if ($v -notmatch '\Av?(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?\z') { exit 1 }"
if errorlevel 1 (
    echo ERROR: version.txt must contain SemVer with an optional v prefix and no build metadata, for example v1.2.3 or v1.2.3-rc.1.
    exit /b 1
)
set "CONFIG_SYNC_NEEDED=0"
"%PYTHON_EXE%" "%REPO_ROOT%\tools\sync_config_version.py" "%REPO_ROOT%\config.yaml" "%TAG%" --check >nul
set "CONFIG_SYNC_EXIT=!ERRORLEVEL!"
if "!CONFIG_SYNC_EXIT!"=="2" (
    set "CONFIG_SYNC_NEEDED=1"
) else if not "!CONFIG_SYNC_EXIT!"=="0" (
    echo ERROR: Could not validate the Docker tag in config.yaml.
    exit /b 1
)
if "%DRY_RUN%"=="0" if "%CONFIRMED%"=="0" (
    echo ERROR: Creating and publishing a GitHub release requires explicit confirmation.
    echo Re-run with --yes, or use --dry-run to validate without changing GitHub.
    exit /b 1
)

pushd "%REPO_ROOT%" >nul
if errorlevel 1 exit /b 1

where git >nul 2>nul
if errorlevel 1 (
    echo ERROR: git was not found in PATH.
    goto :fail
)
where gh >nul 2>nul
if errorlevel 1 if exist "%ProgramFiles%\GitHub CLI\gh.exe" set "PATH=%ProgramFiles%\GitHub CLI;%PATH%"
where gh >nul 2>nul
if errorlevel 1 (
    echo ERROR: GitHub CLI ^(gh^) was not found. Install it from https://cli.github.com/
    goto :fail
)
gh auth status >nul 2>nul
if errorlevel 1 (
    echo ERROR: GitHub CLI is not authenticated. Run: gh auth login
    goto :fail
)

git remote get-url origin | powershell -NoProfile -Command "$u=($input | Out-String).Trim(); $allowed=@('https://github.com/Cellular-Imaging-Amsterdam-UMC/cideconvolve.git','git@github.com:Cellular-Imaging-Amsterdam-UMC/cideconvolve.git'); if ($allowed -notcontains $u) { exit 1 }"
if errorlevel 1 (
    echo ERROR: origin does not point to Cellular-Imaging-Amsterdam-UMC/cideconvolve.
    goto :fail
)
echo Origin repository verified: Cellular-Imaging-Amsterdam-UMC/cideconvolve

git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>nul
if errorlevel 1 (
    echo ERROR: Current branch has no upstream. Run: git push -u origin HEAD
    goto :fail
)
git update-index -q --refresh
set "DIRTY="
for /f "usebackq delims=" %%A in (`git status --porcelain`) do set "DIRTY=1"
if defined DIRTY (
    echo ERROR: Working tree has uncommitted changes.
    git status --short
    goto :fail
)

echo Fetching latest remote refs...
git fetch --prune
if errorlevel 1 goto :fail
for /f "tokens=1,2" %%A in ('git rev-list --left-right --count HEAD...@{u}') do (
    set "AHEAD=%%A"
    set "BEHIND=%%B"
)
if not "!AHEAD!"=="0" (
    echo ERROR: Local branch is !AHEAD! commit^(s^) ahead. Push first.
    goto :fail
)
if not "!BEHIND!"=="0" (
    echo ERROR: Local branch is !BEHIND! commit^(s^) behind. Pull or rebase first.
    goto :fail
)

git rev-parse -q --verify "refs/tags/%TAG%" >nul 2>nul
if not errorlevel 1 (
    echo ERROR: Local tag %TAG% already exists.
    goto :fail
)
git ls-remote --exit-code --tags origin "refs/tags/%TAG%" >nul 2>nul
if not errorlevel 1 (
    echo ERROR: Remote tag %TAG% already exists.
    goto :fail
)
gh release view "%TAG%" >nul 2>nul
if not errorlevel 1 (
    echo ERROR: GitHub release %TAG% already exists.
    goto :fail
)

if "%CONFIG_SYNC_NEEDED%"=="1" (
    echo Synchronizing config.yaml Docker tag with %TAG%...
    call :run_or_echo "%PYTHON_EXE%" "%REPO_ROOT%\tools\sync_config_version.py" "%REPO_ROOT%\config.yaml" "%TAG%"
    if errorlevel 1 goto :fail
    call :run_or_echo git add -- config.yaml
    if errorlevel 1 goto :fail
    call :run_or_echo git commit -m "Set config Docker tag to %TAG%"
    if errorlevel 1 goto :fail
    call :run_or_echo git push origin HEAD
    if errorlevel 1 goto :fail
)

echo Creating annotated tag %TAG%...
call :run_or_echo git tag -a "%TAG%" -m "Release %TAG%"
if errorlevel 1 goto :fail
echo Pushing tag %TAG%...
call :run_or_echo git push origin "%TAG%"
if errorlevel 1 goto :fail_after_tag
echo Creating GitHub release %TAG%...
call :run_or_echo gh release create "%TAG%" --verify-tag --title "%TAG%" --generate-notes
if errorlevel 1 goto :fail_after_push

popd >nul
if "%DRY_RUN%"=="1" (
    echo Dry run completed. No tag or GitHub release was created.
) else (
    echo Done. GitHub release %TAG% created.
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

:fail_after_push
echo ERROR: Release creation failed; tag %TAG% is already on origin.
goto :fail
:fail_after_tag
echo ERROR: Push failed; local tag %TAG% was created.
:fail
set "EXITCODE=%ERRORLEVEL%"
if "%EXITCODE%"=="0" set "EXITCODE=1"
popd >nul
endlocal & exit /b %EXITCODE%