@echo off
setlocal

set "DATA_PATH=%~dp0"
if "%DATA_PATH:~-1%"=="\" set "DATA_PATH=%DATA_PATH:~0,-1%"

REM Derive image name from descriptor.json (strip namespace for local run)
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Content '%DATA_PATH%\descriptor.json' | ConvertFrom-Json).'container-image'.image.Split('/')[-1]"`) do set "DEFAULT_IMAGE=%%I"
if "%IMAGE%"=="" set "IMAGE=%DEFAULT_IMAGE%"

docker run --rm --gpus all ^
	-v "%DATA_PATH%\infolder:/data/in" ^
	-v "%DATA_PATH%\outfolder:/data/out" ^
	-v "%DATA_PATH%\gtfolder:/data/gt" ^
	%IMAGE% ^
	--infolder /data/in ^
	--outfolder /data/out ^
	--gtfolder /data/gt ^
	--local ^
	%*

endlocal
