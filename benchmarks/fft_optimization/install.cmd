@echo off
setlocal
set "PY=C:\Users\p000881\AppData\Local\miniconda3\envs\deconvolve\python.exe"
"%PY%" -m pip install -r "%~dp0requirements.txt"
exit /b %errorlevel%

