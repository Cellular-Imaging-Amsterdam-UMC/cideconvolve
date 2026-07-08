@echo off
setlocal
set "PYTHON=C:\Users\p000881\AppData\Local\miniconda3\envs\deconvolve\python.exe"
set "SCRIPT=%~dp0leica_nyquist_finder.py"
"%PYTHON%" "%SCRIPT%" %*
