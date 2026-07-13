@echo off
setlocal
cd /d "%~dp0\..\.."
set "PY=C:\Users\p000881\AppData\Local\miniconda3\envs\deconvolve\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" tools\omero_import_metadata_probe\run_dividingcell_wrapper_roundtrip.py %*
