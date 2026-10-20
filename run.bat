@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 py -3 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import faster_whisper" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import json; exit(0 if json.load(open('config.json', encoding='utf-8')).get('device') == 'cuda' else 1)" >nul 2>&1
if not errorlevel 1 (
    ".venv\Scripts\python.exe" -c "from importlib.metadata import version; version('nvidia-cublas-cu12'); version('nvidia-cudnn-cu12')" >nul 2>&1
    if errorlevel 1 (
        ".venv\Scripts\python.exe" -m pip install --index-url https://pypi.org/simple -r requirements-gpu-windows.txt
        if errorlevel 1 goto failed
    )
)
".venv\Scripts\python.exe" run.py
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%
:failed
echo Setup failed. Check Python and network access.
pause
exit /b 1
