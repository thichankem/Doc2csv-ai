@echo off
REM Doc2CSV-AI launcher - uses Anaconda Python (where deps are installed)
setlocal

set "PY=%USERPROFILE%\anaconda3\python.exe"

if not exist "%PY%" (
    echo [LOI] Khong tim thay Anaconda Python tai: %PY%
    echo Vui long sua bien PY trong run.bat tro toi python.exe co cai deps.
    pause
    exit /b 1
)

cd /d "%~dp0"
"%PY%" app.py
if errorlevel 1 pause
endlocal
