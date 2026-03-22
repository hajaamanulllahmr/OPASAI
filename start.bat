@echo off
title OPASAI — Offline Personal AI Assistant
color 0B
cd /d "%~dp0"

echo.
echo  ============================================
echo    OPASAI — Offline Personal AI Assistant
echo  ============================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python is not installed.
    echo  Download Python 3.10 or newer from: https://www.python.org/downloads/
    echo  Make sure to tick "Add Python to PATH" during install.
    echo.
    pause & exit /b 1
)
python --version

:: ── Install dependencies ──────────────────────────────────────────────────────
echo.
echo [2/4] Installing Python packages...
pip install -r requirements.txt -q --no-warn-script-location
echo  Done.

:: ── Check Ollama ─────────────────────────────────────────────────────────────
echo.
echo [3/4] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Ollama is not installed.
    echo  Download it from: https://ollama.com/download
    echo  Install it, then re-run this script.
    echo.
    pause & exit /b 1
)
echo  Ollama found.

:: Start Ollama server if it isn't already running
curl -s --connect-timeout 2 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo  Starting Ollama server in the background...
    start /min "Ollama" ollama serve
    echo  Waiting for Ollama to start...
    timeout /t 5 /nobreak >nul
) else (
    echo  Ollama is already running.
)

:: Pull Phi-3 if it hasn't been downloaded yet
echo.
echo [4/4] Checking Phi-3 model...
ollama list 2>nul | findstr /i "phi3" >nul
if errorlevel 1 (
    echo  Phi-3 not found — downloading now (about 2.2 GB, one-time only)...
    ollama pull phi3
    if errorlevel 1 (
        echo  ERROR: Failed to download Phi-3. Check your internet connection.
        pause & exit /b 1
    )
) else (
    echo  Phi-3 model is ready.
)

:: ── Choose interface ──────────────────────────────────────────────────────────
echo.
echo  ============================================
echo   All checks passed! How do you want to run?
echo.
echo   [1] Desktop App   (native window — recommended)
echo   [2] Web UI        (browser at http://localhost:8000)
echo   [3] Terminal CLI  (command line)
echo  ============================================
echo.
set /p CHOICE=  Your choice (1/2/3) [default=1]: 

if "%CHOICE%"=="2" goto :web
if "%CHOICE%"=="3" goto :cli

:desktop
echo.
echo  Launching OPASAI Desktop App...
python desktop_app.py
goto :end

:web
echo.
echo  Launching OPASAI Web Server...
start "" http://localhost:8000
python server.py
goto :end

:cli
echo.
echo  Launching OPASAI Terminal...
python cli.py
goto :end

:end
echo.
echo  OPASAI has exited. Press any key to close.
pause
