@echo off
title AI Offline Assistant
set KMP_DUPLICATE_LIB_OK=TRUE

echo ============================================
echo   AI Offline Personal Assistant
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://www.python.org
    pause
    exit /b 1
)

echo [1/4] Checking and installing required packages...
pip install faster-whisper sounddevice numpy pyttsx3 customtkinter pillow SpeechRecognition pyaudio ctranslate2 fastapi uvicorn --quiet --disable-pip-version-check

echo.
echo [2/4] Extracting profile data...
cd /d "%~dp0"
python -c "import zipfile, os; z=zipfile.ZipFile('data/sk.zip'); z.extractall('data')" 2>nul

echo.
echo [3/4] Make sure Ollama is running (ollama serve) before chatting.
echo.
echo [4/4] Launching Desktop App...
echo.

python desktop_app.py

if errorlevel 1 (
    echo.
    echo [ERROR] App crashed. Check errors above.
    pause
)