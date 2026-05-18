@echo off
cd /d "%~dp0LLM"
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo Starting LLM server only...
echo.
echo This window closes when the LLM server stops.
echo LLM URL: http://127.0.0.1:8000
echo.
"%PYTHON_EXE%" -m uvicorn main:app --host 127.0.0.1 --port 8000
