@echo off
cd /d "%~dp0"
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo Starting Cooking Agent...
echo.
echo Keep this window open.
echo The LLM server starts automatically in the background.
echo Browser URL: http://127.0.0.1:3000
echo.
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:3000"
cd /d "%~dp0Web-main\Front"
"%PYTHON_EXE%" abc.py
pause
