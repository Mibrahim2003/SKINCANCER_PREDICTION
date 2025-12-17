@echo off
setlocal
cd /d "%~dp0"

echo Starting Skin Cancer AI System...
echo The browser will open automatically.

:: 1) Build path to uvicorn inside the local venv
set UVICORN_PATH=%~dp0.venv\Scripts\uvicorn.exe

if not exist "%UVICORN_PATH%" (
	echo [ERROR] uvicorn not found at "%UVICORN_PATH%". Ensure the venv is created and dependencies installed.
	pause
	exit /b 1
)

:: 2) Start FastAPI server in the background on port 8000 (empty title argument avoids URI handling)
start "" /B "%UVICORN_PATH%" app.main:app --reload --port 8000

:: 3) Wait 3 seconds for the server to wake up
timeout /t 3 /nobreak >nul

:: 4) Open the browser to the local app
start http://127.0.0.1:8000

echo System Running. Close this window to stop the server.

endlocal
