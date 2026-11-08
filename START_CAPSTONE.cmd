@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
set "CAPSTONE_URL=http://127.0.0.1:8765/"

rem If Capstone is already running, do not start a second server.
powershell -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%CAPSTONE_URL%'; if ($response.StatusCode -eq 200) { exit 0 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    start "" "%CAPSTONE_URL%"
    exit /b 0
)

rem Prefer the project environment only when its Python runtime is healthy.
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -c "import encodings" >nul 2>&1
    if not errorlevel 1 goto python_ready
)

rem The original virtual environment may point to an expired Codex runtime.
rem Fall back to the user's installed Python instead of failing silently.
set "PYTHON_EXE="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE (
    echo ERROR: No working Python installation was found.
    echo Install Python 3.12 or newer, then run this file again.
    pause
    exit /b 1
)

:python_ready
"%PYTHON_EXE%" -c "import traffic_edge.web_ui" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was found, but the Capstone dependencies are unavailable.
    echo Python used: %PYTHON_EXE%
    echo See README.md for the installation command.
    pause
    exit /b 1
)

echo Starting Capstone Traffic Event Analyzer...
echo Python: %PYTHON_EXE%
echo URL: %CAPSTONE_URL%
"%PYTHON_EXE%" -m traffic_edge.web_ui --host 127.0.0.1 --port 8765
if errorlevel 1 (
    echo.
    echo Capstone stopped because an error occurred.
    pause
)
