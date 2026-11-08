@echo off
setlocal
cd /d "%~dp0"
set "COMPUTE=%~1"
if not defined COMPUTE set "COMPUTE=cpu"
where py >nul 2>&1
if not errorlevel 1 (
    py -3 tools\setup.py --compute "%COMPUTE%"
) else (
    python tools\setup.py --compute "%COMPUTE%"
)
if errorlevel 1 (
    echo Setup failed. Install Python 3.11-3.13 and run python tools\setup.py --compute %COMPUTE%.
    pause
    exit /b 1
)
pause
