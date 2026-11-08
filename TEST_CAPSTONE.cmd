@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
"%~dp0.venv\Scripts\python.exe" -m compileall -q src tests
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" -m traffic_edge.web_ui --smoke-test
if errorlevel 1 exit /b 1
set "CAPSTONE_TEST_NODE="
for /f "delims=" %%N in ('where node 2^>nul') do if not defined CAPSTONE_TEST_NODE set "CAPSTONE_TEST_NODE=%%N"
if defined CAPSTONE_TEST_NODE (
    "%CAPSTONE_TEST_NODE%" "%~dp0tests\test_calibration_ui.js"
    if errorlevel 1 exit /b 1
    "%CAPSTONE_TEST_NODE%" "%~dp0tests\test_annotation_ui.js"
    if errorlevel 1 exit /b 1
    "%CAPSTONE_TEST_NODE%" "%~dp0tests\test_workspace_ui.js"
    if errorlevel 1 exit /b 1
    "%CAPSTONE_TEST_NODE%" "%~dp0tests\test_collision_ui.js"
    if errorlevel 1 exit /b 1
    "%CAPSTONE_TEST_NODE%" "%~dp0tests\test_study_ui.js"
    if errorlevel 1 exit /b 1
)
echo Source compilation, automated tests and the Web UI smoke test passed.
