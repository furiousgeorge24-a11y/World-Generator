@echo off
title pipeline_b webui
cd /d "%~dp0"
rem port 5001 so pipeline_a's preview (5000) can run alongside for comparison
rem open the browser once the server has had a moment to bind
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start "" http://127.0.0.1:5001"
py -3.14 ..\webui\serve.py --backend webui_adapter --root . --port 5001
