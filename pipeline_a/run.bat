@echo off
title mapgen webui
cd /d "%~dp0"
rem open the browser once the server has had a moment to bind
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start "" http://127.0.0.1:5000"
py -3.14 ..\webui\serve.py --backend mapgen.webui_adapter --root .
