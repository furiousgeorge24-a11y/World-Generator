@echo off
setlocal
cd /d "%~dp0"
py -3.14 ..\webui\serve.py --backend webui_adapter --root . --port 5002
