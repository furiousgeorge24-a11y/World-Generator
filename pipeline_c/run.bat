@echo off
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0.") do set "PIPELINE_C_ROOT=%%~fI"
for %%I in ("%~dp0..\webui\serve.py") do set "WEBUI_SERVER=%%~fI"
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0prepare_webui.ps1" -Port 5002 -ServerScript "%WEBUI_SERVER%" -Backend webui_adapter -Root "%PIPELINE_C_ROOT%"
if errorlevel 1 exit /b %errorlevel%
start "" /b powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -Command ^
  "$url='http://127.0.0.1:5002/'; $probe=$url+'api/registry'; for ($attempt=0; $attempt -lt 120; $attempt++) { try { $meta=Invoke-RestMethod -Uri $probe -TimeoutSec 1; if ($meta.name -eq 'pipeline_c land-origin lab') { Start-Process $url; exit 0 } } catch {}; Start-Sleep -Milliseconds 250 }; exit 1"
py -3.14 "%WEBUI_SERVER%" --backend webui_adapter --root "%PIPELINE_C_ROOT%" --port 5002
