@echo off
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0.") do set "PIPELINE_C_ROOT=%%~fI"
for %%I in ("%~dp0search_server.py") do set "SEARCH_SERVER=%%~fI"
rem The search runs its worlds in a process pool. A reloader would restart
rem this process on any edit, leak the pool's children, and orphan the run,
rem so the server pins a single process and never enables one.
set "WEBUI_RELOAD=0"
rem One BLAS thread per process: the block model solves a dense system
rem every step, and OpenBLAS threads across eight workers starve the run.
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0prepare_webui.ps1" -Port 5004 -ServerScript "%SEARCH_SERVER%" -Backend explore_adapter -Root "%PIPELINE_C_ROOT%" -RegistryName "pipeline_c regime search"
if errorlevel 1 exit /b %errorlevel%
start "" /b powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -Command ^
  "$url='http://127.0.0.1:5004/'; $probe=$url+'api/config'; for ($attempt=0; $attempt -lt 120; $attempt++) { try { $config=Invoke-RestMethod -Uri $probe -TimeoutSec 1; if ($config.knobs) { Start-Process $url; exit 0 } } catch {}; Start-Sleep -Milliseconds 250 }; exit 1"
py -3.14 "%SEARCH_SERVER%" --backend explore_adapter --root "%PIPELINE_C_ROOT%" --port 5004
