@echo off
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0.") do set "PIPELINE_C_ROOT=%%~fI"
for %%I in ("%~dp0..\webui\serve.py") do set "WEBUI_SERVER=%%~fI"
rem The exploration lab runs its worlds in a process pool. The shell's
rem reloader restarts the server process on any edit, which would leak the
rem pool's children, so this launcher pins a single process. The lab falls
rem back to sequential generation if it ever finds the reloader on.
set "WEBUI_RELOAD=0"
rem One BLAS thread per process: the block model solves a dense system
rem every step, and OpenBLAS threads across eight workers starve the run.
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0prepare_webui.ps1" -Port 5003 -ServerScript "%WEBUI_SERVER%" -Backend explore_adapter -Root "%PIPELINE_C_ROOT%" -RegistryName "pipeline_c exploration lab"
if errorlevel 1 exit /b %errorlevel%
start "" /b powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -Command ^
  "$url='http://127.0.0.1:5003/'; $probe=$url+'api/registry'; for ($attempt=0; $attempt -lt 120; $attempt++) { try { $meta=Invoke-RestMethod -Uri $probe -TimeoutSec 1; if ($meta.name -eq 'pipeline_c exploration lab') { Start-Process $url; exit 0 } } catch {}; Start-Sleep -Milliseconds 250 }; exit 1"
py -3.14 "%WEBUI_SERVER%" --backend explore_adapter --root "%PIPELINE_C_ROOT%" --port 5003
