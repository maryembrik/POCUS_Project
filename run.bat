@echo off
REM Launch POCUS-Emergency. One deployment, one port: http://localhost:8501
REM
REM `python serve.py` alone usually fails on this machine for two reasons that look the same:
REM Anaconda's Scripts directory is not on PATH, and `python` on PATH often resolves to the
REM Windows Store stub at AppData\Local\Microsoft\WindowsApps\python.exe, which runs nothing
REM and offers to install Python instead. This script finds a real interpreter.

setlocal
set "PY="

for %%P in (
  "%USERPROFILE%\anaconda3\python.exe"
  "%USERPROFILE%\miniconda3\python.exe"
  "%LOCALAPPDATA%\Continuum\anaconda3\python.exe"
  "C:\ProgramData\anaconda3\python.exe"
) do (
  if not defined PY if exist %%~P (
    %%~P -c "import fastapi" >nul 2>&1 && set "PY=%%~P"
  )
)

if not defined PY (
  echo.
  echo Could not find a Python interpreter with fastapi installed.
  echo Run it directly, for example:
  echo     C:\Users\%USERNAME%\anaconda3\python.exe serve.py
  echo.
  exit /b 1
)

REM Required here: torch and MKL each link their own OpenMP runtime, and without this the
REM interpreter aborts at import with OMP error #15.
set KMP_DUPLICATE_LIB_OK=TRUE
set PORT=8501

echo Using %PY%
echo Open http://localhost:%PORT%
"%PY%" "%~dp0serve.py"
endlocal
