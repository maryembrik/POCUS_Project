@echo off
REM Launch the POCUS Copilot application.
REM
REM `streamlit run app.py` fails on a standard Anaconda install because Anaconda's Scripts
REM directory is not on PATH. Worse, `python` on PATH often resolves to the Windows Store stub
REM at AppData\Local\Microsoft\WindowsApps\python.exe, which does not run anything -- it offers
REM to install Python from the Store. This script finds a real interpreter instead of relying
REM on either.

setlocal
set "PY="

REM Prefer an interpreter that actually has streamlit installed.
for %%P in (
  "%USERPROFILE%\anaconda3\python.exe"
  "%USERPROFILE%\miniconda3\python.exe"
  "%LOCALAPPDATA%\Continuum\anaconda3\python.exe"
  "C:\ProgramData\anaconda3\python.exe"
) do (
  if not defined PY if exist %%~P (
    %%~P -c "import streamlit" >nul 2>&1 && set "PY=%%~P"
  )
)

if not defined PY (
  echo.
  echo Could not find a Python interpreter with streamlit installed.
  echo.
  echo Run it directly with the full path to your interpreter, for example:
  echo     C:\Users\%USERNAME%\anaconda3\python.exe -m streamlit run app.py
  echo.
  exit /b 1
)

REM Required on this machine: two OpenMP runtimes are linked into the process (torch and MKL),
REM and without this the interpreter aborts at import with OMP error #15.
set KMP_DUPLICATE_LIB_OK=TRUE

echo Using %PY%
"%PY%" -m streamlit run "%~dp0app.py" %*
endlocal
