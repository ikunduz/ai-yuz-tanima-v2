@echo off
setlocal
set ROOT=%~dp0\..

if not exist "%ROOT%\.venv\Scripts\python.exe" (
  echo .venv bulunamadi.
  echo Once Python 64-bit kurup su komutlari calistirin:
  echo   py -3.10 -m venv .venv
  echo   .venv\Scripts\python -m pip install -U pip setuptools wheel pyinstaller -r requirements.txt
  exit /b 1
)

"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\build_windows_apps.py"
exit /b %errorlevel%
