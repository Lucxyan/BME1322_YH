@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -X utf8 gui.py %*
) else (
    python -X utf8 gui.py %*
)
pause
