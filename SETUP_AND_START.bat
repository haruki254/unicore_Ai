@echo off
title Trading Intelligence — Full Setup + Start
cd /d "%~dp0"
echo.
echo ============================================
echo   Trading Intelligence System
echo   Full setup: generate data + train + start
echo ============================================
echo.
python run.py --fresh
pause