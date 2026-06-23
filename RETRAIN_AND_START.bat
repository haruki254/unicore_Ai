@echo off
title Trading Intelligence — Retrain + Start
cd /d "%~dp0"
echo.
echo ============================================
echo   Retraining AI models then starting server
echo ============================================
echo.
python run.py --retrain
pause
