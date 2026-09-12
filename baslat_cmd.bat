@echo off
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
title Bitcoin Puzzle CMD Motoru v3.0 - 16 CPU + AMD GPU
color 0B
chcp 65001 >nul
cls

echo ==============================================================================
echo   ⚡ BITCOIN PUZZLE ARKA PLAN MOTORU (CMD v3.0)
echo   🚀 16 CPU Cekirdegi (coincurve C) + AMD GPU Hizlandirma
echo ==============================================================================
echo.
echo   [1/1] 16 CPU Cekirdekli CMD Tarama Motoru Baslatiliyor...
echo.
python "%~dp0cmd_motor.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [HATA] Python motoru baslatilamadi!
    pause
)
