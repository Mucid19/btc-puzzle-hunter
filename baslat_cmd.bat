@echo off
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
title Bitcoin Puzzle CMD Motoru v3.0
color 0B
chcp 65001 >nul
cls

echo ==============================================================================
echo   ⚡ BITCOIN PUZZLE ARKA PLAN MOTORU (CMD v3.0)
echo   🚀 Cok Cekirdekli + GPU Hizlandirma
echo ==============================================================================
echo.
echo [*] CMD Motoru ve Web Arayuzu Baslatiliyor...
echo.

:: Dogrudan python ile calistirmayi dene
python "%~dp0cmd_motor.py"
if %ERRORLEVEL% EQU 0 goto :end

:: Python komutu calismadiysa py ile dene
py "%~dp0cmd_motor.py"
if %ERRORLEVEL% EQU 0 goto :end

:: Kutuphane eksikligi varsa websockets yukleyip tekrar dene
echo.
echo [*] websockets kutuphanesi kontrol ediliyor...
python -m pip install websockets >nul 2>&1
python "%~dp0cmd_motor.py"
if %ERRORLEVEL% EQU 0 goto :end

echo.
echo [HATA] Python motoru baslatilamadi!
echo Lutfen bu bilgisayarda Python kurulu oldugundan emin olun.
pause

:end
