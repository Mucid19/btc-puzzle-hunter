@echo off
chcp 65001 >nul
title SiriuS Secp256k1 Puzzle Hunter
echo =======================================================
echo    SiriuS Secp256k1 Puzzle Hunter
echo =======================================================
echo.
echo [1/2] Yerel HTTP Sunucusu Baslatiliyor (Port: 8000)...
start "" http://localhost:8000/
echo [2/2] Tarayici Acildi: http://localhost:8000/
echo.
echo Kapatmak icin bu pencereyi kapatabilirsiniz.
echo -------------------------------------------------------
python -m http.server 8000
pause
