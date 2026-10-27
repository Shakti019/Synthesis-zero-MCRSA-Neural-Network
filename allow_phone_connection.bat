@echo off
title SYNTHESIS-ZERO · Network & Firewall Setup
echo =================================================================
echo   SYNTHESIS-ZERO · Hotspot & Firewall Unblocker
echo =================================================================

net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Elevating privileges to configure Windows Firewall...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"%~dp0allow_phone_connection.bat\"' -Verb RunAs"
    exit /b
)

echo.
echo [1/2] Setting Hotspot Network 'realme narzo 60 5G 466' to Private...
powershell -Command "Set-NetConnectionProfile -Name 'realme narzo 60 5G 466' -NetworkCategory Private -ErrorAction SilentlyContinue"

echo [2/2] Opening Inbound Port 5000 in Windows Defender Firewall...
netsh advfirewall firewall delete rule name="SynthesisZeroPort5000" >nul 2>&1
netsh advfirewall firewall add rule name="SynthesisZeroPort5000" dir=in action=allow protocol=TCP localport=5000 profile=any >nul 2>&1

echo.
echo =================================================================
echo   SUCCESS! Port 5000 is now unlocked for all incoming connections.
echo.
echo   Your phone can now directly open:
echo   http://10.133.123.81:5000/camera
echo =================================================================
echo.
pause
