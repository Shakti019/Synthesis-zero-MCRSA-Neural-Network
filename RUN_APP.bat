@echo off
title Synthesis-Zero Full-Stack Launcher
echo ==========================================================
echo  SYNTHESIS-ZERO: Full-Stack Quantum Solar Lab Launcher
echo ==========================================================
echo.

:: Detect Local IP Address for Realme Narzo Phone Pairing
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /r "IPv4.*192\.168\."') do (
    for /f "tokens=1" %%b in ("%%a") do set LOCAL_IP=%%b
)
if "%LOCAL_IP%"=="" (
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /r "IPv4"') do (
        for /f "tokens=1" %%b in ("%%a") do set LOCAL_IP=%%b
    )
)
if "%LOCAL_IP%"=="" set LOCAL_IP=localhost

echo [Detected LAN IP: %LOCAL_IP%]
echo.

:: 1. Launch Python Quantum Backend on port 5000 in background
echo [1/2] Starting Python Quantum Engine (Port 5000)...
start "Synthesis-Zero Backend (Port 5000)" cmd /k "title Backend (Port 5000) && cd /d "%~dp0" && .\.venv\Scripts\python.exe app.py"

:: 2. Launch Next.js Production Lab on port 3000 in background with HTTPS
echo [2/2] Starting Next.js Frontend with HTTPS (Port 3000)...
start "Synthesis-Zero Frontend (Port 3000)" cmd /k "title Frontend (Port 3000) && cd /d "%~dp0frontend" && npm run dev"

echo.
echo ==========================================================
echo  BOTH APPLICATIONS ARE RUNNING WITH HTTPS!
echo  Desktop Lab:     https://localhost:3000/lab
echo  Mobile Probe:    https://%LOCAL_IP%:3000/node
echo  Python GNN API:  http://localhost:5000
echo ==========================================================
echo.
echo Opening browser in 4 seconds...
timeout /t 4 >nul
start https://localhost:3000/lab

