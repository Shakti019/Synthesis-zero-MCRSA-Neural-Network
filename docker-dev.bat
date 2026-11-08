@echo off
title Synthesis-Zero Development Launcher
echo ==========================================================
echo  SYNTHESIS-ZERO: Single Container / Dev Launcher
echo  Python GNN (Port 5000) + Next.js (Port 3000)
echo ==========================================================
echo.

:: Check if Docker daemon is responsive
echo [Checking Docker Desktop status...]
docker ps >nul 2>&1
if %errorlevel% equ 0 (
    echo [Docker engine is active!]
    echo Starting single container with live hot-reload...
    docker-compose up --build
    goto end
)

:: Docker is not running: try starting Docker Desktop
echo [Docker Desktop is not running. Attempting to start Docker Desktop...]
if exist "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe" (
    start "" "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
    echo Waiting up to 15 seconds for Docker engine to initialize...
    for /l %%i in (1,1,15) do (
        docker ps >nul 2>&1
        if !errorlevel! equ 0 goto start_docker
        timeout /t 1 >nul
    )
)

:start_docker
docker ps >nul 2>&1
if %errorlevel% equ 0 (
    echo [Docker Desktop is ready!]
    echo Starting container with live hot-reload...
    docker-compose up --build
    goto end
)

:: If Docker still cannot be reached, automatically launch in Direct Local Dev Mode
echo.
echo ==========================================================
echo  Docker Desktop is not started or took too long to boot.
echo  AUTOMATICALLY LAUNCHING DIRECT LOCAL DEV MODE (NO DOCKER NEEDED)...
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

echo [1/2] Launching Python Quantum Engine (Port 5000)...
start "Synthesis-Zero Backend (Port 5000)" cmd /k "title Backend (Port 5000) && cd /d "%~dp0" && .\.venv\Scripts\python.exe app.py"

echo [2/2] Launching Next.js Production Lab (Port 3000)...
start "Synthesis-Zero Frontend (Port 3000)" cmd /k "title Frontend (Port 3000) && cd /d "%~dp0frontend" && npm run dev"

echo.
echo ==========================================================
echo  APPLICATIONS ARE RUNNING!
echo  Desktop Lab:     http://localhost:3000/lab
echo  Realme Narzo:    http://%LOCAL_IP%:3000/node
echo  Python GNN API:  http://localhost:5000
echo ==========================================================
echo.
echo Opening browser in 3 seconds...
timeout /t 3 >nul
start http://localhost:3000/lab

:end
pause

