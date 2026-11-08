# SYNTHESIS-ZERO: Full-Stack Solar Lab & Hardware Telemetry Launcher
Write-Host "==========================================================" -ForegroundColor Amber
Write-Host " SYNTHESIS-ZERO: Quantum Solar Lab & Realme Narzo Bridge " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Amber

$LocalIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi*", "Ethernet*" | Where-Object { $_.IPAddress -notlike "169.254*" } | Select-Object -ExpandProperty IPAddress -First 1)
if (-not $LocalIP) { $LocalIP = "localhost" }

Write-Host "Local Machine IP detected: http://${LocalIP}:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Starting Python Quantum GNN Engine (Port 5000)..." -ForegroundColor White
$BackendProc = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "app.py" -PassThru

Write-Host "2. Starting Next.js Production Lab & Sensor Bridge (Port 3000)..." -ForegroundColor White
Set-Location frontend
$FrontendProc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c npm run start" -PassThru

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " SYSTEM READY!" -ForegroundColor Green
Write-Host " Desktop Lab URL:     http://localhost:3000/lab" -ForegroundColor Yellow
Write-Host " Realme Narzo QR URL: http://${LocalIP}:3000/node" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Press Ctrl+C to terminate services."

try {
    Wait-Process -Id $FrontendProc.Id
} finally {
    Stop-Process -Id $BackendProc.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $FrontendProc.Id -Force -ErrorAction SilentlyContinue
}

