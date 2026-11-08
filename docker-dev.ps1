# SYNTHESIS-ZERO: Single Container Docker Development Runner
Write-Host "==========================================================" -ForegroundColor Yellow
Write-Host " SYNTHESIS-ZERO: Single Docker Container Dev Environment  " -ForegroundColor Green
Write-Host " Python GNN Backend (Port 5000) + Next.js (Port 3000)     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Yellow

# Check if Docker daemon is responsive
$dockerRunning = $false
try {
    $proc = Start-Process -FilePath "docker" -ArgumentList "ps" -NoNewWindow -PassThru -Wait -ErrorAction SilentlyContinue
    if ($proc.ExitCode -eq 0) { $dockerRunning = $true }
} catch {}

if (-not $dockerRunning) {
    Write-Host "Starting Docker Desktop..." -ForegroundColor Yellow
    $dockerPath = "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"
    if (Test-Path $dockerPath) {
        Start-Process -FilePath $dockerPath
        Write-Host "Waiting for Docker daemon to become ready (up to 30s)..." -ForegroundColor Gray
        for ($i = 0; $i -lt 15; $i++) {
            Start-Sleep -Seconds 2
            try {
                $check = Start-Process -FilePath "docker" -ArgumentList "ps" -NoNewWindow -PassThru -Wait -ErrorAction SilentlyContinue
                if ($check.ExitCode -eq 0) {
                    $dockerRunning = $true
                    break
                }
            } catch {}
        }
    }
}

Write-Host "Building and launching single container in development mode..." -ForegroundColor Green
Write-Host "Live volume sync active: any code edits on host will hot-reload inside the container!" -ForegroundColor Gray
Write-Host ""

docker-compose up --build

