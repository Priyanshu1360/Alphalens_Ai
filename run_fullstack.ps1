$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python not found in venv at: $python"
}

Write-Host "Starting FastAPI Backend on Port 8000..." -ForegroundColor Green
Start-Process -FilePath $python -ArgumentList "-m uvicorn src.api.routes:app --port 8000" -WindowStyle Normal

Write-Host "Starting Next.js Frontend on Port 3000..." -ForegroundColor Cyan
Set-Location "$PSScriptRoot\frontend"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k npm run dev" -WindowStyle Normal

Write-Host "Both servers started in separate windows." -ForegroundColor Yellow
Write-Host "API: http://localhost:8000"
Write-Host "UI: http://localhost:3000"

Write-Host "Waiting for servers to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 3
Write-Host "Opening UI in your default browser..." -ForegroundColor Green
Start-Process "http://localhost:3000"
