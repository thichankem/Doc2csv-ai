# Doc2CSV-AI launcher (PowerShell) - uses Anaconda Python (where deps are installed)
$ErrorActionPreference = "Stop"

$py = Join-Path $env:USERPROFILE "anaconda3\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[LOI] Khong tim thay Anaconda Python tai: $py" -ForegroundColor Red
    Write-Host "Vui long sua bien `$py trong run.ps1 tro toi python.exe co cai deps." -ForegroundColor Yellow
    Read-Host "Nhan Enter de thoat"
    exit 1
}

Set-Location $PSScriptRoot
& $py app.py
