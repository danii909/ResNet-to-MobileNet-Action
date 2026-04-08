#!/usr/bin/env pwsh
# Download pesi pretrained per il progetto KD Action Recognition
# Uso: .\download_weights.ps1

$weightsDir = "experiments\checkpoints"
New-Item -ItemType Directory -Force -Path $weightsDir | Out-Null

$weights = @(
    @{
        Name = "SLOW_8x8_R50.pyth"
        Url  = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth"
        Desc = "SlowOnly R50 (Kinetics-400) - Teacher"
    }
)

foreach ($w in $weights) {
    $dest = Join-Path $weightsDir $w.Name
    if (Test-Path $dest) {
        $size = [math]::Round((Get-Item $dest).Length / 1MB, 1)
        Write-Host "[OK] $($w.Desc) gia' presente ($size MB)" -ForegroundColor Green
    } else {
        Write-Host "[DOWNLOAD] $($w.Desc)..." -ForegroundColor Cyan
        Write-Host "  $($w.Url)"
        Invoke-WebRequest -Uri $w.Url -OutFile $dest
        $size = [math]::Round((Get-Item $dest).Length / 1MB, 1)
        Write-Host "[OK] Scaricato: $dest ($size MB)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Pesi pronti in $weightsDir/" -ForegroundColor Green
Write-Host "Ora fai: .\sync_cluster.ps1 -Action upload"
