<#
.SYNOPSIS
    Sincronizza il progetto tra la macchina locale e il cluster DMI.

.DESCRIPTION
    Upload del codice sorgente sul cluster, download di risultati (logs, checkpoints, wandb runs).

.PARAMETER Action
    upload              — Carica codice sorgente e config sul cluster
    download            — Scarica tutto (logs + checkpoints + wandb)
    download-logs       — Scarica solo logs e figure
    download-checkpoints — Scarica solo checkpoint
    download-wandb      — Scarica solo wandb offline runs

.PARAMETER User
    Username per SSH (codice fiscale). Default: legge da $env:CLUSTER_USER.

.PARAMETER Host
    Hostname del cluster. Default: gcluster.dmi.unict.it

.PARAMETER RemoteDir
    Directory remota del progetto. Default: ~/dl26-projects

.EXAMPLE
    .\sync_cluster.ps1 -Action upload
    .\sync_cluster.ps1 -Action download -User ABCDEF12G34H567I
    .\sync_cluster.ps1 -Action download-wandb
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("upload", "download", "download-logs", "download-checkpoints", "download-wandb")]
    [string]$Action,

    [string]$User = "brbdnl01e03e017o",
    [string]$ClusterHost = "gcluster.dmi.unict.it",
    [string]$RemoteDir = "~/dl26-projects"
)

if (-not $User) {
    Write-Error "Specifica -User <codice-fiscale> o imposta `$env:CLUSTER_USER"
    exit 1
}

$Remote = "${User}@${ClusterHost}"
$RemotePath = "${Remote}:${RemoteDir}"

# Cartelle/file da escludere nell'upload
$ExcludeUpload = @(
    ".venv", "__pycache__", ".git/objects", "data",
    "experiments/logs", "experiments/checkpoints",
    "wandb", "*.pyc", ".env"
)

function Upload {
    Write-Host "[UPLOAD] Upload codice su ${RemotePath}..." -ForegroundColor Cyan

    # Crea struttura remota
    ssh $Remote "mkdir -p $RemoteDir/experiments/configs $RemoteDir/experiments/logs $RemoteDir/experiments/checkpoints $RemoteDir/logs $RemoteDir/data $RemoteDir/figures $RemoteDir/cluster"

    # Usa scp per i file essenziali (Windows non ha rsync nativo)
    # Nota: per experiments/configs copiamo il CONTENUTO nella cartella remota corretta.
    Write-Host "  Copiando src/..."
    scp -r "src" "${RemotePath}/"

    Write-Host "  Copiando cluster/..."
    scp -r "cluster" "${RemotePath}/"

    Write-Host "  Copiando experiments/configs/..."
    scp -r "experiments/configs/." "${RemotePath}/experiments/configs/"

    # File singoli nella root
    $rootFiles = @("environment.yml", "README.md", "CONTRIBUTING.md", "INSTRUCTIONS.md", "LICENSE")
    foreach ($f in $rootFiles) {
        if (Test-Path $f) {
            scp "$f" "${RemotePath}/"
        }
    }

    # Pesi pretrained (solo se non gia' presenti sul cluster)
    $weightsDir = "experiments/checkpoints"
    $weightsFiles = Get-ChildItem -Path $weightsDir -Filter "*.pyth" -ErrorAction SilentlyContinue
    foreach ($w in $weightsFiles) {
        $remotePath_w = "${RemoteDir}/experiments/checkpoints/$($w.Name)"
        $exists = ssh $Remote "test -f $remotePath_w && echo yes || echo no"
        if ($exists -eq "no") {
            $sizeMB = [math]::Round($w.Length / 1MB, 1)
            Write-Host "  Copiando pesi $($w.Name) ($sizeMB MB)..."
            scp "$($w.FullName)" "${RemotePath}/experiments/checkpoints/"
        } else {
            Write-Host "  Pesi $($w.Name) gia' presenti sul cluster, skip."
        }
    }

    Write-Host "[OK] Upload completato." -ForegroundColor Green
}

function Download {
    param([string]$What = "all")

    switch ($What) {
        "all" {
            Write-Host "[DOWNLOAD] Download completo da ${RemotePath}..." -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path "experiments/logs" | Out-Null
            New-Item -ItemType Directory -Force -Path "experiments/checkpoints" | Out-Null
            New-Item -ItemType Directory -Force -Path "figures" | Out-Null
            scp -r "${RemotePath}/experiments/logs/*" "experiments/logs/" 2>$null
            scp -r "${RemotePath}/experiments/checkpoints/*" "experiments/checkpoints/" 2>$null
            scp -r "${RemotePath}/figures/*" "figures/" 2>$null
            scp -r "${RemotePath}/logs/*" "logs/" 2>$null
        }
        "logs" {
            Write-Host "[DOWNLOAD] Download logs e figure..." -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path "experiments/logs" | Out-Null
            New-Item -ItemType Directory -Force -Path "figures" | Out-Null
            New-Item -ItemType Directory -Force -Path "logs" | Out-Null
            scp -r "${RemotePath}/experiments/logs/*" "experiments/logs/" 2>$null
            scp -r "${RemotePath}/figures/*" "figures/" 2>$null
            scp -r "${RemotePath}/logs/*" "logs/" 2>$null
        }
        "checkpoints" {
            Write-Host "[DOWNLOAD] Download checkpoints..." -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path "experiments/checkpoints" | Out-Null
            scp -r "${RemotePath}/experiments/checkpoints/*" "experiments/checkpoints/"
        }
        "wandb" {
            Write-Host "[DOWNLOAD] Download wandb offline runs..." -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path "experiments/logs" | Out-Null
            scp -r "${RemotePath}/experiments/logs/*/wandb" "experiments/logs/" 2>$null
            Write-Host ""
            Write-Host "Per sincronizzare con W`&B cloud:" -ForegroundColor Yellow
            Write-Host "  wandb sync experiments\logs\wandb\offline-run-*"
        }
    }

    Write-Host "[OK] Download completato." -ForegroundColor Green
}

switch ($Action) {
    "upload"               { Upload }
    "download"             { Download -What "all" }
    "download-logs"        { Download -What "logs" }
    "download-checkpoints" { Download -What "checkpoints" }
    "download-wandb"       { Download -What "wandb" }
}
