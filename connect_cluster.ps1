Write-Host "Connessione a clusterone in corso..." -ForegroundColor Cyan
ssh -t clusterone "cd dl26-projects/ && bash --rcfile cluster/aliases.sh"
