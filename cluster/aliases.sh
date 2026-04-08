#!/bin/bash
# ============================================================================
# Alias utili per il cluster DMI -- KD Action Recognition
#
# Uso:
#   source cluster/aliases.sh
#
# Per caricarli automaticamente, aggiungi al tuo ~/.bashrc:
#   source ~/dl26-projects/cluster/aliases.sh
# ============================================================================

PROJ_DIR="$HOME/dl26-projects"

# -- Job management -----------------------------------------------------------

# Controlla i miei job attivi
alias myjobs='squeue --me --format="%.10i %.20j %.8T %.10M %.6D %.20R %o"'

# Info dettagliata su un job (uso: jobinfo <JOB_ID>)
jobinfo() {
    if [ -z "$1" ]; then
        echo "Uso: jobinfo <JOB_ID>"
        return 1
    fi
    scontrol show job "$1"
}

# Cancella un job (uso: killjob <JOB_ID>)
alias killjob='scancel'

# Cancella tutti i miei job
alias killalljobs='scancel --me'

# -- Log monitoring -----------------------------------------------------------

# Segui il log di un job di training (uso: trainlog <JOB_ID>)
trainlog() {
    if [ -z "$1" ]; then
        echo "Uso: trainlog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-train-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Mostra l'ultimo log -- uso: lastlog [N_RIGHE]
# Senza argomento: tail -f (segui). Con argomento: mostra ultime N righe.
lastlog() {
    local logfile
    logfile=$(ls -t "$PROJ_DIR"/logs/slurm*.log 2>/dev/null | head -1)
    if [ -z "$logfile" ]; then
        echo "Nessun log trovato in $PROJ_DIR/logs/"
        return 1
    fi
    echo "==> $logfile <=="
    if [ -n "$1" ]; then
        tail -n "$1" "$logfile"
    else
        tail -f "$logfile"
    fi
}

# Mostra le ultime metriche dal log (uso: metrics [JOB_ID])
metrics() {
    local logfile
    if [ -n "$1" ]; then
        logfile="$PROJ_DIR/logs/slurm-train-${1}.log"
    else
        logfile=$(ls -t "$PROJ_DIR"/logs/slurm*.log 2>/dev/null | head -1)
    fi
    if [ -z "$logfile" ] || [ ! -f "$logfile" ]; then
        echo "Log non trovato"
        return 1
    fi
    echo "==> $logfile <=="
    grep -E "^Epoch " "$logfile" | tail -10
}

# -- Filesystem ---------------------------------------------------------------

# Tree ricorsivo di una cartella (uso: tree <DIR> [DEPTH])
tree() {
    local dir="${1:-.}"
    local depth="${2:-3}"
    find "$dir" -maxdepth "$depth" | sed -e "s|[^/]*/|  |g" -e "s|  |+-|"
}

# -- GPU & risorse ------------------------------------------------------------

# Stato GPU
alias gpu='nvidia-smi'

# Uso disco del progetto
alias quota='quota -s'

# Uso disco dettagliato
diskusage() {
    echo "=== Uso disco ==="
    du -sh "$HOME"/.cache/huggingface/ 2>/dev/null | sed 's/^/  HF cache:  /'
    du -sh "$PROJ_DIR"/experiments/checkpoints/ 2>/dev/null | sed 's/^/  Checkpoints: /'
    du -sh "$PROJ_DIR"/wandb/ 2>/dev/null | sed 's/^/  W&B logs:  /'
    du -sh "$PROJ_DIR"/logs/ 2>/dev/null | sed 's/^/  SLURM logs: /'
    echo "---"
    du -sh "$HOME" 2>/dev/null | sed 's/^/  TOTALE:    /'
}

# -- Quick commands -----------------------------------------------------------

# Vai alla directory del progetto
alias proj='cd "$PROJ_DIR"'

# Mostra i checkpoint disponibili
ckpts() {
    local ckdir="$PROJ_DIR/experiments/checkpoints"
    echo "=== Checkpoints ==="
    for f in "$ckdir"/*.pyth "$ckdir"/*.pth "$ckdir"/*.pt; do
        [ -f "$f" ] || continue
        local size=$(du -sh "$f" 2>/dev/null | cut -f1)
        echo "  $(basename "$f")  ($size)"
    done
    echo ""
    echo "=== Best models ==="
    for f in "$ckdir"/*best*.pth "$ckdir"/*best*.pt; do
        [ -f "$f" ] || continue
        local size=$(du -sh "$f" 2>/dev/null | cut -f1)
        echo "  $(basename "$f")  ($size)"
    done
}

# Mostra i config disponibili
configs() {
    echo "Config disponibili:"
    ls -1 "$PROJ_DIR"/experiments/configs/*.yaml 2>/dev/null | while read -r f; do
        echo "  $(basename "$f")"
    done
}

# -- Training -----------------------------------------------------------------

# Lancia training (uso: train CONFIG [extra args...])
# Esempio: train teacher.yaml
#          train distillation.yaml --override training.lr=0.005
train() {
    local config="$1"
    shift
    if [ -z "$config" ]; then
        echo "Uso: train CONFIG [extra args...]"
        echo ""
        configs
        return 1
    fi
    # Se e' solo un nome file, anteponi il path
    if [[ "$config" != */* ]]; then
        config="experiments/configs/$config"
    fi
    if [ ! -f "$PROJ_DIR/$config" ]; then
        echo "Config non trovato: $config"
        echo ""
        configs
        return 1
    fi
    cd "$PROJ_DIR" && CONFIG="$config" EXTRA_ARGS="$*" sbatch cluster/train.sh
}

# Setup ambiente (uso: setup)
alias setup='cd "$PROJ_DIR" && bash cluster/setup.sh'

# -- Monitor ------------------------------------------------------------------

# Monitor live del training (uso: monitor [--poll N])
monitor() {
    cd "$PROJ_DIR" && python3 -u -m src.utils.monitor "$@"
}

# -- Pip / Environment --------------------------------------------------------

# Pulisci tutti i pacchetti --user
pip-clean() {
    echo "Rimozione pacchetti pip --user..."
    rm -rf ~/.local/lib/python3.*/site-packages/*
    rm -rf ~/.local/bin/*
    echo "~/.local ripulito"
}

# (Re)installa dipendenze da setup.sh
pip-setup() {
    echo "Installazione dipendenze..."
    cd "$PROJ_DIR" && bash cluster/setup.sh
}

# Pulisci e reinstalla da zero
pip-reset() {
    pip-clean
    pip-setup
}

# -- Pulizia ------------------------------------------------------------------

# Pulizia workspace (uso: clean [--force])
clean() {
    local force=0
    [ "$1" = "--force" ] && force=1

    echo "=== Pulizia workspace ==="
    echo ""

    # W&B offline runs
    local wandb_size=$(du -sh "$PROJ_DIR/wandb" 2>/dev/null | cut -f1)
    local wandb_count=$(ls -d "$PROJ_DIR/wandb"/offline-run-* 2>/dev/null | wc -l)
    echo "  W&B offline runs: $wandb_count ($wandb_size)"

    # SLURM logs
    local log_count=$(ls "$PROJ_DIR/logs"/slurm*.log 2>/dev/null | wc -l)
    local log_size=$(du -sh "$PROJ_DIR/logs" 2>/dev/null | cut -f1)
    echo "  SLURM logs: $log_count ($log_size)"

    # HF cache
    local hf_size=$(du -sh "$HOME/.cache/huggingface" 2>/dev/null | cut -f1)
    echo "  HF cache: $hf_size"

    echo ""
    if [ "$force" -eq 1 ]; then
        rm -rf "$PROJ_DIR/wandb"/offline-run-*
        echo "  W&B offline runs cancellati"
        # Keep last 5 logs
        ls -t "$PROJ_DIR/logs"/slurm*.log 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
        echo "  Log vecchi cancellati (ultimi 5 mantenuti)"
    else
        echo "  Dry run. Usa: clean --force per cancellare"
    fi
}

# -- Meta ---------------------------------------------------------------------

_KD_ALIASES="myjobs jobinfo killjob killalljobs trainlog lastlog metrics tree gpu quota diskusage proj ckpts configs train setup monitor pip-clean pip-setup pip-reset clean daniele unload-aliases install-aliases uninstall-aliases"

# Submit multiple training configs as a dependency chain (one active job at a time)
train-chain() {
    if [ "$#" -lt 1 ]; then
        echo "Uso: train-chain CONFIG1 [CONFIG2 ...]"
        echo "Esempio:"
        echo "  train-chain teacher.yaml baseline.yaml distillation.yaml"
        return 1
    fi

    local args=()
    local cfg
    for cfg in "$@"; do
        if [[ "$cfg" != */* ]]; then
            cfg="experiments/configs/$cfg"
        fi
        if [ ! -f "$PROJ_DIR/$cfg" ]; then
            echo "Config non trovato: $cfg"
            return 1
        fi
        args+=("$cfg")
    done

    cd "$PROJ_DIR" && bash cluster/submit_chain.sh "${args[@]}"
}

# Submit one SLURM job that runs multiple trainings sequentially.
# Useful when QOS blocks multiple submitted jobs (QOSMaxSubmitJobPerUserLimit).
train-seq() {
    if [ "$#" -lt 1 ]; then
        echo "Uso: train-seq CONFIG1 [CONFIG2 ...]"
        echo "Esempio:"
        echo "  train-seq teacher.yaml baseline.yaml distillation.yaml"
        return 1
    fi

    local args=()
    local cfg
    for cfg in "$@"; do
        if [[ "$cfg" != */* ]]; then
            cfg="experiments/configs/$cfg"
        fi
        if [ ! -f "$PROJ_DIR/$cfg" ]; then
            echo "Config non trovato: $cfg"
            return 1
        fi
        args+=("$cfg")
    done

    local cfgs_string="${args[*]}"
    cd "$PROJ_DIR" && CONFIGS="$cfgs_string" sbatch cluster/train_sequential.sh
}

# Mostra i comandi disponibili
sas() {
    echo "Comandi KD disponibili:"
    echo ""
    echo "-- Job management --"
    echo "   myjobs            -- lista job attivi"
    echo "   jobinfo <ID>      -- dettagli job"
    echo "   killjob <ID>      -- cancella job"
    echo "   killalljobs       -- cancella tutti i miei job"
    echo ""
    echo "-- Log monitoring --"
    echo "   trainlog <ID>     -- segui log training"
    echo "   lastlog [N]       -- segui l'ultimo log (N=ultime N righe)"
    echo "   metrics [ID]      -- mostra ultime metriche (Epoch ...)"
    echo ""
    echo "-- Training --"
    echo "   train CONFIG [extra args...]"
    echo "                     -- lancia training"
    echo "   train-chain C1 [C2 ...]"
    echo "                     -- lancia una catena di training (afterok)"
    echo "   train-seq C1 [C2 ...]"
    echo "                     -- un solo job SLURM con training sequenziali (usa train_sequential.sh)"
    echo "   setup             -- (ri)lancia cluster/setup.sh"
    echo "   configs           -- mostra config disponibili"
    echo "   monitor [--poll N] -- monitor live del training"
    echo ""
    echo "-- Utilita' --"
    echo "   proj              -- cd al progetto"
    echo "   tree <DIR> [N]    -- albero cartelle (profondita' N)"
    echo "   gpu               -- stato GPU"
    echo "   quota             -- uso disco"
    echo "   diskusage         -- uso disco dettagliato"
    echo "   ckpts             -- mostra checkpoint"
    echo ""
    echo "-- Pulizia --"
    echo "   clean [--force]   -- pulizia workspace"
    echo "   pip-clean         -- rimuovi pacchetti pip --user"
    echo "   pip-setup         -- (re)installa dipendenze"
    echo "   pip-reset         -- pip-clean + pip-setup"
    echo ""
    echo "-- Meta --"
    echo "   sas               -- mostra questo messaggio"
    echo "   install-aliases   -- aggiungi alias al .bashrc"
    echo "   uninstall-aliases -- rimuovi alias dal .bashrc"
}

# Rimuovi tutti gli alias e funzioni custom (solo sessione corrente)
unload-aliases() {
    for cmd in $_KD_ALIASES; do
        unalias "$cmd" 2>/dev/null
        unset -f "$cmd" 2>/dev/null
    done
    unset _KD_ALIASES PROJ_DIR
    echo "Alias KD rimossi (sessione corrente)."
}

_ALIASES_SOURCE_LINE="source ~/dl26-projects/cluster/aliases.sh"

# Aggiungi alias al .bashrc (caricati ad ogni login)
install-aliases() {
    if grep -qF "$_ALIASES_SOURCE_LINE" ~/.bashrc 2>/dev/null; then
        echo "Alias gia' presenti in ~/.bashrc"
    else
        echo "$_ALIASES_SOURCE_LINE" >> ~/.bashrc
        echo "Alias aggiunti a ~/.bashrc (attivi dal prossimo login)"
    fi
}

# Rimuovi alias dal .bashrc (non piu' caricati al login)
uninstall-aliases() {
    if grep -qF "$_ALIASES_SOURCE_LINE" ~/.bashrc 2>/dev/null; then
        sed -i "\|$_ALIASES_SOURCE_LINE|d" ~/.bashrc
        echo "Alias rimossi da ~/.bashrc"
    else
        echo "Alias non presenti in ~/.bashrc"
    fi
    unload-aliases
}

echo "Alias KD caricati. Digita 'sas' per la lista comandi."
