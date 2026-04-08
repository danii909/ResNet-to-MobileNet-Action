#!/bin/bash
# ============================================================================
# Setup one-tantum per il cluster DMI -- KD Action Recognition
#
# Uso (dal login node):
#   cd ~/dl26-projects
#   bash cluster/setup.sh
#
# Lo script rilancia se stesso dentro srun + Apptainer automaticamente.
# ============================================================================

# -- 0. Auto-rilancio dentro srun + Apptainer se siamo sul login node ----------
if [ -z "$APPTAINER_CONTAINER" ]; then
    echo "Login node rilevato -> rilancio inside srun + Apptainer..."
    ACCOUNT="${SLURM_ACCOUNT:-dl-course-q2}"
    exec srun --account "$ACCOUNT" --partition "$ACCOUNT" --qos gpu-xlarge \
         --gres=gpu:1 --gres=shard:22000 --mem=48G --cpus-per-task=8 \
         apptainer run --nv /shared/sifs/latest.sif \
         bash "$0" "$@"
fi

set -e

echo "=== Setup KD Action Recognition (Cluster DMI) ==="
echo ""

# -- 1. Verifica GPU -----------------------------------------------------------
echo "Rilevamento GPU..."

# Trova il comando python disponibile
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERRORE: Python non trovato nel container!"
    exit 1
fi
echo "   Python: $($PY --version 2>&1)"

GPU_INFO=$($PY -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability()
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {name} (CC {cc[0]}.{cc[1]}, {vram:.1f} GB)')
    print(f'CC_MAJOR={cc[0]}')
else:
    print('  GPU: NESSUNA GPU rilevata')
    print('CC_MAJOR=0')
") || { echo "ERRORE: Errore nel rilevamento GPU"; exit 1; }

echo "$GPU_INFO" | grep -v CC_MAJOR
CC_MAJOR=$(echo "$GPU_INFO" | grep CC_MAJOR | cut -d= -f2)

# -- 2. Installa dipendenze ----------------------------------------------------
echo ""
echo "Installazione dipendenze..."
pip install --user torch torchvision torchaudio 2>/dev/null || true
pip install --user pytorchvideo wandb decord av seaborn datasets \
    numpy pandas matplotlib scikit-learn pyyaml tensorboard tqdm

echo "   Dipendenze installate."

# -- 3. Scarica dataset UCF-101 da Hugging Face --------------------------------
cd "$HOME/dl26-projects"

export HF_TOKEN="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"

# Controlliamo se il dataset e' gia' in cache HF
HF_CACHE_CHECK=$($PY -c "
from datasets import load_dataset
try:
    ds = load_dataset('flwrlabs/ucf101', split='train', download_mode='reuse_dataset_if_exists')
    print('CACHED')
except Exception:
    print('MISSING')
" 2>/dev/null || echo "MISSING")

if [ "$HF_CACHE_CHECK" = "CACHED" ]; then
    echo ""
    echo "Dataset UCF-101 gia' presente nella cache HF."
    $PY -c "
from datasets import load_dataset
ds = load_dataset('flwrlabs/ucf101', split='train')
print(f'  Train: {len(ds)} frame')
ds = load_dataset('flwrlabs/ucf101', split='test')
print(f'  Test: {len(ds)} frame')
"
else
    echo ""
    echo "Download dataset UCF-101 da Hugging Face (cache HF)..."
    echo "   Questo potrebbe richiedere tempo (~2.5M frame)..."

    $PY -c "
from datasets import load_dataset
print('Scaricamento train split...')
ds_train = load_dataset('flwrlabs/ucf101', split='train')
print(f'  Train: {len(ds_train)} frame')
print('Scaricamento test split...')
ds_test = load_dataset('flwrlabs/ucf101', split='test')
print(f'  Test: {len(ds_test)} frame')
print('Dataset scaricato e in cache.')
"
    echo "   Dataset UCF-101 pronto nella cache HF."

    # Rimuovi i raw download per risparmiare ~54GB di spazio
    echo "   Pulizia raw downloads HF per risparmiare spazio..."
    rm -rf "$HOME/.cache/huggingface/hub/datasets--flwrlabs--ucf101"
    echo "   Cache raw rimossa."
fi

# -- 3b. Verifica pesi pretrained slow_r50 --------------------------------------
WEIGHTS_FILE="$HOME/dl26-projects/experiments/checkpoints/SLOW_8x8_R50.pyth"
if [ -f "$WEIGHTS_FILE" ]; then
    echo ""
    echo "Pesi slow_r50 presenti in $WEIGHTS_FILE"
else
    echo ""
    echo "ATTENZIONE: Pesi slow_r50 non trovati in $WEIGHTS_FILE"
    echo "   Caricali dal PC locale con: .\\sync_cluster.ps1 -Action upload"
fi

# -- 4. Verifica installazione -------------------------------------------------
echo ""
echo "Verifica installazione..."
$PY -c "
import torch, torchvision
print(f'  PyTorch:      {torch.__version__}')
print(f'  TorchVision:  {torchvision.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
try:
    import pytorchvideo
    print(f'  PyTorchVideo: OK')
except ImportError:
    print(f'  PyTorchVideo: NON installato')
try:
    import wandb
    print(f'  WandB:        {wandb.__version__}')
except ImportError:
    print(f'  WandB:        NON installato')
try:
    import decord
    print(f'  decord:       OK')
except ImportError:
    print(f'  decord:       NON installato (fallback: torchvision)')
try:
    import datasets
    print(f'  datasets:     {datasets.__version__}')
except ImportError:
    print(f'  datasets:     NON installato')
"

# -- 5. Verifica struttura progetto ---------------------------------------------
echo ""
echo "Verifica struttura progetto..."
MISSING=0
for f in src/training/train.py src/evaluation/evaluate.py src/models/teacher.py src/models/student.py; do
    if [ ! -f "$f" ]; then
        echo "   MANCANTE: $f"
        MISSING=1
    fi
done
for f in experiments/configs/teacher.yaml experiments/configs/baseline.yaml experiments/configs/distillation.yaml; do
    if [ ! -f "$f" ]; then
        echo "   MANCANTE: $f"
        MISSING=1
    fi
done
if [ "$MISSING" = "0" ]; then
    echo "   OK: Tutti i file necessari sono presenti."
fi

echo ""
echo "=== Setup completato! ==="
echo ""
echo "Prossimi passi:"
echo "  1. Modifica cluster/train.sh con la tua queue, email e QoS"
echo "  2. Lancia: CONFIG=experiments/configs/teacher.yaml sbatch cluster/train.sh"
