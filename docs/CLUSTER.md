# Guida al Cluster GPU — KD Action Recognition

Guida per eseguire il training sul cluster GPU del DMI UniCT.

---

## 1. Hardware e QoS

| Nodo     | GPU     | VRAM       | CC  | bf16 | Note                 |
| -------- | ------- | ---------- | --- | ---- | -------------------- |
| gnode1–4 | 1× K80  | 22 GB      | 3.7 | ❌   | Solo fp32/fp16       |
| gnode5   | 4× V100 | 16 GB each | 7.0 | ❌   | Riservato dottorandi |
| gnode10  | 4× L40S | 48 GB each | 8.9 | ✅   | Tutto supportato     |

| QoS        | CPU | RAM   | GPU VRAM | Tempo max |
| ---------- | --- | ----- | -------- | --------- |
| gpu-small  | 1   | 4 GB  | 2.8 GB   | 4h        |
| gpu-medium | 2   | 8 GB  | 5.5 GB   | 6h        |
| gpu-large  | 4   | 16 GB | 11 GB    | 12h       |
| gpu-xlarge | 8   | 48 GB | 22 GB    | 12h       |

> **Per questo progetto** serve almeno `gpu-large` (11 GB VRAM).
> Consigliato `gpu-xlarge` per il teacher 3D ResNet-50 (fine-tune).

---

## 2. Accesso e Trasferimento

### SSH

```bash
ssh <codice-fiscale>@gcluster.dmi.unict.it
```

### Upload codice (da Windows)

```powershell
.\sync_cluster.ps1 -Action upload -User <CODICE-FISCALE>
```

### Download risultati

```powershell
.\sync_cluster.ps1 -Action download -User <CODICE-FISCALE>
.\sync_cluster.ps1 -Action download-logs -User <CODICE-FISCALE>
.\sync_cluster.ps1 -Action download-checkpoints -User <CODICE-FISCALE>
.\sync_cluster.ps1 -Action download-wandb -User <CODICE-FISCALE>
```

---

## 3. Setup Iniziale (una tantum)

```bash
# Sul cluster, dal login node:
cd ~/dl26-projects
bash cluster/setup.sh
```

Lo script:

- Rileva la GPU disponibile
- Installa le dipendenze Python (`pip install --user`)
- Scarica ed estrae il dataset UCF-101
- Verifica l'installazione

---

## 4. Lanciare il Training

### 4.1. Configura lo script batch

Modifica `cluster/train.sh` con i tuoi parametri:

```bash
nano cluster/train.sh
```

Cambia queste righe:

```bash
#SBATCH --account=dl-course-q1      # ← la tua queue
#SBATCH --partition=dl-course-q1    # ← idem
#SBATCH --qos=gpu-xlarge            # ← il tuo QoS
#SBATCH --mail-user=tua@email.com   # ← la tua email
```

### 4.2. Lancia i job

**Step 1 — Fine-tune Teacher:**

```bash
CONFIG=experiments/configs/teacher.yaml sbatch cluster/train.sh
```

**Step 2 — Baseline Student:**

```bash
CONFIG=experiments/configs/baseline.yaml sbatch cluster/train.sh
```

**Step 3 — Knowledge Distillation:**

```bash
CONFIG=experiments/configs/distillation.yaml sbatch cluster/train.sh
```

**Step 4 — KD + Attention Transfer:**

```bash
CONFIG=experiments/configs/attention_transfer.yaml sbatch cluster/train.sh
```

**Con override di parametri:**

```bash
CONFIG=experiments/configs/distillation.yaml EXTRA_ARGS="--override distillation.temperature=10" sbatch cluster/train.sh
```

**Job chaining (esegui distillation dopo che teacher è completato):**

```bash
TEACHER_JOB=$(CONFIG=experiments/configs/teacher.yaml sbatch --parsable cluster/train.sh)
CONFIG=experiments/configs/distillation.yaml sbatch --dependency=afterok:$TEACHER_JOB cluster/train.sh
```

**Catena multipla (teacher -> baseline -> distillation):**

```bash
CONFIGS="experiments/configs/teacher.yaml experiments/configs/baseline.yaml experiments/configs/distillation.yaml" \
sbatch cluster/train_sequential.sh
```

Questo e' il comando da usare per la sequenza completa quando vuoi eseguire tutti i training in ordine dentro un solo job SLURM.
Se un run fallisce, lo script interrompe la sequenza e marca il job come `FAILED`.

**Alias comodo da shell:**

```bash
source cluster/aliases.sh
train-seq teacher.yaml baseline.yaml distillation.yaml
```

**Se ricevi `QOSMaxSubmitJobPerUserLimit`:**

Alcuni account permettono solo 1 job totale (running + pending). In quel caso usa sempre un solo job SLURM che esegue piu training in sequenza:

```bash
CONFIGS="experiments/configs/teacher.yaml experiments/configs/baseline.yaml experiments/configs/distillation.yaml" \
sbatch cluster/train_sequential.sh
```

Questo evita submission multiple e rispetta il limite QoS.
Nota: il tempo totale resta soggetto al wall-time massimo del job (es. 12h).

> Nota: aggirare limiti di slot, QoS o wall-time del cluster (ad esempio eseguendo training GPU fuori scheduler) non è una pratica corretta. Usa sempre `sbatch`/`srun` e le policy ufficiali del corso/cluster.

### 4.3. Evaluation

```bash
CONFIG=experiments/configs/teacher.yaml \
CHECKPOINT=experiments/checkpoints/teacher_finetune_best.pth \
sbatch cluster/eval.sh
```

### 4.4. Alias utili

Se carichi gli alias con `source cluster/aliases.sh`, i comandi piu' utili per il training sono:

- `train teacher.yaml` per lanciare un singolo training
- `train-chain teacher.yaml baseline.yaml distillation.yaml` per una catena di job con dipendenze `afterok`
- `train-seq teacher.yaml baseline.yaml distillation.yaml` per un singolo job SLURM con i training eseguiti in sequenza

Per la sequenza completa, usa `train-seq` oppure direttamente `sbatch cluster/train_sequential.sh`.

---

## 5. Monitorare i Job

```bash
squeue -u $USER                        # Job attivi
scontrol show job <JOB_ID>             # Dettagli job
tail -f logs/slurm-train-<JOB_ID>.log  # Log in real-time
scancel <JOB_ID>                       # Cancella job
```

### Log strutturati per training

Ogni job di training crea automaticamente una cartella:

```bash
~/dl26-projects/experiments/logs/slurm-train-<JOB_ID>/
```

Contenuto utile:

- `metrics_epoch.csv`: metriche per epoca (train/test loss, train/test accuracy, top-5, lr, best_acc)
- `metrics_epoch.jsonl`: stesso contenuto in formato JSONL
- `training_summary.txt`: riepilogo finale (best accuracy, best epoch, final train/test performance)
- `job_meta.txt`: metadati SLURM e exit code
- `slurm-stdout.log`: copia del log standard SLURM
- `metric_*` e `status_*`: file marker rapidi per controllo via `ls -la`

Esempio controllo rapido:

```bash
ls -la ~/dl26-projects/experiments/logs/slurm-train-<JOB_ID>
```

---

## 6. W&B Offline

Il cluster non ha accesso a internet per gli studenti.
Gli script impostano `WANDB_MODE=offline` automaticamente.

Per sincronizzare dopo il training:

```powershell
# Da Windows:
.\sync_cluster.ps1 -Action download-wandb -User <CF>
wandb sync experiments\logs\wandb\offline-run-*
```

---

## 7. Troubleshooting

### CUDA out of memory

Riduci batch size nel config:

```bash
CONFIG=experiments/configs/teacher.yaml EXTRA_ARGS="--override training.batch_size=8" sbatch cluster/train.sh
```

### Job non parte

```bash
sacctmgr show associations user=$USER format=Account,Partition,QOS
```

Verifica che account/partition/qos nello script SLURM corrispondano.

### Pulizia workspace

```bash
bash cluster/clean.sh          # dry-run
bash cluster/clean.sh --force  # cancella davvero
```
