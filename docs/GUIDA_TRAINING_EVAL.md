# Guida Completta all'Uso del Cluster (Training & Evaluation)
Questo documento serve come guida di riferimento rapido per l'avvio di job di addestramento (Training) e valutazione (Evaluation/Inference) sul cluster DMI per il progetto **KD Action Recognition**.

---

## 📌 Struttura delle Cartelle
* **`experiments/configs/`**: Contiene tutti i file di configurazione (`.yaml`) organizzati per tipologia (es. `experiments/configs/experiments/` o `experiments/configs/valid/`).
* **`experiments/checkpoints/`**: Cartella dove vengono salvati e letti i pesi dei modelli (`.pth`).
* **`cluster/`**: Contiene tutti gli script SLURM (`.sh`) e i file di utilità/alias per interagire con il cluster.
* **`logs/`**: Contiene i file di log di output generati da SLURM (es. `slurm-train-%j.log` e `slurm-eval-%j.log`).

---

## ⚡ Caricare gli Alias (Consigliatissimo!)
Il progetto fornisce un set di alias e funzioni bash utilissime che semplificano enormemente i comandi.
Per caricarli nella sessione corrente del cluster:
```bash
source cluster/aliases.sh
```

> [!TIP]
> Puoi installarli permanentemente nel tuo profilo (in modo che vengano caricati automaticamente ad ogni login) digitando:
> ```bash
> install-aliases
> ```
> Digitando il comando **`sas`** vedrai la lista completa dei comandi custom disponibili!

---

## 📊 Guida alla Valutazione (Evaluation)
La valutazione esegue l'inferenza del modello sul **set di test** e calcola le metriche (Top-1 Accuracy, Top-5 Accuracy, velocità di inferenza in ms, dimensione modello).

### 1. Metodo Standard con `sbatch` (Senza alias)
Si usa lo script `cluster/eval.sh` passando due variabili d'ambiente:
* `CONFIG`: Il percorso del file di configurazione del modello.
* `CHECKPOINT`: Il percorso del file dei pesi `.pth` da valutare.

```bash
CONFIG=<percorso_config> CHECKPOINT=<percorso_checkpoint> sbatch cluster/eval.sh
```

#### Esempi di comandi diretti:
* **Teacher Model:**
  ```bash
  CONFIG=experiments/configs/experiments/teacher.yaml \
  CHECKPOINT=experiments/checkpoints/teacher_finetune_best.pth \
  sbatch cluster/eval.sh
  ```
* **Attention Transfer Symmetric (Run 4511):**
  ```bash
  CONFIG=experiments/configs/valid/at_symmetric_p50.yaml \
  CHECKPOINT=experiments/checkpoints/slurm-train-eval-4511/at_symmetric/distillation_at_best.pth \
  sbatch cluster/eval.sh
  ```
* **Attention Transfer Temporal Only (Run 4511):**
  ```bash
  CONFIG=experiments/configs/valid/at_temporal_p50.yaml \
  CHECKPOINT=experiments/checkpoints/slurm-train-eval-4511/at_temporal_only/distillation_at_best.pth \
  sbatch cluster/eval.sh
  ```

---

### 2. Metodo Rapido con Alias `evaluate` (Se attivi)
Se hai caricato gli alias tramite `source cluster/aliases.sh`, puoi usare la funzione `evaluate`:

* **Valutazione con Checkpoint Specifico:**
  ```bash
  evaluate <config> <checkpoint>
  ```
  *Esempio:*
  ```bash
  evaluate experiments/configs/valid/at_symmetric_p50.yaml experiments/checkpoints/slurm-train-eval-4511/at_symmetric/distillation_at_best.pth
  ```

* **Valutazione con Checkpoint di Default:**
  Se non specifichi il checkpoint, l'alias mapperà automaticamente la configurazione al rispettivo checkpoint "best" standard (es. per il teacher):
  ```bash
  evaluate experiments/configs/experiments/teacher.yaml
  ```

---

### 3. Monitoraggio dei Risultati dell'Evaluation
Dopo aver lanciato il job:
1. **Segui i log live** usando il Job ID restituito:
   ```bash
   tail -f logs/slurm-eval-<JOB_ID>.log
   # Oppure se usi gli alias, per vedere l'ultimo log lanciato:
   lastlog
   ```
2. **Leggi il sommario finale**:
   Alla fine del job, viene generato un sommario testuale e JSON con tutti i risultati nella cartella dell'esperimento:
   ```bash
   cat experiments/logs/slurm-eval-<JOB_ID>/evaluation_summary.txt
   ```

---

## 🏋️ Guida all'Addestramento (Training)
L'addestramento avvia il ciclo di training sul cluster usando i parametri definiti nel file `.yaml`.

### 1. Metodo Standard con `sbatch` (Senza alias)
Si usa lo script `cluster/train.sh` specificando la variabile d'ambiente `CONFIG`:

```bash
CONFIG=<percorso_config> sbatch cluster/train.sh
```
*Esempio:*
```bash
CONFIG=experiments/configs/experiments/baseline.yaml sbatch cluster/train.sh
```

---

### 2. Metodo Rapido con Alias `train` (Se attivi)
Se hai gli alias caricati, puoi semplicemente lanciare:
```bash
train <config_name> [eventuali_override...]
```
*Esempi:*
```bash
train baseline.yaml
train distillation.yaml --override training.lr=0.0005
```

---

### 3. Training Avanzato (Più modelli in sequenza o catena)
Se devi addestrare più modelli consecutivamente per non superare i limiti di GPU attive sul cluster:

* **Esecuzione Sequenziale (Un unico Job SLURM):**
  Lancia un unico job che esegue i training uno dopo l'altro.
  ```bash
  train-seq teacher.yaml baseline.yaml distillation.yaml
  ```
* **Esecuzione in Catena (Più Job con dipendenze `afterok`):**
  Sottomette più job distinti, ma ognuno partirà solo se il precedente si è concluso con successo.
  ```bash
  train-chain teacher.yaml baseline.yaml distillation.yaml
  ```
* **Pipeline Completa Auto-gestita:**
  Lancia in un colpo solo il training sequenziale di Teacher, Baseline e Distillation seguito dalle rispettive valutazioni sul test set:
  ```bash
  train-and-eval
  ```

---

## 🛠️ Comandi Utili per la Gestione del Cluster

### Monitoraggio Risorse e Code
* **`myjobs`**: Mostra tutti i tuoi job attivi o in coda con formattazione pulita.
* **`jobinfo <JOB_ID>`**: Informazioni dettagliate sullo stato di un job specifico.
* **`killjob <JOB_ID>`**: Cancella un job.
* **`killalljobs`**: Cancella tutti i tuoi job attivi.
* **`gpu`**: Esegue `nvidia-smi` per vedere lo stato delle GPU sul nodo corrente.
* **`diskusage`**: Mostra l'uso del disco suddiviso per cache HF, checkpoints e W&B.

### Manutenzione Spazio su Disco
Se ricevi errori di spazio esaurito (`Disk quota exceeded` o `No space left on device`):
```bash
# Esegue un dry-run per farti vedere quanto spazio puoi liberare
clean

# Rimuove i log vecchi di SLURM (mantiene solo gli ultimi 5) e le vecchie run offline di W&B
clean --force

# Pulisce completamente i pacchetti installati nella cartella locale
pip-clean
```
