# Piano di Miglioramento — Track 6: Knowledge Distillation for Mobile Action Recognition

## Stato attuale del progetto

### Risultati del run più affidabile (`slurm-train-eval-4201`, protocollo group-aware)

| Modello | Top-1 (test) | Top-5 (test) | Parametri | Size | Latency |
|---|---|---|---|---|---|
| **Teacher** (3D ResNet-50) | **88.71%** | 98.18% | 31.8M | 121.5 MB | 2.75 ms |
| **Baseline Student** (MobileNet3D) | 59.48% | 82.77% | 2.5M | 9.5 MB | 1.79 ms |
| **Distillation Student** (KD T=8, α=0.7) | **65.27%** | 87.73% | 2.5M | 9.5 MB | 1.80 ms |

**Compressione**: 12.8x parametri, 12.8x size — **KD guadagna +5.8 pp** rispetto al baseline.

### Gap da colmare
Il gap teacher→student distilled è ancora **23.4pp** su top-1. Le cause principali:
1. **Capacity gap enorme** (12.8x): il MobileNet3D ha poca capacità per replicare un ResNet-50 3D
2. **Distillation solo logit-based** (KDLoss): nessun trasferimento di feature intermedie nella run principale
3. **Augmentation conservativa**: `color_jitter=0.1`, no random resized crop, no temporal jitter
4. **Pool temporale identico** nel MobileNet3D: nessuna attenzione temporale esplicita

---

## Obiettivi minimi già implementati ✅
1. ✅ Teacher Model (3D ResNet-50 slow_r50 pretrained su Kinetics-400)
2. ✅ Baseline Student (MobileNet3D from scratch)
3. ✅ Knowledge Transfer (logit-based KD con T e α configurabili)
4. ✅ Evaluation completa (Top-1, Top-5, size MB, latency ms)

## Target da implementare

### Da obiettivi minimi (raffinamenti):
- Aumentare i frame: 24 → **32** (più contesto temporale)
- Ottimizzare l'augmentation (random temporal stride, RRC)
- Migliorare il training del teacher (più epoche, LR più bassa)

### Extra obiettivi richiesti (escluso temperature ablation):
1. ✅ **Attention Transfer** (`distillation_at`) — già implementato in codice, mai usato nel run principale group-aware → **da lanciare sul cluster**
2. **t-SNE visualizations** — confronto embedding space teacher vs baseline vs distilled student
3. **Analisi comprensione temporale** — confusion matrix e per-class accuracy analysis

---

## Modifiche al codice proposte

### 1. Nuovi config YAML ottimizzati

#### [NEW] `experiments/configs/distillation_at_24f_v2.yaml`
Config attention transfer con augmentation 24f, group-aware eval split, tuned per il run corretto.

#### [MODIFY] `experiments/configs/teacher_24f_evalsplit.yaml`  
Aggiunta `training.epochs: 60` (da 50) e LR schedule più gentile.

#### [NEW] `experiments/configs/distillation_t8_a07_24f_v2.yaml`
Versione pulita per il run definitivo con label smoothing 0.05 (invece di 0.1).

### 2. Visualizzazione t-SNE

#### [NEW] `src/evaluation/tsne_visualizer.py`
Script che:
- Carica teacher, baseline, distilled student
- Estrae embedding pre-classifier su test set
- Produce t-SNE plots 2D con plotly (colori per classe)
- Salva PNG e HTML interattivo

#### [NEW] `src/evaluation/confusion.py`
Script per confusion matrix e per-class accuracy analysis.

### 3. Script di submit aggiornato

#### [NEW] `cluster/submit_improved_v2.sh`
Pipeline completa: teacher (60 ep) → baseline (75 ep) → KD standard (75 ep) → KD+AT (75 ep) → t-SNE.

---

## Strategia di miglioramento delle performance

### A. Attention Transfer (Extra Objective #2)
L'AT è già implementato nel codice (`distillation_at` mode, `CombinedKDATLoss`). Il problema è che il run group-aware corrente ha usato solo KD standard. Lanciare `distillation_at` con:
- T=8, α=0.7, β=0.05 (abbassato da 0.1 per stabilità)
- teacher_keys=[3,4,5], student_keys=[2,4,6]
- 24 frame, same augmentation del run corrente

### B. Più epoche + LR warmup
Il baseline converge lentamente (best epoch 58/60): aumentare a **75 epoche** per una convergenza più completa.

### C. Augmentation temporale più ricca
Abilitare `max_temporal_stride=2` per diversificare la copertura temporale durante training.

### D. Teacher migliore
Il teacher attuale ottiene 88.71%. Rifinire con 60 epoche e LR decrescente nel tempo può portarlo sopra 90%.

---

## Piano di esecuzione (cluster)

```
Run 1: teacher (60 ep) → eval
Run 2: baseline (75 ep) → eval  [stesse condizioni precedenti per confronto fair]
Run 3: distillation standard (75 ep, T=8, α=0.7) → eval
Run 4: distillation_at (75 ep, T=8, α=0.7, β=0.05) → eval
Post-training: t-SNE analysis + confusion matrix
```

> [!IMPORTANT]
> L'ablazione temperature (T=1,5,10,20) è ESCLUSA dal run attuale per limitare i tempi. Può essere aggiunta in un secondo momento come job separato.

---

## Piano di verifica

### Test automatici
- Esecuzione del pipeline SLURM → check `pipeline_summary.txt`
- Verifica `eval/status_SUCCESS` per ogni fase

### Metriche attese (target)
| Modello | Top-1 atteso |
|---|---|
| Teacher (60 ep) | ~89-91% |
| Baseline (75 ep) | ~60-63% |
| KD standard (75 ep) | ~66-69% |
| KD + AT (75 ep) | ~68-71% |

### Visualizzazioni
- t-SNE su embedding di test set: clustering per classe più compatto nel distilled vs baseline
- Confusion matrix: analisi quali classi beneficiano di più dalla distillazione

---

## Open questions

> [!IMPORTANT]
> Il training del teacher richiede già 6-8 ore sul cluster. Aumentare le epoch da 50 a 60 è marginale. Confermi di voler ritrainingare il teacher o usare il checkpoint esistente (88.71%) per i nuovi run di distillation?

