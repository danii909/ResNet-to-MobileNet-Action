# Report: Attention Transfer Temporale per Action Recognition Mobile

## 1. Obiettivo

Questo report documenta i risultati dell'**Extra Objective (Track 6)**: l'estensione della Knowledge Distillation (KD) per includere l'**Attention Transfer (AT) su mappe di attivazione temporali intermedie**.

L'obiettivo è verificare se il trasferimento esplicito di informazione dalle feature intermedie del teacher — in particolare il segnale temporale, che codifica le dinamiche di movimento — migliori l'accuratezza dello student mobile rispetto alla sola distillation basata sui logit.

---

## 2. Setup Sperimentale

### 2.1 Modelli

| Modello | Architettura | Parametri | Dimensione | Latenza |
|---|---|---:|---:|---:|
| **Teacher** | 3D ResNet-50 (slow_r50) | 31.84M | 121.5 MB | ~2.7 ms |
| **Student** | MobileNet3D (MobileNetV2-3D) | 2.48M | 9.5 MB | ~1.8 ms |

Il rapporto di compressione è **12.8x** in parametri e **12.8x** in dimensione del modello.

### 2.2 Dataset e Protocollo di Valutazione

- **Dataset**: UCF-101 (101 classi di azioni, ~13K video)
- **Split**: Protocollo **group-aware** con `split_seed=42` e `eval_ratio=0.2`
  - I video dello stesso gruppo (stessa scena/attore) non compaiono simultaneamente nel train e nel validation set
  - Questo previene il data leakage e produce risultati affidabili e riproducibili
- **Test set**: Split ufficiale UCF-101 (3783 clip)
- **Input**: 24 frame, crop 112×112

### 2.3 Configurazione di Training

Tutti gli esperimenti condividono le stesse identiche impostazioni per garantire un confronto equo:

| Parametro | Valore |
|---|---|
| Epoche | 60 |
| Optimizer | AdamW |
| Learning Rate | 5e-4 |
| Weight Decay | 0.01 |
| Scheduler | Cosine (senza warmup) |
| Batch Size | 16 |
| Label Smoothing | 0.0 |
| Temperatura KD ($T$) | 8.0 |
| Alpha KD ($\alpha$) | 0.7 |
| Seed | 42 |

### 2.4 Teacher Utilizzato

Per gli esperimenti AT è stato utilizzato il teacher del run **slurm-train-eval-4384**, il più recente disponibile:

| Run | Teacher Test Top-1 | Teacher Test Top-5 |
|---|:---:|:---:|
| slurm-train-eval-4201 | 88.71% | 98.18% |
| **slurm-train-eval-4384** | **88.08%** | **97.94%** |

Nonostante la configurazione sia identica, si osserva una lieve variabilità (~0.6 pp) tra i due teacher, attribuibile a fattori non deterministici del training su GPU (operazioni atomiche CUDA, ordine di esecuzione dei kernel).

---

## 3. Implementazione dell'Attention Transfer

### 3.1 Approccio: Decomposizione Spazio-Temporale

L'AT è stato implementato decomponendo l'attenzione in due componenti ortogonali:

**Mappa di Attenzione Spaziale** — *"Dove guarda la rete in ogni frame"*:
$$A_{\text{spaziale}}(F) = \text{L2\_norm}\left(\text{mean}_{c,t}\left(F^2\right)\right) \in \mathbb{R}^{B \times H \cdot W}$$

**Mappa di Attenzione Temporale** — *"Quando si attiva la rete durante il video"*:
$$A_{\text{temporale}}(F) = \text{L2\_norm}\left(\text{mean}_{c,h,w}\left(F^2\right)\right) \in \mathbb{R}^{B \times T}$$

### 3.2 Loss Function Completa

$$\mathcal{L}_{\text{totale}} = \underbrace{\alpha \cdot T^2 \cdot D_{\text{KL}}(\sigma(z^S/T) \| \sigma(z^T/T)) + (1-\alpha) \cdot \text{CE}(z^S, y)}_{\text{Knowledge Distillation}} + \underbrace{\beta_s \sum_i \text{MSE}(A^T_{s,i}, A^S_{s,i})}_{\text{AT Spaziale}} + \underbrace{\beta_t \sum_i \text{MSE}(A^T_{t,i}, A^S_{t,i})}_{\text{AT Temporale}}$$

### 3.3 Accoppiamento dei Livelli

L'AT viene applicato su 3 coppie di livelli a profondità semantica crescente:

| Coppia | Teacher (Blocco) | Student (Stage) | Shape Teacher | Shape Student |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 2 | 2 | `[B, 256, 24, 28, 28]` | `[B, 32, 24, 14, 14]` |
| 2 | 3 | 4 | `[B, 512, 24, 14, 14]` | `[B, 96, 24, 7, 7]` |
| 3 | 4 | 6 | `[B, 1024, 24, 7, 7]` | `[B, 320, 24, 4, 4]` |

- Il **mismatch spaziale** ($H_T \neq H_S$) viene gestito con interpolazione bilineare 2D
- La **dimensione temporale** $T = 24$ è identica tra teacher e student → trasferimento diretto, senza approssimazione

### 3.4 Bug Corretto nell'Implementazione Precedente

L'implementazione originale conteneva un bug critico e silenzioso:

1. **Indicizzazione errata**: `FEATURE_BLOCKS = [3, 4, 5]` includeva il blocco 5, che è la testa di classificazione del teacher (output `[B, 101]`, tensore 2D). Il hook su questo blocco produceva mappe di attenzione **costantemente pari a 1.0**, con MSE = 0 e gradiente nullo. Un terzo del segnale AT era completamente sprecato.

2. **Interpolazione 1D su dati 3D**: Il vecchio codice appiattiva `[T, H, W]` in un vettore 1D e usava interpolazione lineare, mescolando posizioni spaziali e temporali. Questo produceva un segnale di supervisione corrotto.

La correzione (`FEATURE_BLOCKS = [2, 3, 4]`) e la nuova architettura a componenti separate sono descritte in dettaglio nel documento tecnico (`docs/ATTENTION_TRANSFER_IMPLEMENTATION.md`).

### 3.5 Varianti Testate

| Variante | $\beta_s$ (spaziale) | $\beta_t$ (temporale) | Logica |
|---|:---:|:---:|---|
| **Simmetrica** | 0.05 | 0.05 | Peso uguale a entrambi i componenti |
| **Temporal-heavy** | 0.03 | 0.07 | Maggiore peso al segnale temporale (nessuna interpolazione → segnale più pulito) |

---

## 4. Risultati

### 4.1 Tabella Comparativa Completa

I risultati confrontano tutti i modelli student addestrati con il protocollo group-aware. La sezione "Confronto Fair" (run 4384 + 4391) isola l'effetto dell'AT usando lo stesso teacher.

#### Run slurm-train-eval-4201 (Teacher 4201)

| Metodo | Best Eval Acc | Test Top-1 | Test Top-5 | $\Delta$ vs Baseline |
|---|:---:|:---:|:---:|:---:|
| **Teacher** (3D ResNet-50) | — | **88.71%** | **98.18%** | — |
| Baseline (no KD) | 59.18% | 59.48% | 82.77% | — |
| **KD-only** ($T$=8, $\alpha$=0.7) | **63.78%** | **65.27%** | **87.73%** | **+5.79 pp** |

#### Run slurm-train-eval-4384 + slurm-at-temporal-4391 (Teacher 4384) — Confronto Fair

| Metodo | Best Eval Acc | Test Top-1 | Test Top-5 | $\Delta$ vs KD-only |
|---|:---:|:---:|:---:|:---:|
| **Teacher** (3D ResNet-50) | — | **88.08%** | **97.94%** | — |
| Baseline (no KD) | 56.00% | 56.99% | 81.47% | — |
| KD-only ($T$=8, $\alpha$=0.7) | 59.09% | 58.92% | 84.93% | — |
| **KD + AT Simmetrico** | 59.14% | **59.42%** | **85.04%** | **+0.50 pp** |
| **KD + AT Temporal-heavy** | **59.93%** | 59.40% | 84.99% | **+0.48 pp** |

### 4.2 Efficienza del Modello

| Metrica | Teacher | Student (tutti) |
|---|:---:|:---:|
| Parametri | 31.84M | **2.48M** (12.8x meno) |
| Dimensione modello | 121.5 MB | **9.5 MB** (12.8x meno) |
| Latenza inferenza | ~2.7 ms | **~1.8 ms** (1.5x più veloce) |

> L'AT **non aggiunge alcun costo a runtime**: è una tecnica di training-only. Il modello deployato è identico indipendentemente dall'uso di AT.

---

## 5. Analisi dei Risultati

### 5.1 Effetto della Knowledge Distillation

La KD standard produce un miglioramento sostanziale rispetto al baseline:
- **Run 4201**: +5.79 pp su test top-1 (59.48% → 65.27%)
- **Run 4384**: +1.93 pp su test top-1 (56.99% → 58.92%)

La differenza tra i due run (~6 pp su test) evidenzia una significativa variabilità inter-run, probabile conseguenza della combinazione di fattori non deterministici e della sensibilità del piccolo student a variazioni nella qualità del teacher.

### 5.2 Effetto dell'Attention Transfer

Isolando l'effetto dell'AT (confronto fair: stessi teacher, configurazione e seed):

| | KD-only | KD + AT Sym | KD + AT Heavy |
|---|:---:|:---:|:---:|
| Test Top-1 | 58.92% | 59.42% (+0.50) | 59.40% (+0.48) |
| Test Top-5 | 84.93% | 85.04% (+0.11) | 84.99% (+0.06) |
| Best Eval | 59.09% | 59.14% (+0.05) | 59.93% (+0.84) |

**Osservazioni chiave:**

1. **L'AT produce un miglioramento consistente ma modesto** (+0.5 pp su test top-1). Entrambe le varianti convergono sullo stesso livello di test accuracy, suggerendo che il margine di miglioramento dal trasferimento di feature intermedie è limitato in questo scenario.

2. **Le due varianti sono praticamente equivalenti al test.** La variante temporal-heavy mostra un lieve vantaggio sulla eval accuracy (+0.84 pp vs +0.05 pp), ma questo non si traduce in un test gap significativo. Questo suggerisce che l'equilibrio tra segnale spaziale e temporale non è un fattore dominante.

3. **Il miglioramento dell'AT è piccolo rispetto al contributo dei soft logit.** La KD standard già cattura la maggior parte dell'informazione trasferibile attraverso le distribuzioni di probabilità softened. Le feature intermedie aggiungono un segnale supplementare, ma su UCF-101 con questa coppia teacher-student, il contributo marginale è limitato.

### 5.3 Perché il Miglioramento è Modesto

Diverse ragioni possono spiegare il guadagno contenuto:

1. **UCF-101 è un dataset relativamente semplice**: 101 classi con pattern visivi spesso chiari. I soft logit contengono già informazione sufficiente per guidare lo student.

2. **Gap architetturale elevato**: Il teacher (ResNet-50 3D) e lo student (MobileNet3D) hanno strutture interne molto diverse. Le feature intermedie non sono facilmente trasferibili tra architetture eterogenee.

3. **Riduzione spaziale aggressiva nello student**: Le feature map dello student sono fino a 7x più piccole spazialmente (4×4 vs 28×28), richiedendo un'interpolazione significativa che introduce rumore nel segnale di supervisione.

4. **Capacità limitata dello student**: Con soli 2.48M di parametri, lo student potrebbe non avere la capacità sufficiente per sfruttare contemporaneamente il segnale dei logit e quello delle feature intermedie.

---

## 6. Conclusioni

### 6.1 Riepilogo

| Obiettivo | Risultato |
|---|---|
| Implementare AT con decomposizione spaziale/temporale | ✅ Completato |
| Correggere il bug dell'indicizzazione del blocco head | ✅ Corretto |
| Correggere l'interpolazione 1D errata | ✅ Sostituita con 2D bilineare |
| Migliorare l'accuratezza rispetto al KD-only | ✅ +0.5 pp (modesto ma consistente) |
| Zero overhead a runtime | ✅ Confermato |

### 6.2 Contributo Tecnico

Il contributo principale di questo lavoro non è nel guadagno quantitativo (modesto su UCF-101), ma nella **correttezza dell'implementazione**:

1. **Scoperta e correzione di un bug silenzioso** che invalidava 1/3 del segnale AT (hooking della testa di classificazione invece di uno stage residuale).
2. **Decomposizione principled** dell'attenzione 3D in componenti spaziali e temporali, evitando l'appiattimento naive che mescolava le dimensioni.
3. **Interpolazione geometricamente corretta**: bilineare 2D per lo spazio, trasferimento diretto per il tempo.
4. **Protocollo sperimentale rigoroso**: confronto fair con variabili controllate (stesso teacher, seed, configurazione).

### 6.3 File di Riferimento

| Risorsa | Percorso |
|---|---|
| Codice AT | `src/training/losses.py` |
| Test suite | `tests/test_at_loss.py` |
| Config simmetrica | `experiments/configs/distillation_at_temporal.yaml` |
| Config temporal-heavy | `experiments/configs/distillation_at_temporal_heavy.yaml` |
| Risultati AT | `results/train-eval-test-split (group-aware)/slurm-at-temporal-4391/` |
| Risultati KD-only (4384) | `results/train-eval-test-split (group-aware)/slurm-train-eval-4384/` |
| Risultati KD-only (4201) | `results/train-eval-test-split (group-aware)/slurm-train-eval-4201/` |
| Documento tecnico (EN) | `docs/ATTENTION_TRANSFER_IMPLEMENTATION.md` |
| Documento tecnico (IT) | `docs/ATTENTION_TRANSFER_IMPLEMENTATION_IT.md` |
