# Attention Transfer per Action Recognition 3D: Dettagli Implementativi

## Abstract

Questo documento descrive la progettazione, l'implementazione e il debugging del modulo di Attention Transfer (AT) utilizzato nella nostra pipeline di Knowledge Distillation per la classificazione di azioni video (action recognition) su dispositivi mobili. Il meccanismo di AT trasferisce le rappresentazioni delle feature intermedie da un modello "teacher" 3D ResNet-50 a un modello "student" MobileNet3D sul dataset UCF-101. Vengono descritti i fondamenti teorici, il bug critico scoperto nell'implementazione originale (indicizzazione errata dei blocchi di feature e interpolazione spaziale non corretta) e l'approccio corretto che decompone l'attenzione in componenti spaziali e temporali separate.

---

## 1. Background

### 1.1 Attention Transfer nella Knowledge Distillation

L'Attention Transfer (Zagoruyko & Komodakis, "Paying More Attention to Attention", ICLR 2017) estende la standard Knowledge Distillation basata sui logit trasferendo le informazioni dalle feature map intermedie del teacher a quelle dello student. L'idea centrale è che la distribuzione spaziale delle attivazioni — ovvero dove la rete "presta attenzione" — trasporta informazioni discriminative da cui lo student può imparare, al di là di quanto trasmettano i soli soft logit.

Per la classificazione di immagini 2D, dato un tensore di feature $F \in \mathbb{R}^{B \times C \times H \times W}$, la mappa di attenzione spaziale è definita come:

$$A(F) = \text{normalize}\left(\sum_{c=1}^{C} |F_c|^2\right) \in \mathbb{R}^{B \times H \times W}$$

La loss di AT è quindi:

$$\mathcal{L}_{\text{AT}} = \beta \sum_{i} \text{MSE}\left(A(F^T_i),\ A(F^S_i)\right)$$

dove $F^T_i$ e $F^S_i$ sono coppie di feature map intermedie del teacher e dello student, e $\beta$ è un coefficiente di peso.

### 1.2 Estensione alle Feature Video 3D

Nella comprensione dei video, le feature intermedie sono tensori a 5 dimensioni:

$$F \in \mathbb{R}^{B \times C \times T \times H \times W}$$

dove $T$ è la dimensione temporale (numero di frame mantenuti in quel livello). Un'estensione ingenua (naive) dell'AT 2D al 3D consiste nel sommare sui canali e appiattire (flattening) tutte le dimensioni rimanenti in un singolo vettore. Questo è esattamente ciò che faceva l'implementazione originale nella nostra codebase. Come mostreremo di seguito, questo approccio è sia teoricamente errato che bacato a livello pratico.

---

## 2. Panoramica dell'Architettura

### 2.1 Teacher: 3D ResNet-50 (slow_r50)

Il modello teacher è la `slow_r50` di PyTorchVideo, pre-addestrata su Kinetics-400 e sottoposta a fine-tuning su UCF-101. Il modello è costituito da 6 blocchi sequenziali (indici 0–5):

| Indice Blocco | Tipo | Shape di Output (circa) | Descrizione |
|:-----------:|------|:----------------------:|-------------|
| 0 | `ResStage` | `[B, 64, T, 56, 56]` | Convoluzione iniziale + pool |
| 1 | `ResStage` | `[B, 64, T, 56, 56]` | Primo stage residuale |
| 2 | `ResStage` | `[B, 256, T, 28, 28]` | Secondo stage residuale |
| 3 | `ResStage` | `[B, 512, T, 14, 14]` | Terzo stage residuale |
| 4 | `ResStage` | `[B, 1024, T, 7, 7]` | Quarto stage residuale |
| **5** | **`ResNetBasicHead`** | **`[B, num_classes]`** | **Global avg pool → dropout → proiezione lineare** |

> **Osservazione critica:** Il blocco 5 **non** è uno stage di estrazione di feature. È la testa di classificazione (classification head) che esegue il global average pooling su tutte le dimensioni spaziali e temporali, seguito da una proiezione lineare a `num_classes` dimensioni. Il suo output è un **tensore 2D** `[B, 101]`, non una feature map 5D.

### 2.2 Student: MobileNet3D

Lo student è un'architettura MobileNetV2 adattata per le convoluzioni 3D. Processa l'input attraverso una convoluzione iniziale (stem) seguita da 7 stage residuali invertiti (indici 0–6):

| Indice Stage | Canali di Output | Stride | Shape di Output (circa) |
|:-----------:|:---------------:|:------:|:----------------------:|
| 0 | 16 | 1 | `[B, 16, T, 56, 56]` |
| 1 | 24 | 2 | `[B, 24, T, 28, 28]` |
| **2** | **32** | **2** | **`[B, 32, T, 14, 14]`** |
| 3 | 64 | 2 | `[B, 64, T, 7, 7]` |
| **4** | **96** | **1** | **`[B, 96, T, 7, 7]`** |
| 5 | 160 | 2 | `[B, 160, T, 4, 4]` |
| **6** | **320** | **1** | **`[B, 320, T, 4, 4]`** |

Entrambi i modelli preservano la dimensione temporale $T$ attraverso tutti gli stage di estrazione delle feature (lo stride temporale è 1 ovunque), il che significa che $T$ è identico tra teacher e student ad ogni livello accoppiato.

### 2.3 Accoppiamento delle Feature per l'AT

L'accoppiamento corretto abbina i livelli con livelli semantici simili e risoluzioni spaziali approssimativamente compatibili:

| Coppia | Blocco Teacher | Shape Teacher | Stage Student | Shape Student |
|:----:|:------------:|:-------------:|:-------------:|:-------------:|
| 1 | Blocco **2** | `[B, 256, T, 28, 28]` | Stage **2** | `[B, 32, T, 14, 14]` |
| 2 | Blocco **3** | `[B, 512, T, 14, 14]` | Stage **4** | `[B, 96, T, 7, 7]` |
| 3 | Blocco **4** | `[B, 1024, T, 7, 7]` | Stage **6** | `[B, 320, T, 4, 4]` |

---

## 3. Il Bug: Indicizzazione Errata dei Blocchi di Feature

### 3.1 Cosa era sbagliato

L'implementazione originale in `teacher.py` definiva:

```python
FEATURE_BLOCKS = [3, 4, 5]  # SBAGLIATO
```

Questo registrava forward hook sui blocchi 3, 4 e **5**. Poiché il blocco 5 è la `ResNetBasicHead` (testa di classificazione), l'hook ne catturava l'output: un **tensore 2D** di forma `[B, num_classes]`.

### 3.2 Perché l'errore era silenzioso

La funzione originale per la mappa di attenzione spaziale non validava la dimensionalità dell'input:

```python
def _spatial_attention_map(features):
    attn = (features ** 2).sum(dim=1)   # Su 2D: [B, 101] → [B] (uno scalare per sample!)
    attn = attn.view(attn.size(0), -1)  # [B] → [B, 1]
    attn = F.normalize(attn, p=2, dim=1)  # normalizza un singolo valore → sempre 1.0
    return attn
```

Quando applicato all'output 2D della testa `[B, 101]`:
- `.sum(dim=1)` sommava lungo i 101 logit delle classi, producendo uno **scalare per sample** `[B]`
- `.view(B, -1)` ne faceva il reshape a `[B, 1]`
- `F.normalize(..., p=2, dim=1)` su un singolo elemento restituisce sempre `1.0`

Le "mappe di attenzione" risultanti per la terza coppia erano **sempre costanti** (un vettore di 1). L'MSE tra due vettori composti solo da '1' è zero, non contribuendo in alcun modo alla loss e non producendo alcun errore. Questo significava che:

1. **Un terzo del segnale AT veniva sprecato** — la coppia di feature di livello più alto contribuiva con gradiente zero.
2. **Nessun errore a runtime veniva generato** — il bug era completamente silenzioso.
3. **Nessun test l'aveva rilevato** — senza controlli (assert) sulla dimensionalità, il codice sembrava funzionare.

### 3.3 Il problema dell'interpolazione

Anche per le due coppie di feature valide (blocchi 3 e 4), il codice originale aveva un secondo problema. Quando le dimensioni spaziali del teacher e dello student differivano, appiattiva (flattening) la mappa di attenzione 5D in 1D e applicava un'**interpolazione lineare**:

```python
# Originale (sbagliato)
attn = (features ** 2).sum(dim=1)  # [B, T, H, W]
attn = attn.view(attn.size(0), -1)  # [B, T*H*W] — appiattito!
# ...
s_attn = F.interpolate(s_attn.unsqueeze(1), size=t_attn.shape[1], mode="linear")
```

Questo è matematicamente scorretto. Consideriamo le feature del teacher con shape `[B, C, 24, 14, 14]` e quelle dello student con shape `[B, C, 24, 4, 4]`. Dopo la somma sui canali e il flatten:

- Teacher: `[B, 24 * 14 * 14]` = `[B, 4704]`
- Student: `[B, 24 * 4 * 4]` = `[B, 384]`

L'interpolazione lineare 1D da 384 a 4704 le tratta come segnali 1D. Ma i dati sottostanti hanno la struttura `[T, H, W]` — tre dimensioni intrecciate. Interpolare lungo un singolo asse **mescola le posizioni spaziali e temporali**. Per esempio, il valore interpolato alla posizione 400 nel vettore flat del teacher corrisponde al timestep $t = \lfloor 400 / 196 \rfloor = 2$, e alla posizione spaziale $(h, w) = (400 \mod 196)$, ma la corrispondente posizione nello student $400 \times (384/4704) \approx 33$ viene mappata in un $(t, h, w)$ completamente diverso nello spazio delle feature dello student.

---

## 4. La Soluzione: Attenzione Spaziale e Temporale Separate

### 4.1 Logica di Progettazione

Per le feature video 3D, le dimensioni spaziali $(H, W)$ e la dimensione temporale $(T)$ trasportano informazioni semantiche fondamentalmente diverse:

- L'**attenzione spaziale** risponde a: *"Dove si concentra la rete in ogni frame?"* — questo cattura la posizione dell'oggetto, le regioni salienti e i pattern visivi.
- L'**attenzione temporale** risponde a: *"Quando si ha il picco di attivazione durante il video?"* — questo cattura le dinamiche di movimento, le fasi dell'azione e la struttura temporale.

Far collassare tutto questo in un singolo vettore piatto, come faceva l'implementazione originale, distrugge questa distinzione e rende impossibile per la loss fornire gradienti separati per le correzioni spaziali rispetto a quelle temporali.

### 4.2 Formulazione Matematica

Dato un tensore di feature 5D $F \in \mathbb{R}^{B \times C \times T \times H \times W}$, definiamo:

**Mappa di Attenzione Spaziale:**
$$A_{\text{spaziale}}(F) = \text{L2\_normalize}\left(\text{mean}_{c,t}\left(F^2\right)\right) \in \mathbb{R}^{B \times H \cdot W}$$

Questa operazione calcola la media su tutti i canali e tutti i timestep, producendo una singola heatmap spaziale che riassume *dove* la rete presta attenzione per l'intera clip.

**Mappa di Attenzione Temporale:**
$$A_{\text{temporale}}(F) = \text{L2\_normalize}\left(\text{mean}_{c,h,w}\left(F^2\right)\right) \in \mathbb{R}^{B \times T}$$

Questa operazione calcola la media su tutti i canali e tutte le posizioni spaziali, producendo un profilo temporale che riassume *quando* si raggiungono i picchi di intensità di attivazione.

**Loss AT Combinata:**
$$\mathcal{L}_{\text{AT}} = \beta_s \sum_{i} \text{MSE}\left(A^T_{\text{spaziale},i},\ A^S_{\text{spaziale},i}\right) + \beta_t \sum_{i} \text{MSE}\left(A^T_{\text{temporale},i},\ A^S_{\text{temporale},i}\right)$$

dove $\beta_s$ e $\beta_t$ sono coefficienti di peso indipendenti per i componenti spaziale e temporale.

### 4.3 Gestione dei Mismatch Dimensionali

**Mismatch spaziale** ($H_T \times W_T \neq H_S \times W_S$): La mappa di attenzione spaziale dello student viene trasformata (reshaped) in `[B, 1, H_s, W_s]`, interpolata usando l'**interpolazione bilineare 2D** a `[B, 1, H_t, W_t]`, per poi essere nuovamente appiattita e ri-normalizzata. Questo preserva la struttura spaziale 2D e interpola correttamente tra le posizioni della griglia.

```python
# Corretto: l'interpolazione bilineare 2D preserva la struttura spaziale
s_spatial_2d = s_spatial.view(-1, 1, H_s, W_s)
s_spatial_2d = F.interpolate(s_spatial_2d, size=(H_t, W_t), mode="bilinear", align_corners=False)
s_spatial = s_spatial_2d.view(B, -1)
s_spatial = F.normalize(s_spatial, p=2, dim=1)
```

**Mismatch temporale** ($T_T \neq T_S$): Nella nostra architettura, sia il teacher che lo student hanno stride temporale pari a 1 a tutti i livelli, quindi $T_T = T_S = T_{\text{input}}$. **Nessuna interpolazione è necessaria.** Questo è un vantaggio significativo del componente di attenzione temporale: il segnale viene trasferito direttamente senza alcuna approssimazione. È comunque stato incluso un meccanismo di fallback di sicurezza che utilizza l'interpolazione lineare 1D per maggiore robustezza, ma non ci si aspetta che venga attivato in condizioni normali.

### 4.4 Funzione di Loss Totale

La loss completa di distillation con AT spaziale e temporale è:

$$\mathcal{L}_{\text{totale}} = \underbrace{\alpha \cdot T^2 \cdot D_{\text{KL}}\left(\sigma(z^S / T)\ ||\ \sigma(z^T / T)\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(z^S, y)}_{\text{Knowledge Distillation}} + \underbrace{\beta_s \sum_i \text{MSE}(A^T_{s,i}, A^S_{s,i})}_{\text{AT Spaziale}} + \underbrace{\beta_t \sum_i \text{MSE}(A^T_{t,i}, A^S_{t,i})}_{\text{AT Temporale}}$$

---

## 5. Dettagli Implementativi

### 5.1 Struttura del Codice

L'implementazione risiede in `src/training/losses.py` e consiste in:

| Componente | Descrizione |
|---|---|
| `_spatial_attention_map(F)` | Calcola l'attenzione spaziale `[B, H*W]` da feature 4D o 5D |
| `_temporal_attention_map(F)` | Calcola l'attenzione temporale `[B, T]` da feature 5D |
| `AttentionTransferLoss` | Modulo `nn.Module` che calcola AT spaziale + temporale sui livelli appaiati |
| `CombinedKDATLoss` | Modulo `nn.Module` che combina la loss di KD + la loss di AT nell'obiettivo finale |

### 5.2 Mappa di Attenzione Spaziale

```python
def _spatial_attention_map(features: torch.Tensor) -> torch.Tensor:
    if features.dim() == 5:
        attn = (features ** 2).mean(dim=(1, 2))  # media su C e T → [B, H, W]
    elif features.dim() == 4:
        attn = (features ** 2).mean(dim=1)        # media su C → [B, H, W]
    else:
        raise ValueError(f"Attesi tensori 4D o 5D, ricevuto {features.dim()}D")
    attn = attn.view(attn.size(0), -1)            # [B, H*W]
    attn = F.normalize(attn, p=2, dim=1)
    return attn
```

Scelte progettuali chiave:
- **`mean` (media) invece di `sum` (somma)**: Usare la media sui canali e sul tempo rende la magnitudine indipendente dal numero di canali $C$ (che differisce tra teacher e student) e dalla lunghezza temporale $T$. Questo produce scale comparabili senza bisogno di normalizzazioni aggiuntive.
- **Controllo dimensionale esplicito**: Lancia un `ValueError` in caso di input inattesi (es. l'output 2D della testa di classificazione), impedendo fallimenti silenziosi.

### 5.3 Mappa di Attenzione Temporale

```python
def _temporal_attention_map(features: torch.Tensor) -> torch.Tensor:
    if features.dim() != 5:
        raise ValueError(f"L'attenzione temporale richiede feature 5D, ricevuto {features.dim()}D")
    attn = (features ** 2).mean(dim=(1, 3, 4))  # media su C, H, W → [B, T]
    attn = F.normalize(attn, p=2, dim=1)
    return attn
```

La mappa temporale è definita rigorosamente per input 5D. Produce un vettore $T$-dimensionale per ogni sample, dove ciascun valore rappresenta l'intensità aggregata di attivazione a quel timestep. La normalizzazione L2 assicura che il profilo temporale sia confrontato in base alla sua forma (distribuzione relativa), non in base alla magnitudine assoluta.

### 5.4 Retrocompatibilità (Backward Compatibility)

Il trainer legge gli iperparametri per l'AT dalla configurazione YAML con un meccanismo di fallback:

```python
at_beta_fallback = kd_cfg.get("at_beta", 0.05)
beta_spatial = kd_cfg.get("at_beta_spatial", at_beta_fallback)
beta_temporal = kd_cfg.get("at_beta_temporal", at_beta_fallback)
```

Questo assicura che i vecchi file di configurazione che specificano un singolo valore `at_beta` continuino a funzionare correttamente: il valore viene usato sia per il componente spaziale che per quello temporale. Le nuove configurazioni possono specificare `at_beta_spatial` e `at_beta_temporal` in modo indipendente.

---

## 6. Configurazione Sperimentale

### 6.1 Protocollo per un Confronto Equo (Fair Comparison)

Per isolare l'effetto dell'AT dalle altre scelte iperparametriche, gli esperimenti AT utilizzano **impostazioni identiche** rispetto al run di base di distillation validato (`slurm-train-eval-4201`):

| Parametro | Valore | Nota |
|---|---|---|
| Epochs | 60 | Come 4201 |
| Scheduler | `cosine` | Come 4201 (no warmup) |
| Learning rate | 0.0005 | Come 4201 |
| Batch size | 16 | Come 4201 |
| Label smoothing | 0.0 | Come 4201 (default) |
| Temperatura $T$ | 8.0 | Come 4201 |
| Alpha $\alpha$ | 0.7 | Come 4201 |
| Num frames | 24 | Come 4201 |
| Eval split | Group-aware, seed=42 | Come 4201 |

Le **uniche** aggiunte sono: `at_beta_spatial`, `at_beta_temporal`, `teacher_keys`, e `student_keys`.

### 6.2 Varianti di Iperparametri

Vengono testate due configurazioni:

| Variante | $\beta_s$ | $\beta_t$ | Logica (Rationale) |
|---|:---:|:---:|---|
| **Simmetrica** | 0.05 | 0.05 | Peso uguale ai segnali spaziali e temporali |
| **Temporal-heavy** | 0.03 | 0.07 | L'AT temporale non presenta rumore da interpolazione (corrispondenza esatta di $T$), quindi può trasportare un segnale più forte. L'AT spaziale con l'interpolazione bilineare introduce invece un certo errore di approssimazione, giustificando un peso inferiore. |

---

## 7. Verifica (Verification)

### 7.1 Test Automatici

Una test suite (`tests/test_at_loss.py`) verifica:

1. **Correttezza delle shape**: la mappa spaziale produce `[B, H*W]`, quella temporale produce `[B, T]`
2. **Normalizzazione L2**: le norme in output vengono verificate affinché siano pari a 1.0
3. **Controlli di dimensionalità**: un input 4D alla mappa temporale solleva un `ValueError`
4. **Gestione del mismatch spaziale**: l'interpolazione bilineare 2D produce le shape corrette quando $H_T \neq H_S$
5. **Calcolo su coppie multiple**: 3 coppie di feature producono una loss finita e non negativa
6. **Flusso del gradiente (Gradient flow)**: il backward pass propaga i gradienti sia alle feature dello student che alla testa di classificazione dello student
7. **Retrocompatibilità**: una vecchia configurazione con solo `at_beta` produce $\beta_s = \beta_t$ in modo identico
8. **Isolamento dei componenti**: impostando $\beta_s = 0$ si produce una loss spaziale nulla con una loss temporale non nulla

Tutti e 10 i test passano con successo.

### 7.2 Validazione dell'Integrazione

La pipeline SLURM (`cluster/submit_at_temporal.sh`) addestra entrambe le configurazioni sequenzialmente con una valutazione dopo ciascuna, producendo un `pipeline_summary.txt` che permette di fare un confronto diretto con l'accuratezza in fase di valutazione (63.78%) ottenuta dal run di base solo KD.

---

## 8. Riepilogo dei Cambiamenti (Summary of Changes)

| File | Cambiamento | Scopo |
|---|---|---|
| `src/models/teacher.py` | `FEATURE_BLOCKS`: `[3,4,5]` → `[2,3,4]` | Esclusione della testa di classificazione (blocco 5) |
| `src/models/student.py` | Aggiornamento commenti | Documentare il corretto accoppiamento dei blocchi |
| `src/training/losses.py` | Riscritto | AT spaziale + temporale separati con interpolazione corretta |
| `src/training/trainer.py` | Aggiornamento `_build_criterion` | Supporto per `at_beta_spatial` / `at_beta_temporal` con fallback |
| `experiments/configs/distillation_at_temporal.yaml` | Nuova config | AT Simmetrico ($\beta_s = \beta_t = 0.05$) |
| `experiments/configs/distillation_at_temporal_heavy.yaml` | Nuova config | AT Temporal-heavy ($\beta_s = 0.03$, $\beta_t = 0.07$) |
| `experiments/configs/*.yaml` (3 file) | Fix `teacher_keys` | Tutte le config AT esistenti sono state aggiornate a `[2,3,4]` |
| `cluster/submit_at_temporal.sh` | Nuovo script | Singolo job SLURM per entrambi gli esperimenti AT |
| `tests/test_at_loss.py` | Nuovi test | 10 test di verifica automatici |

---

## Riferimenti (References)

1. Zagoruyko, S., & Komodakis, N. (2017). *Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer*. ICLR 2017.
2. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. NeurIPS Workshop.
3. Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). *SlowFast Networks for Video Recognition*. ICCV 2019.
4. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018.
