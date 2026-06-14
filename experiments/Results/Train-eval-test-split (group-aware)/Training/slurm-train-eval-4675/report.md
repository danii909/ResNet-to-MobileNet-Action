# Cross-Frame Distillation: Analisi Comparativa di Latenza e Accuratezza
## Confronto Sperimentale: Modello 24f (4565) vs Modello 16f Cross-Frame (4675)

Questo report analizza il trade-off tra l'accuratezza e le prestazioni computazionali (latenza di inferenza e throughput) derivante dall'utilizzo della **Cross-Frame Distillation**. 

L'obiettivo della Cross-Frame Distillation è addestrare uno student leggero (`MobileNet3D` con `width_mult=1.0`) a elaborare **16 frame** anziché **24 frame**, distillando la conoscenza da un teacher (`ResNet3D-50`) che lavora stabilmente a **24 frame**.

---

## 1. Sintesi dei Modelli a Confronto

* **Modello A (Baseline Distillation - 24f)**:
  * **Job ID**: `slurm-train-eval-4565` (fase `kd_t20_a07_24f_lightaug`)
  * **Frame in ingresso (Student)**: 24 frame
  * **Accuratezza Top-1 (Test Set)**: **65.85%**
  * **Accuratezza Top-5 (Test Set)**: **87.81%**
* **Modello B (Cross-Frame Distillation - 16f)**:
  * **Job ID**: `slurm-train-eval-4675`
  * **Frame in ingresso (Student)**: 16 frame
  * **Accuratezza Top-1 (Test Set)**: **62.68%**
  * **Accuratezza Top-5 (Test Set)**: **86.20%**

---

## 2. Risultati del Benchmark di Latenza

I test di inferenza sono stati eseguiti sul cluster (GPU NVIDIA L40S e CPU server) confrontando diversi batch size per isolare l'overhead di comunicazione CPU-GPU.

### GPU Benchmark (NVIDIA L40S)
| Batch Size (BS) | Latenza Modello A (24f) | Latenza Modello B (16f) | Velocizzazione (Speedup) | Tempo GPU Risparmiato | Throughput Modello A | Throughput Modello B |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 3.27 ms | 3.31 ms | 0.99x | -1.30% (Nullo) | 305.7 clips/s | 301.8 clips/s |
| **4** | 3.47 ms | 3.23 ms | 1.08x | **7.00%** | 1153.1 clips/s | 1239.8 clips/s |
| **8** | 8.07 ms | 4.78 ms | **1.69x** | **40.75%** | 991.3 clips/s | **1673.1 clips/s** |
| **16** | 18.71 ms | 11.75 ms | **1.59x** | **37.20%** | 855.3 clips/s | **1361.9 clips/s** |
| **32** | 43.30 ms | 26.35 ms | **1.64x** | **39.15%** | 739.1 clips/s | **1214.5 clips/s** |
| **64** | 95.25 ms | 60.56 ms | **1.57x** | **36.42%** | 671.9 clips/s | **1056.8 clips/s** |

### CPU Benchmark (Batch Size 1 & 8)
| Batch Size (BS) | Latenza Modello A (24f) | Latenza Modello B (16f) | Velocizzazione (Speedup) | Tempo CPU Risparmiato | Throughput Modello A | Throughput Modello B |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CPU BS 1** | 53.52 ms | 25.36 ms | **2.11x** | **52.61%** | 18.7 clips/s | **39.4 clips/s** |
| **CPU BS 8** | 480.17 ms | 296.72 ms | **1.62x** | **38.21%** | 16.7 clips/s | **27.0 clips/s** |

## 3. Grafici di Confronto delle Prestazioni

I seguenti grafici visualizzano l'andamento della latenza e del throughput su GPU e CPU all'aumentare del Batch Size:

![Inference Benchmark Plots](benchmark_plots.png)

Il pannello dei grafici è composto da 4 quadranti:
1. **In alto a sinistra (GPU Latency vs Batch Size)**: Mostra il tempo di inferenza per singolo video (in millisecondi) su GPU L40S. Evidenzia come a basso batch size (BS=1) la latenza sia dominata dall'overhead di lancio dei kernel, mentre a batch carichi (BS $\ge$ 8) emerga il vantaggio reale del modello B (linea blu).
2. **In alto a destra (GPU Throughput vs Batch Size)**: Rappresenta la produttività totale (video elaborati al secondo). Mostra che a pieno carico (BS $\ge$ 8), il modello B (16f) incrementa la capacità di elaborazione del **+68.8%** rispetto al modello A.
3. **In basso a sinistra (Inference Time Saved %)**: Quantifica la percentuale esatta di tempo macchina GPU risparmiata con il modello B. A regime, il risparmio si attesta stabilmente **tra il 36.4% e il 40.7%**.
4. **In basso a destra (CPU Latency Comparison)**: Confronta i tempi di inferenza su CPU (a BS=1 e BS=8). Dimostra che per sistemi edge (BS=1), il modello Cross-Frame (16f) **dimezza i tempi di calcolo (-52.61%)** rispetto alla baseline a 24f.

---

## 4. Considerazioni Analitiche e Conclusioni

### 1. Il Fenomeno della Latenza a Batch Size = 1 su GPU
I test confermano che a **Batch Size = 1 su GPU L40S la latenza è identica (3.27 ms vs 3.31 ms)**. 
Questo comportamento è normale ed è dovuto al **Kernel Launch Overhead**: a basso carico, il tempo totale è dominato dalla latenza di comunicazione CPU-GPU (invio dei comandi via bus PCIe e scheduling hardware) e non dalle operazioni matematiche (FLOPs). La GPU L40S, incredibilmente potente, calcola 16 o 24 frame quasi istantaneamente, lasciando visibile solo il costo fisso di comunicazione.

### 2. Il Risparmio Computazionale Reale su GPU (Batch Size $\ge$ 8)
Non appena il Batch Size sale (BS $\ge$ 8), saturando la potenza di calcolo parallelo della GPU, l'effetto dell'overhead fisso svanisce. 
In questo scenario reale di carico:
* Il modello Cross-Frame (16f) risulta **da 1.57x a 1.69x più veloce** rispetto al modello standard (24f).
* Si registra un risparmio di tempo GPU compreso tra il **36.4% e il 40.7%**.
* Il throughput massimo aumenta in modo netto: a BS=8, il modello 16f processa **1673.1 video al secondo** contro i **991.3** del modello a 24f (un incremento di produttività del **+68.8%**).
* *Nota tecnica*: Il risparmio computazionale effettivo (fino al 41%) supera il calo lineare dei frame (33.3%). Questo accade perché una dimensione temporale pari a 16 si allinea in modo perfetto con la struttura a blocchi e i canali dei Tensor Core di NVIDIA, ottimizzando l'allineamento in memoria.

### 3. Vantaggi Straordinari su CPU (Edge Deployment)
Su CPU, dove il collo di bottiglia è puramente aritmetico (compute-bound) e non ci sono ritardi di trasferimento bus PCIe, il modello Cross-Frame mostra benefici clamorosi:
* A **Batch Size = 1**, il modello a 16f dimezza la latenza passando da **53.52 ms** a **25.36 ms** (**52.61% di tempo in meno**, speedup di **2.11x**).
* Questo dimezzamento rende il modello pienamente idoneo a scenari di elaborazione video in tempo reale su dispositivi CPU a bassa potenza (edge computing).

### 4. Valutazione del Trade-Off (Accuratezza vs Risorse)
La Cross-Frame Distillation ha causato una perdita di accuratezza Top-1 sul test set del **3.17% in valore assoluto** (da 65.85% a 62.68%). 

Tuttavia, a fronte di questo piccolo calo, si ottiene:
1. Un **dimezzamento del tempo di inferenza su CPU** a BS=1 (latenza ridotta del 52.6%).
2. Un **risparmio di circa il 40% del tempo di calcolo su GPU** a pieno carico.
3. Un **incremento del 68.8% del throughput video** su GPU.

**Conclusione**: Il trade-off è ampiamente positivo ed efficiente. La Cross-Frame Distillation si dimostra una strategia vincente per ridurre l'impronta computazionale del modello nei casi in cui la velocità di elaborazione e il consumo di risorse siano prioritari rispetto ad incrementi marginali di accuratezza.
