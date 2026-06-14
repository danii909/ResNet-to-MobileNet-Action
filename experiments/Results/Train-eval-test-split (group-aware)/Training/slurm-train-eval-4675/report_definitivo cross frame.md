# Relazione Finale di Progetto: Knowledge Distillation per Mobile Action Recognition su UCF-101

**Studente:** Daniele Barbagallo  
**Corso:** Laurea Magistrale in Computer Science, Università degli Studi di Catania (UniCT)  

---

## 1. Introduzione e Obiettivi
L'obiettivo di questo progetto è l'ottimizzazione e la compressione di modelli di Deep Learning per l'**Action Recognition** in ambito video (dataset UCF-101). Nello specifico, il lavoro indaga la possibilità di trasferire la conoscenza di una rete pesante e computazionalmente costosa (Teacher) verso una rete leggera (Student) pensata per l'inferenza su dispositivi edge.

* **Teacher:** 3D ResNet-50 (Modello ad alta capacità, computazionalmente oneroso).
* **Student:** MobileNet3D (Architettura ottimizzata per mobile, ~2.48M parametri).

Il progetto esplora diverse tecniche avanzate di compressione: **Knowledge Distillation (KD)** basata sui logit, **Attention Transfer (AT)**, approcci iterativi **Born-Again**, e infine un approccio architetturale asimmetrico definito **Cross-Frame Distillation**.

---

## 2. Fase 1: Ottimizzazione della Distillazione (Modelli a 24 Frame)
La prima fase della sperimentazione ha mantenuto invariata la dimensione temporale per entrambi i modelli ($T=24$ frame).

### 2.1 Standard Knowledge Distillation e Ablation sulla Temperatura
È stato effettuato uno sweep dell'iperparametro *Temperature* ($T$) nella Loss di Kullback-Leibler Divergence.
* **Baseline MobileNet3D (Senza KD):** ~59.5% - 60% Top-1 Accuracy.
* **Teacher ResNet-50:** L'oracolo ha fornito una supervisione di altissima qualità.
* **Risultati KD:** Aumentando la temperatura, lo Student ha mostrato capacità di generalizzazione nettamente superiori. Il picco prestazionale è stato raggiunto con **$T=20$** e $lpha=0.7$, ottenendo un'accuratezza sul Test Set del **65.85%** (Top-1) e dell'87.81% (Top-5). 
* **Analisi t-SNE:** Lo spazio latente del modello a $T=20$ ha rivelato cluster di classi molto più definiti e coesi rispetto alle temperature standard (es. $T=1$), confermando che l'innalzamento della temperatura (che addolcisce le distribuzioni di probabilità) è cruciale su un dataset complesso come UCF-101 per superare le limitazioni di capacità della MobileNet.

### 2.2 Fallimento Architetturale: Attention Transfer e Born-Again
Per tentare di migliorare ulteriormente i risultati, si è intervenuti sulle *feature maps* intermedie:
* **Attention Transfer (AT):** L'allineamento forzato dell'attenzione spaziale e temporale tra ResNet-50 e MobileNet3D ha portato a un degrado delle performance (sotto il 60%). Il fallimento è imputabile al **Capacity Gap**: le convoluzioni *depthwise* della MobileNet faticano a replicare la ricchezza spaziale delle convoluzioni standard della ResNet, portando a interpolazioni distruttive.
* **Born-Again Networks:** L'approccio iterativo (addestrare lo Student dallo Student della generazione precedente) ha innescato un rapido **Knowledge Collapse** dalla terza generazione in poi, con i logit che si degradavano invece di raffinarsi.

---

## 3. Fase 2: Il Trade-off Ottimale (Cross-Frame Distillation)
Constatato che la KD sui logit a $T=20$ fosse il limite superiore di accuratezza a parità di contesto temporale, il focus del progetto si è spostato sull'efficienza di calcolo (*Throughput* e *Latenza*).

È stata implementata una tecnica di **Cross-Frame Distillation**:
* Il Teacher inquadra l'intera scena temporale analizzando **24 frame** in inferenza.
* Lo Student viene addestrato da zero a imitare le predizioni del Teacher ricevendo un tensore sotto-campionato di soli **16 frame** (una riduzione del 33.3% del carico computazionale).

### 3.1 Impatto sull'Accuratezza
Nonostante la perdita di informazione causata dall'ingresso temporale ridotto (che corrisponde a soli ~0.5 secondi a 30fps), lo Student è riuscito a raggiungere un'eccellente accuratezza del **62.68%** (Top-1) e dell'**86.20%** (Top-5). Il calo rispetto al modello a 24f ($T=20$) è stato contenuto in un modesto **-3.17%**, confermando che la soft-loss distillata dal Teacher compensa ampiamente il contesto visivo mancante.

---

## 4. Profilazione Hardware ed Edge Deployment
Il reale successo della tecnica Cross-Frame a 16 frame risiede nei formidabili vantaggi infrastrutturali misurati tramite benchmark fisici. 
Il numero $16$ (potenza di 2) permette un allineamento di memoria perfetto per i Tensor Core e l'architettura dei registri, massimizzando l'efficienza.

### 4.1 Deployment su Edge (Latenza CPU, Compute-Bound)
Su processori standard, in condizioni limitate dal calcolo aritmetico puro (BS=1):
* **Latenza Modello 24f:** ~53.5 ms
* **Latenza Modello 16f:** ~25.4 ms
* **Vantaggio:** Speedup di **2.11x**, con un dimezzamento netto del tempo di risposta (**-52.6%**). Questo rende la MobileNet3D Cross-Frame perfettamente idonea all'analisi video *Real-Time* a circa 40 FPS su dispositivi a bassa potenza (Edge Computing/IoT).

### 4.2 Scalabilità Server (Throughput GPU, Memory-Bound)
Su GPU di fascia enterprise (NVIDIA L40S) a pieno carico (Batch Size $\ge$ 8), dove viene superato il *Kernel Launch Overhead*:
* **Throughput Modello 24f (BS=8):** 991.3 clips/sec
* **Throughput Modello 16f (BS=8):** 1673.1 clips/sec
* **Vantaggio:** Incremento della scalabilità di calcolo parallelo del **+68.8%**, risparmiando quasi il **40%** del tempo di utilizzo hardware a parità di batch analizzato.

---

## 5. Conclusioni
Il progetto dimostra che comprimere un modello 3D non è un processo lineare che si risolve con la sola Knowledge Distillation. Architetture fortemente eterogenee (ResNet vs MobileNet) mal sopportano allineamenti forzati interni (Attention Transfer), traendo invece un beneficio massiccio dall'ottimizzazione a temperatura elevata ($T=20$) sul *decision boundary* finale.

Infine, la manipolazione della dimensione temporale con l'approccio **Cross-Frame Distillation a 16 frame** ha permesso di colpire il *punto di ottimo paretiano*: una riduzione drastica dell'impronta computazionale e dei tempi di latenza a fronte di una perdita marginale e controllata di accuratezza. Il risultato è un sistema Action Recognition pronto per il deployment su risorse di bordo senza compromessi di fluidità.
