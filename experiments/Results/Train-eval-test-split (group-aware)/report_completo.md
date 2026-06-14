# Report Tecnico: Ottimizzazione della Knowledge Distillation per Action Recognition

Questo documento analizza l'intero ciclo di sperimentazione e debugging affrontato per comprimere una rete 3D ResNet-50 in una MobileNet3D sul dataset UCF-101. L'analisi è suddivisa in tre fasi principali, ciascuna caratterizzata da specifici problemi tecnici, configurazioni SLURM e relativi impatti sulle prestazioni.

---

## 1. Il Disallineamento dei Layer e il "Capacity Gap" (Job 4384 vs 4391)

La prima fase di testing dell'Attention Transfer (AT) ha evidenziato un limite strutturale nel mappare feature tra due architetture profondamente eterogenee.

### Il Problema dei Layer (Il Bug "Key 5")
Inizialmente, l'AT tentava di agganciare i layer `[3, 4, 5]` del Teacher. Il blocco 5, tuttavia, non era un tensore convoluzionale, ma la testa di classificazione (output 2D). Questo generava mappe di attenzione costanti a 1.0, vanificando il segnale e portando al collasso dell'apprendimento. 

Una volta corretto il mapping accoppiando stage semanticamente affini (`[2, 3, 4]` per la ResNet e `[2, 4, 6]` per la MobileNet), il training si è stabilizzato, ma il *capacity gap* spaziale (convoluzioni standard contro depthwise) ha limitato i guadagni.

### Perché Proprio Questi Layer?
La scelta dei layer `[2, 3, 4]` per la ResNet-50 e `[2, 4, 6]` per la MobileNet3D è stata guidata da due vincoli tecnici complementari. Da un lato, questi stage rappresentano livelli di astrazione e risoluzione spaziale comparabili, tipicamente nell'ordine di $28\times28$, $14\times14$ e $7\times7$, quindi riducono il costo e l'errore introdotti dall'interpolazione della loss di Attention Transfer. Dall'altro lato, il mapping corretto evita di agganciare la testa di classificazione finale della ResNet, che nel vecchio schema `[3, 4, 5]` produceva mappe di attenzione degeneri e costanti a 1.0, invalidando di fatto il segnale sul blocco finale.

### Variazioni Successive del Mapping
Nel corso della sperimentazione il mapping non è rimasto fisso, ma è stato raffinato per testare scenari più aggressivi contro il *capacity gap*. In una fase successiva abbiamo provato un approccio *Late-Stage*, limitando il trasferimento all'ultimo stadio utile prima del pooling con `teacher_keys: [4]` e `student_keys: [6]`. In un'altra variante, la configurazione *Temporal-Only AT*, abbiamo mantenuto le chiavi `[2, 3, 4]` e `[2, 4, 6]` ma azzerato la componente spaziale della loss (`\beta_s = 0.0`), concentrando il trasferimento sulle dinamiche temporali. Infine, nello scenario *Born Again* MobileNet $\rightarrow$ MobileNet, le chiavi naturali sono diventate identiche (`[2, 4, 6]` su entrambi i modelli), perché tra architetture omogenee i tensori corrispondenti combaciano senza bisogno di interpolazione.

### Risultati e Configurazioni
| SLURM ID | Esperimento | Configurazione Distillazione | Test Top-1 | Test Top-5 |
| :---: | :--- | :--- | :---: | :---: |
| **4384** | Baseline KD | `mode: distillation`, $T=8$, $\alpha=0.7$ | **58.92%** | **84.93%** |
| **4391** | AT Simmetrico | `mode: distillation_at`, $\beta_s=0.05, \beta_t=0.05$, $keys=[2,3,4]$ | **59.42%** | **85.04%** |
| **4391** | AT Heavy Temp | `mode: distillation_at`, $\beta_s=0.03, \beta_t=0.07$, $keys=[2,3,4]$ | **59.40%** | **84.99%** |

**Considerazioni:** L'inserimento dell'AT corretto ha portato un modesto +0.50% (dal 58.92% al 59.42%). Il trasferimento dell'attenzione spaziale, che richiede interpolazioni aggressive per adattare i volumi $14\times14$ ai $7\times7$, disperde il segnale, agendo più come rumore che come guida utile.

---

## 2. Il Muro del 60%, Bug Silenziosi e l'Efficacia del Warmup (Job 4431 / Exp1)

Durante i tentativi di superare la soglia del 60%, un disallineamento nei parametri ha bloccato temporaneamente i progressi.

### Problemi Affrontati
* **Il Bug Silenzioso "Mode":** Diverse configurazioni AT non producevano variazioni perché il file YAML manteneva l'impostazione `mode: distillation`, bypassando di fatto il codice dell'Attention Transfer e replicando la baseline standard.
* **KeyError sul Fallback:** Una volta forzato `distillation_at`, un fallback predefinito nel codice cercava ancora le chiavi `[3, 4, 5]` non più coerenti con la configurazione corretta `[2, 3, 4]` / `[2, 4, 6]`, causando un crash all'Epoca 6 al termine del warmup iniziale.

Definiti rigidamente i layer e le modalità operative, si sono ottenuti i seguenti numeri:

### Risultati e Configurazioni
| SLURM ID | Esperimento | Configurazione Distillazione | Test Top-1 | Test Top-5 |
| :---: | :--- | :--- | :---: | :---: |
| **4431** | Late-Stage AT | `distillation_at`, $\beta_s=0.05, \beta_t=0.05$, $keys=[4] \rightarrow [6]$ | **61.59%** | **87.52%** |
| **4431** | Temporal-Only AT | `distillation_at`, **$\beta_s=0.0, \beta_t=0.1$**, $keys=[2,3,4] \rightarrow [2,4,6]$ | **62.07%** | **87.05%** |
| **Exp 1** | KD Ottimizzata | `distillation`, **$T=10$, $\alpha=0.9$, `warmup_epochs: 10`** | **62.62%** | **87.02%** |

**Considerazioni:**
1. **Dinamica Temporale Pura (+0.64% rispetto a KD base 61.43%):** L'eliminazione dell'AT spaziale ($\beta_s=0.0$) ha rimosso l'overhead distruttivo dell'interpolazione. L'allineamento naturale sui 24 frame ha permesso alla MobileNet di apprendere ottimamente i pattern temporali.
2. **Warmup come acceleratore decisivo (+1.19%):** Fornire 10 epoche di classificazione pura tramite *ground truth* ha permesso all'architettura inizializzata *from scratch* di consolidare filtri visivi di base, massimizzando poi la ricezione di soft-targets ad alta temperatura ($T=10$).

---

## 3. L'Illusione del "Born Again" e il Knowledge Collapse (Job 4446)

Al fine di azzerare il gap capacitivo spaziale, è stata testata una *Iterative Self-Distillation* (MobileNet $\rightarrow$ MobileNet), partendo da un Teacher con un solido 65% in Test.

### Risultati e Configurazioni
| SLURM ID | Generazione | Setup Distillazione e Teacher | Test Top-1 | Test Top-5 |
| :---: | :---: | :--- | :---: | :---: |
| **4201** | Gen 0 (Teacher) | `distillation` (Run storico lightaug) | **~65.00%** | **-** |
| **4446** | Gen 1 | `distillation`, $T=8$, $\alpha=0.7$, Teacher: Gen 0 | **61.83%** | **86.39%** |
| **4446** | Gen 2 | `distillation`, $T=8$, $\alpha=0.7$, Teacher: Gen 1 | **58.21%** | **84.80%** |
| **4446** | Gen 3 | `distillation`, $T=8$, $\alpha=0.7$, Teacher: Gen 2 | **55.75%** | **82.26%** |

### Problemi Affrontati (Il Collasso Generazionale)
* **Temperatura Eccessiva per Modelli Non-Oracolo:** Il parametro $T=8.0$ era ottimizzato per la *confidence* estrema della ResNet-50. Applicato alle predizioni meno sicure di un Teacher al 65%, ha generato probabilità piatte, equivalenti a rumore bianco. Ogni nuova generazione ha appreso da logit progressivamente più corrotti, causando il calo dal 61.8% al 55.7%.
* **Spreco dell'Omogeneità Architetturale:** L'assenza dell'opzione `distillation_at` ha inibito l'Attention Transfer. In uno scenario MobileNet-to-MobileNet, i tensori spaziali combaciano naturalmente con chiavi identiche `[2, 4, 6]`, senza necessità di interpolazione. Spegnere l'AT ha quindi annullato il principale beneficio teorico del setup *Reborn*.
