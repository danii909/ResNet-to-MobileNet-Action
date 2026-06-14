# Report

## (AT) Disallineamento dei Layer e Capacity Gap (Job 4384 vs 4391)

### Key 5 bug
Inizialmente, l'AT tentava di agganciare i layer `[3, 4, 5]` del Teacher. Il blocco 5, tuttavia, non era un tensore convoluzionale, ma la testa di classificazione (output 2D). Questo generava mappe di attenzione costanti a 1.0, vanificando il segnale e portando al collasso dell'apprendimento. 

Una volta corretto il mapping accoppiando stage semanticamente affini (`[2, 3, 4]` per la ResNet e `[2, 4, 6]` per la MobileNet, risoluzione spaziale comparabili, tipicamente nell'ordine di $28\times28$, $14\times14$ e $7\times7$), il training si è stabilizzato, ma il capacity gap spaziale (convoluzioni standard contro depthwise) ha limitato i guadagni.

### Risultati AT standard
| SLURM ID | Esperimento | Configurazione Distillazione | Test Top-1 | Test Top-5 |
| :---: | :--- | :--- | :---: | :---: |
| **4384** | Baseline KD | `mode: distillation`, $T=8$, $\alpha=0.7$ | **58.92%** | **84.93%** |
| **4391** | AT Simmetrico | `mode: distillation_at`, $\beta_s=0.05, \beta_t=0.05$, $keys=[2,3,4]$ | **59.42%** | **85.04%** |
| **4391** | AT Heavy Temp | `mode: distillation_at`, $\beta_s=0.03, \beta_t=0.07$, $keys=[2,3,4]$ | **59.40%** | **84.99%** |

---

### Risultati ATv2 (Job 4431 / Exp1)
| SLURM ID | Esperimento | Configurazione Distillazione | Test Top-1 | Test Top-5 |
| :---: | :--- | :--- | :---: | :---: |
| **4431** | Late-Stage AT | `distillation_at`, $\beta_s=0.05, \beta_t=0.05$, $keys=[4] \rightarrow [6]$ | **61.59%** | **87.52%** |
| **4431** | Temporal-Only AT | `distillation_at`, **$\beta_s=0.0, \beta_t=0.1$**, $keys=[2,3,4] \rightarrow [2,4,6]$ | **62.07%** | **87.05%** |
| **Exp 1** | KD Ottimizzata | `distillation`, **$T=10$, $\alpha=0.9$, `warmup_epochs: 10`** | **62.62%** | **87.02%** |

Approccio Late-Stage nel quale si limita il trasferimento all'ultimo stadio utile prima del pooling con `teacher_keys: [4]` e `student_keys: [6]`. 
Approccio *Temporal-Only AT*, nel quale vengono mantenute le chiavi `[2, 3, 4]` e `[2, 4, 6]` ma azzerata la componente spaziale della loss (`\beta_s = 0.0`), concentrando il trasferimento sulle dinamiche temporali. 

**Considerazioni:**
2. **Warmup come acceleratore decisivo (+1.19%):** Fornire 10 epoche di classificazione pura tramite *ground truth* ha permesso all'architettura inizializzata *from scratch* di consolidare filtri visivi di base, massimizzando poi la ricezione di soft-targets ad alta temperatura ($T=10$).

---

## Born Again e Knowledge Collapse (Job 4446)
3 student. Punto di partenza: distillation pura con 65% in Test.

### Risultati e Configurazioni
| SLURM ID | Generazione | Setup Distillazione e Teacher | Test Top-1 | Test Top-5 |
| :---: | :---: | :--- | :---: | :---: |
| **4201** | Gen 0 (Teacher) | `distillation` $T=8$, $\alpha=0.7$, Teacher: Resnet50 | **~65.00%** | **~90%** |
| **4446** | Gen 1 | `distillation`, $T=8$, $\alpha=0.7$, Teacher: Gen 0 | **61.83%** | **86.39%** |
| **4446** | Gen 2 | `distillation`, $T=8$, $\alpha=0.7$, Teacher: Gen 1 | **58.21%** | **84.80%** |
| **4446** | Gen 3 | `distillation`, $T=8$, $\alpha=0.7$, Teacher: Gen 2 | **55.75%** | **82.26%** |

### Problemi Affrontati (Il Collasso Generazionale)
* **Temperatura Eccessiva per Modelli Non-Oracolo:** Il parametro $T=8.0$ era ottimizzato per la *confidence* estrema della ResNet-50. Applicato alle predizioni meno sicure di un Teacher al 65%, ha generato probabilità piatte, equivalenti a rumore bianco. Ogni nuova generazione ha appreso da logit progressivamente più corrotti, causando il calo dal 61.8% al 55.7%.
* **Spreco dell'Omogeneità Architetturale:** L'assenza dell'opzione `distillation_at` ha inibito l'Attention Transfer. In uno scenario MobileNet-to-MobileNet, i tensori spaziali combaciano naturalmente con chiavi identiche `[2, 4, 6]`, senza necessità di interpolazione. Spegnere l'AT ha quindi annullato il principale beneficio teorico del setup *Reborn*.
