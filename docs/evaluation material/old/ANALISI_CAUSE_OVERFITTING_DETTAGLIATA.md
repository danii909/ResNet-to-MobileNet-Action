# Analisi Dettagliata e Prolissa delle Cause Più Plausibili di Overfitting

## Obiettivo del documento

Questo documento riscrive in forma estesa, ragionata e realistica l'analisi del fenomeno di overfitting osservato nelle run del progetto. L'intento non è solo elencare sintomi, ma spiegare perché i sintomi sono coerenti con il comportamento del sistema completo: dataset, preprocessing, architetture, protocollo di valutazione, training regime e strategie di distillazione.

La lettura va quindi intesa come una diagnosi probabilistica. In altre parole, non si afferma una singola causa assoluta, ma si costruisce una gerarchia di cause plausibili basata sui dati disponibili.

## Contesto e perimetro

### Scope dell'analisi

L'analisi copre le run principali riportate nei risultati, con particolare attenzione ai gruppi in [results/slurm-train-seq-3998](../results/slurm-train-seq-3998), [results/slurm-train-eval-4041](../results/slurm-train-eval-4041), [results/slurm-multiple-runs-4077](../results/slurm-multiple-runs-4077), [results/slurm-recovery-24f-sweep-4137](../results/slurm-recovery-24f-sweep-4137), [results/slurm-24f-refine3-a-4141](../results/slurm-24f-refine3-a-4141) e [results/slurm-24f-refine3-b-4142](../results/slurm-24f-refine3-b-4142).

Le osservazioni comparative sono allineate anche con la sintesi in [docs/EXPERIMENTS_COMPARISON.md](EXPERIMENTS_COMPARISON.md).

### Vincolo richiesto

La run seed43 con eval acc circa 71 è stata volutamente esclusa dalla diagnosi comparativa principale, perché rappresenta uno studio di sensibilità al seed e non un confronto iperparametrico equo con seed costante.

## Quadro empirico: quale pattern emerge davvero

Il pattern osservato non è un caso isolato. Nelle run student, soprattutto nelle configurazioni meno robuste, la train accuracy cresce moltissimo mentre la test accuracy rimane sensibilmente più bassa.

In termini semplici, il modello impara fortemente il training set ma trasferisce solo una parte di quell'apprendimento fuori campione.

La differenza può essere descritta con il gap di generalizzazione:

$$
\Delta = \text{train acc} - \text{test acc}
$$

Nelle run peggiori il gap è enorme. Nelle run migliori 24 frame light augmentation il gap si riduce, ma non sparisce. Questo è un punto centrale: il problema è strutturale, non solo episodico.

## Spiegazione causale: gerarchia delle cause più plausibili

Di seguito, le cause ordinate per probabilità e impatto con razionale tecnico.

### 1. Protocollo di selezione del checkpoint sul test set

Livello di plausibilità: alto.

Nel trainer, il modello migliore viene selezionato con la metrica calcolata sul loader chiamato test. Questo comportamento è visibile nel flusso di training e valutazione in [src/training/trainer.py](../src/training/trainer.py) e nell'entrypoint in [src/training/train.py](../src/training/train.py), dove il trainer riceve train e test senza una validation separata.

Perché è importante:

- Non è il classico leakage train-test diretto sui campioni, ma è comunque una forma di ottimizzazione indiretta sul test.
- Dopo molte prove, sweep e confronti, il test smette di essere una stima veramente neutra.
- Il rischio è duplice: sovrastima delle performance reali e instabilità delle conclusioni quando si cambia seed o dettagli minori.

Perché non spiega tutto da solo:

- Anche con questo bias, i gap train-test molto ampi in diverse run indicano comunque overfitting reale.
- Quindi questa causa amplifica e confonde la misura, ma non è l'unica origine del fenomeno.

### 2. Student addestrato da zero in regime video ad alta complessità

Livello di plausibilità: alto.

Lo student in [src/models/student.py](../src/models/student.py) è un MobileNet3D addestrato da zero. Al contrario, il teacher in [src/models/teacher.py](../src/models/teacher.py) parte da pretraining su Kinetics (quando i pesi sono presenti), poi viene fine-tuned.

Implicazione pratica:

- Addestrare da zero un modello video, anche leggero, su UCF101 espone facilmente a memorizzazione del train se la regolarizzazione non è perfettamente bilanciata.
- Sessanta epoche con ottimizzazione efficace possono portare rapidamente la train accuracy molto in alto.
- Se la variabilità intrinseca del dataset non è sufficiente a vincolare l'apprendimento verso pattern davvero generali, il test resta più basso.

Perché è coerente coi numeri:

- Le run baseline o KD deboli mostrano train molto elevata e test distante.
- Le run KD migliori migliorano il test, ma spesso con train ancora molto alta, segnale che la capacità di memorizzazione non è stata eliminata.

### 3. Mismatch tra distribuzione di training e protocollo di test a clip singola

Livello di plausibilità: medio-alto.

Nel dataset loader [src/datasets/ucf101.py](../src/datasets/ucf101.py), durante il training ci sono sampling temporale stocastico e trasformazioni spaziali casuali; nel test il campionamento è più deterministico e la crop è centrale.

Questo non è necessariamente sbagliato, ma crea una dinamica nota:

- Il modello apprende una famiglia ampia di viste train-time.
- In valutazione, una sola vista/clip può non catturare bene il contenuto discriminante del video.
- La metrica test può oscillare di più e sottostimare la robustezza media su più viste, oppure evidenziare fragilità reali di rappresentazione.

In entrambi i casi, il gap train-test può rimanere alto anche quando il training procede correttamente.

### 4. Distillazione utile ma non sufficiente a cancellare l'overfitting

Livello di plausibilità: medio-alto.

La loss di distillazione in [src/training/losses.py](../src/training/losses.py) combina componente soft (KL con temperatura) e componente hard (cross-entropy su label vere).

In teoria questo regolarizza perché il teacher fornisce informazione più ricca delle sole hard labels. In pratica:

- La KD migliora la generalizzazione rispetto ad alcuni baseline.
- Però non trasforma automaticamente il problema in uno scenario senza overfitting.
- Se il modello studente resta molto capace rispetto al segnale utile disponibile, continua a poter memorizzare una parte importante del train.

Quindi la KD è una mitigazione parziale, non una cura totale.

### 5. Finestra stretta di augmentation: troppo poco memorizza, troppo forte degrada

Livello di plausibilità: medio.

Dalle config emerge che strategie strong augmentation possono peggiorare sensibilmente i risultati. Le configurazioni come [experiments/configs/distillation_t10_a07_strongaug.yaml](../experiments/configs/distillation_t10_a07_strongaug.yaml) e [experiments/configs/distillation_at_t10_a07_strongaug.yaml](../experiments/configs/distillation_at_t10_a07_strongaug.yaml) aumentano aggressività spaziale/temporale e includono stride temporale massimo più alto.

Interpretazione realistica:

- Se l'augmentation è troppo leggera, il modello può imparare scorciatoie e sovra-adattarsi al train.
- Se è troppo forte o distruttiva, si perde informazione semantica/temporale utile e cala anche la generalizzazione.
- Il sistema quindi richiede una calibrazione fine, non una semplice regola più augmentation uguale meno overfitting.

### 6. Sensibilità al seed e varianza stocastica non trascurabile

Livello di plausibilità: medio.

Anche senza includere la run seed43 nel confronto principale, l'intero storico suggerisce che il seed può spostare in modo visibile la metrica finale.

Effetto sulla diagnosi:

- Una singola run può sovra-rappresentare o sotto-rappresentare la qualità reale della configurazione.
- Serve ragionare su medie multi-seed o almeno conferme su 2-3 seed nei candidati migliori.
- Parte del pattern osservato è strutturale, ma la sua ampiezza precisa è influenzata dal rumore stocastico.

## Perché il teacher sembra più stabile dello student

Il teacher fine-tuned tende a mostrare dinamiche più sane di generalizzazione. Questo non sorprende:

- parte da un pretraining forte su dominio video;
- ha rappresentazioni già strutturate;
- richiede meno sforzo per trovare un minimo che generalizza su UCF101.

Lo student da zero deve invece costruire la rappresentazione quasi interamente dai dati target e, in questo contesto, è più incline a convergere verso soluzioni ad alta performance sul train ma più fragili fuori campione.

## Cosa è probabile e cosa è meno probabile

### Probabile

- overfitting reale e ricorrente nello student regime;
- bias di protocollo dovuto all'uso del test per selezione checkpoint;
- sensibilità elevata a scelte di sampling/augmentation.

### Meno probabile

- bug grossolano nelle label o nel mapping classi (i trend sono troppo coerenti tra molte run);
- errore singolo che spiega tutto il fenomeno.

## Onestà metodologica: cosa si può migliorare davvero

Risposta sincera: l'overfitting non si elimina completamente in questo scenario, ma si può ridurre in modo sostanziale.

Le leve più realistiche, in ordine pratico:

1. introdurre split train/val/test separati e usare val per selezionare il best checkpoint;
2. mantenere sampling temporale conservativo (evitare stride aggressivi);
3. usare augmentation moderata e coerente con il dominio;
4. validare le migliori config su più seed;
5. considerare pretraining o inizializzazione più informata anche per lo student.

Aspettativa realistica:

- il gap train-test può calare, ma non andare a zero;
- la stabilità tra run può migliorare;
- il ranking tra configurazioni diventa più affidabile.

## Conclusione finale

Il pattern di overfitting è reale, diffuso e tecnicamente coerente con il setup. Non è identico in tutte le run, ma segue una logica: quando il regime è meglio calibrato (ad esempio 24f light augmentation) il gap si riduce; quando il regime è sbilanciato (config deboli o augmentazioni troppo aggressive) il problema cresce o cambia natura.

La causa non è una sola. È l'effetto combinato di protocollo di valutazione, capacità e inizializzazione dello student, dinamica di training video e delicatezza del preprocessing temporale-spaziale.

Questa diagnosi è quindi severa ma costruttiva: il progetto è recuperabile e migliorabile, ma richiede disciplina sperimentale e protocolli più puliti per trasformare i guadagni osservati in risultati veramente affidabili.