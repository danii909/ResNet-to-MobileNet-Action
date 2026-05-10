# Organizzazione dei risultati

Questa cartella raccoglie i log e gli output sperimentali prodotti durante le prove sul dataset UCF101. I risultati non sono tutti equivalenti: appartengono a tre protocolli diversi di split, con livelli diversi di affidabilita delle metriche.

## Struttura generale

Ogni cartella `slurm-...` corrisponde a un job lanciato in cluster. Dentro ogni job si trovano una o piu configurazioni sperimentali, ad esempio `baseline/`, `teacher/`, `distillation/` oppure varianti piu specifiche come `baseline_ls005_24f_lightaug/` o `kd_t10_a07_24f_lightaug/`.

In ogni esperimento la struttura tipica e:

- `train/`: log e metriche della fase di training.
- `eval/`: output della valutazione.
- `pipeline_summary.txt`: riepilogo sintetico del job e delle fasi eseguite.

In alcuni casi compare anche una fase di `test` separata, ma nella pratica la distinzione importante qui e capire su quale split viene fatta la valutazione durante il training.

## Come e organizzato UCF101

UCF101 e un dataset di action recognition composto da video organizzati per classe. Nella struttura classica ci sono cartelle per azione, ad esempio `ApplyEyeMakeup/`, `Archery/`, `BasketballDunk/` e cosi via, e dentro ogni cartella ci sono i video `.avi`.

Un dettaglio cruciale e che i video della stessa classe possono appartenere allo stesso gruppo di origine: UCF101 raccoglie infatti video provenienti da clip piu lunghe e correlate tra loro. La documentazione ufficiale del dataset raccomanda di tenere separati i video dello stesso gruppo tra train e test, per evitare leakage e metriche artificialmente alte.

Nel progetto questo punto e importante perche il modello non lavora su un singolo video intero per volta, ma su clip campionate nel tempo. Se si spezza lo split solo per frame o per clip senza vincolare il gruppo video, clip dello stesso video originale possono finire in train e in evaluation, rendendo la valutazione troppo facile.

## Famiglie di esperimenti

### 1. `train-test-split/`

Questa e la famiglia piu vecchia e contiene esperimenti condotti con lo split stock di UCF101 fornito da Hugging Face, cioe train e test ufficiali.

Caratteristica chiave:

- durante il training la fase di evaluation veniva fatta sul test set ufficiale;
- in alcuni esperimenti veniva eseguita anche una evaluation finale post-training sullo stesso test set.

Conseguenza:

- le metriche di training possono risultare ottimistiche, perche il test set viene usato anche come riferimento per scegliere il checkpoint migliore;
- questi risultati sono utili come storico sperimentale, ma non sono i piu adatti per una stima rigorosa della generalizzazione.

Questa cartella include i job piu importanti della prima fase del progetto, tra cui:

- `slurm-train-seq-3998`
- `slurm-train-eval-4041`
- `slurm-multiple-runs-4077`
- `slurm-strongaug-runs-4129`
- `slurm-recovery-24f-sweep-4137`
- `slurm-24f-refine3-a-4141`
- `slurm-24f-refine3-b-4142`
- `slurm-baseline-24f-phase2-4147`
- `slurm-baseline-24f-phase3-4162`

### 2. `train-eval-test-split (no group-aware)/`

Questa cartella nasce per correggere il problema precedente introducendo un terzo split interno, ricavato dal training set, da usare come evaluation durante il training. In questo modo il test set ufficiale dovrebbe restare riservato solo alla valutazione finale post-training.

Il limite di questa prima correzione e che lo split interno era fatto per label, ma non per gruppo di video. In pratica si partiva dalle clip o dai sample della stessa classe, senza bloccare in modo esplicito i video di origine.

Per UCF101 questo e un problema serio: se due clip provengono dallo stesso video sorgente, una puo finire in train e l'altra in eval. Il modello allora riconosce pattern quasi duplicati invece di generalizzare davvero. Il risultato tipico e un training molto alto, ma troppo bello per essere vero.

Questa cartella contiene solo gli esperimenti piu promettenti della cartella precedente, ma va letta con cautela per il rischio di leakage. I run principali qui sono:

- `TEACHER/slurm-train-4168`
- `slurm-base-kd-t8-v3-4184`

### 3. `train-eval-test-split (group-aware)/`

Questa e la versione corretta dell'approccio precedente. Mantiene tre set separati, ma fa lo split interno in modo group-aware: train ed eval vengono separati sia per label sia per video di origine, cosi nessun video contribuisce a entrambe le parti.

Questo e il protocollo da considerare piu affidabile nella cartella dei risultati, perche riduce in modo sostanziale il leakage e rende piu interpretabile la differenza tra validation interna e test ufficiale.

Qui si trovano solo gli esperimenti migliori selezionati dalla fase precedente, con il protocollo corretto. Il run di riferimento e:

- `slurm-train-eval-4201`

## Come leggere i numeri

Quando confronti i risultati, non mescolare metriche provenienti da protocolli diversi. In particolare:

- non confrontare direttamente i migliori valori di training dei run storici con le metriche finali dei run group-aware;
- usa il test set solo per la stima finale delle performance;
- usa l'evaluation interna solo come supporto alla selezione del checkpoint, ma solo se lo split e group-aware.

In sintesi:

- `train-test-split/` = storico sperimentale sullo split ufficiale, ma con evaluation sul test durante il training;
- `train-eval-test-split (no group-aware)/` = tentativo di correzione, ma ancora vulnerabile al leakage tra clip dello stesso video;
- `train-eval-test-split (group-aware)/` = soluzione corretta, da preferire per interpretare i risultati in modo affidabile.
