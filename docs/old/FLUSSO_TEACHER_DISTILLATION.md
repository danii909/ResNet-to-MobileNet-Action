# Flusso Teacher -> Distillation

Questo documento spiega come funziona il passaggio tra il training del teacher e la distillation dello student.

## Idea generale

Il progetto segue questo flusso:

1. Si allena il teacher su UCF-101 con fine-tuning.
2. Al termine, il training salva il miglior checkpoint del teacher su disco.
3. La distillation carica quel checkpoint e usa il teacher come modello guida per lo student.

Quindi lo student non "impara da zero" il teacher in memoria: legge un checkpoint già salvato in precedenza.

## Quale teacher viene usato

La distillation usa il path indicato nel config YAML.

Nel file [experiments/configs/distillation.yaml](experiments/configs/distillation.yaml) il parametro chiave è:

```yaml
distillation:
  teacher_checkpoint: experiments/checkpoints/teacher_finetune_best.pth
```

Questo significa che il teacher usato dalla distillation e' il checkpoint finetunato salvato in:

- [experiments/checkpoints/teacher_finetune_best.pth](experiments/checkpoints/teacher_finetune_best.pth)

Non viene usato direttamente il teacher "base" pretrainato su Kinetics-400, se non come punto di partenza iniziale del fine-tuning.

## Da dove arrivano i pesi

Il teacher viene costruito in [src/models/teacher.py](src/models/teacher.py).

Il comportamento e' questo:

- prima viene creato `slow_r50`
- se `pretrained: true`, il modello prova a caricare i pesi Kinetics-400 da cache locale o da file locale
- se nel costruttore viene passato `checkpoint_path`, quel checkpoint viene caricato sopra il modello

Per la distillation, il `checkpoint_path` viene preso dal config e punta al teacher finetunato.

## Cosa succede se ci sono piu' run di teacher

Qui conta il path del checkpoint, non il concetto astratto di "ultimo teacher".

Se piu' run di teacher salvano sullo stesso file:

- [experiments/checkpoints/teacher_finetune_best.pth](experiments/checkpoints/teacher_finetune_best.pth)

allora l'ultimo run che ha scritto quel file puo' sovrascrivere il precedente.

Quindi, di fatto, la distillation usera' sempre il checkpoint presente in quel path al momento del lancio.

Se vuoi distinguere piu' esperimenti, devi salvare i teacher in path diversi e poi cambiare `distillation.teacher_checkpoint`.

## Distillation in run separata

Puoi lanciare lo student in un momento diverso dal teacher. Non serve che i due training siano nello stesso job o nella stessa esecuzione.

L'unica condizione e': il file del teacher deve esistere prima di avviare la distillation.

Esempio:

1. lanci il teacher finetune
2. si crea [experiments/checkpoints/teacher_finetune_best.pth](experiments/checkpoints/teacher_finetune_best.pth)
3. lanci la distillation dello student
4. la distillation carica quel file come teacher

## Sequenza consigliata

Per il progetto, la sequenza piu' chiara e' questa:

```bash
sbatch cluster/train_sequential.sh
```

oppure, se usi gli alias:

```bash
source cluster/aliases.sh
train-seq teacher.yaml baseline.yaml distillation.yaml
```

In entrambi i casi il teacher viene prima finetunato, poi il suo best checkpoint viene riusato dalla distillation.

## In sintesi

- La distillation usa il teacher finetunato, non quello base.
- Il teacher arriva dal checkpoint indicato nel config.
- Se fai piu' run, il file checkpoint determina quale teacher viene usato.
- Se vuoi separare i run, cambia il path del checkpoint o usa override da CLI.
