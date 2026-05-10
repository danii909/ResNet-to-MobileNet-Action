# Guida rapida: passare il checkpoint in evaluation

Questo documento spiega come indicare in modo esplicito quale checkpoint usare durante l'evaluation.

## Regola principale

Lo script di evaluation carica il checkpoint che passi nella variabile `CHECKPOINT`.

Non seleziona automaticamente "l'ultimo run". Se usi un nome fisso che viene sovrascritto, allora sembrera' prendere sempre l'ultimo.

## Comando standard

```bash
CONFIG=<config_model.yaml> CHECKPOINT=<path_checkpoint.pth> sbatch cluster/eval.sh
```

## Esempi pratici

### Teacher

```bash
CONFIG=experiments/configs/teacher.yaml \
CHECKPOINT=experiments/checkpoints/teacher_finetune_best.pth \
sbatch cluster/eval.sh
```

### Student baseline

```bash
CONFIG=experiments/configs/baseline.yaml \
CHECKPOINT=experiments/checkpoints/baseline_best.pth \
sbatch cluster/eval.sh
```

### Student distillation

```bash
CONFIG=experiments/configs/distillation.yaml \
CHECKPOINT=experiments/checkpoints/distillation_best.pth \
sbatch cluster/eval.sh
```

### Checkpoint custom (esperimento specifico)

```bash
CONFIG=experiments/configs/distillation.yaml \
CHECKPOINT=experiments/checkpoints/distillation_expA_epoch60.pth \
sbatch cluster/eval.sh
```

## Cosa succede internamente

1. [cluster/eval.sh](cluster/eval.sh) riceve `CONFIG` e `CHECKPOINT`.
2. Lo script passa `--override evaluation.checkpoint=<CHECKPOINT>` al comando Python.
3. [src/evaluation/evaluate.py](src/evaluation/evaluate.py) legge `evaluation.checkpoint` e carica esattamente quel file.

## Come evitare ambiguita' con piu' run

- Evita di riusare sempre lo stesso nome file per i checkpoint finali.
- Salva checkpoint con nomi distinti per esperimento.
- In evaluation passa sempre il path esplicito nel `CHECKPOINT`.

In questo modo sei sicuro al 100% di valutare il modello che vuoi.
