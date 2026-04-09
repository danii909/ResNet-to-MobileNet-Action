# TODO prossime run (focus obiettivi minimi)

Questo piano mantiene il focus solo sugli obiettivi minimi della consegna:
- confronto Teacher vs Baseline Student vs Distilled Student
- metriche deployment (size MB e inference ms)
- nessun obiettivo extra (niente Attention Transfer, niente t-SNE)

## 0) Vincolo fondamentale: riuso del teacher gia validato

Teacher di riferimento del job:
- Job root: /home/brbdnl01e03e017o/dl26-projects/experiments/logs/slurm-train-eval-4041
- Checkpoint da riusare: /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth

Perche questo controllo e importante:
- Evita di distillare da un teacher diverso per errore.
- Mantiene confrontabili i risultati con la pipeline gia chiusa (teacher top1=88.21).
- Evita il costo di una nuova run teacher (non necessaria).

Checklist di verifica prima del lancio:
- [ ] esiste /home/brbdnl01e03e017o/dl26-projects/experiments/logs/slurm-train-eval-4041/pipeline_summary.txt
- [ ] nel summary il blocco [teacher] ha train_status=SUCCESS ed eval_status=SUCCESS
- [ ] esiste /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth
- [ ] tutte le run distillation passano esplicitamente distillation.teacher_checkpoint con quel path

## 1) Run pianificate (senza teacher)

Le run sono ordinate per massimizzare il rapporto costo/beneficio. Ogni run salva checkpoint in una cartella separata per evitare overwrite.

### Run A - baseline_wd002_ls01

Obiettivo:
- Ridurre overfitting del baseline aumentando la regolarizzazione.

Cosa cambia:
- config base: experiments/configs/baseline.yaml
- training.weight_decay=0.02
- training.checkpoint_dir=experiments/checkpoints/next_runs_4041/baseline_wd002_ls01
- training.run_log_dir=experiments/logs/next_runs_4041/baseline_wd002_ls01/train
- logging.run_name=baseline-wd002-ls01

Cosa ci aspettiamo:
- train accuracy piu bassa del baseline precedente, test piu stabile.
- miglioramento piccolo ma utile del lower bound student.

### Run B - kd_t6_a06

Obiettivo:
- Distillation piu bilanciata (piu peso ai target hard rispetto alla run alpha=0.7).

Cosa cambia:
- config base: experiments/configs/distillation.yaml
- distillation.teacher_checkpoint=/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth
- distillation.temperature=6.0
- distillation.alpha=0.6
- training.lr=0.0006
- training.weight_decay=0.01
- training.kd_warmup_epochs=5
- training.checkpoint_dir=experiments/checkpoints/next_runs_4041/kd_t6_a06
- training.run_log_dir=experiments/logs/next_runs_4041/kd_t6_a06/train
- logging.run_name=kd-t6-a06

Cosa ci aspettiamo:
- miglior compromesso tra stabilita e trasferimento soft targets.

### Run C - kd_t10_a07

Obiettivo:
- Testare logits piu smussati mantenendo alpha originale.

Cosa cambia:
- config base: experiments/configs/distillation.yaml
- distillation.teacher_checkpoint=/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth
- distillation.temperature=10.0
- distillation.alpha=0.7
- training.lr=0.0006
- training.weight_decay=0.01
- training.kd_warmup_epochs=5
- training.checkpoint_dir=experiments/checkpoints/next_runs_4041/kd_t10_a07
- training.run_log_dir=experiments/logs/next_runs_4041/kd_t10_a07/train
- logging.run_name=kd-t10-a07

Cosa ci aspettiamo:
- possibile incremento top1/top5 se la distribuzione del teacher risulta piu informativa per lo student.

### Run D - kd_t6_a06_lr5e4_wd002

Obiettivo:
- Stabilizzare ulteriormente la distillation per ridurre overfitting finale.

Cosa cambia:
- config base: experiments/configs/distillation.yaml
- distillation.teacher_checkpoint=/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth
- distillation.temperature=6.0
- distillation.alpha=0.6
- training.lr=0.0005
- training.weight_decay=0.02
- training.kd_warmup_epochs=5
- training.checkpoint_dir=experiments/checkpoints/next_runs_4041/kd_t6_a06_lr5e4_wd002
- training.run_log_dir=experiments/logs/next_runs_4041/kd_t6_a06_lr5e4_wd002/train
- logging.run_name=kd-t6-a06-lr5e4-wd002

Cosa ci aspettiamo:
- training meno aggressivo, curva test piu regolare.

## 2) Valutazione per ogni run

Per ogni training, eseguire evaluation dedicata con:
- EVAL_LOG_DIR dedicata
- checkpoint best della run

Checkpoint usati in evaluation:
- Run A: experiments/checkpoints/next_runs_4041/baseline_wd002_ls01/baseline_best.pth
- Run B: experiments/checkpoints/next_runs_4041/kd_t6_a06/distillation_best.pth
- Run C: experiments/checkpoints/next_runs_4041/kd_t10_a07/distillation_best.pth
- Run D: experiments/checkpoints/next_runs_4041/kd_t6_a06_lr5e4_wd002/distillation_best.pth

## 3) Criterio di scelta finale (semplice)

Confrontare tutte le run con i baseline gia ottenuti nel job 4041:
- baseline precedente: top1 62.54
- distillation precedente: top1 65.11

Scegliere come best model la run con:
- top1 massimo in evaluation
- a parita (<=0.2 top1), preferire quella con inferenza media piu bassa

## 4) Script di lancio (vincolo: un solo job)

Usare un solo submit:
- sbatch cluster/submit_multiple_runs.sh

Lo script:
- verifica prima il teacher del job 4041
- esegue tutte le run in sequenza nello stesso job SLURM (train poi eval per ogni run)
- usa parametri espliciti per ogni run
- salva summary unico in experiments/logs/slurm-multiple-runs-<JOBID>/pipeline_summary.txt
