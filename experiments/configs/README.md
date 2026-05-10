# Experiment Configurations

Store your `.yaml` or `.json` files here to manage experiment parameters (hyperparameters, paths).

Recommended next runs based on slurm-multiple-runs-4077 analysis:

- distillation_t10_a07_strongaug.yaml
	- Best KD recipe (T=10, alpha=0.7) + stronger spatial augmentation + temporal stride jitter.

- distillation_t10_a07_strongaug_24f.yaml
	- Same as above, but with 24 frames per clip for longer temporal context.

- distillation_at_t10_a07_strongaug.yaml
	- KD + Attention Transfer with the same strong augmentation setup.

Recovery sweep (recommended after strongaug regression):

- distillation_t10_a07_24f_minaug.yaml
	- 24 frames with minimal augmentation (stability-first baseline).

- distillation_t10_a07_24f_lightaug.yaml
	- 24 frames with conservative augmentation.

- distillation_t10_a07_24f_lightaug_lr6e4.yaml
	- Same as lightaug, with slightly higher LR for faster adaptation.
