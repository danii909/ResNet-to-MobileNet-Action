# Attention Transfer for 3D Action Recognition: Implementation Details

## Abstract

This document describes the design, implementation, and debugging of the Attention Transfer (AT) module used in our Knowledge Distillation pipeline for mobile action recognition. The AT mechanism transfers intermediate feature representations from a 3D ResNet-50 teacher to a MobileNet3D student on the UCF-101 dataset. We detail the theoretical foundations, the critical bug discovered in the original implementation (incorrect feature block indexing and improper spatial interpolation), and the corrected approach that decomposes attention into separate spatial and temporal components.

---

## 1. Background

### 1.1 Attention Transfer in Knowledge Distillation

Attention Transfer (Zagoruyko & Komodakis, "Paying More Attention to Attention", ICLR 2017) extends standard logit-based Knowledge Distillation by transferring information from the teacher's intermediate feature maps to the student's. The core idea is that the spatial distribution of activations — where the network "pays attention" — carries discriminative information that the student can learn from, beyond what the soft logits alone convey.

For 2D image classification, given a feature tensor $F \in \mathbb{R}^{B \times C \times H \times W}$, the spatial attention map is defined as:

$$A(F) = \text{normalize}\left(\sum_{c=1}^{C} |F_c|^2\right) \in \mathbb{R}^{B \times H \times W}$$

The AT loss is then:

$$\mathcal{L}_{\text{AT}} = \beta \sum_{i} \text{MSE}\left(A(F^T_i),\ A(F^S_i)\right)$$

where $F^T_i$ and $F^S_i$ are paired intermediate feature maps from teacher and student, and $\beta$ is a weighting coefficient.

### 1.2 Extension to 3D Video Features

In video understanding, intermediate features are 5-dimensional tensors:

$$F \in \mathbb{R}^{B \times C \times T \times H \times W}$$

where $T$ is the temporal dimension (number of frames retained at that layer). A naive extension of 2D AT to 3D is to sum over channels and flatten all remaining dimensions into a single vector. This is what the original implementation in our codebase did. As we show below, this approach is both theoretically unsound and practically buggy.

---

## 2. Architecture Overview

### 2.1 Teacher: 3D ResNet-50 (slow_r50)

The teacher model is PyTorchVideo's `slow_r50`, pretrained on Kinetics-400 and fine-tuned on UCF-101. The model consists of 6 sequential blocks (indices 0–5):

| Block Index | Type | Output Shape (approx.) | Description |
|:-----------:|------|:----------------------:|-------------|
| 0 | `ResStage` | `[B, 64, T, 56, 56]` | Initial convolution + pool |
| 1 | `ResStage` | `[B, 64, T, 56, 56]` | First residual stage |
| 2 | `ResStage` | `[B, 256, T, 28, 28]` | Second residual stage |
| 3 | `ResStage` | `[B, 512, T, 14, 14]` | Third residual stage |
| 4 | `ResStage` | `[B, 1024, T, 7, 7]` | Fourth residual stage |
| **5** | **`ResNetBasicHead`** | **`[B, num_classes]`** | **Global avg pool → dropout → linear projection** |

> **Critical observation:** Block 5 is **not** a feature extraction stage. It is the classification head that performs global average pooling over all spatial and temporal dimensions, followed by a linear projection to `num_classes` dimensions. Its output is a **2D tensor** `[B, 101]`, not a 5D feature map.

### 2.2 Student: MobileNet3D

The student is a MobileNetV2 architecture adapted to 3D convolutions. It processes input through a stem convolution followed by 7 inverted residual stages (indices 0–6):

| Stage Index | Output Channels | Stride | Output Shape (approx.) |
|:-----------:|:---------------:|:------:|:----------------------:|
| 0 | 16 | 1 | `[B, 16, T, 56, 56]` |
| 1 | 24 | 2 | `[B, 24, T, 28, 28]` |
| **2** | **32** | **2** | **`[B, 32, T, 14, 14]`** |
| 3 | 64 | 2 | `[B, 64, T, 7, 7]` |
| **4** | **96** | **1** | **`[B, 96, T, 7, 7]`** |
| 5 | 160 | 2 | `[B, 160, T, 4, 4]` |
| **6** | **320** | **1** | **`[B, 320, T, 4, 4]`** |

Both models preserve the temporal dimension $T$ throughout all feature extraction stages (temporal stride is 1 everywhere), meaning $T$ is identical between teacher and student at every paired layer.

### 2.3 Feature Pairing for AT

The correct pairing matches layers at similar semantic levels and roughly compatible spatial resolutions:

| Pair | Teacher Block | Teacher Shape | Student Stage | Student Shape |
|:----:|:------------:|:-------------:|:-------------:|:-------------:|
| 1 | Block **2** | `[B, 256, T, 28, 28]` | Stage **2** | `[B, 32, T, 14, 14]` |
| 2 | Block **3** | `[B, 512, T, 14, 14]` | Stage **4** | `[B, 96, T, 7, 7]` |
| 3 | Block **4** | `[B, 1024, T, 7, 7]` | Stage **6** | `[B, 320, T, 4, 4]` |

---

## 3. The Bug: Incorrect Feature Block Indexing

### 3.1 What was wrong

The original implementation in `teacher.py` defined:

```python
FEATURE_BLOCKS = [3, 4, 5]  # WRONG
```

This registered forward hooks on blocks 3, 4, and **5**. Since block 5 is the `ResNetBasicHead` (classification head), the hook captured its output: a **2D tensor** of shape `[B, num_classes]`.

### 3.2 Why it was silent

The original spatial attention function did not validate input dimensionality:

```python
def _spatial_attention_map(features):
    attn = (features ** 2).sum(dim=1)   # On 2D: [B, 101] → [B] (scalar per sample!)
    attn = attn.view(attn.size(0), -1)  # [B] → [B, 1]
    attn = F.normalize(attn, p=2, dim=1)  # normalize a single value → always 1.0
    return attn
```

When applied to the 2D head output `[B, 101]`:
- `.sum(dim=1)` summed across the 101 class logits, producing a **scalar per sample** `[B]`
- `.view(B, -1)` reshaped to `[B, 1]`
- `F.normalize(..., p=2, dim=1)` on a single element always returns `1.0`

The resulting "attention maps" for the third pair were **always constant** (all ones). The MSE between two all-ones vectors is zero, contributing nothing to the loss but producing no error. This meant:

1. **One third of the AT signal was wasted** — the highest-level feature pair contributed zero gradient.
2. **No runtime error was raised** — the bug was completely silent.
3. **No test caught it** — without dimensionality assertions, the code appeared to function.

### 3.3 The interpolation problem

Even for the two valid feature pairs (blocks 3 and 4), the original code had a second issue. When teacher and student spatial dimensions differed, it flattened the 5D attention map to 1D and applied **linear interpolation**:

```python
# Original (incorrect)
attn = (features ** 2).sum(dim=1)  # [B, T, H, W]
attn = attn.view(attn.size(0), -1)  # [B, T*H*W] — flattened!
# ...
s_attn = F.interpolate(s_attn.unsqueeze(1), size=t_attn.shape[1], mode="linear")
```

This is mathematically incorrect. Consider teacher features with shape `[B, C, 24, 14, 14]` and student features with shape `[B, C, 24, 4, 4]`. After channel summation and flattening:

- Teacher: `[B, 24 * 14 * 14]` = `[B, 4704]`
- Student: `[B, 24 * 4 * 4]` = `[B, 384]`

The 1D linear interpolation from 384 to 4704 treats these as 1D signals. But the underlying data has the structure `[T, H, W]` — three interleaved dimensions. Interpolating along a single axis **mixes temporal and spatial positions**. For example, the interpolated value at position 400 in the teacher's flat vector corresponds to timestep $t = \lfloor 400 / 196 \rfloor = 2$, spatial position $(h, w) = (400 \mod 196)$, but the corresponding student position $400 \times (384/4704) \approx 33$ maps to a completely different $(t, h, w)$ in the student's feature space.

---

## 4. The Solution: Separated Spatial and Temporal Attention

### 4.1 Design Rationale

For 3D video features, the spatial dimensions $(H, W)$ and the temporal dimension $(T)$ carry fundamentally different semantic information:

- **Spatial attention** answers: *"Where in each frame does the network focus?"* — this captures object location, salient regions, and texture patterns.
- **Temporal attention** answers: *"When during the video does activation peak?"* — this captures motion dynamics, action phases, and temporal structure.

Collapsing these into a single flat vector, as the original implementation did, destroys this distinction and makes it impossible for the loss to provide separate gradients for spatial vs. temporal corrections.

### 4.2 Mathematical Formulation

Given a 5D feature tensor $F \in \mathbb{R}^{B \times C \times T \times H \times W}$, we define:

**Spatial Attention Map:**
$$A_{\text{spatial}}(F) = \text{L2\_normalize}\left(\text{mean}_{c,t}\left(F^2\right)\right) \in \mathbb{R}^{B \times H \cdot W}$$

This averages over all channels and all timesteps, producing a single spatial heatmap that summarizes *where* the network attends across the entire clip.

**Temporal Attention Map:**
$$A_{\text{temporal}}(F) = \text{L2\_normalize}\left(\text{mean}_{c,h,w}\left(F^2\right)\right) \in \mathbb{R}^{B \times T}$$

This averages over all channels and all spatial positions, producing a temporal profile that summarizes *when* activation intensity peaks.

**Combined AT Loss:**
$$\mathcal{L}_{\text{AT}} = \beta_s \sum_{i} \text{MSE}\left(A^T_{\text{spatial},i},\ A^S_{\text{spatial},i}\right) + \beta_t \sum_{i} \text{MSE}\left(A^T_{\text{temporal},i},\ A^S_{\text{temporal},i}\right)$$

where $\beta_s$ and $\beta_t$ are independent weighting coefficients for the spatial and temporal components.

### 4.3 Handling Dimension Mismatches

**Spatial mismatch** ($H_T \times W_T \neq H_S \times W_S$): The student's spatial attention map is reshaped to `[B, 1, H_s, W_s]`, interpolated using **2D bilinear interpolation** to `[B, 1, H_t, W_t]`, then re-flattened and re-normalized. This preserves the 2D spatial structure and correctly interpolates between grid positions.

```python
# Correct: 2D bilinear interpolation preserving spatial structure
s_spatial_2d = s_spatial.view(-1, 1, H_s, W_s)
s_spatial_2d = F.interpolate(s_spatial_2d, size=(H_t, W_t), mode="bilinear", align_corners=False)
s_spatial = s_spatial_2d.view(B, -1)
s_spatial = F.normalize(s_spatial, p=2, dim=1)
```

**Temporal mismatch** ($T_T \neq T_S$): In our architecture, both teacher and student have temporal stride 1 at all layers, so $T_T = T_S = T_{\text{input}}$. **No interpolation is needed.** This is a significant advantage of the temporal attention component: the signal is transferred directly without any approximation. A defensive fallback using 1D linear interpolation is included for robustness but is not expected to trigger under normal conditions.

### 4.4 Total Loss Function

The complete distillation loss with spatial and temporal AT is:

$$\mathcal{L}_{\text{total}} = \underbrace{\alpha \cdot T^2 \cdot D_{\text{KL}}\left(\sigma(z^S / T)\ ||\ \sigma(z^T / T)\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(z^S, y)}_{\text{Knowledge Distillation}} + \underbrace{\beta_s \sum_i \text{MSE}(A^T_{s,i}, A^S_{s,i})}_{\text{Spatial AT}} + \underbrace{\beta_t \sum_i \text{MSE}(A^T_{t,i}, A^S_{t,i})}_{\text{Temporal AT}}$$

---

## 5. Implementation Details

### 5.1 Code Structure

The implementation resides in `src/training/losses.py` and consists of:

| Component | Description |
|---|---|
| `_spatial_attention_map(F)` | Computes spatial attention `[B, H*W]` from 4D or 5D features |
| `_temporal_attention_map(F)` | Computes temporal attention `[B, T]` from 5D features |
| `AttentionTransferLoss` | `nn.Module` that computes spatial + temporal AT over paired layers |
| `CombinedKDATLoss` | `nn.Module` combining KD loss + AT loss into the final objective |

### 5.2 Spatial Attention Map

```python
def _spatial_attention_map(features: torch.Tensor) -> torch.Tensor:
    if features.dim() == 5:
        attn = (features ** 2).mean(dim=(1, 2))  # mean over C and T → [B, H, W]
    elif features.dim() == 4:
        attn = (features ** 2).mean(dim=1)        # mean over C → [B, H, W]
    else:
        raise ValueError(f"Expected 4D or 5D features, got {features.dim()}D")
    attn = attn.view(attn.size(0), -1)            # [B, H*W]
    attn = F.normalize(attn, p=2, dim=1)
    return attn
```

Key design choices:
- **`mean` instead of `sum`**: Using mean over channels and time makes the magnitude independent of the number of channels $C$ (which differs between teacher and student) and the temporal length $T$. This produces comparable scales without additional normalization.
- **Explicit dimensionality check**: Raises `ValueError` for unexpected inputs (e.g., the 2D head output), preventing silent failures.

### 5.3 Temporal Attention Map

```python
def _temporal_attention_map(features: torch.Tensor) -> torch.Tensor:
    if features.dim() != 5:
        raise ValueError(f"Temporal attention requires 5D features, got {features.dim()}D")
    attn = (features ** 2).mean(dim=(1, 3, 4))  # mean over C, H, W → [B, T]
    attn = F.normalize(attn, p=2, dim=1)
    return attn
```

The temporal map is strictly defined for 5D inputs. It produces a $T$-dimensional vector per sample, where each value represents the aggregate activation intensity at that timestep. L2 normalization ensures the temporal profile is compared by shape (relative distribution), not by absolute magnitude.

### 5.4 Backward Compatibility

The trainer reads AT hyperparameters from the YAML configuration with a fallback mechanism:

```python
at_beta_fallback = kd_cfg.get("at_beta", 0.05)
beta_spatial = kd_cfg.get("at_beta_spatial", at_beta_fallback)
beta_temporal = kd_cfg.get("at_beta_temporal", at_beta_fallback)
```

This ensures that older configuration files specifying a single `at_beta` value continue to function correctly: the value is used for both spatial and temporal components. New configurations can specify `at_beta_spatial` and `at_beta_temporal` independently.

---

## 6. Experimental Configuration

### 6.1 Fair Comparison Protocol

To isolate the effect of AT from other hyperparameter choices, the AT experiments use **identical settings** to the validated baseline distillation run (`slurm-train-eval-4201`):

| Parameter | Value | Note |
|---|---|---|
| Epochs | 60 | Same as 4201 |
| Scheduler | `cosine` | Same as 4201 (no warmup) |
| Learning rate | 0.0005 | Same as 4201 |
| Batch size | 16 | Same as 4201 |
| Label smoothing | 0.0 | Same as 4201 (default) |
| Temperature $T$ | 8.0 | Same as 4201 |
| Alpha $\alpha$ | 0.7 | Same as 4201 |
| Num frames | 24 | Same as 4201 |
| Eval split | Group-aware, seed=42 | Same as 4201 |

The **only** additions are: `at_beta_spatial`, `at_beta_temporal`, `teacher_keys`, and `student_keys`.

### 6.2 Hyperparameter Variants

Two configurations are tested:

| Variant | $\beta_s$ | $\beta_t$ | Rationale |
|---|:---:|:---:|---|
| **Symmetric** | 0.05 | 0.05 | Equal weight to spatial and temporal signals |
| **Temporal-heavy** | 0.03 | 0.07 | Temporal AT has no interpolation noise (exact $T$ match), so it can carry a stronger signal. Spatial AT with bilinear interpolation introduces approximation error, justifying a lower weight. |

---

## 7. Verification

### 7.1 Automated Tests

A test suite (`tests/test_at_loss.py`) validates:

1. **Shape correctness**: spatial map produces `[B, H*W]`, temporal map produces `[B, T]`
2. **L2 normalization**: output norms are verified to be 1.0
3. **Dimensionality guards**: 4D input to temporal map raises `ValueError`
4. **Spatial mismatch handling**: 2D bilinear interpolation produces correct shapes when $H_T \neq H_S$
5. **Multi-pair computation**: 3 feature pairs produce finite, non-negative loss
6. **Gradient flow**: backward pass propagates gradients to both student features and student head
7. **Backward compatibility**: single `at_beta` config produces identical $\beta_s = \beta_t$
8. **Component isolation**: setting $\beta_s = 0$ produces zero spatial loss with non-zero temporal loss

All 10 tests pass successfully.

### 7.2 Integration Validation

The SLURM pipeline (`cluster/submit_at_temporal.sh`) trains both configurations sequentially with evaluation after each, producing a `pipeline_summary.txt` that directly compares against the 63.78% best eval accuracy from the KD-only baseline.

---

## 8. Summary of Changes

| File | Change | Purpose |
|---|---|---|
| `src/models/teacher.py` | `FEATURE_BLOCKS`: `[3,4,5]` → `[2,3,4]` | Exclude classifier head (block 5) |
| `src/models/student.py` | Comment update | Document correct block pairing |
| `src/training/losses.py` | Full rewrite | Separated spatial + temporal AT with correct interpolation |
| `src/training/trainer.py` | `_build_criterion` update | Support `at_beta_spatial` / `at_beta_temporal` with fallback |
| `experiments/configs/distillation_at_temporal.yaml` | New config | Symmetric AT ($\beta_s = \beta_t = 0.05$) |
| `experiments/configs/distillation_at_temporal_heavy.yaml` | New config | Temporal-heavy AT ($\beta_s = 0.03$, $\beta_t = 0.07$) |
| `experiments/configs/*.yaml` (3 files) | `teacher_keys` fix | All existing AT configs updated to `[2,3,4]` |
| `cluster/submit_at_temporal.sh` | New script | Single SLURM job for both AT experiments |
| `tests/test_at_loss.py` | New tests | 10 automated verification tests |

---

## References

1. Zagoruyko, S., & Komodakis, N. (2017). *Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer*. ICLR 2017.
2. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. NeurIPS Workshop.
3. Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). *SlowFast Networks for Video Recognition*. ICCV 2019.
4. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018.
