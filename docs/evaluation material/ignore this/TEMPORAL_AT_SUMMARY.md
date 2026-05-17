# Track 6: Knowledge Distillation for Mobile Action Recognition
## Riassunto Progetto e Opzioni di Fix per Attention Transfer Temporale

---

## 📋 Contesto Generale

**Obiettivo:** Comprimere un modello 3D ResNet-50 (teacher) pesante in un MobileNet3D (student) leggero (5-10x più piccolo) mantenendo accuratezza su video action recognition.

**Dataset:** UCF-101 (101 azioni, video 24 frame a 112×112)

**Vincoli:** Non si possono modificare le architetture dei modelli, solo i hyperparameter di training/distillation.

---

## 🏗️ Architetture Attuali

| Componente | Dettagli |
| --- | --- |
| **Teacher** | 3D ResNet-50 pretrained Kinetics-400, fine-tuned su UCF-101 → **92.31% accuracy** |
| **Student** | MobileNet3D (3-5M param), ~5.5x compression vs teacher |
| **Input** | [B, 3, T=24, H=112, W=112] |

---

## 📊 Implementazione Attuale (KD Classica + AT)

### Modalità Training Disponibili

#### Mode: `distillation` (Response-Based Only)
```
Loss KD: α·T²·KL(soft_student || soft_teacher) + (1-α)·CE(student, labels)

Parametri:
  - T (temperature): 8.0
  - α (KD weight): 0.7
  
Risultato: best_eval_acc = 52.16% ✓
```

#### Mode: `distillation_at` (Response-Based + Feature-Based)
```
Loss KD:  α·T²·KL(soft_student || soft_teacher) + (1-α)·CE(student, labels)
Loss AT:  β·Σ_i MSE(attention_teacher_i, attention_student_i)
Loss TOT: KD + AT

Parametri:
  - T (temperature): 8.0
  - α (KD weight): 0.7
  - β (AT weight): 0.05
  - Feature pairs: teacher blocks [3,4,5] → student stages [2,4,6]
  
Risultato: best_eval_acc = 52.06% ❌ (non migliora)
```

---

## 🔴 Problema Identificato

**L'Attention Transfer attualmente è SOLO SPAZIALE:**

```
Input features: [B, C, T, H, W]
   ↓
Somma sui canali → [B, T, H, W]
   ↓
Appiattisce tutto → [B, T×H×W]
   ↓
L2 normalize → [B, N] (attention map)
   ↓
MSE Loss

❌ NON distingue pattern temporali (movimento) da pattern spaziali (texture)
```

**Perché è un problema per action recognition?**
- Il movimento nel tempo (come evolvono i frame) è **cruciale** per riconoscere le azioni
- Ma viene "perso" mescolato nello spazio
- L'AT spaziale insegna solo "quale zona guardare" ma non "come cambia nel tempo"

---

## 📈 Risultati Attuali vs Attesi

| Metrica | Teacher | Baseline Student | KD (logits-only) | KD+AT (spaziale) |
| --- | --- | --- | --- | --- |
| Best eval acc | 92.31% | 45.31% | 52.16% ✓ | 52.06% ❌ |
| Improvement over baseline | — | baseline | +6.85% | +6.75% |

**Osservazione:** AT spaziale non porta vantaggio netto ma neppure degrada.

---

## 🎯 Extra Objective (dal progetto)

**Expand KD to include Attention Transfer across intermediate temporal activation mappings.**

Significa: **Implementare AT che cattura specificamente i pattern TEMPORALI, non solo spaziali.**

---

## 3️⃣ OPZIONI DI FIX

### **Opzione 1: Attenzione Separata Spaziale + Temporale** ⭐ CONSIGLIATA

#### Cosa Fa
Calcola due mappe di attenzione distinte:
- **Spaziale:** somma canali+tempo → [B, H, W] → "quale zona guardare?"
- **Temporale:** somma canali+spazio → [B, T] → "come evolve nel tempo?"

#### Implementazione
```python
def _spatial_attention_map(features: torch.Tensor) -> torch.Tensor:
    """Spatial attention: sum over channels and time."""
    attn = (features ** 2).mean(dim=(1, 2))  # [B, C, T, H, W] → [B, H, W]
    attn = attn.view(attn.size(0), -1)       # [B, H*W]
    attn = F.normalize(attn, p=2, dim=1)
    return attn

def _temporal_attention_map(features: torch.Tensor) -> torch.Tensor:
    """Temporal attention: sum over channels and space."""
    attn = (features ** 2).mean(dim=(1, 3, 4))  # [B, C, T, H, W] → [B, T]
    attn = F.normalize(attn, p=2, dim=1)
    return attn
```

#### Loss Computation
```python
# In CombinedKDATLoss
for t_key, s_key in zip(teacher_keys, student_keys):
    # Spatial AT
    t_spatial = _spatial_attention_map(teacher_features[t_key])
    s_spatial = _spatial_attention_map(student_features[s_key])
    loss_spatial += F.mse_loss(s_spatial, t_spatial)
    
    # Temporal AT
    t_temporal = _temporal_attention_map(teacher_features[t_key])
    s_temporal = _temporal_attention_map(student_features[s_key])
    loss_temporal += F.mse_loss(s_temporal, t_temporal)

# Total AT loss
at_loss = self.beta_spatial * loss_spatial + self.beta_temporal * loss_temporal
```

#### Vantaggi
- ✅ Semplice, interpretabile, low risk
- ✅ Insegna al student come il movimento evolve nel tempo
- ✅ Minima modifica al codice (~30-40 righe)
- ✅ Separa correttamente spazio e tempo
- ✅ Pragmatico per Track 6 (Small)

#### Svantaggi
- ❌ Potrebbe non catturare interazioni spazio-temporali complesse

#### Effort
**Basso: ~1-2 ore di implementazione + test**

---

### **Opzione 2: Attenzione Multi-Scala Temporale** 

#### Cosa Fa
Cattura movimento a diverse velocità:
- **Fast:** differenza frame-a-frame (movimenti rapidi, balzi)
- **Medium:** differenza ogni 4 frame (movimenti medi)
- **Slow:** differenza ogni 8 frame (movimenti lenti, fluidi)

#### Implementazione Snippet
```python
def temporal_attention_multi_scale(features: torch.Tensor):
    """features: [B, C, T, H, W]"""
    B, C, T, H, W = features.shape
    attn_maps = {}
    
    # Somma canali: [B, T, H, W]
    feature_norm = (features ** 2).sum(dim=1)
    
    # Fast: movimento frame-to-frame
    diff_fast = feature_norm[:, 1:] - feature_norm[:, :-1]
    attn_maps['fast'] = diff_fast.abs().mean(dim=(2, 3))  # [B, T-1]
    
    # Medium: movimento su 4 frame
    diff_medium = feature_norm[:, 4:] - feature_norm[:, :-4]
    attn_maps['medium'] = diff_medium.abs().mean(dim=(2, 3))  # [B, T-4]
    
    # Slow: movimento su 8 frame
    diff_slow = feature_norm[:, 8:] - feature_norm[:, :-8]
    attn_maps['slow'] = diff_slow.abs().mean(dim=(2, 3))  # [B, T-8]
    
    return attn_maps
```

#### Vantaggi
- ✅ Cattura salti vs fluidità di movimento
- ✅ Rich temporal patterns a multi-scala
- ✅ Potrebbe migliorare performance su azioni complesse

#### Svantaggi
- ⚠️ Più complessa, più hyperparameter da bilanciare (β_fast, β_medium, β_slow)
- ⚠️ Effort medio (~3-4 ore)
- ❌ Rischio overfitting se hyperparameter non ben tuned

#### Effort
**Medio: ~3-4 ore di implementazione + tuning**

---

### **Opzione 3: Attenzione con 3D Convolution** (Advanced)

#### Cosa Fa
Usa Conv3D per imparare automaticamente interazioni spazio-temporali:
- La rete apprende quale pattern spazio-temporale è importante
- End-to-end learning, massima flessibilità

#### Implementazione Snippet
```python
class TemporalAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1)
    
    def forward(self, features):
        # [B, C, T, H, W] → [B, 1, T, H, W]
        attn = self.conv3d(features)
        # Pool to [B, 1]
        attn = attn.abs().mean(dim=(2, 3, 4))
        return attn
```

#### Vantaggi
- ✅ Learned end-to-end, massima flessibilità
- ✅ Cattura relazioni spazio-temporali complesse
- ✅ Potenziale massimo di improvement

#### Svantaggi
- ❌ Aggiunge trainable parameters (verifica vincoli del progetto!)
- ⚠️ Più difficile da debuggare
- ⚠️ Effort alto (~4-5 ore + tuning)
- ❌ Overhead computazionale

#### Effort
**Alto: ~4-5 ore di implementazione + tuning**

---

## 📋 Tabella Comparativa

| Criterio | Opzione 1 | Opzione 2 | Opzione 3 |
| --- | --- | --- | --- |
| **Efficienza effort/miglioramento** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Interpretabilità** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Potenziale di improvement** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Align ai vincoli** | ✅ Ok | ✅ Ok | ⚠️ Check |
| **Implementazione** | <2h | 3-4h | 4-5h |
| **Testing** | ~1h | ~2h | ~3h |
| **Risk Level** | Low | Medium | High |

---

## 🏆 Raccomandazione

### **→ Opzione 1 (Separata Spaziale+Temporale)**

#### Motivi
1. **Basso sforzo, alto payoff:** 30-40 righe di codice, test veloce
2. **Concettualmente corretta:** Separa correttamente spazio/tempo per video
3. **Safe baseline:** Se non funziona, facile rollback; se funziona, upgrade a opzione 2
4. **Align al progetto:** Track 6 è "Small" → pragmatismo > sofisticazione
5. **Quick win per il report:** Dimostra riflessione sul problema temporale

#### Flow Suggerito
1. **Implementa Opzione 1** (Spaziale+Temporale)
2. **Run 3-4 training experiment** con β_spatial, β_temporal variabili
   - Suggested: β_spatial=0.05, β_temporal=0.05 (symmetric)
   - Or: β_spatial=0.03, β_temporal=0.07 (temporal-heavy)
3. **Se improvement >= 0.5%:** Documenta, move on ✓
4. **Se nessun improvement:** Try Opzione 2 (multi-scala)
5. **Se tempo/computational budget esaurito:** Opzione 3 rimane in "Future Work"

---

## 📝 Mettriche da Monitorare

Durante i test, trackare:
- **best_eval_acc:** Accuracy massima nel validation set
- **best_epoch:** A quale epoca si raggiunge il massimo
- **final_train_acc:** Accuracy finale sul training
- **eval_acc_top5:** Top-5 accuracy
- **eval_loss:** Loss di validazione (deve calare)

**Target minimo:** +0.5% improvement su best_eval_acc rispetto a KD logits-only (52.16%)

---

## 🔧 File da Modificare

Se procedi con Opzione 1:

1. **[src/training/losses.py](src/training/losses.py)**
   - Aggiungi `_temporal_attention_map()` function
   - Modifica `AttentionTransferLoss.forward()` per calcolare sia spatial che temporal
   - Update docstring con nuova formula della loss

2. **[src/training/trainer.py](src/training/trainer.py)**
   - Eventualmente log separati per spatial_at_loss e temporal_at_loss

3. **Config file (es. [experiments/configs/distillation_at_24f_v2.yaml](experiments/configs/distillation_at_24f_v2.yaml))**
   - Aggiungi parametri:
     ```yaml
     distillation:
       at_beta_spatial: 0.05
       at_beta_temporal: 0.05
     ```

---

## ✅ Checklist Pre-Implementazione

- [ ] Leggere e comprendere il codice attuale in `_spatial_attention_map()`
- [ ] Verificare che non ci siano vincoli su trainable parameters nel progetto
- [ ] Pianificare 3-4 config di test con β variabili
- [ ] Preparare script di plotting per confrontare risultati
- [ ] Documentare nel report qualsiasi risultato (anche se negativo)

---

## 📚 Riferimenti nel Codice

- **Feature extraction:** [src/models/teacher.py](src/models/teacher.py) (hooks su blocchi 3,4,5)
- **Feature extraction student:** [src/models/student.py](src/models/student.py) (stages 2,4,6)
- **Loss attuale:** [src/training/losses.py](src/training/losses.py) (AttentionTransferLoss)
- **Training loop:** [src/training/trainer.py](src/training/trainer.py) (_compute_loss method)

---

**Data:** 4 Maggio 2026  
**Track:** 6 - Knowledge Distillation for Mobile Action Recognition  
**Team:** G24
