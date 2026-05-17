# Track 6: Knowledge Distillation for Mobile Action Recognition
## Riassunto Progetto e Opzioni di Fix per Attention Transfer Temporale

---

## 📋 Contesto Generale

**Obiettivo:** Comprimere un modello 3D ResNet-50 (teacher) pesante in un MobileNet3D (student) leggero (5-10x più piccolo) mantenendo accuratezza su video action recognition.

**Dataset:** UCF-101 (101 azioni, video 24 frame a 112×112)

**Vincoli:** Non si possono modificare le architetture dei modelli, solo i hyperparameter di training/distillation.

## 🔴 Problema Identificato

**L'Attention Transfer attualmente era SOLO SPAZIALE:**

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

52.06% test accuracy.

---

## 🎯 Extra Objective (dal progetto)

**Expand KD to include Attention Transfer across intermediate temporal activation mappings.**

Significa: **Implementare AT che cattura specificamente i pattern TEMPORALI, non solo spaziali.**

---

## 3️⃣ OPZIONe DI FIX

### **Attenzione Separata Spaziale + Temporale** ⭐ CONSIGLIATA

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

