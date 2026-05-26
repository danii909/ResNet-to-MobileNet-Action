
# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_PRIMARY)
add_rect(sl, Inches(0), Inches(0), Inches(13.33), Inches(0.07), fill_color=C_HIGHLIGHT2)
add_textbox(sl, "Knowledge Distillation", Inches(0.6), Inches(1.6), Inches(12), Inches(1.2), font_size=46, bold=True, color=C_WHITE)
add_textbox(sl, "for Mobile Action Recognition", Inches(0.6), Inches(2.7), Inches(12), Inches(0.8), font_size=34, color=RGBColor(0xAA, 0xBB, 0xFF))
add_textbox(sl, "Compressing a 3D ResNet-50 teacher into a MobileNet3D student\nfor efficient video action recognition on UCF-101", Inches(0.6), Inches(3.45), Inches(11), Inches(0.7), font_size=14, color=RGBColor(0xCC, 0xCC, 0xCC))
add_rect(sl, Inches(0.6), Inches(4.3), Inches(3.0), Inches(0.04), fill_color=C_HIGHLIGHT2)
add_multiline(sl, [
    ("DEEP LEARNING — ADVANCED MODELS AND METHODS", True, RGBColor(0x88, 0x99, 0xFF), 9),
    ("Daniele Barbagallo   |   Student ID 1000015334", False, RGBColor(0xCC, 0xCC, 0xCC), 11),
    ("Università di Catania — DMI   |   A.A. 2025/2026", False, RGBColor(0x99, 0x99, 0xBB), 10),
    ("Advisor: Prof. Antonino Furnari", False, RGBColor(0x99, 0x99, 0xBB), 10),
], Inches(0.6), Inches(4.5), Inches(9), Inches(2.0))
footer(sl, 1)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — PROBLEM & MOTIVATION (Schematized)
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "PROBLEM & MOTIVATION")
slide_title(sl, "Problem Statement & Objective")

# Block 1: Goal
add_rect(sl, Inches(0.45), Inches(1.7), Inches(12.4), Inches(1.4), fill_color=C_BOX)
add_textbox(sl, "✓  Goal", Inches(0.6), Inches(1.8), Inches(11), Inches(0.4), font_size=16, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    ("Compress a 3D ResNet-50 teacher (~128 MB, pretrained on Kinetics-400)", False, C_PRIMARY, 13),
    ("into a MobileNet3D student (~9.5 MB) for on-device video action recognition", False, C_PRIMARY, 13),
    ("on UCF-101 (101 classes) at zero size overhead.", False, C_PRIMARY, 13),
], Inches(0.8), Inches(2.2), Inches(11.5), Inches(0.8), line_spacing=18)

# Block 2: Capacity Gap
add_rect(sl, Inches(0.45), Inches(3.4), Inches(12.4), Inches(3.2), fill_color=C_BOX2)
add_textbox(sl, "📉  Capacity Gap", Inches(0.6), Inches(3.5), Inches(11), Inches(0.4), font_size=16, bold=True, color=C_HIGHLIGHT2)

# Sub-blocks for gap
add_rect(sl, Inches(0.8), Inches(4.2), Inches(5.0), Inches(2.0), fill_color=C_WHITE, line_color=C_LGRAY)
add_textbox(sl, "88.82%", Inches(0.8), Inches(4.5), Inches(5.0), Inches(0.8), font_size=36, bold=True, color=C_HIGHLIGHT, align=PP_ALIGN.CENTER)
add_textbox(sl, "Teacher Top-1 (ResNet-50)", Inches(0.8), Inches(5.3), Inches(5.0), Inches(0.5), font_size=12, color=C_GRAY, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(6.0), Inches(4.2), Inches(5.0), Inches(2.0), fill_color=C_WHITE, line_color=C_LGRAY)
add_textbox(sl, "59.48%", Inches(6.0), Inches(4.5), Inches(5.0), Inches(0.8), font_size=36, bold=True, color=C_GRAY, align=PP_ALIGN.CENTER)
add_textbox(sl, "Baseline Student Top-1", Inches(6.0), Inches(5.3), Inches(5.0), Inches(0.5), font_size=12, color=C_GRAY, align=PP_ALIGN.CENTER)

add_textbox(sl, "Gap: −29.3 pp", Inches(0.45), Inches(6.8), Inches(6.2), Inches(0.4), font_size=16, bold=True, color=C_ERROR, align=PP_ALIGN.CENTER)
add_textbox(sl, "Size Ratio: 13.5×", Inches(6.65), Inches(6.8), Inches(6.2), Inches(0.4), font_size=16, bold=True, color=C_PRIMARY, align=PP_ALIGN.CENTER)

footer(sl, 2)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — EXPERIMENTAL SETUP (Schematized)
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EXPERIMENTAL SETUP")
slide_title(sl, "Dataset, Group-Aware Split & Training Config")

# Setup Left
add_rect(sl, Inches(0.45), Inches(1.7), Inches(4.0), Inches(5.0), fill_color=C_BOX)
add_textbox(sl, "UCF-101 Dataset", Inches(0.6), Inches(1.8), Inches(3.7), Inches(0.4), font_size=14, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "• 101 action classes",
    "• 13,320 video clips (YouTube)",
    "• 24 frames/clip (112×112)",
    "• Test split: 3,783 clips"
], Inches(0.6), Inches(2.2), Inches(3.7), Inches(1.2), font_size=12)

add_textbox(sl, "Training Configuration", Inches(0.6), Inches(3.8), Inches(3.7), Inches(0.4), font_size=14, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    ("Student (MobileNet3D):", True, C_PRIMARY, 11),
    "AdamW, LR 5e-4, 60 epochs",
    "Batch 16, AMP, Label Smooth 0.1",
    "",
    ("Teacher (ResNet-50):", True, C_PRIMARY, 11),
    "SGD, LR 7e-4, 50 epochs"
], Inches(0.6), Inches(4.2), Inches(3.7), Inches(2.0), font_size=11)

# Group-Aware Right
add_rect(sl, Inches(4.6), Inches(1.7), Inches(8.2), Inches(5.0), fill_color=C_WHITE, line_color=C_HIGHLIGHT)
add_rect(sl, Inches(4.6), Inches(1.7), Inches(8.2), Inches(0.6), fill_color=C_HIGHLIGHT)
add_textbox(sl, "Group-Aware Eval Split", Inches(4.8), Inches(1.8), Inches(7.8), Inches(0.4), font_size=16, bold=True, color=C_WHITE)

add_textbox(sl, "Motivation", Inches(4.8), Inches(2.5), Inches(7.8), Inches(0.4), font_size=14, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "UCF-101 contains clips extracted from the same original source videos.",
    "Standard random splitting results in data leakage, creating overly",
    "optimistic, fake validation metrics.",
    "The training set was split in a wise, group-aware manner based on",
    "source video IDs (extracted via regex, e.g. '_gXX_') to obtain an",
    "internal evaluation set which was otherwise absent."
], Inches(5.0), Inches(2.9), Inches(7.4), Inches(1.5), font_size=12, line_spacing=16)

add_textbox(sl, "Impact", Inches(4.8), Inches(4.8), Inches(7.8), Inches(0.4), font_size=14, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "Evaluation set metrics dropped from fictitiously high values to a realistic",
    "60-65% validation range, exposing the true generalization gap and",
    "ensuring reliable hyperparameter tuning."
], Inches(5.0), Inches(5.2), Inches(7.4), Inches(1.0), font_size=12, line_spacing=16)

footer(sl, 3)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — LOGIT-BASED KD
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "LOGIT-BASED KNOWLEDGE DISTILLATION")
slide_title(sl, "Method — Logit-Based KD", "Temperature-scaled soft targets from teacher logits")
# Add a schematic representation of KD
add_rect(sl, Inches(1.0), Inches(2.5), Inches(3.0), Inches(1.5), fill_color=C_BOX)
add_textbox(sl, "Teacher\nResNet-50", Inches(1.0), Inches(2.9), Inches(3.0), Inches(0.8), font_size=20, bold=True, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(9.3), Inches(2.5), Inches(3.0), Inches(1.5), fill_color=C_BOX2)
add_textbox(sl, "Student\nMobileNet3D", Inches(9.3), Inches(2.9), Inches(3.0), Inches(0.8), font_size=20, bold=True, align=PP_ALIGN.CENTER)

add_textbox(sl, "KL Divergence Loss\n(Temperature T, Weight α)", Inches(4.5), Inches(2.8), Inches(4.3), Inches(0.8), font_size=16, bold=True, color=C_HIGHLIGHT, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(1.0), Inches(5.0), Inches(11.3), Inches(1.5), fill_color=C_WHITE, line_color=C_LGRAY)
add_multiline(sl, [
    "• Ground Truth Labels: Hard 0/1 targets (Cross-Entropy).",
    "• Teacher Soft Targets: Teacher logits smoothed by Temperature T.",
    "• KD Warmup: Train with Cross-Entropy only for 10 epochs, then introduce KD loss."
], Inches(1.2), Inches(5.2), Inches(10.9), Inches(1.0), font_size=14, line_spacing=20)
footer(sl, 4)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — KD TEMPERATURE ABLATION Top-1
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "LOGIT-BASED KD — TEMPERATURE ABLATION")
slide_title(sl, "Test Set Top-1 Accuracy — All Configurations")
pic(sl, HIST_TOP1, Inches(2.6), Inches(1.7), Inches(8), Inches(4.5))
add_textbox(sl, "KD T=20 achieves 65.85% (+6.37 pp vs baseline 59.48%). Performance is monotonically non-decreasing with temperature. AT symmetric and Born-Again Gen3 fall below baseline.",
            Inches(0.45), Inches(6.5), Inches(12.4), Inches(0.5), font_size=12, color=C_PRIMARY)
footer(sl, 5)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — KD TEMPERATURE ABLATION Top-5
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "LOGIT-BASED KD — TEMPERATURE ABLATION")
slide_title(sl, "Test Set Top-5 Accuracy — All Configurations")
pic(sl, HIST_TOP5, Inches(2.6), Inches(1.7), Inches(8), Inches(4.5))
add_textbox(sl, "KD T=20 also achieves best Top-5: 89.24% (+6.47 pp vs baseline 82.77%). Soft targets consistently improve ranking quality across all temperature settings.",
            Inches(0.45), Inches(6.5), Inches(12.4), Inches(0.5), font_size=12, color=C_PRIMARY)
footer(sl, 6)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — KD TEMPERATURE ABLATION EXPLANATION
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "LOGIT-BASED KD — TEMPERATURE ABLATION")
slide_title(sl, "Temperature Ablation Insights")

add_rect(sl, Inches(0.45), Inches(1.7), Inches(12.4), Inches(1.6), fill_color=C_BOX)
add_textbox(sl, "The Role of Temperature (T)", Inches(0.6), Inches(1.8), Inches(12.0), Inches(0.4), font_size=16, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "Higher temperatures smooth out the predicted class distributions of the teacher.",
    "This reveals rich 'dark knowledge' about inter-class relationships (e.g., how similar running is to jogging),",
    "which provides a much denser training signal than one-hot labels."
], Inches(0.6), Inches(2.3), Inches(12.0), Inches(1.0), font_size=14, line_spacing=20)

add_rect(sl, Inches(0.45), Inches(3.6), Inches(5.8), Inches(2.8), fill_color=C_WHITE, line_color=C_SUCCESS)
add_textbox(sl, "✓ Optimal Balance: T=20", Inches(0.6), Inches(3.7), Inches(5.5), Inches(0.4), font_size=16, bold=True, color=C_SUCCESS)
add_multiline(sl, [
    "T=20 yields the optimal balance:",
    "It softens the logits enough to effectively transfer structural relationships without completely washing out the primary class signal.",
    "Achieves the peak 65.85% Top-1 accuracy."
], Inches(0.6), Inches(4.2), Inches(5.5), Inches(1.8), font_size=13, line_spacing=18)

add_rect(sl, Inches(7.05), Inches(3.6), Inches(5.8), Inches(2.8), fill_color=C_WHITE, line_color=C_ERROR)
add_textbox(sl, "⚠️ Extreme Temperatures", Inches(7.2), Inches(3.7), Inches(5.5), Inches(0.4), font_size=16, bold=True, color=C_ERROR)
add_multiline(sl, [
    "Pushing T even higher starts to degrade performance due to over-smoothing.",
    "Specific classes that rely on sharp, distinct visual boundaries (e.g., Hammering or PizzaTossing) suffer because their unique features get blended with noise."
], Inches(7.2), Inches(4.2), Inches(5.5), Inches(1.8), font_size=13, line_spacing=18)

footer(sl, 7)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — t-SNE
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "QUALITATIVE ANALYSIS — LATENT SPACE")
slide_title(sl, "t-SNE Visualization", "Pre-classifier embeddings for 10 random UCF-101 classes (KD T=20)")
pic(sl, f"{T20_TSNE}/tsne_comparison_4565.png", Inches(0.2), Inches(1.7), Inches(12.9), Inches(4.5))
add_textbox(sl, "Teacher (ResNet-50): Compact, well-separated | Baseline MobileNet3D: Significant inter-class overlap | Distilled Student: Tighter clusters, improved class separation",
            Inches(0.45), Inches(6.5), Inches(12.4), Inches(0.5), font_size=12, color=C_PRIMARY)
footer(sl, 8)
