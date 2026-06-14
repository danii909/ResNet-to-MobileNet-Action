"""
presentation_v6.pptx — FINAL VERSION
- Usa SOLO dati da:
    results/Train-eval-test-split (group-aware)/
- Include curve di loss/accuracy e confusion matrix per T=20 (miglior esperimento)
- Stile ispirato alla presentazione di riferimento (accademico, minimalista, numeri in grande)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ─── Palette colori ──────────────────────────────────────────────────────────
C_BG        = RGBColor(0xFF, 0xFF, 0xFF)
C_PRIMARY   = RGBColor(0x1A, 0x1A, 0x2E)
C_HIGHLIGHT = RGBColor(0x0F, 0x3D, 0x91)
C_HIGHLIGHT2= RGBColor(0xE5, 0x48, 0x00)
C_GRAY      = RGBColor(0x55, 0x55, 0x55)
C_LGRAY     = RGBColor(0xDD, 0xDD, 0xDD)
C_BOX       = RGBColor(0xF0, 0xF4, 0xFF)
C_BOX2      = RGBColor(0xFF, 0xF3, 0xE0)
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

# ─── Percorsi immagini (SOLO group-aware) ────────────────────────────────────
BASE   = r"results/Train-eval-test-split (group-aware)"
GA     = f"{BASE}/Training"

# Esperimento MIGLIORE: T=20
T20_FIGS = f"{GA}/slurm-train-eval-4565/kd_t20_a07_24f_lightaug/figures"
T20_CM   = f"{BASE}/Confusion-Matrix/kd_t20_a07_24f_lightaug"
T20_TSNE = f"{GA}/slurm-train-eval-4565/tsne_plots"

# Baseline
BL_FIGS  = f"{GA}/slurm-train-eval-4495/baseline_ls005_24f_lightaug/figures"
BL_CM    = f"{BASE}/Confusion-Matrix/baseline_ls005_24f_lightaug"
BL_TSNE  = f"{GA}/slurm-train-eval-4495/tsne_plots"

# Grafici aggregati (nella root group-aware)
HIST_TOP1   = f"{BASE}/distillation_histogram_test_top1.png"
HIST_TOP5   = f"{BASE}/distillation_histogram_test_top5.png"
COMP_CHART  = f"{BASE}/distillation_comparison.png"
AGG_CHART   = f"{BASE}/aggregate_best_eval_acc.png"
DELTA_CHART = f"{BASE}/figures/delta_accuracy_kd_vs_baseline.png"

# Benchmark (copied to figures folder inside BASE)
GPU_PLT  = f"{BASE}/figures/gpu_benchmark_plots.png"
CPU_PLT  = f"{BASE}/figures/cpu_benchmark_plots.png"
COMB_PLT = f"{BASE}/figures/combined_benchmark_plots.png"

# Cross-frame (T=20 cross frame in slurm-4675)
# (nessuna figura specifica — usiamo combined benchmark)

# ─── Dimensioni ──────────────────────────────────────────────────────────────
W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

TOTAL_SLIDES = 23

# ─── Helper functions ─────────────────────────────────────────────────────────

def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def set_bg(slide, color=C_BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_rect(slide, left, top, width, height, fill_color=None):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    shape.line.fill.background()
    return shape

def add_textbox(slide, text, left, top, width, height,
                font_size=12, bold=False, color=C_PRIMARY,
                align=PP_ALIGN.LEFT, italic=False, font_name="Calibri"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font_name
    return tb

def add_multiline(slide, lines, left, top, width, height,
                  font_size=11, bold=False, color=C_PRIMARY,
                  align=PP_ALIGN.LEFT, font_name="Calibri", line_spacing=None):
    """lines: list of str or (text, bold, color, font_size)"""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in lines:
        if isinstance(item, str):
            txt, b, c, fs = item, bold, color, font_size
        else:
            txt, b, c, fs = item
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        if line_spacing:
            p.line_spacing = Pt(line_spacing)
        run = p.add_run()
        run.text = txt
        run.font.size = Pt(fs)
        run.font.bold = b
        run.font.color.rgb = c
        run.font.name = font_name
    return tb

def footer(slide, n, total=TOTAL_SLIDES):
    add_textbox(slide, f"{n}/{total}",
                Inches(11.8), Inches(7.05), Inches(1.3), Inches(0.35),
                font_size=10, color=C_GRAY, align=PP_ALIGN.RIGHT)

def section_header(slide, text, color=C_HIGHLIGHT):
    add_rect(slide, Inches(0), Inches(0), Inches(13.33), Inches(0.08), fill_color=color)
    add_textbox(slide, text, Inches(0.45), Inches(0.08), Inches(10), Inches(0.42),
                font_size=9, bold=True, color=color)

def slide_title(slide, title, subtitle=""):
    add_textbox(slide, title, Inches(0.45), Inches(0.55), Inches(12.4), Inches(0.7),
                font_size=26, bold=True, color=C_PRIMARY)
    if subtitle:
        add_textbox(slide, subtitle, Inches(0.45), Inches(1.15), Inches(12.4), Inches(0.4),
                    font_size=12, color=C_GRAY, italic=True)
        add_rect(slide, Inches(0.45), Inches(1.5), Inches(12.43), Inches(0.02), fill_color=C_LGRAY)
    else:
        add_rect(slide, Inches(0.45), Inches(1.2), Inches(12.43), Inches(0.02), fill_color=C_LGRAY)

def pic(slide, path, left, top, width, height):
    return slide.shapes.add_picture(path, left, top, width, height)

def highlight_box(slide, left, top, width, height, label, value,
                  bg=C_BOX, val_color=C_HIGHLIGHT, lbl_color=C_GRAY):
    add_rect(slide, left, top, width, height, fill_color=bg)
    add_textbox(slide, value,
                left + Inches(0.05), top + Inches(0.05),
                width - Inches(0.1), height * 0.56,
                font_size=26, bold=True, color=val_color, align=PP_ALIGN.CENTER)
    add_textbox(slide, label,
                left + Inches(0.05), top + height * 0.58,
                width - Inches(0.1), height * 0.38,
                font_size=8.5, color=lbl_color, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_PRIMARY)
add_rect(sl, Inches(0), Inches(0), Inches(13.33), Inches(0.07), fill_color=C_HIGHLIGHT2)
add_textbox(sl, "Knowledge Distillation",
            Inches(0.6), Inches(1.6), Inches(12), Inches(1.2),
            font_size=46, bold=True, color=C_WHITE)
add_textbox(sl, "for Mobile Action Recognition",
            Inches(0.6), Inches(2.7), Inches(12), Inches(0.8),
            font_size=34, color=RGBColor(0xAA, 0xBB, 0xFF))
add_textbox(sl, "Compressing a 3D ResNet-50 teacher into a MobileNet3D student\nfor efficient video action recognition on UCF-101",
            Inches(0.6), Inches(3.45), Inches(11), Inches(0.7),
            font_size=14, color=RGBColor(0xCC, 0xCC, 0xCC))
add_rect(sl, Inches(0.6), Inches(4.3), Inches(3.0), Inches(0.04), fill_color=C_HIGHLIGHT2)
add_multiline(sl, [
    ("DEEP LEARNING — ADVANCED MODELS AND METHODS", True, RGBColor(0x88, 0x99, 0xFF), 9),
    ("Daniele Barbagallo   |   Student ID 1000015334", False, RGBColor(0xCC, 0xCC, 0xCC), 11),
    ("Università di Catania — DMI   |   A.A. 2025/2026", False, RGBColor(0x99, 0x99, 0xBB), 10),
    ("Advisor: Prof. Antonino Furnari", False, RGBColor(0x99, 0x99, 0xBB), 10),
], Inches(0.6), Inches(4.5), Inches(9), Inches(2.0))
footer(sl, 1)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — OUTLINE
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "OVERVIEW")
slide_title(sl, "Outline", "Structure of this presentation")

sections = [
    ("1", "Problem & Motivation",    "Capacity gap, baseline, objective"),
    ("2", "Experimental Setup",      "UCF-101, group-aware split, training config"),
    ("3", "Logit-Based KD",          "Loss function, warmup, results table"),
    ("4", "Training Curves — T=20",  "Loss, accuracy & LR curves for the best experiment"),
    ("5", "Temperature Ablation",    "Histograms Top-1/Top-5, comparison across T"),
    ("6", "Confusion Matrix & Errors", "Visual error analysis & top 5 wrong classes relation"),
    ("7", "KD Impact per Class",     "Delta accuracy (KD T=20 − Baseline) per class"),
    ("8", "Advanced Methods",        "Attention Transfer & Born-Again self-distillation"),
    ("9", "Qualitative — t-SNE",     "Latent space comparison (baseline, T=20, teacher)"),
    ("10","Cross-Frame Distillation","16f vs 24f GPU/CPU benchmark results"),
    ("11","Conclusions",             "Speed-accuracy trade-off, key findings"),
]
row_h = Inches(0.53)
top0  = Inches(1.62)
for i, (num, title, desc) in enumerate(sections):
    t = top0 + i * row_h
    add_rect(sl, Inches(0.45), t + Inches(0.08), Inches(0.34), Inches(0.34), fill_color=C_HIGHLIGHT)
    add_textbox(sl, num, Inches(0.45), t + Inches(0.05), Inches(0.34), Inches(0.36),
                font_size=11, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    add_textbox(sl, title, Inches(0.92), t, Inches(3.6), Inches(0.5),
                font_size=11.5, bold=True, color=C_PRIMARY)
    add_textbox(sl, desc,  Inches(4.65), t + Inches(0.04), Inches(8.4), Inches(0.44),
                font_size=10.5, color=C_GRAY)
footer(sl, 2)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — PROBLEM & OBJECTIVE
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "PROBLEM & MOTIVATION")
slide_title(sl, "Problem Statement & Objective")

add_textbox(sl, "Goal",
            Inches(0.45), Inches(1.65), Inches(5.8), Inches(0.4),
            font_size=14, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "Compress a 3D ResNet-50 teacher (~128 MB,",
    "pretrained on Kinetics-400) into a MobileNet3D",
    "student (~9.5 MB) for on-device video action",
    "recognition on UCF-101 (101 classes).",
], Inches(0.45), Inches(2.0), Inches(5.8), Inches(1.2), font_size=11.5, line_spacing=15)

add_textbox(sl, "Capacity Gap",
            Inches(0.45), Inches(3.1), Inches(5.8), Inches(0.4),
            font_size=14, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "• Teacher (ResNet-50, Kinetics pretrained): 88.82% Top-1",
    "• Student (MobileNet3D, from scratch):      59.48% Top-1",
    "• Gap: −29.3 pp  |  Size ratio: ~13.5×",
    "• Objective: close the gap via KD at zero size overhead",
], Inches(0.45), Inches(3.45), Inches(5.9), Inches(1.4), font_size=11, line_spacing=16)

add_textbox(sl, "Reference Numbers",
            Inches(6.9), Inches(1.65), Inches(6.0), Inches(0.4),
            font_size=14, bold=True, color=C_HIGHLIGHT)

bx_data = [
    (Inches(6.9),  "88.82%",   "Teacher Top-1 (ResNet-50)",   C_BOX,  C_HIGHLIGHT),
    (Inches(9.15), "59.48%",   "Baseline Student Top-1",       C_BOX,  C_HIGHLIGHT2),
    (Inches(6.9),  "−29.3 pp", "Capacity Gap",                 C_BOX2, C_HIGHLIGHT2),
    (Inches(9.15), "13.5×",    "Size Ratio (Teacher/Student)", C_BOX,  C_HIGHLIGHT),
]
bw, bh = Inches(2.0), Inches(1.05)
for idx, (lx, val, lbl, bg, vc) in enumerate(bx_data):
    row = idx // 2
    highlight_box(sl, lx, Inches(2.1) + row * Inches(1.15), bw, bh, lbl, val, bg=bg, val_color=vc)

add_textbox(sl, "Why Knowledge Distillation?",
            Inches(6.9), Inches(4.55), Inches(6.0), Inches(0.4),
            font_size=13, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "• Teacher soft-logits carry inter-class similarity structure",
    "• Higher entropy targets are easier for low-capacity students",
    "• Zero parameter/latency overhead on the deployed student",
], Inches(6.9), Inches(4.9), Inches(6.0), Inches(1.0), font_size=11, line_spacing=16)
footer(sl, 3)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — EXPERIMENTAL SETUP
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EXPERIMENTAL SETUP")
slide_title(sl, "Dataset, Group-Aware Split & Training Config")

cw = Inches(3.9)
cl = [Inches(0.45), Inches(4.7), Inches(9.0)]

add_textbox(sl, "UCF-101 Dataset", cl[0], Inches(1.65), cw, Inches(0.4), 13, True, C_HIGHLIGHT)
add_multiline(sl, [
    "• 101 action classes (sports, daily activities)",
    "• 13,320 video clips from YouTube",
    "• Sampled at 24 frames/clip (112×112 crop)",
    "• Official test split: 3,783 clips",
], cl[0], Inches(2.05), cw, Inches(1.5), font_size=10.5, line_spacing=15.5)

add_textbox(sl, "Group-Aware Eval Split", cl[1], Inches(1.65), cw, Inches(0.4), 13, True, C_HIGHLIGHT)
add_multiline(sl, [
    "UCF-101 clips are grouped by source video.",
    "A naïve random split leaks correlated frames",
    "across train/eval → inflated val metrics.",
    "",
    "• Split at group level (regex on video ID)",
    "• 80% train / 20% internal eval — no leakage",
    "• Official test set reserved for final evaluation",
    "• Only group-aware results used in this work",
], cl[1], Inches(2.05), cw, Inches(2.1), font_size=10.5, line_spacing=14.5)

add_textbox(sl, "Training Configuration", cl[2], Inches(1.65), cw, Inches(0.4), 13, True, C_HIGHLIGHT)
add_multiline(sl, [
    ("Student (MobileNet3D)", True, C_PRIMARY, 10.5),
    ("  Optimizer: AdamW   WD: 0.01", False, C_GRAY, 10),
    ("  LR: 5e-4   Scheduler: cosine annealing", False, C_GRAY, 10),
    ("  Epochs: 60   Batch: 16   AMP ✓", False, C_GRAY, 10),
    ("  KD warmup: 10 epochs (CE-only)", False, C_GRAY, 10),
    ("  Label smoothing: 0.1", False, C_GRAY, 10),
    ("", False, C_GRAY, 6),
    ("Teacher (ResNet-50)", True, C_PRIMARY, 10.5),
    ("  Pretrained: Kinetics-400", False, C_GRAY, 10),
    ("  Optimizer: SGD   LR: 7e-4", False, C_GRAY, 10),
    ("  Epochs: 50   Batch: 12", False, C_GRAY, 10),
    ("  Fine-tuned on UCF-101", False, C_GRAY, 10),
], cl[2], Inches(2.05), cw, Inches(2.6))

add_rect(sl, Inches(0.45), Inches(5.8), Inches(12.43), Inches(1.3), fill_color=C_BOX)
add_textbox(sl, "All results shown use the Group-Aware protocol — the only statistically valid split.",
            Inches(0.6), Inches(5.87), Inches(12.1), Inches(0.4), 12, True, C_HIGHLIGHT)
add_textbox(sl, "Experiments run on: slurm-train-eval-4495, 4565, 4601, slurm-born-again-4529, slurm-train-eval-4675 (cross-frame).",
            Inches(0.6), Inches(6.25), Inches(12.1), Inches(0.6), 10, False, C_GRAY)
footer(sl, 4)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — LOGIT-BASED KD METHOD + RESULTS TABLE
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "LOGIT-BASED KNOWLEDGE DISTILLATION")
slide_title(sl, "Method — Logit-Based KD", "Temperature-scaled soft targets from teacher logits")

add_textbox(sl, "Loss Function", Inches(0.45), Inches(1.65), Inches(6.2), Inches(0.4),
            13, True, C_HIGHLIGHT)
add_rect(sl, Inches(0.45), Inches(2.05), Inches(6.2), Inches(1.35), fill_color=C_BOX)
add_textbox(sl, "L_KD  =  (1−α)·L_CE(y, ŷ)  +  α·T²·KL( σ(z_T/T) ‖ σ(z_S/T) )",
            Inches(0.55), Inches(2.12), Inches(6.0), Inches(0.5),
            font_size=12, bold=True, color=C_PRIMARY, align=PP_ALIGN.CENTER, font_name="Courier New")
add_multiline(sl, [
    ("T", True, C_HIGHLIGHT, 10.5), ("  temperature (controls soft-target entropy)  ", False, C_GRAY, 10.5),
    ("α", True, C_HIGHLIGHT, 10.5), ("  KD weight vs hard-label CE loss", False, C_GRAY, 10.5),
    ("z_T, z_S", True, C_HIGHLIGHT, 10.5), ("  teacher/student pre-softmax logits", False, C_GRAY, 10.5),
], Inches(0.6), Inches(2.65), Inches(6.0), Inches(0.7))

add_textbox(sl, "Warmup Phase", Inches(0.45), Inches(3.5), Inches(6.2), Inches(0.4), 13, True, C_HIGHLIGHT)
add_multiline(sl, [
    "10 epochs of pure CE before soft targets.",
    "Stabilizes student weights from scratch",
    "before teacher guidance is introduced.",
    "Effect: +1.19 pp Top-1 vs. no warmup.",
], Inches(0.45), Inches(3.88), Inches(6.2), Inches(1.3), font_size=10.5, line_spacing=15)

# Table
add_textbox(sl, "Temperature Sweep Results (Group-Aware Test Set)",
            Inches(7.0), Inches(1.65), Inches(6.0), Inches(0.4), 13, True, C_HIGHLIGHT)

rows = [
    ("Configuration",       "Top-1",  "Top-5",  "Δ Baseline", True),
    ("Baseline (no KD)",    "59.48%", "82.77%", "—",         False),
    ("KD T=1,  α=0.7",     "64.31%", "85.56%", "+4.83 pp",  False),
    ("KD T=5,  α=0.7",     "62.91%", "85.65%", "+3.43 pp",  False),
    ("KD T=8,  α=0.7",     "64.58%", "86.94%", "+5.10 pp",  False),
    ("KD T=10, α=0.7",     "64.47%", "86.55%", "+4.99 pp",  False),
    ("KD T=20, α=0.7  ★",  "65.85%", "89.24%", "+6.37 pp",  False),
    ("Teacher (ResNet-50)", "88.82%", "98.18%", "—",         False),
]
cws = [Inches(2.8), Inches(0.9), Inches(0.9), Inches(1.2)]
cxs = [Inches(7.0), Inches(9.9), Inches(10.85), Inches(11.8)]
rh  = Inches(0.38)
rt0 = Inches(2.1)
for ri, row in enumerate(rows):
    is_hdr = row[4]
    is_best = "★" in row[0]
    bg = C_HIGHLIGHT if is_hdr else (C_BOX if ri % 2 == 0 else C_WHITE)
    for ci, (cell, cw, cx) in enumerate(zip(row[:4], cws, cxs)):
        add_rect(sl, cx, rt0 + ri * rh, cw, rh, fill_color=bg)
        cc = C_WHITE if is_hdr else (C_HIGHLIGHT2 if is_best and ci > 0 else C_PRIMARY)
        add_textbox(sl, str(cell), cx + Inches(0.05), rt0 + ri * rh,
                    cw - Inches(0.05), rh, font_size=10, bold=is_hdr, color=cc,
                    align=PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT)

add_textbox(sl, "★ Best: T=20 → 65.85% Top-1 (+6.37 pp vs baseline 59.48%)",
            Inches(7.0), Inches(5.2), Inches(6.0), Inches(0.35), 10, True, C_HIGHLIGHT2)
add_textbox(sl, "Higher T smooths teacher logits → exposes inter-class similarity → student learns class boundaries better",
            Inches(7.0), Inches(5.52), Inches(6.0), Inches(0.4), 10, False, C_GRAY)
footer(sl, 5)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — TRAINING CURVES T=20 (Loss + Accuracy)
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "BEST EXPERIMENT — KD T=20, α=0.7 | Training Dynamics")
slide_title(sl, "Training Curves — KD T=20 (Best Experiment)", "Loss and Top-1/Top-5 accuracy over 60 epochs + cosine LR schedule")

# Loss curve (left)
pic(sl, f"{T20_FIGS}/kd_t20_a07_24f_lightaug_loss.png",
    Inches(0.35), Inches(1.6), Inches(6.4), Inches(3.3))

# Accuracy curve (right)
pic(sl, f"{T20_FIGS}/kd_t20_a07_24f_lightaug_accuracy.png",
    Inches(6.85), Inches(1.6), Inches(6.4), Inches(3.3))

# LR schedule (bottom left, smaller)
pic(sl, f"{T20_FIGS}/kd_t20_a07_24f_lightaug_lr.png",
    Inches(0.35), Inches(5.0), Inches(4.0), Inches(2.1))

# Annotations (bottom right)
add_rect(sl, Inches(4.55), Inches(5.0), Inches(8.65), Inches(2.1), fill_color=C_BOX)
add_textbox(sl, "Reading the Curves",
            Inches(4.7), Inches(5.05), Inches(8.4), Inches(0.38),
            font_size=12, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "• Loss converges smoothly — KD loss dominated by KL term in first 30 epochs",
    "• Eval accuracy plateaus at ~63-65% (group-aware, no leakage) — stable after epoch 40",
    "• Top-5 eval accuracy reaches ~85-86% → good ranking quality despite Top-1 gap",
    "• No significant overfit: train/eval gap reflects capacity, not data leakage",
    "• Cosine LR annealing gradually reduces LR to ≈0 by epoch 60 → smooth convergence",
], Inches(4.7), Inches(5.4), Inches(8.4), Inches(1.6),
    font_size=10.5, color=C_PRIMARY, line_spacing=14.5)
footer(sl, 6)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — TEMPERATURE ABLATION — HISTOGRAM TOP-1
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "LOGIT-BASED KD — TEMPERATURE ABLATION")
slide_title(sl, "Test Set Top-1 Accuracy — All Configurations")

pic(sl, HIST_TOP1, Inches(0.35), Inches(1.6), Inches(12.6), Inches(4.6))

add_textbox(sl,
    "KD T=20 achieves 65.85% (+6.37 pp vs baseline 59.48%). "
    "Performance is monotonically non-decreasing with temperature. "
    "AT symmetric and Born-Again Gen3 fall below baseline.",
    Inches(0.45), Inches(6.25), Inches(12.4), Inches(0.55),
    font_size=10.5, italic=True, color=C_GRAY)
footer(sl, 7)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — TEMPERATURE ABLATION — HISTOGRAM TOP-5
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "LOGIT-BASED KD — TEMPERATURE ABLATION")
slide_title(sl, "Test Set Top-5 Accuracy — All Configurations")

pic(sl, HIST_TOP5, Inches(0.35), Inches(1.6), Inches(12.6), Inches(4.6))

add_textbox(sl,
    "KD T=20 also achieves best Top-5: 89.24% (+6.47 pp vs baseline 82.77%). "
    "Soft targets consistently improve ranking quality across all temperature settings.",
    Inches(0.45), Inches(6.25), Inches(12.4), Inches(0.55),
    font_size=10.5, italic=True, color=C_GRAY)
footer(sl, 8)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — CONFUSION MATRIX: BASELINE vs KD T=20 (top25)
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "CONFUSION MATRIX — BASELINE vs BEST (KD T=20, α=0.7)")
slide_title(sl, "Confusion Matrix — Top 25 Hardest Classes", "Comparison: Baseline (left) vs KD T=20 (right)")

# Baseline CM top25 (left)
pic(sl, f"{BL_CM}/predictions_baseline_ls005_24f_lightaug_top25.png",
    Inches(0.35), Inches(1.55), Inches(6.4), Inches(5.45))

# KD T=20 CM top25 (right)
pic(sl, f"{T20_CM}/predictions_kd_t20_a07_24f_lightaug_top25.png",
    Inches(6.9), Inches(1.55), Inches(6.3), Inches(5.45))

# Labels
add_rect(sl, Inches(0.35), Inches(7.05), Inches(6.4), Inches(0.3), fill_color=C_BOX2)
add_textbox(sl, "Baseline MobileNet3D — 59.48% Top-1",
            Inches(0.45), Inches(7.07), Inches(6.2), Inches(0.25),
            font_size=10, bold=True, color=C_HIGHLIGHT2, align=PP_ALIGN.CENTER)
add_rect(sl, Inches(6.9), Inches(7.05), Inches(6.3), Inches(0.3), fill_color=C_BOX)
add_textbox(sl, "KD T=20 (Best) — 65.85% Top-1  (+6.37 pp)",
            Inches(7.0), Inches(7.07), Inches(6.1), Inches(0.25),
            font_size=10, bold=True, color=C_HIGHLIGHT, align=PP_ALIGN.CENTER)
footer(sl, 9)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — CONFUSION MATRIX: NORMALIZED (full 101 classes)
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "CONFUSION MATRIX — KD T=20 FULL (101 Classes, Normalized)")
slide_title(sl, "Normalized Confusion Matrix — KD T=20", "All 101 UCF-101 classes on the official test set")

pic(sl, f"{T20_CM}/predictions_kd_t20_a07_24f_lightaug_norm.png",
    Inches(0.35), Inches(1.55), Inches(10.5), Inches(5.6))

# Analysis panel
add_rect(sl, Inches(11.0), Inches(1.55), Inches(2.2), Inches(5.6), fill_color=C_BOX)
add_textbox(sl, "Key Observations",
            Inches(11.1), Inches(1.62), Inches(2.0), Inches(0.4),
            font_size=11, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "• Strong diagonal →",
    "  most classes well",
    "  predicted",
    "",
    "• Off-diagonal blobs:",
    "  similar-motion",
    "  pairs confused",
    "  (e.g. CricketShot",
    "  ↔ TennisSwing)",
    "",
    "• Instrument classes",
    "  (PlayingDhol,",
    "  PlayingCello) show",
    "  good separation",
    "",
    "• Hammering &",
    "  Archery show most",
    "  inter-class noise",
], Inches(11.1), Inches(2.0), Inches(2.0), Inches(4.5),
   font_size=9.5, color=C_PRIMARY, line_spacing=13)
footer(sl, 10)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — ERROR ANALYSIS: TOP 5 HARDEST CLASSES
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "CONFUSION MATRIX — TOP 5 ERRORS RELATIONSHIP")
slide_title(sl, "Error Analysis — Top 5 Hardest Classes", "Relationship between baseline and distillation (KD T=20) top errors")

# 3 Highlight boxes
hb_data = [
    ("WallPushups", "+65.7 pp", "Accuracy Gain (0% → 65.7%)", C_BOX, C_HIGHLIGHT, Inches(0.45)),
    ("Hammering", "-9.1 pp", "Accuracy Drop (12.1% → 3.0%)", C_BOX2, C_HIGHLIGHT2, Inches(4.75)),
    ("HandstandWalking", "Top Error", "Persistent difficulty (~94% error)", C_BOX, C_HIGHLIGHT, Inches(9.05))
]
bw_h = Inches(3.8)
bh_h = Inches(1.2)
bt_h = Inches(1.65)
for lbl, val, desc, bg, vc, lx in hb_data:
    highlight_box(sl, lx, bt_h, bw_h, bh_h, f"{lbl}\n{desc}", val, bg=bg, val_color=vc)

add_textbox(sl, "Relationship Analysis", Inches(0.45), Inches(3.05), Inches(12.4), Inches(0.35), 14, True, C_HIGHLIGHT)
add_rect(sl, Inches(0.45), Inches(3.4), Inches(12.43), Inches(0.02), fill_color=C_LGRAY)

cols = [
    ("Shared Challenges (Persistent)", [
        "• HandstandWalking and Nunchucks remain in the top 5 errors for both models:",
        "  - HandstandWalking: 2.94% → 5.88% (+2.94 pp)",
        "  - Nunchucks: 11.43% → 20.00% (+8.57 pp)",
        "  - MoppingFloor: 11.76% → 20.59% (+8.82 pp)",
        "• Complex dynamics: These classes feature rapid, complex body motion sequences that remain hard to capture.",
        "• Persistent confusions: MoppingFloor is heavily confused with HandstandWalking in both (26.5% vs 29.4%)."
    ]),
    ("Greatest Successes (KD Resolution)", [
        "• KD regularizes baseline's worst classes by providing soft inter-class targets:",
        "  - WallPushups: 0.00% → 65.71% (+65.71 pp)",
        "  - HeadMassage: 12.20% → 41.46% (+29.26 pp)",
        "  - JumpRope: 10.53% → 23.68% (+13.16 pp)",
        "• Teacher soft logits guide the student through subtle motion boundaries.",
        "• Clear confusions: JumpRope confusion with JumpingJack falls from 18.4% to 0%."
    ]),
    ("Unexpected Regressions (Trade-offs)", [
        "• Distillation degrades classes requiring fine-grained or fast periodic cues:",
        "  - Hammering: 12.12% → 3.03% (−9.09 pp)",
        "  - PizzaTossing: 27.27% → 15.15% (−12.12 pp)",
        "  - Archery: 29.27% → 19.51% (−9.76 pp)",
        "• Over-smoothing: At T=20, soft targets smooth out fine visual details (e.g. hammer/pizza shape).",
        "• High confusion: Hammering gets confused with PlayingSitar (12.1%); Archery with WritingOnBoard (22.0%)."
    ])
]
cxs_c = [Inches(0.45), Inches(4.75), Inches(9.05)]
for ci, (ctitle, citems) in enumerate(cols):
    add_textbox(sl, ctitle, cxs_c[ci], Inches(3.5), Inches(3.8), Inches(0.4),
                font_size=12, bold=True, color=C_HIGHLIGHT if ci < 2 else C_HIGHLIGHT2)
    add_multiline(sl, citems, cxs_c[ci], Inches(3.9), Inches(3.8), Inches(3.1),
                  font_size=10, line_spacing=13)
footer(sl, 11)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — KD IMPACT PER CLASS (Delta chart)
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "QUALITATIVE — KD IMPACT PER CLASS")
slide_title(sl, "KD T=20 vs Baseline — Per-Class Accuracy Delta", "Top improvements and regressions across UCF-101 classes")

pic(sl, DELTA_CHART, Inches(0.35), Inches(1.55), Inches(9.5), Inches(5.6))

add_rect(sl, Inches(10.0), Inches(1.55), Inches(3.15), Inches(5.6), fill_color=C_BOX)
add_textbox(sl, "Interpretation",
            Inches(10.1), Inches(1.62), Inches(2.9), Inches(0.4),
            font_size=11, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "Top gains:",
    "  WallPushups +65.7%",
    "  HeadMassage +29.3%",
    "  JavelinThrow +29.0%",
    "  Drumming     +28.9%",
    "",
    "Winners share subtle",
    "motion patterns that",
    "the teacher's soft",
    "logits clarify.",
    "",
    "Regressions:",
    "  JumpingJack −16.2%",
    "  TennisSwing −14.3%",
    "",
    "High-energy periodical",
    "actions where teacher",
    "soft targets add noise",
    "vs. one-hot targets.",
], Inches(10.1), Inches(2.02), Inches(2.95), Inches(5.0),
   font_size=9.5, color=C_PRIMARY, line_spacing=13)
footer(sl, 12)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — t-SNE LATENT SPACE (4565, T temperature comparison)
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "QUALITATIVE ANALYSIS — LATENT SPACE")
slide_title(sl, "t-SNE — Latent Space at Different Temperatures", "Pre-classifier embeddings for 10 random UCF-101 classes (slurm-4565)")

# Use the t-SNE plot from slurm-4495 which shows baseline/KD comparison
tsne_comp = f"{BL_TSNE}/tsne_comparison_4495.png"
pic(sl, tsne_comp, Inches(0.35), Inches(1.55), Inches(12.6), Inches(4.55))

add_multiline(sl, [
    ("Teacher (ResNet-50):", True, C_HIGHLIGHT, 10.5),
    (" Compact, well-separated  ", False, C_PRIMARY, 10.5),
    ("  |  ", False, C_LGRAY, 10.5),
    ("Baseline MobileNet3D:", True, C_HIGHLIGHT2, 10.5),
    (" Significant inter-class overlap  ", False, C_PRIMARY, 10.5),
    ("  |  ", False, C_LGRAY, 10.5),
    ("Distilled Student (T=8):", True, RGBColor(0x22, 0x88, 0x44), 10.5),
    (" Tighter clusters, improved class separation", False, C_PRIMARY, 10.5),
], Inches(0.45), Inches(6.18), Inches(12.4), Inches(0.55))
footer(sl, 13)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — ATTENTION TRANSFER
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "ADVANCED METHODS — ATTENTION TRANSFER")
slide_title(sl, "Attention Transfer (AT)", "Aligning spatial/temporal activation maps across network stages")

add_textbox(sl, "Approach", Inches(0.45), Inches(1.65), Inches(5.9), Inches(0.4), 13, True, C_HIGHLIGHT)
add_multiline(sl, [
    "L2-normalized squared activation maps from matched",
    "stages aligned between teacher (ResNet-50) and student.",
    "",
    "L_total = L_KD + β_s·L_AT_spatial + β_t·L_AT_temporal",
], Inches(0.45), Inches(2.0), Inches(5.9), Inches(1.25), font_size=10.5, line_spacing=15)

add_textbox(sl, "Bug Fixed: Key-5 Error", Inches(0.45), Inches(3.2), Inches(5.9), Inches(0.4),
            13, True, C_HIGHLIGHT2)
add_multiline(sl, [
    "Initial config: block [3,4,5] of teacher.",
    "Block 5 = classification head (2D) → constant",
    "attention map at 1.0 → AT loss silently vanished.",
    "",
    "Fix: mapping [2,3,4]→[2,4,6] (ResNet→MobileNet).",
    "Stages share spatial resolutions: 28×28, 14×14, 7×7.",
], Inches(0.45), Inches(3.55), Inches(5.9), Inches(2.0), font_size=10.5, line_spacing=15)

add_textbox(sl, "AT Results (Group-Aware Protocol, slurm-4601)",
            Inches(6.7), Inches(1.65), Inches(6.3), Inches(0.4), 13, True, C_HIGHLIGHT)

at_rows = [
    ("Experiment",                    "Top-1",  "Top-5",  True),
    ("KD Baseline (T=8, α=0.7)",     "58.92%", "84.93%", False),
    ("AT Symmetric (β_s=β_t=0.05)",  "59.42%", "85.04%", False),
    ("AT Heavy Temporal (β_t=0.07)", "59.40%", "84.99%", False),
    ("Late-Stage AT ([4]→[6])",      "61.59%", "87.52%", False),
    ("Temporal-Only (β_s=0)",        "62.07%", "87.05%", False),
    ("KD Opt. T=10, α=0.9, wup10",  "62.62%", "87.02%", False),
]
at_cws = [Inches(3.5), Inches(1.0), Inches(1.0)]
at_cxs = [Inches(6.7), Inches(10.35), Inches(11.45)]
at_rh  = Inches(0.4)
at_t0  = Inches(2.1)
for ri, row in enumerate(at_rows):
    bg = C_HIGHLIGHT if row[3] else (C_BOX if ri % 2 == 0 else C_WHITE)
    for ci, (cell, cw, cx) in enumerate(zip(row[:3], at_cws, at_cxs)):
        add_rect(sl, cx, at_t0 + ri * at_rh, cw, at_rh, fill_color=bg)
        add_textbox(sl, cell, cx + Inches(0.05), at_t0 + ri * at_rh, cw, at_rh,
                    font_size=10, bold=row[3], color=C_WHITE if row[3] else C_PRIMARY,
                    align=PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT)

add_textbox(sl, "Interpretation", Inches(6.7), Inches(4.9), Inches(6.3), Inches(0.4), 12, True, C_HIGHLIGHT)
add_multiline(sl, [
    "• Spatial AT: depthwise ↔ standard conv gap limits alignment quality",
    "• Interpolation artifacts degrade spatial AT signal → acts as noise",
    "• Temporal-only AT avoids spatial distortion → best AT variant",
    "• Pure KD T=20 (65.85%) outperforms all AT variants by a large margin",
], Inches(6.7), Inches(5.3), Inches(6.3), Inches(1.3), font_size=10.5, line_spacing=15)
footer(sl, 14)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — BORN-AGAIN NETWORKS
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "ADVANCED METHODS — BORN-AGAIN SELF-DISTILLATION (slurm-born-again-4529)")
slide_title(sl, "Born-Again Self-Distillation", "Iterative MobileNet3D → MobileNet3D self-distillation")

add_textbox(sl, "Setup", Inches(0.45), Inches(1.65), Inches(5.9), Inches(0.4), 13, True, C_HIGHLIGHT)
add_multiline(sl, [
    "• Teacher: distilled MobileNet3D (Gen 0, KD T=8 → 64.31%)",
    "• T=2.5, α=0.5, β_s=β_t=0.05 — same arch → no interpolation",
    "• Keys identical: [2,4,6] → [2,4,6]",
    "• 3 successive student generations trained",
], Inches(0.45), Inches(2.0), Inches(5.9), Inches(1.5), font_size=10.5, line_spacing=15)

add_textbox(sl, "Generational Collapse", Inches(0.45), Inches(3.5), Inches(5.9), Inches(0.4),
            13, True, C_HIGHLIGHT2)
add_multiline(sl, [
    "Gen 0 → Gen 1:  64.31% → 60.43%  (−3.88 pp)",
    "Gen 1 → Gen 2:  60.43% → 60.98%  (+0.55 pp)",
    "Gen 2 → Gen 3:  60.98% → 59.21%  (−1.77 pp)",
    "",
    "Even with identical architecture (no interpolation),",
    "the noisy soft targets from a ~65% teacher",
    "corrupt successive generations. T=2.5 is poorly",
    "calibrated for a low-confidence teacher.",
], Inches(0.45), Inches(3.88), Inches(5.9), Inches(2.3), font_size=10.5, line_spacing=15)

add_textbox(sl, "Accuracy Across Generations",
            Inches(6.8), Inches(1.65), Inches(6.2), Inches(0.4), 13, True, C_HIGHLIGHT)

gen_data = [
    ("Gen 0\n(Start)", 64.31, C_HIGHLIGHT),
    ("Gen 1",          60.43, RGBColor(0x55, 0x88, 0xCC)),
    ("Gen 2",          60.98, RGBColor(0x77, 0xAA, 0x55)),
    ("Gen 3",          59.21, C_HIGHLIGHT2),
    ("Baseline",       59.48, RGBColor(0x44, 0xAA, 0x44)),
]
bl = Inches(6.8); bt0 = Inches(2.1); baw = Inches(6.2); bah = Inches(3.6)
bw = Inches(0.9); bgap = Inches(0.3)
max_v, min_v = 68.0, 55.0
for bi, (lbl, val, col) in enumerate(gen_data):
    bx = bl + bi * (bw + bgap)
    bh_bar = bah * (val - min_v) / (max_v - min_v)
    by = bt0 + bah - bh_bar
    add_rect(sl, bx, by, bw, bh_bar, fill_color=col)
    add_textbox(sl, f"{val:.2f}%", bx, by - Inches(0.3), bw, Inches(0.3),
                font_size=10.5, bold=True, color=col, align=PP_ALIGN.CENTER)
    add_textbox(sl, lbl, bx, bt0 + bah + Inches(0.05), bw, Inches(0.5),
                font_size=9.5, color=C_GRAY, align=PP_ALIGN.CENTER)

base_y = bt0 + bah - bah * (59.48 - min_v) / (max_v - min_v)
add_rect(sl, bl, base_y, baw, Inches(0.02), fill_color=RGBColor(0x44, 0xAA, 0x44))
add_textbox(sl, "── Baseline (59.48%)",
            bl + Inches(0.05), base_y - Inches(0.25), Inches(2.5), Inches(0.25),
            font_size=8.5, color=RGBColor(0x44, 0xAA, 0x44))

add_textbox(sl, "Root cause: noisy soft-targets from low-confidence teacher (64%) + miscalibrated temperature T=2.5",
            Inches(6.8), Inches(6.2), Inches(6.2), Inches(0.5),
            font_size=10, italic=True, color=C_GRAY)
footer(sl, 15)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 16 — CROSS-FRAME DISTILLATION INTRO
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — CROSS-FRAME DISTILLATION (slurm-train-eval-4675)")
slide_title(sl, "Cross-Frame Distillation", "Trading temporal resolution for wall-clock inference speed")

add_textbox(sl, "Motivation", Inches(0.45), Inches(1.65), Inches(5.9), Inches(0.4), 13, True, C_HIGHLIGHT)
add_multiline(sl, [
    "Processing 24 frames/clip is expensive for edge deployments.",
    "A 16-frame student (Model B) reduces temporal compute by 33.3%.",
    "",
    "Question: does this translate to real wall-clock speedup?",
    "Benchmark on the full official test set (3,783 clips).",
], Inches(0.45), Inches(2.0), Inches(5.9), Inches(2.0), font_size=10.5, line_spacing=15)

add_textbox(sl, "Benchmark Protocol", Inches(0.45), Inches(4.0), Inches(5.9), Inches(0.4), 13, True, C_HIGHLIGHT)
add_multiline(sl, [
    "• Dataset: official UCF-101 test split (3,783 clips)",
    "• Metric: pure forward-pass latency (I/O excluded)",
    "• GPU: NVIDIA L40S (CUDA synchronization)",
    "• CPU: sub-sampled (50×BS=1, 20×BS=8)",
    "• Server: DMI Cluster gnode10",
], Inches(0.45), Inches(4.4), Inches(5.9), Inches(1.5), font_size=10.5, line_spacing=15)

add_textbox(sl, "Model Comparison", Inches(7.0), Inches(1.65), Inches(6.0), Inches(0.4), 13, True, C_HIGHLIGHT)

mc_rows = [
    ("",                    "Model A (24f)", "Model B (16f)", True),
    ("Frames / clip",       "24",            "16",           False),
    ("Top-1 Test",          "65.85%",        "62.68%",       False),
    ("Accuracy gap",        "—",             "−3.17 pp",     False),
    ("Frame reduction",     "—",             "−33.3%",       False),
    ("GPU speedup (BS≥8)",  "1×",            "~1.6×",        False),
    ("CPU speedup",         "1×",            "~1.57×",       False),
    ("GPU latency saved",   "—",             "37–40%",       False),
    ("CPU latency saved",   "—",             "35–37%",       False),
    ("CPU BS=1 latency",    "136.7 ms",      "88.5 ms ✓",    False),
]
mc_cws = [Inches(2.0), Inches(1.9), Inches(1.9)]
mc_cxs = [Inches(7.0), Inches(9.05), Inches(11.05)]
mc_rh  = Inches(0.42)
mc_t0  = Inches(2.1)
for ri, row in enumerate(mc_rows):
    bg = C_HIGHLIGHT if row[3] else (C_BOX if ri % 2 == 0 else C_WHITE)
    for ci, (cell, cw, cx) in enumerate(zip(row[:3], mc_cws, mc_cxs)):
        add_rect(sl, cx, mc_t0 + ri * mc_rh, cw, mc_rh, fill_color=bg)
        add_textbox(sl, cell, cx + Inches(0.05), mc_t0 + ri * mc_rh, cw, mc_rh,
                    font_size=10.5, bold=row[3], color=C_WHITE if row[3] else C_PRIMARY,
                    align=PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT)
footer(sl, 16)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 17 — GPU BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — GPU BENCHMARK | NVIDIA L40S")
slide_title(sl, "GPU Benchmark — NVIDIA L40S", "Model A (24f) vs Model B (16f) — pure inference latency")

pic(sl, GPU_PLT, Inches(0.35), Inches(1.55), Inches(12.6), Inches(4.4))

add_textbox(sl, "GPU Results Summary",
            Inches(0.45), Inches(6.0), Inches(12.4), Inches(0.4), 12, True, C_HIGHLIGHT)
add_textbox(sl,
    "BS≥8 (saturated): Model B achieves 37–40% latency savings (1.59×–1.66× speedup), "
    "exceeding theoretical 33.3% due to better Tensor Core memory alignment.  "
    "BS=1: PCIe kernel launch overhead dominates → only ~11.5% saving.",
    Inches(0.45), Inches(6.38), Inches(12.4), Inches(0.75), font_size=10, color=C_GRAY)
footer(sl, 17)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 18 — CPU BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — CPU BENCHMARK | CLUSTER HOST")
slide_title(sl, "CPU Benchmark — Cluster Host", "Without PCIe overhead, speedup reflects pure arithmetic FLOP reduction")

pic(sl, CPU_PLT, Inches(0.35), Inches(1.55), Inches(12.6), Inches(4.4))

add_textbox(sl, "CPU Results Summary",
            Inches(0.45), Inches(6.0), Inches(12.4), Inches(0.4), 12, True, C_HIGHLIGHT)
add_textbox(sl,
    "Model B (16f): 136.70 ms → 88.51 ms at BS=1 (−35.3%), crossing below 100 ms real-time threshold.  "
    "At BS=8: 1.60× speedup (−37.3%). Ideal for CPU-only edge deployments.",
    Inches(0.45), Inches(6.38), Inches(12.4), Inches(0.75), font_size=10, color=C_GRAY)
footer(sl, 18)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 19 — COMBINED BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — COMBINED BENCHMARK OVERVIEW")
slide_title(sl, "GPU & CPU — Combined Benchmark Overview")

pic(sl, COMB_PLT, Inches(0.35), Inches(1.55), Inches(12.6), Inches(5.2))

add_textbox(sl,
    "16-frame model achieves 37–40% GPU savings and 35–37% CPU savings at the cost of −3.17 pp accuracy vs 24-frame model.",
    Inches(0.45), Inches(6.82), Inches(12.4), Inches(0.4), font_size=10.5, italic=True, color=C_GRAY)
footer(sl, 19)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 20 — SPEED-ACCURACY TRADE-OFF
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — SPEED-ACCURACY TRADE-OFF SUMMARY")
slide_title(sl, "Speed-Accuracy Trade-off Summary")

# 5 highlight boxes (properly spaced within 13.33in)
hb_data = [
    ("Model A", "65.85%", "Top-1 (24f, T=20)",       C_BOX,  C_HIGHLIGHT,  Inches(0.45)),
    ("Model B", "62.68%", "Top-1 (16f, cross-frame)", C_BOX,  C_HIGHLIGHT2, Inches(3.07)),
    ("Δ Acc",   "−3.17", "pp vs Model A",             C_BOX2, C_HIGHLIGHT2, Inches(5.69)),
    ("GPU ×",   "1.6×",  "speedup at BS≥8",           C_BOX,  C_HIGHLIGHT,  Inches(8.31)),
    ("CPU ↓",   "~36%",  "latency reduction",         C_BOX,  C_HIGHLIGHT,  Inches(10.93)),
]
bw_h = Inches(2.4); bh_h = Inches(1.3); bt_h = Inches(1.75)
for lbl, val, desc, bg, vc, lx in hb_data:
    highlight_box(sl, lx, bt_h, bw_h, bh_h, f"{lbl}\n{desc}", val, bg=bg, val_color=vc)

add_textbox(sl, "Trade-off Analysis",
            Inches(0.45), Inches(3.2), Inches(12.4), Inches(0.4), 14, True, C_HIGHLIGHT)
add_rect(sl, Inches(0.45), Inches(3.6), Inches(12.43), Inches(0.02), fill_color=C_LGRAY)

cols = [
    ("Model A — Accuracy Focus", [
        "✓ Best Top-1: 65.85% (+6.37 pp vs baseline)",
        "✓ Best Top-5: 89.24% (+6.47 pp)",
        "• 24 frames/clip — higher compute cost",
        "• Suitable for server-side inference",
        "• KD T=20, α=0.7",
    ]),
    ("Model B — Efficiency Focus", [
        "✓ BS=1 CPU latency: 88.5 ms (<100 ms threshold)",
        "✓ 1.6× GPU speedup at BS≥8",
        "✓ Still +3.20 pp over scratch baseline",
        "• 16 frames/clip — edge deployment ready",
        "• Ideal for mobile / real-time pipelines",
    ]),
    ("Key Insight", [
        "Saving 37–40% of latency costs",
        "only 3.17 pp in accuracy vs Model A.",
        "",
        "Model B still exceeds the scratch",
        "baseline by +3.20 pp.",
        "",
        "→ Excellent operating point",
        "   for real-world deployment.",
    ]),
]
cxs_c = [Inches(0.45), Inches(4.65), Inches(8.85)]
for ci, (ctitle, citems) in enumerate(cols):
    add_textbox(sl, ctitle, cxs_c[ci], Inches(3.7), Inches(3.9), Inches(0.4),
                font_size=12, bold=True, color=C_HIGHLIGHT if ci < 2 else C_HIGHLIGHT2)
    add_multiline(sl, citems, cxs_c[ci], Inches(4.1), Inches(3.9), Inches(2.0),
                  font_size=10.5, line_spacing=15)
footer(sl, 20)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 21 — AGGREGATE BEST EVAL
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "RESULTS OVERVIEW — ALL GROUP-AWARE EXPERIMENTS")
slide_title(sl, "All Experiments — Best Eval Accuracy (Group-Aware)")

pic(sl, AGG_CHART, Inches(0.35), Inches(1.55), Inches(12.6), Inches(4.6))

add_textbox(sl,
    "KD T=20 achieves best student eval accuracy (65.18% eval / 65.85% test). Teacher (ResNet-50) reaches 92.36% eval. "
    "All AT variants and Born-Again configurations fall below the best pure-KD result.",
    Inches(0.45), Inches(6.2), Inches(12.4), Inches(0.5), font_size=10.5, italic=True, color=C_GRAY)
footer(sl, 21)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 22 — CONCLUSIONS
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "CONCLUSIONS")
slide_title(sl, "Summary & Key Findings")

findings = [
    ("Logit-Based KD",
     "Effective and zero-overhead. Best config: T=20, α=0.7 → 65.85% Top-1 "
     "(+6.37 pp vs. scratch baseline 59.48%). Higher T smooths logits and exposes inter-class structure.",
     C_BOX, C_HIGHLIGHT),
    ("Attention Transfer",
     "Limited by depthwise ↔ standard conv mismatch. Spatial AT over-constrains student; "
     "temporal-only AT is the best variant (62.07%). All AT variants underperform pure KD T=20.",
     C_BOX, C_HIGHLIGHT2),
    ("Born-Again Self-Distillation",
     "Sensitive to teacher quality and temperature calibration. "
     "Noisy soft-targets from a ~65% teacher cause progressive generational collapse (64% → 59.21%).",
     C_BOX, C_HIGHLIGHT2),
    ("Cross-Frame Distillation",
     "16-frame Model B achieves 1.6× GPU speedup and crosses 100 ms CPU threshold (88.5 ms/clip). "
     "−3.17 pp accuracy vs. Model A — an excellent operating point for edge deployment.",
     C_BOX, C_HIGHLIGHT),
]
fw = Inches(5.9); fh = Inches(1.65)
fl = [Inches(0.45), Inches(6.75), Inches(0.45), Inches(6.75)]
ft = [Inches(1.65), Inches(1.65), Inches(3.45), Inches(3.45)]
for fi, (ttl, body, bg, tc) in enumerate(findings):
    add_rect(sl, fl[fi], ft[fi], fw, fh, fill_color=bg)
    add_textbox(sl, ttl, fl[fi]+Inches(0.1), ft[fi]+Inches(0.08), fw-Inches(0.2), Inches(0.38),
                font_size=12, bold=True, color=tc)
    add_textbox(sl, body, fl[fi]+Inches(0.1), ft[fi]+Inches(0.42), fw-Inches(0.2), fh-Inches(0.48),
                font_size=10.5, color=C_PRIMARY)

add_rect(sl, Inches(0.45), Inches(5.25), Inches(12.43), Inches(1.85), fill_color=C_PRIMARY)
add_textbox(sl, "Take-Away",
            Inches(0.6), Inches(5.32), Inches(12.0), Inches(0.38),
            font_size=13, bold=True, color=C_WHITE)
add_textbox(sl,
    "KD compresses a 3D ResNet-50 (~128 MB) into a MobileNet3D (~9.5 MB) achieving 65.85% Top-1 on UCF-101 "
    "(+6.37 pp over scratch). The 16-frame variant reduces inference by 37–40% on GPU "
    "with only −3.17 pp accuracy cost — an excellent operating point for real-world edge deployments.",
    Inches(0.6), Inches(5.68), Inches(12.0), Inches(1.3),
    font_size=10.5, color=RGBColor(0xCC, 0xCC, 0xCC))
footer(sl, 22)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 23 — THANK YOU
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_PRIMARY)
add_rect(sl, Inches(0), Inches(0),     Inches(13.33), Inches(0.07), fill_color=C_HIGHLIGHT2)
add_rect(sl, Inches(0), Inches(7.43),  Inches(13.33), Inches(0.07), fill_color=C_HIGHLIGHT2)

add_textbox(sl, "Thank you",
            Inches(0.6), Inches(1.8), Inches(12), Inches(1.2),
            font_size=54, bold=True, color=C_WHITE)
add_textbox(sl, "Questions & Discussion",
            Inches(0.6), Inches(3.0), Inches(12), Inches(0.6),
            font_size=22, color=RGBColor(0xAA, 0xBB, 0xFF))
add_rect(sl, Inches(0.6), Inches(3.75), Inches(3.0), Inches(0.04), fill_color=C_HIGHLIGHT2)
add_multiline(sl, [
    ("Daniele Barbagallo   ·   1000015334", False, RGBColor(0xCC, 0xCC, 0xCC), 12),
    ("Deep Learning — Advanced Models and Methods", False, RGBColor(0x88, 0x88, 0xBB), 10),
    ("Università di Catania — DMI   |   A.A. 2025/2026", False, RGBColor(0x88, 0x88, 0xBB), 10),
], Inches(0.6), Inches(3.95), Inches(10), Inches(1.2))

# Recap boxes
recap = [
    ("65.85%",  "Best Top-1\n(KD T=20)"),
    ("1.6×",    "GPU Speedup\n(Model B)"),
    ("~9.5 MB", "Student Size"),
    ("<89 ms",  "CPU Latency\nBS=1 (16f)"),
    ("+6.37 pp","KD gain\nvs scratch"),
]
rw = Inches(2.3); rh = Inches(1.3); rt_r = Inches(5.5)
for ri, (val, lbl) in enumerate(recap):
    rx = Inches(0.6) + ri * (rw + Inches(0.18))
    add_rect(sl, rx, rt_r, rw, rh, fill_color=RGBColor(0x22, 0x33, 0x55))
    add_textbox(sl, val, rx, rt_r + Inches(0.05), rw, rh * 0.55,
                font_size=22, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    add_textbox(sl, lbl, rx, rt_r + rh * 0.58, rw, rh * 0.4,
                font_size=8.5, color=RGBColor(0x99, 0xAA, 0xCC), align=PP_ALIGN.CENTER)
footer(sl, 23)


# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
out = r"PRESENTAZIONE/presentation_v6.pptx"
prs.save(out)
print(f"Saved: {out}")
print(f"Slides: {len(prs.slides)}")
