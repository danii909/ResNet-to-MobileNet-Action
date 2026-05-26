"""
presentation_v7.pptx
Generated automatically based on user's v5 structure with schematic updates.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.chart import XL_LABEL_POSITION

# Colors
C_BG        = RGBColor(0xFF, 0xFF, 0xFF)
C_PRIMARY   = RGBColor(0x1A, 0x1A, 0x2E)
C_HIGHLIGHT = RGBColor(0x0F, 0x3D, 0x91)
C_HIGHLIGHT2= RGBColor(0xE5, 0x48, 0x00)
C_GRAY      = RGBColor(0x55, 0x55, 0x55)
C_LGRAY     = RGBColor(0xDD, 0xDD, 0xDD)
C_BOX       = RGBColor(0xF0, 0xF4, 0xFF)
C_BOX2      = RGBColor(0xFF, 0xF3, 0xE0)
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
C_SUCCESS   = RGBColor(0x28, 0xA7, 0x45)
C_ERROR     = RGBColor(0xDC, 0x35, 0x45)

# Paths
BASE   = r"results/Train-eval-test-split (group-aware)"
BENCH  = r"results/Test_Set_Benchmark"
GA     = f"{BASE}/Training"

T20_FIGS = f"{GA}/slurm-train-eval-4565/kd_t20_a07_24f_lightaug/figures"
T20_CM   = f"{BASE}/Confusion-Matrix/kd_t20_a07_24f_lightaug"
T20_TSNE = f"{GA}/slurm-train-eval-4565/tsne_plots"
BL_FIGS  = f"{GA}/slurm-train-eval-4495/baseline_ls005_24f_lightaug/figures"
BL_CM    = f"{BASE}/Confusion-Matrix/baseline_ls005_24f_lightaug"
BL_TSNE  = f"{GA}/slurm-train-eval-4495/tsne_plots"

HIST_TOP1   = f"{BASE}/distillation_histogram_test_top1.png"
HIST_TOP5   = f"{BASE}/distillation_histogram_test_top5.png"
COMP_CHART  = f"{BASE}/distillation_comparison.png"
DELTA_CHART = f"{BASE}/figures/delta_accuracy_kd_vs_baseline.png"

GPU_PLT  = f"{BASE}/figures/gpu_benchmark_plots.png"
CPU_PLT  = f"{BASE}/figures/cpu_benchmark_plots.png"
COMB_PLT = f"{BASE}/figures/combined_benchmark_plots.png"

AT_TSNE_SYM = f"{GA}/slurm-train-eval-4601/tsne_plots/tsne_teacher_baseline_distill_at_symmetric.png"
AT_TSNE_TMP = f"{GA}/slurm-train-eval-4601/tsne_plots/tsne_teacher_baseline_distill_at_temporal.png"

W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H
TOTAL_SLIDES = 23

def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def set_bg(slide, color=C_BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_rect(slide, left, top, width, height, fill_color=None, line_color=None):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1.5)
    else:
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
    try:
        return slide.shapes.add_picture(path, left, top, width, height)
    except:
        return add_rect(slide, left, top, width, height, fill_color=C_LGRAY)

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
# SLIDE 3 — DATASET & EVALUATION SPLIT
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EXPERIMENTAL SETUP")
slide_title(sl, "Dataset & Group-Aware Split")

# Dataset Left
add_rect(sl, Inches(0.45), Inches(1.7), Inches(4.0), Inches(5.0), fill_color=C_BOX)
add_textbox(sl, "UCF-101 Dataset", Inches(0.6), Inches(1.8), Inches(3.7), Inches(0.4), font_size=16, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "• 101 action classes",
    "• 13,320 video clips (YouTube)",
    "• 24 frames/clip (112×112)",
    "• Test split: 3,783 clips"
], Inches(0.6), Inches(2.4), Inches(3.7), Inches(1.2), font_size=14, line_spacing=24)

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

add_textbox(sl, "Impact", Inches(4.8), Inches(4.8), Inches(4.2), Inches(0.4), font_size=14, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "Evaluation set metrics dropped from fictitiously high values to a realistic",
    "60-65% validation range, exposing the true generalization gap and",
    "ensuring reliable hyperparameter tuning."
], Inches(5.0), Inches(5.2), Inches(4.0), Inches(1.0), font_size=12, line_spacing=16)

chart_data = CategoryChartData()
chart_data.categories = ['Train', 'Eval', 'Test']
chart_data.add_series('Clips', (7630, 1907, 3783))

x, y, cx, cy = Inches(9.0), Inches(4.2), Inches(3.5), Inches(2.3)
chart = sl.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, cx, cy, chart_data
).chart
chart.has_legend = False

footer(sl, 3)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — TRAINING CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EXPERIMENTAL SETUP")
slide_title(sl, "Training Configuration & Light Augmentation")

add_rect(sl, Inches(0.45), Inches(1.7), Inches(5.8), Inches(5.0), fill_color=C_BOX)
add_textbox(sl, "Model Training Protocols", Inches(0.6), Inches(1.8), Inches(5.5), Inches(0.4), font_size=16, bold=True, color=C_HIGHLIGHT)

add_multiline(sl, [
    ("Student (MobileNet3D):", True, C_PRIMARY, 14),
    "• Optimizer: AdamW",
    "• Learning Rate: 5e-4 (Cosine Schedule)",
    "• Epochs: 60",
    "• Batch Size: 16 (with AMP)",
    "• Label Smoothing: 0.1",
    "",
    ("Teacher (ResNet-50):", True, C_PRIMARY, 14),
    "• Optimizer: SGD",
    "• Learning Rate: 7e-4",
    "• Epochs: 50"
], Inches(0.6), Inches(2.5), Inches(5.5), Inches(3.0), font_size=13, line_spacing=20)

add_rect(sl, Inches(6.5), Inches(1.7), Inches(6.3), Inches(5.0), fill_color=C_BOX2)
add_textbox(sl, "Light Augmentation (lightaug)", Inches(6.7), Inches(1.8), Inches(5.9), Inches(0.4), font_size=16, bold=True, color=C_HIGHLIGHT2)

add_multiline(sl, [
    "To combat overfitting caused by the small dataset size (13k clips) and the fact that the MobileNet3D student is trained from scratch:",
    "",
    "A 'light augmentation' pipeline was introduced:",
    "• Random Horizontal Flip",
    "• Random Crop",
    "• Color Jitter (Brightness/Contrast/Saturation)",
    "",
    "This pipeline significantly helped the student generalize better to the unseen validation and test sets."
], Inches(6.7), Inches(2.5), Inches(5.9), Inches(3.0), font_size=14, line_spacing=22)

footer(sl, 4)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — LOGIT-BASED KD
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
footer(sl, 5)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — KD TEMPERATURE ABLATION Top-1
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "LOGIT-BASED KD — TEMPERATURE ABLATION")
slide_title(sl, "Test Set Top-1 Accuracy — All Configurations")
pic(sl, HIST_TOP1, Inches(2.6), Inches(1.7), Inches(8), Inches(4.5))
add_textbox(sl, "KD T=20 achieves 65.85% (+6.37 pp vs baseline 59.48%). Performance is monotonically non-decreasing with temperature. AT symmetric and Born-Again Gen3 fall below baseline.",
            Inches(0.45), Inches(6.5), Inches(12.4), Inches(0.5), font_size=12, color=C_PRIMARY)
footer(sl, 6)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — KD TEMPERATURE ABLATION Top-5
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "LOGIT-BASED KD — TEMPERATURE ABLATION")
slide_title(sl, "Test Set Top-5 Accuracy — All Configurations")
pic(sl, HIST_TOP5, Inches(2.6), Inches(1.7), Inches(8), Inches(4.5))
add_textbox(sl, "KD T=20 also achieves best Top-5: 89.24% (+6.47 pp vs baseline 82.77%). Soft targets consistently improve ranking quality across all temperature settings.",
            Inches(0.45), Inches(6.5), Inches(12.4), Inches(0.5), font_size=12, color=C_PRIMARY)
footer(sl, 7)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — KD TEMPERATURE ABLATION EXPLANATION
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

footer(sl, 8)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — t-SNE
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "QUALITATIVE ANALYSIS — LATENT SPACE")
slide_title(sl, "t-SNE Visualization", "Pre-classifier embeddings for 10 random UCF-101 classes (KD T=20)")
pic(sl, f"{T20_TSNE}/tsne_comparison_4565.png", Inches(0.2), Inches(1.7), Inches(12.9), Inches(4.5))
add_textbox(sl, "Teacher (ResNet-50): Compact, well-separated | Baseline MobileNet3D: Significant inter-class overlap | Distilled Student: Tighter clusters, improved class separation",
            Inches(0.45), Inches(6.5), Inches(12.4), Inches(0.5), font_size=12, color=C_PRIMARY)
footer(sl, 9)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — TRAINING CURVES
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "BEST EXPERIMENT — KD T=20 | Training Dynamics")
slide_title(sl, "Training Curves — KD T=20", "Loss and Top-1/Top-5 accuracy over 60 epochs + cosine LR schedule")
pic(sl, f"{T20_FIGS}/training_curves.png", Inches(0.2), Inches(1.8), Inches(9.8), Inches(5.0))

add_rect(sl, Inches(10.2), Inches(1.8), Inches(2.8), Inches(4.5), fill_color=C_BOX)
add_textbox(sl, "Overfitting Analysis", Inches(10.3), Inches(1.9), Inches(2.6), Inches(0.4), font_size=14, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "Overfitting is observed in the curves.",
    "",
    "This is due to the small size of the UCF-101 dataset and because the student MobileNet3D is trained from scratch (unlike the pre-trained ResNet-50 teacher).",
    "",
    "To combat this, 'lightaug' (light data augmentation) was introduced.",
    "",
    "This helped the student generalize better to the validation set, even though it resulted in a slightly higher training loss."
], Inches(10.3), Inches(2.4), Inches(2.6), Inches(3.8), font_size=12, line_spacing=16)
footer(sl, 10)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — CONFUSION MATRIX
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "CONFUSION MATRIX — KD T=20 FULL")
slide_title(sl, "Confusion Matrix — KD T=20", "All 101 UCF-101 classes on the official test set")
pic(sl, f"{T20_CM}/confusion_matrix_normalized.png", Inches(0.2), Inches(1.7), Inches(5.5), Inches(5.5))

add_rect(sl, Inches(6.0), Inches(1.8), Inches(6.8), Inches(5.0), fill_color=C_WHITE, line_color=C_HIGHLIGHT)
add_rect(sl, Inches(6.0), Inches(1.8), Inches(6.8), Inches(0.6), fill_color=C_HIGHLIGHT)
add_textbox(sl, "Analytical Considerations", Inches(6.2), Inches(1.9), Inches(6.4), Inches(0.4), font_size=16, bold=True, color=C_WHITE)

add_multiline(sl, [
    ("• Highly Diagonal Structure", True, C_PRIMARY, 14),
    "The strong diagonal visually confirms the solid overall performance (65.85% Top-1) and effective learning across the majority of the 101 classes.",
    "",
    ("• Persistent Confusions", True, C_PRIMARY, 14),
    "Actions with highly similar visual appearance or overlapping semantic context (e.g., HandstandWalking vs. MoppingFloor) remain challenging and form visible off-diagonal clusters.",
    "",
    ("• Success of Distillation", True, C_PRIMARY, 14),
    "KD successfully resolved boundaries for many previously confused classes. The teacher's soft targets effectively guided the student through subtle motion boundaries that were missed when training from scratch."
], Inches(6.3), Inches(2.7), Inches(6.2), Inches(3.8), font_size=13, line_spacing=18)
footer(sl, 11)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — CONFUSION MATRIX BASELINE VS KD T=20
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "CONFUSION MATRIX — BASELINE vs BEST (KD T=20)")
slide_title(sl, "Confusion Matrix — Top 25 Hardest Classes", "Comparison: Baseline (left) vs KD T=20 (right)")

add_rect(sl, Inches(0.45), Inches(1.7), Inches(6.0), Inches(5.0), fill_color=C_BOX)
pic(sl, f"{BL_CM}/confusion_matrix_top25.png", Inches(0.55), Inches(1.8), Inches(5.8), Inches(4.3))
add_textbox(sl, "Baseline MobileNet3D — 59.48% Top-1", Inches(0.45), Inches(6.2), Inches(6.0), Inches(0.4), font_size=14, bold=True, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(6.85), Inches(1.7), Inches(6.0), Inches(5.0), fill_color=C_BOX2)
pic(sl, f"{T20_CM}/confusion_matrix_top25.png", Inches(6.95), Inches(1.8), Inches(5.8), Inches(4.3))
add_textbox(sl, "KD T=20 (Best) — 65.85% Top-1  (+6.37 pp)", Inches(6.85), Inches(6.2), Inches(6.0), Inches(0.4), font_size=14, bold=True, align=PP_ALIGN.CENTER)
footer(sl, 12)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — TOP 5 ERRORS RELATIONSHIP
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "CONFUSION MATRIX — TOP 5 ERRORS RELATIONSHIP")
slide_title(sl, "Error Analysis — Top 5 Hardest Classes", "Relationship between baseline and distillation (KD T=20) top errors")

highlight_box(sl, Inches(0.45), Inches(1.7), Inches(3.9), Inches(1.1), "Accuracy Gain (0% → 65.7%)", "+65.7 pp", val_color=C_SUCCESS)
add_textbox(sl, "WallPushups", Inches(0.45), Inches(1.7), Inches(3.9), Inches(0.3), font_size=10, color=C_PRIMARY, bold=True, align=PP_ALIGN.CENTER)

highlight_box(sl, Inches(4.7), Inches(1.7), Inches(3.9), Inches(1.1), "Accuracy Drop (12.1% → 3.0%)", "-9.1 pp", val_color=C_ERROR)
add_textbox(sl, "Hammering", Inches(4.7), Inches(1.7), Inches(3.9), Inches(0.3), font_size=10, color=C_PRIMARY, bold=True, align=PP_ALIGN.CENTER)

highlight_box(sl, Inches(8.95), Inches(1.7), Inches(3.9), Inches(1.1), "Persistent difficulty (~94% error)", "Top Error", val_color=C_HIGHLIGHT2)
add_textbox(sl, "HandstandWalking", Inches(8.95), Inches(1.7), Inches(3.9), Inches(0.3), font_size=10, color=C_PRIMARY, bold=True, align=PP_ALIGN.CENTER)

add_textbox(sl, "Relationship Analysis", Inches(0.45), Inches(3.0), Inches(12.4), Inches(0.4), font_size=16, bold=True, color=C_HIGHLIGHT)

add_rect(sl, Inches(0.45), Inches(3.5), Inches(3.9), Inches(3.3), fill_color=C_BOX)
add_textbox(sl, "Shared Challenges", Inches(0.55), Inches(3.6), Inches(3.7), Inches(0.3), font_size=12, bold=True)
add_multiline(sl, [
    "HandstandWalking and Nunchucks remain in the top 5 errors for both models.",
    "These classes feature rapid, complex sequences that remain hard to capture.",
    "Persistent confusions: MoppingFloor is heavily confused with HandstandWalking in both."
], Inches(0.55), Inches(4.0), Inches(3.7), Inches(2.7), font_size=11)

add_rect(sl, Inches(4.7), Inches(3.5), Inches(3.9), Inches(3.3), fill_color=C_BOX)
add_textbox(sl, "Greatest Successes", Inches(4.8), Inches(3.6), Inches(3.7), Inches(0.3), font_size=12, bold=True)
add_multiline(sl, [
    "KD regularizes baseline's worst classes by providing soft targets (WallPushups 0%→65%).",
    "Teacher soft logits guide the student through subtle motion boundaries.",
    "Clear confusions like JumpRope vs JumpingJack fall from 18.4% to 0%."
], Inches(4.8), Inches(4.0), Inches(3.7), Inches(2.7), font_size=11)

add_rect(sl, Inches(8.95), Inches(3.5), Inches(3.9), Inches(3.3), fill_color=C_BOX)
add_textbox(sl, "Unexpected Regressions", Inches(9.05), Inches(3.6), Inches(3.7), Inches(0.3), font_size=12, bold=True)
add_multiline(sl, [
    "Distillation degrades classes requiring fine-grained or fast periodic cues (e.g. Hammering, PizzaTossing).",
    "Over-smoothing: At T=20, soft targets smooth out fine visual details like shape.",
    "Hammering gets confused with PlayingSitar."
], Inches(9.05), Inches(4.0), Inches(3.7), Inches(2.7), font_size=11)

footer(sl, 13)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — KD IMPACT PER CLASS
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "QUALITATIVE — KD IMPACT PER CLASS")
slide_title(sl, "KD T=20 vs Baseline — Per-Class Accuracy Delta", "Top improvements and regressions across UCF-101 classes")

pic(sl, DELTA_CHART, Inches(0.45), Inches(1.7), Inches(9.0), Inches(5.0))

add_rect(sl, Inches(9.7), Inches(1.7), Inches(3.2), Inches(5.0), fill_color=C_BOX)
add_textbox(sl, "Interpretation", Inches(9.8), Inches(1.8), Inches(3.0), Inches(0.4), font_size=14, bold=True, color=C_HIGHLIGHT)

add_multiline(sl, [
    ("Top gains:", True, C_SUCCESS, 11),
    "WallPushups +65.7%",
    "HeadMassage +29.3%",
    "JavelinThrow +29.0%",
    "Drumming +28.9%",
    "",
    "Winners share subtle motion patterns that the teacher's soft logits clarify.",
    "",
    ("Regressions:", True, C_ERROR, 11),
    "JumpingJack −16.2%",
    "TennisSwing −14.3%",
    "",
    "High-energy periodical actions where teacher soft targets add noise vs. one-hot targets."
], Inches(9.8), Inches(2.2), Inches(3.0), Inches(4.4), font_size=10, line_spacing=12)

footer(sl, 14)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — ATTENTION TRANSFER
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "ADVANCED METHODS — ATTENTION TRANSFER")
slide_title(sl, "Attention Transfer (AT)", "Aligning spatial/temporal activation maps across network stages")

# Approach Block
add_rect(sl, Inches(0.45), Inches(1.7), Inches(12.4), Inches(1.2), fill_color=C_BOX)
add_textbox(sl, "Approach", Inches(0.6), Inches(1.8), Inches(12.0), Inches(0.3), font_size=14, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "L2-normalized squared activation maps from matched stages aligned between teacher (ResNet-50) and student.",
    "Loss = L_KD + β_s·L_AT_spatial + β_t·L_AT_temporal"
], Inches(0.6), Inches(2.1), Inches(12.0), Inches(0.8), font_size=12, line_spacing=16)

# Bug Fix Block
add_rect(sl, Inches(0.45), Inches(3.1), Inches(12.4), Inches(1.4), fill_color=C_BOX2)
add_textbox(sl, "Bug Fixed: Key-5 Error", Inches(0.6), Inches(3.2), Inches(12.0), Inches(0.3), font_size=14, bold=True, color=C_HIGHLIGHT2)
add_multiline(sl, [
    "Initial config: block [3,4,5] of teacher. Block 5 = classification head (2D) → constant attention map at 1.0 → AT loss silently vanished.",
    "Fix: mapping [2,3,4]→[2,4,6] (ResNet→MobileNet). Stages share spatial resolutions: 28×28, 14×14, 7×7."
], Inches(0.6), Inches(3.5), Inches(12.0), Inches(0.8), font_size=12, line_spacing=16)

# Results Block
add_textbox(sl, "Results (Group-Aware Protocol, slurm-4601)", Inches(0.45), Inches(4.7), Inches(12.4), Inches(0.3), font_size=14, bold=True, color=C_PRIMARY)

add_rect(sl, Inches(0.45), Inches(5.1), Inches(3.9), Inches(1.4), fill_color=C_WHITE, line_color=C_LGRAY)
add_textbox(sl, "64.58%", Inches(0.45), Inches(5.3), Inches(3.9), Inches(0.5), font_size=32, bold=True, color=C_PRIMARY, align=PP_ALIGN.CENTER)
add_textbox(sl, "AT Symmetric (β_s=0.05, β_t=0.05)", Inches(0.45), Inches(5.9), Inches(3.9), Inches(0.4), font_size=11, color=C_GRAY, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(4.7), Inches(5.1), Inches(3.9), Inches(1.4), fill_color=C_WHITE, line_color=C_LGRAY)
add_textbox(sl, "65.03%", Inches(4.7), Inches(5.3), Inches(3.9), Inches(0.5), font_size=32, bold=True, color=C_PRIMARY, align=PP_ALIGN.CENTER)
add_textbox(sl, "AT Temporal-Only (β_s=0)", Inches(4.7), Inches(5.9), Inches(3.9), Inches(0.4), font_size=11, color=C_GRAY, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(8.95), Inches(5.1), Inches(3.9), Inches(1.4), fill_color=C_WHITE, line_color=C_HIGHLIGHT)
add_textbox(sl, "65.85%", Inches(8.95), Inches(5.3), Inches(3.9), Inches(0.5), font_size=32, bold=True, color=C_HIGHLIGHT, align=PP_ALIGN.CENTER)
add_textbox(sl, "Pure KD T=20", Inches(8.95), Inches(5.9), Inches(3.9), Inches(0.4), font_size=11, color=C_HIGHLIGHT, align=PP_ALIGN.CENTER)

footer(sl, 15)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 16 — ATTENTION TRANSFER t-SNE
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "ADVANCED METHODS — ATTENTION TRANSFER")
slide_title(sl, "Qualitative AT Analysis", "Latent space representations for AT Variants")

add_textbox(sl, "AT Symmetric", Inches(0.45), Inches(1.7), Inches(6.0), Inches(0.4), font_size=16, bold=True, align=PP_ALIGN.CENTER)
pic(sl, AT_TSNE_SYM, Inches(0.45), Inches(2.2), Inches(6.0), Inches(4.5))

add_textbox(sl, "AT Temporal", Inches(6.85), Inches(1.7), Inches(6.0), Inches(0.4), font_size=16, bold=True, align=PP_ALIGN.CENTER)
pic(sl, AT_TSNE_TMP, Inches(6.85), Inches(2.2), Inches(6.0), Inches(4.5))

add_rect(sl, Inches(0.45), Inches(6.8), Inches(12.4), Inches(0.5), fill_color=C_BOX)
add_textbox(sl, "Interpretation: Spatial AT alignment is limited by depthwise ↔ standard conv gap, introducing noise. Temporal-only AT avoids spatial distortion.", Inches(0.6), Inches(6.9), Inches(12.0), Inches(0.3), font_size=11, color=C_PRIMARY)

footer(sl, 16)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 17 — BORN-AGAIN
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "ADVANCED METHODS — BORN-AGAIN SELF-DISTILLATION (slurm-born-again-4529)")
slide_title(sl, "Born-Again Self-Distillation", "Iterative MobileNet3D → MobileNet3D self-distillation")

# Setup Block
add_rect(sl, Inches(0.45), Inches(1.7), Inches(5.8), Inches(1.5), fill_color=C_BOX)
add_textbox(sl, "Setup", Inches(0.6), Inches(1.8), Inches(5.5), Inches(0.3), font_size=14, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "• Teacher: distilled MobileNet3D (Gen 0, 64.31%)",
    "• Same arch → no interpolation needed",
    "• Keys identical: [2,4,6] → [2,4,6]",
    "• 3 successive student generations trained"
], Inches(0.6), Inches(2.1), Inches(5.5), Inches(1.0), font_size=11, line_spacing=14)

# Temperature Choice Block
add_rect(sl, Inches(0.45), Inches(3.4), Inches(5.8), Inches(1.7), fill_color=C_BOX2)
add_textbox(sl, "Temperature Choice (T=2.5)", Inches(0.6), Inches(3.5), Inches(5.5), Inches(0.3), font_size=14, bold=True, color=C_HIGHLIGHT2)
add_multiline(sl, [
    "Temperature was intentionally set very low (T=2.5) because the self-distillation teacher has a much lower confidence (~64%) compared to ResNet.",
    "A high temperature would have over-smoothed the already-smooth logits and destroyed remaining sharp boundaries."
], Inches(0.6), Inches(3.8), Inches(5.5), Inches(1.2), font_size=11, line_spacing=14)

# Results Block
add_rect(sl, Inches(6.5), Inches(1.7), Inches(6.3), Inches(3.4), fill_color=C_WHITE, line_color=C_HIGHLIGHT)
add_rect(sl, Inches(6.5), Inches(1.7), Inches(6.3), Inches(0.6), fill_color=C_HIGHLIGHT)
add_textbox(sl, "Generational Collapse", Inches(6.7), Inches(1.8), Inches(5.9), Inches(0.4), font_size=16, bold=True, color=C_WHITE)

add_multiline(sl, [
    "Gen 0 (Teacher): 64.31%",
    "Gen 1:  60.43%  (−3.88 pp)",
    "Gen 2:  60.98%  (+0.55 pp)",
    "Gen 3:  59.21%  (−1.77 pp)",
    "Baseline: 59.48%"
], Inches(6.7), Inches(2.5), Inches(5.9), Inches(1.5), font_size=14, bold=True, line_spacing=20)

add_textbox(sl, "Root cause: Even with T=2.5, noisy soft-targets from a low-confidence teacher corrupt successive generations.", Inches(0.45), Inches(5.4), Inches(12.4), Inches(0.6), font_size=14, color=C_ERROR, align=PP_ALIGN.CENTER, bold=True)

footer(sl, 17)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 18 — CROSS-FRAME INTRO
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — CROSS-FRAME DISTILLATION (slurm-train-eval-4675)")
slide_title(sl, "Cross-Frame Distillation", "Trading temporal resolution for wall-clock inference speed")

# Motivation
add_rect(sl, Inches(0.45), Inches(1.7), Inches(12.4), Inches(1.4), fill_color=C_BOX)
add_textbox(sl, "Motivation", Inches(0.6), Inches(1.8), Inches(12.0), Inches(0.4), font_size=16, bold=True, color=C_HIGHLIGHT)
add_multiline(sl, [
    "Processing 24 frames/clip is expensive for edge deployments.",
    "A 16-frame student (Model B) reduces temporal compute by 33.3%.",
    "Question: does this translate to real wall-clock speedup?"
], Inches(0.6), Inches(2.3), Inches(12.0), Inches(0.8), font_size=14, line_spacing=18)

# Schematized Comparison
add_textbox(sl, "Model Comparison", Inches(0.45), Inches(3.4), Inches(12.4), Inches(0.4), font_size=16, bold=True)

add_rect(sl, Inches(1.5), Inches(4.0), Inches(4.5), Inches(2.5), fill_color=C_WHITE, line_color=C_PRIMARY)
add_textbox(sl, "Model A (24f)", Inches(1.5), Inches(4.2), Inches(4.5), Inches(0.4), font_size=18, bold=True, align=PP_ALIGN.CENTER)
add_multiline(sl, [
    "• Frames / clip: 24",
    "• Top-1 Test: 65.85%",
    "• GPU latency: 1x baseline",
    "• CPU latency: 1x baseline"
], Inches(1.8), Inches(4.8), Inches(4.0), Inches(1.5), font_size=14, line_spacing=20)

add_rect(sl, Inches(7.33), Inches(4.0), Inches(4.5), Inches(2.5), fill_color=C_BOX2, line_color=C_HIGHLIGHT2)
add_textbox(sl, "Model B (16f)", Inches(7.33), Inches(4.2), Inches(4.5), Inches(0.4), font_size=18, bold=True, color=C_HIGHLIGHT2, align=PP_ALIGN.CENTER)
add_multiline(sl, [
    "• Frames / clip: 16 (-33.3%)",
    "• Top-1 Test: 62.68% (-3.17 pp)",
    "• GPU Speedup: ~1.6x (BS≥8)",
    "• CPU Speedup: ~1.57x"
], Inches(7.63), Inches(4.8), Inches(4.0), Inches(1.5), font_size=14, bold=True, color=C_PRIMARY, line_spacing=20)

footer(sl, 18)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 19 — GPU BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — GPU BENCHMARK | NVIDIA L40S")
slide_title(sl, "GPU Benchmark — NVIDIA L40S", "Model A (24f) vs Model B (16f) — pure inference latency")
pic(sl, GPU_PLT, Inches(1.2), Inches(1.7), Inches(11.0), Inches(4.5))
add_textbox(sl, "GPU Results Summary: BS≥8 (saturated): Model B achieves 37–40% latency savings (1.59×–1.66× speedup), exceeding theoretical 33.3% due to better Tensor Core memory alignment. BS=1: PCIe kernel launch overhead dominates → only ~11.5% saving.",
            Inches(0.45), Inches(6.5), Inches(12.4), Inches(0.5), font_size=11, color=C_PRIMARY)
footer(sl, 19)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 20 — CPU BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — CPU BENCHMARK | CLUSTER HOST")
slide_title(sl, "CPU Benchmark — Cluster Host", "Without PCIe overhead, speedup reflects pure arithmetic FLOP reduction")
pic(sl, CPU_PLT, Inches(1.2), Inches(1.7), Inches(11.0), Inches(4.5))
add_textbox(sl, "CPU Results Summary: Model B (16f): 136.70 ms → 88.51 ms at BS=1 (−35.3%), crossing below 100 ms real-time threshold. At BS=8: 1.60× speedup (−37.3%). Ideal for CPU-only edge deployments.",
            Inches(0.45), Inches(6.5), Inches(12.4), Inches(0.5), font_size=11, color=C_PRIMARY)
footer(sl, 20)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 21 — SPEED ACCURACY
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — SPEED-ACCURACY TRADE-OFF SUMMARY")
slide_title(sl, "Speed-Accuracy Trade-off Summary")

add_rect(sl, Inches(1.0), Inches(1.7), Inches(2.5), Inches(1.5), fill_color=C_BOX)
add_textbox(sl, "65.85%", Inches(1.0), Inches(2.0), Inches(2.5), Inches(0.5), font_size=32, bold=True, color=C_HIGHLIGHT, align=PP_ALIGN.CENTER)
add_textbox(sl, "Model A Top-1", Inches(1.0), Inches(2.6), Inches(2.5), Inches(0.4), font_size=11, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(4.0), Inches(1.7), Inches(2.5), Inches(1.5), fill_color=C_BOX)
add_textbox(sl, "62.68%", Inches(4.0), Inches(2.0), Inches(2.5), Inches(0.5), font_size=32, bold=True, color=C_PRIMARY, align=PP_ALIGN.CENTER)
add_textbox(sl, "Model B Top-1", Inches(4.0), Inches(2.6), Inches(2.5), Inches(0.4), font_size=11, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(7.0), Inches(1.7), Inches(2.5), Inches(1.5), fill_color=C_WHITE, line_color=C_LGRAY)
add_textbox(sl, "−3.17 pp", Inches(7.0), Inches(2.0), Inches(2.5), Inches(0.5), font_size=32, bold=True, color=C_ERROR, align=PP_ALIGN.CENTER)
add_textbox(sl, "Δ Acc vs Model A", Inches(7.0), Inches(2.6), Inches(2.5), Inches(0.4), font_size=11, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(10.0), Inches(1.7), Inches(2.5), Inches(1.5), fill_color=C_BOX2, line_color=C_HIGHLIGHT2)
add_textbox(sl, "1.6×", Inches(10.0), Inches(2.0), Inches(2.5), Inches(0.5), font_size=32, bold=True, color=C_SUCCESS, align=PP_ALIGN.CENTER)
add_textbox(sl, "GPU Speedup", Inches(10.0), Inches(2.6), Inches(2.5), Inches(0.4), font_size=11, align=PP_ALIGN.CENTER)

add_rect(sl, Inches(2.0), Inches(3.6), Inches(4.2), Inches(3.2), fill_color=C_BOX)
add_textbox(sl, "Model A — Accuracy Focus", Inches(2.2), Inches(3.8), Inches(3.8), Inches(0.4), font_size=14, bold=True)
add_multiline(sl, [
    "✓ Best Top-1: 65.85% (+6.37 pp)",
    "✓ Best Top-5: 89.24% (+6.47 pp)",
    "• 24 frames/clip — higher compute cost",
    "• Suitable for server-side inference",
    "• KD T=20, α=0.7"
], Inches(2.2), Inches(4.3), Inches(3.8), Inches(2.2), font_size=12, line_spacing=18)

add_rect(sl, Inches(7.13), Inches(3.6), Inches(4.2), Inches(3.2), fill_color=C_BOX2)
add_textbox(sl, "Model B — Efficiency Focus", Inches(7.33), Inches(3.8), Inches(3.8), Inches(0.4), font_size=14, bold=True)
add_multiline(sl, [
    "✓ BS=1 CPU latency: 88.5 ms (<100 ms)",
    "✓ 1.6× GPU speedup at BS≥8",
    "✓ Still +3.20 pp over scratch baseline",
    "• 16 frames/clip — edge deployment ready",
    "• Ideal for mobile / real-time pipelines"
], Inches(7.33), Inches(4.3), Inches(3.8), Inches(2.2), font_size=12, line_spacing=18)

footer(sl, 21)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 22 — CONCLUSIONS
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "CONCLUSIONS")
slide_title(sl, "Summary & Key Findings")

add_rect(sl, Inches(0.45), Inches(1.7), Inches(12.4), Inches(1.0), fill_color=C_BOX)
add_textbox(sl, "Logit-Based KD", Inches(0.6), Inches(1.8), Inches(3.0), Inches(0.4), font_size=14, bold=True)
add_textbox(sl, "Effective and zero-overhead. Best config: T=20, α=0.7 → 65.85% Top-1 (+6.37 pp vs. scratch baseline 59.48%). Higher T smooths logits and exposes inter-class structure.", Inches(3.6), Inches(1.8), Inches(9.0), Inches(0.8), font_size=12)

add_rect(sl, Inches(0.45), Inches(2.9), Inches(12.4), Inches(1.0), fill_color=C_BOX)
add_textbox(sl, "Attention Transfer", Inches(0.6), Inches(3.0), Inches(3.0), Inches(0.4), font_size=14, bold=True)
add_textbox(sl, "Limited by depthwise ↔ standard conv mismatch. Spatial AT over-constrains student; temporal-only AT is the best variant (65.03%). All AT variants underperform pure KD T=20.", Inches(3.6), Inches(3.0), Inches(9.0), Inches(0.8), font_size=12)

add_rect(sl, Inches(0.45), Inches(4.1), Inches(12.4), Inches(1.0), fill_color=C_BOX)
add_textbox(sl, "Born-Again Self-Distillation", Inches(0.6), Inches(4.2), Inches(3.0), Inches(0.4), font_size=14, bold=True)
add_textbox(sl, "Sensitive to teacher quality and temperature calibration. Noisy soft-targets from a ~64% teacher cause progressive generational collapse (64% → 59.21%).", Inches(3.6), Inches(4.2), Inches(9.0), Inches(0.8), font_size=12)

add_rect(sl, Inches(0.45), Inches(5.3), Inches(12.4), Inches(1.0), fill_color=C_BOX)
add_textbox(sl, "Cross-Frame Distillation", Inches(0.6), Inches(5.4), Inches(3.0), Inches(0.4), font_size=14, bold=True)
add_textbox(sl, "16-frame Model B achieves 1.6× GPU speedup and crosses 100 ms CPU threshold (88.5 ms/clip). −3.17 pp accuracy vs. Model A — an excellent operating point for edge deployment.", Inches(3.6), Inches(5.4), Inches(9.0), Inches(0.8), font_size=12)

footer(sl, 22)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 23 — THANK YOU
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_PRIMARY)

add_textbox(sl, "Thank you", Inches(0.6), Inches(2.0), Inches(12), Inches(1.2), font_size=54, bold=True, color=C_WHITE)
add_textbox(sl, "Questions & Discussion", Inches(0.6), Inches(3.2), Inches(12), Inches(0.8), font_size=32, color=RGBColor(0xAA, 0xBB, 0xFF))

add_multiline(sl, [
    ("Daniele Barbagallo   ·   1000015334", False, RGBColor(0xCC, 0xCC, 0xCC), 14),
    ("Deep Learning — Advanced Models and Methods", False, RGBColor(0x99, 0x99, 0xBB), 12),
    ("Università di Catania — DMI   |   A.A. 2025/2026", False, RGBColor(0x99, 0x99, 0xBB), 12),
], Inches(0.6), Inches(4.5), Inches(9), Inches(1.5), line_spacing=20)

add_rect(sl, Inches(0.6), Inches(4.2), Inches(3.0), Inches(0.04), fill_color=C_HIGHLIGHT2)

footer(sl, 23, 23)

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
out = r"PRESENTAZIONE/presentation_v7.pptx"
prs.save(out)
print(f"Salvato correttamente in: {out}")