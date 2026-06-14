"""
Generate a professional PowerPoint presentation for the KD_Project.
Output: PRESENTAZIONE/presentation_v4.pptx (16:9, dark theme)
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os


# ── Color palette ────────────────────────────────────────────────────
BG         = RGBColor(18, 18, 28)
CARD_BG    = RGBColor(28, 28, 40)
CARD_EDGE  = RGBColor(50, 50, 68)
WHITE      = RGBColor(240, 240, 248)
BODY       = RGBColor(195, 195, 210)
MUTED      = RGBColor(140, 140, 158)
TEAL       = RGBColor(56, 210, 190)
GREEN      = RGBColor(60, 210, 150)
AMBER      = RGBColor(245, 190, 50)
VIOLET     = RGBColor(160, 135, 245)


# ── Helpers ──────────────────────────────────────────────────────────
def _bg(slide):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = BG


def _header(slide, title, tag=None):
    if tag:
        b = slide.shapes.add_textbox(Inches(.8), Inches(.35), Inches(11), Inches(.3))
        p = b.text_frame.paragraphs[0]
        p.text = tag.upper()
        p.font.name = "Segoe UI"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = TEAL
        b.text_frame.margin_left = b.text_frame.margin_right = 0
        b.text_frame.margin_top = b.text_frame.margin_bottom = 0
    b = slide.shapes.add_textbox(Inches(.8), Inches(.6), Inches(11), Inches(.75))
    p = b.text_frame.paragraphs[0]
    p.text = title
    p.font.name = "Segoe UI"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    b.text_frame.margin_left = b.text_frame.margin_right = 0
    b.text_frame.margin_top = b.text_frame.margin_bottom = 0
    # thin accent line
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(.8), Inches(1.45), Inches(1.6), Pt(2.5))
    ln.fill.solid()
    ln.fill.fore_color.rgb = TEAL
    ln.line.fill.background()


def _card(slide, l, t, w, h):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = CARD_BG
    s.line.color.rgb = CARD_EDGE
    s.line.width = Pt(1)
    s.adjustments[0] = 0.04
    return s


def _tb(slide, l, t, w, h, wrap=True):
    """Add a textbox and return its text_frame."""
    b = slide.shapes.add_textbox(l, t, w, h)
    tf = b.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    return tf


def _p(tf, text, sz=13, color=BODY, bold=False, after=8, first=False):
    """Append (or use first) paragraph."""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.text = text
    p.font.name = "Segoe UI"
    p.font.size = Pt(sz)
    p.font.bold = bold
    p.font.color.rgb = color
    p.space_after = Pt(after)
    p.line_spacing = 1.2
    return p


def _section(tf, text):
    """Card section header in teal."""
    _p(tf, text, sz=11, color=TEAL, bold=True, after=10, first=(len(tf.paragraphs) == 1 and tf.paragraphs[0].text == ""))


def _table(slide, rows, cols, data, l, t, w, h,
           header_bg=RGBColor(38, 38, 55), cell_bg=CARD_BG,
           highlights=None):
    """Create a styled table. highlights is a dict (r,c) -> RGBColor."""
    highlights = highlights or {}
    ts = slide.shapes.add_table(rows, cols, l, t, w, h)
    tbl = ts.table
    for r in range(rows):
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_bg if r == 0 else cell_bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = data[r][c]
            p.font.name = "Segoe UI"
            p.font.size = Pt(10)
            p.font.bold = (r == 0) or (c == 0)
            p.font.color.rgb = highlights.get((r, c), WHITE if r == 0 else BODY)
            p.alignment = PP_ALIGN.CENTER
    return tbl


def _img_fit(slide, path, l, t, max_w, max_h):
    """Insert image preserving aspect ratio, centered in box."""
    if not os.path.exists(path):
        print(f"Warning: image path not found: {path}")
        return
    pic = slide.shapes.add_picture(path, l, t)
    ow, oh = pic.width, pic.height
    s = min(max_w / ow, max_h / oh)
    nw, nh = int(ow * s), int(oh * s)
    pic.width, pic.height = nw, nh
    pic.left = int(l + (max_w - nw) / 2)
    pic.top  = int(t + (max_h - nh) / 2)


# ── Slide builders ───────────────────────────────────────────────────

def slide_title(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(.35), Inches(7.5))
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()

    tf = _tb(slide, Inches(1.0), Inches(1.4), Inches(7.2), Inches(4.5))
    _p(tf, "DEEP LEARNING — ADVANCED MODELS AND METHODS", 11, TEAL, True, 20, first=True)
    _p(tf, "Knowledge Distillation\nfor Mobile Action Recognition", 40, WHITE, True, 16)
    _p(tf, "Compressing a 3D ResNet-50 teacher into a MobileNet3D student\nfor efficient video action recognition on UCF-101", 15, BODY, after=0)

    _card(slide, Inches(8.9), Inches(1.1), Inches(3.6), Inches(5.2))
    tf = _tb(slide, Inches(9.15), Inches(1.35), Inches(3.1), Inches(4.8))
    _section(tf, "PROJECT INFO")
    for label, val in [
        ("Author",       "Daniele Barbagallo"),
        ("Student ID",   "1000015334"),
        ("Advisor",      "Prof. Antonino Furnari"),
        ("Course",       "Deep Learning (AMM)"),
        ("University",   "Università di Catania — DMI"),
        ("Year",         "2025 / 2026"),
    ]:
        _p(tf, label.upper(), 8, MUTED, True, 1)
        _p(tf, val, 13, WHITE, True, 12)


def slide_objectives(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Objective & Capacity Gap", "Introduction")

    tf = _tb(slide, Inches(.8), Inches(1.7), Inches(5.8), Inches(5))
    _section(tf, "PROBLEM STATEMENT")
    _p(tf, "Goal: compress a large 3D ResNet-50 teacher (~128 MB, pretrained on Kinetics-400) into a MobileNet3D student (~9.5 MB) for on-device video action recognition.", after=10, first=False)
    _p(tf, "The student trained from scratch with standard cross-entropy reaches only 59.48% Test Top-1 — a 29.3 pp gap with the teacher (88.82%).", after=10)
    _p(tf, "Knowledge Distillation transfers the teacher's soft probability distributions to the student, closing part of this gap without changing the student architecture or increasing its size.", after=0)

    _card(slide, Inches(7.2), Inches(1.7), Inches(5.3), Inches(4.5))
    tf2 = _tb(slide, Inches(7.4), Inches(2.0), Inches(4.9), Inches(.5))
    _section(tf2, "REFERENCE MODELS")
    _table(slide, 3, 4,
        [["Model", "Size", "Test Top-1", "Test Top-5"],
         ["Teacher (3D ResNet-50)", "~128 MB", "88.82 %", "98.18 %"],
         ["Student Baseline",      "~9.5 MB", "59.48 %", "82.77 %"]],
        Inches(7.4), Inches(2.7), Inches(4.9), Inches(1.5),
        highlights={(1,0): TEAL, (2,0): VIOLET})

    tf3 = _tb(slide, Inches(7.4), Inches(4.5), Inches(4.9), Inches(1.5))
    _p(tf3, "The teacher is pretrained and fine-tuned; the student is trained from scratch. All experiments use the official UCF-101 test set for final evaluation.", 11, MUTED, after=0, first=True)


def slide_setup(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Experimental Setup", "Methodology")

    tf = _tb(slide, Inches(.8), Inches(1.7), Inches(5.6), Inches(5))
    _section(tf, "TRAINING CONFIGURATION")
    bullets = [
        "Optimizer: AdamW (weight decay 0.01)",
        "Scheduler: cosine annealing (LR 5e-4 student, 7e-4 teacher)",
        "Epochs: 60 (+ 10-epoch CE-only warmup for KD runs)",
        "Batch size: 16 (student) / 12 (teacher)",
        "Mixed precision (AMP), label smoothing 0.1, gradient clipping 1.0",
        "Augmentation: resize short-side 128 → crop 112×112, horizontal flip, color jitter 0.1, random erasing 0.05",
    ]
    for b in bullets:
        _p(tf, f"•  {b}", 13, BODY, after=7)

    _card(slide, Inches(6.9), Inches(1.7), Inches(5.6), Inches(4.6))
    tf2 = _tb(slide, Inches(7.15), Inches(2.0), Inches(5.1), Inches(4.0))
    _section(tf2, "GROUP-AWARE EVALUATION SPLIT")
    _p(tf2, "UCF-101 videos are organized into groups originating from the same source clip. A naïve random split leaks correlated frames between train and eval sets, inflating metrics.", 13, BODY, after=10)
    _p(tf2, "The training set was split at the group level (regex on video IDs) to create a clean evaluation partition (20 %), preventing data leakage.", 13, WHITE, bold=True, after=10)
    _p(tf2, "This reduced validation accuracy from fictitiously high values to a realistic 60-65 % range, ensuring reliable model selection.", 13, BODY, after=0)


def slide_kd_main(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Logit-Based Knowledge Distillation", "Knowledge Distillation")

    tf = _tb(slide, Inches(.8), Inches(1.7), Inches(5.6), Inches(5))
    _section(tf, "METHOD")
    _p(tf, "The distillation loss combines standard cross-entropy on hard labels with KL divergence on temperature-softened teacher logits.", 13, BODY, after=10)
    _p(tf, "A 10-epoch warmup phase (CE-only) stabilizes student weights before soft targets are introduced.", 13, BODY, after=10)
    _section(tf, "STANDARD CONFIGURATION")
    _p(tf, "T = 8,  α = 0.7  →  Test Top-1: 64.31 %  (+4.83 pp vs baseline)", 13, WHITE, bold=True, after=10)
    _p(tf, "The best overall result was obtained at higher temperatures (see ablation slides). The distilled student retains the same 9.5 MB size and inference latency.", 13, BODY, after=0)

    _card(slide, Inches(6.9), Inches(1.7), Inches(5.6), Inches(4.6))
    tf2 = _tb(slide, Inches(7.15), Inches(2.0), Inches(5.1), Inches(.5))
    _section(tf2, "RESULTS COMPARISON")
    _table(slide, 4, 5,
        [["Model",            "Val Top-1", "Test Top-1", "Test Top-5", "Δ vs BL"],
         ["Teacher (R50)",    "92.36 %",   "88.82 %",   "98.18 %",   "—"],
         ["Baseline",         "59.18 %",   "59.48 %",   "82.77 %",   "ref."],
         ["KD (T=8, α=0.7)",  "64.57 %",   "64.31 %",   "87.68 %",   "+4.83"]],
        Inches(7.15), Inches(2.7), Inches(5.1), Inches(2.2),
        highlights={(3,2): TEAL, (3,4): GREEN})

    tf3 = _tb(slide, Inches(7.15), Inches(5.2), Inches(5.1), Inches(1))
    _p(tf3, "Distillation closes roughly one-sixth of the teacher-student gap, with zero size or latency overhead.", 11, MUTED, after=0, first=True)


def slide_histogram_top1(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Top-1 Test Accuracy", "Knowledge Distillation")

    img_path = "results/Train-eval-test-split (group-aware)/distillation_histogram_test_top1.png"
    _card(slide, Inches(.8), Inches(1.7), Inches(11.73), Inches(4.3))
    if os.path.exists(img_path):
        _img_fit(slide, img_path, Inches(0.95), Inches(1.85), Inches(11.43), Inches(4.0))
    else:
        tf_err = _tb(slide, Inches(1.0), Inches(3.5), Inches(11.3), Inches(1))
        _p(tf_err, f"[{img_path} not found]", 13, AMBER, True, 0, True)

    # Caption at the bottom
    tf_cap = _tb(slide, Inches(.8), Inches(6.15), Inches(11.73), Inches(0.85))
    _p(tf_cap, "Key Insights: Overall Top-1 test accuracy across configurations. The distilled student peaks at 65.85% with T = 20 (+6.37 percentage points vs. baseline 59.48%). This confirms that higher temperatures smooth teacher logits, helping the lower-capacity student learn class boundaries. Conversely, spatial Attention Transfer (AT) and Born-Again Network (BAN) self-distillation suffer from structural mismatches and error propagation, leading to regressions.", 12, BODY, first=True)


def slide_histogram_top5(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Top-5 Test Accuracy", "Knowledge Distillation")

    img_path = "results/Train-eval-test-split (group-aware)/distillation_histogram_test_top5.png"
    _card(slide, Inches(.8), Inches(1.7), Inches(11.73), Inches(4.3))
    if os.path.exists(img_path):
        _img_fit(slide, img_path, Inches(0.95), Inches(1.85), Inches(11.43), Inches(4.0))
    else:
        tf_err = _tb(slide, Inches(1.0), Inches(3.5), Inches(11.3), Inches(1))
        _p(tf_err, f"[{img_path} not found]", 13, AMBER, True, 0, True)

    # Caption at the bottom
    tf_cap = _tb(slide, Inches(.8), Inches(6.15), Inches(11.73), Inches(0.85))
    _p(tf_cap, "Key Insights: Overall Top-5 test accuracy. Standard KD at T = 20 achieves the highest performance at 89.24% (+6.47 pp vs. baseline 82.77%). Similar to the Top-1 trends, soft logit targets consistently outperform scratch baselines across sweeps. Restricting the student with spatial attention matching or relying on noise-prone uncalibrated self-distillation (BAN) fails to yield competitive Top-5 results.", 12, BODY, first=True)


def slide_temp(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Temperature Ablation (α = 0.7)", "Knowledge Distillation")

    # Left card - Table
    _card(slide, Inches(.8), Inches(1.7), Inches(5.8), Inches(4.6))
    tf = _tb(slide, Inches(1.0), Inches(2.0), Inches(5.4), Inches(.5))
    _section(tf, "TEMPERATURE SWEEP")

    _table(slide, 6, 5,
        [["T",          "Train Acc", "Val Top-1", "Test Top-1", "Δ vs BL"],
         ["1",          "98.44 %",   "62.75 %",   "62.91 %",   "+3.43"],
         ["5",          "95.87 %",   "63.12 %",   "62.07 %",   "+2.59"],
         ["8",          "97.59 %",   "64.57 %",   "64.31 %",   "+4.83"],
         ["10",         "98.28 %",   "64.85 %",   "64.47 %",   "+4.99"],
         ["20  (best)", "99.05 %",   "65.18 %",   "65.85 %",   "+6.37"]],
        Inches(1.0), Inches(2.6), Inches(5.4), Inches(2.8),
        highlights={(5,3): TEAL, (5,4): GREEN, (5,0): TEAL})

    # Right card - Text Box (reintroduced for visual consistency)
    _card(slide, Inches(6.9), Inches(1.7), Inches(5.6), Inches(4.6))
    tf2 = _tb(slide, Inches(7.15), Inches(2.0), Inches(5.1), Inches(4.0))
    _section(tf2, "ANALYSIS")
    _p(tf2, "Performance improves monotonically with temperature. At T = 20 the student reaches 65.85 % Test Top-1, a +6.37 pp gain over the baseline.", 13, WHITE, bold=True, after=12)
    _p(tf2, "Higher temperatures smooth the teacher's output distribution, exposing inter-class similarity structure that a lower-capacity student can approximate more effectively.", 13, BODY, after=12)
    _p(tf2, "At T = 1 the softmax outputs are nearly one-hot, which provides minimal additional signal beyond the hard labels.", 13, BODY, after=0)


def slide_at(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Attention Transfer", "Advanced Methods")

    tf = _tb(slide, Inches(.8), Inches(1.7), Inches(5.6), Inches(5))
    _section(tf, "APPROACH & DEBUGGING")
    _p(tf, "Spatial and temporal attention maps (L2-normalized squared activations) are aligned between teacher and student at matched network stages.", 13, BODY, after=10)
    _p(tf, "An initial configuration targeted block 5 of the teacher, which corresponds to the classification head (2D tensor). This produced constant attention maps and silently nullified the AT loss.", 13, AMBER, bold=True, after=10)
    _p(tf, "After correcting the mapping to [2,3,4] → [2,4,6], the AT loss became active but still resulted in a performance regression.", 13, BODY, after=10)
    _section(tf, "CONFIGURATIONS TESTED")
    _p(tf, "•  Symmetric: β_s = 0.05, β_t = 0.05", 13, BODY, after=4)
    _p(tf, "•  Temporal-Only: β_s = 0.0, β_t = 0.10", 13, BODY, after=0)

    # Increased card height to 5.0 to prevent textbox overflow
    _card(slide, Inches(6.9), Inches(1.7), Inches(5.6), Inches(5.0))
    tf2 = _tb(slide, Inches(7.15), Inches(2.0), Inches(5.1), Inches(.5))
    _section(tf2, "RESULTS")
    _table(slide, 4, 3,
        [["Config",            "Val Top-1", "Δ vs KD"],
         ["KD (T=8, no AT)",   "64.57 %",   "ref."],
         ["AT Symmetric",      "58.25 %",   "−6.32"],
         ["AT Temporal-Only",  "59.04 %",   "−5.53"]],
        Inches(7.15), Inches(2.6), Inches(5.1), Inches(1.6),
        highlights={(2,2): AMBER, (3,2): AMBER})

    tf3 = _tb(slide, Inches(7.15), Inches(4.4), Inches(5.1), Inches(2.0))
    _section(tf3, "INTERPRETATION")
    _p(tf3, "The spatial capacity gap between standard 3D convolutions (teacher) and depthwise separable convolutions (student) makes it structurally difficult for the student to reproduce the teacher's attention patterns.", 12, BODY, after=8)
    _p(tf3, "The additional AT loss acts as an over-constraint, degrading performance rather than helping convergence.", 12, BODY, after=0)


def slide_ban(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Born-Again Networks", "Advanced Methods")

    tf = _tb(slide, Inches(.8), Inches(1.7), Inches(5.6), Inches(5))
    _section(tf, "SETUP")
    _p(tf, "Iterative self-distillation with homogeneous architecture (MobileNet3D → MobileNet3D) across 3 generations.", 13, BODY, after=8)
    _p(tf, "Starting point (Gen 0): logit-distilled student (KD T = 8, 64.31 % Test Top-1).", 13, BODY, after=8)
    _p(tf, "Mode: distillation_at — logit KD combined with Attention Transfer using identical feature keys [2, 4, 6] on both sides (no interpolation needed).", 13, BODY, after=8)
    _p(tf, "Hyper-parameters: T = 2.5, α = 0.5, β_s = β_t = 0.05, 50 epochs, AdamW, cosine scheduler.", 13, BODY, after=10)

    # Balanced layout: card height 5.0, analysis text moved under the table on the right
    _card(slide, Inches(6.9), Inches(1.7), Inches(5.6), Inches(5.0))
    tf2 = _tb(slide, Inches(7.15), Inches(2.0), Inches(5.1), Inches(.5))
    _section(tf2, "GENERATIONAL COLLAPSE")
    _table(slide, 5, 4,
        [["Gen",   "Teacher",       "Test Top-1", "Δ vs Gen 0"],
         ["Gen 0", "ResNet-50",     "64.31 %",    "ref."],
         ["Gen 1", "Gen 0 Student", "60.98 %",    "−3.33"],
         ["Gen 2", "Gen 1 Student", "60.43 %",    "−3.88"],
         ["Gen 3", "Gen 2 Student", "59.21 %",    "−5.10"]],
        Inches(7.15), Inches(2.5), Inches(5.1), Inches(1.9),
        highlights={(2,3): AMBER, (3,3): AMBER, (4,3): AMBER})

    tf3 = _tb(slide, Inches(7.15), Inches(4.65), Inches(5.1), Inches(1.9))
    _section(tf3, "ANALYSIS")
    _p(tf3, "Despite perfectly matching features (exact alignment, no spatial interpolation), a progressive collapse is observed: Gen 3 drops below baseline.", 12, BODY, after=6, first=True)
    _p(tf3, "The student-teacher (64%) provides noisy soft targets. Each successive generation inherits and amplifies these errors, degrading the training signal.", 12, BODY, after=0)


def slide_tsne(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Latent Space — t-SNE Visualization", "Qualitative Analysis")

    # Centered card aligned to slide margins (0.80 inches)
    img = "results/Train-eval-test-split (group-aware)/Training/slurm-train-eval-4565/tsne_plots/tsne_temperature_t20.png"
    _card(slide, Inches(.8), Inches(1.7), Inches(11.73), Inches(3.28))
    if os.path.exists(img):
        _img_fit(slide, img, Inches(0.95), Inches(1.85), Inches(11.43), Inches(2.98))
    else:
        tf_err = _tb(slide, Inches(1.0), Inches(2.8), Inches(11.3), Inches(1))
        _p(tf_err, f"[{img} not found]", 13, AMBER, True, 0, True)

    # Bottom text box aligned with standard margins
    tf = _tb(slide, Inches(.8), Inches(5.15), Inches(11.73), Inches(1.55))
    _section(tf, "EMBEDDING ANALYSIS")
    _p(tf, "t-SNE projections of pre-classifier embeddings (10 random classes). Teacher clusters are compact and well-separated. Baseline student exhibits significant inter-class overlap.", 13, BODY, after=6)
    _p(tf, "The best distilled student (KD T = 20) shows tighter clustering and improved class separation compared to the baseline, confirming effective structural knowledge transfer.", 13, WHITE, bold=True, after=0)


def slide_crossframe_intro(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Cross-Frame Distillation", "Efficiency")

    tf = _tb(slide, Inches(.8), Inches(1.7), Inches(5.6), Inches(5))
    _section(tf, "MOTIVATION")
    _p(tf, "Processing 24 frames per clip is computationally expensive for edge and mobile deployments.", 13, BODY, after=10)
    _p(tf, "A 16-frame student (Model B) reduces temporal computation by 33.3 % compared to the 24-frame model (Model A).", 13, WHITE, bold=True, after=10)
    _p(tf, "The benchmark measures whether this theoretical saving translates into real wall-clock speedup on GPU and CPU hardware.", 13, BODY, after=0)

    # Right card height adjusted to 4.60 to avoid overflow
    _card(slide, Inches(6.9), Inches(1.7), Inches(5.6), Inches(4.6))
    tf2 = _tb(slide, Inches(7.15), Inches(2.0), Inches(5.1), Inches(4.0))
    _section(tf2, "BENCHMARK PROTOCOL")
    _p(tf2, "•  Dataset: official UCF-101 test split (3 783 clips)", 13, BODY, after=8)
    _p(tf2, "•  Metric: pure forward-pass latency (I/O excluded)", 13, BODY, after=8)
    _p(tf2, "•  GPU: NVIDIA L40S with CUDA synchronization", 13, BODY, after=8)
    _p(tf2, "•  CPU: sub-sampled (50 batches BS=1, 20 batches BS=8) to avoid excessive runtime while preserving statistical accuracy", 13, BODY, after=8)
    _p(tf2, "•  Server: DMI Cluster (gnode10)", 13, BODY, after=0)


def slide_gpu_charts(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "GPU Benchmark — NVIDIA L40S", "Efficiency")

    _card(slide, Inches(.8), Inches(1.7), Inches(11.73), Inches(4.3))
    
    # Left and Right charts side-by-side inside card
    img_left = "results/Test_Set_Benchmark/individual_plots/gpu_latency.png"
    img_right = "results/Test_Set_Benchmark/individual_plots/gpu_throughput.png"
    
    if os.path.exists(img_left):
        _img_fit(slide, img_left, Inches(1.05), Inches(1.85), Inches(5.40), Inches(4.00))
    if os.path.exists(img_right):
        _img_fit(slide, img_right, Inches(6.85), Inches(1.85), Inches(5.40), Inches(4.00))

    # Bottom caption box
    tf_cap = _tb(slide, Inches(.8), Inches(6.15), Inches(11.73), Inches(0.85))
    _p(tf_cap, "Key Insights: GPU latency per batch (left) and throughput in clips/sec (right). At batch sizes >= 8, the GPU is saturated, and the 16-frame model (Model B) achieves a stable 1.59x–1.66x speedup (saving 37–40% of GPU compute time). At BS=1, the performance gain is limited to 1.13x due to PCIe kernel launch overhead dominance.", 12, BODY, first=True)


def slide_gpu_table(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "GPU Benchmark — NVIDIA L40S", "Efficiency")

    # Left Card
    _card(slide, Inches(.8), Inches(1.7), Inches(5.6), Inches(4.6))
    img_path = "results/Test_Set_Benchmark/individual_plots/gpu_latency_saved.png"
    if os.path.exists(img_path):
        _img_fit(slide, img_path, Inches(0.95), Inches(1.85), Inches(5.30), Inches(4.30))

    # Right Card
    _card(slide, Inches(6.9), Inches(1.7), Inches(5.6), Inches(4.6))
    tf2 = _tb(slide, Inches(7.15), Inches(2.0), Inches(5.1), Inches(.5))
    _section(tf2, "PERFORMANCE SUMMARY")

    _table(slide, 5, 5,
        [["BS",  "Model A (24f)", "Model B (16f)", "Speedup", "Saved"],
         ["1",   "4.54 ms",       "4.02 ms",       "1.13×",   "11.5 %"],
         ["8",   "1.03 ms",       "0.62 ms",       "1.65×",   "39.3 %"],
         ["16",  "1.18 ms",       "0.74 ms",       "1.59×",   "37.2 %"],
         ["32",  "1.37 ms",       "0.82 ms",       "1.66×",   "39.7 %"]],
        Inches(7.15), Inches(2.5), Inches(5.1), Inches(1.8),
        highlights={(1,3): GREEN, (1,4): GREEN,
                    (2,3): GREEN, (2,4): GREEN,
                    (3,3): GREEN, (3,4): GREEN,
                    (4,3): GREEN, (4,4): GREEN})

    tf3 = _tb(slide, Inches(7.15), Inches(4.5), Inches(5.1), Inches(1.6))
    _p(tf3, "At batch sizes ≥ 8 the GPU is fully saturated and Model B achieves 37-40 % latency savings, exceeding the linear 33 % frame reduction due to better alignment in memory.", 12, BODY, after=6, first=True)
    _p(tf3, "At BS = 1, PCIe kernel-launch overhead dominates, limiting the latency savings to ~11.5 %.", 12, BODY, after=0)


def slide_cpu_charts(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "CPU Benchmark — Cluster Host", "Efficiency") # Corrected title

    _card(slide, Inches(.8), Inches(1.7), Inches(11.73), Inches(4.3))
    
    # Left and Right CPU charts
    img_left = "results/Test_Set_Benchmark/individual_plots/cpu_latency.png"
    img_right = "results/Test_Set_Benchmark/individual_plots/cpu_throughput.png"
    
    if os.path.exists(img_left):
        _img_fit(slide, img_left, Inches(1.05), Inches(1.85), Inches(5.40), Inches(4.00))
    if os.path.exists(img_right):
        _img_fit(slide, img_right, Inches(6.85), Inches(1.85), Inches(5.40), Inches(4.00))

    # Bottom caption box
    tf_cap = _tb(slide, Inches(.8), Inches(6.15), Inches(11.73), Inches(0.85))
    _p(tf_cap, "Key Insights: CPU latency per batch (left) and throughput (right). Without PCIe bus delays, the speedup on CPU directly reflects the arithmetic FLOP reduction. The 16-frame model (Model B) achieves a stable 1.54x–1.60x speedup (saving 35–37% of latency) across batch sizes, making it highly suitable for real-time edge execution.", 12, BODY, first=True)


def slide_cpu_table(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "CPU Benchmark — Cluster Host", "Efficiency") # Corrected title

    # Left Card
    _card(slide, Inches(.8), Inches(1.7), Inches(5.6), Inches(4.6))
    img_path = "results/Test_Set_Benchmark/individual_plots/cpu_latency_saved.png"
    if os.path.exists(img_path):
        _img_fit(slide, img_path, Inches(0.95), Inches(1.85), Inches(5.30), Inches(4.30))

    # Right Card
    _card(slide, Inches(6.9), Inches(1.7), Inches(5.6), Inches(4.6))
    tf2 = _tb(slide, Inches(7.15), Inches(2.0), Inches(5.1), Inches(.5))
    _section(tf2, "PERFORMANCE SUMMARY")

    _table(slide, 3, 5,
        [["BS", "Model A (24f)", "Model B (16f)", "Speedup", "Saved"],
         ["1",  "136.70 ms",     "88.51 ms",      "1.54×",   "35.3 %"],
         ["8",  "113.51 ms",     "71.13 ms",      "1.60×",   "37.3 %"]],
        Inches(7.15), Inches(2.5), Inches(5.1), Inches(1.2),
        highlights={(1,3): GREEN, (1,4): GREEN,
                    (2,3): GREEN, (2,4): GREEN})

    tf3 = _tb(slide, Inches(7.15), Inches(3.9), Inches(5.1), Inches(2.2))
    _p(tf3, "Without PCIe overhead, the speedup directly reflects the arithmetic FLOP reduction, showing consistent gains at all batch sizes.", 12, BODY, after=6, first=True)
    _p(tf3, "Single-clip latency drops from 137 ms to 89 ms, crossing below the 100 ms threshold for real-time video processing, making Model B ideal for CPU-only edge deployments.", 12, BODY, after=0)


def slide_conclusion(prs, layout):
    slide = prs.slides.add_slide(layout)
    _bg(slide); _header(slide, "Conclusions", "Summary")

    tf = _tb(slide, Inches(.8), Inches(1.7), Inches(5.6), Inches(5))
    _section(tf, "SPEED-ACCURACY TRADE-OFF")
    _p(tf, "Model A (24 f, KD T = 20):  65.85 % Test Top-1  (+6.37 pp vs baseline)", 13, BODY, after=8)
    _p(tf, "Model B (16 f, cross-frame):  62.68 % Test Top-1", 13, BODY, after=8)
    _p(tf, "Saving 37-40 % latency costs only 3.17 pp in accuracy compared to Model A.  Model B still exceeds the scratch baseline by +3.20 pp.", 13, WHITE, bold=True, after=0)

    _card(slide, Inches(6.9), Inches(1.7), Inches(5.6), Inches(4.6))
    tf2 = _tb(slide, Inches(7.15), Inches(2.0), Inches(5.1), Inches(4))
    _section(tf2, "KEY FINDINGS")
    _p(tf2, "•  Logit-based KD is effective and zero-overhead; higher temperatures better bridge large capacity gaps.", 13, BODY, after=10)
    _p(tf2, "•  Attention Transfer is limited by the architectural mismatch between standard and depthwise convolutions.", 13, BODY, after=10)
    _p(tf2, "•  Born-Again self-distillation is sensitive to temperature calibration and prone to generational collapse without an oracle teacher.", 13, BODY, after=10)
    _p(tf2, "•  Cross-frame distillation offers an effective latency-accuracy trade-off for real-time edge deployment.", 13, WHITE, bold=True, after=0)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    prs = Presentation()
    prs.slide_width  = Inches(13.333)
    prs.slide_height = Inches(7.5)
    bl = prs.slide_layouts[6]  # blank slide layout

    slide_title(prs, bl)
    slide_objectives(prs, bl)
    slide_setup(prs, bl)
    slide_kd_main(prs, bl)
    slide_histogram_top1(prs, bl)
    slide_histogram_top5(prs, bl)
    slide_temp(prs, bl)
    slide_at(prs, bl)
    slide_ban(prs, bl)
    slide_tsne(prs, bl)
    slide_crossframe_intro(prs, bl)
    slide_gpu_charts(prs, bl)
    slide_gpu_table(prs, bl)
    slide_cpu_charts(prs, bl)
    slide_cpu_table(prs, bl)
    slide_conclusion(prs, bl)

    out_dir = "PRESENTAZIONE"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "presentation_v4.pptx")
    prs.save(out)
    print(f"Saved -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
