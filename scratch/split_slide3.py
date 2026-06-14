import re

with open('scratch/make_presentation_v7.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Update TOTAL_SLIDES
text = text.replace('TOTAL_SLIDES = 22', 'TOTAL_SLIDES = 23')
text = text.replace('TOTAL_SLIDES = 23', 'TOTAL_SLIDES = 24')

# Replacement for Slide 3 logic
slide3_original = """# ════════════════════════════════════════════════════════════════════════════
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

footer(sl, 3)"""

slide3_and_4_new = """# ════════════════════════════════════════════════════════════════════════════
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

footer(sl, 4)"""

if slide3_original in text:
    text = text.replace(slide3_original, slide3_and_4_new)
else:
    print("Warning: Slide 3 original text not matched exactly.")

# Update slide numbers for slides 4 onwards.
# They were previously SLIDE 4, SLIDE 5 etc.
# We will match `# SLIDE (\d+)` and `footer\(sl, (\d+)\)`

def replace_comment(m):
    num = int(m.group(1))
    if num >= 4:
        return f"# SLIDE {num+1}"
    return m.group(0)

def replace_footer(m):
    num = int(m.group(1))
    if num >= 4:
        # Wait, slide 4 became 5, etc.
        return f"footer(sl, {num+1})"
    return m.group(0)

def replace_footer_2(m):
    num = int(m.group(1))
    num2 = int(m.group(2))
    if num >= 4:
        return f"footer(sl, {num+1}, {num2+1})"
    return m.group(0)

# Slide comments
text = re.sub(r'# SLIDE (\d+)', replace_comment, text)

# Footers with one argument
text = re.sub(r'footer\(sl, (\d+)\)', replace_footer, text)

# Footers with two arguments (like Slide 22 -> 23)
text = re.sub(r'footer\(sl, (\d+),\s*(\d+)\)', replace_footer_2, text)


with open('scratch/make_presentation_v7.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated make_presentation_v7.py successfully.")
