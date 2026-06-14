from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # Blank slide

    # Colors
    bg_color = RGBColor(15, 15, 22)        # Very dark blue-gray
    card_color = RGBColor(26, 26, 36)      # Slate gray for card background
    border_color = RGBColor(45, 45, 60)    # Subtle card border
    text_title = RGBColor(245, 245, 250)   # Off-white for headers
    text_body = RGBColor(200, 200, 215)    # Light gray for standard text
    text_muted = RGBColor(150, 150, 165)   # Muted gray
    
    # Accent Colors
    color_teal = RGBColor(45, 212, 191)    # Pastel Teal (KD, Teacher)
    color_violet = RGBColor(167, 139, 250) # Pastel Violet (Baseline, Latency)
    color_green = RGBColor(52, 211, 153)   # Pastel Green (Saved time, improvements)
    color_yellow = RGBColor(251, 191, 36)  # Pastel Gold (Warnings, bugs)
    color_white = RGBColor(255, 255, 255)

    def set_slide_background(slide):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = bg_color

    def add_slide_header(slide, title_text, category_text=None):
        # Category/Tracker (Teal small text)
        if category_text:
            cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
            tf = cat_box.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = category_text.upper()
            p.font.name = "Segoe UI"
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = color_teal
        
        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Segoe UI"
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = text_title

    def add_card(slide, left, top, width, height):
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = card_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
        # Remove default adjustments to make corners subtler
        shape.adjustments[0] = 0.05
        return shape

    def format_paragraph(p, text, size_pt, color_rgb, bold=False, space_after=6, line_spacing=1.15):
        p.text = text
        p.font.name = "Segoe UI"
        p.font.size = Pt(size_pt)
        p.font.bold = bold
        p.font.color.rgb = color_rgb
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing

    # ----------------------------------------------------
    # SLIDE 1: Title Slide (Copertina)
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)

    # Large decorative accent block on the left
    accent_bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.4), Inches(7.5)
    )
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = color_teal
    accent_bar.line.fill.background()

    # Left content block (Title & Subtitle)
    title_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(7.5), Inches(4.5))
    tf = title_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_course = tf.paragraphs[0]
    format_paragraph(p_course, "DEEP LEARNING: ADVANCED MODELS AND METHODS", 12, color_teal, bold=True, space_after=18)
    
    p_title = tf.add_paragraph()
    format_paragraph(p_title, "Knowledge Distillation for\nMobile Action Recognition", 38, text_title, bold=True, space_after=18)
    
    p_sub = tf.add_paragraph()
    format_paragraph(p_sub, "Compressing 3D ResNet-50 into MobileNet3D: Optimization, Attention Transfer, and Cross-Frame Distillation", 16, text_body, space_after=0)

    # Right Content Card (Author Details)
    card_left, card_top, card_width, card_height = Inches(8.8), Inches(1.2), Inches(3.7), Inches(5.1)
    add_card(slide, card_left, card_top, card_width, card_height)

    info_box = slide.shapes.add_textbox(Inches(9.1), Inches(1.5), Inches(3.1), Inches(4.5))
    itf = info_box.text_frame
    itf.word_wrap = True
    
    p_info_hdr = itf.paragraphs[0]
    format_paragraph(p_info_hdr, "PROJECT DETAILS", 11, color_teal, bold=True, space_after=14)

    labels_data = [
        ("Author", "Daniele Barbagallo"),
        ("Student ID", "1000015334"),
        ("Advisor", "Prof. Antonino Furnari"),
        ("Course", "Deep Learning (AMM)"),
        ("Institution", "University of Catania"),
        ("Department", "DMI"),
        ("Academic Year", "2025 / 2026")
    ]
    for lbl, val in labels_data:
        p_lbl = itf.add_paragraph()
        format_paragraph(p_lbl, lbl.upper(), 9, text_muted, bold=True, space_after=2)
        p_val = itf.add_paragraph()
        format_paragraph(p_val, val, 14, text_title, bold=True, space_after=12)

    # ----------------------------------------------------
    # SLIDE 2: Objectives & Capacity Gap
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "Project Objectives & The Capacity Gap", "Introduction")

    # Left text box
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.8))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = ltf.margin_right = ltf.margin_bottom = 0
    
    p = ltf.paragraphs[0]
    format_paragraph(p, "THE COMPRESSION CHALLENGE", 12, color_teal, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Mobile Action Recognition: Video processing is computationally heavy, demanding compact and fast models for mobile or edge deployment.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Target: Compress a heavy 3D ResNet-50 Teacher (~128 MB, Kinetics-400 pretrained) into an ultra-lightweight MobileNet3D Student (~9.5 MB) trained from scratch.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• The Capacity Gap: MobileNet3D trained from scratch from raw labels reaches a performance ceiling of 59.48% Test Top-1, creating a massive 29.34% gap with the Teacher (88.82%).", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• The Solution: Employ Knowledge Distillation (KD) to transfer the Teacher's dark knowledge and structural representations to close this performance gap.", 14, text_body, space_after=0)

    # Right Card: Model comparison table
    add_card(slide, Inches(7.4), Inches(1.8), Inches(5.1), Inches(4.5))
    table_box = slide.shapes.add_textbox(Inches(7.6), Inches(2.1), Inches(4.7), Inches(3.9))
    ttf = table_box.text_frame
    ttf.word_wrap = True
    p = ttf.paragraphs[0]
    format_paragraph(p, "ARCHITECTURAL COMPARISON", 12, color_teal, bold=True, space_after=15)
    
    # Add a table shape inside the card
    rows, cols = 3, 4
    table_shape = slide.shapes.add_table(rows, cols, Inches(7.6), Inches(2.7), Inches(4.7), Inches(2.2))
    table = table_shape.table
    
    # Table headers
    headers = ["Model", "Size", "Top-1", "Top-5"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(38, 38, 54)
        p = cell.text_frame.paragraphs[0]
        format_paragraph(p, h, 11, text_title, bold=True, space_after=0)
        p.alignment = PP_ALIGN.CENTER
        
    # Table data
    data = [
        ["Teacher (3D R50)", "128.0 MB", "88.82%", "98.18%"],
        ["Student Baseline", "9.5 MB", "59.48%", "82.77%"]
    ]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r+1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(26, 26, 36)
            p = cell.text_frame.paragraphs[0]
            # Color model names specifically
            color = color_teal if r == 0 and c == 0 else (color_violet if r == 1 and c == 0 else text_body)
            format_paragraph(p, val, 11, color, bold=(c==0), space_after=0)
            p.alignment = PP_ALIGN.CENTER

    desc_box = slide.shapes.add_textbox(Inches(7.6), Inches(5.2), Inches(4.7), Inches(1.0))
    dtf = desc_box.text_frame
    dtf.word_wrap = True
    p = dtf.paragraphs[0]
    format_paragraph(p, "The Teacher is ~13.5x larger than the Student baseline, presenting a clear compression opportunity via logit and attention alignment.", 11, text_muted, space_after=0)

    # ----------------------------------------------------
    # SLIDE 3: Experimental Setup & Group-Aware Split
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "Experimental Setup & Group-Aware Dataset Split", "Methodology")

    # Left Box - Setup
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = ltf.margin_right = ltf.margin_bottom = 0
    p = ltf.paragraphs[0]
    format_paragraph(p, "TRAINING HYPERPARAMETERS & CONFIG", 12, color_teal, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Optimizer: AdamW with weight decay of 0.01.", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• LR Scheduler: Cosine annealing with base learning rate of 0.0005 for students, 0.0007 for teacher.", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Training Budget: 60 epochs (with 10-epoch warmup for KD/AT where only Cross-Entropy loss is active).", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Regularization: AMP (Automatic Mixed Precision) enabled, Label Smoothing of 0.1.", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Data Augmentation (Light): Short-side resize to 128, crop to 112x112, random horizontal flip, color jitter (0.1), random erasing (0.05), max temporal stride 1.", 14, text_body, space_after=0)

    # Right Card - Group Aware Split
    add_card(slide, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.5))
    right_box = slide.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(3.9))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    format_paragraph(p, "GROUP-AWARE SPLIT DESIGN", 12, color_teal, bold=True, space_after=12)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Motivation: UCF-101 contains clips extracted from the same original source videos. Standard random splitting results in data leakage, creating overly optimistic, fake validation metrics.", 13, text_body, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Smart Division: The training set was split in a wise, group-aware manner based on source video IDs (extracted via regex, e.g. '_gXX_') to obtain an internal evaluation set which was otherwise absent.", 13, text_title, bold=True, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Impact: Evaluation set metrics dropped from fittitiouly high values to a realistic 60-65% validation range, exposing the true generalization gap and ensuring reliable hyperparameter tuning.", 13, text_body, space_after=0)

    # ----------------------------------------------------
    # SLIDE 4: Logit-Based Knowledge Distillation
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "Logit-Based Knowledge Distillation Results", "Knowledge Distillation")

    # Left Column
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = ltf.margin_right = ltf.margin_bottom = 0
    p = ltf.paragraphs[0]
    format_paragraph(p, "LOGIT ALIGNMENT METHODOLOGY", 12, color_teal, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Loss Formulation: Combines standard Cross-Entropy (hard labels) with KL Divergence on soft targets scaled by temperature T.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Hyperparameters: Best standard baseline configuration uses temperature T = 8.0, alpha = 0.7, batch size 16.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Warmup Effect: An initial 10-epoch classification warmup using pure ground truth is crucial to stabilize Student weights before introducing teacher soft targets.", 14, text_title, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Parameter Overhead: +0% extra parameters. The distilled student maintains the exact 9.5 MB size and low latency of the baseline.", 14, text_body, space_after=0)

    # Right Card - Results
    add_card(slide, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.5))
    right_box = slide.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(3.9))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    format_paragraph(p, "QUANTITATIVE PERFORMANCE COMPARISON", 12, color_teal, bold=True, space_after=15)

    # Table
    table_shape = slide.shapes.add_table(4, 5, Inches(7.1), Inches(2.7), Inches(5.2), Inches(2.5))
    table = table_shape.table
    
    headers = ["Model", "Val Top-1", "Test Top-1", "Test Top-5", "Delta"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(38, 38, 54)
        p = cell.text_frame.paragraphs[0]
        format_paragraph(p, h, 10, text_title, bold=True, space_after=0)
        p.alignment = PP_ALIGN.CENTER
        
    data = [
        ["Teacher (3D R50)", "92.36%", "88.82%", "98.18%", "Upper Bound"],
        ["Student Baseline", "59.18%", "59.48%", "82.77%", "Reference"],
        ["KD Student (T=8)", "64.57%", "64.31%", "87.68%", "+4.83%"]
    ]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r+1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(26, 26, 36)
            p = cell.text_frame.paragraphs[0]
            color = color_teal if r == 2 and c == 2 else (color_green if r == 2 and c == 4 else text_body)
            bold = (r == 2 and (c == 2 or c == 4)) or (c == 0)
            format_paragraph(p, val, 10, color, bold=bold, space_after=0)
            p.alignment = PP_ALIGN.CENTER

    desc_box = slide.shapes.add_textbox(Inches(7.1), Inches(5.3), Inches(5.2), Inches(0.9))
    dtf = desc_box.text_frame
    dtf.word_wrap = True
    p = dtf.paragraphs[0]
    format_paragraph(p, "Knowledge Distillation yields a significant +4.83% absolute improvement in Test Top-1 and +4.91% in Top-5, without modifying the Student architecture.", 11, text_muted, space_after=0)

    # ----------------------------------------------------
    # SLIDE 5: Temperature Scaling Ablation
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "Temperature Scaling Ablation Study", "Knowledge Distillation")

    # Left Card - Table of sweep
    add_card(slide, Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.5))
    left_box = slide.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.6), Inches(3.9))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    p = ltf.paragraphs[0]
    format_paragraph(p, "TEMPERATURE ABLATION SWEEP (ALPHA = 0.7)", 12, color_teal, bold=True, space_after=15)

    table_shape = slide.shapes.add_table(6, 5, Inches(1.0), Inches(2.7), Inches(5.6), Inches(2.8))
    table = table_shape.table
    
    headers = ["Temp (T)", "Train Acc", "Val Top-1", "Test Top-1", "Delta vs BL"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(38, 38, 54)
        p = cell.text_frame.paragraphs[0]
        format_paragraph(p, h, 10, text_title, bold=True, space_after=0)
        p.alignment = PP_ALIGN.CENTER

    data = [
        ["T = 1", "98.44%", "62.75%", "62.91%", "+3.43%"],
        ["T = 5", "95.87%", "63.12%", "62.07%", "+2.59%"],
        ["T = 8", "97.59%", "64.57%", "64.31%", "+4.83%"],
        ["T = 10", "98.28%", "64.85%", "64.47%", "+4.99%"],
        ["T = 20 (Best)", "99.05%", "65.18%", "65.85%", "+6.37%"]
    ]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r+1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(26, 26, 36)
            p = cell.text_frame.paragraphs[0]
            is_best = (r == 4)
            color = color_green if c == 4 else (color_teal if is_best and c == 3 else text_body)
            format_paragraph(p, val, 10, color, bold=is_best, space_after=0)
            p.alignment = PP_ALIGN.CENTER

    # Right Column - Analysis
    right_box = slide.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.2), Inches(4.5))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    rtf.margin_left = rtf.margin_top = rtf.margin_right = rtf.margin_bottom = 0
    p = rtf.paragraphs[0]
    format_paragraph(p, "KEY TEMPERATURE TRENDS & FINDINGS", 12, color_teal, bold=True, space_after=12)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Monotonic Increase: Accuracy shows a clear upward trend with temperature, peaking at T = 20 with a massive +6.37% gain over baseline.", 13, text_title, bold=True, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Softening Probability Distributions: With T=1, the teacher's soft targets are too peaked, behaving like standard hard labels. This makes it difficult for the student to capture relationships between classes.", 13, text_body, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Bridging the Capacity Gap: Setting T=20 smooths out the distributions, revealing the 'dark knowledge' (inter-class similarity structures) that a lower-capacity student is capable of mimicking.", 13, text_body, space_after=0)

    # ----------------------------------------------------
    # SLIDE 6: Attention Transfer & Mismatch Debugging
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "Attention Transfer: Debugging & Results", "Advanced KD Methods")

    # Left Column
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = ltf.margin_right = ltf.margin_bottom = 0
    p = ltf.paragraphs[0]
    format_paragraph(p, "THE 'KEY 5' BUG & FEATURE PAIRING", 12, color_teal, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Concept: Align intermediate spatial and temporal attention maps (L2-normalized activations) from matching network stages.", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• The Bug: Early attempts targeted block 5 of the Teacher (ResNet-50 classification head, 2D). This generated constant 1.0 attention maps, silently nullifying the loss.", 14, color_yellow, bold=True, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Correction: Mapped stages with comparable spatial dimensions: `[2,3,4]` of ResNet to `[2,4,6]` of MobileNet.", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Configs: Symmetric (spatial beta=0.05, temporal beta=0.05) and Temporal-Only (spatial beta=0.0, temporal beta=0.10).", 14, text_body, space_after=0)

    # Right Card - Results
    add_card(slide, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.5))
    right_box = slide.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(3.9))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    format_paragraph(p, "PERFORMANCE REGRESSION & ANALYSIS", 12, color_teal, bold=True, space_after=12)

    table_shape = slide.shapes.add_table(4, 3, Inches(7.1), Inches(2.6), Inches(5.2), Inches(1.6))
    table = table_shape.table
    
    headers = ["Configuration", "Val Top-1", "Delta vs KD (T=8)"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(38, 38, 54)
        p = cell.text_frame.paragraphs[0]
        format_paragraph(p, h, 10, text_title, bold=True, space_after=0)
        p.alignment = PP_ALIGN.CENTER
        
    data = [
        ["KD (T=8, no AT)", "64.57%", "Reference"],
        ["AT Symmetric", "58.25%", "-6.32%"],
        ["AT Temporal-Only", "59.04%", "-5.53%"]
    ]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r+1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(26, 26, 36)
            p = cell.text_frame.paragraphs[0]
            color = color_yellow if r > 0 and c == 2 else text_body
            bold = (r > 0 and c == 2) or (c == 0)
            format_paragraph(p, val, 10, color, bold=bold, space_after=0)
            p.alignment = PP_ALIGN.CENTER

    desc_box = slide.shapes.add_textbox(Inches(7.1), Inches(4.4), Inches(5.2), Inches(1.8))
    dtf = desc_box.text_frame
    dtf.word_wrap = True
    p = dtf.paragraphs[0]
    format_paragraph(p, "Why did Attention Transfer fail?", 12, color_teal, bold=True, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "1. Capacity Gap: Standard 3D convolutions (Teacher) produce rich, high-dimensional spatial maps. MobileNet3D's depthwise separable convolutions are too limited to mimic them, causing learning collapse.", 11, text_body, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "2. Over-Regularization: Forcing intermediate layers to strictly align acts as an excessive constraint, preventing the Student from finding alternative solutions.", 11, text_body, space_after=0)

    # ----------------------------------------------------
    # SLIDE 7: Born-Again Networks & Generational Collapse
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "Born-Again Networks: Generational Collapse", "Advanced KD Methods")

    # Left Column
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = ltf.margin_right = ltf.margin_bottom = 0
    p = ltf.paragraphs[0]
    format_paragraph(p, "ITERATIVE SELF-DISTILLATION SETUP", 12, color_teal, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Method: Train a student model from a teacher. In the next generation, this student becomes the new teacher, distilling its knowledge to a structurally identical student.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Homogeneous Architecture: MobileNet3D (Teacher) to MobileNet3D (Student). Spatial features match perfectly without interpolation.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Initial Target (Gen 0): The best-performing logit-distilled student (KD T=8, 64.31% Test Top-1) serves as the starting point.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Configuration: Temperature T = 8.0, alpha = 0.7, optimizer AdamW, cosine scheduler, light augmentation.", 14, text_body, space_after=0)

    # Right Card - Results & Analysis
    add_card(slide, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.5))
    right_box = slide.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(3.9))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    format_paragraph(p, "THE COLLAPSE PHENOMENON", 12, color_teal, bold=True, space_after=10)

    table_shape = slide.shapes.add_table(5, 4, Inches(7.1), Inches(2.5), Inches(5.2), Inches(1.8))
    table = table_shape.table
    
    headers = ["Generation", "Teacher", "Test Top-1", "Delta vs Gen 0"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(38, 38, 54)
        p = cell.text_frame.paragraphs[0]
        format_paragraph(p, h, 10, text_title, bold=True, space_after=0)
        p.alignment = PP_ALIGN.CENTER
        
    data = [
        ["Gen 0", "ResNet-50", "64.31%", "Reference"],
        ["Gen 1", "Gen 0 Student", "60.98%", "-3.33%"],
        ["Gen 2", "Gen 1 Student", "60.43%", "-3.88%"],
        ["Gen 3", "Gen 2 Student", "59.21%", "-5.10%"]
    ]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r+1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(26, 26, 36)
            p = cell.text_frame.paragraphs[0]
            color = color_yellow if r > 0 and c == 3 else text_body
            bold = (r > 0 and c == 3) or (c == 0)
            format_paragraph(p, val, 10, color, bold=bold, space_after=0)
            p.alignment = PP_ALIGN.CENTER

    desc_box = slide.shapes.add_textbox(Inches(7.1), Inches(4.5), Inches(5.2), Inches(1.8))
    dtf = desc_box.text_frame
    dtf.word_wrap = True
    p = dtf.paragraphs[0]
    format_paragraph(p, "Why did the generations collapse?", 12, color_teal, bold=True, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "1. Uncalibrated Temperature: T=8.0 was tuned for the high-confidence ResNet-50. When applied to the less confident MobileNet3D (64%), it flattened probabilities into near-white noise.", 11, text_body, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "2. Error Propagation: Without an oracle (ground truth teacher), systematic mistakes and artifacts accumulate and propagate across generations, leading to overfitting on errors.", 11, text_body, space_after=0)

    # ----------------------------------------------------
    # SLIDE 8: Latent Space Analysis (t-SNE)
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "Latent Space Analysis via t-SNE Projections", "Qualitative Analysis")

    # Left Column
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.2), Inches(4.8))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = ltf.margin_right = ltf.margin_bottom = 0
    p = ltf.paragraphs[0]
    format_paragraph(p, "VISUALIZING LEARNED EMBEDDINGS", 12, color_teal, bold=True, space_after=12)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Method: Apply t-SNE dimensionality reduction to pre-classifier embeddings across 10 randomly selected classes.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Teacher (ResNet-50): Left panel shows highly compact, well-separated, and distinct semantic clusters.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Student Baseline: Middle panel displays fuzzy cluster boundaries, significant class overlap, and mixed clusters.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Distilled Student (KD T=8): Right panel shows dramatically improved cluster definitions and separation, validating the success of structural knowledge transfer.", 14, text_title, bold=True, space_after=0)

    # Right - Image
    tsne_img_path = "docs/latex/parts/cap6_risultati/tsne/tsne_comparison_4495.png"
    if os.path.exists(tsne_img_path):
        # We place a card backing
        add_card(slide, Inches(6.4), Inches(1.8), Inches(6.1), Inches(4.5))
        # Add the image
        slide.shapes.add_picture(tsne_img_path, Inches(6.5), Inches(2.0), Inches(5.9), Inches(4.1))
    else:
        # Placeholder card if not found
        add_card(slide, Inches(6.4), Inches(1.8), Inches(6.1), Inches(4.5))
        err_box = slide.shapes.add_textbox(Inches(6.6), Inches(3.5), Inches(5.7), Inches(1.0))
        p = err_box.text_frame.paragraphs[0]
        format_paragraph(p, "t-SNE comparison plot image not found.", 14, color_yellow, bold=True)
        p.alignment = PP_ALIGN.CENTER

    # ----------------------------------------------------
    # SLIDE 9: Cross-Frame Distillation for Edge Deployment
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "Cross-Frame Distillation for Edge Deployment", "Cross-Frame Distillation")

    # Left Column
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = ltf.margin_right = ltf.margin_bottom = 0
    p = ltf.paragraphs[0]
    format_paragraph(p, "EFFICIENCY VS ACCURACY GOALS", 12, color_teal, bold=True, space_after=12)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Deployment Challenge: Processing 24 frames per video clip is computationally expensive, especially on resource-constrained CPUs and older edge GPUs.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Cross-Frame Distillation: Train a Model B using only 16 frames per clip, representing a 33.3% theoretical reduction in temporal FLOPs compared to the 24-frame Model A.", 14, text_title, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Core Inquiry: Verify if this 33.3% computational savings translates into actual wall-clock execution speedups on hardware (GPU and CPU).", 14, text_body, space_after=0)

    # Right Card
    add_card(slide, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.5))
    right_box = slide.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(3.9))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    format_paragraph(p, "BENCHMARK METRICS & METHODOLOGY", 12, color_teal, bold=True, space_after=12)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Dataset: Official UCF-101 test set, containing 3,783 real video clips.", 14, text_body, space_after=8)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Measurement: Focuses solely on forward inference time (model(clips)) to isolate network speed from disk I/O.", 14, text_body, space_after=8)
    p = rtf.add_paragraph()
    format_paragraph(p, "• GPU Sync: Uses PyTorch CUDA synchronization to ensure asynchronous kernels are fully completed before measuring times.", 14, text_body, space_after=8)
    p = rtf.add_paragraph()
    format_paragraph(p, "• CPU Sampling: CPU benchmarks are sub-sampled (50 batches for BS=1, 20 batches for BS=8) to avoid excessively long test runs while retaining statistical accuracy.", 14, text_body, space_after=0)

    # ----------------------------------------------------
    # SLIDE 10: GPU Benchmark Performance
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "GPU Inference Benchmark: NVIDIA L40S", "Cross-Frame Distillation")

    # Left Column - Table
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = ltf.margin_right = ltf.margin_bottom = 0
    p = ltf.paragraphs[0]
    format_paragraph(p, "GPU BENCHMARK (MS PER CLIP)", 12, color_teal, bold=True, space_after=12)

    table_shape = slide.shapes.add_table(5, 5, Inches(0.8), Inches(2.4), Inches(5.6), Inches(2.2))
    table = table_shape.table
    
    headers = ["Batch Size", "Model A (24f)", "Model B (16f)", "Speedup", "Saved Time"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(38, 38, 54)
        p = cell.text_frame.paragraphs[0]
        format_paragraph(p, h, 9, text_title, bold=True, space_after=0)
        p.alignment = PP_ALIGN.CENTER
        
    data = [
        ["BS = 1", "4.54 ms", "4.02 ms", "1.13x", "11.49%"],
        ["BS = 8", "1.03 ms", "0.62 ms", "1.65x", "39.32%"],
        ["BS = 16", "1.18 ms", "0.74 ms", "1.59x", "37.15%"],
        ["BS = 32", "1.37 ms", "0.82 ms", "1.66x", "39.69%"]
    ]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r+1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(26, 26, 36)
            p = cell.text_frame.paragraphs[0]
            is_high_bs = (r > 0)
            color = color_green if is_high_bs and (c == 3 or c == 4) else (color_teal if c == 2 else text_body)
            bold = is_high_bs and (c == 3 or c == 4)
            format_paragraph(p, val, 9.5, color, bold=bold, space_after=0)
            p.alignment = PP_ALIGN.CENTER

    desc_box = slide.shapes.add_textbox(Inches(0.8), Inches(4.8), Inches(5.6), Inches(2.0))
    dtf = desc_box.text_frame
    dtf.word_wrap = True
    dtf.margin_left = dtf.margin_top = dtf.margin_right = dtf.margin_bottom = 0
    p = dtf.paragraphs[0]
    format_paragraph(p, "Key GPU Findings:", 12, color_teal, bold=True, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "• GPU Saturation: At BS >= 8, GPU is fully saturated, yielding a massive 37%-40% speedup. This exceeds the linear 33.3% frame reduction due to better alignment of 16-frame blocks on Tensor Cores.", 11, text_body, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "• PCIe Overhead: At BS = 1, PCIe bus overhead dominates, limiting speedup to 1.13x.", 11, text_body, space_after=0)

    # Right Card - Image
    gpu_plot_path = "results/Test_Set_Benchmark/gpu_benchmark_plots.png"
    if os.path.exists(gpu_plot_path):
        add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
        slide.shapes.add_picture(gpu_plot_path, Inches(6.9), Inches(2.2), Inches(5.5), Inches(3.7))
    else:
        add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
        err_box = slide.shapes.add_textbox(Inches(7.0), Inches(3.5), Inches(5.3), Inches(1.0))
        p = err_box.text_frame.paragraphs[0]
        format_paragraph(p, "GPU benchmark plots image not found.", 14, color_yellow, bold=True)
        p.alignment = PP_ALIGN.CENTER

    # ----------------------------------------------------
    # SLIDE 11: CPU Benchmark Performance
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "CPU Inference Benchmark: Host Cluster", "Cross-Frame Distillation")

    # Left Column - Table
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = ltf.margin_right = ltf.margin_bottom = 0
    p = ltf.paragraphs[0]
    format_paragraph(p, "CPU BENCHMARK (MS PER CLIP)", 12, color_teal, bold=True, space_after=12)

    table_shape = slide.shapes.add_table(3, 5, Inches(0.8), Inches(2.5), Inches(5.6), Inches(1.8))
    table = table_shape.table
    
    headers = ["Batch Size", "Model A (24f)", "Model B (16f)", "Speedup", "Saved Time"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(38, 38, 54)
        p = cell.text_frame.paragraphs[0]
        format_paragraph(p, h, 9, text_title, bold=True, space_after=0)
        p.alignment = PP_ALIGN.CENTER
        
    data = [
        ["BS = 1", "136.70 ms", "88.51 ms", "1.54x", "35.26%"],
        ["BS = 8", "113.51 ms", "71.13 ms", "1.60x", "37.33%"]
    ]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r+1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(26, 26, 36)
            p = cell.text_frame.paragraphs[0]
            color = color_green if (c == 3 or c == 4) else (color_teal if c == 2 else text_body)
            bold = (c == 3 or c == 4)
            format_paragraph(p, val, 9.5, color, bold=bold, space_after=0)
            p.alignment = PP_ALIGN.CENTER

    desc_box = slide.shapes.add_textbox(Inches(0.8), Inches(4.6), Inches(5.6), Inches(2.2))
    dtf = desc_box.text_frame
    dtf.word_wrap = True
    dtf.margin_left = dtf.margin_top = dtf.margin_right = dtf.margin_bottom = 0
    p = dtf.paragraphs[0]
    format_paragraph(p, "Key CPU Findings:", 12, color_teal, bold=True, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "• FLOPs-Dominated Speedup: Without the PCIe bus bottleneck, CPU speedup closely reflects the theoretical temporal FLOP reduction.", 11, text_body, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "• Edge Suitability: Reducing single-clip latency (BS=1) from 136.7 ms to 88.5 ms (below the 100 ms threshold) makes the 16f model highly suitable for real-time edge processing.", 11, text_body, space_after=0)

    # Right Card - Image
    cpu_plot_path = "results/Test_Set_Benchmark/cpu_benchmark_plots.png"
    if os.path.exists(cpu_plot_path):
        add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
        slide.shapes.add_picture(cpu_plot_path, Inches(6.9), Inches(2.2), Inches(5.5), Inches(3.7))
    else:
        add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
        err_box = slide.shapes.add_textbox(Inches(7.0), Inches(3.5), Inches(5.3), Inches(1.0))
        p = err_box.text_frame.paragraphs[0]
        format_paragraph(p, "CPU benchmark plots image not found.", 14, color_yellow, bold=True)
        p.alignment = PP_ALIGN.CENTER

    # ----------------------------------------------------
    # SLIDE 12: Conclusions & Speed-Accuracy Trade-off
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)
    add_slide_header(slide, "Conclusions & Speed-Accuracy Trade-off", "Conclusions")

    # Left Column
    left_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = ltf.margin_right = ltf.margin_bottom = 0
    p = ltf.paragraphs[0]
    format_paragraph(p, "THE SPEED-ACCURACY TRADE-OFF", 12, color_teal, bold=True, space_after=12)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Model A (24f, Best KD T=20): Achieves 65.85% Test Top-1 (a substantial +6.37% gain over the baseline scratch student).", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Model B (16f, Cross-Frame KD): Reaches 62.68% Test Top-1.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Trade-off Evaluation: Saving 37%-40% in latency/FLOPs costs only 3.17% in Test Top-1 compared to Model A.", 14, text_title, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Net Improvement: Model B (16f) still outperforms the original Model A baseline from scratch (59.48%) by a solid +3.20% while executing significantly faster.", 14, text_body, space_after=0)

    # Right Card - Key Takeaways
    add_card(slide, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.5))
    right_box = slide.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(3.9))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    format_paragraph(p, "KEY SUMMARY TAKEAWAYS", 12, color_teal, bold=True, space_after=12)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Logit-Based KD is a powerful and zero-overhead method. Higher temperatures (T=20) are critical to soften representations and bridge large capacity gaps.", 13.5, text_body, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Attention Transfer is limited by architectural heterogeneity (standard vs depthwise convolutions) and risk of over-regularization.", 13.5, text_body, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Self-Distillation (BAN) is highly sensitive to temperature and prone to generational collapse when the teacher is not an oracle.", 13.5, text_body, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Cross-Frame Distillation provides an outstanding Pareto-optimal trade-off for practical mobile/edge real-time deployments.", 13.5, text_title, bold=True, space_after=0)

    # Save
    output_path = "presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_presentation()
