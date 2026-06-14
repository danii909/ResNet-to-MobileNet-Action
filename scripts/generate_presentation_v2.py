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

    def add_image_fit(slide, image_path, left, top, max_width, max_height):
        # Add picture at dummy location first
        pic = slide.shapes.add_picture(image_path, left, top)
        orig_w = pic.width
        orig_h = pic.height
        
        # Calculate aspect-ratio preserving scaling factor
        scale = min(max_width / orig_w, max_height / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Center the image inside the bounding box
        pic.width = new_w
        pic.height = new_h
        pic.left = int(left + (max_width - new_w) / 2)
        pic.top = int(top + (max_height - new_h) / 2)
        return pic

    # ----------------------------------------------------
    # SLIDE 1: Title Slide (Copertina)
    # ----------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide)

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
    format_paragraph(p, "• Target: My objective is to compress a heavy 3D ResNet-50 Teacher (~128 MB, Kinetics-400 pretrained) into an ultra-lightweight MobileNet3D Student (~9.5 MB).", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• The Capacity Gap: When training the student baseline from scratch on raw labels, I observed a performance ceiling of 59.48% Test Top-1, resulting in a massive 29.34% gap with the Teacher (88.82%).", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Strategy: I applied Knowledge Distillation (KD) to transfer the Teacher's dark knowledge and structural representations in order to bridge this gap.", 14, text_body, space_after=0)

    # Right Card: Model comparison table
    add_card(slide, Inches(7.4), Inches(1.8), Inches(5.1), Inches(4.5))
    table_box = slide.shapes.add_textbox(Inches(7.6), Inches(2.1), Inches(4.7), Inches(3.9))
    ttf = table_box.text_frame
    ttf.word_wrap = True
    p = ttf.paragraphs[0]
    format_paragraph(p, "ARCHITECTURAL COMPARISON", 12, color_teal, bold=True, space_after=15)
    
    # Table inside the card
    rows, cols = 3, 4
    table_shape = slide.shapes.add_table(rows, cols, Inches(7.6), Inches(2.7), Inches(4.7), Inches(2.2))
    table = table_shape.table
    
    headers = ["Model", "Size", "Top-1", "Top-5"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(38, 38, 54)
        p = cell.text_frame.paragraphs[0]
        format_paragraph(p, h, 11, text_title, bold=True, space_after=0)
        p.alignment = PP_ALIGN.CENTER
        
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
    format_paragraph(p, "TRAINING CONFIGURATION", 12, color_teal, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Optimizer & Schedule: I trained the students using the AdamW optimizer with a weight decay of 0.01 and a cosine annealing scheduler.", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Learning Rates: I selected a base learning rate of 0.0005 for student runs and 0.0007 for teacher fine-tuning.", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Training Budget: I trained the models for 60 epochs (adding a 10-epoch warmup for KD and AT runs where only Cross-Entropy loss is active).", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Regularization & Speed: I enabled AMP (Automatic Mixed Precision) and applied a label smoothing factor of 0.1.", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Data Augmentation (Light): I applied a short-side resize to 128, a random crop to 112x112, random horizontal flip (0.5), color jitter (0.1), random erasing (0.05), and a max temporal stride of 1.", 14, text_body, space_after=0)

    # Right Card - Group Aware Split
    add_card(slide, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.5))
    right_box = slide.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(3.9))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    format_paragraph(p, "GROUP-AWARE EVALUATION SPLIT", 12, color_teal, bold=True, space_after=12)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Split Methodology: To address the absence of an internal evaluation set, I divided the training set in a wise, group-aware manner.", 13, text_title, bold=True, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Handling Video Groups: Since UCF-101 contains clips originating from the same source videos, a standard random split would cause data leakage. I grouped files by extracting video source identifiers (e.g. '_gXX_') via regex.", 13, text_body, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Observation: By splitting at the video group level, I prevented data leakage. This yielded realistic validation metrics in the 60-65% range, providing a reliable proxy for true generalization.", 13, text_body, space_after=0)

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
    format_paragraph(p, "METHODOLOGY & OBSERVATIONS", 12, color_teal, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Logit-Based KD: I combined standard Cross-Entropy on hard labels with KL Divergence on soft targets scaled by temperature T.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Warmup Strategy: I used a 10-epoch classification warmup using pure ground truth. I observed that this is crucial to stabilize scratch student weights before introducing soft targets.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Performance Boost: In the standard run (T=8, alpha=0.7), I observed a Test Top-1 improvement of +4.83% over the baseline student, reaching 64.31% with zero size overhead.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Hyperparameters: I used T=8.0, alpha=0.7, BS=16 as the standard KD setup, while my optimal configuration was achieved at T=20.", 14, text_title, bold=True, space_after=0)

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
    format_paragraph(p, "I observed that logit-based KD bridges the capacity gap by a solid margin without modifying the Student architecture size or latency.", 11, text_muted, space_after=0)

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
    format_paragraph(p, "KEY TEMPERATURE OBSERVED TRENDS", 12, color_teal, bold=True, space_after=12)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Best Configuration: I obtained the best results at temperature T=20, reaching a Test Top-1 accuracy of 65.85% (a +6.37% gain over the baseline scratch student).", 13, text_title, bold=True, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Effect of Scaling: I observed that higher temperatures smooth the teacher's probability distributions. This exposes structural inter-class correlations that are easier for a lower-capacity student to mimic.", 13, text_body, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Peak Limitation: At lower temperatures (T=1), the teacher's logits behave like hard targets. I observed that the student struggles to capture dark knowledge under this peaky distribution.", 13, text_body, space_after=0)

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
    format_paragraph(p, "FEATURE PAIRING & THE 'KEY 5' BUG", 12, color_teal, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Concept: I integrated spatial and temporal attention maps (using L2-normalized activations) into the distillation loss to guide intermediate student layers.", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Bug Isolation: I debugged a silent issue where early configurations targeted block 5 of the Teacher (classification head, 2D). This yielded constant 1.0 attention maps, nullifying the loss.", 14, color_yellow, bold=True, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Resolution: I resolved the bug by mapping stages with matching spatial dimensions: `[2,3,4]` of ResNet to `[2,4,6]` of MobileNet.", 14, text_body, space_after=8)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Configurations: I tested Symmetric AT (spatial/temporal beta=0.05) and Temporal-Only AT (temporal beta=0.10).", 14, text_body, space_after=0)

    # Right Card - Results
    add_card(slide, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.5))
    right_box = slide.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(3.9))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    format_paragraph(p, "OBSERVED REGRESSION & ANALYSIS", 12, color_teal, bold=True, space_after=12)

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
    format_paragraph(p, "I hypothesize that this regression is caused by:", 12, color_teal, bold=True, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "1. Capacity Gap: The teacher's standard 3D convolutions generate rich spatial maps that the student's depthwise separable convolutions are structurally limited to capture.", 11, text_body, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "2. Over-Regularization: Forcing student intermediate layers to strictly mimic the teacher acts as an excessive constraint, preventing convergence.", 11, text_body, space_after=0)

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
    format_paragraph(p, "• Method: I implemented iterative self-distillation (homogeneous MobileNet3D-to-MobileNet3D) for 3 successive generations to verify if the student could improve recursively.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Configurations: I used temperature T = 2.5, alpha = 0.7, the AdamW optimizer with a cosine scheduler, and light data augmentation.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Teacher Init (Gen 0): I selected the best-performing logit-distilled student (KD T=8, 64.31% Test Top-1) as the starting point for Gen 1 training.", 14, text_body, space_after=0)

    # Right Card - Results & Analysis
    add_card(slide, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.5))
    right_box = slide.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(3.9))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    format_paragraph(p, "OBSERVED COLLAPSE & ANALYSIS", 12, color_teal, bold=True, space_after=10)

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
    format_paragraph(p, "I observed a progressive performance collapse. The causes reside in:", 12, color_teal, bold=True, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "1. Uncalibrated Temperature: The temperature parameter T=2.5, optimized for an oracle teacher (ResNet-50), produced overly flat probability distributions when applied to the less confident MobileNet3D teacher (64%), diluting the training signal.", 11, text_body, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "2. Error Propagation: Without an oracle, systematic errors and artifacts accumulate and propagate across generations, leading to overfitting on predictions containing errors.", 11, text_body, space_after=0)

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
    format_paragraph(p, "• Method: I applied t-SNE dimensionality reduction to pre-classifier embeddings across 10 randomly selected classes.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Teacher (ResNet-50): The embeddings form compact, cleanly separated, and distinct semantic clusters.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Student Baseline: The embeddings show significant class overlap, fuzzy cluster boundaries, and mixed clusters.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Best Distilled Student (KD T=20): I observed that the T=20 distilled student exhibits tighter clusters and much cleaner semantic separation than the baseline, confirming that higher temperatures enhance structural knowledge transfer.", 14, text_title, bold=True, space_after=0)

    # Right - Image (Centered, preserving aspect ratio)
    tsne_img_path = "docs/latex/parts/cap6_risultati/tsne/tsne_temperature_t20.png"
    add_card(slide, Inches(6.4), Inches(1.8), Inches(6.1), Inches(4.5))
    if os.path.exists(tsne_img_path):
        add_image_fit(slide, tsne_img_path, Inches(6.5), Inches(2.0), Inches(5.9), Inches(4.1))
    else:
        err_box = slide.shapes.add_textbox(Inches(6.6), Inches(3.5), Inches(5.7), Inches(1.0))
        p = err_box.text_frame.paragraphs[0]
        format_paragraph(p, "t-SNE T=20 plot image not found.", 14, color_yellow, bold=True)
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
    format_paragraph(p, "• Edge Constraint: Processing 24 frames per video clip is computationally expensive for real-time mobile and edge deployments.", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Cross-Frame Distillation: I designed a Model B that uses only 16 frames per clip, representing a 33.3% theoretical reduction in temporal computation compared to the 24-frame Model A.", 14, text_title, bold=True, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Core Objective: I evaluated whether this computational reduction translates into actual wall-clock speedups on target hardware (GPU and CPU).", 14, text_body, space_after=0)

    # Right Card
    add_card(slide, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.5))
    right_box = slide.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(3.9))
    rtf = right_box.text_frame
    rtf.word_wrap = True
    p = rtf.paragraphs[0]
    format_paragraph(p, "BENCHMARK METRICS & METHODOLOGY", 12, color_teal, bold=True, space_after=12)
    p = rtf.add_paragraph()
    format_paragraph(p, "• Dataset & Focus: I ran the benchmark on the official UCF-101 test set (3,783 clips) and measured the raw forward inference latency, excluding I/O.", 14, text_body, space_after=8)
    p = rtf.add_paragraph()
    format_paragraph(p, "• CUDA Sync: I used PyTorch CUDA synchronization to ensure precise timing of asynchronous GPU kernels.", 14, text_body, space_after=8)
    p = rtf.add_paragraph()
    format_paragraph(p, "• CPU Sampling: Due to the high cost of CPU inference, I sub-sampled CPU runs (50 batches for BS=1, 20 batches for BS=8) to prevent timeout while keeping statistically identical means.", 14, text_body, space_after=0)

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
    format_paragraph(p, "I observed that on GPU:", 12, color_teal, bold=True, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "• Saturation: At batch sizes >= 8, the GPU is fully saturated, yielding a 37%-40% speedup. This exceeds the linear 33.3% frame reduction due to better alignment of 16-frame blocks on Tensor Cores.", 11, text_body, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "• Overhead: At BS=1, PCIe bus overhead dominates, limiting the speedup to 1.13x.", 11, text_body, space_after=0)

    # Right Card - Centered Image
    gpu_plot_path = "results/Test_Set_Benchmark/gpu_benchmark_plots.png"
    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
    if os.path.exists(gpu_plot_path):
        add_image_fit(slide, gpu_plot_path, Inches(6.9), Inches(2.0), Inches(5.5), Inches(4.1))
    else:
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
    format_paragraph(p, "I observed that on CPU:", 12, color_teal, bold=True, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "• Arithmetic Scaling: Without PCIe bus delays, the speedup directly reflects the reduction in arithmetic FLOPs.", 11, text_body, space_after=4)
    p = dtf.add_paragraph()
    format_paragraph(p, "• Real-Time Target: Reducing latency (BS=1) from 136.7 ms to 88.5 ms (below the 100 ms threshold) makes the 16f model highly suitable for real-time edge processing.", 11, text_body, space_after=0)

    # Right Card - Centered Image
    cpu_plot_path = "results/Test_Set_Benchmark/cpu_benchmark_plots.png"
    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
    if os.path.exists(cpu_plot_path):
        add_image_fit(slide, cpu_plot_path, Inches(6.9), Inches(2.0), Inches(5.5), Inches(4.1))
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
    format_paragraph(p, "• Model A (24f, Best KD T=20): Reached 65.85% Test Top-1 (a substantial +6.37% gain over the baseline scratch student).", 14, text_body, space_after=10)
    p = ltf.add_paragraph()
    format_paragraph(p, "• Model B (16f, Cross-Frame KD): Reached 62.68% Test Top-1.", 14, text_body, space_after=10)
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
    format_paragraph(p, "• I observed that logit-based KD is a powerful, zero-overhead method. Higher temperatures (T=20) are critical to soften representations and bridge large capacity gaps.", 13.5, text_body, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• I observed that Attention Transfer is limited by architectural heterogeneity (standard vs depthwise convolutions) and risks over-regularization.", 13.5, text_body, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• I observed that Self-Distillation (BAN) is highly sensitive to temperature and prone to generational collapse when the teacher is not an oracle.", 13.5, text_body, space_after=10)
    p = rtf.add_paragraph()
    format_paragraph(p, "• I conclude that Cross-Frame Distillation provides an outstanding Pareto-optimal trade-off for practical mobile/edge real-time deployments.", 13.5, text_title, bold=True, space_after=0)

    # Save
    output_path = "presentation_final.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_presentation()
