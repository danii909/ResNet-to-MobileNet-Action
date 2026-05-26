
# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — TRAINING CURVES
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
footer(sl, 9)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — CONFUSION MATRIX
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
footer(sl, 10)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — CONFUSION MATRIX BASELINE VS KD T=20
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
footer(sl, 11)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — TOP 5 ERRORS RELATIONSHIP
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

footer(sl, 12)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — KD IMPACT PER CLASS
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

footer(sl, 13)
