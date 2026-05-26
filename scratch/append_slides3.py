
# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — ATTENTION TRANSFER
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

footer(sl, 14)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — ATTENTION TRANSFER t-SNE
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

footer(sl, 15)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 16 — BORN-AGAIN
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

footer(sl, 16)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 17 — CROSS-FRAME INTRO
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

footer(sl, 17)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 18 — GPU BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — GPU BENCHMARK | NVIDIA L40S")
slide_title(sl, "GPU Benchmark — NVIDIA L40S", "Model A (24f) vs Model B (16f) — pure inference latency")
pic(sl, GPU_PLT, Inches(1.2), Inches(1.7), Inches(11.0), Inches(4.5))
add_textbox(sl, "GPU Results Summary: BS≥8 (saturated): Model B achieves 37–40% latency savings (1.59×–1.66× speedup), exceeding theoretical 33.3% due to better Tensor Core memory alignment. BS=1: PCIe kernel launch overhead dominates → only ~11.5% saving.",
            Inches(0.45), Inches(6.5), Inches(12.4), Inches(0.5), font_size=11, color=C_PRIMARY)
footer(sl, 18)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 19 — CPU BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
sl = blank_slide(prs)
set_bg(sl, C_BG)
section_header(sl, "EFFICIENCY — CPU BENCHMARK | CLUSTER HOST")
slide_title(sl, "CPU Benchmark — Cluster Host", "Without PCIe overhead, speedup reflects pure arithmetic FLOP reduction")
pic(sl, CPU_PLT, Inches(1.2), Inches(1.7), Inches(11.0), Inches(4.5))
add_textbox(sl, "CPU Results Summary: Model B (16f): 136.70 ms → 88.51 ms at BS=1 (−35.3%), crossing below 100 ms real-time threshold. At BS=8: 1.60× speedup (−37.3%). Ideal for CPU-only edge deployments.",
            Inches(0.45), Inches(6.5), Inches(12.4), Inches(0.5), font_size=11, color=C_PRIMARY)
footer(sl, 19)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 20 — SPEED ACCURACY
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

footer(sl, 20)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 21 — CONCLUSIONS
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

footer(sl, 21)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 22 — THANK YOU
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

footer(sl, 22, 22)

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
out = r"PRESENTAZIONE/presentation_v7.pptx"
prs.save(out)
print(f"Salvato correttamente in: {out}")
