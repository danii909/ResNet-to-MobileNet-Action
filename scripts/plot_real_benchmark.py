import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Data from slurm-benchmark-testset-4700.log
    
    # GPU Data (per-clip latency)
    gpu_bs = [1, 8, 16, 32]
    gpu_latency_clip_a = [4.54, 1.03, 1.18, 1.37]
    gpu_throughput_a = [220.4, 971.6, 846.8, 731.1]
    gpu_latency_clip_b = [4.02, 0.62, 0.74, 0.82]
    gpu_throughput_b = [249.0, 1601.3, 1347.3, 1212.4]
    gpu_saved = [11.49, 39.32, 37.15, 39.69]

    # Convert to Total Batch Latency (latency_per_clip * batch_size)
    gpu_latency_a = [lat * bs for lat, bs in zip(gpu_latency_clip_a, gpu_bs)]
    gpu_latency_b = [lat * bs for lat, bs in zip(gpu_latency_clip_b, gpu_bs)]

    # CPU Data (per-clip latency)
    cpu_bs = [1, 8]
    cpu_latency_clip_a = [136.70, 113.51]
    cpu_latency_clip_b = [88.51, 71.13]
    # Re-calculate exact throughput from ms latency: 1000 / latency
    cpu_throughput_a = [1000.0 / x for x in cpu_latency_clip_a]
    cpu_throughput_b = [1000.0 / x for x in cpu_latency_clip_b]
    cpu_saved = [35.26, 37.33]

    # Convert to Total Batch Latency
    cpu_latency_a = [lat * bs for lat, bs in zip(cpu_latency_clip_a, cpu_bs)]
    cpu_latency_b = [lat * bs for lat, bs in zip(cpu_latency_clip_b, cpu_bs)]

    # Colors
    color_a = '#e74c3c'  # Soft Red
    color_b = '#3498db'  # Soft Blue
    color_sgreen = '#2ecc71'  # Soft Green
    color_dgreen = '#1e8449'  # Dark Green
    color_purple = '#8e44ad'  # Violet/Purple from old plots

    # Setup styles
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Ensure output directories exist
    base_dir = Path("results/Test_Set_Benchmark")
    indiv_dir = base_dir / "individual_plots"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. GPU BENCHMARK PLOTS (1x3 Panel)
    # =========================================================================
    fig_gpu, axs_gpu = plt.subplots(1, 3, figsize=(18, 6))
    fig_gpu.suptitle('Test set inference Benchmark (GPU): Model A (24f) (Top-1: 65.85%) vs Cross Frame Model B (16f) (Top-1: 62.68%)', 
                     fontsize=14, fontweight='bold', y=1.02)

    # 1.1 GPU Latency
    axs_gpu[0].plot(gpu_bs, gpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_gpu[0].plot(gpu_bs, gpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            y_mid = (gpu_latency_a[i] + gpu_latency_b[i]) / 2.0
            axs_gpu[0].annotate("", xy=(bs, gpu_latency_a[i]), xytext=(bs, gpu_latency_b[i]),
                                arrowprops=dict(arrowstyle="<->", color=color_purple, lw=1.2))
            axs_gpu[0].text(bs * 1.12, y_mid, f"-{gpu_saved[i]:.1f}%", 
                        color=color_purple, fontweight='bold', fontsize=10,
                        va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_purple, lw=1))
    axs_gpu[0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_gpu[0].set_ylabel('Total Batch Latency (ms)', fontweight='bold')
    axs_gpu[0].set_title('GPU Batch Latency vs Batch Size (Lower is Better)', fontsize=12, fontweight='bold')
    axs_gpu[0].set_xscale('log', base=2)
    axs_gpu[0].set_xticks(gpu_bs)
    axs_gpu[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_gpu[0].legend(loc='upper left', frameon=True)
    axs_gpu[0].grid(True, which="both", ls="--", alpha=0.5)

    # 1.2 GPU Throughput
    axs_gpu[1].plot(gpu_bs, gpu_throughput_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_gpu[1].plot(gpu_bs, gpu_throughput_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            gain_pct = (gpu_throughput_b[i] - gpu_throughput_a[i]) / gpu_throughput_a[i] * 100.0
            y_mid = (gpu_throughput_a[i] + gpu_throughput_b[i]) / 2.0
            axs_gpu[1].annotate("", xy=(bs, gpu_throughput_b[i]), xytext=(bs, gpu_throughput_a[i]),
                                arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
            axs_gpu[1].text(bs * 1.12, y_mid, f"+{gain_pct:.1f}%", 
                        color=color_dgreen, fontweight='bold', fontsize=10,
                        va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_gpu[1].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_gpu[1].set_ylabel('Throughput (clips/second)', fontweight='bold')
    axs_gpu[1].set_title('GPU Throughput vs Batch Size (Higher is Better)', fontsize=12, fontweight='bold')
    axs_gpu[1].set_xscale('log', base=2)
    axs_gpu[1].set_xticks(gpu_bs)
    axs_gpu[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_gpu[1].legend(loc='upper left', frameon=True)
    axs_gpu[1].grid(True, which="both", ls="--", alpha=0.5)

    # 1.3 GPU Latency Saved %
    bars_gpu = axs_gpu[2].bar([str(bs) for bs in gpu_bs], gpu_saved, color=color_sgreen, edgecolor='none', width=0.5)
    axs_gpu[2].set_xlabel('Batch Size', fontweight='bold')
    axs_gpu[2].set_ylabel('Latency Saved (%)', fontweight='bold')
    axs_gpu[2].set_title('Inference Time Saved (%) on GPU', fontsize=12, fontweight='bold')
    axs_gpu[2].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    axs_gpu[2].set_ylim(0, max(gpu_saved) + 8)
    for bar in bars_gpu:
        yval = bar.get_height()
        axs_gpu[2].text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"+{yval:.1f}%", 
                        ha='center', va='bottom', fontweight='bold', color='black')
    axs_gpu[2].grid(True, which="both", ls="--", alpha=0.5)

    dest_gpu = base_dir / "gpu_benchmark_plots.png"
    fig_gpu.savefig(dest_gpu, dpi=300, bbox_inches='tight')
    plt.close(fig_gpu)

    # =========================================================================
    # 2. CPU BENCHMARK PLOTS (1x3 Panel)
    # =========================================================================
    fig_cpu, axs_cpu = plt.subplots(1, 3, figsize=(18, 6))
    fig_cpu.suptitle('Test set inference Benchmark (CPU): Model A (24f) (Top-1: 65.85%) vs Cross Frame Model B (16f) (Top-1: 62.68%)', 
                     fontsize=14, fontweight='bold', y=1.02)

    # 2.1 CPU Latency
    axs_cpu[0].plot(cpu_bs, cpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_cpu[0].plot(cpu_bs, cpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        y_mid = (cpu_latency_a[i] + cpu_latency_b[i]) / 2.0
        axs_cpu[0].annotate("", 
                        xy=(bs, cpu_latency_a[i]), 
                        xytext=(bs, cpu_latency_b[i]),
                        arrowprops=dict(arrowstyle="<->", color=color_purple, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        axs_cpu[0].text(bs * offset_fac, y_mid, f"-{cpu_saved[i]:.1f}%", 
                    color=color_purple, fontweight='bold', fontsize=10,
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_purple, lw=1))
    axs_cpu[0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_cpu[0].set_ylabel('Total Batch Latency (ms)', fontweight='bold')
    axs_cpu[0].set_title('CPU Batch Latency vs Batch Size (Lower is Better)', fontsize=12, fontweight='bold')
    axs_cpu[0].set_xscale('log', base=2)
    axs_cpu[0].set_xticks(cpu_bs)
    axs_cpu[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_cpu[0].legend(loc='upper left', frameon=True)
    axs_cpu[0].grid(True, which="both", ls="--", alpha=0.5)

    # 2.2 CPU Throughput
    axs_cpu[1].plot(cpu_bs, cpu_throughput_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_cpu[1].plot(cpu_bs, cpu_throughput_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        cpu_gain_pct = (cpu_throughput_b[i] - cpu_throughput_a[i]) / cpu_throughput_a[i] * 100.0
        y_mid = (cpu_throughput_a[i] + cpu_throughput_b[i]) / 2.0
        axs_cpu[1].annotate("", 
                        xy=(bs, cpu_throughput_b[i]), 
                        xytext=(bs, cpu_throughput_a[i]),
                        arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        axs_cpu[1].text(bs * offset_fac, y_mid, f"+{cpu_gain_pct:.1f}%", 
                    color=color_dgreen, fontweight='bold', fontsize=10,
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_cpu[1].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_cpu[1].set_ylabel('Throughput (clips/second)', fontweight='bold')
    axs_cpu[1].set_title('CPU Throughput vs Batch Size (Higher is Better)', fontsize=12, fontweight='bold')
    axs_cpu[1].set_xscale('log', base=2)
    axs_cpu[1].set_xticks(cpu_bs)
    axs_cpu[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_cpu[1].legend(loc='upper left', frameon=True)
    axs_cpu[1].grid(True, which="both", ls="--", alpha=0.5)

    # 2.3 CPU Latency Saved %
    bars_cpu = axs_cpu[2].bar([str(bs) for bs in cpu_bs], cpu_saved, color=color_sgreen, edgecolor='none', width=0.2)
    axs_cpu[2].set_xlabel('Batch Size', fontweight='bold')
    axs_cpu[2].set_ylabel('Latency Saved (%)', fontweight='bold')
    axs_cpu[2].set_title('Inference Time Saved (%) on CPU', fontsize=12, fontweight='bold')
    axs_cpu[2].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    axs_cpu[2].set_ylim(0, max(cpu_saved) + 8)
    for bar in bars_cpu:
        yval = bar.get_height()
        axs_cpu[2].text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"+{yval:.1f}%", 
                        ha='center', va='bottom', fontweight='bold', color='black')
    axs_cpu[2].grid(True, which="both", ls="--", alpha=0.5)

    dest_cpu = base_dir / "cpu_benchmark_plots.png"
    fig_cpu.savefig(dest_cpu, dpi=300, bbox_inches='tight')
    plt.close(fig_cpu)

    # =========================================================================
    # 3. COMBINED GPU & CPU BENCHMARK PLOTS (2x3 Panel)
    # =========================================================================
    fig_comb, axs_comb = plt.subplots(2, 3, figsize=(18, 12))
    fig_comb.suptitle('Test set inference Benchmark (GPU & CPU): Model A (24f) (Top-1: 65.85%) vs Cross Frame Model B (16f) (Top-1: 62.68%)', 
                      fontsize=15, fontweight='bold', y=0.98)

    # GPU Latency (Total Batch Latency)
    axs_comb[0, 0].plot(gpu_bs, gpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_comb[0, 0].plot(gpu_bs, gpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            y_mid = (gpu_latency_a[i] + gpu_latency_b[i]) / 2.0
            axs_comb[0, 0].annotate("", xy=(bs, gpu_latency_a[i]), xytext=(bs, gpu_latency_b[i]),
                                    arrowprops=dict(arrowstyle="<->", color=color_purple, lw=1.2))
            axs_comb[0, 0].text(bs * 1.12, y_mid, f"-{gpu_saved[i]:.1f}%", 
                                color=color_purple, fontweight='bold', fontsize=9,
                                va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_purple, lw=1))
    axs_comb[0, 0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_comb[0, 0].set_ylabel('Total Batch Latency (ms)', fontweight='bold')
    axs_comb[0, 0].set_title('GPU Batch Latency (Lower is Better)', fontsize=12, fontweight='bold')
    axs_comb[0, 0].set_xscale('log', base=2)
    axs_comb[0, 0].set_xticks(gpu_bs)
    axs_comb[0, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_comb[0, 0].legend(loc='upper left', frameon=True)
    axs_comb[0, 0].grid(True, which="both", ls="--", alpha=0.5)

    # GPU Throughput
    axs_comb[0, 1].plot(gpu_bs, gpu_throughput_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_comb[0, 1].plot(gpu_bs, gpu_throughput_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            gain_pct = (gpu_throughput_b[i] - gpu_throughput_a[i]) / gpu_throughput_a[i] * 100.0
            y_mid = (gpu_throughput_a[i] + gpu_throughput_b[i]) / 2.0
            axs_comb[0, 1].annotate("", xy=(bs, gpu_throughput_b[i]), xytext=(bs, gpu_throughput_a[i]),
                                    arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
            axs_comb[0, 1].text(bs * 1.12, y_mid, f"+{gain_pct:.1f}%", 
                                color=color_dgreen, fontweight='bold', fontsize=9,
                                va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_comb[0, 1].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_comb[0, 1].set_ylabel('Throughput (clips/second)', fontweight='bold')
    axs_comb[0, 1].set_title('GPU Throughput vs Batch Size (Higher is Better)', fontsize=12, fontweight='bold')
    axs_comb[0, 1].set_xscale('log', base=2)
    axs_comb[0, 1].set_xticks(gpu_bs)
    axs_comb[0, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_comb[0, 1].legend(loc='upper left', frameon=True)
    axs_comb[0, 1].grid(True, which="both", ls="--", alpha=0.5)

    # GPU Saved %
    bars_gpu_comb = axs_comb[0, 2].bar([str(bs) for bs in gpu_bs], gpu_saved, color=color_sgreen, edgecolor='none', width=0.5)
    axs_comb[0, 2].set_xlabel('Batch Size', fontweight='bold')
    axs_comb[0, 2].set_ylabel('Latency Saved (%)', fontweight='bold')
    axs_comb[0, 2].set_title('Inference Time Saved (%) on GPU', fontsize=12, fontweight='bold')
    axs_comb[0, 2].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    axs_comb[0, 2].set_ylim(0, max(gpu_saved) + 8)
    for bar in bars_gpu_comb:
        yval = bar.get_height()
        axs_comb[0, 2].text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"+{yval:.1f}%", 
                            ha='center', va='bottom', fontweight='bold', color='black')
    axs_comb[0, 2].grid(True, which="both", ls="--", alpha=0.5)

    # CPU Latency (Total Batch Latency)
    axs_comb[1, 0].plot(cpu_bs, cpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_comb[1, 0].plot(cpu_bs, cpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        y_mid = (cpu_latency_a[i] + cpu_latency_b[i]) / 2.0
        axs_comb[1, 0].annotate("", xy=(bs, cpu_latency_a[i]), xytext=(bs, cpu_latency_b[i]),
                                arrowprops=dict(arrowstyle="<->", color=color_purple, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        axs_comb[1, 0].text(bs * offset_fac, y_mid, f"-{cpu_saved[i]:.1f}%", 
                            color=color_purple, fontweight='bold', fontsize=9,
                            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_purple, lw=1))
    axs_comb[1, 0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_comb[1, 0].set_ylabel('Total Batch Latency (ms)', fontweight='bold')
    axs_comb[1, 0].set_title('CPU Batch Latency (Lower is Better)', fontsize=12, fontweight='bold')
    axs_comb[1, 0].set_xscale('log', base=2)
    axs_comb[1, 0].set_xticks(cpu_bs)
    axs_comb[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_comb[1, 0].legend(loc='upper left', frameon=True)
    axs_comb[1, 0].grid(True, which="both", ls="--", alpha=0.5)

    # CPU Throughput
    axs_comb[1, 1].plot(cpu_bs, cpu_throughput_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_comb[1, 1].plot(cpu_bs, cpu_throughput_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        cpu_gain_pct = (cpu_throughput_b[i] - cpu_throughput_a[i]) / cpu_throughput_a[i] * 100.0
        y_mid = (cpu_throughput_a[i] + cpu_throughput_b[i]) / 2.0
        axs_comb[1, 1].annotate("", xy=(bs, cpu_throughput_b[i]), xytext=(bs, cpu_throughput_a[i]),
                                arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        axs_comb[1, 1].text(bs * offset_fac, y_mid, f"+{cpu_gain_pct:.1f}%", 
                            color=color_dgreen, fontweight='bold', fontsize=9,
                            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_comb[1, 1].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_comb[1, 1].set_ylabel('Throughput (clips/second)', fontweight='bold')
    axs_comb[1, 1].set_title('CPU Throughput vs Batch Size (Higher is Better)', fontsize=12, fontweight='bold')
    axs_comb[1, 1].set_xscale('log', base=2)
    axs_comb[1, 1].set_xticks(cpu_bs)
    axs_comb[1, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_comb[1, 1].legend(loc='upper left', frameon=True)
    axs_comb[1, 1].grid(True, which="both", ls="--", alpha=0.5)

    # CPU Saved %
    bars_cpu_comb = axs_comb[1, 2].bar([str(bs) for bs in cpu_bs], cpu_saved, color=color_sgreen, edgecolor='none', width=0.2)
    axs_comb[1, 2].set_xlabel('Batch Size', fontweight='bold')
    axs_comb[1, 2].set_ylabel('Latency Saved (%)', fontweight='bold')
    axs_comb[1, 2].set_title('Inference Time Saved (%) on CPU', fontsize=12, fontweight='bold')
    axs_comb[1, 2].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    axs_comb[1, 2].set_ylim(0, max(cpu_saved) + 8)
    for bar in bars_cpu_comb:
        yval = bar.get_height()
        axs_comb[1, 2].text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"+{yval:.1f}%", 
                            ha='center', va='bottom', fontweight='bold', color='black')
    axs_comb[1, 2].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    dest_comb = base_dir / "combined_benchmark_plots.png"
    fig_comb.savefig(dest_comb, dpi=300, bbox_inches='tight')
    plt.close(fig_comb)

    # =========================================================================
    # 4. WRAPPED GPU & CPU PLOTS (3x2 Panel)
    # =========================================================================
    fig_wrap, axs_wrap = plt.subplots(3, 2, figsize=(14, 18))
    fig_wrap.suptitle('Test set inference Benchmark (GPU & CPU): Model A (24f) (Top-1: 65.85%) vs Cross Frame Model B (16f) (Top-1: 62.68%)', 
                      fontsize=15, fontweight='bold', y=0.99)

    # 4.1 GPU Latency (Total Batch Latency)
    axs_wrap[0, 0].plot(gpu_bs, gpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_wrap[0, 0].plot(gpu_bs, gpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            y_mid = (gpu_latency_a[i] + gpu_latency_b[i]) / 2.0
            axs_wrap[0, 0].annotate("", xy=(bs, gpu_latency_a[i]), xytext=(bs, gpu_latency_b[i]),
                                    arrowprops=dict(arrowstyle="<->", color=color_purple, lw=1.2))
            axs_wrap[0, 0].text(bs * 1.12, y_mid, f"-{gpu_saved[i]:.1f}%", 
                                color=color_purple, fontweight='bold', fontsize=9,
                                va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_purple, lw=1))
    axs_wrap[0, 0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_wrap[0, 0].set_ylabel('Total Batch Latency (ms)', fontweight='bold')
    axs_wrap[0, 0].set_title('GPU Batch Latency (Lower is Better)', fontsize=12, fontweight='bold')
    axs_wrap[0, 0].set_xscale('log', base=2)
    axs_wrap[0, 0].set_xticks(gpu_bs)
    axs_wrap[0, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_wrap[0, 0].legend(loc='upper left', frameon=True)
    axs_wrap[0, 0].grid(True, which="both", ls="--", alpha=0.5)

    # 4.2 GPU Throughput
    axs_wrap[0, 1].plot(gpu_bs, gpu_throughput_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_wrap[0, 1].plot(gpu_bs, gpu_throughput_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            gain_pct = (gpu_throughput_b[i] - gpu_throughput_a[i]) / gpu_throughput_a[i] * 100.0
            y_mid = (gpu_throughput_a[i] + gpu_throughput_b[i]) / 2.0
            axs_wrap[0, 1].annotate("", xy=(bs, gpu_throughput_b[i]), xytext=(bs, gpu_throughput_a[i]),
                                    arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
            axs_wrap[0, 1].text(bs * 1.12, y_mid, f"+{gain_pct:.1f}%", 
                                color=color_dgreen, fontweight='bold', fontsize=9,
                                va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_wrap[0, 1].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_wrap[0, 1].set_ylabel('Throughput (clips/second)', fontweight='bold')
    axs_wrap[0, 1].set_title('GPU Throughput vs Batch Size (Higher is Better)', fontsize=12, fontweight='bold')
    axs_wrap[0, 1].set_xscale('log', base=2)
    axs_wrap[0, 1].set_xticks(gpu_bs)
    axs_wrap[0, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_wrap[0, 1].legend(loc='upper left', frameon=True)
    axs_wrap[0, 1].grid(True, which="both", ls="--", alpha=0.5)

    # 4.3 GPU Saved %
    bars_gpu_wrap = axs_wrap[1, 0].bar([str(bs) for bs in gpu_bs], gpu_saved, color=color_sgreen, edgecolor='none', width=0.5)
    axs_wrap[1, 0].set_xlabel('Batch Size', fontweight='bold')
    axs_wrap[1, 0].set_ylabel('Latency Saved (%)', fontweight='bold')
    axs_wrap[1, 0].set_title('Inference Time Saved (%) on GPU', fontsize=12, fontweight='bold')
    axs_wrap[1, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    axs_wrap[1, 0].set_ylim(0, max(gpu_saved) + 8)
    for bar in bars_gpu_wrap:
        yval = bar.get_height()
        axs_wrap[1, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"+{yval:.1f}%", 
                            ha='center', va='bottom', fontweight='bold', color='black')
    axs_wrap[1, 0].grid(True, which="both", ls="--", alpha=0.5)

    # 4.4 CPU Latency (Total Batch Latency)
    axs_wrap[1, 1].plot(cpu_bs, cpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_wrap[1, 1].plot(cpu_bs, cpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        y_mid = (cpu_latency_a[i] + cpu_latency_b[i]) / 2.0
        axs_wrap[1, 1].annotate("", xy=(bs, cpu_latency_a[i]), xytext=(bs, cpu_latency_b[i]),
                                arrowprops=dict(arrowstyle="<->", color=color_purple, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        axs_wrap[1, 1].text(bs * offset_fac, y_mid, f"-{cpu_saved[i]:.1f}%", 
                            color=color_purple, fontweight='bold', fontsize=9,
                            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_purple, lw=1))
    axs_wrap[1, 1].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_wrap[1, 1].set_ylabel('Total Batch Latency (ms)', fontweight='bold')
    axs_wrap[1, 1].set_title('CPU Batch Latency (Lower is Better)', fontsize=12, fontweight='bold')
    axs_wrap[1, 1].set_xscale('log', base=2)
    axs_wrap[1, 1].set_xticks(cpu_bs)
    axs_wrap[1, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_wrap[1, 1].legend(loc='upper left', frameon=True)
    axs_wrap[1, 1].grid(True, which="both", ls="--", alpha=0.5)

    # 4.5 CPU Throughput
    axs_wrap[2, 0].plot(cpu_bs, cpu_throughput_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_wrap[2, 0].plot(cpu_bs, cpu_throughput_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        cpu_gain_pct = (cpu_throughput_b[i] - cpu_throughput_a[i]) / cpu_throughput_a[i] * 100.0
        y_mid = (cpu_throughput_a[i] + cpu_throughput_b[i]) / 2.0
        axs_wrap[2, 0].annotate("", xy=(bs, cpu_throughput_b[i]), xytext=(bs, cpu_throughput_a[i]),
                                arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        axs_wrap[2, 0].text(bs * offset_fac, y_mid, f"+{cpu_gain_pct:.1f}%", 
                            color=color_dgreen, fontweight='bold', fontsize=9,
                            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_wrap[2, 0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_wrap[2, 0].set_ylabel('Throughput (clips/second)', fontweight='bold')
    axs_wrap[2, 0].set_title('CPU Throughput vs Batch Size (Higher is Better)', fontsize=12, fontweight='bold')
    axs_wrap[2, 0].set_xscale('log', base=2)
    axs_wrap[2, 0].set_xticks(cpu_bs)
    axs_wrap[2, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_wrap[2, 0].legend(loc='upper left', frameon=True)
    axs_wrap[2, 0].grid(True, which="both", ls="--", alpha=0.5)

    # 4.6 CPU Latency Saved %
    bars_cpu_wrap = axs_wrap[2, 1].bar([str(bs) for bs in cpu_bs], cpu_saved, color=color_sgreen, edgecolor='none', width=0.2)
    axs_wrap[2, 1].set_xlabel('Batch Size', fontweight='bold')
    axs_wrap[2, 1].set_ylabel('Latency Saved (%)', fontweight='bold')
    axs_wrap[2, 1].set_title('Inference Time Saved (%) on CPU', fontsize=12, fontweight='bold')
    axs_wrap[2, 1].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    axs_wrap[2, 1].set_ylim(0, max(cpu_saved) + 8)
    for bar in bars_cpu_wrap:
        yval = bar.get_height()
        axs_wrap[2, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"+{yval:.1f}%", 
                            ha='center', va='bottom', fontweight='bold', color='black')
    axs_wrap[2, 1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    dest_wrap = base_dir / "combined_benchmark_plots_3x2.png"
    fig_wrap.savefig(dest_wrap, dpi=300, bbox_inches='tight')
    plt.close(fig_wrap)

    # =========================================================================
    # 5. INDIVIDUAL PLOTS (6 separate 1x1 Figures - Total Batch Latency)
    # =========================================================================
    # 5.1 GPU Latency
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gpu_bs, gpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    ax.plot(gpu_bs, gpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            y_mid = (gpu_latency_a[i] + gpu_latency_b[i]) / 2.0
            ax.annotate("", xy=(bs, gpu_latency_a[i]), xytext=(bs, gpu_latency_b[i]),
                        arrowprops=dict(arrowstyle="<->", color=color_purple, lw=1.2))
            ax.text(bs * 1.08, y_mid, f"-{gpu_saved[i]:.1f}%", 
                    color=color_purple, fontweight='bold', fontsize=9,
                    va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_purple, lw=1))
    ax.set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    ax.set_ylabel('Total Batch Latency (ms)', fontweight='bold')
    ax.set_title('GPU Batch Latency vs Batch Size', fontsize=11, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(gpu_bs)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.savefig(indiv_dir / "gpu_latency.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 5.2 GPU Throughput
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gpu_bs, gpu_throughput_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    ax.plot(gpu_bs, gpu_throughput_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            gain_pct = (gpu_throughput_b[i] - gpu_throughput_a[i]) / gpu_throughput_a[i] * 100.0
            y_mid = (gpu_throughput_a[i] + gpu_throughput_b[i]) / 2.0
            ax.annotate("", xy=(bs, gpu_throughput_b[i]), xytext=(bs, gpu_throughput_a[i]),
                        arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
            ax.text(bs * 1.08, y_mid, f"+{gain_pct:.1f}%", 
                    color=color_dgreen, fontweight='bold', fontsize=9,
                    va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    ax.set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    ax.set_ylabel('Throughput (clips/second)', fontweight='bold')
    ax.set_title('GPU Throughput vs Batch Size', fontsize=11, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(gpu_bs)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=True)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.savefig(indiv_dir / "gpu_throughput.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 5.3 GPU Latency Saved %
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([str(bs) for bs in gpu_bs], gpu_saved, color=color_sgreen, edgecolor='none', width=0.5)
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Latency Saved (%)', fontweight='bold')
    ax.set_title('Inference Time Saved (%) on GPU', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax.set_ylim(0, max(gpu_saved) + 8)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"+{yval:.1f}%", 
                ha='center', va='bottom', fontweight='bold', color='black')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.savefig(indiv_dir / "gpu_latency_saved.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 5.4 CPU Latency
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cpu_bs, cpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    ax.plot(cpu_bs, cpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        y_mid = (cpu_latency_a[i] + cpu_latency_b[i]) / 2.0
        ax.annotate("", xy=(bs, cpu_latency_a[i]), xytext=(bs, cpu_latency_b[i]),
                    arrowprops=dict(arrowstyle="<->", color=color_purple, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        ax.text(bs * offset_fac, y_mid, f"-{cpu_saved[i]:.1f}%", 
                color=color_purple, fontweight='bold', fontsize=9,
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_purple, lw=1))
    ax.set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    ax.set_ylabel('Total Batch Latency (ms)', fontweight='bold')
    ax.set_title('CPU Batch Latency vs Batch Size', fontsize=11, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(cpu_bs)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.savefig(indiv_dir / "cpu_latency.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 5.5 CPU Throughput
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cpu_bs, cpu_throughput_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    ax.plot(cpu_bs, cpu_throughput_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        cpu_gain_pct = (cpu_throughput_b[i] - cpu_throughput_a[i]) / cpu_throughput_a[i] * 100.0
        y_mid = (cpu_throughput_a[i] + cpu_throughput_b[i]) / 2.0
        ax.annotate("", xy=(bs, cpu_throughput_b[i]), xytext=(bs, cpu_throughput_a[i]),
                    arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        ax.text(bs * offset_fac, y_mid, f"+{cpu_gain_pct:.1f}%", 
                color=color_dgreen, fontweight='bold', fontsize=9,
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    ax.set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    ax.set_ylabel('Throughput (clips/second)', fontweight='bold')
    ax.set_title('CPU Throughput vs Batch Size', fontsize=11, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(cpu_bs)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=True)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.savefig(indiv_dir / "cpu_throughput.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 5.6 CPU Latency Saved %
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([str(bs) for bs in cpu_bs], cpu_saved, color=color_sgreen, edgecolor='none', width=0.2)
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Latency Saved (%)', fontweight='bold')
    ax.set_title('Inference Time Saved (%) on CPU', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax.set_ylim(0, max(cpu_saved) + 8)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"+{yval:.1f}%", 
                ha='center', va='bottom', fontweight='bold', color='black')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.savefig(indiv_dir / "cpu_latency_saved.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"GPU plots saved to: {dest_gpu}")
    print(f"CPU plots saved to: {dest_cpu}")
    print(f"Combined plots (2x3) saved to: {dest_comb}")
    print(f"Combined plots (3x2) saved to: {dest_wrap}")
    print(f"6 Individual plots saved in folder: {indiv_dir}")

if __name__ == "__main__":
    main()
