import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Data from slurm-benchmark-testset-4700.log
    
    # GPU Data
    gpu_bs = [1, 8, 16, 32]
    gpu_latency_a = [4.54, 1.03, 1.18, 1.37]
    gpu_throughput_a = [220.4, 971.6, 846.8, 731.1]
    gpu_latency_b = [4.02, 0.62, 0.74, 0.82]
    gpu_throughput_b = [249.0, 1601.3, 1347.3, 1212.4]
    gpu_saved = [11.49, 39.32, 37.15, 39.69]

    # CPU Data
    cpu_bs = [1, 8]
    cpu_latency_a = [136.70, 113.51]
    cpu_latency_b = [88.51, 71.13]
    # Re-calculate exact throughput from ms latency: 1000 / latency
    cpu_throughput_a = [1000.0 / x for x in cpu_latency_a]
    cpu_throughput_b = [1000.0 / x for x in cpu_latency_b]
    cpu_saved = [35.26, 37.33]

    # Colors
    color_a = '#e74c3c'  # Soft Red
    color_b = '#3498db'  # Soft Blue
    color_sgreen = '#2ecc71'  # Soft Green
    color_dgreen = '#1e8449'  # Dark Green

    # Setup styles
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # =========================================================================
    # 1. GPU BENCHMARK PLOTS (1x3 Panel)
    # =========================================================================
    fig_gpu, axs_gpu = plt.subplots(1, 3, figsize=(18, 6))
    fig_gpu.suptitle('Real Test Set GPU Inference Benchmark: Model A (24f) vs Model B (16f)', 
                     fontsize=14, fontweight='bold', y=1.05)

    # 1.1 GPU Latency
    axs_gpu[0].plot(gpu_bs, gpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_gpu[0].plot(gpu_bs, gpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            y_mid = (gpu_latency_a[i] + gpu_latency_b[i]) / 2.0
            axs_gpu[0].annotate("", xy=(bs, gpu_latency_a[i]), xytext=(bs, gpu_latency_b[i]),
                                arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
            axs_gpu[0].text(bs * 1.12, y_mid, f"-{gpu_saved[i]:.1f}%", 
                        color=color_dgreen, fontweight='bold', fontsize=10,
                        va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_gpu[0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_gpu[0].set_ylabel('Inference Latency (ms)', fontweight='bold')
    axs_gpu[0].set_title('GPU Latency per clip (Lower is Better)', fontsize=12, fontweight='bold')
    axs_gpu[0].set_xscale('log', base=2)
    axs_gpu[0].set_xticks(gpu_bs)
    axs_gpu[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_gpu[0].legend(loc='upper right', frameon=True)
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

    dest_gpu = Path("results/Test_Set_Benchmark/gpu_benchmark_plots.png")
    dest_gpu.parent.mkdir(parents=True, exist_ok=True)
    fig_gpu.savefig(dest_gpu, dpi=300, bbox_inches='tight')
    plt.close(fig_gpu)

    # =========================================================================
    # 2. CPU BENCHMARK PLOTS (1x3 Panel)
    # =========================================================================
    fig_cpu, axs_cpu = plt.subplots(1, 3, figsize=(18, 6))
    fig_cpu.suptitle('Real Test Set CPU Inference Benchmark: Model A (24f) vs Model B (16f)', 
                     fontsize=14, fontweight='bold', y=1.05)

    # 2.1 CPU Latency
    axs_cpu[0].plot(cpu_bs, cpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_cpu[0].plot(cpu_bs, cpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        y_mid = (cpu_latency_a[i] + cpu_latency_b[i]) / 2.0
        axs_cpu[0].annotate("", xy=(bs, cpu_latency_a[i]), xytext=(bs, cpu_latency_b[i]),
                        arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        axs_cpu[0].text(bs * offset_fac, y_mid, f"-{cpu_saved[i]:.1f}%", 
                    color=color_dgreen, fontweight='bold', fontsize=10,
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_cpu[0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_cpu[0].set_ylabel('Inference Latency (ms)', fontweight='bold')
    axs_cpu[0].set_title('CPU Latency per clip (Lower is Better)', fontsize=12, fontweight='bold')
    axs_cpu[0].set_xscale('log', base=2)
    axs_cpu[0].set_xticks(cpu_bs)
    axs_cpu[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_cpu[0].legend(loc='upper right', frameon=True)
    axs_cpu[0].grid(True, which="both", ls="--", alpha=0.5)

    # 2.2 CPU Throughput
    axs_cpu[1].plot(cpu_bs, cpu_throughput_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_cpu[1].plot(cpu_bs, cpu_throughput_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        cpu_gain_pct = (cpu_throughput_b[i] - cpu_throughput_a[i]) / cpu_throughput_a[i] * 100.0
        y_mid = (cpu_throughput_a[i] + cpu_throughput_b[i]) / 2.0
        axs_cpu[1].annotate("", xy=(bs, cpu_throughput_b[i]), xytext=(bs, cpu_throughput_a[i]),
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
    bars_cpu = axs_cpu[2].bar([str(bs) for bs in cpu_bs], cpu_saved, color=color_sgreen, edgecolor='none', width=0.4)
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

    dest_cpu = Path("results/Test_Set_Benchmark/cpu_benchmark_plots.png")
    fig_cpu.savefig(dest_cpu, dpi=300, bbox_inches='tight')
    plt.close(fig_cpu)

    # =========================================================================
    # 3. COMBINED GPU & CPU BENCHMARK PLOTS (2x3 Panel)
    # =========================================================================
    fig_comb, axs_comb = plt.subplots(2, 3, figsize=(18, 12))
    fig_comb.suptitle('Real Test Set Inference Benchmark: GPU vs CPU Performance Comparison', 
                      fontsize=16, fontweight='bold', y=0.98)

    # GPU
    axs_comb[0, 0].plot(gpu_bs, gpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_comb[0, 0].plot(gpu_bs, gpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            y_mid = (gpu_latency_a[i] + gpu_latency_b[i]) / 2.0
            axs_comb[0, 0].annotate("", xy=(bs, gpu_latency_a[i]), xytext=(bs, gpu_latency_b[i]),
                                    arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
            axs_comb[0, 0].text(bs * 1.12, y_mid, f"-{gpu_saved[i]:.1f}%", 
                                color=color_dgreen, fontweight='bold', fontsize=9,
                                va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_comb[0, 0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_comb[0, 0].set_ylabel('Inference Latency (ms)', fontweight='bold')
    axs_comb[0, 0].set_title('GPU Latency per clip (Lower is Better)', fontsize=12, fontweight='bold')
    axs_comb[0, 0].set_xscale('log', base=2)
    axs_comb[0, 0].set_xticks(gpu_bs)
    axs_comb[0, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_comb[0, 0].legend(loc='upper right', frameon=True)
    axs_comb[0, 0].grid(True, which="both", ls="--", alpha=0.5)

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

    # CPU
    axs_comb[1, 0].plot(cpu_bs, cpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_comb[1, 0].plot(cpu_bs, cpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        y_mid = (cpu_latency_a[i] + cpu_latency_b[i]) / 2.0
        axs_comb[1, 0].annotate("", xy=(bs, cpu_latency_a[i]), xytext=(bs, cpu_latency_b[i]),
                                arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        axs_comb[1, 0].text(bs * offset_fac, y_mid, f"-{cpu_saved[i]:.1f}%", 
                            color=color_dgreen, fontweight='bold', fontsize=9,
                            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_comb[1, 0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_comb[1, 0].set_ylabel('Inference Latency (ms)', fontweight='bold')
    axs_comb[1, 0].set_title('CPU Latency per clip (Lower is Better)', fontsize=12, fontweight='bold')
    axs_comb[1, 0].set_xscale('log', base=2)
    axs_comb[1, 0].set_xticks(cpu_bs)
    axs_comb[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_comb[1, 0].legend(loc='upper right', frameon=True)
    axs_comb[1, 0].grid(True, which="both", ls="--", alpha=0.5)

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

    bars_cpu_comb = axs_comb[1, 2].bar([str(bs) for bs in cpu_bs], cpu_saved, color=color_sgreen, edgecolor='none', width=0.4)
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
    dest_comb = Path("results/Test_Set_Benchmark/combined_benchmark_plots.png")
    fig_comb.savefig(dest_comb, dpi=300, bbox_inches='tight')
    plt.close(fig_comb)

    # =========================================================================
    # 4. WRAPPED GPU & CPU PLOTS (3x2 Panel) - Goes to next line every 2 plots
    # =========================================================================
    fig_wrap, axs_wrap = plt.subplots(3, 2, figsize=(14, 18))
    fig_wrap.suptitle('Real Test Set Inference Benchmark: GPU & CPU Wrapped Performance Layout', 
                      fontsize=16, fontweight='bold', y=0.99)

    # 4.1.1 GPU Latency -> [0, 0]
    axs_wrap[0, 0].plot(gpu_bs, gpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_wrap[0, 0].plot(gpu_bs, gpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(gpu_bs):
        if bs >= 8:
            y_mid = (gpu_latency_a[i] + gpu_latency_b[i]) / 2.0
            axs_wrap[0, 0].annotate("", xy=(bs, gpu_latency_a[i]), xytext=(bs, gpu_latency_b[i]),
                                    arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
            axs_wrap[0, 0].text(bs * 1.12, y_mid, f"-{gpu_saved[i]:.1f}%", 
                                color=color_dgreen, fontweight='bold', fontsize=9,
                                va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_wrap[0, 0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_wrap[0, 0].set_ylabel('Inference Latency (ms)', fontweight='bold')
    axs_wrap[0, 0].set_title('GPU Latency per clip (Lower is Better)', fontsize=12, fontweight='bold')
    axs_wrap[0, 0].set_xscale('log', base=2)
    axs_wrap[0, 0].set_xticks(gpu_bs)
    axs_wrap[0, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_wrap[0, 0].legend(loc='upper right', frameon=True)
    axs_wrap[0, 0].grid(True, which="both", ls="--", alpha=0.5)

    # 4.1.2 GPU Throughput -> [0, 1]
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

    # 4.1.3 GPU Latency Saved % -> [1, 0]
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

    # 4.2.1 CPU Latency -> [1, 1]
    axs_wrap[1, 1].plot(cpu_bs, cpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs_wrap[1, 1].plot(cpu_bs, cpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    for i, bs in enumerate(cpu_bs):
        y_mid = (cpu_latency_a[i] + cpu_latency_b[i]) / 2.0
        axs_wrap[1, 1].annotate("", xy=(bs, cpu_latency_a[i]), xytext=(bs, cpu_latency_b[i]),
                                arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
        offset_fac = 1.15 if bs == 1 else 0.85
        axs_wrap[1, 1].text(bs * offset_fac, y_mid, f"-{cpu_saved[i]:.1f}%", 
                            color=color_dgreen, fontweight='bold', fontsize=9,
                            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))
    axs_wrap[1, 1].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs_wrap[1, 1].set_ylabel('Inference Latency (ms)', fontweight='bold')
    axs_wrap[1, 1].set_title('CPU Latency per clip (Lower is Better)', fontsize=12, fontweight='bold')
    axs_wrap[1, 1].set_xscale('log', base=2)
    axs_wrap[1, 1].set_xticks(cpu_bs)
    axs_wrap[1, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs_wrap[1, 1].legend(loc='upper right', frameon=True)
    axs_wrap[1, 1].grid(True, which="both", ls="--", alpha=0.5)

    # 4.2.2 CPU Throughput -> [2, 0]
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

    # 4.2.3 CPU Latency Saved % -> [2, 1]
    bars_cpu_wrap = axs_wrap[2, 1].bar([str(bs) for bs in cpu_bs], cpu_saved, color=color_sgreen, edgecolor='none', width=0.4)
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
    dest_wrap = Path("results/Test_Set_Benchmark/combined_benchmark_plots_3x2.png")
    fig_wrap.savefig(dest_wrap, dpi=300, bbox_inches='tight')
    plt.close(fig_wrap)

    print(f"GPU plots saved to: {dest_gpu}")
    print(f"CPU plots saved to: {dest_cpu}")
    print(f"Combined plots (2x3) saved to: {dest_comb}")
    print(f"Combined plots (3x2) saved to: {dest_wrap}")

if __name__ == "__main__":
    main()
