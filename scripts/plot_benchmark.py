import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Data
    batch_sizes = [1, 4, 8, 16, 32, 64]
    gpu_latency_a = [3.27, 3.47, 8.07, 18.71, 43.30, 95.25]
    gpu_throughput_a = [305.7, 1153.1, 991.3, 855.3, 739.1, 671.9]

    gpu_latency_b = [3.31, 3.23, 4.78, 11.75, 26.35, 60.56]
    gpu_throughput_b = [301.8, 1239.8, 1673.1, 1361.9, 1214.5, 1056.8]

    saved_pct = [-1.30, 7.00, 40.75, 37.20, 39.15, 36.42]

    cpu_bs = [1, 8]
    cpu_latency_a = [53.52, 480.17]
    cpu_latency_b = [25.36, 296.72]

    # Setup styles
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Test set inference Benchmark: Model A (24f) (Top-1: 65.85%) vs Cross Frame Model B (16f) (Top-1: 62.68%)', fontsize=14, fontweight='bold', y=0.98)

    # Colors
    color_a = '#e74c3c'  # Soft Red
    color_b = '#3498db'  # Soft Blue
    color_sgreen = '#2ecc71'  # Soft Green
    color_dgreen = '#1e8449'  # Dark Green
    color_purple = '#8e44ad'  # Purple

    # 1. GPU Latency
    axs[0, 0].plot(batch_sizes, gpu_latency_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs[0, 0].plot(batch_sizes, gpu_latency_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    
    # Annotate GPU Latency Gap
    for i, bs in enumerate(batch_sizes):
        if bs >= 8:
            y_mid = (gpu_latency_a[i] + gpu_latency_b[i]) / 2.0
            # Vertical double arrow
            axs[0, 0].annotate("", 
                               xy=(bs, gpu_latency_a[i]), 
                               xytext=(bs, gpu_latency_b[i]),
                               arrowprops=dict(arrowstyle="<->", color=color_purple, lw=1.2))
            # Text box with percentage saved
            axs[0, 0].text(bs * 1.12, y_mid, f"-{saved_pct[i]:.1f}%", 
                           color=color_purple, fontweight='bold', fontsize=9,
                           va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_purple, lw=1))

    axs[0, 0].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs[0, 0].set_ylabel('Inference Latency (ms)', fontweight='bold')
    axs[0, 0].set_title('GPU Latency vs Batch Size (Lower is Better)', fontsize=12, fontweight='bold')
    axs[0, 0].set_xscale('log', base=2)
    axs[0, 0].set_xticks(batch_sizes)
    axs[0, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs[0, 0].legend(loc='upper left', frameon=True)
    axs[0, 0].grid(True, which="both", ls="--", alpha=0.5)

    # 2. GPU Throughput
    axs[0, 1].plot(batch_sizes, gpu_throughput_a, 'o-', color=color_a, label='Model A', linewidth=2.5, markersize=8)
    axs[0, 1].plot(batch_sizes, gpu_throughput_b, 's-', color=color_b, label='Model B', linewidth=2.5, markersize=8)
    
    # Annotate GPU Throughput Gap (percentage increase)
    for i, bs in enumerate(batch_sizes):
        if bs >= 8:
            gain_pct = (gpu_throughput_b[i] - gpu_throughput_a[i]) / gpu_throughput_a[i] * 100.0
            y_mid = (gpu_throughput_a[i] + gpu_throughput_b[i]) / 2.0
            # Vertical double arrow
            axs[0, 1].annotate("", 
                               xy=(bs, gpu_throughput_b[i]), 
                               xytext=(bs, gpu_throughput_a[i]),
                               arrowprops=dict(arrowstyle="<->", color=color_dgreen, lw=1.2))
            # Text box with percentage gain
            axs[0, 1].text(bs * 1.12, y_mid, f"+{gain_pct:.1f}%", 
                           color=color_dgreen, fontweight='bold', fontsize=9,
                           va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_dgreen, lw=1))

    axs[0, 1].set_xlabel('Batch Size (Log Scale)', fontweight='bold')
    axs[0, 1].set_ylabel('Throughput (clips/second)', fontweight='bold')
    axs[0, 1].set_title('GPU Throughput vs Batch Size (Higher is Better)', fontsize=12, fontweight='bold')
    axs[0, 1].set_xscale('log', base=2)
    axs[0, 1].set_xticks(batch_sizes)
    axs[0, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axs[0, 1].legend(frameon=True)
    axs[0, 1].grid(True, which="both", ls="--", alpha=0.5)

    # 3. GPU Latency Saved %
    bars = axs[1, 0].bar([str(bs) for bs in batch_sizes], saved_pct, color=[color_a if x < 0 else color_sgreen for x in saved_pct], edgecolor='none', width=0.5)
    axs[1, 0].set_xlabel('Batch Size', fontweight='bold')
    axs[1, 0].set_ylabel('Latency Saved (%)', fontweight='bold')
    axs[1, 0].set_title('Inference Time Saved (%) with Model B (16f)', fontsize=12, fontweight='bold')
    axs[1, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    axs[1, 0].set_ylim(min(saved_pct) - 5, max(saved_pct) + 8)
    
    for bar in bars:
        yval = bar.get_height()
        va_dir = 'bottom' if yval >= 0 else 'top'
        y_offset = 1.5 if yval >= 0 else -1.5
        axs[1, 0].text(bar.get_x() + bar.get_width()/2.0, yval + y_offset, f"{yval:+.1f}%", ha='center', va=va_dir, fontweight='bold', color='black')

    # 4. CPU Latency Comparison
    x = np.arange(len(cpu_bs))
    width = 0.35
    axs[1, 1].bar(x - width/2, cpu_latency_a, width, label='Model A', color=color_a)
    axs[1, 1].bar(x + width/2, cpu_latency_b, width, label='Model B', color=color_b)
    
    # Annotate CPU savings
    for i, bs in enumerate(cpu_bs):
        val_a = cpu_latency_a[i]
        val_b = cpu_latency_b[i]
        cpu_saved = (val_a - val_b) / val_a * 100.0
        y_mid = (val_a + val_b) / 2.0
        # Text box in the middle of the CPU bars
        axs[1, 1].text(i, y_mid, f"-{cpu_saved:.1f}%", 
                       color='white', fontweight='bold', fontsize=10,
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", fc=color_purple, ec="none", alpha=0.9))

    axs[1, 1].set_ylabel('Latency (ms)', fontweight='bold')
    axs[1, 1].set_title('CPU Latency Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels([f"BS {bs}" for bs in cpu_bs], fontweight='bold')
    axs[1, 1].legend(frameon=True)
    axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)

    # Add text labels on CPU bars
    for i, (val_a, val_b) in enumerate(zip(cpu_latency_a, cpu_latency_b)):
        axs[1, 1].text(i - width/2, val_a + 5, f"{val_a:.1f}ms", ha='center', va='bottom', fontsize=9)
        axs[1, 1].text(i + width/2, val_b + 5, f"{val_b:.1f}ms", ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to both folders
    dest1 = Path("slurm-train-eval-4675/benchmark_plots.png")
    dest2 = Path("results/Train-eval-test-split (group-aware)/Training/slurm-train-eval-4675/benchmark_plots.png")
    
    dest1.parent.mkdir(parents=True, exist_ok=True)
    dest2.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(dest1, dpi=300, bbox_inches='tight')
    plt.savefig(dest2, dpi=300, bbox_inches='tight')
    print(f"Plots saved successfully to:")
    print(f"  - {dest1}")
    print(f"  - {dest2}")

if __name__ == "__main__":
    main()
