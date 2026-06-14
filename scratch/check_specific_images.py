import os
import hashlib
from PIL import Image

def get_img_info(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    h = hashlib.md5(data).hexdigest()
    with Image.open(path) as img:
        size = img.size
    return {"md5": h, "size": size, "path": path}

def main():
    extracted_dir = "scratch/extracted_images"
    extracted = {}
    for f in os.listdir(extracted_dir):
        p = os.path.join(extracted_dir, f)
        info = get_img_info(p)
        if info:
            extracted[f] = info

    candidates = [
        "results/Train-eval-test-split (group-aware)/distillation_histogram_test_top1.png",
        "results/Train-eval-test-split (group-aware)/distillation_histogram_test_top5.png",
        "results/Test_Set_Benchmark/gpu_benchmark_plots.png",
        "results/Test_Set_Benchmark/cpu_benchmark_plots.png",
        "results/Test_Set_Benchmark/combined_benchmark_plots.png",
        "results/Test_Set_Benchmark/combined_benchmark_plots_3x2.png",
        "results/Test_Set_Benchmark/individual_plots/gpu_latency.png",
        "results/Test_Set_Benchmark/individual_plots/gpu_latency_saved.png",
        "results/Test_Set_Benchmark/individual_plots/cpu_latency.png",
        "results/Test_Set_Benchmark/individual_plots/cpu_latency_saved.png",
        "results/Test_Set_Benchmark/old plots/individual_plots/gpu_latency.png",
        "results/Test_Set_Benchmark/old plots/individual_plots/gpu_latency_saved.png",
        "results/Test_Set_Benchmark/old plots/individual_plots/cpu_latency.png",
        "results/Test_Set_Benchmark/old plots/individual_plots/cpu_latency_saved.png"
    ]
    
    candidate_infos = []
    for c in candidates:
        info = get_img_info(c)
        if info:
            candidate_infos.append(info)

    print("--- Extracted Images ---")
    for name, info in sorted(extracted.items()):
        print(f"{name}: size={info['size']}, md5={info['md5']}")
        # Find matches
        matches = []
        for ci in candidate_infos:
            if ci['md5'] == info['md5']:
                matches.append(f"MD5 match: {ci['path']}")
            elif ci['size'] == info['size']:
                matches.append(f"Size match: {ci['path']}")
        for m in matches:
            print(f"  -> {m}")

if __name__ == "__main__":
    main()
