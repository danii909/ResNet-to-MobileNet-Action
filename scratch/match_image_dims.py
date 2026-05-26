import os
from PIL import Image

def get_image_info(path):
    try:
        with Image.open(path) as img:
            return img.size, img.mode
    except Exception:
        return None

def main():
    # 1. Get info for extracted images
    extracted_info = {}
    ext_dir = "scratch/extracted_images"
    for file in os.listdir(ext_dir):
        path = os.path.join(ext_dir, file)
        info = get_image_info(path)
        if info:
            extracted_info[file] = info
            print(f"Extracted: {file} -> Size: {info[0]}, Mode: {info[1]}")

    print("\nSearching in results/ for size matches...")
    # 2. Search in results/
    for root, dirs, files in os.walk("results"):
        for file in files:
            if file.lower().endswith(".png"):
                path = os.path.join(root, file)
                info = get_image_info(path)
                if info:
                    # check if it matches any extracted size
                    for ext_name, ext_size in extracted_info.items():
                        if info[0] == ext_size[0]:
                            print(f"Match size {info[0]}! Extracted: {ext_name} <==> Repo file: {path}")

if __name__ == "__main__":
    main()
