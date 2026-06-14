import os
from PIL import Image

def main():
    paths = []
    for root, dirs, files in os.walk("results/Test_Set_Benchmark"):
        for f in files:
            if f.lower().endswith(".png"):
                paths.append(os.path.join(root, f))
                
    for p in paths:
        try:
            with Image.open(p) as img:
                print(f"File: {p}, Size: {img.size}")
        except Exception as e:
            print(f"Error {p}: {e}")

if __name__ == "__main__":
    main()
