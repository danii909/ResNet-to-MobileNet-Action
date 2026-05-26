import os
from PIL import Image

def main():
    temp_dir = r"C:\Users\danie\.gemini\antigravity-ide\brain\252529aa-c502-4354-bf5e-4b146a704881\.tempmediaStorage"
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            if f.lower().endswith(".png"):
                p = os.path.join(temp_dir, f)
                try:
                    with Image.open(p) as img:
                        print(f"Temp file: {f}, Size: {img.size}")
                except Exception as e:
                    print(f"Error {f}: {e}")
    else:
        print("Temp dir does not exist.")

if __name__ == "__main__":
    main()
