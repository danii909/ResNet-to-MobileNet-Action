import hashlib
import os
import glob
from pptx import Presentation

def get_md5(data):
    return hashlib.md5(data).hexdigest()

def get_file_md5(path):
    with open(path, "rb") as f:
        return get_md5(f.read())

def main():
    # 1. Map all PNG files in the repo and in AppData to their MD5
    png_map = {}
    search_dirs = [
        ".",
        r"C:\Users\danie\.gemini\antigravity-ide\brain\252529aa-c502-4354-bf5e-4b146a704881\.tempmediaStorage"
    ]
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for root, dirs, files in os.walk(sdir):
            if ".venv" in root or ".git" in root:
                continue
            for file in files:
                if file.lower().endswith(".png") or file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                    path = os.path.join(root, file)
                    try:
                        h = get_file_md5(path)
                        png_map[h] = path
                    except Exception as e:
                        pass

    # 2. Inspect slides in PPTX
    prs = Presentation("PRESENTAZIONE/presentation_v3.pptx")
    for i, slide in enumerate(prs.slides):
        print(f"\n--- Slide {i+1} ---")
        for j, shape in enumerate(slide.shapes):
            if shape.shape_type == 13: # Picture
                image = shape.image
                blob = image.blob
                h = get_md5(blob)
                matched_path = png_map.get(h, "UNKNOWN")
                print(f"Shape {j} ({shape.name}): Image MD5={h}, Matches file={matched_path}")
                print(f"  Pos: Left={shape.left/914400:.2f}in, Top={shape.top/914400:.2f}in, Width={shape.width/914400:.2f}in, Height={shape.height/914400:.2f}in")

if __name__ == "__main__":
    main()
