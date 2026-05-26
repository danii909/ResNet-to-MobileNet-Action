"""
Preview the first few slides of the generated presentation
by converting them to images using LibreOffice or similar.
"""
import sys, io, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Check file size
import os
path = r"PRESENTAZIONE/presentation_v5.pptx"
size = os.path.getsize(path)
print(f"File size: {size/1024/1024:.2f} MB")

# Check slide count with python-pptx
from pptx import Presentation
prs = Presentation(path)
print(f"Slides: {len(prs.slides)}")
for i, slide in enumerate(prs.slides):
    shapes = [s.name for s in slide.shapes]
    print(f"  Slide {i+1}: {len(slide.shapes)} shapes — {shapes[:4]}")
