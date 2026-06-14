import sys

try:
    import pypdf
    reader = pypdf.PdfReader("PRESENTAZIONE/presentation (1).pdf")
    with open("scratch/pdf_dump_utf8.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            f.write(f"--- PAGE {i+1} ---\n")
            f.write(page.extract_text() + "\n")
except ImportError:
    import fitz # PyMuPDF
    doc = fitz.open("PRESENTAZIONE/presentation (1).pdf")
    with open("scratch/pdf_dump_utf8.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(doc):
            f.write(f"--- PAGE {i+1} ---\n")
            f.write(page.get_text() + "\n")
