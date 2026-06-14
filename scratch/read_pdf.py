import sys
sys.stdout.reconfigure(encoding='utf-8')
try:
    import pypdf
    reader = pypdf.PdfReader("PRESENTAZIONE/presentation (1).pdf")
    for i, page in enumerate(reader.pages):
        print(f"--- PAGE {i+1} ---")
        print(page.extract_text())
except ImportError:
    import fitz # PyMuPDF
    doc = fitz.open("PRESENTAZIONE/presentation (1).pdf")
    for i, page in enumerate(doc):
        print(f"--- PAGE {i+1} ---")
        print(page.get_text())
