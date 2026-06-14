import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Try to read the PDF
try:
    import fitz  # PyMuPDF
    doc = fitz.open(r'PRESENTAZIONE/presentation (1).pdf')
    print(f"Pages: {len(doc)}")
    for i, page in enumerate(doc):
        text = page.get_text()
        print(f"\n=== Page {i+1} ===")
        print(text[:600])
    doc.close()
except ImportError:
    print("PyMuPDF not available, trying pypdf")
    try:
        from pypdf import PdfReader
        reader = PdfReader(r'PRESENTAZIONE/presentation (1).pdf')
        print(f"Pages: {len(reader.pages)}")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"\n=== Page {i+1} ===")
            print(text[:600] if text else "(no text)")
    except ImportError:
        print("pypdf also not available")
