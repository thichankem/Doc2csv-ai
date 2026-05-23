"""PDF text extraction using pdfplumber."""
from pathlib import Path

import pdfplumber


def extract_pdf(path: str) -> str:
    """Extract all text from a PDF, page-by-page with blank-line separators."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File không tồn tại: {path}")

    pages = []
    with pdfplumber.open(p) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n\n".join(pages)
