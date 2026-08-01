"""
pdf_processor.py
----------------
Extracts text from a PDF page-by-page, keeping track of the page number
for every piece of text.
"""

import fitz  # PyMuPDF
from dataclasses import dataclass


@dataclass
class PageText:
    """One page of a PDF, with its number and cleaned text."""
    doc_name: str
    page_number: int  # 1-indexed, human-friendly
    text: str


def clean_text(text: str) -> str:
    """Collapse weird whitespace/line breaks that PDFs love to produce."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop empty lines
    return " ".join(lines)


def extract_pages(pdf_path: str) -> list[PageText]:
    """
    Open a PDF and return a list of PageText objects, one per page.
    Skips pages with no extractable text (e.g. blank pages, pure images)
    and prints a warning so you know if OCR might be needed.
    """
    doc_name = pdf_path.split("/")[-1]
    pages = []

    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            raw_text = page.get_text()
            cleaned = clean_text(raw_text)

            if not cleaned:
                print(f"[warning] {doc_name} page {i + 1} has no extractable "
                      f"text (likely scanned/image-only — OCR would be needed).")
                continue

            pages.append(PageText(doc_name=doc_name, page_number=i + 1, text=cleaned))

    if not pages:
        raise ValueError(
            f"No extractable text found anywhere in {doc_name}. "
            f"This PDF is probably scanned images, not real text."
        )

    return pages


if __name__ == "__main__":
    # Quick manual test — run: python pdf_processor.py path/to/file.pdf
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path_to_pdf>")
    else:
        result = extract_pages(sys.argv[1])
        print(f"Extracted {len(result)} pages.")
        print(f"Page 1 preview: {result[0].text[:200]}...")