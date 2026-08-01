"""
chunker.py
----------
Splits each page's text into overlapping chunks, while keeping the
doc_name + page_number attached to every single chunk.
"""

from dataclasses import dataclass
from pdf_processor import PageText


@dataclass
class Chunk:
    doc_name: str
    page_number: int
    chunk_id: int
    text: str


def chunk_page(page: PageText, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split one page's text into overlapping substrings of ~chunk_size
    characters.
    """
    text = page.text
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap  # step back so we don't lose boundary text
    return chunks


def chunk_all_pages(pages: list[PageText], chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Chunk every page and return a flat list of Chunk objects with metadata intact."""
    all_chunks = []
    global_id = 0

    for page in pages:
        page_chunks = chunk_page(page, chunk_size=chunk_size, overlap=overlap)
        for text in page_chunks:
            all_chunks.append(
                Chunk(
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    chunk_id=global_id,
                    text=text,
                )
            )
            global_id += 1

    return all_chunks


if __name__ == "__main__":
    from pdf_processor import extract_pages
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    pages = extract_pages(path)
    chunks = chunk_all_pages(pages)

    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    for c in chunks:
        print(f"  [chunk {c.chunk_id}] {c.doc_name} p.{c.page_number} -> {c.text[:80]}...")