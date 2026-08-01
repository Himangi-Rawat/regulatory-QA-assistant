# 📋 Regulatory Q&A Assistant

An AI-powered chatbot that lets engineers and analysts ask natural-language
questions about regulatory and compliance PDFs and get answers with exact
page-level citations — instead of manually searching through long documents.

Built as a **Retrieval-Augmented Generation (RAG)** system: instead of
asking an LLM to answer from memory (which risks hallucinated or outdated
information), the system retrieves the actual relevant text from the
uploaded PDF first, then asks the LLM to answer *only* from that text —
with every claim tied back to a specific document and page number.

---

## Problem it solves

Checking compliance rules or prepping tender responses currently means
manually digging through long regulatory PDFs — slow, and risky, since
misquoting or missing a clause can have real consequences. This tool turns
that into: type a question, get an answer, see the exact clause and page
it came from, verifiable in one click.

---

## How it works

```
PDF file
   │
   ▼
Extract text PAGE BY PAGE (keeping page numbers)
   │
   ▼
Split each page into overlapping chunks, tagged with {doc_name, page_number}
   │
   ▼
Embed each chunk (Gemini embedding API) and store in a FAISS vector index
   │
   ▼
(index saved to disk — survives restarts, no re-uploading needed)

User question
   │
   ▼
Embed the question → retrieve the most relevant chunks
   │
   ▼
Ask Gemini to answer using ONLY those chunks, citing [Source: file, Page X]
   │
   ▼
Answer shown with expandable "view source" — the real text behind every citation
```

### Key design decisions

- **Page-level extraction before chunking** — prevents chunks from losing
  their correct page number, which would break citations.
- **Overlapping chunks** — stops a clause from being cut in half at a
  chunk boundary.
- **Batched embedding calls** — chunks are embedded in batches of 50 per
  API call instead of one call per chunk, keeping the system usable even
  on large documents without hitting API rate limits.
- **Citation-enforcing prompt** — the model is explicitly instructed to
  say "I could not find this in the provided documents" rather than
  guess, and to cite a source for every claim it makes.
- **Persistence** — processed documents are saved to disk, so the index
  survives an app restart and doesn't need re-uploading for every session.
- **Password-gated + shared API key** — a lightweight access gate backed
  by Streamlit secrets, so a small team can share one tool without each
  person needing their own API key.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| PDF text extraction | PyMuPDF (`fitz`) | Reliable page-level extraction |
| Chunking | Custom, character-based with overlap | Simple, predictable, dependency-free |
| Embeddings | Gemini `gemini-embedding-001` | Keeps the whole stack on one API |
| Vector search | FAISS (`IndexFlatL2`) | Runs locally, no external database needed |
| Answer generation | Gemini `gemini-flash-latest` | Fast, cost-efficient for grounded Q&A |
| UI | Streamlit | Fast to build and demo |

---

## Setup

### 1. Get a free Gemini API key
https://aistudio.google.com/apikey

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure secrets
Create `.streamlit/secrets.toml` (never committed to GitHub):
```toml
APP_PASSWORD = "your-chosen-password"
GEMINI_API_KEY = "your-actual-gemini-api-key"
```

### 4. Run
```bash
streamlit run app.py
```

---

## Using it

1. Enter the office password
2. Upload one or more compliance/regulatory PDFs
3. Click **Process documents**
4. Ask a question — get an answer with clickable, verifiable citations

---

## File structure

```
regulatory_qa_assistant/
├── app.py              # Streamlit UI — entry point
├── pdf_processor.py    # PDF → per-page text extraction
├── chunker.py          # Per-page text → overlapping chunks with metadata
├── vector_store.py     # Embeddings + FAISS index (batched, rate-limit safe)
├── rag_pipeline.py     # Retrieval + citation-enforced Gemini prompt
├── requirements.txt
├── .streamlit/
│   ├── config.toml     # Theme (committed)
│   └── secrets.toml    # API key + password (NOT committed)
```

---

## Known limitations / future work

- **Scanned/image-only PDFs aren't supported** — the app detects this and
  raises a clear error rather than silently failing. OCR (e.g. via
  `pytesseract`) would be the natural next addition.
- **Free-tier API limits** — the Gemini free tier has both per-minute and
  daily request caps; batching keeps usage well within them for normal use,
  but a production deployment would move to a paid tier for guaranteed headroom.
- **Single shared API key** — fine for a small trusted team behind a
  password gate; a larger rollout would move the key to a properly
  managed server-side secret with per-user usage tracking.
- **No multi-turn memory** — each question is answered independently;
  follow-up questions don't yet carry context from earlier in the conversation.
- **Exact brute-force vector search** — fine at current scale; a very
  large document set would benefit from an approximate index like
  `IndexIVFFlat`.

---

## Author

Built by Himangi Rawat as an internship deliverable — a proof-of-concept
RAG pipeline demonstrating retrieval-grounded, citation-safe question
answering over regulatory and compliance documents.
