# 📋 Regulatory Q&A Assistant

### AI-Powered Regulatory & Compliance Question Answering using RAG

> An intelligent Retrieval-Augmented Generation (RAG) chatbot that enables engineers, analysts, and compliance teams to ask natural-language questions about regulatory and compliance documents and receive accurate, citation-backed answers directly from the source PDFs.

Built using **Google Gemini, FAISS, Streamlit, and PyMuPDF**, the assistant retrieves relevant document sections before generating responses, ensuring every answer is grounded in the uploaded documents rather than the model's internal knowledge.

---

## 🎯 Project Overview

Organizations often rely on lengthy regulatory documents, standards, and compliance manuals. Finding the correct clause manually is time-consuming and increases the risk of missing or misquoting important information.

The **Regulatory Q&A Assistant** simplifies this process by allowing users to upload regulatory PDFs and ask questions in natural language. Instead of searching documents manually, users receive concise answers along with exact page-level citations that can be verified instantly.

---

## ✨ Features

- 📄 Upload one or multiple regulatory/compliance PDFs
- 🤖 Natural-language question answering
- 📑 Accurate page-level citations
- 🔍 Semantic search using vector embeddings
- 💾 Persistent FAISS index (documents remain processed after restart)
- 🔐 Password-protected access using Streamlit Secrets
- 📚 Expandable source viewer for complete transparency
- ⚡ Fast retrieval with batched embedding generation

---

# 🚀 How It Works

```
                    Regulatory PDFs
                           │
                           ▼
             Page-wise Text Extraction (PyMuPDF)
                           │
                           ▼
         Chunking with Overlap + Metadata
       {Document Name + Page Number}
                           │
                           ▼
      Gemini Embeddings → FAISS Vector Store
                           │
─────────────────────────────────────────────────
                           │
                    User Question
                           │
                           ▼
              Semantic Vector Search
                           │
                           ▼
          Retrieve Relevant Document Chunks
                           │
                           ▼
         Gemini Flash generates an answer
      using ONLY retrieved document context
                           │
                           ▼
      Answer + Page-Level Citations + Source
```

---

# 🏗️ System Design

The application follows a Retrieval-Augmented Generation (RAG) architecture.

### 1. Document Processing

- PDFs are processed page-by-page.
- Page numbers are preserved before chunking.
- Each chunk stores:
  - Document name
  - Page number
  - Text content

### 2. Embedding Generation

Document chunks are converted into vector embeddings using:

**Gemini Embedding API (`gemini-embedding-001`)**

Embeddings are generated in batches to improve efficiency and reduce API usage.

### 3. Vector Storage

Embeddings are stored locally using **FAISS**, enabling fast semantic similarity search without requiring an external vector database.

### 4. Retrieval

For every user question:

- Generate embedding
- Search FAISS
- Retrieve top matching chunks

### 5. Answer Generation

Retrieved chunks are passed to **Gemini Flash** with a prompt that instructs the model to:

- Answer **only** using retrieved information
- Never hallucinate
- Cite every factual statement
- Clearly state when information is unavailable

---

# ⭐ Key Design Decisions

### Page-Level Extraction

Extracting text before chunking preserves accurate page numbers, enabling reliable citations.

### Overlapping Chunks

Chunk overlap prevents important clauses from being split across chunk boundaries.

### Batched Embeddings

Embedding requests are processed in batches of 50 chunks, reducing latency and staying within API rate limits.

### Citation Enforcement

The prompt explicitly instructs the model to avoid unsupported answers and provide citations for every response.

### Persistent Storage

Processed documents are saved locally, eliminating the need to reprocess PDFs after restarting the application.

### Secure Access

The application uses Streamlit Secrets for API key management and password-based authentication.

---

# 🛠️ Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF (fitz) |
| Text Chunking | Custom Overlapping Chunker |
| Embeddings | Gemini Embedding API |
| Vector Database | FAISS |
| LLM | Gemini Flash |
| Secret Management | Streamlit Secrets |

---

# 📂 Project Structure

```
regulatory_qa_assistant/

├── app.py
├── pdf_processor.py
├── chunker.py
├── vector_store.py
├── rag_pipeline.py
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Himangi-Rawat/regulatory-qa-assistant.git
cd regulatory-qa-assistant
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Secrets

Create:

```
.streamlit/secrets.toml
```

```toml
APP_PASSWORD = "your-password"

GEMINI_API_KEY = "your-api-key"
```

---

## Run

```bash
streamlit run app.py
```

---

# 💻 Usage

1. Launch the application.
2. Enter the application password.
3. Upload one or more regulatory PDFs.
4. Click **Process Documents**.
5. Ask questions in natural language.
6. View answers with page-level citations and expandable source text.

---

# 📌 Example Questions

- What are the fire safety requirements?
- What documents are required for tender submission?
- What are the environmental compliance guidelines?
- What penalties are mentioned for non-compliance?
- Which clause discusses employee safety?

---

# 📊 Architecture Highlights

✔ Retrieval-Augmented Generation (RAG)

✔ Semantic Search

✔ Citation-Based Responses

✔ Persistent Vector Store

✔ Secure API Key Handling

✔ Multi-document Question Answering

---

# ⚠️ Current Limitations

- OCR is not supported for scanned PDFs.
- Each query is independent (no conversational memory).
- Uses exact FAISS search (IndexFlatL2).
- Optimized for small to medium document collections.
- Free Gemini API limits apply.

---

# 🚀 Future Enhancements

- OCR support using Tesseract
- Multi-turn conversational memory
- Hybrid Search (BM25 + Vector Search)
- Metadata filtering
- Approximate FAISS indexing
- Cloud deployment
- User authentication
- Conversation history
- Feedback collection

---


# 👩‍💻 Author

**Himangi Rawat**

B.Tech Computer Science & Engineering

Jaypee Institute of Information Technology, Noida

---

# 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

### Built to simplify regulatory compliance through trustworthy, citation-backed AI.

</div>
