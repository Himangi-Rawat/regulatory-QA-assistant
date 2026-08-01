"""
app.py
------
Streamlit front-end for the Regulatory Q&A Assistant.

Flow:
  1. User enters an office password (gate) before anything else loads.
  2. App uses a shared Gemini API key from .streamlit/secrets.toml if one is
     set there; otherwise falls back to asking the user to paste their own key.
  3. User uploads one or more PDFs -> app extracts + chunks + embeds + indexes them.
  4. User asks a question -> app retrieves relevant chunks -> Gemini answers with citations.
  5. Answer is shown along with an expandable "View source" for every citation,
     so the engineer can verify the answer against the real clause text.
  6. Processed documents are saved to disk (saved_index.faiss/.pkl), so they survive
     an app restart or a new visitor opening the app — no need to re-upload every time.

Run with:  streamlit run app.py
"""

import streamlit as st
import os
import json
import re
from pdf_processor import extract_pages
from chunker import chunk_all_pages
from vector_store import VectorStore, configure_gemini
from rag_pipeline import answer_question

INDEX_PATH = "saved_index"  # files on disk: saved_index.faiss, saved_index.pkl
PROCESSED_FILES_PATH = "saved_index_files.json"  # tracks which filenames are indexed

CITATION_PATTERN = re.compile(r"\[Source:\s*([^,\]]+),\s*Page\s*([^\]]+)\]")


def style_citations(answer_text: str) -> str:
    """Turn '[Source: file.pdf, Page 4]' into a styled inline chip, so
    citations are visually distinct from the rest of the answer at a glance."""
    def replace(match):
        doc_name, page = match.group(1).strip(), match.group(2).strip()
        return f'<span class="citation-chip">{doc_name} · p.{page}</span>'
    return CITATION_PATTERN.sub(replace, answer_text)

st.set_page_config(page_title="Regulatory Q&A Assistant", page_icon="📋", layout="wide")

# ---------- Custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Source Serif 4', serif !important;
    color: #1B3A5C !important;
    letter-spacing: -0.01em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #EEF1F5;
    border-right: 1px solid #D8DEE6;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #4A5568 !important;
}

/* Buttons */
.stButton>button {
    background-color: #1B3A5C;
    color: #FAF9F6;
    border-radius: 6px;
    border: none;
    font-weight: 500;
    padding: 0.45rem 1.1rem;
}
.stButton>button:hover {
    background-color: #142C46;
    color: #FAF9F6;
}

/* Q&A cards */
.qa-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E6EC;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 2px rgba(27, 58, 92, 0.05);
}
.qa-question {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #1B3A5C;
    margin-bottom: 0.6rem;
}

/* Citation chips */
.citation-chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    background-color: #FCEFD8;
    color: #8A5A0A;
    border: 1px solid #EAD3A0;
    border-radius: 4px;
    padding: 0.05rem 0.45rem;
    margin: 0 0.15rem;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# ---------- Password gate ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("📋 Regulatory Q&A Assistant")
    entered_password = st.text_input("Enter office password to continue", type="password")
    if st.button("Enter"):
        if entered_password == st.secrets.get("APP_PASSWORD", ""):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ---------- Session state setup ----------
if "store" not in st.session_state:
    st.session_state.store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "gemini_configured" not in st.session_state:
    st.session_state.gemini_configured = False

# ---------- Auto-load previously saved index, if one exists on disk ----------
if st.session_state.store is None and os.path.exists(f"{INDEX_PATH}.faiss"):
    st.session_state.store = VectorStore.load(INDEX_PATH)
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, "r") as f:
            st.session_state.processed_files = json.load(f)

# ---------- Sidebar: API key + PDF upload ----------
with st.sidebar:
    st.header("Setup")

    shared_key = st.secrets.get("GEMINI_API_KEY", "")

    if shared_key:
        if not st.session_state.gemini_configured:
            configure_gemini(shared_key)
            st.session_state.gemini_configured = True
        st.caption("✓ Using shared office API key")
    else:
        api_key = st.text_input("Gemini API Key", type="password",
                                 help="Get one free at aistudio.google.com/apikey.")
        if api_key and not st.session_state.gemini_configured:
            configure_gemini(api_key)
            st.session_state.gemini_configured = True
            st.success("Gemini configured for this session.")

    st.divider()
    st.header("Upload compliance PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process documents", disabled=not (uploaded_files and st.session_state.gemini_configured)):
        if not st.session_state.gemini_configured:
            st.error("Enter your Gemini API key first.")
        else:
            store = st.session_state.store or VectorStore()
            with st.spinner("Extracting, chunking, and embedding..."):
                for f in uploaded_files:
                    if f.name in st.session_state.processed_files:
                        continue
                    os.makedirs("temp_uploads", exist_ok=True)
                    tmp_path = os.path.join("temp_uploads", f.name)
                    with open(tmp_path, "wb") as out:
                        out.write(f.read())

                    try:
                        pages = extract_pages(tmp_path)
                        chunks = chunk_all_pages(pages)
                        store.add(chunks, batch_size=50, batch_delay=1.0)
                        st.session_state.processed_files.append(f.name)
                        st.success(f"Indexed {f.name}: {len(pages)} pages -> {len(chunks)} chunks.")
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Failed to process {f.name}: {e}")

            st.session_state.store = store

            store.save(INDEX_PATH)
            with open(PROCESSED_FILES_PATH, "w") as f:
                json.dump(st.session_state.processed_files, f)

    if st.session_state.processed_files:
        st.divider()
        st.caption("Indexed documents:")
        for name in st.session_state.processed_files:
            st.text(f"✓ {name}")

        if st.button("Clear saved index"):
            for path in [f"{INDEX_PATH}.faiss", f"{INDEX_PATH}.pkl", PROCESSED_FILES_PATH]:
                if os.path.exists(path):
                    os.remove(path)
            st.session_state.store = None
            st.session_state.processed_files = []
            st.session_state.chat_history = []
            st.rerun()

# ---------- Main panel: Chat ----------
st.title("📋 Regulatory Q&A Assistant")
st.caption("Ask a question about your uploaded compliance/regulatory PDFs. "
           "Every answer is grounded in the actual document text, with page-level citations.")

if not st.session_state.processed_files:
    st.info("Upload and process a PDF from the sidebar to get started.")
else:
    question = st.text_input("Ask a question about the uploaded documents:",
                              placeholder="e.g. What is the required test pressure?")

    col1, col2 = st.columns([1, 5])
    ask_clicked = col1.button("Ask", type="primary")

    if ask_clicked and question.strip():
        if not st.session_state.gemini_configured:
            st.error("Enter your Gemini API key in the sidebar first.")
        else:
            with st.spinner("Retrieving relevant clauses and generating answer..."):
                try:
                    result = answer_question(st.session_state.store, question, top_k=5)
                    st.session_state.chat_history.append((question, result["answer"], result["sources"]))
                except Exception as e:
                    st.error(f"Something went wrong calling Gemini: {e}")

    for q, a, sources in reversed(st.session_state.chat_history):
        styled_answer = style_citations(a)
        st.markdown(f"""
        <div class="qa-card">
            <div class="qa-question">Q: {q}</div>
            <div>{styled_answer}</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander(f"View {len(sources)} source snippet(s) used for this answer"):
            for s in sources:
                st.markdown(f"**{s['doc_name']} — Page {s['page_number']}**")
                st.text(s["snippet"])
                st.divider()